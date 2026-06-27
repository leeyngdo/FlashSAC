from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.managers import CommandTerm
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_conjugate,
    quat_error_magnitude,
    quat_mul,
    sample_uniform,
)

from .hand_properties import HandPropertiesMixin
from .object_properties import ObjectPropertiesMixin
from .motion_library import MotionLibrary
from .sdf import bake_object_sdf_grid as _bake_object_sdf_grid

from .motion_tracking_cfg import MotionTrackingCommandCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


class MotionTrackingCommand(HandPropertiesMixin, ObjectPropertiesMixin, CommandTerm):
    """Command term for hand motion tracking against a MANO reference.

    Loads motion.npz with robot (warm-start) and MANO (tracking targets).
    Robot joint data initializes the hand on reset.
    MANO wrist/joint data provides tracking targets for rewards.
    """

    cfg: MotionTrackingCommandCfg
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: MotionTrackingCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.entity_name]
        self.finger_names = cfg.finger_names
        self.motion_lib = MotionLibrary(
            cfg.motion_file,
            self.robot,
            finger_names=cfg.finger_names,
            device=self.device,
        )
        self._side_list = list(self.motion_lib.hand_sides)

        self._init_site_ids(cfg)
        self._init_joint_ids(cfg)
        self._init_object_sdf(cfg)
        self._init_robot_body_ids(cfg)
        self._init_mano_body_ids(cfg)
        self._init_sampling(cfg, env)
        self._init_metrics()
        self._eval_mode = False
        self._ghost_model = None

    # ── Init helpers ─────────────────────────────────────────────────────

    def _init_site_ids(self, cfg: MotionTrackingCommandCfg) -> None:
        """Resolve per-side palm, fingertip, and contact site indices."""
        self._palm_site_ids: dict[str, int] = {}
        self._tip_site_ids: dict[str, list[int]] = {}
        self._contact_site_ids: dict[str, list[int]] = {}
        for side in self.motion_lib.hand_sides:
            self._palm_site_ids[side] = self.robot.site_names.index(
                cfg.site_names["palm"][side]
            )
            self._tip_site_ids[side], _ = self.robot.find_sites(
                cfg.site_names["tip"][side], preserve_order=True
            )
            self._contact_site_ids[side], _ = self.robot.find_sites(
                cfg.site_names["contact"][side], preserve_order=True
            )

    def _init_joint_ids(self, cfg: MotionTrackingCommandCfg) -> None:
        """Resolve wrist (trans/rot) and finger joint IDs for reset noise."""

        def _to_ids(names: list[str]) -> torch.Tensor:
            ids, _ = self.robot.find_joints_by_actuator_names(names)
            return torch.tensor(ids, dtype=torch.long, device=self.device)

        self._wrist_trans_ids = _to_ids(cfg.joint_names["wrist_trans"])
        self._wrist_rot_ids = _to_ids(cfg.joint_names["wrist_rot"])
        self._finger_joint_ids = _to_ids(cfg.joint_names["finger"])

    def _init_robot_body_ids(self, cfg: MotionTrackingCommandCfg) -> None:
        """Resolve robot body IDs for all, level1, level2 mappings."""
        all_body_mano = cfg.body_mapping["all"]
        level1_mapping = cfg.body_mapping["level1"]
        level2_mapping = cfg.body_mapping["level2"]

        self._level1_body_ids: dict[str, list[int]] = {}
        self._level2_body_ids: dict[str, list[int]] = {}
        self._all_body_ids: dict[str, list[int]] = {}

        for side in self.motion_lib.hand_sides:
            self._level1_body_ids[side] = [
                self.robot.body_names.index(f"{side}_{level1_mapping[f][0]}")
                for f in self.finger_names
            ]
            self._level2_body_ids[side] = [
                self.robot.body_names.index(f"{side}_{level2_mapping[f][0]}")
                for f in self.finger_names
            ]
            self._all_body_ids[side] = [
                self.robot.body_names.index(f"{side}_{rb}") for rb, _ in all_body_mano
            ]

    def _init_mano_body_ids(self, cfg: MotionTrackingCommandCfg) -> None:
        """Resolve MANO joint IDs for all, level1, level2 mappings."""
        all_body_mano = cfg.body_mapping["all"]
        level1_mapping = cfg.body_mapping["level1"]
        level2_mapping = cfg.body_mapping["level2"]

        self._level1_mano_ids: dict[str, list[int]] = {}
        self._level2_mano_ids: dict[str, list[int]] = {}
        self._all_mano_ids: dict[str, list[int]] = {}

        for side in self.motion_lib.hand_sides:
            joint_names = self.motion_lib.mano_joint_names[side]
            self._level1_mano_ids[side] = [
                joint_names.index(level1_mapping[f][1]) for f in self.finger_names
            ]
            self._level2_mano_ids[side] = [
                joint_names.index(level2_mapping[f][1]) for f in self.finger_names
            ]
            self._all_mano_ids[side] = [
                joint_names.index(mj) for _, mj in all_body_mano
            ]

    def _init_object_sdf(self, cfg: MotionTrackingCommandCfg) -> None:
        """Bake per-side object SDF grids from visual meshes."""
        self._obj_sdf_grids: dict[str, torch.Tensor] = {}
        self._obj_sdf_extent: float = float(cfg.object.sdf.grid_extent)
        self._obj_sdf_n: int = int(cfg.object.sdf.grid_n)
        if cfg.object.mesh_paths is not None:
            scales = cfg.object.mesh_scales or {}
            for side, mesh_path in cfg.object.mesh_paths.items():
                if side not in self.motion_lib.hand_sides or mesh_path is None:
                    continue
                self._obj_sdf_grids[side] = _bake_object_sdf_grid(
                    mesh_path,
                    float(scales.get(side, 1.0)),
                    self._obj_sdf_extent,
                    self._obj_sdf_n,
                    device=str(self.device),
                )

    def _init_sampling(
        self,
        cfg: MotionTrackingCommandCfg,
        env: ManagerBasedRlEnv,
    ) -> None:
        """Initialize time-stepping state and adaptive sampling buffers."""
        self.motion_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.motion_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        M = self.motion_lib.num_trajectories
        self.bins_per_traj = (
            self.motion_lib._motion_num_frames.float() // (1 / env.step_dt)
        ).long() + 1
        self.bin_count = int(self.bins_per_traj.max().item())
        self.bin_failed_count = torch.zeros(
            M, self.bin_count, dtype=torch.float, device=self.device
        )
        self._current_bin_failed = torch.zeros(
            M, self.bin_count, dtype=torch.float, device=self.device
        )
        self.kernel = torch.tensor(
            [cfg.sampling.lambda_**i for i in range(cfg.sampling.kernel_size)],
            device=self.device,
        )
        self.kernel /= self.kernel.sum()

    def _init_metrics(self) -> None:
        """Allocate per-side tracking and contact metric buffers."""
        side_prefixes = {"right": "r", "left": "l"}
        for side in self.motion_lib.hand_sides:
            p = side_prefixes[side]
            for key in (
                "error_wrist_trans",
                "error_wrist_rot",
                "error_wrist_lin_vel",
                "error_wrist_ang_vel",
                "error_joint_vel",
            ):
                self.metrics[f"{key}_{p}"] = torch.zeros(
                    self.num_envs, device=self.device
                )
            for finger in self.finger_names:
                self.metrics[f"error_tip_trans_{p}_{finger}"] = torch.zeros(
                    self.num_envs, device=self.device
                )
            for level in (1, 2):
                self.metrics[f"error_level{level}_{p}"] = torch.zeros(
                    self.num_envs, device=self.device
                )
            if self.has_objects:
                for key in (
                    "error_obj_trans",
                    "error_obj_rot",
                    "error_obj_lin_vel",
                    "error_obj_ang_vel",
                ):
                    self.metrics[f"{key}_{p}"] = torch.zeros(
                        self.num_envs, device=self.device
                    )

        self._contact_sum: dict[str, torch.Tensor] = {}
        self._contact_count_ref: dict[str, torch.Tensor] = {}
        self._contact_count_contact: dict[str, torch.Tensor] = {}
        if self.has_objects:
            for side in self.motion_lib.hand_sides:
                p = side_prefixes[side]
                for finger in self.finger_names:
                    k = f"{p}_{finger}"
                    self._contact_count_ref[k] = torch.zeros(
                        self.num_envs, device=self.device
                    )
                    self._contact_count_contact[k] = torch.zeros(
                        self.num_envs, device=self.device
                    )
                    for name in ("ref_dist", "pen", "force"):
                        self._contact_sum[f"contact_{name}_{k}"] = torch.zeros(
                            self.num_envs, device=self.device
                        )

        n_sides, n_fingers = len(self._side_list), len(self.finger_names)
        self.contact_miss_counter = torch.zeros(
            self.num_envs, n_sides, n_fingers, device=self.device
        )
        self.contact_miss_max = torch.zeros_like(self.contact_miss_counter)

        self.pin_fired_this_step = torch.zeros(
            self.num_envs, n_sides, dtype=torch.bool, device=self.device
        )

    # --- Eval mode ---

    def set_eval_mode(
        self,
        *,
        sampling_mode: str,
        noise_to_initial_level: float,
        start_frame: int,
    ) -> None:
        """Swap to caller-supplied eval-side overrides (cfg.eval.command.*).

        Stashes the train values so ``set_train_mode`` can restore them.
        """
        self._train_sampling_mode = self.cfg.sampling.mode
        self._train_noise_level = self.cfg.hand.noise_to_initial_level
        self._train_start_frame = self.cfg.sampling.start_frame
        self.cfg.sampling.mode = sampling_mode
        self.cfg.hand.noise_to_initial_level = noise_to_initial_level
        self.cfg.sampling.start_frame = start_frame
        self._eval_mode = True

    def set_train_mode(self) -> None:
        """Restore training-side sampling mode + noise multiplier."""
        self.cfg.sampling.mode = self._train_sampling_mode
        self.cfg.hand.noise_to_initial_level = self._train_noise_level
        self.cfg.sampling.start_frame = self._train_start_frame
        self._eval_mode = False

    @property
    def motion_num_frames(self) -> torch.Tensor:
        """Per-env motion length in frames. Shape: (B,)."""
        return self.motion_lib._motion_num_frames[self.motion_ids]

    @property
    def motion_completed(self) -> torch.Tensor:
        """Per-env bool: True if motion_steps reached the trajectory end."""
        return self.motion_steps >= self.motion_num_frames

    # --- Flat-index helpers ---

    @property
    def _motion_flat_ids(self) -> torch.Tensor:
        """Flat index into motion library tensors: length_starts[traj] + time_step."""
        return self.motion_lib.length_starts[self.motion_ids] + self.motion_steps

    # --- Command property ---

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.ref_joint_pos, self.ref_joint_vel], dim=1)

    # --- Reference joint state (for reset) ---

    @property
    def ref_joint_pos(self) -> torch.Tensor:
        return self.motion_lib.robot_joint_pos[self._motion_flat_ids]

    @property
    def ref_joint_vel(self) -> torch.Tensor:
        return self.motion_lib.robot_joint_vel[self._motion_flat_ids]

    # --- CommandTerm abstract methods ---

    def _update_metrics(self) -> None:
        side_prefixes = {"right": "r", "left": "l"}
        for si, side in enumerate(self._side_list):
            p = side_prefixes[side]

            # Wrist position (m)
            self.metrics[f"error_wrist_trans_{p}"] = torch.norm(
                self.mano_wrist_trans_w[:, si] - self.robot_wrist_trans_w[:, si], dim=-1
            )

            # Wrist rotation (rad)
            self.metrics[f"error_wrist_rot_{p}"] = quat_error_magnitude(
                self.mano_wrist_quat_w[:, si], self.robot_wrist_quat_w[:, si]
            )

            # Per-finger fingertip position (m)
            tip_err = torch.norm(
                self.mano_tip_trans_w[:, si] - self.robot_tip_trans_w[:, si], dim=-1
            )  # (B, 5)
            for fi, finger in enumerate(self.finger_names):
                self.metrics[f"error_tip_trans_{p}_{finger}"] = tip_err[:, fi]

            # Level 1, 2 joint position (m, mean over 5 joints)
            for level in (1, 2):
                mano_trans = self.mano_level_trans_w(side, level)  # (B, 5, 3)
                robot_trans = self.robot_level_trans_w(side, level)  # (B, 5, 3)
                self.metrics[f"error_level{level}_{p}"] = torch.norm(
                    mano_trans - robot_trans, dim=-1
                ).mean(dim=-1)

            # Wrist linear / angular velocity (m/s, rad/s; mean over xyz)
            self.metrics[f"error_wrist_lin_vel_{p}"] = torch.mean(
                torch.abs(
                    self.mano_wrist_lin_vel_w[:, si] - self.robot_wrist_lin_vel_w[:, si]
                ),
                dim=-1,
            )
            self.metrics[f"error_wrist_ang_vel_{p}"] = torch.mean(
                torch.abs(
                    self.mano_wrist_ang_vel_w[:, si] - self.robot_wrist_ang_vel_w[:, si]
                ),
                dim=-1,
            )

            # All-17-bodies joint velocity (matches joints_vel_error_exp's calc)
            body_delta = self.mano_all_joints_lin_vel_w(
                side
            ) - self.robot_all_joints_lin_vel_w(side)  # (B, 12, 3)
            tip_mano_vel = self.mano_tip_lin_vel_w[:, si]  # (B, 5, 3)
            tip_robot_vel = self.robot_all_joints_lin_vel_w(side)[:, -5:]  # (B, 5, 3)
            all_delta = torch.cat([body_delta, tip_mano_vel - tip_robot_vel], dim=1)
            self.metrics[f"error_joint_vel_{p}"] = (
                all_delta.abs().mean(dim=-1).mean(dim=-1)
            )

            # Object tracking — mirror obj_{trans,rot,lin_vel,ang_vel}_error_exp rewards.
            if self.has_objects:
                self.metrics[f"error_obj_trans_{p}"] = torch.norm(
                    self.ref_obj_trans_w[:, si] - self.sim_obj_trans_w[:, si], dim=-1
                )
                self.metrics[f"error_obj_rot_{p}"] = quat_error_magnitude(
                    self.ref_obj_quat_w[:, si], self.sim_obj_quat_w[:, si]
                )
                self.metrics[f"error_obj_lin_vel_{p}"] = torch.mean(
                    torch.abs(
                        self.ref_obj_lin_vel_w[:, si] - self.sim_obj_lin_vel_w[:, si]
                    ),
                    dim=-1,
                )
                self.metrics[f"error_obj_ang_vel_{p}"] = torch.mean(
                    torch.abs(
                        self.ref_obj_ang_vel_w[:, si] - self.sim_obj_ang_vel_w[:, si]
                    ),
                    dim=-1,
                )

                # Per-finger contact performance. Two gates per (side, finger):
                #   - ref_flag==1            → ref_dist + found (hit rate)
                #   - ref_flag==1 AND found  → pen + force (in-contact behavior)
                pen_sensor: ContactSensor = self._env.scene[
                    f"{p}_fingertip_penetration"
                ]
                force_sensor: ContactSensor = self._env.scene[f"{p}_fingertip_contact"]
                for fi, finger in enumerate(self.finger_names):
                    k = f"{p}_{finger}"
                    flag = self.ref_contact_flags[:, si, fi]
                    found = (pen_sensor.data.found[:, fi] > 0).to(flag.dtype)
                    contact_gate = flag * found
                    ref_dist = torch.norm(
                        self.ref_contact_trans_w[:, si, fi]
                        - self.robot_tip_trans_w[:, si, fi],
                        dim=-1,
                    )
                    pen = torch.clamp(-pen_sensor.data.dist[:, fi], min=0.0)
                    force = torch.norm(force_sensor.data.force[:, fi], dim=-1)
                    self._contact_sum[f"contact_ref_dist_{k}"] += ref_dist * flag
                    self._contact_sum[f"contact_pen_{k}"] += pen * contact_gate
                    self._contact_sum[f"contact_force_{k}"] += force * contact_gate
                    self._contact_count_ref[k] += flag
                    self._contact_count_contact[k] += contact_gate
                    # Consecutive-miss counter: increments on (flag==1 AND found==0),
                    # resets on (flag==0 OR found==1). The single-line form below
                    # captures both: miss=1 → (ctr+1)*1=ctr+1; miss=0 → (ctr+1)*0=0.
                    miss = flag * (1.0 - found)
                    self.contact_miss_counter[:, si, fi] = (
                        self.contact_miss_counter[:, si, fi] + 1.0
                    ) * miss
                    self.contact_miss_max[:, si, fi] = torch.maximum(
                        self.contact_miss_max[:, si, fi],
                        self.contact_miss_counter[:, si, fi],
                    )

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if not self.has_objects:
            return extras
        side_prefixes = {"right": "r", "left": "l"}
        for si, side in enumerate(self._side_list):
            p = side_prefixes[side]
            for fi, finger in enumerate(self.finger_names):
                k = f"{p}_{finger}"
                count_ref = self._contact_count_ref[k]
                count_contact = self._contact_count_contact[k]
                total_ref = count_ref[env_ids].sum().clamp(min=1.0)
                total_contact = count_contact[env_ids].sum().clamp(min=1.0)
                # ref_dist + found gated on ref_flag==1; pen + force gated on ref_flag==1 AND found.
                extras[f"contact_ref_dist_{k}"] = (
                    self._contact_sum[f"contact_ref_dist_{k}"][env_ids].sum()
                    / total_ref
                ).item()
                extras[f"contact_found_{k}"] = (
                    count_contact[env_ids].sum() / total_ref
                ).item()
                extras[f"contact_pen_{k}"] = (
                    self._contact_sum[f"contact_pen_{k}"][env_ids].sum() / total_contact
                ).item()
                extras[f"contact_force_{k}"] = (
                    self._contact_sum[f"contact_force_{k}"][env_ids].sum()
                    / total_contact
                ).item()
                # Per-finger episode-max consecutive-miss streak (mean over resetting envs).
                extras[f"contact_miss_max_{k}"] = (
                    self.contact_miss_max[env_ids, si, fi].mean().item()
                )
                for name in ("ref_dist", "pen", "force"):
                    self._contact_sum[f"contact_{name}_{k}"][env_ids] = 0.0
                count_ref[env_ids] = 0.0
                count_contact[env_ids] = 0.0
                self.contact_miss_counter[env_ids, si, fi] = 0.0
                self.contact_miss_max[env_ids, si, fi] = 0.0
        return extras

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # In eval mode, keep assigned motion_ids; only reset motion_steps.
        if not self._eval_mode and self.motion_lib.num_trajectories > 1:
            # Random per-reset trajectory assignment: each resetting env draws a
            # fresh traj_idx uniformly from [0, M). For M=1 this is a no-op.
            self.motion_ids[env_ids] = torch.randint(
                0,
                self.motion_lib.num_trajectories,
                (len(env_ids),),
                device=self.device,
                dtype=torch.long,
            )

        if self.cfg.sampling.mode == "start":
            sf = int(self.cfg.sampling.start_frame)
            if sf < 0:
                raise ValueError(f"sampling.start_frame must be >= 0, got {sf}")
            # Per-trajectory clamp: a resetting env whose trajectory is shorter
            # than sf starts at its own last frame (deterministic, correct for
            # mixed-length objects), not silently at 0.
            t_m = self.motion_lib._motion_num_frames[self.motion_ids[env_ids]]
            self.motion_steps[env_ids] = torch.minimum(
                torch.full_like(t_m, sf), t_m - 1
            )
        elif self.cfg.sampling.mode == "uniform":
            self._uniform_sampling(env_ids)
        else:
            self._adaptive_sampling(env_ids)

        # Warm-start with per-component init noise (training-time DR). Each
        # effective sigma = baseline × global multiplier (ASAP-style):
        #   sigma_wrist_pos = init_noise_scale.wrist_pos * noise_to_initial_level
        # ``finger_vel`` REPLACES ref finger vel rather than adding — matches
        # ManipTrans behavior. mjlab events fire BEFORE command_manager.reset, so
        # the noise CAN'T live as an EventTermCfg — has to ride on top of ref pose
        # inside the same reset hook.
        joint_pos = self.ref_joint_pos[env_ids].clone()
        joint_vel = self.ref_joint_vel[env_ids].clone()

        soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        mult = float(self.cfg.hand.noise_to_initial_level)
        ns = self.cfg.hand.init_noise_scale
        N = len(env_ids)

        def _randn(n_dofs: int) -> torch.Tensor:
            return torch.randn(N, n_dofs, device=self.device)

        wrist_trans_sigma = float(ns.get("wrist_trans", 0.0)) * mult
        wrist_rot_sigma = math.radians(float(ns.get("wrist_rot_deg", 0.0))) * mult
        finger_range_frac = float(ns.get("finger_range_frac", 0.0)) * mult
        wrist_trans_vel_sigma = float(ns.get("wrist_trans_vel", 0.0)) * mult
        wrist_rot_vel_sigma = float(ns.get("wrist_rot_vel", 0.0)) * mult
        finger_vel_sigma = float(ns.get("finger_vel", 0.0)) * mult

        joint_pos[:, self._wrist_trans_ids] += (
            _randn(len(self._wrist_trans_ids)) * wrist_trans_sigma
        )
        joint_pos[:, self._wrist_rot_ids] += (
            _randn(len(self._wrist_rot_ids)) * wrist_rot_sigma
        )
        finger_range = (
            soft_limits[:, self._finger_joint_ids, 1]
            - soft_limits[:, self._finger_joint_ids, 0]
        )
        joint_pos[:, self._finger_joint_ids] += _randn(len(self._finger_joint_ids)) * (
            finger_range * finger_range_frac
        )
        joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])

        joint_vel[:, self._wrist_trans_ids] += (
            _randn(len(self._wrist_trans_ids)) * wrist_trans_vel_sigma
        )
        joint_vel[:, self._wrist_rot_ids] += (
            _randn(len(self._wrist_rot_ids)) * wrist_rot_vel_sigma
        )
        # Finger velocity REPLACES ref vel (ManipTrans behavior); sigma=0 keeps ref.
        if finger_vel_sigma > 0.0:
            joint_vel[:, self._finger_joint_ids] = (
                _randn(len(self._finger_joint_ids)) * finger_vel_sigma
            )

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.reset(env_ids=env_ids)

        # Reset objects to demo trajectory pose. Seed pos+quat+vel from the
        # motion ref regardless of pin_mode; without seeding, ``obj.reset``
        # falls back to the entity's default spawn pose (typically above the
        # ground), so the object drops in free-fall at t=0 even when the
        # policy is supposed to be carrying it.
        if self.has_objects:
            for side in self._side_list:
                if side not in self.motion_lib.obj_trans:
                    continue
                obj_name = self.cfg.object.entity_names[side]
                obj: Entity = self._env.scene[obj_name]
                obj_trans = self.ref_obj_trans_w[env_ids][
                    :, self._side_list.index(side)
                ]  # (N, 3)
                obj_quat = self.ref_obj_quat_w[env_ids][
                    :, self._side_list.index(side)
                ]  # (N, 4)
                obj_lin_vel = self.ref_obj_lin_vel_w[env_ids][
                    :, self._side_list.index(side)
                ]
                obj_ang_vel = self.motion_lib.obj_ang_vel[side][
                    self.motion_lib.length_starts[self.motion_ids[env_ids]]
                    + self.motion_steps[env_ids]
                ]
                root_state = torch.cat(
                    [obj_trans, obj_quat, obj_lin_vel, obj_ang_vel], dim=-1
                )
                obj.write_root_state_to_sim(root_state, env_ids=env_ids)
                obj.reset(env_ids=env_ids)

    def _update_command(self) -> None:
        self.motion_steps += 1

        # In eval mode, clamp at end of motion (don't wrap/resample).
        if self._eval_mode:
            max_frames = self.motion_lib._motion_num_frames[self.motion_ids]
            self.motion_steps.clamp_(max=max_frames - 1)
        else:
            # Wrap around envs that exceeded their trajectory's length.
            wrap_ids = torch.where(
                self.motion_steps >= self.motion_lib._motion_num_frames[self.motion_ids]
            )[0]
            if wrap_ids.numel() > 0:
                self._resample_command(wrap_ids)

        # Reset per-step pin buffer; will be overwritten below when pinning fires.
        self.pin_fired_this_step = torch.zeros_like(self.pin_fired_this_step)

        # Pin objects to reference trajectory every step
        if self.cfg.object.pin_objects and self.has_objects:
            pin_mode = self.cfg.object.pin_mode
            if pin_mode == "xfrc":
                # DexMachina-style soft PD on the object freejoint via xfrc_applied.
                # The object is a plain freejoint entity (single body, gravity on);
                # we inject a world-frame wrench computed from reference vs. sim error.
                #   force  = kp.pos * (ref_trans - sim_trans) + kd.pos * (ref_lin_vel - sim_lin_vel)
                #   torque = kp.rot * axis_angle(ref_q sim_q^-1) + kd.rot * (ref_w - sim_w)
                # Axis-angle is in world frame (consistent with sim_obj_ang_vel_w).
                # Gravity is unmodified — PD must carry the weight at steady state.
                xfrc = self.cfg.object.xfrc
                kp_pos = xfrc.kp.pos
                kd_pos = xfrc.kd.pos
                kp_rot = xfrc.kp.rot
                kd_rot = xfrc.kd.rot
                sim_trans_all = self.sim_obj_trans_w  # (B, n_sides, 3)
                sim_quat_all = self.sim_obj_quat_w  # (B, n_sides, 4)
                sim_lin_vel_all = self.sim_obj_lin_vel_w  # (B, n_sides, 3)
                sim_ang_vel_all = self.sim_obj_ang_vel_w  # (B, n_sides, 3)
                ref_trans_all = self.ref_obj_trans_w  # (B, n_sides, 3)
                ref_quat_all = self.ref_obj_quat_w  # (B, n_sides, 4)
                for side in self._side_list:
                    if side not in self.motion_lib.obj_trans:
                        continue
                    si = self._side_list.index(side)
                    obj_name = self.cfg.object.entity_names[side]
                    obj: Entity = self._env.scene[obj_name]

                    ref_trans = ref_trans_all[:, si]
                    ref_quat = ref_quat_all[:, si]
                    ref_lin_vel = self.motion_lib.obj_lin_vel[side][
                        self._motion_flat_ids
                    ]
                    ref_ang_vel = self.motion_lib.obj_ang_vel[side][
                        self._motion_flat_ids
                    ]
                    sim_trans = sim_trans_all[:, si]
                    sim_quat = sim_quat_all[:, si]
                    sim_lin_vel = sim_lin_vel_all[:, si]
                    sim_ang_vel = sim_ang_vel_all[:, si]

                    # Linear PD
                    force = kp_pos * (ref_trans - sim_trans) + kd_pos * (
                        ref_lin_vel - sim_lin_vel
                    )
                    # Rotational PD: axis-angle of delta_q = ref_q * sim_q^{-1}
                    delta_q = quat_mul(ref_quat, quat_conjugate(sim_quat))
                    axis_angle = axis_angle_from_quat(delta_q)
                    omega_rot = float(xfrc.omega_rot)
                    if omega_rot > 0.0:
                        # Anisotropic inertia-tensor mode. I_body = R(iquat)·diag(body_inertia)·R(iquat)^T
                        # is the full inertia tensor in body frame (body_iquat maps principal→body
                        # frame; identity if MuJoCo auto-aligned body frame to principal axes).
                        # I_world = R_sim · I_body · R_sim^T per env per step.
                        body_id = int(obj.indexing.body_ids[0])
                        I_diag_p = self._env.sim.model.body_inertia[0, body_id].to(
                            sim_quat.device, dtype=sim_quat.dtype
                        )  # (3,) principal-frame diagonal
                        iquat = self._env.sim.model.body_iquat[0, body_id].to(
                            sim_quat.device, dtype=sim_quat.dtype
                        )  # (4,) principal→body
                        R_ip = matrix_from_quat(iquat.unsqueeze(0))[0]  # (3, 3)
                        I_body = (
                            R_ip @ torch.diag(I_diag_p) @ R_ip.T
                        )  # (3, 3) full body-frame inertia
                        R_sim = matrix_from_quat(sim_quat)  # (B, 3, 3)
                        I_world = R_sim @ I_body @ R_sim.transpose(-1, -2)  # (B, 3, 3)
                        zeta_rot = float(xfrc.zeta_rot)
                        kp_rot_eff = omega_rot * omega_rot
                        kd_rot_eff = 2.0 * zeta_rot * omega_rot
                        ang_err = ref_ang_vel - sim_ang_vel
                        torque = kp_rot_eff * torch.einsum(
                            "bij,bj->bi", I_world, axis_angle
                        ) + kd_rot_eff * torch.einsum("bij,bj->bi", I_world, ang_err)
                    else:
                        torque = kp_rot * axis_angle + kd_rot * (
                            ref_ang_vel - sim_ang_vel
                        )

                    # Freejoint entity has a single body; apply to it. Shape (B, 1, 3).
                    obj.write_external_wrench_to_sim(
                        forces=force.unsqueeze(1),
                        torques=torque.unsqueeze(1),
                    )
            elif pin_mode == "none":
                pass
            else:
                raise ValueError(
                    f"unknown pin_mode={pin_mode!r}; expected 'xfrc' or 'none'"
                )

        # Update adaptive sampling statistics
        if self.cfg.sampling.mode == "adaptive":
            self.bin_failed_count = (
                self.cfg.sampling.alpha * self._current_bin_failed
                + (1 - self.cfg.sampling.alpha) * self.bin_failed_count
            )
            self._current_bin_failed.zero_()

    # --- Sampling helpers ---

    def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
        """Per-trajectory adaptive resampling.

        `self.bin_failed_count` has shape (M, bin_count) — one EMA-smoothed
        failure histogram per trajectory. `bins_per_traj[m]` is the number of
        valid bins for traj m; bins beyond that are masked out so they never get
        sampled and don't distort the kernel-smoothed distribution.
        """
        # --- 1. Scatter this step's failures into (traj, bin). ---
        episode_failed = self._env.termination_manager.terminated[env_ids]
        self._current_bin_failed.zero_()
        if torch.any(episode_failed):
            fail_envs = env_ids[episode_failed]
            fail_tr = self.motion_ids[fail_envs]
            fail_t = self.motion_steps[fail_envs]
            fail_T_m = self.motion_lib._motion_num_frames[fail_tr].clamp_min(1)
            fail_bpt = self.bins_per_traj[fail_tr]
            fail_bin = torch.clamp(
                (fail_t * fail_bpt) // fail_T_m, 0, self.bin_count - 1
            )
            self._current_bin_failed.index_put_(
                (fail_tr, fail_bin),
                torch.ones_like(fail_bin, dtype=torch.float),
                accumulate=True,
            )

        # --- 2. Build per-traj sampling distribution (M, bin_count). ---
        valid_mask = (
            torch.arange(self.bin_count, device=self.device)[None, :]
            < self.bins_per_traj[:, None]
        ).float()  # (M, bin_count)
        probs = self.bin_failed_count + self.cfg.sampling.uniform_ratio / float(
            self.bin_count
        )
        # Mask before smoothing so invalid bins contribute zero; the kernel's
        # right-edge replicate pad then replicates an already-zero cell and the
        # smoothing stays inside each traj's valid range.
        probs = probs * valid_mask
        probs = torch.nn.functional.pad(
            probs.unsqueeze(1),
            (0, self.cfg.sampling.kernel_size - 1),
            mode="replicate",
        )
        probs = torch.nn.functional.conv1d(probs, self.kernel.view(1, 1, -1)).squeeze(
            1
        )  # (M, bin_count)
        probs = probs * valid_mask
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # --- 3. Sample one bin per resetting env from its traj's row. ---
        tr = self.motion_ids[env_ids]
        probs_per_env = probs[tr]  # (N, bin_count)
        sampled_bins = torch.multinomial(probs_per_env, 1, replacement=True).squeeze(
            -1
        )  # (N,)

        # --- 4. Convert bin → frame using per-env traj length. ---
        bpt = self.bins_per_traj[tr].float()
        T_m = self.motion_lib._motion_num_frames[tr].float()
        self.motion_steps[env_ids] = (
            (
                sampled_bins
                + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
            )
            / bpt
            * (T_m - 1)
        ).long()

    def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
        # ManipTrans: floor(T_m * 0.99 * rand) per env, using each env's own
        # trajectory length (self.motion_ids was already sampled by
        # _resample_command before this helper runs).
        tr = self.motion_ids[env_ids]
        T_m = self.motion_lib._motion_num_frames[tr].float()
        self.motion_steps[env_ids] = (
            torch.rand(len(env_ids), device=self.device) * 0.99 * T_m
        ).long()

    # --- Debug visualization (ghost reference + MANO keypoints) ---

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """Render reference hand+object ghost and/or MANO keypoint markers."""
        show_robot = self.cfg.viz_robot_ghost
        show_object = self.cfg.viz_object_ghost and self.has_objects

        if show_robot or show_object:
            if self._ghost_model is None:
                self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
                collision = (self._ghost_model.geom_contype != 0) | (
                    self._ghost_model.geom_conaffinity != 0
                )
                object_body_ids: list[int] = []
                if self.has_objects:
                    for entity_name in self.cfg.object.entity_names.values():
                        entity = self._env.scene[entity_name]
                        object_body_ids.extend(
                            entity.indexing.body_ids.cpu().numpy().tolist()
                        )
                object_mask = np.isin(self._ghost_model.geom_bodyid, object_body_ids)
                robot_visual = ~object_mask & ~collision

                self._ghost_model.geom_rgba[:] = np.array([0.2, 0.6, 1.0, 0.5])
                self._ghost_model.geom_rgba[collision, 3] = 0.0
                self._ghost_model.geom_rgba[robot_visual, 3] = (
                    0.5 if show_robot else 0.0
                )
                self._ghost_model.geom_rgba[object_mask, 3] = (
                    0.5 if show_object else 0.0
                )

            free_q = self.robot.indexing.free_joint_q_adr.cpu().numpy()
            joint_q = self.robot.indexing.joint_q_adr.cpu().numpy()
            obj_q_adr = {}
            if self.has_objects:
                obj_q_adr = {
                    side: self._env.scene[name].indexing.free_joint_q_adr.cpu().numpy()
                    for side, name in self.cfg.object.entity_names.items()
                }

            # mjlab's visualizer hard-codes mjVIS_TRANSPARENT on its vopt, which
            # in MuJoCo 3.7's offscreen renderer ends up depth-rejecting the
            # opaque real geoms behind the translucent ghost geoms (ghost
            # writes depth even though it's alpha-blended). Temporarily clear
            # the flag around each add_ghost_mesh call so the ghost geoms get
            # added with their authored 0.5 alpha and proper depth handling.
            import mujoco  # local import: viz path only

            _t_flag = int(mujoco.mjtVisFlag.mjVIS_TRANSPARENT)
            prev_transparent = bool(visualizer._vopt.flags[_t_flag])
            visualizer._vopt.flags[_t_flag] = False
            try:
                for batch in visualizer.get_env_indices(self.num_envs):
                    qpos = np.zeros(self._env.sim.mj_model.nq)
                    if free_q.size > 0:
                        qpos[free_q[0:3]] = (
                            self.mano_wrist_trans_w[batch, 0].cpu().numpy()
                        )
                        qpos[free_q[3:7]] = (
                            self.mano_wrist_quat_w[batch, 0].cpu().numpy()
                        )
                    qpos[joint_q] = self.ref_joint_pos[batch].cpu().numpy()
                    for si, side in enumerate(self._side_list):
                        if side not in obj_q_adr:
                            continue
                        adr = obj_q_adr[side]
                        qpos[adr[0:3]] = self.ref_obj_trans_w[batch, si].cpu().numpy()
                        qpos[adr[3:7]] = self.ref_obj_quat_w[batch, si].cpu().numpy()
                    visualizer.add_ghost_mesh(
                        qpos, model=self._ghost_model, label=f"ref_{batch}"
                    )
            finally:
                visualizer._vopt.flags[_t_flag] = prev_transparent

        if self.cfg.viz_human_keypoint:
            RED = (1.0, 0.1, 0.1, 1.0)
            GREEN = (0.2, 0.9, 0.3, 1.0)
            WHITE = (1.0, 1.0, 1.0, 1.0)
            finger_chain: dict[str, dict[str, int]] = {
                f: {} for f in self.finger_names
            }
            for i, (_, mano_joint) in enumerate(self.cfg.body_mapping["all"]):
                for f in self.finger_names:
                    if mano_joint.startswith(f"{f}_"):
                        kind = mano_joint[len(f) + 1 :]
                        finger_chain[f].setdefault(kind, i)
                        break

            for batch in visualizer.get_env_indices(self.num_envs):
                for si, side in enumerate(self._side_list):
                    wrist = self.mano_wrist_trans_w[batch, si].cpu().numpy()
                    tips = self.mano_tip_trans_w[batch, si].cpu().numpy()
                    non_tips = self.mano_all_joints_trans_w(side)[batch].cpu().numpy()
                    visualizer.add_sphere(
                        wrist,
                        radius=0.008,
                        color=WHITE,
                        label=f"mano_kp_{side}_wrist_{batch}",
                    )
                    for i, p in enumerate(tips):
                        visualizer.add_sphere(
                            p,
                            radius=0.005,
                            color=RED,
                            label=f"mano_kp_{side}_tip{i}_{batch}",
                        )
                    for i, p in enumerate(non_tips):
                        visualizer.add_sphere(
                            p,
                            radius=0.004,
                            color=GREEN,
                            label=f"mano_kp_{side}_joint{i}_{batch}",
                        )
                    for fi, finger in enumerate(self.finger_names):
                        chain = finger_chain[finger]
                        if "proximal" not in chain or "intermediate" not in chain:
                            continue
                        prox = non_tips[chain["proximal"]]
                        inter = non_tips[chain["intermediate"]]
                        tip = tips[fi]
                        for seg, (a, b) in enumerate(
                            ((wrist, prox), (prox, inter), (inter, tip))
                        ):
                            visualizer.add_cylinder(
                                a,
                                b,
                                radius=0.0025,
                                color=GREEN,
                                label=f"mano_edge_{side}_{finger}_{seg}_{batch}",
                            )
