from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_inv, quat_mul

from ..cmds.motion_tracking import MotionTrackingCommand
from .base import BaseObs

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _rotate_vec_world_to_wrist(
    vec_w: torch.Tensor, command: MotionTrackingCommand
) -> torch.Tensor:
    """Rotate world-frame 3-vectors into per-side wrist frame."""
    wrist_quat = command.robot_wrist_quat_w
    if vec_w.dim() == wrist_quat.dim():
        return quat_apply_inverse(wrist_quat, vec_w)
    extra = vec_w.dim() - wrist_quat.dim()
    for _ in range(extra):
        wrist_quat = wrist_quat.unsqueeze(2)
    wrist_quat = wrist_quat.expand(*vec_w.shape[:-1], 4)
    return quat_apply_inverse(wrist_quat, vec_w)


def _rotate_quat_world_to_wrist(
    quat_w: torch.Tensor, command: MotionTrackingCommand
) -> torch.Tensor:
    """Express a per-side world-frame quaternion in the per-side wrist frame."""
    return quat_mul(quat_inv(command.robot_wrist_quat_w), quat_w)


def _log_norm_force(force: torch.Tensor) -> torch.Tensor:
    """Log-norm transform: input (..., 3) → output (..., 4) = [unit_dir * log(|f|+1), log(|f|+1)]."""
    norm = force.norm(dim=-1, keepdim=True)
    unit = force / (norm + 1e-6)
    log_mag = torch.log(norm + 1)
    log_xyz = unit * log_mag
    return torch.cat([log_xyz, log_mag], dim=-1)


def _rotate_force_world_to_wrist(
    force: torch.Tensor, command: MotionTrackingCommand, side: str
) -> torch.Tensor:
    """Rotate per-finger world-frame force (B, n_primaries, 3) into wrist frame."""
    si = command._side_list.index(side)
    wrist_quat = command.robot_wrist_quat_w[:, si : si + 1]
    wrist_quat = wrist_quat.expand(force.shape[0], force.shape[1], 4)
    return quat_apply_inverse(wrist_quat, force)


class MotionTrackingObs(BaseObs):
    """Motion-tracking obs.

    Extend by overriding any ``BaseObs`` @staticmethod or adding new ones.
    """

    # ── MANO reference (absolute) ──────────────────────────────────────────

    @staticmethod
    def mano_wrist_lin_vel(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Absolute MANO wrist velocity. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        return command.mano_wrist_lin_vel_w.reshape(
            command.mano_wrist_lin_vel_w.shape[0], -1
        )

    @staticmethod
    def mano_wrist_quat(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Absolute MANO wrist quaternion. Shape: (B, n_sides*4)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        return command.mano_wrist_quat_w.reshape(command.mano_wrist_quat_w.shape[0], -1)

    @staticmethod
    def mano_wrist_ang_vel(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Absolute MANO wrist angular velocity. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        return command.mano_wrist_ang_vel_w.reshape(
            command.mano_wrist_ang_vel_w.shape[0], -1
        )

    @staticmethod
    def mano_joints_vel(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """MANO reference velocities for 17 bodies, in wrist frame. Shape: (B, n_sides*17*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        parts = []
        for side in command._side_list:
            si = command._side_list.index(side)
            body_vel = command.mano_all_joints_lin_vel_w(side)
            tip_vel = command.mano_tip_lin_vel_w[:, si]
            parts.append(torch.cat([body_vel, tip_vel], dim=1))
        result = torch.stack(parts, dim=1)
        result = _rotate_vec_world_to_wrist(result, command)
        return result.reshape(result.shape[0], -1)

    # ── MANO reference (deltas: ref - sim) ─────────────────────────────────

    @staticmethod
    def mano_wrist_trans_delta(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Delta from robot wrist to MANO wrist target. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.mano_wrist_trans_w - command.robot_wrist_trans_w
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def mano_wrist_quat_delta(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Quaternion rotation delta. Shape: (B, n_sides*4)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        mano_quat = command.mano_wrist_quat_w
        robot_quat = command.robot_wrist_quat_w
        delta = quat_mul(mano_quat, quat_inv(robot_quat))
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def mano_fingertip_trans_delta(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Delta from robot to MANO for 17 tracked bodies per side. Shape: (B, n_sides*17*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        parts = []
        for side in command._side_list:
            si = command._side_list.index(side)
            body_delta = command.mano_all_joints_trans_w(
                side
            ) - command.robot_all_joints_trans_w(side)
            tip_delta = command.mano_tip_trans_w[:, si] - command.robot_tip_trans_w[:, si]
            parts.append(torch.cat([body_delta, tip_delta], dim=1))
        delta = torch.stack(parts, dim=1)
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def mano_wrist_lin_vel_delta(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Wrist velocity delta. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.mano_wrist_lin_vel_w - command.robot_wrist_lin_vel_w
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def mano_wrist_ang_vel_delta(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Wrist angular velocity delta. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.mano_wrist_ang_vel_w - command.robot_wrist_ang_vel_w
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def mano_joints_vel_delta(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Delta of 17-body velocities (MANO - robot), in wrist frame. Shape: (B, n_sides*17*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        parts = []
        for side in command._side_list:
            si = command._side_list.index(side)
            body_delta = command.mano_all_joints_lin_vel_w(
                side
            ) - command.robot_all_joints_lin_vel_w(side)
            tip_mano_vel = command.mano_tip_lin_vel_w[:, si]
            tip_robot_vel = command.robot_all_joints_lin_vel_w(side)[:, -5:]
            tip_delta = tip_mano_vel - tip_robot_vel
            parts.append(torch.cat([body_delta, tip_delta], dim=1))
        result = torch.stack(parts, dim=1)
        result = _rotate_vec_world_to_wrist(result, command)
        return result.reshape(result.shape[0], -1)

    # ── Contact / distance helpers ─────────────────────────────────────────

    @staticmethod
    def ref_contact_flags(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Reference binary contact flag per finger per side. Shape: (B, n_sides*5)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        flags = command.ref_contact_flags
        return flags.reshape(flags.shape[0], -1)

    @staticmethod
    def mano_tips_distance_obs(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Precomputed MANO tip-to-object-surface distance. Shape: (B, n_sides*5)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        return command.mano_tips_distance.reshape(command.mano_tips_distance.shape[0], -1)

    @staticmethod
    def robot_obj_distance(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Distance from object center to 18 robot bodies per side. Shape: (B, n_sides*18)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        parts = []
        for side in command._side_list:
            si = command._side_list.index(side)
            obj_trans = command.sim_obj_trans_w[:, si : si + 1]
            wrist_trans = command.robot_wrist_trans_w[:, si : si + 1]
            body_trans = command.robot_all_joints_trans_w(side)
            tip_trans = command.robot_tip_trans_w[:, si]
            all_trans = torch.cat([wrist_trans, body_trans, tip_trans], dim=1)
            dist = torch.norm(obj_trans - all_trans, dim=-1)
            parts.append(dist)
        return torch.cat(parts, dim=-1)

    # ── Object state (sim, in wrist frame) ─────────────────────────────────

    @staticmethod
    def obj_trans_relative(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Object position relative to wrist, in wrist frame. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.sim_obj_trans_w - command.robot_wrist_trans_w
        delta = _rotate_vec_world_to_wrist(delta, command)
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def obj_quat(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Object quaternion expressed in wrist frame. Shape: (B, n_sides*4)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        q = _rotate_quat_world_to_wrist(command.sim_obj_quat_w, command)
        return q.reshape(q.shape[0], -1)

    @staticmethod
    def obj_lin_vel(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Object linear velocity in wrist frame. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        v = _rotate_vec_world_to_wrist(command.sim_obj_lin_vel_w, command)
        return v.reshape(v.shape[0], -1)

    @staticmethod
    def obj_ang_vel(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Object angular velocity in wrist frame. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        w = _rotate_vec_world_to_wrist(command.sim_obj_ang_vel_w, command)
        return w.reshape(w.shape[0], -1)

    # ── Object deltas (ref - sim, in wrist frame) ──────────────────────────

    @staticmethod
    def obj_trans_delta(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Delta from sim to ref object pos, in wrist frame. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.ref_obj_trans_w - command.sim_obj_trans_w
        delta = _rotate_vec_world_to_wrist(delta, command)
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def obj_quat_delta(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Quaternion delta from sim to ref object orientation, in wrist frame. Shape: (B, n_sides*4)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        wrist_inv = quat_inv(command.robot_wrist_quat_w)
        sim_in_wrist = quat_mul(wrist_inv, command.sim_obj_quat_w)
        ref_in_wrist = quat_mul(wrist_inv, command.ref_obj_quat_w)
        delta = quat_mul(ref_in_wrist, quat_inv(sim_in_wrist))
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def obj_lin_vel_delta(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Delta of object linear velocity (ref - sim) in wrist frame. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.ref_obj_lin_vel_w - command.sim_obj_lin_vel_w
        delta = _rotate_vec_world_to_wrist(delta, command)
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def obj_ang_vel_delta(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Delta of object angular velocity (ref - sim) in wrist frame. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.ref_obj_ang_vel_w - command.sim_obj_ang_vel_w
        delta = _rotate_vec_world_to_wrist(delta, command)
        return delta.reshape(delta.shape[0], -1)

    # ── Object auxiliary (next-frame / SDF) ────────────────────────────────

    @staticmethod
    def future_obj_trans_delta(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Delta from sim obj to next-frame ref obj pos, in wrist frame. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.next_obj_trans_w - command.sim_obj_trans_w
        delta = _rotate_vec_world_to_wrist(delta, command)
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def future_obj_lin_vel(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Next-frame ref obj linear velocity, in wrist frame. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        v = _rotate_vec_world_to_wrist(command.next_obj_vel_w, command)
        return v.reshape(v.shape[0], -1)

    @staticmethod
    def obj_local_sdf_at_keypoints(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Per-keypoint SDF + outward normal at palm + 5 fingertips. Shape: (B, n_sides*6*4)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        parts = []
        B = command.robot_wrist_trans_w.shape[0]
        for side in command._side_list:
            si = command._side_list.index(side)
            if side not in command._obj_sdf_grids:
                parts.append(torch.zeros(B, 6, 4, device=command.device))
                continue
            palm_pos = command.robot_wrist_trans_w[:, si]
            tip_pos = command.robot_tip_trans_w[:, si]
            keypoints = torch.cat([palm_pos[:, None, :], tip_pos], dim=1)
            sdf, grad_world = command.sdf_query(keypoints, side)
            wrist_quat = command.robot_wrist_quat_w[:, si : si + 1].expand(B, 6, 4)
            grad_wrist = quat_apply_inverse(wrist_quat, grad_world)
            feat = torch.cat([sdf[..., None], grad_wrist], dim=-1)
            parts.append(feat)
        out = torch.stack(parts, dim=1)
        return out.reshape(out.shape[0], -1)

    # ── Contact-force obs (sensor-driven, in wrist frame) ──────────────────

    @staticmethod
    def contact_force(
        env: ManagerBasedRlEnv,
        sensor_name: str,
        command_name: str,
        side: str,
    ) -> torch.Tensor:
        """Log-norm contact force in wrist frame. Shape: (B, n_primaries*4)."""
        command = cast(
            MotionTrackingCommand, env.command_manager.get_term(command_name)
        )
        sensor: ContactSensor = env.scene[sensor_name]
        force = sensor.data.force
        force = _rotate_force_world_to_wrist(force, command, side)
        log_force = _log_norm_force(force)
        return log_force.reshape(log_force.shape[0], -1)

    @staticmethod
    def contact_force_history(
        env: ManagerBasedRlEnv,
        sensor_name: str,
        command_name: str,
        side: str,
        history_len: int,
    ) -> torch.Tensor:
        """Rolling history of per-finger log-norm contact force. Shape: (B, history_len*n_primaries*4)."""
        key = f"_contact_force_history_{sensor_name}"
        command = cast(
            MotionTrackingCommand, env.command_manager.get_term(command_name)
        )
        sensor: ContactSensor = env.scene[sensor_name]
        force = sensor.data.force
        force = _rotate_force_world_to_wrist(force, command, side)
        current = _log_norm_force(force)
        n_primaries = force.shape[1]

        if key not in env.extras:
            init = torch.zeros(
                env.num_envs,
                history_len,
                n_primaries,
                4,
                device=env.device,
                dtype=torch.float,
            )
            init[..., 3] = 1.0
            env.extras[key] = init

        buf = env.extras[key]

        reset_mask = env.episode_length_buf == 0
        if reset_mask.any():
            buf[reset_mask] = 0.0
            buf[reset_mask, ..., 3] = 1.0

        buf = torch.cat([buf[:, 1:], current[:, None]], dim=1)
        env.extras[key] = buf

        return buf.reshape(env.num_envs, -1)

    @staticmethod
    def r_contact_force(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        return MotionTrackingObs.contact_force(
            env,
            sensor_name="r_fingertip_contact",
            command_name=command_name,
            side="right",
        )

    @staticmethod
    def r_contact_force_history(
        env: ManagerBasedRlEnv,
        command_name: str,
        history_len: int,
    ) -> torch.Tensor:
        return MotionTrackingObs.contact_force_history(
            env,
            sensor_name="r_fingertip_contact",
            command_name=command_name,
            side="right",
            history_len=history_len,
        )
