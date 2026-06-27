from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from ..cmds.motion_tracking import MotionTrackingCommand
from .base import BaseRewards

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _cmd(env: ManagerBasedRlEnv, command_name: str) -> MotionTrackingCommand:
    return cast(MotionTrackingCommand, env.command_manager.get_term(command_name))


def _side_idx(command: MotionTrackingCommand, side: str) -> int:
    return command._side_list.index(side)


class MotionTrackingRewards(BaseRewards):
    # ── Per-side tracking-error rewards (compare ref motion to sim) ────────

    @staticmethod
    def tracking_fingertip_trans_error_exp(
        env: ManagerBasedRlEnv,
        command_name: str,
        side: str,
        finger: str,
        scale: float,
    ) -> torch.Tensor:
        """exp(-scale * ||mano_tip - robot_tip||) for one finger on one side."""
        command = _cmd(env, command_name)
        si = _side_idx(command, side)
        fi = command.finger_names.index(finger)
        error = torch.norm(
            command.mano_tip_trans_w[:, si, fi] - command.robot_tip_trans_w[:, si, fi],
            dim=-1,
        )
        return torch.exp(-scale * error)

    @staticmethod
    def tracking_wrist_lin_vel_error_exp(
        env: ManagerBasedRlEnv,
        command_name: str,
        side: str,
        scale: float,
    ) -> torch.Tensor:
        """exp(-scale * mean(|mano_vel - robot_vel|)) for one side."""
        command = _cmd(env, command_name)
        si = _side_idx(command, side)
        error = torch.mean(
            torch.abs(
                command.mano_wrist_lin_vel_w[:, si]
                - command.robot_wrist_lin_vel_w[:, si]
            ),
            dim=-1,
        )
        return torch.exp(-scale * error)

    @staticmethod
    def tracking_wrist_ang_vel_error_exp(
        env: ManagerBasedRlEnv,
        command_name: str,
        side: str,
        scale: float,
    ) -> torch.Tensor:
        """exp(-scale * mean(|mano_ang_vel - robot_ang_vel|)) for one side."""
        command = _cmd(env, command_name)
        si = _side_idx(command, side)
        error = torch.mean(
            torch.abs(
                command.mano_wrist_ang_vel_w[:, si]
                - command.robot_wrist_ang_vel_w[:, si]
            ),
            dim=-1,
        )
        return torch.exp(-scale * error)

    @staticmethod
    def tracking_wrist_trans_error_exp(
        env: ManagerBasedRlEnv,
        command_name: str,
        side: str,
        scale: float,
    ) -> torch.Tensor:
        """exp(-scale * ||mano_wrist_pos - robot_wrist_pos||) for one side."""
        command = _cmd(env, command_name)
        si = _side_idx(command, side)
        error = torch.norm(
            command.mano_wrist_trans_w[:, si] - command.robot_wrist_trans_w[:, si],
            dim=-1,
        )
        return torch.exp(-scale * error)

    @staticmethod
    def tracking_wrist_rot_error_exp(
        env: ManagerBasedRlEnv,
        command_name: str,
        side: str,
        scale: float,
    ) -> torch.Tensor:
        """exp(-scale * quat_angle(mano_wrist, robot_wrist)) for one side."""
        command = _cmd(env, command_name)
        si = _side_idx(command, side)
        error = quat_error_magnitude(
            command.mano_wrist_quat_w[:, si], command.robot_wrist_quat_w[:, si]
        )
        return torch.exp(-scale * error)

    @staticmethod
    def tracking_level_trans_error_exp(
        env: ManagerBasedRlEnv,
        command_name: str,
        side: str,
        level: int,
        scale: float,
    ) -> torch.Tensor:
        """exp(-scale * mean(||mano_joint - robot_body||)) for level 1 or 2, one side."""
        command = _cmd(env, command_name)
        mano_trans = command.mano_level_trans_w(side, level)  # (B, 5, 3)
        robot_trans = command.robot_level_trans_w(side, level)  # (B, 5, 3)
        error = torch.norm(mano_trans - robot_trans, dim=-1)  # (B, 5)
        return torch.exp(-scale * error.mean(dim=-1))

    @staticmethod
    def tracking_joints_vel_error_exp(
        env: ManagerBasedRlEnv,
        command_name: str,
        side: str,
        scale: float,
    ) -> torch.Tensor:
        """exp(-scale * mean(|mano_vel - robot_vel|)) for all 17 bodies, one side.

        ManipTrans: reward_joints_vel = exp(-1 * diff_joints_vel.abs().mean(dim=-1).mean(-1))
        """
        command = _cmd(env, command_name)
        si = _side_idx(command, side)
        body_delta = command.mano_all_joints_lin_vel_w(
            side
        ) - command.robot_all_joints_lin_vel_w(side)  # (B, 12, 3)
        tip_mano_vel = command.mano_tip_lin_vel_w[:, si]  # (B, 5, 3)
        tip_robot_vel = command.robot_all_joints_lin_vel_w(side)[
            :, -5:
        ]  # link2 body vel as approx
        tip_delta = tip_mano_vel - tip_robot_vel  # (B, 5, 3)
        all_delta = torch.cat([body_delta, tip_delta], dim=1)  # (B, 17, 3)
        return torch.exp(-scale * all_delta.abs().mean(dim=-1).mean(dim=-1))

    @staticmethod
    def tracking_obj_trans_error_exp(
        env: ManagerBasedRlEnv,
        command_name: str,
        side: str,
        scale: float,
    ) -> torch.Tensor:
        """exp(-scale * ||ref_obj_trans - sim_obj_trans||) for one side."""
        command = _cmd(env, command_name)
        si = _side_idx(command, side)
        error = torch.norm(
            command.ref_obj_trans_w[:, si] - command.sim_obj_trans_w[:, si], dim=-1
        )
        return torch.exp(-scale * error)

    @staticmethod
    def tracking_obj_rot_error_exp(
        env: ManagerBasedRlEnv,
        command_name: str,
        side: str,
        scale: float,
    ) -> torch.Tensor:
        """exp(-scale * quat_angle(ref_obj, sim_obj)) for one side."""
        command = _cmd(env, command_name)
        si = _side_idx(command, side)
        error = quat_error_magnitude(
            command.ref_obj_quat_w[:, si], command.sim_obj_quat_w[:, si]
        )
        return torch.exp(-scale * error)

    @staticmethod
    def tracking_obj_lin_vel_error_exp(
        env: ManagerBasedRlEnv,
        command_name: str,
        side: str,
        scale: float,
    ) -> torch.Tensor:
        """exp(-scale * mean(|ref_lin_vel - sim_lin_vel|)) for one side."""
        command = _cmd(env, command_name)
        si = _side_idx(command, side)
        error = torch.mean(
            torch.abs(
                command.ref_obj_lin_vel_w[:, si] - command.sim_obj_lin_vel_w[:, si]
            ),
            dim=-1,
        )
        return torch.exp(-scale * error)

    @staticmethod
    def tracking_obj_ang_vel_error_exp(
        env: ManagerBasedRlEnv,
        command_name: str,
        side: str,
        scale: float,
    ) -> torch.Tensor:
        """exp(-scale * mean(|ref_ang_vel - sim_ang_vel|)) for one side."""
        command = _cmd(env, command_name)
        si = _side_idx(command, side)
        error = torch.mean(
            torch.abs(
                command.ref_obj_ang_vel_w[:, si] - command.sim_obj_ang_vel_w[:, si]
            ),
            dim=-1,
        )
        return torch.exp(-scale * error)

    @staticmethod
    def tracking_contact_match(
        env: ManagerBasedRlEnv,
        command_name: str,
        sensor_name: str,
        side: str,
        finger: str,
        beta: float,
        gamma: float,
        tol: float,
    ) -> torch.Tensor:
        """Additive approach-shaping + contact-bonus reward, per finger per frame.

            reward = ref_flag · ( exp(-β · approach_dist)
                                 + found · exp(-γ · max(-dist − tol, 0)) )

        - ref_flag: preprocessed binary flag, 1 when MANO reference says this
          finger should contact at this frame. Whole reward is 0 when
          ref_flag == 0.
        - approach_dist: ``surface_dist`` = ``max(SDF(robot_tip_site), 0)``.
          Distance from the fingertip to the nearest point on the object
          surface, queried against the baked per-side SDF grid. The policy
          can land on any valid surface point rather than being penalised
          for picking one MANO didn't label.
        - found: count of matched contacts on the penetration (mindist)
          sensor; > 0 when the narrowphase has produced contact between the
          fingertip site and the object body.
        - dist: signed distance from the penetration sensor (< 0 when overlap).

        Two branches:
          ref_flag=0          : reward = 0
          ref_flag=1, found=0 : reward = exp(-β · approach_dist)                  (approach)
          ref_flag=1, found=1 : reward = exp(-β · approach_dist)
                                       + exp(-γ · depth_excess)                   (landing)

        Peak 2.0 at clean landing.
        """
        command = _cmd(env, command_name)
        si = _side_idx(command, side)
        fi = command.finger_names.index(finger)

        robot_pt = command.robot_tip_trans_w[:, si, fi]  # (B, 3)
        sdf, _ = command.sdf_query(robot_pt, side)  # (B,)
        approach_dist = torch.clamp(sdf, min=0.0)

        pen_sensor: ContactSensor = env.scene[sensor_name]
        found = (pen_sensor.data.found[:, fi] > 0).to(approach_dist.dtype)  # (B,) 0/1
        pen_dist = pen_sensor.data.dist[:, fi]  # (B,) signed, <0 = overlap
        excess = torch.clamp(-pen_dist - tol, min=0.0)  # (B,)

        shaping = torch.exp(-beta * approach_dist)  # (B,)
        bonus = found * torch.exp(-gamma * excess)  # (B,)

        flag = command.ref_contact_flags[:, si, fi]  # (B,) 0 or 1
        return flag * (shaping + bonus)

    @staticmethod
    def tracking_contact_force(
        env: ManagerBasedRlEnv,
        command_name: str,
        sensor_name: str,
        side: str,
        scale: float,
    ) -> torch.Tensor:
        """ManipTrans contact reward: exp(-scale / (sum_masked_force + eps)).

        Force is masked by fingertip-object distance: only counts when within
        2-3 cm.  ``mask = clamp((0.03 - dist) / 0.01, 0, 1)``.
        """
        command = _cmd(env, command_name)
        si = _side_idx(command, side)

        sensor: ContactSensor = env.scene[sensor_name]
        force = sensor.data.force  # (B, n_primaries, 3)
        force_mag = torch.norm(force, dim=-1)  # (B, n_primaries)

        # Per-finger distance masking using precomputed MANO tip-to-object-surface distance
        tip_dist = command.mano_tips_distance[:, si]  # (B, 5)
        dist_mask = torch.clamp((0.03 - tip_dist) / 0.01, 0.0, 1.0)  # (B, 5)

        masked_force = force_mag * dist_mask  # (B, 5)
        total_force = masked_force.sum(dim=-1)  # (B,)

        return torch.exp(-scale / (total_force + 1e-5))

    # ── Penalties / soft constraints (not tracking errors) ─────────────────

    @staticmethod
    def power_penalty(
        env: ManagerBasedRlEnv,
        command_name: str,
        action_name: str,
        side: str,
        scale: float,
    ) -> torch.Tensor:
        """exp(-scale * sum(|torque * velocity|)) for finger joints, one side."""
        command = _cmd(env, command_name)
        act_term = env.action_manager._terms[action_name]
        finger_ids = act_term._finger_ids
        n_per_side = len(finger_ids) // len(command._side_list)
        si = _side_idx(command, side)
        side_finger_ids = finger_ids[si * n_per_side : (si + 1) * n_per_side]
        torque = command.robot.data.qfrc_actuator[:, side_finger_ids]
        vel = command.robot.data.joint_vel[:, side_finger_ids]
        power = torch.sum(torch.abs(torque * vel), dim=-1)
        return torch.exp(-scale * power)

    @staticmethod
    def wrist_power_penalty(
        env: ManagerBasedRlEnv,
        command_name: str,
        action_name: str,
        side: str,
        scale: float,
    ) -> torch.Tensor:
        """exp(-scale * sum(|torque * velocity|)) for wrist joints, one side."""
        command = _cmd(env, command_name)
        act_term = env.action_manager._terms[action_name]
        wrist_ids = act_term._wrist_ids
        n_per_side = len(wrist_ids) // len(command._side_list)
        si = _side_idx(command, side)
        side_wrist_ids = wrist_ids[si * n_per_side : (si + 1) * n_per_side]
        torque = command.robot.data.qfrc_actuator[:, side_wrist_ids]
        vel = command.robot.data.joint_vel[:, side_wrist_ids]
        power = torch.sum(torch.abs(torque * vel), dim=-1)
        return torch.exp(-scale * power)

    @staticmethod
    def pin_penalty(
        env: ManagerBasedRlEnv,
        command_name: str,
        side: str,
    ) -> torch.Tensor:
        """Per-step flat penalty: 1.0 when pinning fired on this side, else 0.

        Register with a negative weight (``-pin_penalty_weight``) so pin
        firings subtract from the total reward. Under adaptive pinning the
        policy learns to track well enough that the pin never fires — then
        this term contributes 0. Always available regardless of ``pin_mode``;
        ``pin_fired_this_step`` is the per-step buffer the command updates
        in ``_update_command`` before the pin branch executes.
        """
        command = _cmd(env, command_name)
        si = _side_idx(command, side)
        return command.pin_fired_this_step[:, si].to(torch.float32)

    # ── Per-side wrappers (right side) — yaml-facing names ─────────────────

    @staticmethod
    def tracking_r_wrist_trans_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float
    ) -> torch.Tensor:
        return MotionTrackingRewards.tracking_wrist_trans_error_exp(
            env, command_name, side="right", scale=scale
        )

    @staticmethod
    def tracking_r_wrist_rot_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float
    ) -> torch.Tensor:
        return MotionTrackingRewards.tracking_wrist_rot_error_exp(
            env, command_name, side="right", scale=scale
        )

    @staticmethod
    def tracking_r_wrist_lin_vel_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float
    ) -> torch.Tensor:
        return MotionTrackingRewards.tracking_wrist_lin_vel_error_exp(
            env, command_name, side="right", scale=scale
        )

    @staticmethod
    def tracking_r_wrist_ang_vel_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float
    ) -> torch.Tensor:
        return MotionTrackingRewards.tracking_wrist_ang_vel_error_exp(
            env, command_name, side="right", scale=scale
        )

    @staticmethod
    def tracking_r_thumb_tip_trans_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float
    ) -> torch.Tensor:
        return MotionTrackingRewards.tracking_fingertip_trans_error_exp(
            env, command_name, side="right", finger="thumb", scale=scale
        )

    @staticmethod
    def tracking_r_index_tip_trans_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float
    ) -> torch.Tensor:
        return MotionTrackingRewards.tracking_fingertip_trans_error_exp(
            env, command_name, side="right", finger="index", scale=scale
        )

    @staticmethod
    def tracking_r_middle_tip_trans_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float
    ) -> torch.Tensor:
        return MotionTrackingRewards.tracking_fingertip_trans_error_exp(
            env, command_name, side="right", finger="middle", scale=scale
        )

    @staticmethod
    def tracking_r_ring_tip_trans_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float
    ) -> torch.Tensor:
        return MotionTrackingRewards.tracking_fingertip_trans_error_exp(
            env, command_name, side="right", finger="ring", scale=scale
        )

    @staticmethod
    def tracking_r_pinky_tip_trans_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float
    ) -> torch.Tensor:
        return MotionTrackingRewards.tracking_fingertip_trans_error_exp(
            env, command_name, side="right", finger="pinky", scale=scale
        )

    @staticmethod
    def tracking_r_level1_trans_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float
    ) -> torch.Tensor:
        return MotionTrackingRewards.tracking_level_trans_error_exp(
            env, command_name, side="right", level=1, scale=scale
        )

    @staticmethod
    def tracking_r_level2_trans_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float
    ) -> torch.Tensor:
        return MotionTrackingRewards.tracking_level_trans_error_exp(
            env, command_name, side="right", level=2, scale=scale
        )

    @staticmethod
    def r_finger_power_penalty(
        env: ManagerBasedRlEnv, command_name: str, scale: float
    ) -> torch.Tensor:
        return MotionTrackingRewards.power_penalty(
            env, command_name, action_name="maniptrans", side="right", scale=scale
        )

    @staticmethod
    def r_wrist_power_penalty(
        env: ManagerBasedRlEnv, command_name: str, scale: float
    ) -> torch.Tensor:
        return MotionTrackingRewards.wrist_power_penalty(
            env, command_name, action_name="maniptrans", side="right", scale=scale
        )

    @staticmethod
    def tracking_r_joints_vel_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float
    ) -> torch.Tensor:
        return MotionTrackingRewards.tracking_joints_vel_error_exp(
            env, command_name, side="right", scale=scale
        )

    @staticmethod
    def tracking_r_thumb_contact_match(
        env: ManagerBasedRlEnv,
        command_name: str,
        beta: float,
        gamma: float,
        tol: float,
    ) -> torch.Tensor:
        return MotionTrackingRewards.tracking_contact_match(
            env,
            command_name,
            sensor_name="r_fingertip_penetration",
            side="right",
            finger="thumb",
            beta=beta,
            gamma=gamma,
            tol=tol,
        )

    @staticmethod
    def tracking_r_index_contact_match(
        env: ManagerBasedRlEnv,
        command_name: str,
        beta: float,
        gamma: float,
        tol: float,
    ) -> torch.Tensor:
        return MotionTrackingRewards.tracking_contact_match(
            env,
            command_name,
            sensor_name="r_fingertip_penetration",
            side="right",
            finger="index",
            beta=beta,
            gamma=gamma,
            tol=tol,
        )

    @staticmethod
    def tracking_r_middle_contact_match(
        env: ManagerBasedRlEnv,
        command_name: str,
        beta: float,
        gamma: float,
        tol: float,
    ) -> torch.Tensor:
        return MotionTrackingRewards.tracking_contact_match(
            env,
            command_name,
            sensor_name="r_fingertip_penetration",
            side="right",
            finger="middle",
            beta=beta,
            gamma=gamma,
            tol=tol,
        )

    @staticmethod
    def tracking_r_ring_contact_match(
        env: ManagerBasedRlEnv,
        command_name: str,
        beta: float,
        gamma: float,
        tol: float,
    ) -> torch.Tensor:
        return MotionTrackingRewards.tracking_contact_match(
            env,
            command_name,
            sensor_name="r_fingertip_penetration",
            side="right",
            finger="ring",
            beta=beta,
            gamma=gamma,
            tol=tol,
        )

    @staticmethod
    def tracking_r_pinky_contact_match(
        env: ManagerBasedRlEnv,
        command_name: str,
        beta: float,
        gamma: float,
        tol: float,
    ) -> torch.Tensor:
        return MotionTrackingRewards.tracking_contact_match(
            env,
            command_name,
            sensor_name="r_fingertip_penetration",
            side="right",
            finger="pinky",
            beta=beta,
            gamma=gamma,
            tol=tol,
        )

    @staticmethod
    def tracking_r_obj_trans_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float
    ) -> torch.Tensor:
        return MotionTrackingRewards.tracking_obj_trans_error_exp(
            env, command_name, side="right", scale=scale
        )

    @staticmethod
    def tracking_r_obj_rot_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float
    ) -> torch.Tensor:
        return MotionTrackingRewards.tracking_obj_rot_error_exp(
            env, command_name, side="right", scale=scale
        )

    @staticmethod
    def tracking_r_obj_lin_vel_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float
    ) -> torch.Tensor:
        return MotionTrackingRewards.tracking_obj_lin_vel_error_exp(
            env, command_name, side="right", scale=scale
        )

    @staticmethod
    def tracking_r_obj_ang_vel_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float
    ) -> torch.Tensor:
        return MotionTrackingRewards.tracking_obj_ang_vel_error_exp(
            env, command_name, side="right", scale=scale
        )
