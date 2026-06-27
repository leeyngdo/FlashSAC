from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_error_magnitude

from ..cmds.motion_tracking import MotionTrackingCommand
from .base import BaseTerminations

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _cmd(env: ManagerBasedRlEnv, command_name: str) -> MotionTrackingCommand:
    return cast(MotionTrackingCommand, env.command_manager.get_term(command_name))


class MotionTrackingTerminations(BaseTerminations):
    """Motion-tracking terminations: pulls current refs from MotionTrackingCommand."""

    @staticmethod
    def fingertip_diverged(
        env: ManagerBasedRlEnv,
        command_name: str,
        threshold: float,
        grace_steps: int,
    ) -> torch.Tensor:
        """Terminate if ANY fingertip error exceeds threshold (any side).
        Skipped during first ``grace_steps`` of each episode."""
        command = _cmd(env, command_name)
        error = torch.norm(command.mano_tip_trans_w - command.robot_tip_trans_w, dim=-1)
        exceeded = torch.any(error.reshape(error.shape[0], -1) > threshold, dim=-1)
        return exceeded & (env.episode_length_buf >= grace_steps)

    @staticmethod
    def obj_trans_diverged(
        env: ManagerBasedRlEnv,
        command_name: str,
        threshold: float,
        grace_steps: int,
    ) -> torch.Tensor:
        """Terminate if object position error exceeds threshold (any side)."""
        command = _cmd(env, command_name)
        error = torch.norm(command.ref_obj_trans_w - command.sim_obj_trans_w, dim=-1)
        exceeded = torch.any(error > threshold, dim=-1)
        return exceeded & (env.episode_length_buf >= grace_steps)

    @staticmethod
    def obj_rot_diverged(
        env: ManagerBasedRlEnv,
        command_name: str,
        threshold_deg: float,
        grace_steps: int,
    ) -> torch.Tensor:
        """Terminate if object rotation error exceeds threshold (any side)."""
        command = _cmd(env, command_name)
        error_rad = quat_error_magnitude(command.ref_obj_quat_w, command.sim_obj_quat_w)
        threshold_rad = threshold_deg * math.pi / 180.0
        exceeded = torch.any(error_rad > threshold_rad, dim=-1)
        return exceeded & (env.episode_length_buf >= grace_steps)

    @staticmethod
    def obj_lin_vel_sanity(
        env: ManagerBasedRlEnv,
        command_name: str,
        max_obj_lin_vel: float,
    ) -> torch.Tensor:
        """Terminate if any per-side object linear velocity exceeds threshold."""
        command = _cmd(env, command_name)
        return torch.any(
            torch.norm(command.sim_obj_lin_vel_w, dim=-1) > max_obj_lin_vel, dim=-1
        )

    @staticmethod
    def obj_ang_vel_sanity(
        env: ManagerBasedRlEnv,
        command_name: str,
        max_obj_ang_vel: float,
    ) -> torch.Tensor:
        """Terminate if any per-side object angular velocity exceeds threshold."""
        command = _cmd(env, command_name)
        return torch.any(
            torch.norm(command.sim_obj_ang_vel_w, dim=-1) > max_obj_ang_vel, dim=-1
        )

    @staticmethod
    def nan_guard(
        env: ManagerBasedRlEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Terminate envs whose simulation state contains NaN/Inf.

        Catches what ``*_sanity`` cannot: PyTorch comparison operators return
        False on NaN, so velocity caps silently fail. This explicit isnan
        check fires on the next step's termination cycle, after which the env
        is reset to a clean state.
        """
        command = _cmd(env, command_name)
        hand: Entity = env.scene[asset_cfg.name]
        bad_joint = torch.isnan(hand.data.joint_pos).any(dim=-1) | torch.isnan(
            hand.data.joint_vel
        ).any(dim=-1)
        obj_trans = command.sim_obj_trans_w
        obj_lin_vel = command.sim_obj_lin_vel_w
        bad_obj = torch.isnan(obj_trans).flatten(1).any(dim=-1) | torch.isnan(
            obj_lin_vel
        ).flatten(1).any(dim=-1)
        return bad_joint | bad_obj
