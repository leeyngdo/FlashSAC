from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.terminations import time_out as _mjlab_time_out
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class BaseTerminations:
    """Command-independent termination terms (velocity / joint-vel sanity)."""

    # mjlab built-in: flag-only term that marks episode truncation; the
    # termination manager handles the rest via the ``time_out`` cfg field.
    time_out = _mjlab_time_out

    @staticmethod
    def velocity_diverged(
        env: ManagerBasedRlEnv,
        max_lin_vel: float,
        max_ang_vel: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Terminate if root link velocity is unreasonably high."""
        hand: Entity = env.scene[asset_cfg.name]
        lin = torch.norm(hand.data.root_link_lin_vel_w, dim=-1)
        ang = torch.norm(hand.data.root_link_ang_vel_w, dim=-1)
        return (lin > max_lin_vel) | (ang > max_ang_vel)

    @staticmethod
    def joint_vel_sanity(
        env: ManagerBasedRlEnv,
        max_joint_vel: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Terminate if any joint velocity norm exceeds threshold."""
        hand: Entity = env.scene[asset_cfg.name]
        return torch.norm(hand.data.joint_vel, dim=-1) > max_joint_vel

    @staticmethod
    def joint_vel_mean_sanity(
        env: ManagerBasedRlEnv,
        max_joint_vel_mean: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Terminate if mean absolute joint velocity exceeds threshold.

        ManipTrans: ``torch.abs(current_dof_vel).mean(-1) > 200``.
        """
        hand: Entity = env.scene[asset_cfg.name]
        return hand.data.joint_vel.abs().mean(dim=-1) > max_joint_vel_mean
