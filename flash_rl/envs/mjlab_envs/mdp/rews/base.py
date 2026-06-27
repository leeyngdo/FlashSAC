from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp import rewards as _mdp_rewards
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class BaseRewards:
    """mjlab built-in MDP regularizers exposed under yaml-friendly names."""

    @staticmethod
    def action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
        return _mdp_rewards.action_rate_l2(env)

    @staticmethod
    def joint_limits(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        return _mdp_rewards.joint_pos_limits(env, asset_cfg=asset_cfg)
