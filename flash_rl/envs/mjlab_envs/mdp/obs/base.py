from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class BaseObs:
    """Embodiment-independent obs terms (proprio, last action, contact sensors)."""

    @staticmethod
    def robot_joint_pos(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Robot finger joint positions. Shape: (B, n_dofs)."""
        robot: Entity = env.scene[asset_cfg.name]
        return robot.data.joint_pos[:, asset_cfg.joint_ids]

    @staticmethod
    def robot_joint_cos_sin(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Cos and sin of finger joint positions. Shape: (B, 2*n_dofs)."""
        robot: Entity = env.scene[asset_cfg.name]
        q = robot.data.joint_pos[:, asset_cfg.joint_ids]
        return torch.cat([torch.cos(q), torch.sin(q)], dim=-1)

    @staticmethod
    def wrist_state(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Wrist state with zeroed pos + quat + lin_vel + ang_vel. Shape: (B, 13)."""
        robot: Entity = env.scene[asset_cfg.name]
        return torch.cat(
            [
                torch.zeros_like(robot.data.root_link_pos_w),
                robot.data.root_link_quat_w,
                robot.data.root_link_lin_vel_w,
                robot.data.root_link_ang_vel_w,
            ],
            dim=-1,
        )

    @staticmethod
    def robot_joint_vel(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Robot joint velocities. Shape: (B, n_dofs)."""
        robot: Entity = env.scene[asset_cfg.name]
        return robot.data.joint_vel[:, asset_cfg.joint_ids]

    @staticmethod
    def last_action(
        env: ManagerBasedRlEnv,
        action_name: str = "maniptrans",
    ) -> torch.Tensor:
        """First n_dofs dims of the raw policy action. Shape: (B, n_dofs)."""
        action_term = env.action_manager.get_term(action_name)
        return env.action_manager.action[:, : action_term.n_dofs]

    @staticmethod
    def contact_found_history(
        env: ManagerBasedRlEnv,
        sensor_name: str,
        history_len: int,
    ) -> torch.Tensor:
        """Rolling history of per-finger contact ``found`` flag. Shape: (B, history_len*n_primaries)."""
        key = f"_contact_found_history_{sensor_name}"
        sensor: ContactSensor = env.scene[sensor_name]
        current = (sensor.data.found > 0).to(torch.float)
        n_primaries = current.shape[1]

        if key not in env.extras:
            env.extras[key] = torch.ones(
                env.num_envs, history_len, n_primaries, device=env.device, dtype=torch.float
            )

        buf = env.extras[key]
        reset_mask = env.episode_length_buf == 1
        if reset_mask.any():
            buf[reset_mask] = 1.0

        buf = torch.cat([buf[:, 1:], current[:, None]], dim=1)
        env.extras[key] = buf
        return buf.reshape(env.num_envs, -1)

    @staticmethod
    def tip_penetration(
        env: ManagerBasedRlEnv,
        sensor_name: str,
        dist_clamp_min: float = -0.02,
        dist_clamp_max: float = 0.05,
    ) -> torch.Tensor:
        """Per-finger signed penetration distance + found flag. Shape: (B, n_primaries*2)."""
        sensor: ContactSensor = env.scene[sensor_name]
        dist = sensor.data.dist.clamp(dist_clamp_min, dist_clamp_max)
        found = (sensor.data.found > 0).to(dist.dtype)
        out = torch.stack([dist, found], dim=-1)
        return out.reshape(out.shape[0], -1)

    @staticmethod
    def r_tip_penetration(env: ManagerBasedRlEnv) -> torch.Tensor:
        return BaseObs.tip_penetration(env, sensor_name="r_fingertip_penetration")
