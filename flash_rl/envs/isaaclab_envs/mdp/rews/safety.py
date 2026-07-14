"""Safety reward terms.

Provides the custom ``feet_contact_time`` helper plus a re-export of the
IsaacLab built-in ``undesired_contacts`` penalty so both are addressable through
the local reward registry.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp import undesired_contacts
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["feet_contact_time", "undesired_contacts"]


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Rewards feet that lift off after a short ground-contact duration.

    Args:
        env: The environment instance.
        sensor_cfg: The contact-sensor entity config selecting the foot bodies.
        threshold: The contact-time threshold (in seconds) below which a lift-off
            is rewarded.

    Returns:
        The reward of shape ``(num_envs,)``.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward
