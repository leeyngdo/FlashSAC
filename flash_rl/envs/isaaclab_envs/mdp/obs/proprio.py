"""Proprioceptive observation terms re-exported from IsaacLab builtins.

These are the standard ManagerBased observation functions used by the tracking
policy/critic observation groups. They are re-exported here so the
:data:`OBS_TERMS` registry and ``from .proprio import *`` resolve them as local
module-level names. This module imports ``isaaclab`` and is therefore NOT part
of the import-light path.
"""

from __future__ import annotations

from isaaclab.envs.mdp import (
    base_ang_vel,
    base_lin_vel,
    generated_commands,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
)

__all__ = [
    "base_ang_vel",
    "base_lin_vel",
    "generated_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
]
