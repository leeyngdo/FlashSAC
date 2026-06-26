"""Observation term registry for the motion-tracking task.

Aggregates the custom motion observation terms (:mod:`.motion`) and the
proprioceptive IsaacLab builtins (:mod:`.proprio`) into a single
:data:`OBS_TERMS` registry mapping a string name to its term function.

To add an observation: write the function in ``mdp/obs/<file>.py``, register it
in :data:`OBS_TERMS` here, then add an ``ObsTerm`` referencing it in
``tracking/observations_cfg.py``.
"""

from __future__ import annotations

from collections.abc import Callable

# Motion (custom tracking terms)
from .motion import (
    motion_anchor_ori_b,
    motion_anchor_pos_b,
    robot_anchor_ang_vel_w,
    robot_anchor_lin_vel_w,
    robot_anchor_ori_w,
    robot_body_ori_b,
    robot_body_pos_b,
)

# Proprioception (IsaacLab builtins)
from .proprio import (
    base_ang_vel,
    base_lin_vel,
    generated_commands,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
)

OBS_TERMS: dict[str, Callable] = {
    # Motion (custom tracking terms)
    "motion_anchor_pos_b": motion_anchor_pos_b,
    "motion_anchor_ori_b": motion_anchor_ori_b,
    "robot_anchor_ori_w": robot_anchor_ori_w,
    "robot_anchor_lin_vel_w": robot_anchor_lin_vel_w,
    "robot_anchor_ang_vel_w": robot_anchor_ang_vel_w,
    "robot_body_pos_b": robot_body_pos_b,
    "robot_body_ori_b": robot_body_ori_b,
    # Proprioception (IsaacLab builtins)
    "generated_commands": generated_commands,
    "base_lin_vel": base_lin_vel,
    "base_ang_vel": base_ang_vel,
    "joint_pos_rel": joint_pos_rel,
    "joint_vel_rel": joint_vel_rel,
    "last_action": last_action,
}
