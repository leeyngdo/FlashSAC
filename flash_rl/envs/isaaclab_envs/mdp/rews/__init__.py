"""Reward-term registry for the motion-tracking task.

Exposes ``REW_TERMS``: a mapping from reward-function name to its callable, used for introspection
and lookup. Rewards are a single flat (single-critic) set summed by IsaacLab's ``RewardManager``.

To add a reward: write the function in ``mdp/rews/<file>.py``, add it to ``REW_TERMS`` here, then
add a ``RewTerm`` field to ``tracking/rewards_cfg.py``.
"""

from __future__ import annotations

from typing import Callable

# Regularization
from .regularization import action_rate_l2, joint_acc_l2, joint_pos_limits

# Safety
from .safety import feet_contact_time, undesired_contacts

# Tracking
from .tracking import (
    anti_shake_ang_vel_l2,
    motion_global_anchor_orientation_error_exp,
    motion_global_anchor_position_error_exp,
    motion_global_body_angular_velocity_error_exp,
    motion_global_body_linear_velocity_error_exp,
    motion_local_body_position_error_exp,
    motion_relative_body_orientation_error_exp,
    motion_relative_body_position_error_exp,
)

REW_TERMS: dict[str, Callable] = {
    # Tracking
    "motion_global_anchor_position_error_exp": motion_global_anchor_position_error_exp,
    "motion_global_anchor_orientation_error_exp": motion_global_anchor_orientation_error_exp,
    "motion_relative_body_position_error_exp": motion_relative_body_position_error_exp,
    "motion_local_body_position_error_exp": motion_local_body_position_error_exp,
    "motion_relative_body_orientation_error_exp": motion_relative_body_orientation_error_exp,
    "motion_global_body_linear_velocity_error_exp": motion_global_body_linear_velocity_error_exp,
    "motion_global_body_angular_velocity_error_exp": motion_global_body_angular_velocity_error_exp,
    "anti_shake_ang_vel_l2": anti_shake_ang_vel_l2,
    # Regularization
    "action_rate_l2": action_rate_l2,
    "joint_acc_l2": joint_acc_l2,
    "joint_pos_limits": joint_pos_limits,
    # Safety
    "undesired_contacts": undesired_contacts,
    "feet_contact_time": feet_contact_time,
}
