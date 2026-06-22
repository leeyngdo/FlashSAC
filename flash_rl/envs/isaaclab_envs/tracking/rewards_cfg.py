"""Reward configuration for the G1 motion-tracking task.

A single flat (single-critic) ``RewardsCfg``: every :class:`RewTerm` is summed by IsaacLab's
``RewardManager`` into one scalar reward, which is what FlashSAC SAC consumes. Per-term values are
overridable from the env config / CLI (``env.reward.<term>.{weight,std,enabled}``), and each term is
logged separately as ``rewards/<term>``.

Default weights are the holosoma FastSAC preset (action_rate_l2=-1.0, motion_body_pos=2.0,
motion_global_anchor_pos=1.0), which DIFFER from the raw WBT values. The ``std`` params match WBT.

Adding a reward is a 3-step recipe: write the func in ``mdp/rews/<file>.py`` -> add it to
``REW_TERMS`` -> add a :class:`RewTerm` field here.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .. import mdp as mdp

# Bodies excluded from the undesired-contacts penalty (WBT 4-body regex for the
# vendored G1 URDF: ankle-roll feet links + wrist-yaw end-effector links).
_UNDESIRED_CONTACT_REGEX = (
    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
    r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
)


@configclass
class RewardsCfg:
    """Flat single-critic reward terms for the motion-tracking MDP (summed to one scalar)."""

    # --- Motion tracking ---
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=2.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    # --- Regularization ---
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    # --- Safety ---
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[_UNDESIRED_CONTACT_REGEX]),
            "threshold": 1.0,
        },
    )
