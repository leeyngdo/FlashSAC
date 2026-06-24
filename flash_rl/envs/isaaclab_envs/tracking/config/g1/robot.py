"""Unitree G1 tracking profile."""

from __future__ import annotations

from typing import Any

from isaaclab.managers import SceneEntityCfg

ANCHOR_BODY_NAME = "torso_link"
LOCAL_REWARD_ANCHOR_BODY_NAME = "pelvis"

TRACKED_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

END_EFFECTOR_BODY_NAMES = [
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]

END_EFFECTOR_BODY_OFFSETS = [
    [0.18, -0.025, 0.0],
    [0.18, 0.025, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
]

ANTI_SHAKE_BODY_NAMES = [
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "head_link",
]

FEET_JOINT_NAMES = [".*ankle.*"]
BASE_COM_BODY_NAME = "torso_link"

CONTACT_PENALTY_ALLOWED_BODIES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

UNDESIRED_CONTACT_BODY_REGEX = (
    r"^" + "".join(f"(?!{body}$)" for body in CONTACT_PENALTY_ALLOWED_BODIES) + r".+$"
)


def apply_g1_tracking_profile(env_cfg: Any) -> None:
    """Apply Unitree G1 body and joint groups to the generic tracking cfg."""
    env_cfg.commands.motion.anchor_body_name = ANCHOR_BODY_NAME
    env_cfg.commands.motion.body_names = list(TRACKED_BODY_NAMES)

    env_cfg.rewards.motion_ee_body_pos.params["body_names"] = list(END_EFFECTOR_BODY_NAMES)
    env_cfg.rewards.motion_ee_body_pos.params["body_offsets"] = [list(offset) for offset in END_EFFECTOR_BODY_OFFSETS]
    env_cfg.rewards.motion_ee_body_pos.params["anchor_body_name"] = LOCAL_REWARD_ANCHOR_BODY_NAME
    env_cfg.rewards.anti_shake_ang_vel.params["body_names"] = list(ANTI_SHAKE_BODY_NAMES)
    env_cfg.rewards.feet_acc.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=list(FEET_JOINT_NAMES))
    env_cfg.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=[UNDESIRED_CONTACT_BODY_REGEX]
    )

    env_cfg.terminations.ee_body_pos.params["body_names"] = list(END_EFFECTOR_BODY_NAMES)
    env_cfg.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names=BASE_COM_BODY_NAME)
