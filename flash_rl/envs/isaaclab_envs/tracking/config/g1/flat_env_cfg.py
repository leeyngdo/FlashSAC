"""G1-specific tracking environment configurations.

Vendored from BeyondMimic (``whole_body_tracking``) and rewritten to the FlashSAC
modular layout. The low-frequency PPO variant and all ``rsl_rl`` plumbing are dropped;
only the flat tracking config and its without-state-estimation variant remain.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from flash_rl.envs.isaaclab_envs.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from flash_rl.envs.isaaclab_envs.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    """Flat-terrain motion-tracking config for the Unitree G1.

    Binds the G1 articulation and action scale to the generic tracking assembler and
    wires the motion command to the G1 anchor body plus the 14 tracked body links.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
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


@configclass
class G1FlatWoStateEstimationEnvCfg(G1FlatEnvCfg):
    """Without-state-estimation variant.

    Disables the policy observations that depend on base-frame state estimation
    (the motion anchor position in the base frame and the base linear velocity).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
