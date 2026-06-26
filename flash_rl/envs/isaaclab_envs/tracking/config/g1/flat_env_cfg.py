"""G1-specific tracking environment configurations.

Vendored from BeyondMimic (``whole_body_tracking``) and rewritten to the FlashSAC
modular layout. The low-frequency PPO variant and all ``rsl_rl`` plumbing are dropped;
only the flat tracking config and its without-state-estimation variant remain.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from flash_rl.envs.isaaclab_envs.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from flash_rl.envs.isaaclab_envs.tracking.config.g1.robot import apply_g1_tracking_profile
from flash_rl.envs.isaaclab_envs.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    """Flat-terrain motion-tracking config for the Unitree G1.

    Binds the G1 articulation, action scale, and body/joint groups to the
    generic tracking assembler.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        apply_g1_tracking_profile(self)


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
