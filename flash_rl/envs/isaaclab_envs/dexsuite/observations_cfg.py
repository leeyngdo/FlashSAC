"""Observation configuration for the FlashSAC dexsuite Kuka-Allegro task.

The FlashSAC IsaacLab wrapper only consumes the ``policy`` (and optional ``critic``) group, so
the stock ``policy``/``proprio``/``perception`` groups are merged into a single flat ``policy``
group. Term names must match the stock task so the ADR noise curriculum can be retargeted by
name (see :mod:`.dexsuite_env_cfg`). Term functions come from the :data:`...mdp.OBS_TERMS`
registry.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .. import mdp as mdp

# Finger-tip contact sensors created by the stock KukaAllegroMixinCfg.
_FINGER_TIP_CONTACT_SENSORS = [
    "index_link_3_object_s",
    "middle_link_3_object_s",
    "ring_link_3_object_s",
    "thumb_link_3_object_s",
]


@configclass
class DexsuiteKukaAllegroFlatObservationsCfg:
    """Stock observations merged into a single flat ``policy`` group."""

    @configclass
    class PolicyCfg(ObsGroup):
        """The zero-magnitude noise placeholders are ramped up at runtime by the ADR curriculum."""

        object_quat_b = ObsTerm(func=mdp.OBS_TERMS["object_quat_b"], noise=Unoise(n_min=-0.0, n_max=0.0))
        target_object_pose_b = ObsTerm(func=mdp.OBS_TERMS["generated_commands"], params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.OBS_TERMS["last_action"])
        joint_pos = ObsTerm(func=mdp.OBS_TERMS["joint_pos"], noise=Unoise(n_min=-0.0, n_max=0.0))
        joint_vel = ObsTerm(func=mdp.OBS_TERMS["joint_vel"], noise=Unoise(n_min=-0.0, n_max=0.0))
        hand_tips_state_b = ObsTerm(
            func=mdp.OBS_TERMS["body_state_b"],
            noise=Unoise(n_min=-0.0, n_max=0.0),
            # good behaving number for position in m, velocity in m/s, rad/s,
            # and quaternion are unlikely to exceed -2 to 2 range
            clip=(-2.0, 2.0),
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names=["palm_link", ".*_tip"]),
                "base_asset_cfg": SceneEntityCfg("robot"),
            },
        )
        contact = ObsTerm(
            func=mdp.OBS_TERMS["fingers_contact_force_b"],
            params={"contact_sensor_names": _FINGER_TIP_CONTACT_SENSORS},
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )
        object_point_cloud = ObsTerm(
            func=mdp.OBS_TERMS["object_point_cloud_b"],
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-2.0, 2.0),  # clamp between -2 m to 2 m
            params={"num_points": 64, "flatten": True, "visualize": False},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()
