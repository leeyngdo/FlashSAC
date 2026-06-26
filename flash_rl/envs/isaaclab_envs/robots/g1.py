"""Unitree G1 articulation configuration factory.

This module vendors the BeyondMimic G1 cylinder articulation configuration and refactors the
module-level constants of the upstream definition into a :func:`get_g1_cfg` factory so the robot
configuration can be parameterized at the code level (natural frequency, damping ratio, initial
pose, and soft joint limit factor) while still exposing byte-for-byte identical defaults via the
module-level :data:`G1_CYLINDER_CFG` and :data:`G1_ACTION_SCALE`.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from flash_rl.envs.isaaclab_envs.assets import ASSET_DIR

# Armature constants for the Unitree G1 actuator families.
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

# Default PD-tuning targets used to derive stiffness/damping from armature.
NATURAL_FREQ_DEFAULT = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO_DEFAULT = 2.0

# Back-compat aliases matching the upstream WBT names.
NATURAL_FREQ = NATURAL_FREQ_DEFAULT
DAMPING_RATIO = DAMPING_RATIO_DEFAULT


def get_g1_cfg(
    natural_freq: float = NATURAL_FREQ_DEFAULT,
    damping_ratio: float = DAMPING_RATIO_DEFAULT,
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.76),
    soft_joint_pos_limit_factor: float = 0.9,
) -> ArticulationCfg:
    """Build the Unitree G1 cylinder articulation configuration.

    Stiffness and damping are derived from the per-family armature values using the upstream
    relations ``stiffness = armature * natural_freq**2`` and
    ``damping = 2 * damping_ratio * armature * natural_freq``. With the default arguments the
    returned configuration is byte-for-byte identical to the BeyondMimic upstream definition.

    Args:
        natural_freq: Target natural frequency (rad/s) used to derive joint stiffness.
        damping_ratio: Target damping ratio used to derive joint damping.
        init_pos: Initial base position ``(x, y, z)`` of the articulation in meters.
        soft_joint_pos_limit_factor: Fraction of the hard joint position limits used as soft limits.

    Returns:
        The configured :class:`~isaaclab.assets.articulation.ArticulationCfg` for the G1 robot.
    """
    nf = natural_freq
    dr = damping_ratio

    stiffness_5020 = ARMATURE_5020 * nf**2
    stiffness_7520_14 = ARMATURE_7520_14 * nf**2
    stiffness_7520_22 = ARMATURE_7520_22 * nf**2
    stiffness_4010 = ARMATURE_4010 * nf**2

    damping_5020 = 2.0 * dr * ARMATURE_5020 * nf
    damping_7520_14 = 2.0 * dr * ARMATURE_7520_14 * nf
    damping_7520_22 = 2.0 * dr * ARMATURE_7520_22 * nf
    damping_4010 = 2.0 * dr * ARMATURE_4010 * nf

    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            fix_base=False,
            replace_cylinders_with_capsules=True,
            asset_path=f"{ASSET_DIR}/unitree_description/urdf/g1/main.urdf",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            joint_pos={
                ".*_hip_pitch_joint": -0.312,
                ".*_knee_joint": 0.669,
                ".*_ankle_pitch_joint": -0.363,
                ".*_elbow_joint": 0.6,
                "left_shoulder_roll_joint": 0.2,
                "left_shoulder_pitch_joint": 0.2,
                "right_shoulder_roll_joint": -0.2,
                "right_shoulder_pitch_joint": 0.2,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                ],
                effort_limit_sim={
                    ".*_hip_yaw_joint": 88.0,
                    ".*_hip_roll_joint": 139.0,
                    ".*_hip_pitch_joint": 88.0,
                    ".*_knee_joint": 139.0,
                },
                velocity_limit_sim={
                    ".*_hip_yaw_joint": 32.0,
                    ".*_hip_roll_joint": 20.0,
                    ".*_hip_pitch_joint": 32.0,
                    ".*_knee_joint": 20.0,
                },
                stiffness={
                    ".*_hip_pitch_joint": stiffness_7520_14,
                    ".*_hip_roll_joint": stiffness_7520_22,
                    ".*_hip_yaw_joint": stiffness_7520_14,
                    ".*_knee_joint": stiffness_7520_22,
                },
                damping={
                    ".*_hip_pitch_joint": damping_7520_14,
                    ".*_hip_roll_joint": damping_7520_22,
                    ".*_hip_yaw_joint": damping_7520_14,
                    ".*_knee_joint": damping_7520_22,
                },
                armature={
                    ".*_hip_pitch_joint": ARMATURE_7520_14,
                    ".*_hip_roll_joint": ARMATURE_7520_22,
                    ".*_hip_yaw_joint": ARMATURE_7520_14,
                    ".*_knee_joint": ARMATURE_7520_22,
                },
            ),
            "feet": ImplicitActuatorCfg(
                effort_limit_sim=50.0,
                velocity_limit_sim=37.0,
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                stiffness=2.0 * stiffness_5020,
                damping=2.0 * damping_5020,
                armature=2.0 * ARMATURE_5020,
            ),
            "waist": ImplicitActuatorCfg(
                effort_limit_sim=50,
                velocity_limit_sim=37.0,
                joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
                stiffness=2.0 * stiffness_5020,
                damping=2.0 * damping_5020,
                armature=2.0 * ARMATURE_5020,
            ),
            "waist_yaw": ImplicitActuatorCfg(
                effort_limit_sim=88,
                velocity_limit_sim=32.0,
                joint_names_expr=["waist_yaw_joint"],
                stiffness=stiffness_7520_14,
                damping=damping_7520_14,
                armature=ARMATURE_7520_14,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint",
                ],
                effort_limit_sim={
                    ".*_shoulder_pitch_joint": 25.0,
                    ".*_shoulder_roll_joint": 25.0,
                    ".*_shoulder_yaw_joint": 25.0,
                    ".*_elbow_joint": 25.0,
                    ".*_wrist_roll_joint": 25.0,
                    ".*_wrist_pitch_joint": 5.0,
                    ".*_wrist_yaw_joint": 5.0,
                },
                velocity_limit_sim={
                    ".*_shoulder_pitch_joint": 37.0,
                    ".*_shoulder_roll_joint": 37.0,
                    ".*_shoulder_yaw_joint": 37.0,
                    ".*_elbow_joint": 37.0,
                    ".*_wrist_roll_joint": 37.0,
                    ".*_wrist_pitch_joint": 22.0,
                    ".*_wrist_yaw_joint": 22.0,
                },
                stiffness={
                    ".*_shoulder_pitch_joint": stiffness_5020,
                    ".*_shoulder_roll_joint": stiffness_5020,
                    ".*_shoulder_yaw_joint": stiffness_5020,
                    ".*_elbow_joint": stiffness_5020,
                    ".*_wrist_roll_joint": stiffness_5020,
                    ".*_wrist_pitch_joint": stiffness_4010,
                    ".*_wrist_yaw_joint": stiffness_4010,
                },
                damping={
                    ".*_shoulder_pitch_joint": damping_5020,
                    ".*_shoulder_roll_joint": damping_5020,
                    ".*_shoulder_yaw_joint": damping_5020,
                    ".*_elbow_joint": damping_5020,
                    ".*_wrist_roll_joint": damping_5020,
                    ".*_wrist_pitch_joint": damping_4010,
                    ".*_wrist_yaw_joint": damping_4010,
                },
                armature={
                    ".*_shoulder_pitch_joint": ARMATURE_5020,
                    ".*_shoulder_roll_joint": ARMATURE_5020,
                    ".*_shoulder_yaw_joint": ARMATURE_5020,
                    ".*_elbow_joint": ARMATURE_5020,
                    ".*_wrist_roll_joint": ARMATURE_5020,
                    ".*_wrist_pitch_joint": ARMATURE_4010,
                    ".*_wrist_yaw_joint": ARMATURE_4010,
                },
            ),
        },
    )


def get_g1_action_scale(cfg: ArticulationCfg) -> dict[str, float]:
    """Derive the per-joint action scale from an articulation configuration.

    The action scale for each joint is ``0.25 * effort_limit_sim / stiffness``, matching the
    BeyondMimic upstream derivation. Effort limits and stiffness may be specified either as a
    scalar (broadcast to every joint of the actuator) or a per-joint dictionary.

    Args:
        cfg: The articulation configuration to derive the action scale from.

    Returns:
        Mapping from joint-name expression to its action scale.
    """
    action_scale: dict[str, float] = {}
    for a in cfg.actuators.values():
        e = a.effort_limit_sim
        s = a.stiffness
        names = a.joint_names_expr
        if not isinstance(e, dict):
            e = {n: e for n in names}
        if not isinstance(s, dict):
            s = {n: s for n in names}
        for n in names:
            if n in e and n in s and s[n]:
                action_scale[n] = 0.25 * e[n] / s[n]
    return action_scale


# Module-level defaults (byte-for-byte identical to the BeyondMimic upstream definition).
G1_CYLINDER_CFG = get_g1_cfg()
G1_ACTION_SCALE = get_g1_action_scale(G1_CYLINDER_CFG)
