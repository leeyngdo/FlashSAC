"""XHand right-hand robot cfg + meta (declarative port).

Ports DexManip's ``src/envs/scene/_robot_setup.py`` (``build_robot`` and its
``_build_init_state`` / ``_build_articulation`` / ``_build_actuator`` /
``_build_collisions`` helpers) together with the yaml it consumed,
``config/envs/robot/xhand_ghost_right.yaml``.

Instead of parsing the yaml + dispatching through ``build_robot``, the yaml
values are inlined as Python literals here and the mjlab ``EntityCfg`` is
constructed exactly like ``build_robot`` would have.

Exposes:
  - ``XHAND_ROBOT_CFG`` : ``mjlab.entity.EntityCfg`` for the right hand.
  - ``XHAND_META``      : dict with keys ``entity_name``, ``body_names``,
                          ``site_names``, ``joint_names``, ``body_mapping``
                          (the non-EntityCfg fields of the robot yaml block,
                          consumed downstream by the scene/commands builders).

The DexManip yaml's ``xml_path`` was CWD-relative (``assets/robot/xhand/
right.xml``). Here it is resolved to the vendored MJCF under this package via
an absolute path so the module is import-location independent.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

# Absolute path to the vendored MJCF (meshdir="meshes/" resolves relative to it).
_XML_PATH = str(
    (Path(__file__).parent.parent / "assets/robot/xhand/right.xml").resolve()
)


# ─── Robot EntityCfg (ports build_robot) ─────────────────────────────────────
# init_state: root pose/vel identity + every actuated joint at 0.0
# (yaml init_state.pos/rot/lin_vel/ang_vel/joint_pos/joint_vel).
_JOINT_NAMES_FINGER = (
    # Thumb (3)
    "right_hand_thumb_bend_joint",
    "right_hand_thumb_rota_joint1",
    "right_hand_thumb_rota_joint2",
    # Index (3)
    "right_hand_index_bend_joint",
    "right_hand_index_joint1",
    "right_hand_index_joint2",
    # Middle (2)
    "right_hand_mid_joint1",
    "right_hand_mid_joint2",
    # Ring (2)
    "right_hand_ring_joint1",
    "right_hand_ring_joint2",
    # Pinky (2)
    "right_hand_pinky_joint1",
    "right_hand_pinky_joint2",
)
_JOINT_NAMES_WRIST_TRANS = (
    "R_forearm_pos_x_joint",
    "R_forearm_pos_y_joint",
    "R_forearm_pos_z_joint",
)
_JOINT_NAMES_WRIST_ROT = (
    "R_forearm_rot_z_joint",
    "R_forearm_rot_x_joint",
    "R_forearm_rot_y_joint",
)
# All actuated joints (forearm 6-DoF float base + 12 finger joints).
_ALL_JOINT_NAMES = (
    *_JOINT_NAMES_WRIST_TRANS,
    *_JOINT_NAMES_WRIST_ROT,
    *_JOINT_NAMES_FINGER,
)

_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    rot=(1.0, 0.0, 0.0, 0.0),  # (w, x, y, z) identity
    lin_vel=(0.0, 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),
    joint_pos={name: 0.0 for name in _ALL_JOINT_NAMES},
    joint_vel={name: 0.0 for name in _ALL_JOINT_NAMES},
)

# articulation: single actuator group over every actuated joint;
# soft_joint_pos_limit_factor falls back to the build_articulation default 1.0.
_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(XmlActuatorCfg(target_names_expr=_ALL_JOINT_NAMES),),
    soft_joint_pos_limit_factor=1.0,
)

# collisions: single CollisionCfg over the per-link collision geoms.
_COLLISIONS = (
    CollisionCfg(
        geom_names_expr=(
            # Palm (1)
            "collision_hand_right_palm_0",
            # Thumb (4: 3 phalanges + tip)
            "collision_hand_right_thumb_0",
            "collision_hand_right_thumb_1",
            "collision_hand_right_thumb_2",
            "collision_hand_right_thumb_tip",
            # Index (3: 2 phalanges + tip)
            "collision_hand_right_index_0",
            "collision_hand_right_index_1",
            "collision_hand_right_index_tip",
            # Middle (3)
            "collision_hand_right_middle_0",
            "collision_hand_right_middle_1",
            "collision_hand_right_middle_tip",
            # Ring (3)
            "collision_hand_right_ring_0",
            "collision_hand_right_ring_1",
            "collision_hand_right_ring_tip",
            # Pinky (3)
            "collision_hand_right_pinky_0",
            "collision_hand_right_pinky_1",
            "collision_hand_right_pinky_tip",
        ),
        contype=1,
        conaffinity=0,
        condim=3,
        friction=(1.0, 0.005, 0.0001),
        disable_other_geoms=True,
    ),
)

XHAND_ROBOT_CFG = EntityCfg(
    init_state=_INIT_STATE,
    spec_fn=lambda p=_XML_PATH: mujoco.MjSpec.from_file(p),
    articulation=_ARTICULATION,
    collisions=_COLLISIONS,
)


# ─── Robot meta (non-EntityCfg robot-yaml fields) ────────────────────────────
# Consumed by the scene/commands builders via robot_meta.
XHAND_META: dict = {
    "entity_name": "robot",
    "body_names": {
        "fingers": ["thumb", "index", "middle", "ring", "pinky"],
    },
    "site_names": {
        "palm": {
            "right": "right_palm",
        },
        "tip": {
            "right": [
                "track_hand_right_thumb_tip",
                "track_hand_right_index_tip",
                "track_hand_right_middle_tip",
                "track_hand_right_ring_tip",
                "track_hand_right_pinky_tip",
            ],
        },
        "contact": {
            "right": [
                "contact_right_thumb_tip",
                "contact_right_index_tip",
                "contact_right_middle_tip",
                "contact_right_ring_tip",
                "contact_right_pinky_tip",
            ],
        },
    },
    "joint_names": {
        "wrist_trans": list(_JOINT_NAMES_WRIST_TRANS),
        "wrist_rot": list(_JOINT_NAMES_WRIST_ROT),
        "finger": list(_JOINT_NAMES_FINGER),
    },
    "body_mapping": {
        "all": [
            ["hand_thumb_bend_link", "thumb_proximal"],
            ["hand_thumb_rota_link1", "thumb_proximal"],
            ["hand_thumb_rota_link2", "thumb_intermediate"],
            ["hand_index_bend_link", "index_proximal"],
            ["hand_index_rota_link1", "index_proximal"],
            ["hand_index_rota_link2", "index_intermediate"],
            ["hand_mid_link1", "middle_proximal"],
            ["hand_mid_link2", "middle_intermediate"],
            ["hand_ring_link1", "ring_proximal"],
            ["hand_ring_link2", "ring_intermediate"],
            ["hand_pinky_link1", "pinky_proximal"],
            ["hand_pinky_link2", "pinky_intermediate"],
        ],
        "level1": {
            "thumb": ["hand_thumb_bend_link", "thumb_proximal"],
            "index": ["hand_index_bend_link", "index_proximal"],
            "middle": ["hand_mid_link1", "middle_proximal"],
            "ring": ["hand_ring_link1", "ring_proximal"],
            "pinky": ["hand_pinky_link1", "pinky_proximal"],
        },
        "level2": {
            "thumb": ["hand_thumb_rota_link2", "thumb_intermediate"],
            "index": ["hand_index_rota_link2", "index_intermediate"],
            "middle": ["hand_mid_link2", "middle_intermediate"],
            "ring": ["hand_ring_link2", "ring_intermediate"],
            "pinky": ["hand_pinky_link2", "pinky_intermediate"],
        },
    },
}
