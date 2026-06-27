"""Action wiring for the dexmanip motion-tracking task.

Ports DexManip's ``config/envs/actions/maniptrans.yaml`` (the ManipTrans
trajectory-residual action term) together with the ``build_actions`` builder in
``src/envs/actions/__init__.py``. The yaml values are inlined as Python literals
(no YAML / Hydra / OmegaConf at runtime), and the action term *class*
(``ManipTransActionCfg``) is the verbatim-ported one re-exported from ``..mdp``.

DexManip convention transcribed from ``build_actions``: the yaml carries the
robot ``joint_names`` (``{wrist_trans, wrist_rot, finger}``) and the builder
derives the ``actuator_names`` group dict the dataclass actually expects:

    actuator_names = {
        "wrist":  wrist_trans + wrist_rot,
        "finger": finger,
    }

``joint_names`` is sourced from ``robot_meta['joint_names']`` (i.e. straight from
``config/envs/robot/xhand_ghost_right.yaml``), so this module stays robot-agnostic.

DEPENDS ON: ``mdp.actions`` being ported (``ManipTransActionCfg`` /
``ManipTransAction``). Until the robot metadata is supplied, ``make_actions``
does nothing — it is only invoked at env-construct time.
"""

from __future__ import annotations

from typing import Any

# --- DexManip config/envs/actions/maniptrans.yaml (inlined verbatim) ---------
# Term name + scalar tuning surface.
_ACTION_NAME = "maniptrans"

_ACTION_SCALE = 1.0
_ACTION_OFFSET = 0.0

# yaml ``clip: {".*": [-1.0, 1.0]}`` -> the dataclass expects ``dict[str, tuple]``.
_CLIP: dict[str, tuple] = {".*": (-1.0, 1.0)}

# yaml ``residual_scale: {wrist: 0.05, finger: 1.0}``.
_RESIDUAL_SCALE: dict[str, float] = {"wrist": 0.05, "finger": 1.0}


def make_actions(*, robot_meta: dict[str, Any]) -> dict[str, Any]:
    """Build ``dict[str, ActionTermCfg]`` for the ManipTrans action term.

    Mirrors DexManip ``build_actions`` but sources values from Python literals +
    ``robot_meta`` instead of the parsed Hydra cfg block.

    Args:
        robot_meta: the ``XHAND_META`` dict. Uses ``entity_name`` and
            ``joint_names`` (``{wrist_trans, wrist_rot, finger}``).
    """
    from ..mdp import ManipTransActionCfg

    jn = robot_meta["joint_names"]
    # build_actions: derive grouped ``actuator_names`` from ``joint_names``.
    actuator_names: dict[str, tuple[str, ...]] = {
        "wrist": tuple(jn["wrist_trans"]) + tuple(jn["wrist_rot"]),
        "finger": tuple(jn["finger"]),
    }

    cfg = ManipTransActionCfg(
        entity_name=robot_meta["entity_name"],
        command_name="motion",
        actuator_names=actuator_names,
        residual_scale=dict(_RESIDUAL_SCALE),
        action_scale=_ACTION_SCALE,
        action_offset=_ACTION_OFFSET,
        clip=dict(_CLIP),
    )
    return {_ACTION_NAME: cfg}
