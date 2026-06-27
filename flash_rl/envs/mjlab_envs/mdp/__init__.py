"""Modular MDP-term library for the mjlab dexmanip envs (KraftonLab convention).

Each subpackage holds the term *functions* (the physics/math), ported from the
matching DexManip ``src/envs/<x>/`` module and kept structurally close to
upstream so they stay diff-able against the tracking branch:

    cmds/    <- DexManip src/envs/commands/motion_tracking/   (motion_lib, sdf, props, command)
    obs/     <- DexManip src/envs/observations/
    rews/    <- DexManip src/envs/rewards/
    terms/   <- DexManip src/envs/terminations/
    events/  <- DexManip src/envs/events/
    actions/ <- DexManip src/envs/actions/

The declarative wiring (which terms, weights, params) lives in the task package
``flash_rl.envs.mjlab_envs.dexmanip`` — NOT here. This package is the term pool;
``dexmanip/*_cfg.py`` selects and weights from it.

Re-exports are added here as each term module is ported, so callers can write
``from ..mdp import motion_global_anchor_position_error_exp`` etc., mirroring
``isaaclab_envs.mdp``. Until then this stays an empty namespace.
"""

from __future__ import annotations

from ._common import auto_inject, materialize  # noqa: F401

# Term re-exports (mirror isaaclab_envs.mdp) — convenient ``mdp.<Name>`` access
# from the dexmanip/*_cfg.py wiring. All ported from DexManip src/envs/*.
from .actions import ManipTransAction, ManipTransActionCfg  # noqa: F401
from .cmds import MotionTrackingCommand, MotionTrackingCommandCfg  # noqa: F401
from .events import reset_scene_to_default  # noqa: F401
from .obs import BaseObs, MotionTrackingObs  # noqa: F401
from .rews import BaseRewards, MotionTrackingRewards  # noqa: F401
from .terms import BaseTerminations, MotionTrackingTerminations  # noqa: F401
