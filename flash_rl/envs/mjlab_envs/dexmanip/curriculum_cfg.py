"""Curriculum wiring for the dexmanip motion-tracking task.

Declarative port of DexManip ``config/envs/curriculum/base.yaml`` +
``src/envs/curriculum/__init__.py:build_curriculum``. The curriculum modifies
gravity and the object-pin xfrc schedule over training; faithful to upstream.

NOTE: the council recommended holding the curriculum constant for first SAC
bring-up (its schedules are tuned for 4096-env PPO throughput, not SAC's sample
budget). To do that, override per-term or pass an empty dict — but the default
here matches upstream so native == DexManip behaviorally.
"""

from __future__ import annotations

from typing import Any

# term_name -> params (from config/envs/curriculum/base.yaml).
CURRICULUM_PARAMS: dict[str, dict[str, Any]] = {
    "gravity_curriculum": {"schedule_steps": 0, "full_g": 9.81},
    "xfrc_curriculum": {
        "omega_n_start": 20.0,
        "omega_n_end": 0.0,
        "schedule_steps": 6400,
        "delay_steps": 0,
        "zeta": 1.0,
    },
}


def make_curriculum(command_name: str = "motion", entity_name: str = "robot") -> dict[str, Any]:
    """Build ``dict[str, CurriculumTermCfg]`` — mirrors DexManip ``build_curriculum``."""
    from mjlab.managers.curriculum_manager import CurriculumTermCfg

    from ..mdp import auto_inject
    from ..mdp.curriculum import base as _t

    out: dict[str, Any] = {}
    for name, params in CURRICULUM_PARAMS.items():
        fn = getattr(_t, name)
        p: dict[str, Any] = dict(params)
        auto_inject(fn, p, command_name=command_name, entity_name=entity_name)
        out[name] = CurriculumTermCfg(func=fn, params=p)
    return out
