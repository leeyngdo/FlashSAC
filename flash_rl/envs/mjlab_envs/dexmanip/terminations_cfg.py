"""Termination wiring for the dexmanip motion-tracking task.

REFERENCE IMPLEMENTATION of the pattern (KraftonLab / isaaclab_envs style):
the termination *functions* live in ``..mdp.terms`` (ported from DexManip's
``src/envs/terminations/``), and this module owns the *selection + params*
declaratively. No YAML, no hydra ``get_class``, no ``build_terminations``
indirection.

Ports DexManip ``src/envs/terminations/__init__.py:build_terminations`` and
transcribes ``config/envs/termination/motion_tracking.yaml`` (the
``MotionTrackingTerminations`` preset) verbatim: each entry below mirrors a yaml
term block — ``enabled`` terms only, their ``params``, and ``time_out`` flag.
``command_name`` / ``asset_cfg`` are auto-injected exactly as the builder does.

DEPENDS ON: ``mdp/terms`` being ported (the ``MotionTrackingTerminations`` term
staticmethods, which inherit ``BaseTerminations``) and ``mdp/cmds`` (the
``motion`` command they read). Until then ``make_terminations`` raises at call
time (env-construct time), not import time.
"""

from __future__ import annotations

from typing import Any

# DexManip motion_tracking termination preset, transcribed verbatim from
# config/envs/termination/motion_tracking.yaml. Only ``enabled: true`` terms
# appear (disabled terms are dropped, matching build_terminations). Each value
# is (params, time_out); ``command_name``/``asset_cfg`` are auto-injected.
TERMINATIONS: dict[str, dict[str, Any]] = {
    # ── episode time-out (truncation) ──
    "time_out": {"params": {}, "time_out": True},
    # ── tracking divergence ──
    "obj_trans_diverged": {
        "params": {"threshold": 0.06, "grace_steps": 15},  # metres
        "time_out": False,
    },
    "obj_rot_diverged": {
        "params": {"threshold_deg": 90.0, "grace_steps": 15},  # degrees
        "time_out": False,
    },
    "fingertip_diverged": {
        "params": {"threshold": 0.3, "grace_steps": 15},  # metres
        "time_out": False,
    },
    # ── velocity / physics sanity ──
    "velocity_diverged": {
        "params": {"max_lin_vel": 100.0, "max_ang_vel": 200.0},  # m/s, rad/s
        "time_out": False,
    },
    "joint_vel_mean_sanity": {
        "params": {"max_joint_vel_mean": 200.0},  # rad/s
        "time_out": False,
    },
    "joint_vel_sanity": {
        "params": {"max_joint_vel": 200.0},  # rad/s
        "time_out": False,
    },
    "obj_lin_vel_sanity": {
        "params": {"max_obj_lin_vel": 100.0},  # m/s
        "time_out": False,
    },
    "obj_ang_vel_sanity": {
        "params": {"max_obj_ang_vel": 200.0},  # rad/s
        "time_out": False,
    },
    # ── NaN safety net ──
    "nan_guard": {"params": {}, "time_out": False},
}


def make_terminations(
    *,
    command_name: str = "motion",
    entity_name: str = "robot",
) -> dict[str, Any]:
    """Build ``dict[str, TerminationTermCfg]`` from the owned term table.

    Mirrors DexManip ``build_terminations`` but sources the preset from Python,
    not YAML: each term name resolves to a ``MotionTrackingTerminations``
    staticmethod (``getattr`` by name), params are ``materialize``d, then
    ``command_name``/``asset_cfg`` are auto-injected when the signature accepts
    them.
    """
    from mjlab.managers.termination_manager import TerminationTermCfg

    from ..mdp import auto_inject, materialize
    from ..mdp.terms import MotionTrackingTerminations

    out: dict[str, Any] = {}
    for name, term_cfg in TERMINATIONS.items():
        fn = getattr(MotionTrackingTerminations, name)
        raw_params = dict(term_cfg.get("params", {}) or {})
        params: dict[str, Any] = {k: materialize(v) for k, v in raw_params.items()}
        auto_inject(fn, params, command_name=command_name, entity_name=entity_name)
        out[name] = TerminationTermCfg(
            func=fn,
            params=params,
            time_out=bool(term_cfg.get("time_out", False)),
        )
    return out
