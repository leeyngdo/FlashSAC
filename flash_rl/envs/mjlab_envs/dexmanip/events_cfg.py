"""Event wiring for the dexmanip motion-tracking task.

Ports DexManip ``src/envs/events/__init__.py:build_events`` together with the
default preset ``config/envs/events/no_domain_rand.yaml`` (no domain
randomization: only the mjlab built-in scene reset on episode reset).

The event *functions* live in ``..mdp.events`` (re-exported as ``mdp.<name>``,
ported from DexManip). This module owns the declarative *selection* — which
terms, their mode, params, and optional interval — captured verbatim from the
``no_domain_rand`` yaml as a plain Python table. No YAML, no Hydra, no
``build_events`` indirection.

To match upstream ``build_events`` semantics, each entry mirrors a yaml term
block: ``mode`` (required), optional ``params`` (passed through ``materialize``
exactly as the builder does), and optional ``interval_range_s`` (tupled when
present). ``mode="interval"`` terms additionally carry ``interval_range_s``.
"""

from __future__ import annotations

from typing import Any

# DexManip ``no_domain_rand`` events preset (term_name -> term spec), captured
# verbatim from config/envs/events/no_domain_rand.yaml.
#
# Each spec mirrors a yaml term block:
#   "mode"             : EventMode str (required) — e.g. "reset", "interval".
#   "params"           : raw params dict (optional; default {}), run through
#                        ``materialize`` like the builder.
#   "interval_range_s" : (lo, hi) seconds (optional; required for interval mode).
EVENTS: dict[str, dict[str, Any]] = {
    # mjlab default: reset all entities to their init_state on episode reset.
    "reset_scene_to_default": {
        "mode": "reset",
    },
}


def make_events() -> dict[str, Any]:
    """Build ``dict[str, EventTermCfg]`` from the owned events table.

    Mirrors DexManip ``build_events`` but sources the term specs from Python,
    not YAML: looks up each term function on ``mdp.events`` by name, materializes
    its params, and forwards ``mode`` / ``interval_range_s``.
    """
    from mjlab.managers.event_manager import EventTermCfg

    from .. import mdp
    from ..mdp import materialize

    out: dict[str, Any] = {}
    for name, term_cfg in EVENTS.items():
        fn = getattr(mdp, name)
        raw_params = dict(term_cfg.get("params", {}) or {})
        params: dict[str, Any] = {k: materialize(v) for k, v in raw_params.items()}

        kwargs: dict[str, Any] = dict(func=fn, params=params, mode=term_cfg["mode"])
        if term_cfg.get("interval_range_s") is not None:
            kwargs["interval_range_s"] = tuple(term_cfg["interval_range_s"])
        out[name] = EventTermCfg(**kwargs)
    return out
