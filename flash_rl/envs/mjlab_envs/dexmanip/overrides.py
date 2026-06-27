"""Config-override seam for the dexmanip task.

The "configure terms yourself" surface. ``configs/env/dexmanip.yaml`` carries
``reward: {...}``, ``observation: {...}``, ``event: {...}``, ``cfg_overrides:
{...}`` blocks; this module applies them onto the already-assembled
``ManagerBasedRlEnvCfg`` in place, BEFORE ``ManagerBasedRlEnv`` construction
(mjlab managers consume the cfg at __init__).

Intentionally import-light and duck-typed (works on mjlab term-cfg dicts /
dataclasses without importing mjlab), mirroring
``isaaclab_envs/tracking/overrides.py``.

mjlab manager cfg shapes this operates on (see DexManip ``make_env_cfg``):
  * ``cfg.rewards``       : ``dict[str, RewardTermCfg]``        (.weight, .params)
  * ``cfg.terminations``  : ``dict[str, TerminationTermCfg]``   (.params)
  * ``cfg.events``        : ``dict[str, EventTermCfg]``         (.params)
  * ``cfg.observations``  : ``dict[str, ObservationGroupCfg]``  (.terms[name].{scale,params})
"""

from __future__ import annotations

from typing import Any


# ─── generic helpers (ported from isaaclab_envs/tracking/overrides.py) ──────
def omegaconf_to_plain(value: Any) -> Any:
    """Convert OmegaConf containers to plain Python containers, recursively."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(value, (DictConfig, ListConfig)):
            value = OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    if isinstance(value, dict):
        return {k: omegaconf_to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [omegaconf_to_plain(v) for v in value]
    return value


def _set_by_path(obj: Any, dotted: str, value: Any) -> None:
    """Set a nested dict member or object attribute addressed by a dotted path."""
    parts = dotted.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = cur[part] if isinstance(cur, dict) else getattr(cur, part)
    last = parts[-1]
    if isinstance(cur, dict):
        cur[last] = value
    else:
        setattr(cur, last, value)


def _iter_cfg_override_paths(overrides: dict[str, Any]) -> list[tuple[str, Any]]:
    """Flatten nested override dicts into ``("a.b.c", value)`` pairs.

    Hydra's CLI grammar makes literal dotted dict keys awkward. Accepting nested
    dicts keeps sweep scripts readable while preserving the flat dot-path API.
    Empty dicts are treated as leaf values so ``{curriculum: {}}`` still clears
    the curriculum manager config.
    """
    out: list[tuple[str, Any]] = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict) and value:
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
        else:
            out.append((prefix, value))

    for key, value in overrides.items():
        visit(str(key), value)
    return out


def _apply_term_dict_overrides(term_dict: dict[str, Any] | None, overrides: dict[str, Any] | None) -> None:
    """Apply ``{term: {weight: .., params: {..}, enabled: bool}}`` onto a mjlab term dict."""
    if not term_dict or not overrides:
        return
    for name, ov in overrides.items():
        ov = dict(ov or {})
        if ov.get("enabled") is False:
            term_dict.pop(name, None)
            continue
        term = term_dict.get(name)
        if term is None:
            # Unknown term name: surface loudly rather than silently no-op.
            raise KeyError(f"override targets unknown term '{name}' (have: {sorted(term_dict)})")
        if "weight" in ov:
            term.weight = float(ov["weight"])
        params = ov.get("params")
        if params:
            term.params.update(params)


def _apply_observation_overrides(obs_groups: dict[str, Any] | None, overrides: dict[str, Any] | None) -> None:
    """Apply nested ``{group: {term: {scale: .., params: {..}, enabled: bool}}}`` overrides."""
    if not obs_groups or not overrides:
        return
    for group_name, term_ovs in overrides.items():
        group = obs_groups.get(group_name)
        if group is None:
            raise KeyError(f"override targets unknown obs group '{group_name}' (have: {sorted(obs_groups)})")
        for term_name, ov in dict(term_ovs or {}).items():
            ov = dict(ov or {})
            if ov.get("enabled") is False:
                group.terms.pop(term_name, None)
                continue
            term = group.terms.get(term_name)
            if term is None:
                raise KeyError(f"override targets unknown obs term '{group_name}.{term_name}'")
            if "scale" in ov:
                term.scale = ov["scale"]
            if ov.get("params"):
                term.params.update(ov["params"])


def apply_dexmanip_overrides(
    env_cfg: Any,
    *,
    reward: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
    event: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
    termination: dict[str, Any] | None = None,
    robot: dict[str, Any] | None = None,
    cfg_overrides: dict[str, Any] | None = None,
) -> Any:
    """Apply all FlashSAC-side term overrides onto ``env_cfg`` in place.

    ``cfg_overrides`` (flat dotted paths) is applied LAST so it can reach
    anything the structured blocks don't cover.
    """
    _apply_term_dict_overrides(getattr(env_cfg, "rewards", None), omegaconf_to_plain(reward))
    _apply_observation_overrides(getattr(env_cfg, "observations", None), omegaconf_to_plain(observation))
    _apply_term_dict_overrides(getattr(env_cfg, "events", None), omegaconf_to_plain(event))
    _apply_term_dict_overrides(getattr(env_cfg, "terminations", None), omegaconf_to_plain(termination))

    # TODO(port): action/robot overrides need the actuator/action-scale shapes
    # (see isaaclab_envs/tracking/overrides._apply_robot_overrides). Wire once
    # robots/ + actions are ported.
    if action:
        raise NotImplementedError("dexmanip action overrides not wired yet (see overrides.py TODO)")
    if robot:
        raise NotImplementedError("dexmanip robot overrides not wired yet (see overrides.py TODO)")

    for path, value in _iter_cfg_override_paths(omegaconf_to_plain(cfg_overrides) or {}):
        _set_by_path(env_cfg, path, value)
    return env_cfg
