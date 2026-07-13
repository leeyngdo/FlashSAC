"""Config override helpers for the FlashSAC SimToolReal task.

The SimToolReal env cfg uses flat sectioned sub-configs, so the hydra config blocks
map onto plain fields: ``reward`` → ``env_cfg.reward``, ``observation`` → ``env_cfg.obs``,
``termination`` → ``env_cfg.termination``. ``robot``/``motion`` are not supported;
use dot-path ``cfg_overrides`` for anything else (applied last).

This module is intentionally import-light: it works on duck-typed config objects
and does not import IsaacLab.
"""

from __future__ import annotations

from typing import Any

from ..tracking.overrides import apply_cfg_overrides


def _apply_flat_section_overrides(env_cfg: Any, section_attr: str, block: dict[str, Any] | None) -> None:
    """Set fields of a flat sub-configclass, validating that each field exists."""
    if not block:
        return
    section = getattr(env_cfg, section_attr)
    for key, value in block.items():
        if value is None:
            continue
        if not hasattr(section, key):
            raise AttributeError(f"Unknown SimToolReal cfg field '{section_attr}.{key}'.")
        if isinstance(value, list) and isinstance(getattr(section, key), tuple):
            value = tuple(value)
        setattr(section, key, value)


def apply_simtoolreal_overrides(
    env_cfg: Any,
    reward: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
    termination: dict[str, Any] | None = None,
    robot: dict[str, Any] | None = None,
    motion: dict[str, Any] | None = None,
    cfg_overrides: dict[str, Any] | None = None,
) -> Any:
    """Apply friendly flat-section config blocks, then dot-path overrides last.

    ``robot`` and ``motion`` are accepted for signature parity with ``make_isaaclab_env``
    but are not applicable to the SimToolReal task.
    """
    if robot and any(value is not None for value in robot.values()):
        raise ValueError("The SimToolReal task does not support the 'robot' config block; use cfg_overrides.")
    if motion:
        raise ValueError("The SimToolReal task does not support the 'motion' config block.")
    _apply_flat_section_overrides(env_cfg, "reward", reward)
    _apply_flat_section_overrides(env_cfg, "obs", observation)
    _apply_flat_section_overrides(env_cfg, "termination", termination)
    return apply_cfg_overrides(env_cfg, cfg_overrides)
