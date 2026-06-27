"""Shared MDP-term helpers.

Ported verbatim from DexManip ``src/envs/_common.py`` (DAVIAN-Robotics/DexManip).
Keep close to upstream so term builders that rely on ``auto_inject`` /
``materialize`` stay diff-able against the tracking branch.
"""

from __future__ import annotations

import inspect
from typing import Any

from mjlab.managers.scene_entity_config import SceneEntityCfg
from omegaconf import DictConfig, OmegaConf


def materialize(value: Any) -> Any:
    """Convert config-loaded values into the runtime objects mjlab expects.

    A dict (or OmegaConf ``DictConfig``) carrying a ``"name"`` key becomes a
    :class:`SceneEntityCfg` (typical shape: ``{name: hand, joint_names: [".*"]}``).
    Everything else passes through unchanged.
    """
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict) and "name" in value:
        return SceneEntityCfg(**value)
    return value


def auto_inject(
    fn: Any,
    params: dict[str, Any],
    command_name: str | None = None,
    entity_name: str | None = None,
) -> None:
    """Auto-inject ``command_name`` and ``asset_cfg`` when ``fn`` accepts them.

    - ``command_name``: injected as-is when the signature accepts it.
    - ``asset_cfg``: injected as ``SceneEntityCfg(entity_name, joint_names=(".*",))``
      when accepted and not already present (an explicit ``asset_cfg`` wins).
    """
    sig = inspect.signature(fn)
    if command_name is not None and "command_name" in sig.parameters:
        params.setdefault("command_name", command_name)
    if entity_name is not None and "asset_cfg" in sig.parameters:
        if "asset_cfg" not in params:
            params["asset_cfg"] = SceneEntityCfg(entity_name, joint_names=(".*",))
