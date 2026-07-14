"""Dexsuite termination terms.

This module imports ``isaaclab`` and is therefore NOT part of the import-light path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["out_of_bound"]


class out_of_bound(ManagerTermBase):
    """Termination condition for when the object falls out of bound.

    Unlike the stock implementation, honors runtime updates to ``in_bound_range`` (needed for
    the ``oob_adr`` curriculum term): the world-space bounds are cached and rebuilt per axis
    only when the corresponding range entry changes, keeping the hot path free of
    host-to-device transfers.
    """

    def __init__(self, cfg: Any, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self._object: RigidObject = env.scene[asset_cfg.name]
        # Pre-apply env_origins so we can compare directly against world-space positions.
        self._origins = env.scene.env_origins  # (N, 3)
        self._lower = self._origins.clone()  # (N, 3)
        self._upper = self._origins.clone()  # (N, 3)
        self._cached_axis: list[tuple[float, ...] | None] = [None, None, None]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        in_bound_range: dict[str, tuple[float, float]] = {},
    ) -> torch.Tensor:
        # rebuild only the axes whose bounds changed (curriculum typically only moves one)
        for i, key in enumerate(("x", "y", "z")):
            bounds = tuple(in_bound_range.get(key, (0.0, 0.0)))
            if bounds != self._cached_axis[i]:
                lo, hi = bounds
                self._lower[:, i] = self._origins[:, i] + lo
                self._upper[:, i] = self._origins[:, i] + hi
                self._cached_axis[i] = bounds

        pos_w = self._object.data.root_pos_w
        return ((pos_w < self._lower) | (pos_w > self._upper)).any(dim=1)
