"""Object property mixin for MotionTrackingCommand.

Provides reference trajectory, sim state, next-frame lookahead, and SDF
query for object tracking. Separated from the main class for readability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_from_matrix

if TYPE_CHECKING:
    from mjlab.entity import Entity


class ObjectPropertiesMixin:
    """Reference + sim object state properties and SDF query."""

    # --- Reference object trajectory ---

    @property
    def has_objects(self) -> bool:
        return self.cfg.object.entity_names is not None

    @property
    def ref_obj_trans_w(self) -> torch.Tensor:
        """Reference object positions. Shape: (B, n_sides, 3)."""
        parts = []
        for side in self._side_list:
            if side in self.motion_lib.obj_trans:
                pos = self.motion_lib.obj_trans[side][self._motion_flat_ids]
                pos = pos + self._env.scene.env_origins
                parts.append(pos)
        return torch.stack(parts, dim=1)

    @property
    def ref_obj_rotmat_w(self) -> torch.Tensor:
        """Reference object rotation matrices. Shape: (B, n_sides, 3, 3)."""
        parts = []
        for side in self._side_list:
            if side in self.motion_lib.obj_rot:
                parts.append(self.motion_lib.obj_rot[side][self._motion_flat_ids])
        return torch.stack(parts, dim=1)

    @property
    def ref_obj_quat_w(self) -> torch.Tensor:
        """Reference object quaternions. Shape: (B, n_sides, 4)."""
        return quat_from_matrix(self.ref_obj_rotmat_w)

    @property
    def ref_obj_lin_vel_w(self) -> torch.Tensor:
        """Reference object velocities. Shape: (B, n_sides, 3)."""
        parts = []
        for side in self._side_list:
            if side in self.motion_lib.obj_lin_vel:
                parts.append(self.motion_lib.obj_lin_vel[side][self._motion_flat_ids])
        return torch.stack(parts, dim=1)

    @property
    def ref_obj_ang_vel_w(self) -> torch.Tensor:
        """Reference object angular velocities. Shape: (B, n_sides, 3)."""
        parts = []
        for side in self._side_list:
            if side in self.motion_lib.obj_ang_vel:
                parts.append(self.motion_lib.obj_ang_vel[side][self._motion_flat_ids])
        return torch.stack(parts, dim=1)

    # --- Sim object state ---

    def _obj_mass_body_idx(self, obj) -> int:
        """Index of the body that carries mass/geoms within the entity's body_ids."""
        return len(obj.indexing.body_ids) - 1

    @property
    def sim_obj_trans_w(self) -> torch.Tensor:
        """Sim object positions. Shape: (B, n_sides, 3)."""
        parts = []
        for side in self._side_list:
            obj_name = self.cfg.object.entity_names[side]
            obj: Entity = self._env.scene[obj_name]
            idx = self._obj_mass_body_idx(obj)
            parts.append(obj.data.body_link_pos_w[:, idx])
        return torch.stack(parts, dim=1)

    @property
    def sim_obj_quat_w(self) -> torch.Tensor:
        """Sim object quaternions. Shape: (B, n_sides, 4)."""
        parts = []
        for side in self._side_list:
            obj_name = self.cfg.object.entity_names[side]
            obj: Entity = self._env.scene[obj_name]
            idx = self._obj_mass_body_idx(obj)
            parts.append(obj.data.body_link_quat_w[:, idx])
        return torch.stack(parts, dim=1)

    @property
    def sim_obj_lin_vel_w(self) -> torch.Tensor:
        """Sim object velocities. Shape: (B, n_sides, 3)."""
        parts = []
        for side in self._side_list:
            obj_name = self.cfg.object.entity_names[side]
            obj: Entity = self._env.scene[obj_name]
            idx = self._obj_mass_body_idx(obj)
            parts.append(obj.data.body_link_lin_vel_w[:, idx])
        return torch.stack(parts, dim=1)

    @property
    def sim_obj_ang_vel_w(self) -> torch.Tensor:
        """Sim object angular velocities. Shape: (B, n_sides, 3)."""
        parts = []
        for side in self._side_list:
            obj_name = self.cfg.object.entity_names[side]
            obj: Entity = self._env.scene[obj_name]
            idx = self._obj_mass_body_idx(obj)
            parts.append(obj.data.body_link_ang_vel_w[:, idx])
        return torch.stack(parts, dim=1)

    # --- Future trajectory (1-step lookahead) ---

    def _next_motion_flat_ids(self) -> torch.Tensor:
        """Flat index for next frame per env, clamped to trajectory end."""
        cap = self.motion_lib._motion_num_frames[self.motion_ids] - 1
        next_t = torch.minimum(self.motion_steps + 1, cap)
        return self.motion_lib.length_starts[self.motion_ids] + next_t

    @property
    def next_obj_trans_w(self) -> torch.Tensor:
        """Next-frame object positions. Shape: (B, n_sides, 3)."""
        nfi = self._next_motion_flat_ids()
        parts = []
        for side in self._side_list:
            if side in self.motion_lib.obj_trans:
                pos = self.motion_lib.obj_trans[side][nfi] + self._env.scene.env_origins
                parts.append(pos)
        return torch.stack(parts, dim=1)

    @property
    def next_obj_quat_w(self) -> torch.Tensor:
        """Next-frame object quaternions. Shape: (B, n_sides, 4)."""
        nfi = self._next_motion_flat_ids()
        parts = []
        for side in self._side_list:
            if side in self.motion_lib.obj_rot:
                rotmat = self.motion_lib.obj_rot[side][nfi]
                parts.append(quat_from_matrix(rotmat))
        return torch.stack(parts, dim=1)

    @property
    def next_obj_vel_w(self) -> torch.Tensor:
        """Next-frame object velocities. Shape: (B, n_sides, 3)."""
        nfi = self._next_motion_flat_ids()
        parts = []
        for side in self._side_list:
            if side in self.motion_lib.obj_lin_vel:
                parts.append(self.motion_lib.obj_lin_vel[side][nfi])
        return torch.stack(parts, dim=1)

    # --- SDF query ---

    def sdf_query(
        self, world_pts: torch.Tensor, side: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Query the per-side baked SDF + gradient at arbitrary world-frame points.

        Args:
          world_pts: ``(B, K, 3)`` (or ``(B, 3)``) world-frame query points.
          side: which side's SDF to use.

        Returns:
          ``(sdf, grad_world)`` — signed distance and world-frame gradient.
        """
        if side not in self._obj_sdf_grids:
            raise KeyError(
                f"sdf_query: no SDF grid baked for side {side!r}. Set "
                f"`object_mesh_paths[{side!r}]` on the MotionTrackingCommandCfg."
            )
        grid = self._obj_sdf_grids[side]  # (4, N, N, N)
        si = self._side_list.index(side)
        obj_trans = self.sim_obj_trans_w[:, si]  # (B, 3)
        obj_quat = self.sim_obj_quat_w[:, si]  # (B, 4)

        squeeze_K = world_pts.dim() == 2
        if squeeze_K:
            world_pts = world_pts.unsqueeze(1)  # (B, 1, 3)

        B, K = world_pts.shape[0], world_pts.shape[1]
        delta = world_pts - obj_trans[:, None, :]  # (B, K, 3)
        quat_b = obj_quat[:, None, :].expand(B, K, 4)
        from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

        local = quat_apply_inverse(quat_b, delta)  # (B, K, 3) in object frame

        norm = local / self._obj_sdf_extent  # (B, K, 3)
        norm_zyx = norm.flip(-1)  # (B, K, 3) reordered to (z, y, x)
        sample_grid = norm_zyx.reshape(1, B * K, 1, 1, 3)
        grid_5d = grid.unsqueeze(0)  # (1, 4, N, N, N)
        out = torch.nn.functional.grid_sample(
            grid_5d,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # (1, 4, B*K, 1, 1)
        out = out.view(4, B, K).permute(1, 2, 0)  # (B, K, 4)
        sdf = out[..., 0]  # (B, K)
        grad_local = out[..., 1:]  # (B, K, 3) in object frame
        grad_local = grad_local / (
            grad_local.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        )
        grad_world = quat_apply(quat_b, grad_local)  # (B, K, 3)

        if squeeze_K:
            sdf = sdf.squeeze(1)
            grad_world = grad_world.squeeze(1)
        return sdf, grad_world
