"""Hand (wrist/tip/joint) property mixin for MotionTrackingCommand.

Provides MANO reference and robot state properties for wrist, fingertip,
and body-level joint tracking. Separated from the main class for readability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_from_matrix

if TYPE_CHECKING:
    pass


class HandPropertiesMixin:
    """MANO reference + robot state properties for hand tracking."""

    # --- MANO wrist ---

    @property
    def mano_wrist_trans_w(self) -> torch.Tensor:
        """MANO wrist positions. Shape: (B, n_sides, 3)."""
        parts = []
        for side in self._side_list:
            pos = self.motion_lib.mano_wrist_trans[side][
                self._motion_flat_ids
            ]  # (B, 3)
            pos = pos + self._env.scene.env_origins
            parts.append(pos)
        return torch.stack(parts, dim=1)

    @property
    def mano_wrist_rot_w(self) -> torch.Tensor:
        """MANO wrist rotation matrices. Shape: (B, n_sides, 3, 3)."""
        parts = []
        for side in self._side_list:
            parts.append(self.motion_lib.mano_wrist_rot[side][self._motion_flat_ids])
        return torch.stack(parts, dim=1)

    @property
    def mano_wrist_quat_w(self) -> torch.Tensor:
        """MANO wrist quaternions (from rotmat). Shape: (B, n_sides, 4)."""
        return quat_from_matrix(self.mano_wrist_rot_w)

    @property
    def mano_wrist_lin_vel_w(self) -> torch.Tensor:
        """MANO wrist linear velocities. Shape: (B, n_sides, 3)."""
        parts = []
        for side in self._side_list:
            parts.append(
                self.motion_lib.mano_wrist_lin_vel[side][self._motion_flat_ids]
            )
        return torch.stack(parts, dim=1)

    @property
    def mano_wrist_ang_vel_w(self) -> torch.Tensor:
        """MANO wrist angular velocities. Shape: (B, n_sides, 3)."""
        parts = []
        for side in self._side_list:
            parts.append(
                self.motion_lib.mano_wrist_ang_vel[side][self._motion_flat_ids]
            )
        return torch.stack(parts, dim=1)

    # --- MANO fingertips ---

    @property
    def mano_tip_trans_w(self) -> torch.Tensor:
        """MANO fingertip positions (5 per side). Shape: (B, n_sides, 5, 3)."""
        parts = []
        for side in self._side_list:
            all_joints = self.motion_lib.mano_joint_pos[side][
                self._motion_flat_ids
            ]  # (B, 20, 3)
            tips = all_joints[:, self.motion_lib.tip_ids[side]]  # (B, 5, 3)
            tips = tips + self._env.scene.env_origins[:, None, :]
            parts.append(tips)
        return torch.stack(parts, dim=1)

    @property
    def mano_tip_lin_vel_w(self) -> torch.Tensor:
        """MANO fingertip velocities (5 per side). Shape: (B, n_sides, 5, 3)."""
        parts = []
        for side in self._side_list:
            all_vel = self.motion_lib.mano_joint_vel[side][
                self._motion_flat_ids
            ]  # (B, 20, 3)
            tips = all_vel[:, self.motion_lib.tip_ids[side]]  # (B, 5, 3)
            parts.append(tips)
        return torch.stack(parts, dim=1)

    # --- MANO body-level joints ---

    def mano_all_joints_trans_w(self, side: str) -> torch.Tensor:
        """MANO positions for all 12 non-tip joints. Shape: (B, 12, 3)."""
        mano_ids = self._all_mano_ids[side]
        all_joints = self.motion_lib.mano_joint_pos[side][
            self._motion_flat_ids
        ]  # (B, 20, 3)
        pts = all_joints[:, mano_ids]  # (B, 12, 3)
        return pts + self._env.scene.env_origins[:, None, :]

    def mano_all_joints_lin_vel_w(self, side: str) -> torch.Tensor:
        """MANO velocities for all 12 non-tip joints. Shape: (B, 12, 3)."""
        mano_ids = self._all_mano_ids[side]
        all_vel = self.motion_lib.mano_joint_vel[side][
            self._motion_flat_ids
        ]  # (B, 20, 3)
        return all_vel[:, mano_ids]  # (B, 12, 3)

    def mano_level_trans_w(self, side: str, level: int) -> torch.Tensor:
        """MANO joint positions for level 1 or 2. Shape: (B, 5, 3)."""
        mano_ids = (
            self._level1_mano_ids[side] if level == 1 else self._level2_mano_ids[side]
        )
        all_joints = self.motion_lib.mano_joint_pos[side][
            self._motion_flat_ids
        ]  # (B, 20, 3)
        pts = all_joints[:, mano_ids]  # (B, 5, 3)
        return pts + self._env.scene.env_origins[:, None, :]

    # --- MANO contact / distance ---

    @property
    def ref_contact_trans_w(self) -> torch.Tensor:
        """Reference contact points on object, in world frame. Shape: (B, n_sides, 5, 3)."""
        parts = []
        sim_obj_quat = self.sim_obj_quat_w  # (B, n_sides, 4)
        sim_obj_trans = self.sim_obj_trans_w  # (B, n_sides, 3)
        for side in self._side_list:
            si = self._side_list.index(side)
            local_pts = self.motion_lib.contact_pos_full[side][
                self._motion_flat_ids
            ]  # (B, 5, 3)
            obj_trans = sim_obj_trans[:, si]  # (B, 3)
            obj_rot = matrix_from_quat(sim_obj_quat[:, si])  # (B, 3, 3)
            world_pts = obj_trans[:, None, :] + torch.einsum(
                "bij,bkj->bki", obj_rot, local_pts
            )
            parts.append(world_pts)
        return torch.stack(parts, dim=1)

    @property
    def ref_contact_flags(self) -> torch.Tensor:
        """Binary contact expected per finger per side. Shape: (B, n_sides, 5)."""
        parts = []
        for side in self._side_list:
            parts.append(self.motion_lib.contact_flags[side][self._motion_flat_ids])
        return torch.stack(parts, dim=1)

    @property
    def mano_tips_distance(self) -> torch.Tensor:
        """Precomputed MANO tip-to-object-surface distance. Shape: (B, n_sides, 5)."""
        parts = []
        for side in self._side_list:
            parts.append(self.motion_lib.tips_distance[side][self._motion_flat_ids])
        return torch.stack(parts, dim=1)

    # --- Robot wrist ---

    @property
    def robot_wrist_trans_w(self) -> torch.Tensor:
        """Robot palm site positions. Shape: (B, n_sides, 3)."""
        parts = []
        for side in self._side_list:
            idx = self._palm_site_ids[side]
            parts.append(self.robot.data.site_pos_w[:, idx])
        return torch.stack(parts, dim=1)

    @property
    def robot_wrist_quat_w(self) -> torch.Tensor:
        """Robot palm site quaternions. Shape: (B, n_sides, 4)."""
        parts = []
        for side in self._side_list:
            idx = self._palm_site_ids[side]
            parts.append(self.robot.data.site_quat_w[:, idx])
        return torch.stack(parts, dim=1)

    @property
    def robot_wrist_rot_w(self) -> torch.Tensor:
        """Robot wrist rotation matrices (from quat). Shape: (B, n_sides, 3, 3)."""
        return matrix_from_quat(self.robot_wrist_quat_w)

    @property
    def robot_wrist_lin_vel_w(self) -> torch.Tensor:
        """Robot palm site linear velocities. Shape: (B, n_sides, 3)."""
        parts = []
        for side in self._side_list:
            idx = self._palm_site_ids[side]
            parts.append(self.robot.data.site_lin_vel_w[:, idx])
        return torch.stack(parts, dim=1)

    @property
    def robot_wrist_ang_vel_w(self) -> torch.Tensor:
        """Robot palm site angular velocities. Shape: (B, n_sides, 3)."""
        parts = []
        for side in self._side_list:
            idx = self._palm_site_ids[side]
            parts.append(self.robot.data.site_ang_vel_w[:, idx])
        return torch.stack(parts, dim=1)

    # --- Robot fingertips / contact ---

    @property
    def robot_tip_trans_w(self) -> torch.Tensor:
        """Robot fingertip site positions (5 per side). Shape: (B, n_sides, 5, 3)."""
        parts = []
        for side in self._side_list:
            ids = self._tip_site_ids[side]
            parts.append(self.robot.data.site_pos_w[:, ids])
        return torch.stack(parts, dim=1)

    @property
    def robot_contact_trans_w(self) -> torch.Tensor:
        """Robot contact sensor site positions (5 per side). Shape: (B, n_sides, 5, 3)."""
        parts = []
        for side in self._side_list:
            ids = self._contact_site_ids[side]
            parts.append(self.robot.data.site_pos_w[:, ids])
        return torch.stack(parts, dim=1)

    # --- Robot body-level joints ---

    def robot_all_joints_trans_w(self, side: str) -> torch.Tensor:
        """Robot positions for all 12 non-tip bodies. Shape: (B, 12, 3)."""
        body_ids = self._all_body_ids[side]
        return self.robot.data.body_link_pos_w[:, body_ids]  # (B, 12, 3)

    def robot_all_joints_lin_vel_w(self, side: str) -> torch.Tensor:
        """Robot velocities for all 12 non-tip bodies. Shape: (B, 12, 3)."""
        body_ids = self._all_body_ids[side]
        return self.robot.data.body_link_lin_vel_w[:, body_ids]  # (B, 12, 3)

    def robot_level_trans_w(self, side: str, level: int) -> torch.Tensor:
        """Robot body positions for level 1 or 2. Shape: (B, 5, 3)."""
        body_ids = (
            self._level1_body_ids[side] if level == 1 else self._level2_body_ids[side]
        )
        return self.robot.data.body_link_pos_w[:, body_ids]  # (B, 5, 3)
