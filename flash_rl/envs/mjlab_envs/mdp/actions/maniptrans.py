from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

from ..cmds.motion_tracking import MotionTrackingCommand

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ManipTransActionCfg(ActionTermCfg):
    """Configuration for ManipTrans trajectory-residual action term.

    Actuator-name regex groups and residual scales are passed as group→value
    dicts; the action term splits them into wrist/finger internally. Required
    keys: ``"wrist"`` and ``"finger"``.
    """

    entity_name: str
    command_name: str

    actuator_names: dict[str, tuple[str, ...]]
    
    residual_scale: dict[str, float]
    
    action_scale: float = 1.0
    action_offset: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> ManipTransAction:
        return ManipTransAction(self, env)


class ManipTransAction(ActionTerm):
    """Trajectory-residual action for ManipTrans. action_dim = n_dofs."""

    cfg: ManipTransActionCfg
    _entity: Entity

    def __init__(self, cfg: ManipTransActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self._command_name = cfg.command_name

        # Split grouped actuator_names into wrist/finger internally.
        wrist_ids, wrist_names = self._entity.find_joints_by_actuator_names(
            cfg.actuator_names["wrist"]
        )
        finger_ids, finger_names = self._entity.find_joints_by_actuator_names(
            cfg.actuator_names["finger"]
        )
        self._wrist_ids = torch.tensor(wrist_ids, device=self.device, dtype=torch.long)
        self._finger_ids = torch.tensor(
            finger_ids, device=self.device, dtype=torch.long
        )
        self._all_joint_ids = torch.cat([self._wrist_ids, self._finger_ids])

        limits = self._entity.data.soft_joint_pos_limits[0]
        self._finger_lower = limits[self._finger_ids, 0]
        self._finger_upper = limits[self._finger_ids, 1]
        self._finger_range = self._finger_upper - self._finger_lower

        # Cache residual scales — static across substeps, dict lookup avoided in apply_actions.
        self._wrist_scale = float(cfg.residual_scale["wrist"])
        self._finger_scale = float(cfg.residual_scale["finger"])

        action_dim = len(self._all_joint_ids)
        self._raw_actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._processed_actions = torch.zeros(
            self.num_envs, action_dim, device=self.device
        )

        if cfg.clip is not None:
            self._clip = torch.tensor(
                [[-float("inf"), float("inf")]], device=self.device
            ).repeat(self.num_envs, action_dim, 1)
            idx_list, _, val_list = resolve_matching_names_values(
                dict(cfg.clip), wrist_names + finger_names
            )
            self._clip[:, idx_list] = torch.tensor(
                val_list, device=self.device, dtype=torch.float32
            )
        else:
            self._clip = None

    @property
    def action_dim(self) -> int:
        return len(self._all_joint_ids)

    @property
    def n_dofs(self) -> int:
        """Number of applied action dimensions (hand DoFs). Always n_dofs — Stage 2
        residual composition happens inside the ResidualActor, not here."""
        return len(self._all_joint_ids)

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Apply once-per-policy-step normalization (scale → offset → clip) and
        stash for apply_actions. Mirrors ``mjlab.JointAction.process_actions``."""
        self._raw_actions[:] = actions
        self._processed_actions = (
            self._raw_actions * self.cfg.action_scale + self.cfg.action_offset
        )
        if self._clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

    def apply_actions(self) -> None:
        command = cast(
            MotionTrackingCommand,
            self._env.command_manager.get_term(self._command_name),
        )
        ref_joint_pos = command.ref_joint_pos  # (B, n_dofs)

        action = self._processed_actions  # already clipped in process_actions
        wrist_action = action[:, : len(self._wrist_ids)]
        finger_action = action[:, len(self._wrist_ids) :]

        # Wrist: ref + action * scale (action is already in [-1, 1] thanks to clip).
        ref_wrist = ref_joint_pos[:, self._wrist_ids]
        wrist_target = ref_wrist + wrist_action * self._wrist_scale

        # Fingers
        ref_finger = ref_joint_pos[:, self._finger_ids]
        if self._finger_scale >= 1.0:
            finger_target = (
                0.5 * (finger_action + 1.0) * self._finger_range + self._finger_lower
            )
        else:
            finger_target = (
                ref_finger + finger_action * self._finger_scale * self._finger_range
            )

        # Joint range hard-clamp — separate concern from action clip (above).
        finger_target = torch.clamp(
            finger_target, self._finger_lower, self._finger_upper
        )

        target = torch.cat([wrist_target, finger_target], dim=-1)
        self._entity.set_joint_position_target(target, joint_ids=self._all_joint_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
