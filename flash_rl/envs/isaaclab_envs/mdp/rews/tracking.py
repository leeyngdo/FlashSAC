"""Motion-tracking reward terms.

These reward terms penalize the deviation between the robot and the reference
motion, both in the global anchor frame and in the relative body frame. Each
term returns an exponential tracking reward of shape ``(num_envs,)``.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_error_magnitude

from ..cmds.motion_command import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    """Resolves a list of body names to their indices in the command body set.

    Args:
        command: The motion command term holding the configured body names.
        body_names: The body names to select, or ``None`` to select all bodies.

    Returns:
        The list of indices into ``command.cfg.body_names`` matching the request.
    """
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Exponential reward on the global anchor position error.

    Args:
        env: The environment instance.
        command_name: The name of the motion command term.
        std: The standard deviation of the exponential kernel.

    Returns:
        The reward of shape ``(num_envs,)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Exponential reward on the global anchor orientation error.

    Args:
        env: The environment instance.
        command_name: The name of the motion command term.
        std: The standard deviation of the exponential kernel.

    Returns:
        The reward of shape ``(num_envs,)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Exponential reward on the relative body position error.

    Args:
        env: The environment instance.
        command_name: The name of the motion command term.
        std: The standard deviation of the exponential kernel.
        body_names: The bodies to track, or ``None`` to track all bodies.

    Returns:
        The reward of shape ``(num_envs,)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Exponential reward on the relative body orientation error.

    Args:
        env: The environment instance.
        command_name: The name of the motion command term.
        std: The standard deviation of the exponential kernel.
        body_names: The bodies to track, or ``None`` to track all bodies.

    Returns:
        The reward of shape ``(num_envs,)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Exponential reward on the global body linear-velocity error.

    Args:
        env: The environment instance.
        command_name: The name of the motion command term.
        std: The standard deviation of the exponential kernel.
        body_names: The bodies to track, or ``None`` to track all bodies.

    Returns:
        The reward of shape ``(num_envs,)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Exponential reward on the global body angular-velocity error.

    Args:
        env: The environment instance.
        command_name: The name of the motion command term.
        std: The standard deviation of the exponential kernel.
        body_names: The bodies to track, or ``None`` to track all bodies.

    Returns:
        The reward of shape ``(num_envs,)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)
