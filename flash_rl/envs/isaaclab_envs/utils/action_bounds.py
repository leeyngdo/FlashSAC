"""Action-bound helpers for IsaacLab environments."""

from __future__ import annotations

import torch


def compute_joint_limit_action_bound(
    soft_limits: torch.Tensor,
    default_pos: torch.Tensor,
    action_scale: torch.Tensor,
    fraction: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a zero-centered action window from joint limits.

    The IsaacLab ``JointPositionAction`` target is ``default + scale * action``.
    Following Holosoma FastSAC, action zero should keep the default pose, while
    ``range`` reaches the farther joint limit from the default pose.
    This returns ``(bias, range)`` for ``final = bias + range * tanh_action``.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}.")

    lower = soft_limits[..., 0]
    upper = soft_limits[..., 1]
    zero = action_scale.abs() < 1e-8
    safe_scale = torch.where(zero, torch.ones_like(action_scale), action_scale)

    range_to_lower = torch.abs(lower - default_pos)
    range_to_upper = torch.abs(upper - default_pos)
    action_range = fraction * torch.maximum(range_to_lower, range_to_upper) / safe_scale.abs()
    action_bias = torch.zeros_like(action_range)

    action_range = torch.where(zero, torch.zeros_like(action_range), action_range)
    return action_bias, action_range
