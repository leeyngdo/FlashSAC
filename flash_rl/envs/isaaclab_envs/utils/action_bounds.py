"""Action-bound helpers for IsaacLab environments."""

from __future__ import annotations

import torch


def compute_joint_limit_action_bound(
    soft_limits: torch.Tensor,
    default_pos: torch.Tensor,
    action_scale: torch.Tensor,
    fraction: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an asymmetric affine action window from soft joint limits.

    The IsaacLab ``JointPositionAction`` target is ``default + scale * action``.
    This returns ``(bias, range)`` for ``final = bias + range * tanh_action``.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}.")

    lower = soft_limits[..., 0]
    upper = soft_limits[..., 1]
    zero = action_scale.abs() < 1e-8
    safe_scale = torch.where(zero, torch.ones_like(action_scale), action_scale)

    action_to_lower = fraction * (lower - default_pos) / safe_scale
    action_to_upper = fraction * (upper - default_pos) / safe_scale
    action_low = torch.minimum(action_to_lower, action_to_upper)
    action_high = torch.maximum(action_to_lower, action_to_upper)
    action_bias = 0.5 * (action_high + action_low)
    action_range = 0.5 * (action_high - action_low)

    action_bias = torch.where(zero, torch.zeros_like(action_bias), action_bias)
    action_range = torch.where(zero, torch.zeros_like(action_range), action_range)
    return action_bias, action_range
