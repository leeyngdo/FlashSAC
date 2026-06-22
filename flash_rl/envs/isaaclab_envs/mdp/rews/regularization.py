"""Regularization reward terms.

These are thin re-exports of IsaacLab built-in penalties so they are addressable
through the local reward registry and the per-group override layer.
"""

from __future__ import annotations

from isaaclab.envs.mdp import action_rate_l2, joint_pos_limits

__all__ = ["action_rate_l2", "joint_pos_limits"]
