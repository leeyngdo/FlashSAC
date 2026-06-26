"""Termination MDP terms for the motion-tracking task.

This package aggregates the custom termination functions and exposes them through the
``TERM_TERMS`` registry (a ``dict[str, Callable]`` mapping term name to the term function).
A user adds a termination by: writing the function in ``mdp/terms/<file>.py``, registering it
in ``TERM_TERMS`` here, and adding a ``DoneTerm`` in ``tracking/terminations_cfg.py``.
"""

from __future__ import annotations

from typing import Callable

# Tracking
from .tracking import (
    bad_anchor_ori,
    bad_anchor_pos,
    bad_anchor_pos_z_only,
    bad_motion_body_pos,
    bad_motion_body_pos_z_only,
)

__all__ = [
    "bad_anchor_ori",
    "bad_anchor_pos",
    "bad_anchor_pos_z_only",
    "bad_motion_body_pos",
    "bad_motion_body_pos_z_only",
    "TERM_TERMS",
]

TERM_TERMS: dict[str, Callable] = {
    "bad_anchor_pos": bad_anchor_pos,
    "bad_anchor_pos_z_only": bad_anchor_pos_z_only,
    "bad_anchor_ori": bad_anchor_ori,
    "bad_motion_body_pos": bad_motion_body_pos,
    "bad_motion_body_pos_z_only": bad_motion_body_pos_z_only,
}
