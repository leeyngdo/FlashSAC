"""Dexsuite observation terms re-exported from the IsaacLab dexsuite task.

These are the term functions used by the FlashSAC dexsuite observation group. They are
re-exported here so the :data:`OBS_TERMS` registry resolves them as local module-level names.
This module imports ``isaaclab_tasks`` and is therefore NOT part of the import-light path
(it must only be imported after ``AppLauncher`` has started the simulator).
"""

from __future__ import annotations

from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp.observations import (
    body_state_b,
    fingers_contact_force_b,
    object_point_cloud_b,
    object_quat_b,
)

__all__ = [
    "body_state_b",
    "fingers_contact_force_b",
    "object_point_cloud_b",
    "object_quat_b",
]
