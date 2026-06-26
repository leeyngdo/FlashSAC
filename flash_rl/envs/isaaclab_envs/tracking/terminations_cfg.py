"""Termination configuration for the motion-tracking task.

Note the func-vs-field indirection (matching WBT): the ``anchor_pos`` cfg field
uses the ``bad_anchor_pos_z_only`` func and ``ee_body_pos`` uses the
``bad_motion_body_pos_z_only`` func. ``time_out`` is the IsaacLab builtin driven
by the motion clip length.
"""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .. import mdp as mdp


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.5},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": None,
        },
    )
