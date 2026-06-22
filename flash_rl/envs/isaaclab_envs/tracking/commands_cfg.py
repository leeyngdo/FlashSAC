"""Command configuration for the G1 motion-tracking task."""

from __future__ import annotations

from isaaclab.utils import configclass

from .. import mdp as mdp

# Velocity perturbation range shared by the motion command resampling and the
# ``push_robot`` interval event (see :mod:`.events_cfg`).
VELOCITY_RANGE: dict[str, tuple[float, float]] = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


@configclass
class CommandsCfg:
    """Command specifications for the MDP.

    The ``anchor_body_name`` and ``body_names`` fields of the motion command are
    intentionally left to the per-robot flat env config, and ``motion_files`` is
    supplied at runtime via the override layer / Hydra config.
    """

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )
