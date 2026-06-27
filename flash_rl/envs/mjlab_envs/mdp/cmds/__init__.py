"""Command term: motion tracking.

Ported from DexManip ``src/envs/commands/motion_tracking/`` and kept close to
the tracking branch so term-level changes stay diff-able. The subpackage is
self-contained: all internal imports are intra-package (``.motion_library``,
``.sdf``, ``.object_properties``, ``.hand_properties``,
``.motion_tracking_cfg``), and the only external deps are ``mjlab.*`` (resolved
against the DAVIAN fork in the isolated env) + trimesh.

The motion command eager-loads ``motion.pt`` at __init__ (``motion_library``) and
reads the object pool resolved by the scene builder; no data -> no env.
"""

from .motion_tracking import MotionTrackingCommand
from .motion_tracking_cfg import MotionTrackingCommandCfg

__all__ = ["MotionTrackingCommand", "MotionTrackingCommandCfg"]
