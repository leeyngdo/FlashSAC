"""Observation term functions.

Ported verbatim from DexManip ``src/envs/observations/{base,motion_tracking}.py``
(only the ``..commands.motion_tracking`` import was rewired to ``..cmds.``).
Group/term selection lives in ``dexmanip/observations_cfg.py``.
"""

from .base import BaseObs
from .motion_tracking import MotionTrackingObs

__all__ = ["BaseObs", "MotionTrackingObs"]
