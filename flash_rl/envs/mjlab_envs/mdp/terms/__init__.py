"""Termination term functions.

Ported verbatim from DexManip ``src/envs/terminations/{base,motion_tracking}.py``
(only the ``..commands.motion_tracking`` import was rewired to ``..cmds.``).
The terminated-vs-truncated split is preserved. Selection/params live in
``dexmanip/terminations_cfg.py``.
"""

from .base import BaseTerminations
from .motion_tracking import MotionTrackingTerminations

__all__ = ["BaseTerminations", "MotionTrackingTerminations"]
