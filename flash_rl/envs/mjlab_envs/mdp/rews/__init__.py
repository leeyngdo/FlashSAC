"""Reward term functions.

Ported verbatim from DexManip ``src/envs/rewards/{base,motion_tracking}.py``
(only the ``..commands.motion_tracking`` import was rewired to ``..cmds.``).
Weights/selection are NOT here — they live in ``dexmanip/rewards_cfg.py``.
"""

from .base import BaseRewards
from .motion_tracking import MotionTrackingRewards

__all__ = ["BaseRewards", "MotionTrackingRewards"]
