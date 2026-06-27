"""Event / domain-randomization term functions.

Ported verbatim from DexManip ``src/envs/events/base.py`` (re-exports mjlab
built-ins for getattr lookup). Selection/params live in ``dexmanip/events_cfg.py``.
DexManip's default preset is ``no_domain_rand`` (reset_scene_to_default only).
"""

from .base import reset_scene_to_default

__all__ = ["reset_scene_to_default"]
