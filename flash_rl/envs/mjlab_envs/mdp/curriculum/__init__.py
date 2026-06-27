"""Curriculum term functions.

Ported verbatim from DexManip ``src/envs/curriculum/base.py`` (self-contained;
only imports mjlab/torch). Selection/params live in ``dexmanip/curriculum_cfg.py``.
DexManip's default ``base`` preset: gravity_curriculum + xfrc_curriculum.
"""

from .base import gravity_curriculum, xfrc_curriculum

__all__ = ["gravity_curriculum", "xfrc_curriculum"]
