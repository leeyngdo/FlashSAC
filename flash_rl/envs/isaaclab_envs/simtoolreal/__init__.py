"""Vendored SimToolReal dexterous tool-manipulation task (Direct workflow).

Vendors the Isaac Lab environment from ``tylerlum/simtoolreal`` (commit 8405866):
a Kuka iiwa14 + SHARPA-hand goal-pose-reaching task over procedurally generated
tool objects with the paper's sim2real domain randomization. Deviations from
upstream: asset paths anchored on the package ``ASSET_DIR``, the adjacent-links
self-collision map vendored as :mod:`.adjacent_links`, scalar metrics mirrored
into ``extras["log"]``, and the tolerance curriculum frozen while ``eval_mode``
is set.

This top-level ``__init__`` is intentionally import-light: NO ``isaaclab``/``gym``
imports and NO gym registration at import time; ``make_isaaclab_env`` imports
:mod:`.config.kuka_sharpa` only after ``AppLauncher`` has started the simulator.
"""
