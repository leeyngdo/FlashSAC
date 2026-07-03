"""FlashSAC overrides for the IsaacLab dexsuite Kuka-Allegro reorientation task.

Derives from the stock ``isaaclab_tasks`` dexsuite task (instead of vendoring it) and provides
a tuned config class with FlashSAC-compatible flat ``policy`` observations
(:mod:`.dexsuite_env_cfg`, :mod:`.observations_cfg`) plus the gym registration
(:mod:`.config.kuka_allegro`) for ``Isaac-Dexsuite-Kuka-Allegro-Reorient-v0`` (shadows the
stock id).

This top-level ``__init__`` is intentionally import-light: NO ``isaaclab``/``gym`` imports and
NO gym registration at import time; ``make_isaaclab_env`` imports :mod:`.config.kuka_allegro`
only after ``AppLauncher`` has started the simulator.
"""
