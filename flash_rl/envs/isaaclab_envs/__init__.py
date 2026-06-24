"""Vendored modular IsaacLab motion-tracking task package.

This package vendors the BeyondMimic G1 tracking task (from ``whole_body_tracking``)
into FlashSAC and keeps the task split into modular observations, reward and
termination registries, and a multi-dataset motion loader with adaptive sampling.

This top-level ``__init__`` is intentionally import-light: it contains NO ``isaaclab``
or ``gym`` imports and triggers NO gym registration at import time. Pure submodules
(e.g. :mod:`flash_rl.envs.isaaclab_envs.utils.motion_loader`) therefore remain
importable without IsaacLab installed. Gym registration is triggered explicitly by
``make_isaaclab_env`` via ``import flash_rl.envs.isaaclab_envs.tracking.config.g1``
only after ``AppLauncher`` has started the simulator.
"""
