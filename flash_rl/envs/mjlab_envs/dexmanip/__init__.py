"""DexManip motion-tracking task — content only (cfg assembly + override seam).

Pure env-package content, mirroring ``isaaclab_envs/tracking/``: NO VectorEnv
wrapper here. The wrapper + entry point live in ``flash_rl.envs.mjlab``
(``make_dexmanip_env``), which imports the two callables below — keeping the
dependency top-down (``mjlab.py -> mjlab_envs``), exactly like
``isaaclab.py -> isaaclab_envs``.

Declarative ``*_cfg.py`` modules select & weight terms from ``..mdp``;
``build_dexmanip_env_cfg`` composes them into a ``ManagerBasedRlEnvCfg``;
``apply_dexmanip_overrides`` applies the per-term toggles from
``configs/env/dexmanip.yaml``.
"""

from .dexmanip_env_cfg import build_dexmanip_env_cfg
from .overrides import apply_dexmanip_overrides

__all__ = ["build_dexmanip_env_cfg", "apply_dexmanip_overrides"]
