"""Gym registration for the FlashSAC dexsuite Kuka-Allegro environment.

Importing this module registers the gym id. It is imported lazily by ``make_isaaclab_env``
(after ``AppLauncher`` has started the simulator) so that the package root stays import-light.

Note: re-registering ``Isaac-Dexsuite-Kuka-Allegro-Reorient-v0`` deliberately shadows the
stock config (gymnasium logs an "Overriding environment" warning, which is expected).
"""

import gymnasium as gym

from ... import dexsuite_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Reorient-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dexsuite_env_cfg.DexsuiteKukaAllegroReorientEnvCfg,
    },
)
