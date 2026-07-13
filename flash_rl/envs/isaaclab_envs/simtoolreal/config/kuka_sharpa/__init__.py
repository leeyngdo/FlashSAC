"""Gym registration for the FlashSAC SimToolReal Kuka-Sharpa environment.

Importing this module registers the gym id. It is imported lazily by ``make_isaaclab_env``
(after ``AppLauncher`` has started the simulator) so that the package root stays import-light.
"""

import gymnasium as gym

from ... import simtoolreal_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-SimToolReal-Kuka-Sharpa-Direct-v0",
    entry_point="flash_rl.envs.isaaclab_envs.simtoolreal.simtoolreal_env:SimToolRealEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": simtoolreal_env_cfg.SimToolRealEnvCfg,
    },
)
