"""Gym registration for the G1 tracking environments.

Importing this module registers the G1 tracking gym ids. It is imported lazily by
``make_isaaclab_env`` (after ``AppLauncher`` has started the simulator) so that the
package root stays import-light and free of ``isaaclab`` imports.
"""

import gymnasium as gym

from . import flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
    },
)

gym.register(
    id="Isaac-Tracking-Flat-G1-WoSE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatWoStateEstimationEnvCfg,
    },
)
