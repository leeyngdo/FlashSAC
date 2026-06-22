"""MDP term aggregation for the motion-tracking task.

This package re-exports the IsaacLab built-in MDP terms together with the
vendored, modularized terms for commands, observations, rewards, terminations,
and events. Config files import this package once as ``from . import mdp as mdp``
and reference every term (builtin or custom) as ``mdp.<symbol>``.

The aggregation order is load-bearing: ``isaaclab.envs.mdp`` is star-imported
FIRST so builtins (e.g. ``action_rate_l2``, ``joint_pos_limits``,
``undesired_contacts``, ``time_out``, ``generated_commands``,
``randomize_rigid_body_material``, ``push_by_setting_velocity``) resolve, and the
local submodules are imported afterward so custom funcs are added without
shadowing the builtins they depend on.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .cmds import *  # noqa: F401, F403
from .obs import *  # noqa: F401, F403
from .rews import *  # noqa: F401, F403
from .terms import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403

# Registries exposed for the override layer.
from .obs import OBS_TERMS  # noqa: F401
from .rews import REW_TERMS  # noqa: F401
from .terms import TERM_TERMS  # noqa: F401
