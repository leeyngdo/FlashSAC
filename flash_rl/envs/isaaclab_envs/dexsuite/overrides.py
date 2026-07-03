"""Config override helpers for the FlashSAC dexsuite task.

Tuning and observation layout live in the config classes (:mod:`.dexsuite_env_cfg`); this
module only applies the hydra config blocks (``reward``/``observation``/``termination``/
``robot``) and the dot-path ``cfg_overrides`` on top of the parsed env config, reusing the
generic duck-typed helpers from the tracking task.

This module is intentionally import-light: it works on duck-typed config objects and does not
import IsaacLab.
"""

from __future__ import annotations

from typing import Any

from ..tracking.overrides import (
    _apply_observation_overrides,
    _apply_reward_overrides,
    _apply_robot_overrides,
    _apply_termination_overrides,
    apply_cfg_overrides,
)


def apply_dexsuite_overrides(
    env_cfg: Any,
    reward: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
    termination: dict[str, Any] | None = None,
    robot: dict[str, Any] | None = None,
    motion: dict[str, Any] | None = None,
    cfg_overrides: dict[str, Any] | None = None,
) -> Any:
    """Apply friendly dexsuite config blocks, then dot-path overrides last.

    ``motion`` is accepted for signature parity with ``make_isaaclab_env`` but is not
    applicable to the dexsuite task.
    """
    if motion:
        raise ValueError("The dexsuite task does not support the 'motion' config block.")
    _apply_reward_overrides(env_cfg, reward)
    _apply_observation_overrides(env_cfg, observation)
    _apply_termination_overrides(env_cfg, termination)
    _apply_robot_overrides(env_cfg, robot)
    return apply_cfg_overrides(env_cfg, cfg_overrides)
