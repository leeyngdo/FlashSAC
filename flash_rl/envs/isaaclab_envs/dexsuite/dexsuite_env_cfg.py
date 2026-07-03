"""FlashSAC environment configuration for the dexsuite Kuka-Allegro reorientation task.

Derives from the stock IsaacLab dexsuite config and applies two overrides: reward/termination/
curriculum tuning (:func:`_apply_tuning`) and a single flat ``policy`` observation group
(:mod:`.observations_cfg`).
"""

from __future__ import annotations

from typing import Any

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg import (
    DexsuiteKukaAllegroReorientEnvCfg as _StockReorientEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp import initial_final_interpolate_fn, modify_term_cfg

from .. import mdp as mdp
from .observations_cfg import DexsuiteKukaAllegroFlatObservationsCfg


def _apply_tuning(env_cfg: Any) -> None:
    """Tune rewards/terminations/curriculum (ported from ooctipus/IsaacLab ``dexsuite_fix``)."""
    env_cfg.rewards.fingers_to_object.weight = 0.25
    env_cfg.rewards.position_tracking.params["std"] = 0.1
    env_cfg.rewards.orientation_tracking.params["std"] = 1.0
    env_cfg.rewards.success.params["pos_std"] = 0.05
    # tighter drop bound, relaxed back by the oob_adr curriculum term below
    env_cfg.terminations.object_out_of_bound.func = mdp.TERM_TERMS["out_of_bound"]
    env_cfg.terminations.object_out_of_bound.params["in_bound_range"] = {
        "x": (-1.5, 0.5),
        "y": (-2.0, 2.0),
        "z": (0.3, 2.0),
    }
    if env_cfg.curriculum is not None:
        # keep the ADR promotion tolerances in sync with the new success stds
        env_cfg.curriculum.adr.params["pos_tol"] = env_cfg.rewards.success.params["pos_std"] / 2
        env_cfg.curriculum.adr.params["rot_tol"] = env_cfg.rewards.success.params["rot_std"] / 2
        env_cfg.curriculum.oob_adr = CurrTerm(
            func=modify_term_cfg,
            params={
                "address": "terminations.object_out_of_bound.params.in_bound_range.z",
                "modify_fn": initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": (0.3, 2.0),
                    "final_value": (0.0, 2.0),
                    "difficulty_term_str": "adr",
                },
            },
        )


def _retarget_obs_noise_adr(env_cfg: Any) -> None:
    """Repoint observation-noise curriculum terms at the merged ``policy`` group.

    The stock curriculum addresses noise in the ``proprio``/``perception`` groups, which no
    longer exist after the merge; terms whose observation is absent from the policy group are
    dropped.
    """
    if env_cfg.curriculum is None:
        return
    for name, term in list(vars(env_cfg.curriculum).items()):
        if term is None:
            continue
        address = term.params.get("address", "")
        if not address.startswith("observations."):
            continue
        _, _, obs_term, rest = address.split(".", 3)
        if getattr(env_cfg.observations.policy, obs_term, None) is None:
            setattr(env_cfg.curriculum, name, None)
        else:
            term.params["address"] = f"observations.policy.{obs_term}.{rest}"


@configclass
class DexsuiteKukaAllegroReorientEnvCfg(_StockReorientEnvCfg):
    """Tuned reorient task with the stock observations merged into a flat ``policy`` group."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.observations = DexsuiteKukaAllegroFlatObservationsCfg()
        _apply_tuning(self)
        _retarget_obs_noise_adr(self)
