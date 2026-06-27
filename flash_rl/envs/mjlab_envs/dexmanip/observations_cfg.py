"""Observation wiring for the dexmanip motion-tracking task (teacher preset).

REFERENCE IMPLEMENTATION (KraftonLab / isaaclab_envs style): the observation
*functions* live in ``..mdp.obs`` (``MotionTrackingObs``, ported verbatim from
DexManip ``src/envs/observations/``), and this module owns the *selection +
per-term scales/noise/params* declaratively.

Ports DexManip:
  - builder: ``src/envs/observations/__init__.py``  (``build_observations`` + ``_build_group``)
  - preset:  ``config/envs/obs/motion_tracking_teacher.yaml``
            (``obs_dict.actor_obs`` / ``critic_obs`` term lists, ``obs_params``,
             ``obs_scales``, ``noise_scales``, ``nan_policy``,
             actor/critic ``enable_corruption``)

The YAML values are inlined here as plain Python literals — no yaml / hydra /
OmegaConf. Both groups concatenate their terms (``concatenate_terms=True``).
"""

from __future__ import annotations

from typing import Any

# --- DexManip config/envs/obs/motion_tracking_teacher.yaml (inlined verbatim) ---

# obs_dict.actor_obs (order matters: concatenated in this order).
_ACTOR_OBS: list[str] = [
    # robot proprioception
    "wrist_state",
    "robot_joint_pos",
    "robot_joint_cos_sin",
    "robot_joint_vel",
    "last_action",
    # mano reference
    "mano_wrist_lin_vel",
    "mano_wrist_quat",
    "mano_wrist_ang_vel",
    "mano_joints_vel",
    # mano deltas
    "mano_wrist_trans_delta",
    "mano_wrist_quat_delta",
    "mano_fingertip_trans_delta",
    "mano_wrist_lin_vel_delta",
    "mano_wrist_ang_vel_delta",
    "mano_joints_vel_delta",
    # tactile
    "ref_contact_flags",
    "r_contact_force",
    "r_contact_force_history",
    "r_tip_penetration",
    # finger-to-object distances
    "mano_tips_distance_obs",
    "robot_obj_distance",
    # object state
    "obj_trans_relative",
    "obj_quat",
    "obj_lin_vel",
    "obj_ang_vel",
    # object deltas
    "obj_trans_delta",
    "obj_lin_vel_delta",
    "obj_quat_delta",
    "obj_ang_vel_delta",
    # object auxiliary
    "future_obj_trans_delta",
    "future_obj_lin_vel",
    "obj_local_sdf_at_keypoints",
]

# obs_dict.critic_obs (identical term list to actor in the teacher preset).
_CRITIC_OBS: list[str] = list(_ACTOR_OBS)

# Only terms that need extra params beyond auto-injected command_name/asset_cfg.
_OBS_PARAMS: dict[str, dict[str, Any]] = {
    "r_contact_force_history": {"history_len": 3},
}

# Per-term scalar multiplier. 1.0 == no-op (builder leaves scale=None).
_OBS_SCALES: dict[str, float] = {
    "robot_joint_pos": 1.0,
    "robot_joint_cos_sin": 1.0,
    "wrist_state": 1.0,
    "robot_joint_vel": 1.0,
    "last_action": 1.0,
    "mano_wrist_lin_vel": 1.0,
    "mano_wrist_quat": 1.0,
    "mano_wrist_ang_vel": 1.0,
    "mano_wrist_trans_delta": 1.0,
    "mano_wrist_quat_delta": 1.0,
    "mano_fingertip_trans_delta": 1.0,
    "mano_wrist_lin_vel_delta": 1.0,
    "mano_wrist_ang_vel_delta": 1.0,
    "mano_joints_vel": 1.0,
    "mano_joints_vel_delta": 1.0,
    "ref_contact_flags": 1.0,
    "r_contact_force": 1.0,
    "r_contact_force_history": 1.0,
    "r_tip_penetration": 1.0,
    "mano_tips_distance_obs": 1.0,
    "robot_obj_distance": 1.0,
    "obj_trans_relative": 1.0,
    "obj_quat": 1.0,
    "obj_lin_vel": 1.0,
    "obj_ang_vel": 1.0,
    "obj_trans_delta": 1.0,
    "obj_lin_vel_delta": 1.0,
    "obj_quat_delta": 1.0,
    "obj_ang_vel_delta": 1.0,
    "future_obj_trans_delta": 1.0,
    "future_obj_lin_vel": 1.0,
    "obj_local_sdf_at_keypoints": 1.0,
}

# Per-term symmetric uniform noise amplitude. 0.0 == no noise (builder skips).
# Active only on groups with enable_corruption=True (actor in this preset).
_NOISE_SCALES: dict[str, float] = {
    "robot_joint_pos": 0.0,
    "robot_joint_cos_sin": 0.0,
    "wrist_state": 0.0,
    "robot_joint_vel": 0.0,
    "last_action": 0.0,
    "mano_wrist_lin_vel": 0.0,
    "mano_wrist_quat": 0.0,
    "mano_wrist_ang_vel": 0.0,
    "mano_wrist_trans_delta": 0.0,
    "mano_wrist_quat_delta": 0.0,
    "mano_fingertip_trans_delta": 0.0,
    "mano_wrist_lin_vel_delta": 0.0,
    "mano_wrist_ang_vel_delta": 0.0,
    "mano_joints_vel": 0.0,
    "mano_joints_vel_delta": 0.0,
    "ref_contact_flags": 0.0,
    "r_contact_force": 0.0,
    "r_contact_force_history": 0.0,
    "r_tip_penetration": 0.0,
    "mano_tips_distance_obs": 0.0,
    "robot_obj_distance": 0.0,
    "obj_trans_relative": 0.0,
    "obj_quat": 0.0,
    "obj_lin_vel": 0.0,
    "obj_ang_vel": 0.0,
    "obj_trans_delta": 0.0,
    "obj_lin_vel_delta": 0.0,
    "obj_quat_delta": 0.0,
    "obj_ang_vel_delta": 0.0,
    "future_obj_trans_delta": 0.0,
    "future_obj_lin_vel": 0.0,
    "obj_local_sdf_at_keypoints": 0.0,
}

# actor gets noise corruption, critic does not.
_ACTOR_ENABLE_CORRUPTION: bool = True
_CRITIC_ENABLE_CORRUPTION: bool = False
_NAN_POLICY: str = "sanitize"


def _build_group(
    cls: type,
    term_names: list[str],
    *,
    command_name: str,
    entity_name: str,
    enable_corruption: bool,
) -> Any:
    """Mirror DexManip ``_build_group``: assemble one ObservationGroupCfg."""
    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )
    from mjlab.utils.noise import UniformNoiseCfg

    from ..mdp import auto_inject, materialize

    terms: dict[str, ObservationTermCfg] = {}
    for name in term_names:
        fn = getattr(cls, name)
        params: dict[str, Any] = {}

        if name in _OBS_PARAMS:
            for k, v in dict(_OBS_PARAMS[name]).items():
                params[k] = materialize(v)

        auto_inject(fn, params, command_name=command_name, entity_name=entity_name)

        obs_scale_val = float(_OBS_SCALES.get(name, 1.0))
        obs_scale = obs_scale_val if obs_scale_val != 1.0 else None

        obs_noise_amp = float(_NOISE_SCALES.get(name, 0.0))
        obs_noise = (
            UniformNoiseCfg(n_min=-obs_noise_amp, n_max=obs_noise_amp)
            if obs_noise_amp > 0.0
            else None
        )

        terms[name] = ObservationTermCfg(
            func=fn,
            params=params,
            scale=obs_scale,
            noise=obs_noise,
        )

    return ObservationGroupCfg(
        terms=terms,
        concatenate_terms=True,
        enable_corruption=enable_corruption,
        nan_policy=_NAN_POLICY,
    )


def make_observations(
    *,
    command_name: str = "motion",
    entity_name: str = "robot",
) -> dict[str, Any]:
    """Build actor + critic observation groups (``dict[str, ObservationGroupCfg]``).

    Ports DexManip ``build_observations``: sources term lists / scales / noise /
    params from the inlined teacher preset instead of a parsed ``obs`` block.
    """
    from ..mdp.obs import MotionTrackingObs

    return {
        "actor": _build_group(
            MotionTrackingObs,
            _ACTOR_OBS,
            command_name=command_name,
            entity_name=entity_name,
            enable_corruption=_ACTOR_ENABLE_CORRUPTION,
        ),
        "critic": _build_group(
            MotionTrackingObs,
            _CRITIC_OBS,
            command_name=command_name,
            entity_name=entity_name,
            enable_corruption=_CRITIC_ENABLE_CORRUPTION,
        ),
    }
