"""Reward wiring for the dexmanip motion-tracking task — the tuning surface.

Declarative port of DexManip ``config/envs/rewards/motion_tracking.yaml`` +
``src/envs/rewards/__init__.py:build_rewards``. The reward *functions* live in
``..mdp.rews.MotionTrackingRewards`` (verbatim upstream); this module owns the
*selection + weights + scales + params*. Edit the dicts below to tune, or
override per-term at runtime via ``configs/env/dexmanip.yaml`` →
``overrides.apply_dexmanip_overrides(reward=...)``.

build_rewards semantics (mirrored exactly in ``make_rewards``): for each term,
if it is in REWARD_PARAMS use those params; elif it is in REWARD_SCALES set
``params["scale"]``; then auto-inject ``command_name`` / ``asset_cfg``.
"""

from __future__ import annotations

from typing import Any

# term_name -> weight (summed by mjlab's RewardManager into one scalar).
REWARD_WEIGHTS: dict[str, float] = {
    # --- wrist tracking ---
    "tracking_r_wrist_trans_error_exp": 0.1,
    "tracking_r_wrist_rot_error_exp": 0.6,
    "tracking_r_wrist_lin_vel_error_exp": 0.1,
    "tracking_r_wrist_ang_vel_error_exp": 0.05,
    # --- per-finger tip tracking ---
    "tracking_r_thumb_tip_trans_error_exp": 0.9,
    "tracking_r_index_tip_trans_error_exp": 0.8,
    "tracking_r_middle_tip_trans_error_exp": 0.75,
    "tracking_r_ring_tip_trans_error_exp": 0.6,
    "tracking_r_pinky_tip_trans_error_exp": 0.6,
    # --- joint-level tracking ---
    "tracking_r_level1_trans_error_exp": 0.5,
    "tracking_r_level2_trans_error_exp": 0.3,
    # --- power penalties ---
    "r_finger_power_penalty": 0.5,
    "r_wrist_power_penalty": 0.5,
    # --- joint velocity ---
    "tracking_r_joints_vel_error_exp": 0.1,
    # --- object tracking ---
    "tracking_r_obj_trans_error_exp": 5.0,
    "tracking_r_obj_rot_error_exp": 1.0,
    "tracking_r_obj_lin_vel_error_exp": 0.1,
    "tracking_r_obj_ang_vel_error_exp": 0.1,
    # --- per-finger contact match (object interaction) ---
    "tracking_r_thumb_contact_match": 0.7,
    "tracking_r_index_contact_match": 0.6,
    "tracking_r_middle_contact_match": 0.55,
    "tracking_r_ring_contact_match": 0.45,
    "tracking_r_pinky_contact_match": 0.45,
    # --- mjlab built-in regularizers ---
    "action_rate": -0.01,
    "joint_limits": -1.0,
}

# term_name -> ``scale`` param (exp-kernel sharpness for tracking/power terms).
REWARD_SCALES: dict[str, float] = {
    "tracking_r_wrist_trans_error_exp": 40.0,
    "tracking_r_wrist_rot_error_exp": 1.0,
    "tracking_r_wrist_lin_vel_error_exp": 1.0,
    "tracking_r_wrist_ang_vel_error_exp": 1.0,
    "tracking_r_thumb_tip_trans_error_exp": 100.0,
    "tracking_r_index_tip_trans_error_exp": 90.0,
    "tracking_r_middle_tip_trans_error_exp": 80.0,
    "tracking_r_ring_tip_trans_error_exp": 60.0,
    "tracking_r_pinky_tip_trans_error_exp": 60.0,
    "tracking_r_level1_trans_error_exp": 50.0,
    "tracking_r_level2_trans_error_exp": 40.0,
    "r_finger_power_penalty": 10.0,
    "r_wrist_power_penalty": 2.0,
    "tracking_r_joints_vel_error_exp": 1.0,
    "tracking_r_obj_trans_error_exp": 80.0,
    "tracking_r_obj_rot_error_exp": 8.0,
    "tracking_r_obj_lin_vel_error_exp": 1.0,
    "tracking_r_obj_ang_vel_error_exp": 1.0,
}

# term_name -> explicit params (contact-match terms; take precedence over scale).
_CONTACT_PARAMS = {"beta": 40.0, "gamma": 200.0, "tol": 0.002}
REWARD_PARAMS: dict[str, dict[str, Any]] = {
    "tracking_r_thumb_contact_match": dict(_CONTACT_PARAMS),
    "tracking_r_index_contact_match": dict(_CONTACT_PARAMS),
    "tracking_r_middle_contact_match": dict(_CONTACT_PARAMS),
    "tracking_r_ring_contact_match": dict(_CONTACT_PARAMS),
    "tracking_r_pinky_contact_match": dict(_CONTACT_PARAMS),
}


def make_rewards(command_name: str = "motion", entity_name: str = "robot") -> dict[str, Any]:
    """Build ``dict[str, RewardTermCfg]`` — mirrors DexManip ``build_rewards`` (params if/elif scale)."""
    from mjlab.managers.reward_manager import RewardTermCfg

    from ..mdp import auto_inject
    from ..mdp.rews import MotionTrackingRewards

    out: dict[str, Any] = {}
    for name, weight in REWARD_WEIGHTS.items():
        fn = getattr(MotionTrackingRewards, name)
        params: dict[str, Any] = {}
        if name in REWARD_PARAMS:
            params.update(REWARD_PARAMS[name])
        elif name in REWARD_SCALES:
            params["scale"] = REWARD_SCALES[name]
        auto_inject(fn, params, command_name=command_name, entity_name=entity_name)
        out[name] = RewardTermCfg(func=fn, weight=float(weight), params=params)
    return out
