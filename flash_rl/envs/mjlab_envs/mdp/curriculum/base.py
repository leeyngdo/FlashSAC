from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _set_gravity_z(env: "ManagerBasedRlEnv", gz: float) -> None:
    """Set the shared z-gravity across mjlab's unbatched or batched model views."""
    gravity = env.sim.model.opt.gravity
    if gravity.ndim == 1:
        gravity[2] = gz
        return

    # mjlab may expose model.opt.gravity as shape (1, 3) or as an expanded
    # (num_envs, 3) view with stride 0 on the env dimension. Writing all rows of
    # an expanded view is illegal, but the first row aliases the shared storage.
    if gravity.shape[0] > 1 and gravity.stride(0) == 0:
        gravity[0, 2] = gz
    else:
        gravity[..., 2] = gz


def gravity_curriculum(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    schedule_steps: int = 1920,
    full_g: float = 9.81,
) -> dict[str, torch.Tensor]:
    """Linear ramp of -z gravity from 0 -> -full_g over schedule_steps env-steps.

    Mutates env.sim.model.opt.gravity[2] in place. Held at -full_g once
    schedule_steps is reached.

    Special case: ``schedule_steps <= 0`` skips the ramp entirely — gravity is
    -full_g from the very first step (constant mode).
    """
    s = int(env.common_step_counter)
    if schedule_steps <= 0:
        frac = 1.0
    else:
        frac = min(s / max(1, schedule_steps), 1.0)
    gz = -float(full_g) * frac
    _set_gravity_z(env, gz)
    return {"gravity_z": torch.tensor(gz)}


def xfrc_curriculum(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    command_name: str = "motion",
    omega_n_start: float = 0.0,
    omega_n_end: float = 0.0,
    schedule_steps: int = 0,
    delay_steps: int = 0,
    zeta: float = 1.0,
    drive_omega_rot: bool = True,
) -> dict[str, torch.Tensor]:
    """Mass-normalized soft-attractor curriculum for `pin_mode="xfrc"`.

    Single knob: natural frequency ω_n (rad/s). Per-object mass is auto-detected
    from each object entity's body_mass. Gains derived as:

        kp_pos = m * ω_n^2
        kv_pos = 2 * ζ * sqrt(m * kp_pos) = 2 * ζ * m * ω_n

    Schedule: linear interpolation from ``omega_n_start`` → ``omega_n_end`` over
    ``schedule_steps`` env-steps. After that, held at omega_n_end. ``schedule_steps<=0``
    means held at omega_n_start from step 0 (constant mode — for the baseline).

    Mutates ``cmd.cfg.xfrc_kp_pos`` / ``xfrc_kv_pos`` each tick. Rotational gains
    are NOT touched (left at whatever cfg has — typically 0).

    Returns ``{xfrc_omega_n, xfrc_kp_pos, xfrc_kv_pos}`` for wandb logging
    (mirrors the gravity curriculum's logging pattern).
    """
    s = int(env.common_step_counter)
    s_post_delay = max(0, s - int(delay_steps))
    if schedule_steps <= 0:
        w = float(omega_n_start)
    else:
        frac = min(s_post_delay / max(1, schedule_steps), 1.0)
        w = float(omega_n_start) + frac * (float(omega_n_end) - float(omega_n_start))

    cmd = env.command_manager.get_term(command_name)
    obj_names = list((cmd.cfg.object.entity_names or {}).values())
    obj = env.scene[obj_names[0]]
    # body_mass is shape (B, nbody) batched per env. Index env 0 explicitly,
    # then sum over the entity's body_ids. Earlier bug: bare body_ids indexed
    # the BATCH dim instead, returning the entire env's mass (~10x too high).
    m = float(env.sim.model.body_mass[0, obj.indexing.body_ids].sum())

    kp = m * w * w
    kv = 2.0 * float(zeta) * m * w
    cmd.cfg.object.xfrc.kp.pos = kp
    cmd.cfg.object.xfrc.kd.pos = kv
    if drive_omega_rot:
        cmd.cfg.object.xfrc.omega_rot = float(w)
    return {
        "xfrc_omega_n": torch.tensor(w),
        "xfrc_kp_pos": torch.tensor(kp),
        "xfrc_kv_pos": torch.tensor(kv),
        "xfrc_obj_mass": torch.tensor(m),
        "xfrc_omega_rot": torch.tensor(float(w))
        if drive_omega_rot
        else torch.tensor(0.0),
    }


def build_obj_term_stages(
    delay_steps: int,
    duration_steps: int,
) -> tuple[list[dict], list[dict]]:
    """Build (pos_stages, rot_stages) for ManipTrans-style obj-termination tightening.

    Returns lists of {"step": int, "params": {"threshold"|"threshold_deg": float}}
    suitable for ``mjlab.envs.mdp.curriculums.termination_curriculum``.

    Stages land at fractions 0, 0.25, 0.5, 0.75, 1.0 of the duration window:
    ``step = delay_steps + int(duration_steps * frac)``. Thresholds shrink
    cubically from ~0.058 m / ~87° at frac=0 to 0.02 m / 30° at frac=1.
    """
    pos_stages: list[dict] = []
    rot_stages: list[dict] = []
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        scale = (math.e * 2.0) ** (-frac) * 0.3 + 0.7
        pos_thr = 0.02 / 0.343 * scale**3
        rot_thr = 30.0 / 0.343 * scale**3
        s = delay_steps + int(duration_steps * frac)
        pos_stages.append({"step": s, "params": {"threshold": float(pos_thr)}})
        rot_stages.append({"step": s, "params": {"threshold_deg": float(rot_thr)}})
    return pos_stages, rot_stages
