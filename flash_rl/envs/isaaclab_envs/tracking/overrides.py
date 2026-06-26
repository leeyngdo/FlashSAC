"""Config override helpers for the IsaacLab tracking task.

This module is intentionally import-light: it works on duck-typed config objects
and does not import IsaacLab.
"""

from __future__ import annotations

from typing import Any


def _set_by_path(obj: Any, dotted: str, value: Any) -> None:
    """Set a nested dict member or object attribute addressed by a dotted path."""
    parts = dotted.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = cur[part] if isinstance(cur, dict) else getattr(cur, part)
    last = parts[-1]
    if isinstance(cur, dict):
        cur[last] = value
    else:
        setattr(cur, last, value)


def apply_cfg_overrides(env_cfg: Any, cfg_overrides: dict[str, Any] | None = None) -> Any:
    """Apply flat dotted-path overrides onto ``env_cfg`` in place."""
    for path, value in (cfg_overrides or {}).items():
        _set_by_path(env_cfg, path, value)
    return env_cfg


def omegaconf_to_plain(value: Any) -> Any:
    """Convert OmegaConf containers to plain Python containers, recursively."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(value, (DictConfig, ListConfig)):
            value = OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass

    if isinstance(value, dict):
        return {k: omegaconf_to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [omegaconf_to_plain(v) for v in value]
    return value


def _scale_or_set(value: Any, scale: float | None, absolute: Any = None) -> Any:
    if absolute is not None:
        value = absolute
    if scale is None or scale == 1.0:
        return value
    if isinstance(value, dict):
        return {k: v * scale for k, v in value.items()}
    return value * scale


def _apply_robot_overrides(env_cfg: Any, robot: dict[str, Any] | None) -> None:
    robot = dict(robot or {})
    if not robot:
        return

    groups: dict[str, Any] = dict(robot.get("groups") or {})
    actuators = env_cfg.scene.robot.actuators
    actuator_items = actuators.items() if isinstance(actuators, dict) else vars(actuators).items()

    for name, actuator in actuator_items:
        if actuator is None:
            continue
        group_cfg = groups.get(name) or {}
        actuator.stiffness = _scale_or_set(actuator.stiffness, robot.get("stiffness_scale"), group_cfg.get("stiffness"))
        actuator.damping = _scale_or_set(actuator.damping, robot.get("damping_scale"), group_cfg.get("damping"))
        actuator.effort_limit_sim = _scale_or_set(
            actuator.effort_limit_sim, robot.get("effort_limit_scale"), group_cfg.get("effort_limit_sim")
        )
        actuator.armature = _scale_or_set(actuator.armature, robot.get("armature_scale"), group_cfg.get("armature"))
        actuator.velocity_limit_sim = _scale_or_set(
            actuator.velocity_limit_sim, robot.get("velocity_limit_scale"), group_cfg.get("velocity_limit_sim")
        )

    if robot.get("init_pos") is not None:
        env_cfg.scene.robot.init_state.pos = tuple(robot["init_pos"])
    if robot.get("soft_joint_pos_limit_factor") is not None:
        env_cfg.scene.robot.soft_joint_pos_limit_factor = robot["soft_joint_pos_limit_factor"]
    if robot.get("action_scale_scale") is not None:
        env_cfg.actions.joint_pos.scale = _scale_or_set(
            env_cfg.actions.joint_pos.scale,
            robot["action_scale_scale"],
        )


def _apply_reward_overrides(env_cfg: Any, reward: dict[str, Any] | None) -> None:
    if not reward:
        return
    for term_name, term_cfg in reward.items():
        term_cfg = term_cfg or {}
        if term_cfg.get("enabled") is False:
            setattr(env_cfg.rewards, term_name, None)
            continue
        term = getattr(env_cfg.rewards, term_name)
        if term is None:
            continue
        if term_cfg.get("weight") is not None:
            term.weight = term_cfg["weight"]
        for key, value in term_cfg.items():
            if key not in ("enabled", "weight") and value is not None:
                term.params[key] = value


def _apply_observation_overrides(env_cfg: Any, observation: dict[str, Any] | None) -> None:
    if not observation:
        return
    for group_name, group_cfg in observation.items():
        if not group_cfg:
            continue
        group = getattr(env_cfg.observations, group_name)
        for term_name, term_cfg in group_cfg.items():
            term_cfg = term_cfg or {}
            if term_cfg.get("enabled") is False:
                setattr(group, term_name, None)
                continue
            term = getattr(group, term_name)
            noise_scale = term_cfg.get("noise_scale")
            if (
                term is not None
                and noise_scale is not None
                and noise_scale != 1.0
                and getattr(term, "noise", None) is not None
            ):
                term.noise.n_min = term.noise.n_min * noise_scale
                term.noise.n_max = term.noise.n_max * noise_scale


def _apply_termination_overrides(env_cfg: Any, termination: dict[str, Any] | None) -> None:
    if not termination:
        return
    for term_name, term_cfg in termination.items():
        term_cfg = term_cfg or {}
        if term_cfg.get("enabled") is False:
            setattr(env_cfg.terminations, term_name, None)
            continue
        term = getattr(env_cfg.terminations, term_name)
        if term is None:
            continue
        for key, value in term_cfg.items():
            if key != "enabled":
                term.params[key] = value


def _apply_motion_overrides(env_cfg: Any, motion: dict[str, Any] | None) -> None:
    if not motion:
        return
    cmd = env_cfg.commands.motion

    motion_files = motion.get("motion_files")
    if motion_files is not None:
        cmd.motion_files = [motion_files] if isinstance(motion_files, str) else list(motion_files)
    elif motion.get("motion_file") is not None:
        cmd.motion_files = [motion["motion_file"]]

    if motion.get("balance_mode") is not None:
        cmd.balance_mode = motion["balance_mode"]

    aliases = {
        "kernel_size": "adaptive_kernel_size",
        "lambda": "adaptive_lambda",
        "uniform_ratio": "adaptive_uniform_ratio",
        "alpha": "adaptive_alpha",
    }
    for key, value in (motion.get("adaptive_sampling") or {}).items():
        if value is None:
            continue
        attr = aliases.get(key, key)
        if hasattr(cmd, attr):
            setattr(cmd, attr, value)


def apply_tracking_overrides(
    env_cfg: Any,
    reward: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
    termination: dict[str, Any] | None = None,
    robot: dict[str, Any] | None = None,
    motion: dict[str, Any] | None = None,
    cfg_overrides: dict[str, Any] | None = None,
) -> Any:
    """Apply friendly tracking config blocks, then dot-path overrides last."""
    _apply_reward_overrides(env_cfg, reward)
    _apply_observation_overrides(env_cfg, observation)
    _apply_termination_overrides(env_cfg, termination)
    _apply_robot_overrides(env_cfg, robot)
    _apply_motion_overrides(env_cfg, motion)
    return apply_cfg_overrides(env_cfg, cfg_overrides)
