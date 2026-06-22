from __future__ import annotations

from typing import Any, Union, cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from ..types import F32NDArray, NDArray

# NOTE: There is no way to get the action bounds from the env, so we hardcode them here following FastTD3
ACTION_BOUNDS = {
    "Isaac-Repose-Cube-Shadow-Direct-v0": 1.0,
    "Isaac-Repose-Cube-Allegro-Direct-v0": 1.0,
    "Isaac-Velocity-Flat-G1-v0": 1.0,
    "Isaac-Velocity-Rough-G1-v0": 1.0,
    "Isaac-Velocity-Flat-H1-v0": 1.0,
    "Isaac-Velocity-Rough-H1-v0": 1.0,
    "Isaac-Lift-Cube-Franka-v0": 3.0,
    "Isaac-Open-Drawer-Franka-v0": 3.0,
    "Isaac-Velocity-Flat-Anymal-C-v0": 1.0,
    "Isaac-Velocity-Rough-Anymal-C-v0": 1.0,
    "Isaac-Velocity-Flat-Anymal-D-v0": 1.0,
    "Isaac-Velocity-Rough-Anymal-D-v0": 1.0,
    # NOTE: tanh-squashed action in [-1, 1] is scaled by env_cfg.actions.joint_pos.scale = G1_ACTION_SCALE in the env.
    "Isaac-Tracking-Flat-G1-v0": 1.0,
    "Isaac-Tracking-Flat-G1-WoSE-v0": 1.0,
}


# ----------------------------------------------------------------------------------------------------------------------
# Config-override helpers (PURE: no isaaclab import). These operate on a duck-typed env_cfg so they stay unit-testable
# without IsaacLab installed. See OVERRIDE CONTRACT in the task spec.
# ----------------------------------------------------------------------------------------------------------------------
def _set_by_path(obj: Any, dotted: str, value: Any) -> None:
    """Set a (possibly nested) member on ``obj`` addressed by a dotted path.

    Walks the path component by component, descending through dict members (``cur[part]``) or object attributes
    (``getattr``), and assigns ``value`` to the final component (via ``cur[last]`` for dicts, ``setattr`` otherwise).

    Args:
        obj: The root object (an env_cfg-like structure of nested dataclasses / dicts).
        dotted: A dotted path such as ``"scene.robot.actuators.legs.stiffness"``.
        value: The value to assign at the final path component.
    """
    parts = dotted.split(".")
    cur = obj
    for part in parts[:-1]:
        if isinstance(cur, dict):
            cur = cur[part]
        else:
            cur = getattr(cur, part)
    last = parts[-1]
    if isinstance(cur, dict):
        cur[last] = value
    else:
        setattr(cur, last, value)


def apply_cfg_overrides(env_cfg: Any, cfg_overrides: dict[str, Any] | None = None) -> Any:
    """Apply a flat mapping of dotted-path overrides onto ``env_cfg``.

    Args:
        env_cfg: The env_cfg-like object to mutate in place.
        cfg_overrides: Mapping of dotted path -> value (e.g. ``{"sim.dt": 0.004}``). ``None`` is a no-op.

    Returns:
        The same ``env_cfg`` object (mutated in place), for chaining.
    """
    for path, val in (cfg_overrides or {}).items():
        _set_by_path(env_cfg, path, val)
    return env_cfg


def _omegaconf_to_plain(x: Any) -> Any:
    """Recursively convert OmegaConf ``DictConfig``/``ListConfig`` to plain ``dict``/``list``.

    Plain Python objects pass through unchanged. The OmegaConf import is guarded so this helper works even when
    OmegaConf is not installed.

    Args:
        x: An arbitrary object, possibly an OmegaConf container.

    Returns:
        A plain ``dict``/``list``/scalar mirror of ``x``.
    """
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(x, (DictConfig, ListConfig)):
            x = OmegaConf.to_container(x, resolve=True)
    except ImportError:
        pass
    if isinstance(x, dict):
        return {k: _omegaconf_to_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_omegaconf_to_plain(v) for v in x]
    return x


def _has_observation_group(obs_space: gym.spaces.Space[Any], group_name: str) -> bool:
    """Return whether an IsaacLab Dict observation space contains ``group_name``."""
    return isinstance(obs_space, gym.spaces.Dict) and group_name in obs_space.spaces


def _concat_flat_space_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    """Return the flat shape produced by concatenating observation groups on the last dimension."""
    return (sum(shape[-1] for shape in shapes),)


def _scale_or_set(value: Any, scale: float | None, absolute: Any = None) -> Any:
    """Scale a dict-or-scalar actuator field, or override it with an absolute value.

    Args:
        value: The current field value (a scalar, or a dict of name->scalar).
        scale: A multiplicative factor applied to ``value`` when not ``None``.
        absolute: If not ``None``, used directly (replacing ``value``) before any scaling.

    Returns:
        The new field value (same shape as ``value`` / ``absolute``).
    """
    if absolute is not None:
        value = absolute
    if scale is not None and scale != 1.0:
        if isinstance(value, dict):
            value = {k: v * scale for k, v in value.items()}
        else:
            value = value * scale
    return value


def _apply_robot_overrides(
    env_cfg: Any,
    robot: dict[str, Any] | None,
) -> None:
    """Apply robot / actuator cfg knobs onto ``env_cfg.scene.robot`` and ``env_cfg.actions``.

    Covers global scales (stiffness/damping/effort/armature/velocity), per-actuator-group absolute
    overrides, init pose, soft joint-limit factor, and action-scale scaling.

    Args:
        env_cfg: The env_cfg-like object to mutate in place.
        robot: Robot/actuator override mapping (see OVERRIDE CONTRACT). ``None`` / empty is a no-op.
    """
    robot = dict(robot or {})
    if not robot:
        return

    stiffness_scale = robot.get("stiffness_scale")
    damping_scale = robot.get("damping_scale")
    effort_limit_scale = robot.get("effort_limit_scale")
    armature_scale = robot.get("armature_scale")
    velocity_limit_scale = robot.get("velocity_limit_scale")
    groups: dict[str, Any] = dict(robot.get("groups") or {})

    actuators = env_cfg.scene.robot.actuators
    actuator_items = actuators.items() if isinstance(actuators, dict) else vars(actuators).items()
    for name, actuator in actuator_items:
        if actuator is None:
            continue
        grp_cfg = groups.get(name) or {}
        actuator.stiffness = _scale_or_set(actuator.stiffness, stiffness_scale, grp_cfg.get("stiffness"))
        actuator.damping = _scale_or_set(actuator.damping, damping_scale, grp_cfg.get("damping"))
        actuator.effort_limit_sim = _scale_or_set(
            actuator.effort_limit_sim, effort_limit_scale, grp_cfg.get("effort_limit_sim")
        )
        actuator.armature = _scale_or_set(actuator.armature, armature_scale, grp_cfg.get("armature"))
        actuator.velocity_limit_sim = _scale_or_set(
            actuator.velocity_limit_sim, velocity_limit_scale, grp_cfg.get("velocity_limit_sim")
        )

    init_pos = robot.get("init_pos")
    if init_pos is not None:
        env_cfg.scene.robot.init_state.pos = tuple(init_pos)

    soft_factor = robot.get("soft_joint_pos_limit_factor")
    if soft_factor is not None:
        env_cfg.scene.robot.soft_joint_pos_limit_factor = soft_factor

    action_scale_scale = robot.get("action_scale_scale")
    if action_scale_scale is not None and action_scale_scale != 1.0:
        # joint_pos.scale is a per-joint dict (G1_ACTION_SCALE), so use the dict-aware helper
        # (`dict * float` raises TypeError).
        env_cfg.actions.joint_pos.scale = _scale_or_set(env_cfg.actions.joint_pos.scale, action_scale_scale)


def _apply_reward_overrides(env_cfg: Any, reward: dict[str, Any] | None) -> None:
    """Apply flat (single-critic) reward overrides onto the ``env_cfg.rewards`` RewTerm attrs.

    The ``reward`` mapping is ``{term_name: {weight?, std?, enabled?}}``. ``enabled: false`` disables the term by
    setting the attribute to ``None``; ``weight``/``std`` update the RewTerm in place. All terms are summed by
    IsaacLab's ``RewardManager`` into a single scalar reward.

    Args:
        env_cfg: The env_cfg-like object to mutate in place.
        reward: Flat reward override mapping. ``None`` is a no-op.
    """
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
        if term_cfg.get("std") is not None:
            term.params["std"] = term_cfg["std"]


def _apply_observation_overrides(env_cfg: Any, observation: dict[str, Any] | None) -> None:
    """Apply observation enable/disable + noise-scaling onto ``env_cfg.observations``.

    The ``observation`` mapping is ``{group(policy|critic): {term: {enabled?, noise_scale?}}}``. ``enabled: false``
    disables the term (set to ``None``). ``noise_scale`` multiplies ``term.noise.n_min``/``n_max`` when the term has a
    noise model.

    Args:
        env_cfg: The env_cfg-like object to mutate in place.
        observation: Observation override mapping. ``None`` is a no-op.
    """
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
            if term is None:
                continue
            noise_scale = term_cfg.get("noise_scale")
            if noise_scale is not None and noise_scale != 1.0 and getattr(term, "noise", None) is not None:
                term.noise.n_min = term.noise.n_min * noise_scale
                term.noise.n_max = term.noise.n_max * noise_scale


def _apply_termination_overrides(env_cfg: Any, termination: dict[str, Any] | None) -> None:
    """Apply termination thresholds / disable onto ``env_cfg.terminations``.

    The ``termination`` mapping is ``{term: {enabled?, <param>: value}}``. ``enabled: false`` disables the term (set to
    ``None``); any other key/value pair is written into ``term.params``.

    Args:
        env_cfg: The env_cfg-like object to mutate in place.
        termination: Termination override mapping. ``None`` is a no-op.
    """
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
            if key == "enabled":
                continue
            term.params[key] = value


def _apply_motion_overrides(env_cfg: Any, motion: dict[str, Any] | None) -> None:
    """Apply motion-loader / adaptive-sampling overrides onto ``env_cfg.commands.motion``.

    The ``motion`` mapping accepts ``{motion_files?|motion_file?, balance_mode?, adaptive_sampling?: {...}}``. A single
    ``motion_file`` string is normalized to a one-element ``motion_files`` list. The friendly ``adaptive_sampling``
    block is mapped onto the real WBT cfg fields via the aliases ``kernel_size``->``adaptive_kernel_size``,
    ``lambda``->``adaptive_lambda``, ``uniform_ratio``->``adaptive_uniform_ratio``, ``alpha``->``adaptive_alpha``.
    Power users may instead set the exact ``adaptive_*`` field names directly. Any key that does not resolve to an
    existing cfg attribute (after aliasing) is ignored.

    Args:
        env_cfg: The env_cfg-like object to mutate in place.
        motion: Motion override mapping. ``None`` is a no-op.
    """
    if not motion:
        return
    cmd = env_cfg.commands.motion

    motion_files = motion.get("motion_files")
    motion_file = motion.get("motion_file")
    if motion_files is not None:
        if isinstance(motion_files, str):
            motion_files = [motion_files]
        cmd.motion_files = list(motion_files)
    elif motion_file is not None:
        cmd.motion_files = [motion_file]

    if motion.get("balance_mode") is not None:
        cmd.balance_mode = motion["balance_mode"]

    adaptive = motion.get("adaptive_sampling")
    if adaptive:
        # Friendly schema keys -> real WBT cfg field names. Unknown keys are forwarded verbatim so power users can set
        # the exact adaptive_* fields directly.
        alias = {
            "kernel_size": "adaptive_kernel_size",
            "lambda": "adaptive_lambda",
            "uniform_ratio": "adaptive_uniform_ratio",
            "alpha": "adaptive_alpha",
        }
        for key, value in adaptive.items():
            if value is None:
                continue
            attr = alias.get(key, key)
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
    """Apply the full friendly + dot-path override stack onto a tracking ``env_cfg``.

    Order matters: the friendly grouped blocks are applied first, then the general ``cfg_overrides`` dot-path map LAST
    so that explicit dot-paths always win. All inputs are plain Python (convert OmegaConf with ``_omegaconf_to_plain``
    before calling). This function is PURE (no isaaclab import) and mutates ``env_cfg`` in place.

    Args:
        env_cfg: The tracking env_cfg-like object to mutate in place.
        reward: Flat reward overrides ``{term: {weight?, std?, enabled?}}`` (single-critic, summed).
        observation: Observation overrides ``{policy|critic: {term: {enabled?, noise_scale?}}}``.
        termination: Termination overrides ``{term: {enabled?, <param>: value}}``.
        robot: Robot / actuator cfg overrides (scales + absolutes + per-group knobs).
        motion: Motion-loader overrides ``{motion_files?|motion_file?, balance_mode?, adaptive_sampling?}``.
        cfg_overrides: General dot-path escape hatch applied LAST.

    Returns:
        The same ``env_cfg`` object (mutated in place), for chaining.
    """
    _apply_reward_overrides(env_cfg, reward)
    _apply_observation_overrides(env_cfg, observation)
    _apply_termination_overrides(env_cfg, termination)
    _apply_robot_overrides(env_cfg, robot)
    _apply_motion_overrides(env_cfg, motion)
    # FINALLY: dot-path overrides win.
    apply_cfg_overrides(env_cfg, cfg_overrides)
    return env_cfg


def compute_joint_limit_action_bound(
    soft_limits: torch.Tensor,
    default_pos: torch.Tensor,
    action_scale: torch.Tensor,
    fraction: float = 1.0,
    mode: str = "asymmetric",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-joint affine (bias, range) mapping a tanh action in [-1, 1] to a joint-position window.

    A ``JointPositionAction`` computes ``target = default + scale * action``. We pick the per-joint
    action window so that ``action in [-1, 1]`` keeps ``target`` within ``fraction * [lower, upper]``
    (the soft joint position limits). The env applies ``final = bias + range * clamp(action, -1, 1)``
    in ``step()``; the actor still emits plain ``[-1, 1]`` (buffer/agent untouched).

    Args:
        soft_limits: ``[J, 2]`` (lower, upper) soft joint position limits (already soft-scaled).
        default_pos: ``[J]`` default joint positions (the action offset).
        action_scale: ``[J]`` per-joint JointPositionAction scale.
        fraction: f in (0, 1]; scales how far toward each limit the extremes reach.
        mode: ``"asymmetric"`` (extremes hit lower/upper exactly) or ``"symmetric"`` (``bias=0``).

    Returns:
        ``(bias, range)``, each ``[J]``: the affine params for ``final = bias + range * action``.
    """
    lower = soft_limits[..., 0]
    upper = soft_limits[..., 1]
    eps = 1e-8
    zero = action_scale.abs() < eps
    safe_scale = torch.where(zero, torch.ones_like(action_scale), action_scale)
    # action values whose target hits the lower / upper limit (signed; works for negative scale too).
    a_lo = fraction * (lower - default_pos) / safe_scale
    a_hi = fraction * (upper - default_pos) / safe_scale
    if mode == "symmetric":
        rng = torch.maximum(a_hi.abs(), a_lo.abs())
        bias = torch.zeros_like(rng)
    else:  # asymmetric (default): ±1 maps exactly onto the (soft) limits
        bias = 0.5 * (a_hi + a_lo)
        rng = 0.5 * (a_hi - a_lo)
    # joints with ~zero scale have no actuation effect -> no action range.
    bias = torch.where(zero, torch.zeros_like(bias), bias)
    rng = torch.where(zero, torch.zeros_like(rng), rng)
    return bias, rng


def recursive_to_numpy(
    data: Union[torch.Tensor, dict[str, Any], list[Any], tuple[Any, ...], NDArray],
) -> Union[NDArray, dict[str, Any], list[Any], tuple[Any, ...]]:
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, dict):
        return {k: recursive_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(recursive_to_numpy(v) for v in data)
    else:
        return data


class IsaacLabVectorEnv(
    VectorEnv[Union[torch.Tensor, F32NDArray], Union[torch.Tensor, F32NDArray], Union[torch.Tensor, F32NDArray]]
):
    """
    Gymnasium "SyncVectorEnv" implementation for IsaacLab environments.

    As all jax-based env does, IsaacLab does not internally store the 'state' of the env.

    Args:
        env_name (str): The environment name registered in IsaacLab.
        num_envs (int): The number of parallel environments. This is only used if the env argument is a string
        device (str):
        seed (int):
        action_bounds (float):
        to_numpy (bool): If True, will convert all outputs from jnp.ndarray to np.array.
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        seed: int,
        device: str,
        action_bounds: float,
        to_numpy: bool = True,
        headless: bool = True,
        enable_cameras: bool = False,
        reward: dict[str, Any] | None = None,
        observation: dict[str, Any] | None = None,
        termination: dict[str, Any] | None = None,
        robot: dict[str, Any] | None = None,
        motion: dict[str, Any] | None = None,
        cfg_overrides: dict[str, Any] | None = None,
        action_bound: dict[str, Any] | None = None,
    ):
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(headless=headless, device=device, enable_cameras=enable_cameras or not headless)
        self.simulation_app = app_launcher.app

        is_tracking = env_name.startswith("Isaac-Tracking")
        if is_tracking:
            # NOTE: Registration must happen AFTER AppLauncher (so isaaclab is importable) and BEFORE parse_env_cfg
            # (which needs the task registered to look up its env_cfg_entry_point). The package __init__ stays
            # import-light; importing the g1 config module triggers the gym.register calls.
            import flash_rl.envs.isaaclab_envs.tracking.config.g1  # noqa: F401

        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        env_cfg = parse_env_cfg(
            env_name,
            device=device,
            num_envs=num_envs,
        )
        env_cfg.seed = seed
        self.seed = seed
        self.device = device

        if is_tracking:
            # Convert any OmegaConf containers to plain Python, then apply the full override stack before gym.make.
            apply_tracking_overrides(
                env_cfg,
                reward=_omegaconf_to_plain(reward),
                observation=_omegaconf_to_plain(observation),
                termination=_omegaconf_to_plain(termination),
                robot=_omegaconf_to_plain(robot),
                motion=_omegaconf_to_plain(motion),
                cfg_overrides=_omegaconf_to_plain(cfg_overrides),
            )

        self.envs = gym.make(env_name, cfg=env_cfg, render_mode=None)
        setattr(cast(Any, self.envs.unwrapped), "is_evaluating", False)

        # True terminal-observation capture for off-policy value bootstrapping.
        # IsaacLab auto-resets done envs *inside* step() and recomputes observations, so the obs it
        # returns for a done env is the POST-reset frame. For timeout (truncation) transitions the
        # critic must bootstrap on the genuine terminal obs, not a different episode's reset frame.
        # We hook the underlying env's `_reset_idx` to snapshot observations while the sim still holds
        # the terminal state (before scene/command resets run), and surface it as infos["final_obs"].
        self._final_obs_buf: dict[str, torch.Tensor] | None = None
        _base_env = cast(Any, self.envs.unwrapped)
        if hasattr(_base_env, "_reset_idx") and hasattr(_base_env, "observation_manager"):
            self._install_terminal_obs_capture(_base_env)

        self.num_envs = cast(Any, self.envs.unwrapped).num_envs
        self.max_episode_steps = cast(Any, self.envs.unwrapped).max_episode_length
        self.to_numpy = to_numpy

        # Get observation/action spaces
        # NOTE: Action range: [-1, 1] * action_bounds (https://github.com/google-deepmind/mujoco_playground/issues/19)
        obs_space = cast(Any, self.envs.unwrapped).single_observation_space
        self.obs_size = obs_space["policy"].shape
        self.asymmetric_obs = _has_observation_group(obs_space, "critic")
        if self.asymmetric_obs:
            # NOTE: Env will treat concatenate actor & critic states as the observation,
            # but will give 'actual' observation size in the info.
            self.critic_obs_size = obs_space["critic"].shape
            # NOTE: setting to [0, 0] since we only need the shape and dtype
            self.single_observation_space = gym.spaces.Box(
                low=0.0, high=0.0, shape=_concat_flat_space_shape(self.obs_size, self.critic_obs_size), dtype=np.float32
            )
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        else:
            self.critic_obs_size = 0
            self.single_observation_space = gym.spaces.Box(low=0.0, high=0.0, shape=self.obs_size, dtype=np.float32)
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)

        self.action_size = cast(Any, self.envs.unwrapped).single_action_space.shape
        # Action-bound policy: scalar (default, back-compat) OR per-joint joint-limit affine.
        # In joint-limit mode the actor still emits tanh in [-1, 1] (so single_action_space stays
        # [-1, 1] and the replay buffer / random-init are unchanged); the per-joint window is applied
        # in step() via final = bias + range * clamp(action, -1, 1).
        self._action_bias: torch.Tensor | None = None
        self._action_range: torch.Tensor | None = None
        action_bound = _omegaconf_to_plain(action_bound)
        if isinstance(action_bound, dict) and action_bound.get("type") == "joint_limit":
            self.action_bounds = None
            self._action_bias, self._action_range = self._build_joint_limit_bound(
                fraction=float(action_bound.get("fraction", 1.0)),
                mode=str(action_bound.get("mode", "asymmetric")),
            )
            self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=self.action_size, dtype=np.float32)
        else:
            self.action_bounds = action_bounds
            self.single_action_space = gym.spaces.Box(
                low=-1.0 * self.action_bounds, high=1.0 * self.action_bounds, shape=self.action_size, dtype=np.float32
            )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def _install_terminal_obs_capture(self, base_env: Any) -> None:
        """Hook ``base_env._reset_idx`` to snapshot the true terminal observation before auto-reset.

        IsaacLab resets done envs inside ``step()`` and recomputes observations afterwards, so the
        returned obs for a done env is already the post-reset frame. ``_reset_idx`` is the last point
        where the sim still holds the terminal state (it runs scene/event/command resets internally),
        so we compute observations here and cache the rows for the resetting envs into
        ``self._final_obs_buf``. ``step()`` then exposes this as the (genuine) ``infos["final_obs"]``.
        """
        orig_reset_idx = base_env._reset_idx

        def patched_reset_idx(env_ids: Any, *args: Any, **kwargs: Any) -> Any:
            terminal = base_env.observation_manager.compute()
            if self._final_obs_buf is None:
                self._final_obs_buf = {k: torch.zeros_like(v) for k, v in terminal.items()}
            for k, v in terminal.items():
                self._final_obs_buf[k][env_ids] = v[env_ids]
            return orig_reset_idx(env_ids, *args, **kwargs)

        base_env._reset_idx = patched_reset_idx

    def _build_joint_limit_bound(self, fraction: float, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the per-joint (bias, range) affine from the live robot joint limits + action scale.

        Reads the soft joint position limits and default joint positions from the robot, and the
        per-joint scale from the JointPositionAction term (aligned to the action's joint order), then
        delegates to :func:`compute_joint_limit_action_bound`.
        """
        env = cast(Any, self.envs.unwrapped)
        robot = env.scene["robot"]
        term = env.action_manager.get_term("joint_pos")
        soft = robot.data.soft_joint_pos_limits[0]  # [J_robot, 2]
        default = robot.data.default_joint_pos[0]  # [J_robot]
        joint_ids = getattr(term, "_joint_ids", slice(None))
        if not isinstance(joint_ids, slice):
            idx = torch.as_tensor(joint_ids, device=soft.device, dtype=torch.long)
            soft = soft[idx]
            default = default[idx]
        scale = torch.as_tensor(term._scale, device=soft.device, dtype=torch.float32)
        if scale.ndim > 1:
            scale = scale[0]
        scale = scale.reshape(-1)
        if scale.numel() == 1:
            scale = scale.expand(soft.shape[0])
        bias, rng = compute_joint_limit_action_bound(soft, default, scale, fraction=fraction, mode=mode)
        return bias.to(self.device).float(), rng.to(self.device).float()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        random_start_init: bool = True,
    ) -> tuple[Union[torch.Tensor, F32NDArray], dict[str, Any]]:
        obs_dict, infos = self.envs.reset()
        obs = obs_dict["policy"]
        if self.asymmetric_obs:
            critic_obs = obs_dict["critic"]
            obs = torch.cat((obs, critic_obs), dim=-1)
        else:
            critic_obs = None
        # NOTE: decorrelate episode horizons like RSL‑RL
        # In IsaacLab, `dones` is computed as follows:
        # `time_out = self.episode_length_buf >= self.max_episode_length - 1`
        # While training, this code spreads out the resets to avoid spikes
        # when many environments reset at a similar time.
        if random_start_init:
            # step in current episode (per env)
            cast(Any, self.envs.unwrapped).episode_length_buf = torch.randint_like(
                cast(Any, self.envs.unwrapped).episode_length_buf, high=int(self.max_episode_steps)
            )
        if self.to_numpy:
            obs = obs.cpu().numpy()
            infos = recursive_to_numpy(infos)  # type: ignore
        infos.update({"actor_observation_size": self.obs_size, "asymmetric_obs": self.asymmetric_obs})
        return obs, infos

    def step(self, actions: Union[torch.Tensor, F32NDArray]) -> tuple[
        Union[torch.Tensor, F32NDArray],
        Union[torch.Tensor, F32NDArray],
        Union[torch.Tensor, F32NDArray],
        Union[torch.Tensor, F32NDArray],
        dict[str, Any],
    ]:
        if isinstance(actions, torch.Tensor):
            torch_actions = actions.to(self.device)
        else:
            torch_actions = torch.from_numpy(actions).to(self.device)

        if self._action_bias is not None:
            # Per-joint joint-limit affine: map tanh action in [-1, 1] onto the per-joint window.
            torch_actions = self._action_bias + self._action_range * torch.clamp(torch_actions, -1.0, 1.0)
        elif self.action_bounds is not None:
            torch_actions = torch.clamp(torch_actions, -1.0, 1.0) * self.action_bounds
        obs_dict, rew, terminations, truncations, raw_infos = cast(Any, self.envs.step(torch_actions))
        obs = obs_dict["policy"]
        if self.asymmetric_obs:
            critic_obs = obs_dict["critic"]
            obs = torch.cat((obs, critic_obs), dim=-1)
        else:
            critic_obs = None
        infos = {"time_outs": truncations, "observations": {"critic": critic_obs}}
        # True terminal observation (captured pre-reset in _install_terminal_obs_capture). Rows for
        # envs that reset this step are fresh; other rows are stale but never read (train.py consumes
        # final_obs only for done envs). Falls back to post-reset obs before the first reset occurs.
        # See https://github.com/isaac-sim/IsaacLab/issues/1362
        if self._final_obs_buf is not None:
            final_obs = self._final_obs_buf["policy"]
            if self.asymmetric_obs:
                final_obs = torch.cat((final_obs, self._final_obs_buf["critic"]), dim=-1)
            infos["final_obs"] = final_obs
        else:
            infos["final_obs"] = obs

        # Surface IsaacLab's per-term episodic logs so each reward is logged separately.
        # IsaacLab's managers populate raw_infos["log"] on reset with keys like
        # "Episode_Reward/<term>" (episodic sum / max_episode_length_s, averaged over the envs that
        # reset this step), "Episode_Termination/<term>", and command "Metrics/...". We remap reward
        # terms -> "rewards/<term>" and termination terms -> "terminations/<term>" (others pass
        # through) and only emit on steps where an env actually reset (so values are fresh).
        log = raw_infos.get("log") if isinstance(raw_infos, dict) else None
        if log and bool(terminations.any()) | bool(truncations.any()):
            episode_info: dict[str, float] = {}
            for key, value in log.items():
                scalar = float(value.item()) if hasattr(value, "item") else float(value)
                if key.startswith("Episode_Reward/"):
                    episode_info["rewards/" + key.split("/", 1)[1]] = scalar
                elif key.startswith("Episode_Termination/"):
                    episode_info["terminations/" + key.split("/", 1)[1]] = scalar
                elif key.startswith("Metrics/"):
                    episode_info["metrics/" + key.split("/", 1)[1]] = scalar
                else:
                    episode_info[key] = scalar
            if episode_info:
                infos["episode_info"] = episode_info

        if self.to_numpy:
            obs = obs.cpu().numpy()
            rew = rew.cpu().numpy()
            terminations = terminations.cpu().numpy()
            truncations = truncations.cpu().numpy()
            infos = recursive_to_numpy(infos)
        return obs, rew, terminations, truncations, infos

    def close(self, **kwargs: Any) -> None:
        # self.envs.close(**kwargs)
        # self.simulation_app.close()
        return

    def render(self) -> None:
        raise NotImplementedError("We don't support rendering for IsaacLab environments")


def make_isaaclab_env(
    env_name: str,
    num_envs: int,
    seed: int,
    headless: bool = True,
    enable_cameras: bool = False,
    reward: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
    termination: dict[str, Any] | None = None,
    robot: dict[str, Any] | None = None,
    motion: dict[str, Any] | None = None,
    cfg_overrides: dict[str, Any] | None = None,
    action_bound: dict[str, Any] | None = None,
) -> IsaacLabVectorEnv:
    if env_name not in ACTION_BOUNDS:
        print(f"Action bounds not defined for {env_name}; using default value 1.0.")
    action_bounds = ACTION_BOUNDS.get(env_name, 1.0)
    env = IsaacLabVectorEnv(
        env_name=env_name,
        num_envs=num_envs,
        seed=seed,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        action_bounds=action_bounds,
        to_numpy=True,
        headless=headless,
        enable_cameras=enable_cameras,
        reward=reward,
        observation=observation,
        termination=termination,
        robot=robot,
        motion=motion,
        cfg_overrides=cfg_overrides,
        action_bound=action_bound,
    )
    return env
