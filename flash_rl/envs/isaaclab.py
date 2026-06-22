from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Union, cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from ..types import F32NDArray, NDArray
from .isaaclab_envs.tracking.overrides import apply_tracking_overrides, omegaconf_to_plain
from .isaaclab_envs.utils.action_bounds import compute_joint_limit_action_bound

# IsaacLab does not expose these bounds directly.
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
    "Isaac-Tracking-Flat-G1-v0": 1.0,
    "Isaac-Tracking-Flat-G1-WoSE-v0": 1.0,
}


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
            # Register local tracking tasks after AppLauncher initializes IsaacLab.
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
            apply_tracking_overrides(
                env_cfg,
                reward=omegaconf_to_plain(reward),
                observation=omegaconf_to_plain(observation),
                termination=omegaconf_to_plain(termination),
                robot=omegaconf_to_plain(robot),
                motion=omegaconf_to_plain(motion),
                cfg_overrides=omegaconf_to_plain(cfg_overrides),
            )

        self.envs = gym.make(env_name, cfg=env_cfg, render_mode=None)
        setattr(cast(Any, self.envs.unwrapped), "is_evaluating", False)

        # Capture terminal observations before IsaacLab's same-step autoreset.
        self._final_obs_buf: dict[str, torch.Tensor] | None = None
        _base_env = cast(Any, self.envs.unwrapped)
        if hasattr(_base_env, "_reset_idx") and hasattr(_base_env, "observation_manager"):
            self._install_terminal_obs_capture(_base_env)

        self.num_envs = cast(Any, self.envs.unwrapped).num_envs
        self.max_episode_steps = cast(Any, self.envs.unwrapped).max_episode_length
        self.to_numpy = to_numpy

        obs_space = cast(Any, self.envs.unwrapped).single_observation_space
        self.obs_size = obs_space["policy"].shape
        self.asymmetric_obs = isinstance(obs_space, gym.spaces.Dict) and "critic" in obs_space.spaces
        if self.asymmetric_obs:
            self.critic_obs_size = obs_space["critic"].shape
            self.single_observation_space = gym.spaces.Box(
                low=0.0, high=0.0, shape=(self.obs_size[-1] + self.critic_obs_size[-1],), dtype=np.float32
            )
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        else:
            self.critic_obs_size = 0
            self.single_observation_space = gym.spaces.Box(low=0.0, high=0.0, shape=self.obs_size, dtype=np.float32)
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)

        self.action_size = cast(Any, self.envs.unwrapped).single_action_space.shape
        self._action_bias: torch.Tensor | None = None
        self._action_range: torch.Tensor | None = None
        action_bound = omegaconf_to_plain(action_bound)
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

    @contextmanager
    def evaluation_mode(self) -> Iterator[None]:
        """Temporarily tell IsaacLab command terms to use deterministic evaluation behavior."""
        base_env = cast(Any, self.envs.unwrapped)
        previous = getattr(base_env, "is_evaluating", False)
        setattr(base_env, "is_evaluating", True)
        try:
            yield
        finally:
            setattr(base_env, "is_evaluating", previous)

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
        if random_start_init:
            cast(Any, self.envs.unwrapped).episode_length_buf = torch.randint_like(
                cast(Any, self.envs.unwrapped).episode_length_buf, high=int(self.max_episode_steps)
            )
        if self.to_numpy:
            obs = obs.cpu().numpy()
            infos = recursive_to_numpy(infos)  # type: ignore
        infos.update({"actor_observation_size": self.obs_size, "asymmetric_obs": self.asymmetric_obs})
        return obs, infos

    def step(
        self, actions: Union[torch.Tensor, F32NDArray]
    ) -> tuple[
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
        if self._final_obs_buf is not None:
            final_obs = self._final_obs_buf["policy"]
            if self.asymmetric_obs:
                final_obs = torch.cat((final_obs, self._final_obs_buf["critic"]), dim=-1)
            infos["final_obs"] = final_obs
        else:
            infos["final_obs"] = obs

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
