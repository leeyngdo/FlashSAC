from __future__ import annotations

from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from ..types import F32NDArray, NDArray

# Local (non-registry) mjlab tasks whose cfg + MDP terms live in the
# flash_rl.envs.mjlab_envs content package. Mirrors
# flash_rl.envs.isaaclab.LOCAL_ISAACLAB_TASKS: the wrapper module here
# orchestrates, the env package provides the content (cfg builder + overrides).
LOCAL_MJLAB_TASKS: dict[str, str] = {
    "DexManip-MotionTracking-XHand-Right": "flash_rl.envs.mjlab_envs.dexmanip",
}


class MjlabVectorEnv(VectorEnv[F32NDArray, F32NDArray, F32NDArray]):
    """Gymnasium VectorEnv wrapping mjlab's ManagerBasedRlEnv for FlashSAC.

    Uses auto_reset=False so we can capture the true terminal observation before
    resetting. This populates infos["final_obs"] correctly for off-policy TD
    bootstrapping on truncated episodes — fixing the known limitation in the
    IsaacLab wrapper where terminal obs is unavailable.

    Observations are flattened from mjlab's dict format:
    - If both "actor" and "critic" groups exist: critic obs is stored directly.
    - Otherwise: the actor group is used as-is.

    Actions are passed through unchanged (mjlab action terms handle scaling internally).
    """

    def __init__(
        self,
        task_id: str,
        num_envs: int,
        seed: int,
        device: str = "cuda:0",
        to_numpy: bool = True,
    ) -> None:
        import mjlab.tasks  # noqa: F401  # populates the task registry via side effects
        from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg

        env_cfg = load_env_cfg(task_id)
        env_cfg.scene.num_envs = num_envs
        env_cfg.seed = seed
        env_cfg.auto_reset = False  # we handle resets to preserve the terminal obs

        self._env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
        self._device = device
        self._to_numpy = to_numpy
        self.num_envs = num_envs

        # Determine obs layout
        obs_space = self._env.single_observation_space
        obs_groups = list(obs_space.spaces.keys())
        if "actor" not in obs_groups:
            raise ValueError(f"mjlab env must expose an 'actor' observation group, got {obs_groups}.")
        self._actor_obs_key = "actor"
        self._critic_obs_key = "critic" if "critic" in obs_groups else None
        self._has_critic_obs = self._critic_obs_key is not None
        self._actor_obs_dim = int(obs_space.spaces[self._actor_obs_key].shape[0])
        flat_dim = int(obs_space.spaces[self._critic_obs_key].shape[0]) if self._has_critic_obs else self._actor_obs_dim

        action_dim = int(self._env.single_action_space.shape[0])

        self.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, num_envs)

        # Expose for FlashSAC agent/env setup (mirrors IsaacLabVectorEnv)
        self.obs_size = (flat_dim,)
        self.action_size = (action_dim,)

        # Episode return/length tracking for training-time logging
        self._ep_returns = np.zeros(num_envs, dtype=np.float32)
        self._ep_lengths = np.zeros(num_envs, dtype=np.int32)
        self._dexmanip_eval_mode = False
        self._dexmanip_eval_state: dict[str, Any] | None = None
        self._dexmanip_tracking_perf: Any | None = None

    def _get_motion_command(self) -> Any | None:
        command_manager = getattr(getattr(self, "_env", None), "command_manager", None)
        if command_manager is None:
            return None
        try:
            return command_manager.get_term("motion")
        except Exception:
            return None

    def _info_value_to_numpy(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, dict):
            return {k: self._info_value_to_numpy(v) for k, v in value.items()}
        return value

    def _extract_step_infos(self, extras: dict[str, Any]) -> dict[str, Any]:
        infos: dict[str, Any] = {}
        for key in (
            "success",
            "final_info",
            "success_rate_0.5",
            "success_rate_1.0",
            "success_rate_2.0",
        ):
            if key in extras:
                infos[key] = self._info_value_to_numpy(extras[key])
        return infos

    def _set_eval_terminations(self, enabled: bool) -> None:
        term_manager = getattr(self._env, "termination_manager", None)
        if term_manager is None:
            return

        if enabled:
            if self._dexmanip_eval_state is None:
                self._dexmanip_eval_state = {}
            if "termination_manager" in self._dexmanip_eval_state:
                return

            names = list(getattr(term_manager, "_term_names", []))
            cfgs = list(getattr(term_manager, "_term_cfgs", []))
            term_dones = dict(getattr(term_manager, "_term_dones", {}))
            self._dexmanip_eval_state["termination_manager"] = {
                "names": names,
                "cfgs": cfgs,
                "term_dones": term_dones,
            }

            keep = [i for i, cfg in enumerate(cfgs) if getattr(cfg, "time_out", False)]
            term_manager._term_names = [names[i] for i in keep]
            term_manager._term_cfgs = [cfgs[i] for i in keep]
            term_manager._term_dones = {names[i]: term_dones[names[i]] for i in keep}
            return

        state = self._dexmanip_eval_state or {}
        term_state = state.get("termination_manager")
        if term_state is None:
            return
        term_manager._term_names = term_state["names"]
        term_manager._term_cfgs = term_state["cfgs"]
        term_manager._term_dones = term_state["term_dones"]

    def _set_eval_object_pin(self, enabled: bool) -> None:
        command = self._get_motion_command()
        if command is None:
            return
        obj_cfg = getattr(command.cfg, "object", None)
        if obj_cfg is None:
            return

        if enabled:
            if self._dexmanip_eval_state is None:
                self._dexmanip_eval_state = {}
            if "object_pin" not in self._dexmanip_eval_state:
                self._dexmanip_eval_state["object_pin"] = {
                    "pin_objects": getattr(obj_cfg, "pin_objects", None),
                    "pin_mode": getattr(obj_cfg, "pin_mode", None),
                }
            if hasattr(obj_cfg, "pin_objects"):
                obj_cfg.pin_objects = False
            if hasattr(obj_cfg, "pin_mode"):
                obj_cfg.pin_mode = "none"
            return

        state = (self._dexmanip_eval_state or {}).get("object_pin")
        if state is None:
            return
        if state["pin_objects"] is not None and hasattr(obj_cfg, "pin_objects"):
            obj_cfg.pin_objects = state["pin_objects"]
        if state["pin_mode"] is not None and hasattr(obj_cfg, "pin_mode"):
            obj_cfg.pin_mode = state["pin_mode"]

    def _start_dexmanip_tracking_perf(self) -> None:
        if not self._dexmanip_eval_mode:
            self._dexmanip_tracking_perf = None
            return
        try:
            from callbacks.tracking_performance import TrackingPerformance
        except Exception:
            self._dexmanip_tracking_perf = None
            return

        perf = TrackingPerformance(
            grace_steps=15,
            threshold_ks=[0.5, 1.0, 2.0],
            thresholds={
                "obj_trans": 0.03,
                "tip_trans": 0.06,
                "obj_rot_deg": 30.0,
                "joint_trans": 0.08,
            },
            command_name="motion",
        )
        perf.on_start(self._env)
        self._dexmanip_tracking_perf = perf

    def _update_dexmanip_tracking_perf(self) -> None:
        perf = self._dexmanip_tracking_perf
        if perf is not None:
            perf.on_step(self._env)

    def _dexmanip_success_infos(self, dones: torch.Tensor) -> dict[str, Any]:
        perf = self._dexmanip_tracking_perf
        if perf is None:
            return {}
        try:
            state = perf.collect_state()
            count = state["count_per_env"]
            scored = count > 0
            fail = perf._per_env_fail(state, count.clamp(min=1.0))
            success_flags = (~fail) & scored[:, None]
        except Exception:
            return {}

        out: dict[str, Any] = {}
        terminal_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self._env.device)
        terminal_success[dones] = success_flags[dones, perf.k1_idx]
        out["success"] = terminal_success.detach().cpu().numpy()
        out["final_info"] = {"success": out["success"]}
        for i, k in enumerate(perf.threshold_ks):
            terminal_success_k = torch.zeros(self.num_envs, dtype=torch.float32, device=self._env.device)
            terminal_success_k[dones] = success_flags[dones, i].float()
            out[f"success_rate_{k:.1f}"] = terminal_success_k.detach().cpu().numpy()
        return out

    def _flatten_obs(self, obs_dict: dict[str, Any]) -> F32NDArray:
        obs_key = self._critic_obs_key if self._has_critic_obs else self._actor_obs_key
        assert obs_key is not None
        flat = obs_dict[obs_key]
        return flat.cpu().numpy().astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[F32NDArray, dict[str, Any]]:
        obs_dict, _ = self._env.reset()
        self._start_dexmanip_tracking_perf()
        self._ep_returns[:] = 0.0
        self._ep_lengths[:] = 0
        env_info: dict[str, Any] = {}
        if self._has_critic_obs:
            env_info["actor_observation_size"] = (self._actor_obs_dim,)
        return self._flatten_obs(obs_dict), env_info

    def step(
        self,
        actions: Union[F32NDArray, torch.Tensor],
    ) -> tuple[F32NDArray, F32NDArray, NDArray, NDArray, dict[str, Any]]:
        if isinstance(actions, np.ndarray):
            actions_t = torch.from_numpy(actions).float().to(self._device)
        else:
            actions_t = actions.to(self._device)

        obs_dict, rewards, terminateds, truncateds, extras = self._env.step(actions_t)
        dones = terminateds | truncateds
        self._update_dexmanip_tracking_perf()

        rewards_np = rewards.cpu().numpy().astype(np.float32)
        self._ep_returns += rewards_np
        self._ep_lengths += 1

        # Capture terminal obs BEFORE resetting done envs
        terminal_obs = self._flatten_obs(obs_dict)
        step_infos = self._extract_step_infos(extras)
        step_infos.update(self._dexmanip_success_infos(dones))

        # Reset done envs; mjlab raises RuntimeError on the next step() if we skip this.
        # reset() recomputes obs for ALL envs: done envs get fresh state, non-done envs
        # are unchanged — so the returned buf is already the correct next obs.
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) != self.num_envs:
            step_infos.pop("final_info", None)
        if len(done_ids) > 0:
            reset_obs_dict, _ = self._env.reset(env_ids=done_ids)
            next_obs = self._flatten_obs(reset_obs_dict)
        else:
            next_obs = terminal_obs

        infos: dict[str, Any] = {
            "final_obs": terminal_obs,  # true terminal obs; train.py uses this for done envs
        }
        infos.update(step_infos)

        # Emit episode return/length for done envs, merged with mjlab's per-reward-term extras
        done_ids_np = done_ids.cpu().numpy()
        raw_log = extras.get("log") or {}
        episode_info: dict[str, Any] = {
            k: float(v.mean().item()) if isinstance(v, torch.Tensor) else v for k, v in raw_log.items()
        }
        if len(done_ids_np) > 0:
            episode_info["episode_rewards"] = float(self._ep_returns[done_ids_np].mean())
            episode_info["episode_length"] = float(self._ep_lengths[done_ids_np].mean())
            self._ep_returns[done_ids_np] = 0.0
            self._ep_lengths[done_ids_np] = 0
        if episode_info:
            infos["episode_info"] = episode_info

        return (
            next_obs,
            rewards_np,
            terminateds.cpu().numpy(),
            truncateds.cpu().numpy(),
            infos,
        )

    def set_eval_mode(self, eval_mode: bool) -> None:
        command = self._get_motion_command()
        if command is None:
            return
        if eval_mode:
            self._dexmanip_eval_mode = True
            if self._dexmanip_eval_state is None:
                self._dexmanip_eval_state = {}
            set_eval_mode = getattr(command, "set_eval_mode", None)
            if callable(set_eval_mode):
                set_eval_mode(
                    sampling_mode="start",
                    noise_to_initial_level=0.0,
                    start_frame=0,
                )
            self._set_eval_object_pin(True)
            self._set_eval_terminations(True)
        else:
            set_train_mode = getattr(command, "set_train_mode", None)
            if callable(set_train_mode):
                set_train_mode()
            self._set_eval_terminations(False)
            self._set_eval_object_pin(False)
            self._dexmanip_tracking_perf = None
            self._dexmanip_eval_mode = False
            self._dexmanip_eval_state = None

    def close(self, **kwargs: Any) -> None:
        if hasattr(self, "_env"):
            self._env.close()

    @classmethod
    def from_env(
        cls,
        env: Any,
        to_numpy: bool = True,
    ) -> "MjlabVectorEnv":
        """Wrap an already-created ManagerBasedRlEnv.

        Disables auto_reset on the env so terminal obs can be captured before
        the env resets done workers (required for correct off-policy bootstrapping).
        """
        env.cfg.auto_reset = False

        instance = cls.__new__(cls)
        instance._env = env
        instance._device = str(env.device)
        instance._to_numpy = to_numpy
        instance.num_envs = env.num_envs

        obs_space = env.single_observation_space
        obs_groups = list(obs_space.spaces.keys())
        if "actor" not in obs_groups:
            raise ValueError(f"mjlab env must expose an 'actor' observation group, got {obs_groups}.")
        instance._actor_obs_key = "actor"
        instance._critic_obs_key = "critic" if "critic" in obs_groups else None
        instance._has_critic_obs = instance._critic_obs_key is not None
        instance._actor_obs_dim = int(obs_space.spaces[instance._actor_obs_key].shape[0])
        flat_dim = (
            int(obs_space.spaces[instance._critic_obs_key].shape[0])
            if instance._has_critic_obs
            else instance._actor_obs_dim
        )

        action_dim = int(env.single_action_space.shape[0])

        instance.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )
        instance.observation_space = batch_space(instance.single_observation_space, env.num_envs)
        instance.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        instance.action_space = batch_space(instance.single_action_space, env.num_envs)

        instance.obs_size = (flat_dim,)
        instance.action_size = (action_dim,)
        instance._ep_returns = np.zeros(env.num_envs, dtype=np.float32)
        instance._ep_lengths = np.zeros(env.num_envs, dtype=np.int32)
        instance._dexmanip_eval_mode = False
        instance._dexmanip_eval_state = None
        instance._dexmanip_tracking_perf = None

        return instance


def make_mjlab_env(
    task_id: str,
    num_envs: int,
    seed: int,
    device: str = "cuda:0",
) -> MjlabVectorEnv:
    return MjlabVectorEnv(task_id=task_id, num_envs=num_envs, seed=seed, device=device)


def make_dexmanip_env(
    env_name: str,
    num_envs: int,
    seed: int,
    device: str = "cuda:0",
    *,
    motion: Any = None,
    reward: Optional[dict[str, Any]] = None,
    observation: Optional[dict[str, Any]] = None,
    event: Optional[dict[str, Any]] = None,
    action: Optional[dict[str, Any]] = None,
    termination: Optional[dict[str, Any]] = None,
    robot: Optional[dict[str, Any]] = None,
    cfg_overrides: Optional[dict[str, Any]] = None,
) -> MjlabVectorEnv:
    """Build a local (non-registry) mjlab task from ``mjlab_envs`` and wrap it for SAC.

    The dexmanip analogue of :func:`make_mjlab_env`. Where ``make_mjlab_env`` pulls
    a cfg from mjlab's task *registry* (``load_env_cfg``), this reaches DOWN into the
    ``flash_rl.envs.mjlab_envs`` content package for the cfg assembly + per-term
    override seam — top-down, mirroring ``isaaclab.make_isaaclab_env -> isaaclab_envs``.
    Both paths then share the exact same wrapper (``MjlabVectorEnv.from_env``:
    auto_reset=False, actor/critic obs, terminal-obs capture for SAC bootstrapping).

    ``motion`` is the packed ``motion.pt`` path the motion command eager-loads at
    __init__ (no data -> no env). ``reward``/``observation``/``event``/``action``/
    ``termination``/``robot`` are per-term override dicts from ``configs/env/dexmanip.yaml``.
    """
    from mjlab.envs import ManagerBasedRlEnv

    from .mjlab_envs.dexmanip import apply_dexmanip_overrides, build_dexmanip_env_cfg

    if env_name not in LOCAL_MJLAB_TASKS:
        print(f"[mjlab] '{env_name}' not in LOCAL_MJLAB_TASKS; building with the default dexmanip preset.")

    env_cfg = build_dexmanip_env_cfg(env_name, num_envs=num_envs, seed=seed, device=device, motion=motion)
    apply_dexmanip_overrides(
        env_cfg,
        reward=reward,
        observation=observation,
        event=event,
        action=action,
        termination=termination,
        robot=robot,
        cfg_overrides=cfg_overrides,
    )
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    return MjlabVectorEnv.from_env(env)
