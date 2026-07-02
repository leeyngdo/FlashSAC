import os
from collections import deque
from typing import Any, Optional, cast

import gymnasium as gym
import numpy as np
import torch

from flash_rl.buffers.base_buffer import BaseBuffer, Batch
from flash_rl.types import NDArray

# Mapping from numpy dtypes to torch dtypes
_NP_TO_TORCH_DTYPE: dict[np.dtype[Any], torch.dtype] = {
    np.dtype(np.float64): torch.float32,  # enforce float32
    np.dtype(np.float32): torch.float32,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64,
    np.dtype(np.bool_): torch.bool,
    np.dtype(np.uint8): torch.uint8,
}


def _numpy_dtype_to_torch(dtype: Any) -> torch.dtype:
    """Convert a numpy dtype to a torch dtype, enforcing float32 for float64."""
    dtype = np.dtype(dtype)
    if dtype in _NP_TO_TORCH_DTYPE:
        return _NP_TO_TORCH_DTYPE[dtype]
    return torch.float32


class TorchExponentialSampler:
    """Truncated-geometric ("GEOM") recency sampler.

    Draws positions ``i`` in ``{0, ..., size-1}`` with ``P(i) ∝ 2^(k·i)`` where
    ``k = geom_alpha / max_steps`` — exponentially biased toward the newest position
    (``i = size-1``). ``geom_alpha == 0`` is exactly uniform. Sampling is closed-form
    inverse-CDF; the normalization ``Z = (base^size - 1)/(base - 1)`` and the draw are
    kept in float64 because ``base^size`` with ``base ≈ 1 + 1e-5`` and large ``size``
    (or large alpha, where Z overflows float32) is catastrophically lossy otherwise.
    """

    def __init__(self, geom_alpha: float, max_steps: int, device: torch.device):
        if geom_alpha < 0.0:
            raise ValueError(f"geom_alpha must be non-negative, got {geom_alpha}.")
        if geom_alpha > 700.0:
            # 2^alpha must stay finite in float64 (2^700 ~ 5e210).
            raise ValueError(f"geom_alpha must be <= 700 to stay in float64 range, got {geom_alpha}.")
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}.")
        self.geom_alpha = float(geom_alpha)
        self.max_steps = max_steps
        self.device = device
        self.k = self.geom_alpha / max_steps  # exponential rate per position
        self.base = 2.0**self.k
        self.denom = self.base - 1.0
        self.size = 0
        self.Z = 1.0

    def set_size(self, size: int) -> None:
        """Update the live window size (recomputes Z only when it changed; O(1))."""
        if size == self.size:
            return
        self.size = size
        if size <= 0:
            self.Z = 1.0
        elif self.denom == 0.0:  # alpha == 0 -> uniform
            self.Z = float(size)
        else:
            self.Z = (self.base**size - 1.0) / self.denom

    def sample(self, num_samples: int) -> torch.Tensor:
        """Return (num_samples,) int64 recency positions in [0, size-1] on the device."""
        if self.denom == 0.0:
            return torch.randint(0, self.size, (num_samples,), device=self.device)
        r = torch.rand(num_samples, device=self.device, dtype=torch.float64) * self.Z
        # r in [S(i-1), S(i)) with S(i) = (base^(i+1)-1)/denom  =>  log2(1+r*denom)/k in [i, i+1).
        # (The tg-sampler branch subtracted 1 here — an off-by-one that shifts every bucket down
        # and never samples the newest position.)
        inside = 1.0 + r * self.denom
        i = torch.floor(torch.log2(inside) / self.k).to(torch.int64)
        return i.clamp_(0, self.size - 1)


class TorchUniformBuffer(BaseBuffer):
    """
    A uniform experience replay buffer using PyTorch tensors.
    Mirrors NpyUniformBuffer behavior exactly; data is stored on the given device.

    SAPG: when ``num_agents > 1`` each stored transition is tagged with the id of the agent/block
    that collected it. Blocks are contiguous env-index ranges of size ``num_envs // num_agents``
    (env j -> agent j // block_size), matching how the rollout assigns env blocks to policies.
    ``sample_sapg()`` uses those tags as access control: every policy samples its own block, then
    the leader receives additional follower-block samples retagged with the leader id. With
    ``num_agents == 1`` all ids are 0, so the buffer behaves exactly like vanilla SAC.

    Recency-biased ("GEOM" / truncated-geometric) replay: ``geom_alpha > 0`` biases the global
    ``sample()`` distribution toward recent transitions; ``agent_geom_alphas`` gives each SAPG
    agent its own recency bias for block sampling (the alpha of the block being sampled FROM,
    including the leader's donor draws). Alphas of 0 (default) reproduce uniform replay.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[NDArray],
        n_step: int,
        gamma: float,
        max_length: int,
        min_length: int,
        sample_batch_size: int,
        device_type: str,
        obs_storage_dtype: Optional[torch.dtype] = None,
        num_agents: int = 1,
        geom_alpha: float = 0.0,
        agent_geom_alphas: Optional[list[float]] = None,
    ):
        super(TorchUniformBuffer, self).__init__(
            observation_space,
            action_space,
            n_step,
            gamma,
            max_length,
            min_length,
            sample_batch_size,
        )
        device_type = (
            device_type
            if device_type.startswith("cuda") and ":" in device_type
            else ("cuda:0" if device_type.startswith("cuda") else "cpu")
        )
        self._device = torch.device(device_type)
        self._obs_storage_dtype = obs_storage_dtype
        self._num_agents = num_agents
        if self._num_agents < 1:
            raise ValueError(f"num_agents must be >= 1, got {self._num_agents}.")
        self._geom_alpha = float(geom_alpha)
        if self._geom_alpha < 0.0:
            raise ValueError(f"geom_alpha must be non-negative, got {geom_alpha}.")
        if agent_geom_alphas is not None:
            if len(agent_geom_alphas) != self._num_agents:
                raise ValueError(
                    f"agent_geom_alphas must have length num_agents={self._num_agents}, "
                    f"got {len(agent_geom_alphas)}."
                )
            self._agent_geom_alphas: Optional[list[float]] = [float(a) for a in agent_geom_alphas]
            if any(a < 0.0 for a in self._agent_geom_alphas):
                raise ValueError(f"agent_geom_alphas must be non-negative, got {agent_geom_alphas}.")
        else:
            self._agent_geom_alphas = None
        self.reset()

    def __len__(self) -> int:
        return self._num_in_buffer

    def reset(self) -> None:
        m = self._max_length
        pin = self._device.type == "cpu" and torch.cuda.is_available()

        observation_shape = (self._observation_space.shape[-1],) if self._observation_space.shape is not None else (0,)
        observation_dtype = _numpy_dtype_to_torch(
            self._observation_space.dtype if self._observation_space.dtype is not None else np.float32
        )

        action_shape = (self._action_space.shape[-1],) if self._action_space.shape is not None else (0,)
        action_dtype = _numpy_dtype_to_torch(
            self._action_space.dtype if self._action_space.dtype is not None else np.float32
        )

        obs_storage_dtype = self._obs_storage_dtype or observation_dtype
        self._observations = torch.empty(
            (m,) + observation_shape, dtype=obs_storage_dtype, device=self._device, pin_memory=pin
        )
        self._next_observations = torch.empty(
            (m,) + observation_shape, dtype=obs_storage_dtype, device=self._device, pin_memory=pin
        )
        self._actions = torch.empty((m,) + action_shape, dtype=action_dtype, device=self._device, pin_memory=pin)
        self._rewards = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)
        self._terminateds = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)
        self._truncateds = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)
        # SAPG: id of the agent/block that collected each transition (0 when num_agents == 1).
        self._agent_ids = torch.zeros((m,), dtype=torch.long, device=self._device, pin_memory=pin)

        self._n_step_transitions: deque[dict[str, Any]] = deque(maxlen=self._n_step)
        self._num_in_buffer = 0
        self._current_idx = 0
        # Recency sampling state (lazily built once the add batch size is known).
        self._sampling_add_batch: Optional[int] = None
        self._row_sampler: Optional[TorchExponentialSampler] = None
        self._agent_step_samplers: Optional[list[Optional[TorchExponentialSampler]]] = None

    def _to_tensor(self, value: Any) -> torch.Tensor:
        """Convert a value to a tensor on the buffer device (cloned if already a tensor)."""
        if isinstance(value, torch.Tensor):
            return value.detach().to(self._device, copy=True)
        return torch.tensor(value, device=self._device)

    def _get_n_step_prev_transition(self) -> Batch:
        """
        Processes n_step_transitions to compute the n-step return, done status,
        and next observation. Mirrors NpyUniformBuffer._get_n_step_prev_transition exactly.
        """
        n_step_prev_transition = self._n_step_transitions[0]
        curr_transition = self._n_step_transitions[-1]

        # clone last transition
        n_step_reward = curr_transition["reward"].clone()
        n_step_terminated = curr_transition["terminated"].clone()
        n_step_truncated = curr_transition["truncated"].clone()
        n_step_next_observation = curr_transition["next_observation"].clone()

        for n_step_idx in reversed(range(self._n_step - 1)):
            transition = self._n_step_transitions[n_step_idx]
            reward = transition["reward"]  # (n,)
            terminated = transition["terminated"]  # (n,)
            truncated = transition["truncated"]  # (n,)
            next_observation = transition["next_observation"]  # (n, *obs_shape)

            # compute n-step return
            done = (terminated.bool() | truncated.bool()).float()
            n_step_reward = reward + self._gamma * n_step_reward * (1 - done)

            # assign next observation starting from done
            done_mask = done.bool()
            n_step_terminated[done_mask] = terminated[done_mask]
            n_step_truncated[done_mask] = truncated[done_mask]
            n_step_next_observation[done_mask] = next_observation[done_mask]

        n_step_prev_transition["reward"] = n_step_reward
        n_step_prev_transition["terminated"] = n_step_terminated
        n_step_prev_transition["truncated"] = n_step_truncated
        n_step_prev_transition["next_observation"] = n_step_next_observation

        return cast(Batch, n_step_prev_transition)

    def _compute_agent_ids(self, add_batch_size: int) -> torch.Tensor:
        """Agent/block id for each of the ``add_batch_size`` env transitions (env order)."""
        if self._num_agents <= 1:
            return torch.zeros(add_batch_size, dtype=torch.long, device=self._device)
        if add_batch_size % self._num_agents != 0:
            raise ValueError(
                "SAPG requires the number of parallel envs added per step to be divisible by "
                f"num_agents. Got add_batch_size={add_batch_size}, num_agents={self._num_agents}."
            )
        block_size = max(1, add_batch_size // self._num_agents)
        ids = torch.div(
            torch.arange(add_batch_size, device=self._device), block_size, rounding_mode="floor"
        )
        return ids.clamp_(max=self._num_agents - 1)

    def _sample_global_indices(self, num_samples: int) -> torch.Tensor:
        return torch.randint(0, self._num_in_buffer, (num_samples,), device=self._device)

    def _valid_row_window(self) -> int:
        """Number of sampleable rows in LOGICAL (recency) order; row 0 is the oldest."""
        return self._num_in_buffer

    def _logical_to_storage(self, rows: torch.Tensor) -> torch.Tensor:
        """Map logical (recency-ordered) row positions to ring-storage indices."""
        if self._num_in_buffer == self._max_length:
            return (rows + self._current_idx) % self._max_length
        return rows

    def _geom_alpha_for(self, agent_id: int) -> float:
        if self._agent_geom_alphas is not None:
            return self._agent_geom_alphas[agent_id]
        return self._geom_alpha

    def _get_row_sampler(self) -> TorchExponentialSampler:
        if self._row_sampler is None:
            self._row_sampler = TorchExponentialSampler(
                geom_alpha=self._geom_alpha, max_steps=self._max_length, device=self._device
            )
        return self._row_sampler

    def _get_step_sampler(self, agent_id: int) -> Optional[TorchExponentialSampler]:
        """Per-agent recency sampler over add-step groups (None for alpha == 0 -> uniform)."""
        assert self._sampling_add_batch is not None
        if self._agent_step_samplers is None:
            capacity_steps = max(1, self._max_length // self._sampling_add_batch)
            self._agent_step_samplers = [
                (
                    TorchExponentialSampler(
                        geom_alpha=self._geom_alpha_for(a), max_steps=capacity_steps, device=self._device
                    )
                    if self._geom_alpha_for(a) > 0.0
                    else None
                )
                for a in range(self._num_agents)
            ]
        return self._agent_step_samplers[agent_id]

    def _sample_geometric_rows(self, num_samples: int) -> torch.Tensor:
        """Recency-biased global row sampling (storage indices)."""
        window = self._valid_row_window()
        if window <= 0:
            raise RuntimeError("Cannot sample: no valid transitions are available yet.")
        sampler = self._get_row_sampler()
        sampler.set_size(window)
        return self._logical_to_storage(sampler.sample(num_samples))

    def _sample_agent_indices(self, agent_id: int, num_samples: int) -> torch.Tensor:
        """Sample storage indices belonging to ``agent_id``'s env block, sync-free.

        Rows are written in fixed env order, add-batch at a time, so the env offset of the
        LOGICAL row r is ``(r - num_in_buffer) mod add_batch`` (total writes are always a
        multiple of add_batch; the mod-phase matters once the ring wraps with a capacity
        that is not a multiple of add_batch). We therefore draw an add-step group s (with
        the agent's recency bias, or uniformly for alpha == 0) and an offset j inside the
        agent's block, and map to the logical row arithmetically — no rejection sampling,
        no data-dependent host syncs. Up to one partial add-batch of the newest valid rows
        is ignored when the window is not a multiple of add_batch.
        """
        if num_samples == 0:
            return torch.empty((0,), dtype=torch.long, device=self._device)
        ab = self._sampling_add_batch
        if ab is None:
            raise RuntimeError("Cannot sample per-agent indices before any transition was added.")
        window = self._valid_row_window()
        steps = window // ab
        if steps <= 0:
            raise RuntimeError(
                f"Cannot sample SAPG agent {agent_id}: no full add-batch is in the sampleable window yet."
            )
        block = ab // self._num_agents
        sampler = self._get_step_sampler(agent_id)
        if sampler is not None:
            sampler.set_size(steps)
            s = sampler.sample(num_samples)
        else:
            s = torch.randint(0, steps, (num_samples,), device=self._device)
        j = torch.randint(0, block, (num_samples,), device=self._device)
        t = (agent_id * block + j + self._num_in_buffer) % ab
        return self._logical_to_storage(s * ab + t)

    def add(self, transition: Batch) -> None:
        self._n_step_transitions.append({key: self._to_tensor(value) for key, value in transition.items()})

        if len(self._n_step_transitions) >= self._n_step:
            n_step_prev_transition = cast(dict[str, torch.Tensor], self._get_n_step_prev_transition())

            add_batch_size = len(n_step_prev_transition["observation"])
            end_idx = self._current_idx + add_batch_size

            if end_idx <= self._max_length:
                # Contiguous slice — avoids scatter and tensor allocation
                idxs: Any = slice(self._current_idx, end_idx)
            else:
                idxs = (torch.arange(add_batch_size, device=self._device) + self._current_idx) % self._max_length

            self._observations[idxs] = n_step_prev_transition["observation"].to(self._observations.dtype)
            self._next_observations[idxs] = n_step_prev_transition["next_observation"].to(self._next_observations.dtype)
            self._actions[idxs] = n_step_prev_transition["action"].to(self._actions.dtype)
            self._rewards[idxs] = n_step_prev_transition["reward"].to(self._rewards.dtype)
            self._terminateds[idxs] = n_step_prev_transition["terminated"].to(self._terminateds.dtype)
            self._truncateds[idxs] = n_step_prev_transition["truncated"].to(self._truncateds.dtype)
            self._agent_ids[idxs] = self._compute_agent_ids(add_batch_size)

            self._register_add_batch_size(add_batch_size)
            self._num_in_buffer = min(self._num_in_buffer + add_batch_size, self._max_length)
            self._current_idx = (self._current_idx + add_batch_size) % self._max_length

    def _register_add_batch_size(self, add_batch_size: int) -> None:
        """Track the (constant) per-add batch size that per-agent/recency sampling relies on."""
        if self._sampling_add_batch is None:
            self._sampling_add_batch = add_batch_size
        elif self._sampling_add_batch != add_batch_size:
            if self._num_agents > 1 or self._geom_alpha > 0.0 or self._agent_geom_alphas is not None:
                raise ValueError(
                    "Per-agent / recency-biased sampling requires a constant add batch size; "
                    f"got {add_batch_size} after {self._sampling_add_batch}."
                )
            self._sampling_add_batch = add_batch_size
            self._agent_step_samplers = None

    def can_sample(self) -> bool:
        return self._num_in_buffer >= self._min_length

    def sample(self, sample_idxs: Optional[NDArray] = None) -> Batch:
        if sample_idxs is None:
            if self._geom_alpha > 0.0:
                idxs = self._sample_geometric_rows(self._sample_batch_size)
            else:
                idxs = torch.randint(0, self._num_in_buffer, (self._sample_batch_size,), device=self._device)
        else:
            idxs = torch.as_tensor(sample_idxs, device=self._device, dtype=torch.long)

        batch: Batch = {}
        batch["observation"] = self._observations[idxs]
        batch["action"] = self._actions[idxs]
        batch["reward"] = self._rewards[idxs]
        batch["terminated"] = self._terminateds[idxs]
        batch["truncated"] = self._truncateds[idxs]
        batch["next_observation"] = self._next_observations[idxs]
        batch["agent_id"] = self._agent_ids[idxs]

        if self._obs_storage_dtype is not None:
            batch["observation"] = cast(torch.Tensor, batch["observation"]).to(torch.float32)
            batch["next_observation"] = cast(torch.Tensor, batch["next_observation"]).to(torch.float32)

        return batch

    def sample_sapg(self, leader_id: int, off_policy_ratio: int = 1) -> Batch:
        if self._num_agents <= 1:
            return self.sample()
        if not 0 <= leader_id < self._num_agents:
            raise ValueError(f"leader_id must be in [0, {self._num_agents}), got {leader_id}.")
        if off_policy_ratio < 0:
            raise ValueError(f"off_policy_ratio must be >= 0, got {off_policy_ratio}.")

        base_count = self._sample_batch_size // self._num_agents
        remainder = self._sample_batch_size % self._num_agents
        idx_chunks: list[torch.Tensor] = []
        train_id_chunks: list[torch.Tensor] = []
        for agent_id in range(self._num_agents):
            num_samples = base_count + (remainder if agent_id == leader_id else 0)
            if num_samples == 0:
                continue
            idxs = self._sample_agent_indices(agent_id, num_samples)
            idx_chunks.append(idxs)
            train_id_chunks.append(torch.full((num_samples,), agent_id, dtype=torch.long, device=self._device))

        # SAPG leader-follower aggregate: add up to ``off_policy_ratio`` follower blocks to
        # the leader update, retagged as leader samples. This mirrors original SAPG's
        # use_others_experience="lf" augmentation more closely than replacing the leader's own
        # block with a global sample.
        extra_blocks = min(int(off_policy_ratio), self._num_agents - 1)
        if extra_blocks > 0 and base_count > 0:
            follower_ids = torch.tensor(
                [agent_id for agent_id in range(self._num_agents) if agent_id != leader_id],
                dtype=torch.long,
                device=self._device,
            )
            selected_followers = follower_ids[torch.randperm(follower_ids.numel(), device=self._device)[:extra_blocks]]
            for follower_id in selected_followers.tolist():
                idxs = self._sample_agent_indices(int(follower_id), base_count)
                idx_chunks.append(idxs)
                train_id_chunks.append(torch.full((base_count,), leader_id, dtype=torch.long, device=self._device))

        idxs = torch.cat(idx_chunks, dim=0)
        train_agent_ids = torch.cat(train_id_chunks, dim=0)
        perm = torch.randperm(idxs.numel(), device=self._device)
        batch = self.sample(cast(NDArray, idxs[perm]))
        batch["collector_agent_id"] = batch["agent_id"]
        batch["agent_id"] = train_agent_ids[perm]
        return batch

    def save(self, path: str) -> None:
        """
        Save buffer contents and metadata.
        args:
            path (str): The full file path (e.g. "checkpoints/replay_buffer.pt").
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        n = self._num_in_buffer
        dataset: dict[str, Any] = {
            "observation": self._observations[:n],
            "action": self._actions[:n],
            "reward": self._rewards[:n],
            "terminated": self._terminateds[:n],
            "truncated": self._truncateds[:n],
            "next_observation": self._next_observations[:n],
            "agent_id": self._agent_ids[:n],
            "num_in_buffer": self._num_in_buffer,
            "current_idx": self._current_idx,
        }
        torch.save(dataset, path)

    def load(self, path: str) -> None:
        """
        Load buffer contents and metadata.
        args:
            path (str): The full file path (e.g. "checkpoints/replay_buffer.pt").
        """
        dataset = torch.load(path, map_location=self._device)
        n = dataset["num_in_buffer"]

        self._observations[:n] = dataset["observation"]
        self._next_observations[:n] = dataset["next_observation"]
        self._actions[:n] = dataset["action"]
        self._rewards[:n] = dataset["reward"]
        self._terminateds[:n] = dataset["terminated"]
        self._truncateds[:n] = dataset["truncated"]
        if "agent_id" in dataset:  # backward-compat with buffers saved before SAPG
            self._agent_ids[:n] = dataset["agent_id"]

        self._num_in_buffer = n
        self._current_idx = dataset["current_idx"]
        # Note: _n_step_transitions is intentionally not saved/loaded.
        # At most (n_step - 1) in-flight transitions are lost, which is negligible.
        self._n_step_transitions.clear()

    def get_observations(self) -> torch.Tensor:
        return self._observations[: self._num_in_buffer]


class MemoryEfficientTorchUniformBuffer(TorchUniformBuffer):
    """
    Store only observations and reconstruct n-step next observations by index.

    The newest n_step vector-env batches are not sampled because their future
    observation slots have not been written yet. Episode ends keep a sparse
    copy of final next observations because the following observation slot may
    already contain a reset observation.
    """

    def reset(self) -> None:
        m = self._max_length
        pin = self._device.type == "cpu" and torch.cuda.is_available()

        observation_shape = (self._observation_space.shape[-1],) if self._observation_space.shape is not None else (0,)
        observation_dtype = _numpy_dtype_to_torch(
            self._observation_space.dtype if self._observation_space.dtype is not None else np.float32
        )

        action_shape = (self._action_space.shape[-1],) if self._action_space.shape is not None else (0,)
        action_dtype = _numpy_dtype_to_torch(
            self._action_space.dtype if self._action_space.dtype is not None else np.float32
        )

        obs_storage_dtype = self._obs_storage_dtype or observation_dtype
        self._observations = torch.empty(
            (m,) + observation_shape,
            dtype=obs_storage_dtype,
            device=self._device,
            pin_memory=pin,
        )
        self._actions = torch.empty((m,) + action_shape, dtype=action_dtype, device=self._device, pin_memory=pin)
        self._rewards = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)
        self._terminateds = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)
        self._truncateds = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)
        self._agent_ids = torch.zeros((m,), dtype=torch.long, device=self._device, pin_memory=pin)

        self._n_step_transitions: deque[dict[str, Any]] = deque(maxlen=self._n_step)
        self._num_in_buffer = 0
        self._current_idx = 0
        self._add_batch_size: Optional[int] = None
        self._episode_end_next_observations: dict[int, torch.Tensor] = {}
        # Recency sampling state (lazily built once the add batch size is known).
        self._sampling_add_batch: Optional[int] = None
        self._row_sampler: Optional[TorchExponentialSampler] = None
        self._agent_step_samplers: Optional[list[Optional[TorchExponentialSampler]]] = None

    def _valid_sample_high(self) -> int:
        assert self._add_batch_size is not None
        return self._num_in_buffer - self._n_step * self._add_batch_size

    def _valid_row_window(self) -> int:
        # The newest n_step add-batches have no valid next-observation slots yet; logical
        # row 0 is still the oldest row in the buffer, so the phase arithmetic in
        # _sample_agent_indices is unchanged.
        return self._valid_sample_high()

    def _valid_sample_indices(self) -> torch.Tensor:
        sample_high = self._valid_sample_high()
        idxs = torch.arange(sample_high, device=self._device)
        if self._num_in_buffer == self._max_length:
            idxs = (idxs + self._current_idx) % self._max_length
        return idxs

    def _sample_global_indices(self, num_samples: int) -> torch.Tensor:
        if num_samples == 0:
            return torch.empty((0,), dtype=torch.long, device=self._device)
        sample_high = self._valid_sample_high()
        if sample_high <= 0:
            raise RuntimeError("Cannot sample: no valid n-step next-observation slots are available yet.")
        idxs = torch.randint(0, sample_high, (num_samples,), device=self._device)
        if self._num_in_buffer == self._max_length:
            idxs = (idxs + self._current_idx) % self._max_length
        return idxs

    # _sample_agent_indices is inherited from TorchUniformBuffer: the arithmetic
    # step-x-offset sampler only needs _valid_row_window()/_logical_to_storage(),
    # both of which respect this class's n-step exclusion window.

    def add(self, transition: Batch) -> None:
        self._n_step_transitions.append({key: self._to_tensor(value) for key, value in transition.items()})

        if len(self._n_step_transitions) < self._n_step:
            return

        n_step_prev_transition = cast(dict[str, torch.Tensor], self._get_n_step_prev_transition())
        add_batch_size = len(n_step_prev_transition["observation"])
        if self._add_batch_size is None:
            self._add_batch_size = add_batch_size
            if self._n_step * add_batch_size >= self._max_length:
                raise ValueError("max_length must be larger than n_step * add_batch_size")
        elif add_batch_size != self._add_batch_size:
            raise ValueError("MemoryEfficientTorchUniformBuffer requires a constant add batch size")

        end_idx = self._current_idx + add_batch_size
        idx_tensor: Optional[torch.Tensor] = None
        idxs: Any = slice(self._current_idx, end_idx)
        if end_idx > self._max_length:
            idx_tensor = (torch.arange(add_batch_size, device=self._device) + self._current_idx) % self._max_length
            idxs = idx_tensor

        self._observations[idxs] = n_step_prev_transition["observation"].to(self._observations.dtype)
        self._actions[idxs] = n_step_prev_transition["action"].to(self._actions.dtype)
        self._rewards[idxs] = n_step_prev_transition["reward"].to(self._rewards.dtype)
        self._terminateds[idxs] = n_step_prev_transition["terminated"].to(self._terminateds.dtype)
        self._truncateds[idxs] = n_step_prev_transition["truncated"].to(self._truncateds.dtype)
        self._agent_ids[idxs] = self._compute_agent_ids(add_batch_size)

        if self._episode_end_next_observations:
            if end_idx > self._max_length:
                assert idx_tensor is not None
                for idx in idx_tensor.detach().cpu().tolist():
                    self._episode_end_next_observations.pop(int(idx), None)
            else:
                for idx in range(self._current_idx, end_idx):
                    self._episode_end_next_observations.pop(idx, None)
        episode_end_mask = n_step_prev_transition["terminated"].bool() | n_step_prev_transition["truncated"].bool()
        if episode_end_mask.any():
            if end_idx > self._max_length:
                assert idx_tensor is not None
                episode_end_idxs = idx_tensor[episode_end_mask].detach().cpu().tolist()
            else:
                episode_end_positions = episode_end_mask.nonzero(as_tuple=False).squeeze(-1)
                episode_end_idxs = (episode_end_positions + self._current_idx).detach().cpu().tolist()
            episode_end_obs = n_step_prev_transition["next_observation"][episode_end_mask].to(self._observations.dtype)
            for idx, obs in zip(episode_end_idxs, episode_end_obs):
                self._episode_end_next_observations[int(idx)] = obs.detach().clone()

        self._register_add_batch_size(add_batch_size)
        self._num_in_buffer = min(self._num_in_buffer + add_batch_size, self._max_length)
        self._current_idx = (self._current_idx + add_batch_size) % self._max_length

    def can_sample(self) -> bool:
        return (
            self._num_in_buffer >= self._min_length
            and self._add_batch_size is not None
            and self._num_in_buffer > self._n_step * self._add_batch_size
        )

    def sample(self, sample_idxs: Optional[NDArray] = None) -> Batch:
        assert self._add_batch_size is not None
        if sample_idxs is None:
            if self._geom_alpha > 0.0:
                idxs = self._sample_geometric_rows(self._sample_batch_size)
            else:
                sample_high = self._num_in_buffer - self._n_step * self._add_batch_size
                idxs = torch.randint(0, sample_high, (self._sample_batch_size,), device=self._device)
                if self._num_in_buffer == self._max_length:
                    idxs = (idxs + self._current_idx) % self._max_length
        else:
            idxs = torch.as_tensor(sample_idxs, device=self._device, dtype=torch.long)

        batch: Batch = {}
        batch["observation"] = self._observations[idxs]
        batch["action"] = self._actions[idxs]
        batch["reward"] = self._rewards[idxs]
        batch["terminated"] = self._terminateds[idxs]
        batch["truncated"] = self._truncateds[idxs]
        batch["agent_id"] = self._agent_ids[idxs]

        next_idxs = (idxs + self._n_step * self._add_batch_size) % self._max_length
        batch["next_observation"] = self._observations[next_idxs]
        if self._episode_end_next_observations:
            hits = [
                (pos, obs)
                for pos, idx in enumerate(idxs.detach().cpu().tolist())
                if (obs := self._episode_end_next_observations.get(int(idx))) is not None
            ]
            if hits:
                positions, next_observations = zip(*hits)
                batch["next_observation"][torch.as_tensor(positions, device=self._device)] = torch.stack(
                    list(next_observations)
                ).to(self._device)

        if self._obs_storage_dtype is not None:
            batch["observation"] = cast(torch.Tensor, batch["observation"]).to(torch.float32)
            batch["next_observation"] = cast(torch.Tensor, batch["next_observation"]).to(torch.float32)

        return batch

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        n = self._num_in_buffer
        torch.save(
            {
                "observation": self._observations[:n],
                "action": self._actions[:n],
                "reward": self._rewards[:n],
                "terminated": self._terminateds[:n],
                "truncated": self._truncateds[:n],
                "agent_id": self._agent_ids[:n],
                "num_in_buffer": self._num_in_buffer,
                "current_idx": self._current_idx,
                "add_batch_size": self._add_batch_size,
                "episode_end_next_observations": self._episode_end_next_observations,
            },
            path,
        )

    def load(self, path: str) -> None:
        dataset = torch.load(path, map_location=self._device)
        n = dataset["num_in_buffer"]

        self._observations[:n] = dataset["observation"]
        self._actions[:n] = dataset["action"]
        self._rewards[:n] = dataset["reward"]
        self._terminateds[:n] = dataset["terminated"]
        self._truncateds[:n] = dataset["truncated"]
        if "agent_id" in dataset:
            self._agent_ids[:n] = dataset["agent_id"]
        self._num_in_buffer = n
        self._current_idx = dataset["current_idx"]
        self._add_batch_size = dataset["add_batch_size"]
        # Keep the sampling machinery usable before the first post-restore add().
        if self._add_batch_size is not None:
            self._register_add_batch_size(self._add_batch_size)
        self._episode_end_next_observations = dataset.get(
            "episode_end_next_observations",
            dataset.get("timeout_next_observations", {}),
        )
        self._n_step_transitions.clear()
