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


class TorchUniformBuffer(BaseBuffer):
    """
    A uniform experience replay buffer using PyTorch tensors.
    Mirrors NpyUniformBuffer behavior exactly; data is stored on the given device.
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

        self._n_step_transitions: deque[dict[str, Any]] = deque(maxlen=self._n_step)
        self._num_in_buffer = 0
        self._current_idx = 0

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

            self._num_in_buffer = min(self._num_in_buffer + add_batch_size, self._max_length)
            self._current_idx = (self._current_idx + add_batch_size) % self._max_length

    def can_sample(self) -> bool:
        return self._num_in_buffer >= self._min_length

    def sample(self, sample_idxs: Optional[NDArray] = None) -> Batch:
        if sample_idxs is None:
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

        if self._obs_storage_dtype is not None:
            batch["observation"] = cast(torch.Tensor, batch["observation"]).to(torch.float32)
            batch["next_observation"] = cast(torch.Tensor, batch["next_observation"]).to(torch.float32)

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

        self._num_in_buffer = n
        self._current_idx = dataset["current_idx"]
        # Note: _n_step_transitions is intentionally not saved/loaded.
        # At most (n_step - 1) in-flight transitions are lost, which is negligible.
        self._n_step_transitions.clear()

    def get_observations(self) -> torch.Tensor:
        return self._observations[: self._num_in_buffer]


class TorchExponentialSampler:
    """
    Samples recency positions {0,...,size-1} with probability proportional to
    base^i = 2^(k*i), i.e. exponentially biased toward the newest transitions
    (i = size-1). Truncated-Geometric / "GEOM" sampler. geom_alpha == 0.0 => uniform.

    Mirrors NpyUniformBuffer.ExponentialSampler exactly, on the given device.
    """

    def __init__(self, geom_alpha: float, max_steps: int, device: torch.device):
        if geom_alpha < 0.0:
            raise ValueError("geom_alpha must be non-negative")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.geom_alpha = float(geom_alpha)
        self.max_steps = max_steps
        self.device = device
        # fp64 scalars - precision-critical: base^size with base ~= 1+1e-5 and
        # size ~= 1e6 is catastrophically lossy in fp32, so keep these as Python floats.
        self.k = self.geom_alpha / max_steps  # exponential rate
        self.base = 2.0**self.k  # 2^k
        self.denom = self.base - 1.0  # base - 1
        self.size = 0  # updated externally on add()
        self.Z = 1.0  # normalization constant, updated externally on add()

    def compute_Z(self, size: int) -> float:
        """Finite geometric sum sum_{i=0}^{size-1} base^i, computed in float64."""
        if size <= 0:
            return 1.0
        if self.denom == 0.0:  # alpha == 0 => base == 1 => uniform; Z == size
            return float(size)
        return (self.base**size - 1.0) / self.denom

    def _sample_recency(self, num_samples: int) -> torch.Tensor:
        """CDF-inversion sampling of recency positions in [0, size-1], on device."""
        # r ~ U[0, Z). Drawn in fp32 (cheap on GPU), scaled by the fp64-correct Z.
        r = torch.rand(num_samples, device=self.device, dtype=torch.float32) * float(self.Z)
        # inside = base^(i+1) in [1, base^size]; small number => fp32 is fine here.
        inside = 1.0 + r * float(self.denom)
        # i = log2(inside) / k - 1   (closed-form inverse CDF)
        i = torch.log2(inside) / float(self.k) - 1.0
        i = torch.floor(i).to(torch.int64)
        return i.clamp_(0, self.size - 1)

    def sample(self, batch_size: int) -> torch.Tensor:
        """Return a (batch_size,) int64 tensor of recency positions on device."""
        if self.denom == 0.0:
            # uniform fast path - identical distribution to TorchUniformBuffer.sample()
            return torch.randint(0, self.size, (batch_size,), device=self.device)
        return self._sample_recency(batch_size)


class TorchGeometricBuffer(TorchUniformBuffer):
    """
    Truncated-Geometric ("GEOM" / exponential) replay buffer.

    Identical storage / add path to TorchUniformBuffer; only the sampling
    distribution over recency positions changes. geom_alpha == 0.0 => uniform.
    Mirrors NpyUniformBuffer with an ExponentialSampler.
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
        geom_alpha: float = 10.0,
    ):
        self._geom_alpha = geom_alpha
        # reset() is called inside super().__init__ (TorchUniformBuffer.__init__).
        super().__init__(
            observation_space,
            action_space,
            n_step,
            gamma,
            max_length,
            min_length,
            sample_batch_size,
            device_type,
            obs_storage_dtype=obs_storage_dtype,
        )

    def reset(self) -> None:
        super().reset()
        # _device is set by TorchUniformBuffer.__init__ before reset() is called.
        self.sampler = TorchExponentialSampler(
            geom_alpha=self._geom_alpha,
            max_steps=self._max_length,
            device=self._device,
        )
        self.sampler.size = 0
        self.sampler.Z = 1.0

    def _geom_window(self) -> int:
        """Number of entries the sampler may draw from, anchored at the oldest entry."""
        return self._num_in_buffer

    def _refresh_sampler(self) -> None:
        # compute_Z is O(1); the guard short-circuits once the window stops growing.
        window = self._geom_window()
        if self.sampler.size != window:
            self.sampler.size = window
            self.sampler.Z = self.sampler.compute_Z(window)

    def add(self, transition: Batch) -> None:
        super().add(transition)
        # After super().add advances _num_in_buffer, refresh the sampler size / Z.
        self._refresh_sampler()

    def sample(self, sample_idxs: Optional[NDArray] = None) -> Batch:
        if sample_idxs is None:
            recency_idxs = self.sampler.sample(self._sample_batch_size)  # int64, device, 0..size-1
            # recency (0=oldest) -> absolute ring index, re-anchored to the live window.
            sample_idxs = (self._current_idx - self._num_in_buffer + recency_idxs).remainder(self._max_length)
        return super().sample(sample_idxs=sample_idxs)

    def load(self, path: str) -> None:
        # TorchUniformBuffer.load restores _num_in_buffer / _current_idx but not the
        # sampler's size / Z (which are only refreshed on add()). Refresh them here so a
        # sample() issued after load() but before the next add() (e.g. a resumed run that
        # updates before collecting) maps recency to the correct ring window.
        super().load(path)
        self._refresh_sampler()


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

        self._n_step_transitions: deque[dict[str, Any]] = deque(maxlen=self._n_step)
        self._num_in_buffer = 0
        self._current_idx = 0
        self._add_batch_size: Optional[int] = None
        self._episode_end_next_observations: dict[int, torch.Tensor] = {}

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
        self._num_in_buffer = n
        self._current_idx = dataset["current_idx"]
        self._add_batch_size = dataset["add_batch_size"]
        self._episode_end_next_observations = dataset.get(
            "episode_end_next_observations",
            dataset.get("timeout_next_observations", {}),
        )
        self._n_step_transitions.clear()


class MemoryEfficientTorchGeometricBuffer(TorchGeometricBuffer, MemoryEfficientTorchUniformBuffer):
    """
    Truncated-Geometric ("GEOM") sampling over the memory-efficient storage layout.

    MRO: TorchGeometricBuffer contributes the recency sampler and its super() calls
    resolve to MemoryEfficientTorchUniformBuffer, which contributes storage, add and
    next-observation reconstruction. The only geometric-specific change is the sampler
    window: the newest n_step * add_batch_size slots are excluded because their future
    observation slots have not been written yet (same exclusion as the uniform
    memory-efficient buffer). Recency 0 stays anchored at the oldest entry, so the
    recency -> ring-index mapping in TorchGeometricBuffer.sample is unchanged.
    """

    def _geom_window(self) -> int:
        if self._add_batch_size is None:
            return 0
        return max(0, self._num_in_buffer - self._n_step * self._add_batch_size)
