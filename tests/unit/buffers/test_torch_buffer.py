"""Replay-buffer 1-step pair sampling for the dynamics ensemble.

Layout under test: add() writes one num_envs-sized batch per env step, so slot
i and slot i + num_envs are consecutive time steps of the same env — the 1-step
next observation is read by index with zero extra storage.
"""

import numpy as np
import torch

from flash_rl.buffers.torch_buffer import (
    MemoryEfficientTorchUniformBuffer,
    TorchUniformBuffer,
)

try:  # gym spaces for buffer construction
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym

E = 4  # envs
OBS_DIM = 2  # obs = [env_id, t]
ACT_DIM = 1  # action = [t]


def _make_buffer(cls, max_length: int, n_step: int = 3):
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(E, OBS_DIM), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1e9, high=1e9, shape=(E, ACT_DIM), dtype=np.float32)
    return cls(
        observation_space=obs_space,
        action_space=act_space,
        n_step=n_step,
        gamma=0.99,
        max_length=max_length,
        min_length=E,
        sample_batch_size=8,
        device_type="cpu",
    )


def _fill(buffer, T: int, done_at: dict[int, set[int]] | None = None):
    """T env steps; obs[t] = [env, t], action = [t], reward = t."""
    done_at = done_at or {}
    for t in range(T):
        term = np.zeros(E, dtype=np.float32)
        for env, ts in done_at.items():
            if t in ts:
                term[env] = 1.0
        obs = np.stack([[e, t] for e in range(E)]).astype(np.float32)
        next_obs = np.stack([[e, t + 1] for e in range(E)]).astype(np.float32)
        buffer.add(
            {
                "observation": obs,
                "action": np.full((E, ACT_DIM), t, dtype=np.float32),
                "reward": np.full(E, float(t), dtype=np.float32),
                "terminated": term,
                "truncated": np.zeros(E, dtype=np.float32),
                "next_observation": next_obs,
            }
        )


def _check_alignment(w):
    obs = w["observation"]  # (B, 2) = [env, t]
    # same env, consecutive time step
    assert torch.equal(w["next_observation"][:, 0], obs[:, 0])
    assert torch.equal(w["next_observation"][:, 1], obs[:, 1] + 1)
    # action/reward belong to time t (raw, NOT n-step sums)
    assert torch.equal(w["action"][:, 0], obs[:, 1])
    assert torch.equal(w["reward"], obs[:, 1])


def test_one_step_pairs_are_same_env_consecutive_steps() -> None:
    for cls in (TorchUniformBuffer, MemoryEfficientTorchUniformBuffer):
        torch.manual_seed(0)
        buf = _make_buffer(cls, max_length=200)
        _fill(buf, T=40)
        w = buf.sample_one_step(batch_size=16)
        assert w["observation"].shape == (16, OBS_DIM)
        assert w["next_observation"].shape == (16, OBS_DIM)
        _check_alignment(w)


def test_one_step_pairs_survive_ring_wraparound() -> None:
    for cls in (TorchUniformBuffer, MemoryEfficientTorchUniformBuffer):
        torch.manual_seed(0)
        buf = _make_buffer(cls, max_length=15 * E)  # wraps after 15 stored steps
        _fill(buf, T=60)
        for _ in range(5):
            w = buf.sample_one_step(batch_size=32)
            _check_alignment(w)


def test_one_step_pairs_flag_done_rows() -> None:
    torch.manual_seed(0)
    buf = _make_buffer(MemoryEfficientTorchUniformBuffer, max_length=400)
    _fill(buf, T=50, done_at={1: {20}})
    found = False
    for _ in range(20):
        w = buf.sample_one_step(batch_size=64)
        hits = (w["observation"][:, 0] == 1) & (w["observation"][:, 1] == 20)
        if bool(hits.any()):
            assert (w["done"][hits] == 1.0).all()
            found = True
        others = ~hits
        assert (w["done"][others] == 0.0).all()
    assert found, "no sample hit the done row; increase attempts"
