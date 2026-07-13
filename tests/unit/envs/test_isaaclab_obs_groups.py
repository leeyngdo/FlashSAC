"""Tests for the obs-group concat helper (pure torch, no IsaacSim)."""

import torch

from flash_rl.envs.isaaclab import concat_obs_groups


def test_concat_order_and_shape() -> None:
    obs = {
        "policy": torch.arange(6.0).reshape(2, 3),
        "proprio": torch.arange(4.0).reshape(2, 2) + 100,
        "perception": torch.arange(2.0).reshape(2, 1) + 200,
    }
    out = concat_obs_groups(obs, ["policy", "proprio", "perception"])
    assert out.shape == (2, 6)
    assert torch.allclose(out[0], torch.tensor([0.0, 1.0, 2.0, 100.0, 101.0, 200.0]))


def test_missing_group_raises() -> None:
    try:
        concat_obs_groups({"policy": torch.zeros(2, 3)}, ["policy", "proprio"])
    except KeyError:
        return
    raise AssertionError("expected KeyError")
