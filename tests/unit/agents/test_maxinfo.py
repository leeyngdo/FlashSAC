"""Tests for the MaxInfoRL (MaxInfoSAC) modules and their FlashSAC integration."""

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from flash_rl.agents.flashSAC.agent import FlashSACAgent, FlashSACConfig
from flash_rl.agents.flashSAC.maxinfo import (
    MaxInfoDynamics,
    RunningNormalizer,
    update_dyn_scale,
)
from flash_rl.agents.flashSAC.network import FlashSACTemperature
from flash_rl.agents.utils.network import Network

NUM_ENVS = 8
OBS_DIM = 6
ACT_DIM = 2


def _make_agent(**overrides: Any) -> FlashSACAgent:
    cfg_kwargs: dict[str, Any] = dict(
        seed=0,
        normalize_reward=False,
        normalized_G_max=5.0,
        asymmetric_observation=False,
        device_type="cpu",
        buffer_max_length=1000,
        buffer_min_length=64,
        buffer_device_type="cpu",
        sample_batch_size=32,
        learning_rate_init=3e-4,
        learning_rate_peak=3e-4,
        learning_rate_end=1.5e-4,
        learning_rate_warmup_rate=1e-6,
        learning_rate_warmup_step=1,
        learning_rate_decay_rate=1.0,
        learning_rate_decay_step=100,
        actor_num_blocks=1,
        actor_hidden_dim=16,
        actor_bc_alpha=0.0,
        actor_noise_zeta_mu=2.0,
        actor_noise_zeta_max=4,
        actor_update_period=2,
        critic_num_blocks=1,
        critic_hidden_dim=16,
        critic_num_bins=11,
        critic_min_v=-5.0,
        critic_max_v=5.0,
        critic_target_update_tau=0.01,
        temp_initial_value=0.01,
        temp_target_sigma=0.15,
        temp_target_entropy=None,
        gamma=0.99,
        n_step=1,
        use_compile=False,
        compile_mode="default",
        use_amp=False,
        load_optimizer=True,
        load_reward_normalizer=False,
        buffer_obs_dtype=None,
        buffer_optimize_memory_usage=True,
        maxinfo_enabled=True,
        maxinfo_num_heads=3,
        maxinfo_hidden_dim=16,
        maxinfo_num_hidden_layers=2,
    )
    cfg_kwargs.update(overrides)
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(NUM_ENVS, OBS_DIM), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1, high=1, shape=(NUM_ENVS, ACT_DIM), dtype=np.float32)
    return FlashSACAgent(obs_space, act_space, env_info={}, cfg=FlashSACConfig(**cfg_kwargs))


def _fill_buffer(agent: FlashSACAgent, steps: int = 20) -> None:
    rng = np.random.default_rng(0)
    for _ in range(steps):
        agent.process_transition(
            {
                "observation": rng.normal(size=(NUM_ENVS, OBS_DIM)).astype(np.float32),
                "action": rng.uniform(-1, 1, size=(NUM_ENVS, ACT_DIM)).astype(np.float32),
                "reward": rng.normal(size=NUM_ENVS).astype(np.float32),
                "terminated": np.zeros(NUM_ENVS, dtype=np.float32),
                "truncated": np.zeros(NUM_ENVS, dtype=np.float32),
                "next_observation": rng.normal(size=(NUM_ENVS, OBS_DIM)).astype(np.float32),
            }
        )


# ---------------------------------------------------------------------------
# RunningNormalizer
# ---------------------------------------------------------------------------


def test_running_normalizer_matches_batch_stats() -> None:
    torch.manual_seed(0)
    normalizer = RunningNormalizer(4)
    chunks = [torch.randn(16, 4) * 3.0 + 1.0 for _ in range(5)]
    for chunk in chunks:
        normalizer.update(chunk)
    full = torch.cat(chunks, dim=0)
    assert torch.allclose(normalizer.mean, full.mean(dim=0), atol=1e-4)
    assert torch.allclose(normalizer.std, full.std(dim=0, correction=0), atol=1e-4)
    normalized = normalizer.normalize(full)
    assert torch.allclose(normalized.mean(dim=0), torch.zeros(4), atol=1e-4)


def test_running_normalizer_identity_before_update() -> None:
    normalizer = RunningNormalizer(3)
    x = torch.randn(8, 3)
    assert torch.allclose(normalizer.normalize(x), x)


# ---------------------------------------------------------------------------
# Ensemble dynamics model
# ---------------------------------------------------------------------------


def _make_dynamics(learn_reward: bool = True) -> MaxInfoDynamics:
    torch.manual_seed(0)
    return MaxInfoDynamics(
        obs_dim=OBS_DIM,
        action_dim=ACT_DIM,
        num_heads=3,
        hidden_dim=32,
        num_hidden_layers=2,
        learn_reward=learn_reward,
    )


def test_ensemble_forward_shape_and_head_diversity() -> None:
    model = _make_dynamics()
    x = torch.randn(10, OBS_DIM + ACT_DIM)
    preds = model(x)
    assert preds.shape == (3, 10, OBS_DIM + 1)  # delta-obs + reward heads
    # heads must start distinct, otherwise disagreement degenerates to zero
    assert not torch.allclose(preds[0], preds[1])


def test_ensemble_forward_shape_without_reward() -> None:
    model = _make_dynamics(learn_reward=False)
    preds = model(torch.randn(10, OBS_DIM + ACT_DIM))
    assert preds.shape == (3, 10, OBS_DIM)


def test_ensemble_regression_loss_decreases() -> None:
    model = _make_dynamics()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(64, OBS_DIM + ACT_DIM)
    target = torch.randn(64, OBS_DIM + 1)

    def loss_fn() -> torch.Tensor:
        return model.regression_loss(model(x), target)

    initial = loss_fn().item()
    for _ in range(200):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()
    assert loss_fn().item() < 0.5 * initial


def test_disagreement_higher_off_distribution() -> None:
    model = _make_dynamics()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.manual_seed(1)
    x = torch.rand(256, OBS_DIM + ACT_DIM)  # inputs in [0, 1]
    target = torch.sin(x.sum(dim=-1, keepdim=True)).expand(-1, OBS_DIM + 1).contiguous()
    for _ in range(300):
        optimizer.zero_grad()
        loss = model.regression_loss(model(x), target)
        loss.backward()
        optimizer.step()
    in_dist = model.info_gain(model(x)).mean()
    off_dist = model.info_gain(model(x + 10.0)).mean()
    assert off_dist > in_dist


# ---------------------------------------------------------------------------
# Dyn-scale (beta) auto-tuning
# ---------------------------------------------------------------------------


def _make_dyn_scale() -> Network:
    net = FlashSACTemperature(initial_value=1.0)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-1)
    return Network(network=net, optimizer=optimizer)


def test_dyn_scale_rises_when_gain_below_target() -> None:
    dyn_scale = _make_dyn_scale()
    for _ in range(5):
        update_dyn_scale(
            dyn_scale=dyn_scale,
            info_gain_rows=torch.full((8,), -1.0),
            target_info_gain_rows=torch.full((8,), 1.0),
        )
    assert dyn_scale.network.log_temp.item() > 0.0


def test_dyn_scale_falls_when_gain_above_target() -> None:
    dyn_scale = _make_dyn_scale()
    for _ in range(5):
        update_dyn_scale(
            dyn_scale=dyn_scale,
            info_gain_rows=torch.full((8,), 1.0),
            target_info_gain_rows=torch.full((8,), -1.0),
        )
    assert dyn_scale.network.log_temp.item() < 0.0


def test_dyn_scale_clamped_on_persistent_one_sided_gap() -> None:
    """A persistent gap must not drift beta out of [DYN_SCALE_MIN, DYN_SCALE_MAX]."""
    from flash_rl.agents.flashSAC.maxinfo import DYN_SCALE_MAX, DYN_SCALE_MIN

    for sign, bound in ((-1.0, DYN_SCALE_MAX), (1.0, DYN_SCALE_MIN)):
        dyn_scale = _make_dyn_scale()
        for _ in range(300):
            update_dyn_scale(
                dyn_scale=dyn_scale,
                info_gain_rows=torch.full((8,), sign),
                target_info_gain_rows=torch.full((8,), -sign),
            )
        beta = dyn_scale.network.log_temp.exp().item()
        assert abs(beta - bound) / bound < 1e-3, f"beta={beta} escaped bound={bound}"


# ---------------------------------------------------------------------------
# FlashSACAgent integration
# ---------------------------------------------------------------------------


def test_agent_update_emits_maxinfo_metrics() -> None:
    agent = _make_agent()
    _fill_buffer(agent)
    assert agent.can_start_training()
    info = agent.update()  # update_step 0 -> actor update happens
    for key in (
        "maxinfo/ensemble_loss",
        "maxinfo/info_gain",
        "maxinfo/target_info_gain",
        "maxinfo/next_info_gain",
        "maxinfo/dyn_scale",
        "maxinfo/dyn_scale_loss",
    ):
        assert key in info, f"missing {key}"
        assert np.isfinite(info[key]), f"non-finite {key}"
    assert "actor/loss" in info
    assert np.isfinite(info["actor/loss"])


def _flat_params(net: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).clone() for p in net.parameters()])


def test_agent_update_trains_ensemble_and_targets() -> None:
    agent = _make_agent()
    _fill_buffer(agent)
    assert agent._maxinfo is not None
    ensemble_param = next(iter(agent._maxinfo.ensemble.network.parameters()))
    before_ensemble = ensemble_param.detach().clone()
    before_scale = agent._maxinfo.dyn_scale.network.log_temp.detach().clone()
    before_actor_target = _flat_params(agent._maxinfo.actor_target.network)
    agent.update()
    agent.update()
    assert not torch.allclose(ensemble_param, before_ensemble)
    assert not torch.allclose(agent._maxinfo.dyn_scale.network.log_temp, before_scale)
    # EMA movement is tiny (tau=0.01); exact comparison over all parameters
    assert not torch.equal(_flat_params(agent._maxinfo.actor_target.network), before_actor_target)


def test_agent_fixed_dyn_scale_stays_constant() -> None:
    agent = _make_agent(maxinfo_dyn_scale_auto=False, maxinfo_dyn_scale_init=0.5)
    _fill_buffer(agent)
    assert agent._maxinfo is not None
    before = agent._maxinfo.dyn_scale.network.log_temp.detach().clone()
    info = agent.update()
    assert torch.allclose(agent._maxinfo.dyn_scale.network.log_temp, before)
    assert abs(info["maxinfo/dyn_scale"] - 0.5) < 1e-6


def test_disabled_path_has_no_maxinfo() -> None:
    agent = _make_agent(maxinfo_enabled=False)
    assert agent._maxinfo is None
    _fill_buffer(agent)
    info = agent.update()
    assert not any(k.startswith("maxinfo/") for k in info)


def test_save_load_roundtrip(tmp_path: Any, capsys: Any) -> None:
    agent = _make_agent()
    _fill_buffer(agent)
    agent.update()
    assert agent._maxinfo is not None
    ensemble_param = next(iter(agent._maxinfo.ensemble.network.parameters())).detach().clone()
    log_scale = agent._maxinfo.dyn_scale.network.log_temp.detach().clone()
    gain_mean = agent._maxinfo.ensemble.network.gain_normalizer.mean.detach().clone()
    agent.save(str(tmp_path))
    agent2 = _make_agent()
    agent2.load(str(tmp_path))
    assert agent2._maxinfo is not None
    assert torch.allclose(next(iter(agent2._maxinfo.ensemble.network.parameters())), ensemble_param)
    assert torch.allclose(agent2._maxinfo.dyn_scale.network.log_temp, log_scale)
    assert torch.allclose(agent2._maxinfo.ensemble.network.gain_normalizer.mean, gain_mean)
    # The ensemble intentionally has no LR scheduler; loading it must not warn.
    assert "Skipping scheduler load" not in capsys.readouterr().out
