"""Tests for the MaxInfoRL (MaxInfoSAC) modules and their FlashSAC integration."""

import math
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from flash_rl.agents.flashSAC.agent import FlashSACAgent, FlashSACConfig
from flash_rl.agents.flashSAC.maxinfo import (
    MaxInfoDynamics,
    MaxInfoModules,
    RunningNormalizer,
    actor_info_gains,
    policy_info_gain,
    update_dyn_scale,
    update_ensemble,
)
from flash_rl.agents.flashSAC.network import FlashSACTemperature
from flash_rl.agents.utils.network import Network
from flash_rl.agents.utils.reward_normalization import RewardNormalizer

NUM_ENVS = 8
OBS_DIM = 6
ACT_DIM = 2


def _make_agent(env_info: dict[str, Any] | None = None, **overrides: Any) -> FlashSACAgent:
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
    return FlashSACAgent(obs_space, act_space, env_info=env_info or {}, cfg=FlashSACConfig(**cfg_kwargs))


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


def test_running_normalizer_constant_warmup_does_not_collapse_scale() -> None:
    normalizer = RunningNormalizer(1)
    normalizer.update(torch.full((2048, 1), 100.0))

    normalized = normalizer.normalize(torch.full((2048, 1), 100.001))

    assert normalized.abs().max() < 1.0


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
    in_dist = model.info_gain(model(x))
    off_dist = model.info_gain(model(x + 10.0))
    assert off_dist.mean() > in_dist.mean()
    # Reference log-variance gain is shifted by -log(EPS), so zero disagreement maps to 0.
    assert (in_dist >= 0).all() and (off_dist >= 0).all()


def _make_gain_normalizer() -> RewardNormalizer:
    # G_max mirrors init_maxinfo (normalized_G_max): the G_r_max floor is live.
    return RewardNormalizer(gamma=0.99, G_max=5.0, load_rms=True, device=torch.device("cpu"))


def test_log_var_gain_formula_and_nonnegativity() -> None:
    # Reference learn_std=False gain: mean_j log(EPS + var_j) - log(EPS).
    model = _make_dynamics(learn_reward=False)
    preds = torch.randn(3, 5, OBS_DIM)
    expected = torch.log(1e-6 + preds.var(dim=0)).mean(dim=-1) - math.log(1e-6)
    assert torch.allclose(model.info_gain(preds), expected, atol=1e-5)
    assert (model.info_gain(preds) >= 0).all()
    identical = torch.ones(3, 5, OBS_DIM)
    assert torch.allclose(model.info_gain(identical), torch.zeros(5), atol=1e-4)


def test_gain_two_stage_normalization() -> None:
    modules = _make_modules(learn_reward=False)
    model = modules.ensemble.network
    modules.gain_normalizer.G_rms.var.fill_(4.0)  # stage-2 denominator = 2
    observations = torch.randn(8, OBS_DIM)
    actions = torch.randn(8, ACT_DIM)
    raw = model.info_gain(model(torch.cat([observations, actions], dim=-1)))
    # Critic path normalizes WITHOUT updating the entropy stats (reference), so
    # with untouched stats stage-1 is identity and only stage-2 divides.
    out = policy_info_gain(modules, observations, actions)
    assert model.entropy_normalizer.count.item() < 1.0  # prior count only
    assert torch.allclose(out, raw / 2.0, atol=1e-5)
    # Actor path: update-then-normalize on the same rows (reference position).
    bonus, z = actor_info_gains(modules, observations, actions)
    ref_norm = RunningNormalizer(1)
    ref_norm.update(raw.detach().reshape(-1, 1))
    expected_z = ref_norm.normalize(raw)
    assert model.entropy_normalizer.count.item() > 1.0
    assert torch.allclose(z, expected_z, atol=1e-5)  # beta rows = z-scored gain
    assert torch.allclose(bonus, z / 2.0, atol=1e-5)

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
        "maxinfo/bonus",
        "maxinfo/dyn_scale",
        "maxinfo/dyn_scale_loss",
        "maxinfo/gain_scale",
    ):
        assert key in info, f"missing {key}"
        assert np.isfinite(info[key]), f"non-finite {key}"
    assert "actor/loss" in info
    assert np.isfinite(info["actor/loss"])
    # Batch z-score normalization must not blow up at warmup.
    assert abs(info["maxinfo/info_gain"]) < 1e3
    assert abs(info["maxinfo/next_info_gain"]) < 1e3


def test_gain_stats_gated_until_first_actor_update() -> None:
    agent = _make_agent()
    assert agent._maxinfo is not None
    # Pre-z-stats gains are raw-scale (~ -log EPS, z is identity); feeding them
    # to the return stats would permanently inflate the never-decaying G_r_max
    # floor. Stats must stay untouched until training AND the first actor step
    # (which updates the entropy z stats) have both happened.
    _fill_buffer(agent, steps=4)  # 32 rows < buffer_min_length=64
    assert not agent.can_start_training()
    assert agent._maxinfo.gain_normalizer.G_rms.count.item() == 0.0
    _fill_buffer(agent, steps=8)  # crosses the buffer minimum
    assert agent.can_start_training()
    assert agent._maxinfo.gain_normalizer.G_rms.count.item() == 0.0  # z stats still prior
    agent.update()  # update_step 0 -> actor update -> entropy z stats populated
    _fill_buffer(agent, steps=2)
    assert agent._maxinfo.gain_normalizer.G_rms.count.item() > 0
    assert agent._maxinfo.gain_normalizer.G_r_max.item() != 0.0
    # The floor now reflects centered z returns, not the raw ~13.8 baseline.
    assert agent._maxinfo.gain_normalizer.G_r_max.item() < 10.0


def _flat_params(net: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).clone() for p in net.parameters()])


def test_agent_update_trains_ensemble_and_targets() -> None:
    agent = _make_agent()
    assert agent._maxinfo is not None
    _fill_buffer(agent)
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

def _make_modules(learn_reward: bool = True) -> MaxInfoModules:
    torch.manual_seed(0)
    model = _make_dynamics(learn_reward=learn_reward)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return MaxInfoModules(
        ensemble=Network(network=model, optimizer=optimizer),
        dyn_scale=_make_dyn_scale(),
        actor_target=Network(network=torch.nn.Identity()),
        dyn_scale_auto=False,
        gain_normalizer=_make_gain_normalizer(),
    )


def _make_pairs(B: int = 64, seed: int = 1):
    """Deterministic linear dynamics s' = s + 0.1*A@a, reward = sum(a)."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(ACT_DIM, OBS_DIM, generator=g)
    obs = torch.randn(B, OBS_DIM, generator=g)
    actions = torch.randn(B, ACT_DIM, generator=g)
    next_obs = obs + 0.1 * actions @ A
    rewards = actions.sum(-1)
    return obs, actions, next_obs, rewards


def test_ensemble_update_loss_decreases_on_learnable_dynamics() -> None:
    modules = _make_modules()
    obs, actions, next_obs, rewards = _make_pairs()
    valid = torch.ones(obs.shape[0])
    first = update_ensemble(modules, obs, actions, next_obs, rewards, valid)
    for _ in range(300):
        last = update_ensemble(modules, obs, actions, next_obs, rewards, valid)
    assert last["maxinfo/ensemble_loss"].item() < 0.3 * first["maxinfo/ensemble_loss"].item()
    model = modules.ensemble.network
    assert model.input_normalizer.count.item() > 1  # stats updated from valid rows


def test_ensemble_update_masks_invalid_rows() -> None:
    modules = _make_modules()
    obs, actions, next_obs, rewards = _make_pairs()
    # Corrupt half the batch's targets (episode-boundary rows) and mask them out:
    # the loss must remain finite and unaffected by the garbage.
    B = obs.shape[0]
    next_obs = next_obs.clone()
    next_obs[: B // 2] = float("nan")
    valid = torch.ones(B)
    valid[: B // 2] = 0.0
    info = update_ensemble(modules, obs, actions, next_obs, rewards, valid)
    assert torch.isfinite(info["maxinfo/ensemble_loss"])


def test_agent_ensemble_trains_from_replay() -> None:
    agent = _make_agent()
    assert agent._maxinfo is not None
    _fill_buffer(agent)
    ensemble_param = next(iter(agent._maxinfo.ensemble.network.parameters()))
    before = ensemble_param.detach().clone()
    info = agent.update()
    assert not torch.allclose(ensemble_param, before)
    assert "maxinfo/ensemble_loss" in info and np.isfinite(info["maxinfo/ensemble_loss"])

def test_gain_floor_binds_when_max_return_exceeds_budget() -> None:
    # denominator = max(sqrt(G_var), G_r_max / G_max): with G_r_max=25 and
    # G_max=5 the floor (5) beats sqrt(4)=2, capping the max intrinsic return
    # at the support budget.
    modules = _make_modules(learn_reward=False)
    model = modules.ensemble.network
    modules.gain_normalizer.G_rms.var.fill_(4.0)
    modules.gain_normalizer.G_r_max.fill_(25.0)
    observations = torch.randn(8, OBS_DIM)
    actions = torch.randn(8, ACT_DIM)
    raw = model.info_gain(model(torch.cat([observations, actions], dim=-1)))
    out = policy_info_gain(modules, observations, actions)
    assert torch.allclose(out, raw / 5.0, atol=1e-5)


def test_gain_return_norm_off_returns_reference_z() -> None:
    # With stage-2 disabled the bonus IS the stage-1 z (pure reference sigma units).
    modules = _make_modules()
    modules.gain_return_norm = False
    model = modules.ensemble.network
    modules.gain_normalizer.G_rms.var.fill_(400.0)  # must be ignored
    observations = torch.randn(8, OBS_DIM)
    actions = torch.randn(8, ACT_DIM)
    raw = model.info_gain(model(torch.cat([observations, actions], dim=-1)))
    # Entropy stats untouched -> z is identity on this first call.
    out = policy_info_gain(modules, observations, actions)
    assert torch.allclose(out, raw, atol=1e-6)
    bonus, z = actor_info_gains(modules, observations, actions)
    assert torch.allclose(bonus, z, atol=1e-6)
    assert model.entropy_normalizer.count.item() > 1.0  # actor path updated stats
    assert abs(z.mean().item()) < 0.1  # z-scored: centered

def test_agent_fixed_dyn_scale_stays_constant() -> None:
    agent = _make_agent(maxinfo_dyn_scale_auto=False, maxinfo_dyn_scale_init=0.5)
    _fill_buffer(agent)
    assert agent._maxinfo is not None
    before = agent._maxinfo.dyn_scale.network.log_temp.detach().clone()
    info = agent.update()
    assert torch.allclose(agent._maxinfo.dyn_scale.network.log_temp, before)
    assert abs(info["maxinfo/dyn_scale"] - 0.5) < 1e-6


def test_agent_asymmetric_obs_ensemble_models_critic_view() -> None:
    # FlashSAC lays asymmetric obs out as [actor prefix | privileged rest]; the
    # critic view is the FULL observation, so the ensemble models the full width.
    actor_dim = 4
    agent = _make_agent(
        env_info={"actor_observation_size": (actor_dim,)},
        asymmetric_observation=True,
    )
    assert agent._maxinfo is not None
    assert agent._maxinfo.ensemble.network.obs_dim == OBS_DIM
    _fill_buffer(agent)
    info = agent.update()
    for key in ("maxinfo/ensemble_loss", "maxinfo/info_gain", "maxinfo/next_info_gain"):
        assert key in info and np.isfinite(info[key]), key


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
    gain_G_mean = agent._maxinfo.gain_normalizer.G_rms.mean.detach().clone()
    gain_G_max = agent._maxinfo.gain_normalizer.G_r_max.detach().clone()
    agent.save(str(tmp_path))
    agent2 = _make_agent()
    agent2.load(str(tmp_path))
    assert agent2._maxinfo is not None
    assert torch.allclose(next(iter(agent2._maxinfo.ensemble.network.parameters())), ensemble_param)
    assert torch.allclose(agent2._maxinfo.dyn_scale.network.log_temp, log_scale)
    assert torch.allclose(agent2._maxinfo.gain_normalizer.G_rms.mean, gain_G_mean)
    assert torch.allclose(agent2._maxinfo.gain_normalizer.G_r_max, gain_G_max)
    # The ensemble intentionally has no LR scheduler; loading it must not warn.
    assert "Skipping scheduler load" not in capsys.readouterr().out
