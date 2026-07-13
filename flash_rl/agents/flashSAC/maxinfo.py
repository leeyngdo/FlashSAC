"""MaxInfoRL (MaxInfoSAC) modules for FlashSAC.

Directed exploration via information gain about the environment dynamics, approximated
by the disagreement of an ensemble of forward-dynamics models.
Paper: https://arxiv.org/abs/2412.12098
Reference implementation: https://github.com/sukhijab/maxinforl_torch
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from flash_rl.agents.flashSAC.network import FlashSACActor, FlashSACTemperature
from flash_rl.agents.utils.network import Network
from flash_rl.agents.utils.reward_normalization import RewardNormalizer
from flash_rl.agents.utils.scheduler import warmup_cosine_decay_scheduler
from flash_rl.common.distributed import all_reduce_grads_average_

if TYPE_CHECKING:
    from flash_rl.agents.flashSAC.agent import FlashSACConfig

EPS = 1e-6


class RunningNormalizer(nn.Module):
    """Streaming per-dimension mean/std (population) with in-place buffer updates.

    The small prior count mirrors the RND running-stat normalizer and prevents the
    first near-constant batch from collapsing std to EPS. Buffers are updated with
    copy_ so their addresses stay stable for CUDA-graph replays that read them.
    """

    mean: torch.Tensor
    std: torch.Tensor
    count: torch.Tensor

    def __init__(self, dim: int, epsilon: float = 1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("count", torch.tensor(float(epsilon)))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float()
        batch_count = torch.tensor(float(x.shape[0]), dtype=x.dtype, device=x.device)
        batch_sum = x.sum(dim=0)
        batch_sumsq = x.square().sum(dim=0)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_sumsq, op=dist.ReduceOp.SUM)

        batch_mean = batch_sum / batch_count
        batch_var = torch.clamp(batch_sumsq / batch_count - batch_mean.square(), min=0.0)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / total
        s_n = (
            self.std.square() * self.count + batch_var * batch_count + delta.square() * self.count * batch_count / total
        )
        self.mean.copy_(new_mean)
        self.std.copy_(torch.clamp(torch.sqrt(s_n / total), min=EPS))
        self.count.copy_(total)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class EnsembleLinear(nn.Module):
    """Fused linear layer applied per ensemble head: (E, B, in) -> (E, B, out).

    Init matches the reference EnsembleMLP's per-head nn.Linear defaults:
    kaiming-uniform(a=sqrt(5)) weights, U(+-1/sqrt(fan_in)) bias.
    """

    def __init__(self, num_heads: int, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_heads, output_dim, input_dim))
        self.bias = nn.Parameter(torch.empty(num_heads, output_dim))
        for i in range(num_heads):
            nn.init.kaiming_uniform_(self.weight.data[i], a=math.sqrt(5))
        bound = 1.0 / math.sqrt(input_dim)
        nn.init.uniform_(self.bias.data, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ebi,eoi->ebo", x, self.weight) + self.bias.unsqueeze(1)


class MaxInfoDynamics(nn.Module):
    """Ensemble forward-dynamics model with running input/target/gain normalizers.

    Predicts the normalized [next_obs - obs (, reward)] from (obs, action). The
    disagreement (log epistemic variance) across heads approximates the information
    gain of a transition (learn_std=False path of the reference implementation).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_hidden_layers: int,
        learn_reward: bool,
    ):
        super().__init__()
        if num_heads < 2:
            raise ValueError("maxinfo_num_heads must be >= 2 for ensemble disagreement")
        self.obs_dim = obs_dim
        self.num_heads = num_heads
        self.learn_reward = learn_reward
        input_dim = obs_dim + action_dim
        target_dim = obs_dim + (1 if learn_reward else 0)

        layers = [EnsembleLinear(num_heads, input_dim, hidden_dim)]
        for _ in range(num_hidden_layers - 1):
            layers.append(EnsembleLinear(num_heads, hidden_dim, hidden_dim))
        self.hidden = nn.ModuleList(layers)
        self.head = EnsembleLinear(num_heads, hidden_dim, target_dim)

        self.input_normalizer = RunningNormalizer(input_dim)
        self.target_normalizer = RunningNormalizer(target_dim)
        # Reference entropy_normalizer: running z-score of the policy-gain rows,
        # removing the log-variance gain's -log(EPS)-anchored additive baseline.
        self.entropy_normalizer = RunningNormalizer(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, obs+action) -> per-head predictions (E, B, target_dim)."""
        x = self.input_normalizer.normalize(x)
        x = x.unsqueeze(0).expand(self.num_heads, -1, -1)
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.head(x)  # type: ignore[no-any-return]

    def _split_mean(self, per_dim: torch.Tensor) -> torch.Tensor:
        """Mean over target dims, weighting the reward head like the full obs block.

        Mirrors the reference's per-key 'mean' aggregation over {next_obs, reward}.
        """
        if self.learn_reward:
            return 0.5 * per_dim[..., : self.obs_dim].mean(dim=-1) + 0.5 * per_dim[..., self.obs_dim :].mean(dim=-1)
        return per_dim.mean(dim=-1)

    def regression_loss(
        self, preds: torch.Tensor, target: torch.Tensor, valid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """MSE of per-head predictions against the normalized raw target.

        `valid` (B,) masks rows out of the mean (episode-boundary transitions).
        """
        target = self.target_normalizer.normalize(target)
        squared_error = (preds - target.unsqueeze(0)).square()  # (E, B, D)
        rows = self._split_mean(squared_error)  # (E, B)
        if valid is None:
            return rows.mean()
        v = valid.unsqueeze(0)
        return (rows * v).sum() / torch.clamp(v.sum() * self.num_heads, min=1.0)

    def info_gain(self, preds: torch.Tensor) -> torch.Tensor:
        """Unnormalized information-gain rows (B,) from head disagreement.

        The reference's learn_std=False entropy gain (EnsembleMLP.get_disagreement
        + MaxInfoSAC.get_intrinsic_reward): mean over dims of log(EPS + epistemic
        variance), shifted by -log(EPS) so zero disagreement maps to 0 — i.e.
        mean_j log(1 + sigma_j^2 / EPS), the Gaussian information gain with the
        aleatoric variance pinned at EPS. Carries a slowly-decaying additive
        baseline; the entropy z-score in policy_info_gain removes it.
        """
        epistemic_var = preds.var(dim=0)  # unbiased over heads, like the reference's std()
        return self._split_mean(torch.log(EPS + epistemic_var)) - math.log(EPS)


@dataclass
class MaxInfoModules:
    """Bundle of the MaxInfoSAC-specific training components."""

    ensemble: Network
    dyn_scale: Network
    actor_target: Network
    dyn_scale_auto: bool
    # Return-budget scaling of the gain (RND-style): divides the non-negative
    # gain by max(std of its DISCOUNTED return, G_r_max / G_max) so the intrinsic
    # return lives inside the same ±G_max support budget as the task return.
    # Stats update in process_transition once training starts.
    gain_normalizer: RewardNormalizer
    # Stage-2 on/off: when False the bonus is the stage-1 z alone (pure reference
    # sigma units) — ablation knob for the return-budget scaling.
    gain_return_norm: bool = True


def init_maxinfo(
    actor: Network,
    actor_observation_dim: int,
    observation_dim: int,
    action_dim: int,
    action_bias: torch.Tensor,
    action_range: torch.Tensor,
    cfg: "FlashSACConfig",
    device: torch.device,
) -> MaxInfoModules:
    """Build the ensemble, the exploration scale (beta), and the EMA actor target."""
    use_fused = device.type == "cuda" and torch.cuda.is_available()
    # All maxinfo modules follow the agent-wide warmup-cosine lr schedule, like every
    # other optimizer in this codebase (the reference uses constant lrs instead).
    warmup_cosine_decay_lr = warmup_cosine_decay_scheduler(
        init_value=cfg.learning_rate_init,
        peak_value=cfg.learning_rate_peak,
        end_value=cfg.learning_rate_end,
        warmup_steps=cfg.learning_rate_warmup_step,
        decay_steps=cfg.learning_rate_decay_step,
    )

    def _make_scheduler(optimizer: optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: warmup_cosine_decay_lr(step) / cfg.learning_rate_peak,
        )

    # Compile without CUDA graphs: adding these networks to the CUDA-graph pool
    # alongside the actor/critic graphs corrupts cudagraph-trees pool accounting at
    # high env counts ("live storage data ptrs ... not accounted for" on the actor's
    # warmup). Inductor kernels still apply; only graph capture is skipped.
    compile_mode = "default"

    dynamics = MaxInfoDynamics(
        obs_dim=observation_dim,
        action_dim=action_dim,
        num_heads=cfg.maxinfo_num_heads,
        hidden_dim=cfg.maxinfo_hidden_dim,
        num_hidden_layers=cfg.maxinfo_num_hidden_layers,
        learn_reward=cfg.maxinfo_learn_reward,
    ).to(device)
    ensemble_optimizer = optim.Adam(dynamics.parameters(), lr=cfg.learning_rate_peak, fused=use_fused)
    ensemble = Network(
        network=dynamics,
        optimizer=ensemble_optimizer,
        scheduler=_make_scheduler(ensemble_optimizer),
        compile_network=cfg.use_compile,
        compile_mode=compile_mode,
    )

    dyn_scale_net = FlashSACTemperature(cfg.maxinfo_dyn_scale_init).to(device)
    dyn_scale_optimizer: Optional[optim.Adam] = None
    dyn_scale_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    if cfg.maxinfo_dyn_scale_auto:
        dyn_scale_optimizer = optim.Adam(dyn_scale_net.parameters(), lr=cfg.learning_rate_peak, fused=use_fused)
        dyn_scale_scheduler = _make_scheduler(dyn_scale_optimizer)
    dyn_scale = Network(
        network=dyn_scale_net,
        optimizer=dyn_scale_optimizer,
        scheduler=dyn_scale_scheduler,
        compile_network=cfg.use_compile,
        compile_mode=compile_mode,
    )

    actor_target_net = FlashSACActor(
        num_blocks=cfg.actor_num_blocks,
        input_dim=actor_observation_dim,
        hidden_dim=cfg.actor_hidden_dim,
        action_dim=action_dim,
        action_bias=action_bias,
        action_range=action_range,
    ).to(device)
    source_state = {k.removeprefix("_orig_mod."): v for k, v in actor.network.state_dict().items()}
    actor_target_net.load_state_dict(source_state)
    actor_target = Network(
        network=actor_target_net,
        compile_network=cfg.use_compile,
        compile_mode=compile_mode,
        use_weight_normalization=True,
        ema_source=actor,
        # Same tau as the critic target: the reference polyaks actor_target and
        # critic_target together with one shared tau.
        ema_tau=cfg.critic_target_update_tau,
    )

    # Same class as the task RewardNormalizer, fed from the behavior policy's
    # transitions in process_transition like the reward path. Tracks the discounted
    # return of the stage-1 z-scored gain — the currency that enters the actor and
    # the TD target — so stage-2 keeps the intrinsic return inside the same ±G_max
    # support budget as the task return. Caveat of the zero-mean z stream: the
    # G_r_max floor caps the largest observed |return| at the support edge but a
    # never-decaying max can memorize warmup transients.
    gain_normalizer = RewardNormalizer(
        gamma=cfg.gamma,
        G_max=cfg.normalized_G_max,
        load_rms=cfg.load_reward_normalizer,
        device=device,
    )

    return MaxInfoModules(
        ensemble=ensemble,
        dyn_scale=dyn_scale,
        actor_target=actor_target,
        dyn_scale_auto=cfg.maxinfo_dyn_scale_auto,
        gain_normalizer=gain_normalizer,
        gain_return_norm=cfg.maxinfo_gain_return_norm,
    )


def policy_info_gain(
    maxinfo: MaxInfoModules,
    observations: torch.Tensor,
    actions: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Info-gain rows g(s, a) of shape (batch,), computed in float32.

    Normalization is two-stage: (1) z-score with the running stats of the
    policy-gain distribution — the reference's entropy_normalizer, removing the
    log-variance gain's -log(EPS)-anchored additive baseline that any pure
    division would keep (stats update only in actor_info_gains, the reference's
    update position; this path normalizes without updating, like the reference's
    critic-target path) — then (2) divide by the gain RewardNormalizer's
    discounted-return scale so the intrinsic return fits its budget share of the
    task's ±G_max (skipped when gain_return_norm is off: pure reference sigma
    units). Both stages are affine, so gradients flow through `actions` into the
    (caller-frozen) ensemble; the normalizer stats act as constants. Autocast is
    disabled because head variance is precision-fragile in fp16.
    """
    # The compiled wrapper proxies attribute access to the original module.
    model = cast(MaxInfoDynamics, maxinfo.ensemble.network)
    with torch.autocast(device_type=observations.device.type, enabled=False):
        inp = torch.cat([observations.float(), actions.float()], dim=-1)
        gain = model.info_gain(maxinfo.ensemble(inp))
        if not normalize:
            return gain
        z = model.entropy_normalizer.normalize(gain)
        if not maxinfo.gain_return_norm:
            return z
        return maxinfo.gain_normalizer.normalize_rewards(z)


def actor_info_gains(
    maxinfo: MaxInfoModules,
    observations: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(bonus, z) rows for the actor update's stacked current+target policy batch.

    Updates the z-normalizer on these rows first — the reference's update-then-
    normalize position on the same current+target rows (maxinfo_sac.py train()).
    The z rows feed the dyn-scale loss in the reference's σ units; the bonus rows
    add the return-budget scaling and enter the actor objective. Gradients flow
    through `actions` in both outputs.
    """
    model = cast(MaxInfoDynamics, maxinfo.ensemble.network)
    gain = policy_info_gain(maxinfo, observations, actions, normalize=False)
    model.entropy_normalizer.update(gain.detach().reshape(-1, 1))
    z = model.entropy_normalizer.normalize(gain)
    if not maxinfo.gain_return_norm:
        return z, z
    return maxinfo.gain_normalizer.normalize_rewards(z), z


@torch.no_grad()
def update_gain_stats(
    maxinfo: MaxInfoModules,
    observations: torch.Tensor,
    actions: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
) -> None:
    """Accumulate the discounted return of the z-scored gain of behavior transitions.

    The z rows are the currency that enters the actor/critic, so the return stats
    track them (normalize-only here; z stats update in actor_info_gains). Called
    once per interaction step (like RewardNormalizer.update_reward_stats), so the
    per-env discounted accumulator sees temporally ordered transitions — the
    update path only sees shuffled replay rows and cannot maintain it.
    """
    if not maxinfo.gain_return_norm:
        return
    model = cast(MaxInfoDynamics, maxinfo.ensemble.network)
    # Also wait for the first z-stats update (first actor step): before it the
    # z-score is identity, and the raw-scale log-gain rows (~ -log EPS) would
    # permanently inflate the never-decaying G_r_max floor.
    if model.entropy_normalizer.count.item() <= 1.0:
        return
    gain = policy_info_gain(maxinfo, observations, actions, normalize=False)
    z = model.entropy_normalizer.normalize(gain)
    maxinfo.gain_normalizer.update_reward_stats(
        reward=z.detach(),
        terminated=terminated,
        truncated=truncated,
    )


def update_ensemble(
    maxinfo: MaxInfoModules,
    observations: torch.Tensor,  # (B, obs) true state s_t (critic view)
    actions: torch.Tensor,  # (B, act) executed action a_t
    next_observations: torch.Tensor,  # (B, obs) true next state s_{t+1}
    rewards: torch.Tensor,  # (B,) raw 1-step reward r_t
    valid: torch.Tensor,  # (B,) 1 if the row's target supervises (see agent)
) -> dict[str, torch.Tensor]:
    """One gradient step: 1-step regression of the ensemble on raw replay transitions.

    The buffer reads s_{t+1} by index (slot i + num_envs), so episode-boundary rows
    arrive with a target that belongs to the next episode — they are masked out of
    both the normalizer stats and the loss.
    """
    model = cast(MaxInfoDynamics, maxinfo.ensemble.network)

    with torch.no_grad():
        # Sanitize invalid rows' targets (reset observations, or anything else) so
        # the masked loss never touches them even through 0*x arithmetic — a NaN
        # or wild value times a zero mask still poisons the sum/backward otherwise.
        valid_b = valid.bool()
        next_observations = torch.where(valid_b.unsqueeze(-1), next_observations, observations)
        rewards = torch.where(valid_b, rewards, torch.zeros_like(rewards))

        inp_rows = torch.cat([observations, actions], dim=-1).float()
        tgt_rows = (next_observations - observations).float()
        if model.learn_reward:
            tgt_rows = torch.cat([tgt_rows, rewards.reshape(-1, 1).float()], dim=-1)
        if bool(valid_b.any()):
            model.input_normalizer.update(inp_rows[valid_b])
            model.target_normalizer.update(tgt_rows[valid_b])

    with torch.autocast(device_type=observations.device.type, enabled=False):
        preds = maxinfo.ensemble(inp_rows)  # (E, B, D)
        loss = model.regression_loss(preds, tgt_rows, valid=valid.float())

    assert maxinfo.ensemble.optimizer is not None
    maxinfo.ensemble.optimizer.zero_grad(set_to_none=True)
    loss.backward()  # type: ignore
    all_reduce_grads_average_(maxinfo.ensemble.optimizer)
    maxinfo.ensemble.optimizer.step()
    if maxinfo.ensemble.scheduler is not None:
        maxinfo.ensemble.scheduler.step()

    return {"maxinfo/ensemble_loss": loss.detach()}


def update_dyn_scale(
    dyn_scale: Network,
    info_gain_rows: torch.Tensor,
    target_info_gain_rows: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Auto-tune the exploration scale beta against the delayed policy's info gain.

    Matches the reference log-space loss: beta falls when the current policy already
    gains more information than the EMA target policy.
    """
    value = dyn_scale().detach().clone()
    log_scale = cast(FlashSACTemperature, dyn_scale.network).log_temp
    loss = (log_scale * (info_gain_rows.detach() - target_info_gain_rows.detach())).mean()

    assert dyn_scale.optimizer is not None
    dyn_scale.optimizer.zero_grad(set_to_none=True)
    loss.backward()  # type: ignore[no-untyped-call]
    all_reduce_grads_average_(dyn_scale.optimizer)
    dyn_scale.optimizer.step()
    if dyn_scale.scheduler is not None:
        dyn_scale.scheduler.step()

    return {"maxinfo/dyn_scale": value.mean(), "maxinfo/dyn_scale_loss": loss}
