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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from flash_rl.agents.flashSAC.network import FlashSACActor, FlashSACTemperature
from flash_rl.agents.utils.network import Network
from flash_rl.agents.utils.scheduler import warmup_cosine_decay_scheduler
from flash_rl.common.distributed import all_reduce_grads_average_

if TYPE_CHECKING:
    from flash_rl.agents.flashSAC.agent import FlashSACConfig

EPS = 1e-6

# The beta auto-tuner is a pure integrator on E[g - g_target]; any persistent tiny bias
# in that gap drifts log-beta without bound (observed in BOTH directions at the
# 10G-step scale: collapse to 0 and explosion past 1e7, dragging the actor loss and TD
# targets with it). The reference has no guard — its 1M-step runs never integrate long
# enough to expose this. Clamp keeps the bonus bounded in a usable range.
DYN_SCALE_MIN = 1e-2
DYN_SCALE_MAX = 10.0


class RunningNormalizer(nn.Module):
    """Streaming per-dimension mean/std (population) with in-place buffer updates.

    Buffers are updated with copy_ so their addresses stay stable for CUDA-graph
    replays that read them. Stats are per-rank (not synchronized across data-parallel
    ranks), matching how batch-norm running stats are treated in this codebase.
    """

    mean: torch.Tensor
    std: torch.Tensor
    count: torch.Tensor

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float()
        batch_count = x.shape[0]
        total = self.count + batch_count
        new_mean = (self.mean * self.count + x.sum(dim=0)) / total
        # Chan et al. parallel-merge of the sum of squared deviations.
        s_n = (
            self.std.square() * self.count
            + (x - new_mean).square().sum(dim=0)
            + self.count * (self.mean - new_mean).square()
        )
        self.mean.copy_(new_mean)
        self.std.copy_(torch.clamp(torch.sqrt(s_n / total), min=EPS))
        self.count.copy_(total)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class EnsembleLinear(nn.Module):
    """Fused linear layer applied per ensemble head: (E, B, in) -> (E, B, out)."""

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
        self.gain_normalizer = RunningNormalizer(1)

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

    def regression_loss(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE of per-head predictions against the normalized raw target."""
        target = self.target_normalizer.normalize(target)
        squared_error = (preds - target.unsqueeze(0)).square()  # (E, B, D)
        return self._split_mean(squared_error).mean()

    def info_gain(self, preds: torch.Tensor) -> torch.Tensor:
        """Unnormalized information gain rows (B,) from head disagreement."""
        epistemic_var = preds.std(dim=0).square()  # sample variance over heads, as in the reference
        log_var = torch.log(EPS + epistemic_var)
        return self._split_mean(log_var) - math.log(EPS)


@dataclass
class MaxInfoModules:
    """Bundle of the MaxInfoSAC-specific training components."""

    ensemble: Network
    dyn_scale: Network
    actor_target: Network
    dyn_scale_auto: bool


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

    return MaxInfoModules(
        ensemble=ensemble,
        dyn_scale=dyn_scale,
        actor_target=actor_target,
        dyn_scale_auto=cfg.maxinfo_dyn_scale_auto,
    )


def policy_info_gain(
    maxinfo: MaxInfoModules,
    observations: torch.Tensor,
    actions: torch.Tensor,
    update_stats: bool,
) -> torch.Tensor:
    """Normalized info-gain rows g(s, a) of shape (batch,), computed in float32.

    Gradients flow through `actions` into the (caller-frozen) ensemble; the running
    gain stats act as constants. Autocast is disabled because head variance is
    precision-fragile in fp16.
    """
    # The compiled wrapper proxies attribute access to the original module.
    model = cast(MaxInfoDynamics, maxinfo.ensemble.network)
    with torch.autocast(device_type=observations.device.type, enabled=False):
        inp = torch.cat([observations.float(), actions.float()], dim=-1)
        gain = model.info_gain(maxinfo.ensemble(inp))
        if update_stats:
            model.gain_normalizer.update(gain.unsqueeze(-1))
        return model.gain_normalizer.normalize(gain.unsqueeze(-1)).squeeze(-1)


def update_ensemble(maxinfo: MaxInfoModules, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """One MSE gradient step of the dynamics ensemble on the replay batch."""
    model = cast(MaxInfoDynamics, maxinfo.ensemble.network)
    with torch.no_grad():
        observation = batch["observation"].float()
        inp = torch.cat([observation, batch["action"].float()], dim=-1)
        target = batch["next_observation"].float() - observation
        if model.learn_reward:
            target = torch.cat([target, batch["reward"].float().reshape(-1, 1)], dim=-1)
        model.input_normalizer.update(inp)
        model.target_normalizer.update(target)

    with torch.autocast(device_type=inp.device.type, enabled=False):
        preds = maxinfo.ensemble(inp)
        loss = model.regression_loss(preds, target)

    assert maxinfo.ensemble.optimizer is not None
    maxinfo.ensemble.optimizer.zero_grad(set_to_none=True)
    loss.backward()  # type: ignore
    all_reduce_grads_average_(maxinfo.ensemble.optimizer)
    maxinfo.ensemble.optimizer.step()
    if maxinfo.ensemble.scheduler is not None:
        maxinfo.ensemble.scheduler.step()

    return {"maxinfo/ensemble_loss": loss}


def update_dyn_scale(
    dyn_scale: Network,
    info_gain_rows: torch.Tensor,
    target_info_gain_rows: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Auto-tune the exploration scale beta against the delayed policy's info gain.

    Same fixed point and sign as the reference's log-space loss, expressed in the
    value form used by this codebase's temperature update: beta falls when the
    current policy already gains more information than the EMA target policy.
    """
    value = dyn_scale().clone()
    loss = (value * (info_gain_rows.detach() - target_info_gain_rows.detach())).mean()

    assert dyn_scale.optimizer is not None
    dyn_scale.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    all_reduce_grads_average_(dyn_scale.optimizer)
    dyn_scale.optimizer.step()
    if dyn_scale.scheduler is not None:
        dyn_scale.scheduler.step()

    with torch.no_grad():
        log_temp = cast(FlashSACTemperature, dyn_scale.network).log_temp
        log_temp.clamp_(math.log(DYN_SCALE_MIN), math.log(DYN_SCALE_MAX))

    return {"maxinfo/dyn_scale": value.mean(), "maxinfo/dyn_scale_loss": loss}
