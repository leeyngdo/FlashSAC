from typing import Any, Optional, cast

import torch
from torch.amp.grad_scaler import GradScaler

from flash_rl.agents.utils.network import Network
from flash_rl.buffers import Batch
from flash_rl.common.distributed import all_reduce_grads_average_


def add_prefix_to_keys(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}/{k}": v for k, v in d.items()}


@torch.compile
def _select_min_q_log_probs(
    next_qs: torch.Tensor,  # (2, B)
    next_q_log_probs: torch.Tensor,  # (2, B, num_bins)
) -> torch.Tensor:
    """Select log-probs from the min-Q critic and return the next-obs half (batch, num_bins)."""
    num_bins = next_q_log_probs.shape[-1]
    min_indices = next_qs.argmin(dim=0)  # (B,)
    selected = torch.gather(
        next_q_log_probs,
        dim=0,
        index=min_indices[None, :, None].expand(1, -1, num_bins),
    )[0]  # (B, num_bins)
    return selected


@torch.compile
def _compute_categorical_td_target(
    target_log_probs: torch.Tensor,  # (B, num_bins)
    reward: torch.Tensor,  # (B,)
    done: torch.Tensor,  # (B,)
    actor_entropy: torch.Tensor,  # (B,)
    gamma: float,
    num_bins: int,
    min_v: float,
    max_v: float,
) -> torch.Tensor:
    batch_size = reward.shape[0]

    reward = reward.reshape(-1, 1)
    done = done.reshape(-1, 1)
    actor_entropy = actor_entropy.reshape(-1, 1)

    # Compute target value buckets
    bin_width = (max_v - min_v) / (num_bins - 1)
    bin_values = torch.linspace(
        min_v, max_v, num_bins, device=target_log_probs.device, dtype=target_log_probs.dtype
    ).view(1, -1)

    # target_bin_values
    target_bin_values = reward + gamma * (bin_values - actor_entropy) * (1.0 - done)
    target_bin_values = torch.clamp(target_bin_values, min_v, max_v)

    # update indices
    b = (target_bin_values - min_v) / bin_width
    lower = torch.floor(b).long()
    upper = torch.clamp(lower + 1, 0, num_bins - 1)

    frac = b - lower.float()

    # Compute target probabilities using exp
    target_probs_exp = target_log_probs.exp()
    m_l = target_probs_exp * (1.0 - frac)
    m_u = target_probs_exp * frac

    # Allocate output tensor
    target_probs = torch.zeros(batch_size, num_bins, dtype=target_probs_exp.dtype, device=target_probs_exp.device)

    # Scatter operations
    target_probs.scatter_add_(1, lower, m_l)
    target_probs.scatter_add_(1, upper, m_u)

    return target_probs


def update_actor(
    actor: Network,
    critic: Network,
    temperature: Network,
    batch: Batch,
    bc_alpha: float,
    device: torch.device,
    use_amp: bool,
    grad_scaler: Optional[GradScaler],
    num_agents: int = 1,
    leader_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Update actor network.

    Args:
        actor: Actor network.
        critic: Critic network.
        temperature: Temperature network.
        batch: Batch of transitions.
        bc_alpha: BC regularization coefficient.
        device: Device to use.
        use_amp: Whether to use automatic mixed precision.
        grad_scaler: GradScaler for FP16 AMP.
    """

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        agent_ids = cast(Optional[torch.Tensor], batch.get("agent_id"))
        actor_obs_all = torch.cat([batch["actor_observation"], batch["actor_next_observation"]], dim=0)  # type: ignore
        actor_agent_ids_all = torch.cat([agent_ids, agent_ids], dim=0) if agent_ids is not None else None
        actions_all, info = actor(
            observations=actor_obs_all,
            training=True,
            agent_ids=actor_agent_ids_all,
        )
        log_probs_all = info["log_prob"]

        actions = torch.chunk(actions_all, 2, dim=0)[0]
        log_probs = torch.chunk(log_probs_all, 2, dim=0)[0]

        # Disable critic gradients to prevent CUDA graph overwriting
        critic.network.requires_grad_(False)
        qs, _ = critic(
            observations=batch["observation"],
            actions=actions,
            training=False,
            agent_ids=agent_ids,
        )
        q = torch.minimum(qs[0], qs[1])
        critic.network.requires_grad_(True)

        temp_value = temperature(agent_ids).detach() if agent_ids is not None else temperature().detach()
        actor_loss = (log_probs * temp_value - q).mean()

        if bc_alpha > 0:
            # https://arxiv.org/abs/2306.02451
            q_abs = torch.abs(q).mean().detach()
            bc_loss = ((actions - batch["action"]) ** 2).mean()
            actor_loss = actor_loss + bc_alpha * q_abs * bc_loss

        entropy_samples = -log_probs
        entropy = entropy_samples.mean()
        mean_action = actions.mean()

        agent_entropy_info = {}
        if num_agents > 1 and agent_ids is not None:
            leader_mask = agent_ids == leader_id
            if leader_mask.any():
                agent_entropy_info["entropy_leader"] = entropy_samples[leader_mask].mean()
            for agent_id in range(num_agents):
                agent_mask = agent_ids == agent_id
                if agent_mask.any():
                    agent_entropy_info[f"entropy_agent_{agent_id}"] = entropy_samples[agent_mask].mean()

    # Gradient step
    assert actor.optimizer is not None
    actor.optimizer.zero_grad(set_to_none=True)
    if use_amp:
        assert grad_scaler is not None
        grad_scaler.scale(actor_loss).backward()
        # Average scaled grads before unscale so an AMP overflow on any rank is
        # detected by every rank.
        all_reduce_grads_average_(actor.optimizer)
        grad_scaler.unscale_(actor.optimizer)
        grad_scaler.step(actor.optimizer)
        grad_scaler.update()
    else:
        actor_loss.backward()
        all_reduce_grads_average_(actor.optimizer)
        actor.optimizer.step()

    # LR scheduler
    if actor.scheduler is not None:
        actor.scheduler.step()

    # Weight Normalization
    # NOTE: Make sure you finish all computation before this (e.g., computing info values)
    actor.normalize_parameters()

    update_info = {
        "loss": actor_loss,
        "entropy": entropy,
        "mean_action": mean_action,
        **agent_entropy_info,
    }
    update_info = add_prefix_to_keys(update_info, "actor")
    update_info["_temperature_entropy"] = entropy_samples.detach()
    if agent_ids is not None:
        update_info["_temperature_agent_ids"] = agent_ids.detach()

    return update_info


def update_critic(
    actor: Network,
    critic: Network,
    target_critic: Network,
    temperature: Network,
    batch: Batch,
    min_v: float,
    max_v: float,
    num_bins: int,
    gamma: float,
    n_step: int,
    device: torch.device,
    use_amp: bool,
    grad_scaler: Optional[GradScaler],
) -> dict[str, torch.Tensor]:
    """Update critic network.

    Args:
        actor: Actor network.
        critic: Critic network.
        target_critic: Target critic network.
        temperature: Temperature network.
        batch: Batch of transitions.
        min_v: Minimum value for categorical distribution.
        max_v: Maximum value for categorical distribution.
        num_bins: Number of bins for categorical distribution.
        gamma: Discount factor.
        n_step: N-step return.
        device: Device to use.
        use_amp: Whether to use automatic mixed precision.
        grad_scaler: GradScaler for FP16 AMP.
    """

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        # Compute target values
        with torch.no_grad():
            # SAPG: condition the (target) critic and next-action policy on the policy id selected
            # by the sampler. Followers only receive their own collected rows; the leader receives
            # globally sampled rows retagged with the leader id.
            agent_ids = cast(Optional[torch.Tensor], batch.get("agent_id"))
            next_actions, info = actor(
                observations=batch["actor_next_observation"],
                training=False,
                agent_ids=agent_ids,
            )
            # Clone variables to prevent overwriting
            next_actions = next_actions.clone()
            next_actor_log_probs = info["log_prob"].clone()

            temp_value = temperature(agent_ids) if agent_ids is not None else temperature()

            next_actor_entropy = temp_value * next_actor_log_probs
            obs_all = torch.cat([batch["observation"], batch["next_observation"]], dim=0)  # type: ignore
            act_all = torch.cat([batch["action"], next_actions], dim=0)  # type: ignore
            ids_all = torch.cat([agent_ids, agent_ids], dim=0) if agent_ids is not None else None

            # qs_all: (2, 2*B)
            # q_infos_all['log_probs']: (2, 2*B, num_bins)
            qs_all, q_infos_all = target_critic(
                observations=obs_all,
                actions=act_all,
                training=True,
                agent_ids=ids_all,
            )
            next_qs = qs_all.chunk(2, dim=1)[1]
            next_q_log_probs = q_infos_all["log_prob"].chunk(2, dim=1)[1]
            next_q_log_probs = _select_min_q_log_probs(next_qs, next_q_log_probs)

            # Compute target probs
            target_probs = _compute_categorical_td_target(
                target_log_probs=next_q_log_probs,
                reward=batch["reward"],  # type: ignore
                done=batch["terminated"],  # type: ignore
                actor_entropy=next_actor_entropy,
                gamma=gamma**n_step,
                num_bins=num_bins,
                min_v=min_v,
                max_v=max_v,
            )
            max_entropy_bonus = next_actor_entropy.max()

        # Compute predicted q-value
        pred_qs_all, pred_q_infos = critic(
            observations=obs_all,
            actions=act_all,
            training=True,
            agent_ids=ids_all,
        )
        pred_log_probs = torch.chunk(pred_q_infos["log_prob"], 2, dim=1)[0]

        ce_loss = -(target_probs.unsqueeze(0) * pred_log_probs).sum(dim=-1)  # (2, B)
        critic_loss = ce_loss.mean()

    # Gradient step
    assert critic.optimizer is not None
    critic.optimizer.zero_grad(set_to_none=True)
    if use_amp:
        assert grad_scaler is not None
        grad_scaler.scale(critic_loss).backward()  # type: ignore
        # Average scaled grads before unscale so an AMP overflow on any rank is
        # detected by every rank.
        all_reduce_grads_average_(critic.optimizer)
        grad_scaler.unscale_(critic.optimizer)
        grad_scaler.step(critic.optimizer)
        grad_scaler.update()
    else:
        critic_loss.backward()  # type: ignore
        all_reduce_grads_average_(critic.optimizer)
        critic.optimizer.step()

    # LR scheduler
    if critic.scheduler is not None:
        critic.scheduler.step()

    # Weight Normalization
    critic.normalize_parameters()

    update_info = {
        "loss": critic_loss,
        "max_entropy_bonus": max_entropy_bonus,
    }
    update_info = add_prefix_to_keys(update_info, "critic")

    return update_info


@torch.no_grad()
def update_target_network(
    target_network: Network,
) -> dict[str, torch.Tensor]:
    # Use prepared/compiled EMA function for update
    target_network.ema_update_parameters()
    info: dict[str, torch.Tensor] = {}
    return info


def update_temperature(
    temperature: Network,
    entropy: torch.Tensor,
    target_entropy: float,
    agent_ids: Optional[torch.Tensor] = None,
    target_entropy_offsets: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Update temperature network.

    Args:
        temperature: Temperature network.
        entropy: Current entropy value(s).
        target_entropy: Target entropy value.
        agent_ids: Training agent id per entropy sample. When present, each sample updates
            only that agent's temperature.
        target_entropy_offsets: Per-agent additive offsets on ``target_entropy`` (SAPG
            diversity grading: leader 0, followers > 0). Indexed by ``agent_ids``; ignored
            when either is None.
    """

    temperature_value = temperature().clone()
    selected_temperature = temperature(agent_ids) if agent_ids is not None else temperature_value
    if agent_ids is not None and target_entropy_offsets is not None:
        selected_target = target_entropy + target_entropy_offsets[agent_ids]
    else:
        selected_target = target_entropy
    temperature_loss = (selected_temperature * (entropy.detach() - selected_target)).mean()

    assert temperature.optimizer is not None
    temperature.optimizer.zero_grad(set_to_none=True)
    temperature_loss.backward()
    all_reduce_grads_average_(temperature.optimizer)
    temperature.optimizer.step()
    if temperature.scheduler is not None:
        temperature.scheduler.step()

    update_info = {
        "value": temperature_value.mean(),
        "loss": temperature_loss,
    }
    if temperature_value.numel() > 1:
        for agent_id, value in enumerate(temperature_value):
            update_info[f"value_agent_{agent_id}"] = value
    update_info = add_prefix_to_keys(update_info, "temperature")

    return update_info
