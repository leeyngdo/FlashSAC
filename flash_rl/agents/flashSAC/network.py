import math
from typing import Optional

import torch
import torch.nn as nn

from flash_rl.agents.flashSAC.layer import (
    EnsembleCategoricalValue,
    EnsembleFlashSACBlock,
    EnsembleFlashSACEmbedder,
    EnsembleUnitRMSNorm,
    FlashSACBlock,
    FlashSACEmbedder,
    NormalTanhPolicy,
    UnitRMSNorm,
)


class FlashSACActor(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        action_bias: torch.Tensor,
        action_range: torch.Tensor,
        num_agents: int = 1,
        agent_latent_dim: int = 0,
    ):
        super().__init__()
        # SAPG: a per-agent learned latent (phi) is prepended to the observation so that one shared
        # backbone realizes M different policies. agent_latent_dim == 0 disables this entirely and
        # the actor is identical to vanilla SAC (single policy).
        self.num_agents = num_agents
        self.agent_latent_dim = agent_latent_dim
        if agent_latent_dim > 0:
            self.agent_latents = nn.Parameter(torch.randn(num_agents, agent_latent_dim))
            embed_input_dim = input_dim + agent_latent_dim
        else:
            self.register_parameter("agent_latents", None)
            embed_input_dim = input_dim

        self.embedder = FlashSACEmbedder(input_dim=embed_input_dim, hidden_dim=hidden_dim)
        self.encoder = nn.ModuleList([FlashSACBlock(hidden_dim) for _ in range(num_blocks)])
        self.post_norm = UnitRMSNorm(hidden_dim)
        self.predictor = NormalTanhPolicy(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            action_bias=action_bias,
            action_range=action_range,
        )

    def _augment(self, observations: torch.Tensor, agent_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """Prepend the per-agent latent selected by ``agent_ids`` to the observation (no-op when disabled)."""
        if self.agent_latents is None:
            return observations
        assert agent_ids is not None, "agent_ids is required when agent latents are enabled"
        phi = self.agent_latents[agent_ids]  # [B, agent_latent_dim]
        return torch.cat((phi, observations), dim=-1).contiguous()

    def get_mean_and_std(
        self,
        observations: torch.Tensor,
        training: bool,
        agent_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._augment(observations, agent_ids)
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        mean, std = self.predictor.get_mean_and_std(x, training)
        return mean, std

    def forward(
        self,
        observations: torch.Tensor,
        training: bool,
        agent_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self._augment(observations, agent_ids)
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        actions, info = self.predictor(x, training)
        return actions, info


class FlashSACDoubleCritic(nn.Module):
    """
    Double-Q for Clipped Double Q-learning.
    https://arxiv.org/pdf/1802.09477v3

    Fuses N parallel critic networks into single batched operations.
    All internal computation uses (N, batch, dim) tensor layout.
    """

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        num_qs: int = 2,
        num_agents: int = 1,
        agent_latent_dim: int = 0,
    ):
        super().__init__()
        self.num_qs = num_qs
        # SAPG: a per-agent learned latent (phi) is prepended to [obs, action] so one shared critic
        # represents all M agents' Q-functions Q^{pi_i}. agent_latent_dim == 0 disables it (vanilla
        # SAC). This latent table is the critic's own (separate from the actor's).
        self.num_agents = num_agents
        self.agent_latent_dim = agent_latent_dim
        if agent_latent_dim > 0:
            self.agent_latents = nn.Parameter(torch.randn(num_agents, agent_latent_dim))
            embed_input_dim = input_dim + agent_latent_dim
        else:
            self.register_parameter("agent_latents", None)
            embed_input_dim = input_dim

        self.embedder = EnsembleFlashSACEmbedder(num_qs, embed_input_dim, hidden_dim)
        self.encoder = nn.ModuleList([EnsembleFlashSACBlock(num_qs, hidden_dim) for _ in range(num_blocks)])
        self.post_norm = EnsembleUnitRMSNorm(num_qs, hidden_dim)
        self.predictor = EnsembleCategoricalValue(
            num_ensemble=num_qs,
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        training: bool,
        agent_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = torch.cat((observations, actions), dim=-1)  # [B, obs+act]
        if self.agent_latents is not None:
            assert agent_ids is not None, "agent_ids is required when agent latents are enabled"
            x = torch.cat((self.agent_latents[agent_ids], x), dim=-1)  # [B, phi+obs+act]
        x = x.contiguous()
        x = x.unsqueeze(0).expand(self.num_qs, -1, -1).contiguous()  # [num_qs, B, in_dim]
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        qs, infos = self.predictor(x, training)
        return qs, infos


class FlashSACTemperature(nn.Module):
    def __init__(self, initial_value: float = 0.01, num_agents: int = 1):
        super().__init__()
        self.num_agents = num_agents
        self.log_temp = nn.Parameter(torch.full((num_agents,), math.log(initial_value), dtype=torch.float32))

    def forward(self, agent_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        temp = torch.exp(self.log_temp)
        if agent_ids is None:
            return temp
        return temp[agent_ids]
