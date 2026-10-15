"""Non-generative action-conditioned JEPA with EMA targets and VICReg.

Inputs are already normalized observations, not raw RAM or hidden opponent
records. Defaults use 96 observable scalars and 16 controller action identities.
"""
from __future__ import annotations

import copy
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LatentHJEPAWorldModel(nn.Module):
    """Encode observable state and imagine action-conditioned latent futures."""

    def __init__(self, state_dim: int = 96, action_dim: int = 16,
                 latent_dim: int = 128, ema_decay: float = 0.996) -> None:
        super().__init__()
        if min(state_dim, action_dim, latent_dim) < 1 or not 0 <= ema_decay < 1:
            raise ValueError("positive dimensions and EMA decay in [0,1) required")
        self.state_dim, self.action_dim, self.latent_dim = state_dim, action_dim, latent_dim
        self.ema_decay = ema_decay
        self.online_encoder = nn.Sequential(nn.Linear(state_dim, latent_dim),
            nn.LayerNorm(latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim))
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)
        self.action_emb = nn.Embedding(action_dim, latent_dim)
        self.predictor = nn.Sequential(nn.Linear(2 * latent_dim, 2 * latent_dim),
            nn.LayerNorm(2 * latent_dim), nn.SiLU(), nn.Linear(2 * latent_dim, latent_dim))

    @torch.no_grad()
    def update_target_ema(self) -> None:
        """Update only target parameters; callers invoke this after each optimizer step."""
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters(), strict=True):
            target.mul_(self.ema_decay).add_(online, alpha=1 - self.ema_decay)

    def encode(self, states: Tensor) -> Tensor:
        if states.shape[-1] != self.state_dim:
            raise ValueError("incorrect observation feature dimension")
        return cast(Tensor, self.online_encoder(states))

    def predict(self, latents: Tensor, actions: Tensor) -> Tensor:
        if actions.dtype != torch.long or torch.any((actions < 0) | (actions >= self.action_dim)):
            raise ValueError("actions must be valid long integer identities")
        embedding = cast(Tensor, self.action_emb(actions))
        return cast(Tensor, self.predictor(torch.cat((latents, embedding), dim=-1)))

    @staticmethod
    def _regularization(values: Tensor) -> tuple[Tensor, Tensor]:
        # Unbiased variance is undefined for singleton batches; population
        # variance keeps inference/training edge cases finite.
        flat = values.reshape(-1, values.shape[-1])
        std = torch.sqrt(flat.var(dim=0, unbiased=False) + 1e-4)
        variance = F.relu(1.0 - std).mean()
        centered = flat - flat.mean(dim=0)
        covariance = centered.T @ centered / max(1, flat.shape[0] - 1)
        off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
        return variance, off_diagonal.square().sum() / flat.shape[-1]

    def forward_loss(self, s_t: Tensor, a_t: Tensor, s_next: Tensor,
                     sim_coeff: float = 25.0, var_coeff: float = 25.0,
                     cov_coeff: float = 1.0) -> tuple[Tensor, dict[str, float]]:
        """Predictive invariance plus variance and covariance on both online branches."""
        z = self.encode(s_t)
        prediction = self.predict(z, a_t)
        with torch.no_grad():
            target = cast(Tensor, self.target_encoder(s_next))
        similarity = F.mse_loss(prediction, target)
        v1, c1 = self._regularization(z)
        v2, c2 = self._regularization(prediction)
        variance, covariance = v1 + v2, c1 + c2
        total = sim_coeff * similarity + var_coeff * variance + cov_coeff * covariance
        return total, {"sim": float(similarity.detach()), "var": float(variance.detach()),
                       "cov": float(covariance.detach())}

    def imagine_rollout(self, states: Tensor, actions: Tensor) -> Tensor:
        """Return [batch,horizon,latent] without decoding pixels or RAM."""
        if states.ndim != 2 or actions.ndim != 2 or states.shape[0] != actions.shape[0] or actions.shape[1] < 1:
            raise ValueError("rollouts require states[B,D] and nonempty actions[B,H]")
        latent = self.encode(states)
        future: list[Tensor] = []
        for step in range(actions.shape[1]):
            latent = self.predict(latent, actions[:, step])
            future.append(latent)
        return torch.stack(future, dim=1)

    def forward_sequence_loss(self, states: Tensor, actions: Tensor,
                              horizons: tuple[int, ...] = (1, 2, 4)) -> tuple[Tensor, dict[str, float]]:
        """Train predictions at multiple elapsed times, including overlapping starts."""
        if states.ndim != 3 or actions.ndim != 2 or states.shape[:2] != (actions.shape[0], actions.shape[1] + 1):
            raise ValueError("expected states[B,T+1,D], actions[B,T]")
        if not horizons or min(horizons) < 1:
            raise ValueError("positive prediction horizons required")
        terms: list[Tensor] = []
        metrics: dict[str, float] = {}
        for horizon in horizons:
            if horizon > actions.shape[1]:
                continue
            starts = actions.shape[1] - horizon + 1
            initial = states[:, :starts].reshape(-1, self.state_dim)
            sequence = torch.stack([actions[:, i:i + starts] for i in range(horizon)], dim=-1).reshape(-1, horizon)
            prediction = self.imagine_rollout(initial, sequence)[:, -1]
            with torch.no_grad():
                target = cast(Tensor, self.target_encoder(states[:, horizon:horizon + starts].reshape(-1, self.state_dim)))
            invariance = F.mse_loss(prediction, target)
            variance, covariance = self._regularization(self.encode(initial))
            pv, pc = self._regularization(prediction)
            terms.append(25 * invariance + 25 * (variance + pv) + covariance + pc)
            metrics[f"horizon_{horizon}"] = float(invariance.detach())
        if not terms:
            raise ValueError("all horizons exceed sequence length")
        return torch.stack(terms).mean(), metrics

    @torch.no_grad()
    def plan_actions(self, states: Tensor, candidate_actions: Tensor, goal_states: Tensor,
                     discount: float = 0.95) -> Tensor:
        """Rank [batch,candidates,horizon] by discounted distance to target latents."""
        if candidate_actions.ndim != 3 or not 0 < discount <= 1:
            raise ValueError("expected candidate actions[B,K,H] and discount in (0,1]")
        batch, candidates, horizon = candidate_actions.shape
        repeated = states[:, None].expand(-1, candidates, -1).reshape(batch * candidates, -1)
        future = self.imagine_rollout(repeated, candidate_actions.reshape(batch * candidates, horizon))
        goal = cast(Tensor, self.target_encoder(goal_states))[:, None, None]
        distances = (future.reshape(batch, candidates, horizon, -1) - goal).square().mean(dim=-1)
        weights = torch.pow(torch.tensor(discount, device=states.device), torch.arange(horizon, device=states.device))
        return (distances * weights).sum(dim=-1).argmin(dim=-1)
