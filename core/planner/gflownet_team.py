"""Trajectory-balance sampling of unordered teams and legal movesets."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TeamCandidate:
    """Observable candidate metadata; legal moves must come from ROM learnsets."""
    species_id: int
    types: tuple[int, int]
    legal_moves: tuple[int, ...]
    offensive_coverage: tuple[float, ...] = ()
    defensive_risk: float = 0.0


@dataclass(frozen=True)
class TeamTrajectory:
    species: tuple[int, ...]
    moves: tuple[tuple[int, ...], ...]
    state: Tensor
    log_pf_minus_pb: Tensor


class TeamSynthesisGFlowNet(nn.Module):
    """Probability proportional to positive team reward with uniform parent policy."""

    def __init__(self, num_available_species: int = 150, hidden_dim: int = 128,
                 num_available_moves: int = 355) -> None:
        super().__init__()
        if min(num_available_species, hidden_dim, num_available_moves) < 1:
            raise ValueError("positive catalog and hidden dimensions required")
        self.num_species, self.num_moves = num_available_species, num_available_moves
        self.log_Z = nn.Parameter(torch.tensor(0.0))
        self.forward_policy_net = nn.Sequential(nn.Linear(self.num_species, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, self.num_species))
        self.move_policy_net = nn.Sequential(nn.Linear(self.num_species + self.num_moves, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, self.num_moves))

    def sample_team_trajectory(self, team_size: int = 6, temp: float = 1.0,
                               available_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if not 1 <= team_size <= min(6, self.num_species) or temp <= 0:
            raise ValueError("team size must be 1..6 within catalog; temperature must be positive")
        allowed = torch.ones(self.num_species, dtype=torch.bool, device=self.log_Z.device) if available_mask is None else available_mask.to(device=self.log_Z.device, dtype=torch.bool)
        if allowed.shape != (self.num_species,) or int(allowed.sum()) < team_size:
            raise ValueError("insufficient available candidates")
        state = torch.zeros(self.num_species, device=self.log_Z.device)
        ratio = self.log_Z * 0
        for step in range(team_size):
            logits = cast(Tensor, self.forward_policy_net(state)) / temp
            log_probs = F.log_softmax(logits.masked_fill(~allowed | state.bool(), -torch.inf), dim=-1)
            chosen = int(torch.multinomial(log_probs.exp(), 1).item())
            ratio = ratio + log_probs[chosen] + math.log(step + 1)
            state = state.clone()
            state[chosen] = 1
        return state, ratio

    def sample_team_and_moves(self, catalog: tuple[TeamCandidate, ...], team_size: int = 6,
                              temp: float = 1.0, available_mask: Tensor | None = None) -> TeamTrajectory:
        """Construct unordered species then unordered moves with separate reverse counts."""
        if len(catalog) != self.num_species:
            raise ValueError("candidate ordering must match species network outputs")
        state, ratio = self.sample_team_trajectory(team_size, temp, available_mask)
        selected = [int(i) for i in state.nonzero().flatten().tolist()]
        all_moves: list[tuple[int, ...]] = []
        for idx in selected:
            legal = tuple(sorted(set(catalog[idx].legal_moves)))
            if not legal or min(legal) < 1 or max(legal) >= self.num_moves:
                raise ValueError("each species needs valid nonzero learnset move IDs")
            move_state = torch.zeros(self.num_moves, device=self.log_Z.device)
            allowed = torch.zeros_like(move_state, dtype=torch.bool)
            allowed[list(legal)] = True
            species = torch.zeros_like(state)
            species[idx] = 1
            for step in range(min(4, len(legal))):
                logits = cast(Tensor, self.move_policy_net(torch.cat((species, move_state)))) / temp
                lp = F.log_softmax(logits.masked_fill(~allowed | move_state.bool(), -torch.inf), dim=-1)
                chosen = int(torch.multinomial(lp.exp(), 1).item())
                ratio = ratio + lp[chosen] + math.log(step + 1)
                move_state = move_state.clone()
                move_state[chosen] = 1
            all_moves.append(tuple(int(m) for m in move_state.nonzero().flatten().tolist()))
        return TeamTrajectory(tuple(catalog[i].species_id for i in selected), tuple(all_moves), state, ratio)

    def compute_trajectory_balance_loss(self, state_terminal: Tensor,
                                        log_pf_minus_pb: Tensor, log_reward: Tensor) -> Tensor:
        if state_terminal.shape[-1] != self.num_species or not torch.isfinite(log_reward).all():
            raise ValueError("valid terminal state and finite log reward required")
        return F.mse_loss(self.log_Z + log_pf_minus_pb, log_reward)

    @staticmethod
    def coverage_reward(team: tuple[TeamCandidate, ...]) -> float:
        """Strictly positive reward from declared offense, type diversity and risk."""
        if not team:
            raise ValueError("nonempty team required")
        types = {t for member in team for t in member.types if t != 9}
        dims = max((len(member.offensive_coverage) for member in team), default=0)
        coverage = sum(max((member.offensive_coverage[i] if i < len(member.offensive_coverage) else 0.0)
                           for member in team) for i in range(dims)) / max(1, dims)
        risk = sum(member.defensive_risk for member in team) / len(team)
        return math.exp(max(-20.0, min(20.0, len(types) / 6 + coverage - risk)))
