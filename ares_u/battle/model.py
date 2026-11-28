"""Networks of E1: the entity Transformer actor-critic and the DQN MLP.

The Transformer is written by hand (pre-norm blocks with explicit multi-head attention) because
``nn.TransformerEncoderLayer(norm_first=True)`` fails under torch-directml. Token sequence:
``[field, 6 entities, 6 actions]``; each token type has its own input projection and a learned
type embedding. Action logits are a pointer attention from the field (summary) token over the
action tokens plus a per-token score; illegal actions are masked with a large negative value.
The value head reads the field token. The ``ids`` ablation adds a learned species embedding to
each entity token (hidden foes use index 0).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from .obs import ACTION_DIM, ACTIONS, ENTITIES, ENTITY_DIM, FIELD_DIM, FLAT_DIM, SPECIES_VOCAB, Batch, flat_batch

MASK_VALUE = -1e9
TRAIN_SPECIES_MAX = 151  # internal species ids 1..151 = Gen1, the only generation seen in training


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 128
    layers: int = 3
    heads: int = 4
    ff: int = 256
    species_ids: bool = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class PreNormBlock(nn.Module):
    """x + MHA(LN(x)); x + FF(LN(x)) with explicit attention (DirectML-safe)."""

    def __init__(self, d_model: int, heads: int, ff: int) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, ff)
        self.ff2 = nn.Linear(ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, tokens, 3, self.heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attended = torch.matmul(torch.softmax(scores, dim=-1), v)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, tokens, width))
        output: torch.Tensor = x + self.ff2(torch.relu(self.ff1(self.norm2(x))))
        return output


class EntityTransformer(nn.Module):
    """Actor-critic over field/entity/action tokens."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        width = config.d_model
        self.field_in = nn.Linear(FIELD_DIM, width)
        self.entity_in = nn.Linear(ENTITY_DIM, width)
        self.action_in = nn.Linear(ACTION_DIM, width)
        self.kind = nn.Parameter(torch.zeros(3, width))
        self.slot = nn.Parameter(torch.zeros(ENTITIES + ACTIONS, width))
        self.species = nn.Embedding(SPECIES_VOCAB, width) if config.species_ids else None
        if self.species is not None:
            nn.init.normal_(self.species.weight, std=0.1)
        self.blocks = nn.ModuleList(PreNormBlock(width, config.heads, config.ff) for _ in range(config.layers))
        self.final = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.score = nn.Linear(width, 1)
        self.value = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
        nn.init.normal_(self.kind, std=0.02)
        nn.init.normal_(self.slot, std=0.02)

    def forward(self, field: torch.Tensor, entities: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor,
                species: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns masked logits (B, ACTIONS) and values (B,)."""
        field_token = (self.field_in(field) + self.kind[0]).unsqueeze(1)
        entity_tokens = self.entity_in(entities) + self.kind[1] + self.slot[:ENTITIES]
        if self.species is not None:
            # Species outside the Gen1 training pool share the learned index-0 row (also used for
            # hidden foes) instead of feeding untrained random rows; the ablation then isolates the
            # reliance on species identity rather than the effect of arbitrary noise vectors.
            known = species.le(TRAIN_SPECIES_MAX)
            entity_tokens = entity_tokens + self.species(torch.where(known, species, torch.zeros_like(species)))
        action_tokens = self.action_in(actions) + self.kind[2] + self.slot[ENTITIES:]
        x = torch.cat([field_token, entity_tokens, action_tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.final(x)
        summary = x[:, 0]
        acts = x[:, 1 + ENTITIES:]
        pointer = torch.einsum("bd,bad->ba", self.query(summary), self.key(acts)) / math.sqrt(x.shape[-1])
        logits = pointer + self.score(acts).squeeze(-1)
        logits = logits.masked_fill(~mask, MASK_VALUE)
        return logits, self.value(summary).squeeze(-1)


class QNetwork(nn.Module):
    """DQN baseline: MLP over the flat observation, one Q-value per engine action."""

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FLAT_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, ACTIONS))

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.net(flat)
        return output


Weights = dict[str, NDArray[np.float32]]


def to_numpy(module: nn.Module) -> Weights:
    """CPU float32 copies of the parameters (what is shipped to rollout workers)."""
    return {name: tensor.detach().to("cpu").float().numpy().copy() for name, tensor in module.state_dict().items()}


def load_numpy(module: nn.Module, weights: Weights) -> None:
    module.load_state_dict({name: torch.from_numpy(np.asarray(value)) for name, value in weights.items()})


def batch_tensors(batch: Batch, device: torch.device | str = "cpu"
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (torch.as_tensor(batch.field, device=device), torch.as_tensor(batch.entities, device=device),
            torch.as_tensor(batch.actions, device=device), torch.as_tensor(batch.mask, device=device),
            torch.as_tensor(batch.species, device=device))


def policy_numpy(model: EntityTransformer, batch: Batch) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    """Action probabilities and values for a CPU batch (forward only, no autograd)."""
    with torch.no_grad():
        logits, values = model(*batch_tensors(batch))
        probs = torch.softmax(logits.double(), dim=-1).numpy()
    return probs, values.numpy().astype(np.float32)


def q_numpy(model: QNetwork, batch: Batch) -> NDArray[np.float32]:
    with torch.no_grad():
        values = model(torch.as_tensor(flat_batch(batch))).numpy().astype(np.float32)
    return np.where(batch.mask, values, -np.inf).astype(np.float32)


def build_policy(kind: str) -> EntityTransformer:
    return EntityTransformer(ModelConfig(species_ids=(kind == "ppo_ids")))
