"""Portable Mamba-2 SSD and masked Entity-Transformer actor/critic.

The SSM has one scalar A and time step per head, grouped B/C, a depthwise
causal convolution over X/B/C, gated RMS normalization and the full convolution
and SSM cache. This is the Mamba-2 parameterization, not the PDF's diagonal
Mamba-1 approximation. Chunk execution uses the semiseparable decay kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SSMCache:
    """Head states [B,H,P,N] and pre-convolution history [B,C,K-1]."""
    ssm_state: Tensor
    conv_state: Tensor

    def detach(self) -> SSMCache:
        return SSMCache(self.ssm_state.detach(), self.conv_state.detach())

    def reset(self, done: Tensor) -> SSMCache:
        """Clear terminated environments without resetting ongoing battles."""
        if done.shape != (self.ssm_state.shape[0],):
            raise ValueError("done mask must have one entry per environment")
        keep = (~done.bool()).to(self.ssm_state.dtype)
        return SSMCache(self.ssm_state * keep[:, None, None, None], self.conv_state * keep[:, None, None])


class SelectiveSSMBlock(nn.Module):
    """Mamba-2 with O(1) recurrent inference and portable chunked SSD training."""

    def __init__(self, d_model: int = 128, state_dim: int = 32,
                 expand: int = 2, head_dim: int = 32, conv_width: int = 4,
                 groups: int = 1) -> None:
        super().__init__()
        if min(d_model, state_dim, expand, head_dim, conv_width, groups) < 1:
            raise ValueError("all Mamba dimensions must be positive")
        self.d_model, self.state_dim = d_model, state_dim
        self.inner_dim, self.head_dim = d_model * expand, head_dim
        self.conv_width, self.groups = conv_width, groups
        if self.inner_dim % head_dim or (self.inner_dim // head_dim) % groups:
            raise ValueError("head and group dimensions must divide expanded width")
        self.num_heads = self.inner_dim // head_dim
        self.conv_dim = self.inner_dim + 2 * groups * state_dim
        self.in_proj = nn.Linear(d_model, self.inner_dim + self.conv_dim + self.num_heads, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, conv_width, groups=self.conv_dim)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1, 16, self.num_heads)))
        dt = torch.exp(torch.linspace(-6.0, -2.5, self.num_heads))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.norm_weight = nn.Parameter(torch.ones(self.inner_dim))
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=False)

    def initial_state(self, batch_size: int, device: torch.device | str | None = None,
                      dtype: torch.dtype | None = None) -> SSMCache:
        if batch_size < 1:
            raise ValueError("positive batch size required")
        return SSMCache(torch.zeros(batch_size, self.num_heads, self.head_dim, self.state_dim,
                                   device=device or self.A_log.device, dtype=dtype or self.A_log.dtype),
                        torch.zeros(batch_size, self.conv_dim, self.conv_width - 1,
                                    device=device or self.A_log.device, dtype=dtype or self.A_log.dtype))

    def _validate(self, inputs: Tensor, cache: SSMCache) -> None:
        batch = inputs.shape[0]
        if cache.ssm_state.shape != (batch, self.num_heads, self.head_dim, self.state_dim) or cache.conv_state.shape != (batch, self.conv_dim, self.conv_width - 1):
            raise ValueError("incompatible complete Mamba-2 cache")
        if inputs.shape[-1] != self.d_model:
            raise ValueError("incorrect SSM input dimension")

    def _finish(self, output: Tensor, gate: Tensor, residual: Tensor) -> Tensor:
        gated = output * F.silu(gate)
        normalized = gated * torch.rsqrt(gated.square().mean(dim=-1, keepdim=True) + 1e-5)
        return cast(Tensor, self.out_proj(normalized * self.norm_weight)) + residual

    def _split_conv(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = values[..., :self.inner_dim]
        b = values[..., self.inner_dim:self.inner_dim + self.groups * self.state_dim]
        c = values[..., self.inner_dim + self.groups * self.state_dim:]
        prefix = values.shape[:-1]
        x = x.reshape(*prefix, self.num_heads, self.head_dim)
        b = b.reshape(*prefix, self.groups, self.state_dim).repeat_interleave(self.num_heads // self.groups, dim=-2)
        c = c.reshape(*prefix, self.groups, self.state_dim).repeat_interleave(self.num_heads // self.groups, dim=-2)
        return x, b, c

    def step_recurrent(self, x_t: Tensor, h_prev: SSMCache) -> tuple[Tensor, SSMCache]:
        """Advance one turn, retaining both raw causal convolution and SSD state."""
        if x_t.ndim != 2:
            raise ValueError("one turn must have shape [B,D]")
        self._validate(x_t, h_prev)
        projection = cast(Tensor, self.in_proj(x_t))
        gate = projection[..., :self.inner_dim]
        raw = projection[..., self.inner_dim:self.inner_dim + self.conv_dim]
        dt_raw = projection[..., self.inner_dim + self.conv_dim:]
        history = torch.cat((h_prev.conv_state, raw.unsqueeze(-1)), dim=-1)
        convolved = F.silu(F.conv1d(history, self.conv1d.weight, self.conv1d.bias, groups=self.conv_dim).squeeze(-1))
        x, b, c = self._split_conv(convolved)
        dt = F.softplus(dt_raw + self.dt_bias)
        decay = torch.exp(-self.A_log.exp() * dt)
        state = h_prev.ssm_state * decay[:, :, None, None] + (x * dt[:, :, None])[:, :, :, None] * b[:, :, None, :]
        output = torch.einsum("bhpn,bhn->bhp", state, c) + x * self.D[None, :, None]
        next_cache = SSMCache(state, history[:, :, 1:])
        return self._finish(output.flatten(1), gate, x_t), next_cache

    def forward_sequence(self, inputs: Tensor, state: SSMCache | None = None,
                         chunk_size: int = 32) -> tuple[Tensor, SSMCache]:
        """Evaluate causal SSD chunks, handling arbitrary final chunk lengths."""
        if inputs.ndim != 3 or inputs.shape[1] < 1 or chunk_size < 1:
            raise ValueError("nonempty [B,T,D] sequence and positive chunk size required")
        cache = state or self.initial_state(inputs.shape[0], inputs.device, inputs.dtype)
        self._validate(inputs, cache)
        projected = cast(Tensor, self.in_proj(inputs))
        gate = projected[..., :self.inner_dim]
        raw = projected[..., self.inner_dim:self.inner_dim + self.conv_dim]
        dt_raw = projected[..., self.inner_dim + self.conv_dim:]
        history = torch.cat((cache.conv_state, raw.transpose(1, 2)), dim=-1)
        convolved = F.silu(F.conv1d(history, self.conv1d.weight, self.conv1d.bias, groups=self.conv_dim).transpose(1, 2))
        x, b, c = self._split_conv(convolved)
        dt = F.softplus(dt_raw + self.dt_bias)
        decay_log = (-self.A_log.exp() * dt).transpose(1, 2)
        scaled_x = x * dt.unsqueeze(-1)
        carry = cache.ssm_state
        outputs: list[Tensor] = []
        for start in range(0, inputs.shape[1], chunk_size):
            stop = min(start + chunk_size, inputs.shape[1])
            log_decay = decay_log[:, :, start:stop]
            cumulative = log_decay.cumsum(dim=-1)
            # exp(sum_{k=j+1..i} A_k): a lower-triangular semiseparable kernel.
            differences = cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)
            lower = torch.ones(stop - start, stop - start, dtype=torch.bool, device=inputs.device).tril()
            kernel = differences.masked_fill(~lower, -torch.inf).exp()
            bc, cc, xc = b[:, start:stop], c[:, start:stop], scaled_x[:, start:stop]
            within = torch.einsum("bihn,bjhn,bhij,bjhp->bihp", cc, bc, kernel, xc)
            inherited = torch.einsum("bihn,bhpn,bhi->bihp", cc, carry, cumulative.exp())
            outputs.append(within + inherited + x[:, start:stop] * self.D[None, None, :, None])
            final_decay = (cumulative[:, :, -1:] - cumulative).exp()
            carry = carry * cumulative[:, :, -1, None, None].exp() + torch.einsum("bjhn,bhj,bjhp->bhpn", bc, final_decay, xc)
        output = torch.cat(outputs, dim=1).flatten(2)
        conv_cache = history[:, :, -(self.conv_width - 1):] if self.conv_width > 1 else history[:, :, :0]
        return self._finish(output, gate, inputs), SSMCache(carry, conv_cache)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_sequence(inputs)[0]


class AresBattlePolicyNetwork(nn.Module):
    """Spatial observable entities plus recurrent context; nine actions/three targets."""

    def __init__(self, entity_dim: int = 48, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3) -> None:
        super().__init__()
        if d_model % nhead or min(entity_dim, num_layers) < 1:
            raise ValueError("valid transformer dimensions required")
        self.entity_dim, self.d_model = entity_dim, d_model
        self.entity_encoder = nn.Sequential(nn.Linear(entity_dim, d_model), nn.LayerNorm(d_model),
            nn.GELU(), nn.Linear(d_model, d_model))
        self.role_embedding = nn.Embedding(7, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, 2 * d_model, dropout=0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.spatial_transformer = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.temporal_ssm = SelectiveSSMBlock(d_model)
        self.action_head = nn.Linear(d_model, 9)
        self.target_head = nn.Linear(2 * d_model, 3)
        self.value_head = nn.Linear(d_model, 1)

    def initial_state(self, batch_size: int) -> SSMCache:
        return self.temporal_ssm.initial_state(batch_size)

    @staticmethod
    def _masked(logits: Tensor, mask: Tensor) -> Tensor:
        if mask.shape != logits.shape or not mask.bool().any(dim=-1).all():
            raise ValueError("each action/target row needs at least one legal choice")
        return logits.masked_fill(~mask.bool(), -torch.inf)

    def forward_turn(self, entities: Tensor, roles: Tensor, action_mask: Tensor,
                     ssm_state: SSMCache | None, target_mask: Tensor | None = None,
                     entity_mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, SSMCache]:
        """Missing entities are excluded from attention and pooling, never zero-valued tokens."""
        if entities.ndim != 3 or entities.shape[-1] != self.entity_dim or roles.shape != entities.shape[:2]:
            raise ValueError("entities[B,E,48] and roles[B,E] required")
        present = torch.ones_like(roles, dtype=torch.bool) if entity_mask is None else entity_mask.bool()
        if present.shape != roles.shape or not present.any(dim=-1).all():
            raise ValueError("at least one observed entity required per battle")
        safe_entities = entities.masked_fill(~present.unsqueeze(-1), 0)
        x = cast(Tensor, self.entity_encoder(safe_entities)) + cast(Tensor, self.role_embedding(roles))
        spatial = cast(Tensor, self.spatial_transformer(x, src_key_padding_mask=~present))
        pooled = (spatial * present.unsqueeze(-1)).sum(dim=1) / present.sum(dim=1, keepdim=True)
        active_mask = (roles == 0) & present
        active = (spatial * active_mask.unsqueeze(-1)).sum(dim=1) / active_mask.sum(dim=1, keepdim=True).clamp_min(1)
        active = torch.where(active_mask.any(dim=-1, keepdim=True), active, pooled)
        adapted, cache = self.temporal_ssm.step_recurrent((active + pooled) * 0.5, ssm_state or self.initial_state(entities.shape[0]))
        actions = self._masked(cast(Tensor, self.action_head(adapted)), action_mask)
        targets = cast(Tensor, self.target_head(torch.cat((active, adapted), dim=-1)))
        if target_mask is not None:
            targets = self._masked(targets, target_mask)
        return actions, targets, cast(Tensor, self.value_head(adapted)), cache
