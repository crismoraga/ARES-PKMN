"""Networks for ARES-Nav: Nature-CNN encoder, conv autoencoder, PPO actor-critic and RND.

All networks consume uint8 pixel stacks scaled to [0, 1]; none of them sees RAM.  Only
DirectML-safe layers are used (Conv2d, ConvTranspose2d, pixel (un)shuffle, Linear, ReLU/LeakyReLU).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _conv_out(size: int, kernel: int, stride: int) -> int:
    return (size - kernel) // stride + 1


class SpaceToDepthConv(nn.Module):
    """An 8x8 stride-4 convolution expressed as ``pixel_unshuffle(4)`` + 2x2 stride-1 convolution.

    The two parametrisations describe the same function class (each output sees an aligned 8x8
    window with stride 4), but the second avoids the very slow strided-8x8 weight-gradient
    kernel of DirectML (about 12x faster on an RX 6700 XT).  Needs H and W divisible by 4.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 16, out_channels, 2, stride=1)

    def forward(self, pixels: Tensor) -> Tensor:
        out: Tensor = self.conv(nn.functional.pixel_unshuffle(pixels, 4))
        return out


def depth_to_space_deconv(in_channels: int, out_channels: int) -> nn.Sequential:
    """Transpose of :class:`SpaceToDepthConv`: 2x2 transposed conv then ``pixel_shuffle(4)``."""
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels * 16, 2, stride=1), nn.PixelShuffle(4))


def encoder_grid(height: int, width: int) -> tuple[int, int]:
    """Spatial size of the Nature-CNN feature map for an input of ``height`` x ``width``."""
    h, w = height, width
    for kernel, stride in ((8, 4), (4, 2), (3, 1)):
        h, w = _conv_out(h, kernel, stride), _conv_out(w, kernel, stride)
    if h < 1 or w < 1 or height % 4 or width % 4:
        raise ValueError(f"Input {height}x{width} must be >= 36 and divisible by 4 for the encoder")
    return h, w


class NatureEncoder(nn.Module):
    """Mnih et al. (2015) conv stack: 32x8s4 (as space-to-depth), 64x4s2, 64x3s1, then a linear layer."""

    def __init__(self, in_channels: int = 4, height: int = 84, width: int = 84, feature_dim: int = 512) -> None:
        super().__init__()
        self.in_channels, self.height, self.width, self.feature_dim = in_channels, height, width, feature_dim
        self.grid = encoder_grid(height, width)
        self.conv = nn.Sequential(
            SpaceToDepthConv(in_channels, 32), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(64 * self.grid[0] * self.grid[1], feature_dim), nn.ReLU())
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))  # ReLU gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, pixels: Tensor) -> Tensor:
        """``pixels``: float tensor in [0, 1] of shape (B, C, H, W)."""
        features: Tensor = self.fc(self.conv(pixels).flatten(1))
        return features


class ConvDecoder(nn.Module):
    """Mirror of :class:`NatureEncoder` with transposed convolutions; outputs logits."""

    def __init__(self, out_channels: int = 4, height: int = 84, width: int = 84, feature_dim: int = 512) -> None:
        super().__init__()
        self.height, self.width = height, width
        self.grid = encoder_grid(height, width)
        h1, w1 = _conv_out(height, 8, 4), _conv_out(width, 8, 4)
        h2, w2 = _conv_out(h1, 4, 2), _conv_out(w1, 4, 2)
        self.fc = nn.Sequential(nn.Linear(feature_dim, 64 * self.grid[0] * self.grid[1]), nn.ReLU())
        # output_padding recovers sizes lost to floor division in the encoder.
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, output_padding=(h1 - ((h2 - 1) * 2 + 4), w1 - ((w2 - 1) * 2 + 4))),
            nn.ReLU(),
            depth_to_space_deconv(32, out_channels),
        )
        if (h1 + 1) * 4 != height or (w1 + 1) * 4 != width:
            raise ValueError("Decoder needs H and W divisible by 4")

    def forward(self, features: Tensor) -> Tensor:
        grid = self.fc(features).view(-1, 64, self.grid[0], self.grid[1])
        logits: Tensor = self.deconv(grid)
        return logits


class ConvAutoencoder(nn.Module):
    """Self-supervised pixel autoencoder; its encoder initialises the policy encoder."""

    def __init__(self, channels: int = 4, height: int = 84, width: int = 84, feature_dim: int = 512) -> None:
        super().__init__()
        self.encoder = NatureEncoder(channels, height, width, feature_dim)
        self.decoder = ConvDecoder(channels, height, width, feature_dim)

    def forward(self, pixels: Tensor) -> Tensor:
        """Returns reconstruction logits with the same shape as ``pixels``."""
        logits: Tensor = self.decoder(self.encoder(pixels))
        return logits

    def loss(self, pixels: Tensor) -> Tensor:
        return nn.functional.binary_cross_entropy_with_logits(self(pixels), pixels)


VALUE_STREAMS: tuple[str, ...] = ("rnd", "episodic")


class ActorCritic(nn.Module):
    """Shared encoder with a categorical policy head and one value head per reward stream.

    Stream 0 is the non-episodic RND novelty, stream 1 the episodic frame-key novelty
    (two value heads as in Burda et al., 2018, where each stream has its own discounting/cuts).
    """

    def __init__(self, num_actions: int, channels: int = 4, height: int = 84, width: int = 84,
                 feature_dim: int = 512, value_heads: int = len(VALUE_STREAMS)) -> None:
        super().__init__()
        self.num_actions, self.value_heads = num_actions, value_heads
        self.encoder = NatureEncoder(channels, height, width, feature_dim)
        self.policy = nn.Linear(feature_dim, num_actions)
        self.value = nn.Linear(feature_dim, value_heads)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def forward(self, pixels: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (action logits (B, A), values (B, value_heads))."""
        features = self.encoder(pixels)
        logits: Tensor = self.policy(features)
        values: Tensor = self.value(features)
        return logits, values


def categorical_log_prob_entropy(logits: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
    """Log-prob of ``actions`` and entropy, written with gather/softmax only (DML friendly)."""
    log_probs = torch.log_softmax(logits, dim=-1)
    chosen = log_probs.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    entropy = -(log_probs.exp() * log_probs).sum(-1)
    return chosen, entropy


def _rnd_trunk(in_channels: int, height: int, width: int) -> tuple[nn.Sequential, int]:
    h, w = encoder_grid(height, width)
    trunk = nn.Sequential(
        SpaceToDepthConv(in_channels, 32), nn.LeakyReLU(),
        nn.Conv2d(32, 64, 4, stride=2), nn.LeakyReLU(),
        nn.Conv2d(64, 64, 3, stride=1), nn.LeakyReLU(),
        nn.Flatten(),
    )
    return trunk, 64 * h * w


class RNDModel(nn.Module):
    """Random Network Distillation (Burda et al., 2018) on the newest normalised frame.

    The target network is frozen at its random initialisation; the predictor is trained to
    match it, and the per-sample prediction error is the novelty bonus.
    """

    def __init__(self, in_channels: int = 1, height: int = 84, width: int = 84, out_dim: int = 512) -> None:
        super().__init__()
        target_trunk, flat = _rnd_trunk(in_channels, height, width)
        self.target = nn.Sequential(target_trunk, nn.Linear(flat, out_dim))
        predictor_trunk, flat = _rnd_trunk(in_channels, height, width)
        self.predictor = nn.Sequential(
            predictor_trunk, nn.Linear(flat, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, out_dim))
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

    def forward(self, normalised: Tensor) -> Tensor:
        """Per-sample squared prediction error (B,) for normalised frames (B, 1, H, W)."""
        with torch.no_grad():
            target: Tensor = self.target(normalised)
        prediction: Tensor = self.predictor(normalised)
        return (prediction - target).pow(2).mean(-1)
