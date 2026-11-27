"""Training and evaluation for ARES-Nav (experiment E2).

Pipeline (one seed):
  1. ``collect_frames``      sticky-random rollouts on FireRed -> frame sequences (.npz)
  2. ``pretrain_autoencoder`` conv AE on FireRed frame stacks (self-supervised)
  3. ``train_ppo_rnd``        PPO + RND on FireRed, encoder initialised from the AE
  4. ``train_ppo_rnd``        Emerald from scratch      (method ``scratch``)
     ``train_ppo_rnd``        Emerald from the FireRed policy (method ``transfer``)
  5. ``evaluate``             Emerald: random, scratch, transfer, zero-shot (FireRed policy, no training)

The learner runs on DirectML when available, following the project rules: the first
``backward()`` of a process happens on the learner device, the optimiser is a lerp-free Adam and the
rollout worker processes (``ares_u.nav.env``) never import torch.  The reward is purely intrinsic and
pixel-based, with two streams and two value heads (Burda et al., 2018): the non-episodic RND
novelty of the newest frame and an episodic count bonus on coarse quantised frame keys
(``ares_u.nav.novelty``).  RAM (map, x, y) is only logged as the exploration metric.
"""
from __future__ import annotations

import json
import math
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

from ares_u.nav.env import NUM_ACTIONS, Cell, NavConfig, SubprocVecEnv, make_configs
from ares_u.nav.models import ActorCritic, ConvAutoencoder, RNDModel, categorical_log_prob_entropy
from ares_u.nav.novelty import EpisodicNovelty, PlaceNovelty

UInt8Array = NDArray[np.uint8]
Float64Array = NDArray[np.float64]


# --------------------------------------------------------------------------- utilities

def resolve_device(name: str = "auto") -> torch.device:
    """``auto`` -> DirectML if importable, else CPU."""
    if name in ("auto", "dml"):
        try:
            import torch_directml
            device: torch.device = torch_directml.device()
            return device
        except ImportError:
            if name == "dml":
                raise
    return torch.device(name if name != "auto" else "cpu")


class LerpFreeAdam(Optimizer):
    """Adam (Kingma & Ba) written with mul_/add_/addcmul_/addcdiv_ only.

    ``torch.optim.Adam(foreach=False)`` calls ``lerp_``, which DirectML does not implement and
    silently runs on the CPU (a device round-trip per parameter per step, ~25 ms per step here).
    This implementation is numerically the same update (checked against torch Adam in tests).
    """

    def __init__(self, params: Iterator[nn.Parameter] | list[nn.Parameter], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8) -> None:
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps})

    @torch.no_grad()
    def step(self, closure: None = None) -> None:  # type: ignore[override]
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                state["step"] += 1
                step = int(state["step"])
                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(param.grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(param.grad, param.grad, value=1.0 - beta2)
                bias1 = 1.0 - beta1 ** step
                bias2 = 1.0 - beta2 ** step
                denom = (exp_avg_sq / bias2).sqrt_().add_(group["eps"])
                param.addcdiv_(exp_avg, denom, value=-group["lr"] / bias1)


def make_optimizer(parameters: Iterator[nn.Parameter] | list[nn.Parameter], lr: float) -> Optimizer:
    return LerpFreeAdam(parameters, lr=lr, eps=1e-5)


def backward(loss: Tensor) -> None:
    """Typed wrapper around ``Tensor.backward`` (unannotated in torch's stubs)."""
    torch.autograd.backward(loss)


def pixels_to_device(array: UInt8Array, device: torch.device) -> Tensor:
    """uint8 (B, C, H, W) -> float [0, 1] on ``device`` (transfers uint8, converts on device)."""
    return torch.from_numpy(np.ascontiguousarray(array)).to(device).float().div_(255.0)


class RunningMeanStd:
    """Running moments: all-history parallel variance (Chan et al.) or, with ``decay``, an
    exponential moving average over batches (tracks non-stationary statistics)."""

    def __init__(self, shape: tuple[int, ...] = (), decay: float | None = None) -> None:
        if decay is not None and not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1) (None = all history)")
        self.mean: Float64Array = np.zeros(shape, dtype=np.float64)
        self.var: Float64Array = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        self.decay = decay

    def update(self, batch: NDArray[Any]) -> None:
        values = np.asarray(batch, dtype=np.float64)
        batch_mean, batch_var, n = values.mean(axis=0), values.var(axis=0), values.shape[0]
        delta = batch_mean - self.mean
        if self.decay is not None:
            if self.count < 1.0:
                self.mean, self.var = batch_mean, batch_var
            else:
                keep = self.decay
                self.mean = self.mean + (1.0 - keep) * delta
                self.var = keep * self.var + (1.0 - keep) * batch_var + keep * (1.0 - keep) * np.square(delta)
            self.count += n
            return
        total = self.count + n
        self.mean = self.mean + delta * n / total
        m2 = self.var * self.count + batch_var * n + np.square(delta) * self.count * n / total
        self.var = m2 / total
        self.count = total

    @property
    def std(self) -> Float64Array:
        return np.sqrt(self.var)

    def state(self) -> dict[str, Any]:
        return {"mean": torch.from_numpy(np.array(self.mean, dtype=np.float64)),
                "var": torch.from_numpy(np.array(self.var, dtype=np.float64)),
                "count": float(self.count)}

    def load(self, state: dict[str, Any]) -> None:
        self.mean = np.asarray(state["mean"].numpy(), dtype=np.float64)
        self.var = np.asarray(state["var"].numpy(), dtype=np.float64)
        self.count = float(state["count"])


class RewardForwardFilter:
    """Discounted running return used to normalise intrinsic rewards (Burda et al., 2018).

    With ``dones`` the running return restarts after an episode end (episodic streams)."""

    def __init__(self, gamma: float, num_envs: int) -> None:
        self.gamma = gamma
        self.returns: Float64Array = np.zeros(num_envs, dtype=np.float64)

    def update(self, rewards: Float64Array, dones: NDArray[np.bool_] | None = None) -> Float64Array:
        self.returns = self.returns * self.gamma + rewards
        out = self.returns.copy()
        if dones is not None:
            self.returns = np.where(dones, 0.0, self.returns)
        return out


class StreamNormaliser:
    """Divide a reward stream by the std of its discounted return (EMA over updates by default).

    With ``center`` the running mean reward is subtracted first ("reward centering", Naik et al.,
    2024).  The non-episodic RND stream is a continuing task whose rewards are always positive, so
    without centering its discounted return is dominated by a large constant (~mean/(1-gamma)) that
    the value head must chase while the normaliser moves; centering keeps the value targets O(1)
    and leaves the policy gradient unchanged in expectation (novel frames stay above average)."""

    def __init__(self, gamma: float, num_envs: int, decay: float | None, center: bool = False) -> None:
        self.filter = RewardForwardFilter(gamma, num_envs)
        self.rms = RunningMeanStd((), decay)
        self.center = center
        self.reward_mean = RunningMeanStd((), decay)

    def __call__(self, rewards: Float64Array, dones: NDArray[np.bool_] | None = None) -> Float64Array:
        """``rewards`` and ``dones`` are (T, N); returns the normalised rewards (T, N)."""
        if self.center:
            self.reward_mean.update(rewards.reshape(-1))
            rewards = rewards - float(self.reward_mean.mean)
        filtered = np.stack([self.filter.update(rewards[t], None if dones is None else dones[t])
                             for t in range(rewards.shape[0])])
        self.rms.update(filtered.reshape(-1))
        scale = max(float(np.sqrt(self.rms.var)), 1e-8)
        normalised: Float64Array = rewards / scale
        return normalised

    def state(self) -> dict[str, Any]:
        return {"returns": torch.from_numpy(self.filter.returns.copy()), "rms": self.rms.state(),
                "reward_mean": self.reward_mean.state()}

    def load(self, state: dict[str, Any]) -> None:
        self.filter.returns = np.asarray(state["returns"].numpy(), dtype=np.float64)
        self.rms.load(state["rms"])
        self.reward_mean.load(state["reward_mean"])


def gae_non_episodic(rewards: Float64Array, values: Float64Array, last_value: Float64Array,
                     gamma: float, lam: float) -> Float64Array:
    """GAE over (T, N) without cutting at episode ends (intrinsic return is non-episodic)."""
    steps = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    running = np.zeros_like(last_value)
    for t in reversed(range(steps)):
        next_value = values[t + 1] if t + 1 < steps else last_value
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running
    return advantages


def gae_episodic(rewards: Float64Array, values: Float64Array, last_value: Float64Array,
                 dones: NDArray[np.bool_], gamma: float, lam: float) -> Float64Array:
    """GAE over (T, N) cut at episode ends: ``dones[t]`` means s_{t+1} starts a new episode."""
    steps = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    running = np.zeros_like(last_value)
    for t in reversed(range(steps)):
        next_value = values[t + 1] if t + 1 < steps else last_value
        nonterminal = 1.0 - dones[t].astype(np.float64)
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        running = delta + gamma * lam * nonterminal * running
        advantages[t] = running
    return advantages


def explained_variance(prediction: Float64Array, target: Float64Array) -> float:
    variance = float(np.var(target))
    return float("nan") if variance < 1e-12 else 1.0 - float(np.var(target - prediction)) / variance


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))


# --------------------------------------------------------------------------- frame collection

class StickyRandomPolicy:
    """Uniform random actions that persist with probability ``repeat_prob`` (explorative)."""

    def __init__(self, num_envs: int, seed: int, repeat_prob: float = 0.75) -> None:
        self.rng = np.random.default_rng(seed)
        self.repeat_prob = repeat_prob
        self.current = self.rng.integers(0, NUM_ACTIONS, size=num_envs)

    def __call__(self) -> NDArray[np.int64]:
        resample = self.rng.random(self.current.shape[0]) >= self.repeat_prob
        self.current = np.where(resample, self.rng.integers(0, NUM_ACTIONS, size=self.current.shape[0]),
                                self.current)
        return self.current.astype(np.int64)


def collect_frames(env_config: NavConfig, num_envs: int, steps_per_env: int, seed: int,
                   out_path: Path, repeat_prob: float = 0.75) -> dict[str, Any]:
    """Store newest frames (E, T+1, H, W) and episode-start flags; stacks are rebuilt later.

    With ``env_config.start_states`` every episode starts from a uniformly drawn savestate, which
    gives the autoencoder visually diverse frames (towns, routes, caves, gyms, battles).
    """
    policy = StickyRandomPolicy(num_envs, seed, repeat_prob)
    height, width = env_config.obs_height, env_config.obs_width
    frames = np.zeros((num_envs, steps_per_env + 1, height, width), dtype=np.uint8)
    starts = np.zeros((num_envs, steps_per_env + 1), dtype=np.bool_)
    cells: set[Cell] = set()
    start = time.perf_counter()
    with SubprocVecEnv(make_configs(env_config, num_envs), [seed * 1000 + i for i in range(num_envs)]) as venv:
        obs = venv.reset()
        frames[:, 0], starts[:, 0] = obs[:, -1], True
        for t in range(steps_per_env):
            obs, dones, infos = venv.step(policy())
            frames[:, t + 1] = obs[:, -1]
            starts[:, t + 1] = dones
            cells.update(tuple(c) for info in infos for c in info["step_cells"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, frames=frames, starts=starts, frame_stack=np.int64(env_config.frame_stack))
    stats = {"game": env_config.game, "frames": int(frames.shape[0] * frames.shape[1]), "unique_cells": len(cells),
             "unique_maps": len({(c[0], c[1]) for c in cells}),
             "start_states": [str(p) for p in env_config.resolved_states()],
             "episode_steps": env_config.episode_steps, "repeat_prob": repeat_prob,
             "seconds": time.perf_counter() - start, "path": str(out_path)}
    out_path.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


class FrameStackDataset:
    """Rebuild (C, H, W) frame stacks from stored sequences, clamping at episode starts."""

    def __init__(self, frames: UInt8Array, starts: NDArray[np.bool_], frame_stack: int) -> None:
        if frames.ndim != 4 or starts.shape != frames.shape[:2]:
            raise ValueError("frames must be (E, T, H, W) and starts (E, T)")
        self.frames, self.frame_stack = frames, frame_stack
        steps = np.arange(frames.shape[1])
        marked = np.where(starts, steps[None, :], 0)
        self.episode_start = np.maximum.accumulate(marked, axis=1)
        self.num_envs, self.steps = frames.shape[0], frames.shape[1]

    @classmethod
    def load(cls, path: Path) -> FrameStackDataset:
        with np.load(path) as data:
            return cls(data["frames"], data["starts"], int(data["frame_stack"]))

    def __len__(self) -> int:
        return self.num_envs * self.steps

    def batch(self, indices: NDArray[np.int64]) -> UInt8Array:
        env, step = np.divmod(indices, self.steps)
        offsets = np.arange(-(self.frame_stack - 1), 1)
        times = np.maximum(step[:, None] + offsets[None, :], self.episode_start[env, step][:, None])
        stacks: UInt8Array = self.frames[env[:, None], times]
        return stacks


# --------------------------------------------------------------------------- autoencoder

@dataclass
class AEConfig:
    frames_path: str
    out_path: str
    eval_frames_path: str = ""
    epochs: int = 20
    batch_size: int = 128
    lr: float = 5e-4
    val_fraction: float = 0.05
    seed: int = 0
    device: str = "auto"
    feature_dim: int = 512


def _ae_eval(model: ConvAutoencoder, data: FrameStackDataset, indices: NDArray[np.int64],
             device: torch.device, batch_size: int) -> dict[str, float]:
    """BCE (the training loss), pixel MSE and R^2 against the per-pixel mean image of the set."""
    model.eval()
    bce = mse = 0.0
    pixel_sum: Tensor | None = None
    pixel_sq = 0.0
    with torch.no_grad():
        for begin in range(0, len(indices), batch_size):
            pixels = pixels_to_device(data.batch(indices[begin:begin + batch_size]), device)
            logits = model(pixels)
            bce += float(nn.functional.binary_cross_entropy_with_logits(logits, pixels).item()) * len(pixels)
            mse += float((torch.sigmoid(logits) - pixels).pow(2).mean().item()) * len(pixels)
            batch_sum = pixels.sum(0).cpu()
            pixel_sum = batch_sum if pixel_sum is None else pixel_sum + batch_sum
            pixel_sq += float(pixels.pow(2).mean().item()) * len(pixels)
    model.train()
    n = max(1, len(indices))
    mean_image = (pixel_sum / n) if pixel_sum is not None else torch.zeros(1)
    variance = pixel_sq / n - float(mean_image.pow(2).mean())  # MSE of predicting the mean image
    return {"bce": bce / n, "mse": mse / n, "r2": 1.0 - (mse / n) / max(variance, 1e-12)}


def pretrain_autoencoder(config: AEConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    data = FrameStackDataset.load(Path(config.frames_path))
    rng = np.random.default_rng(config.seed)
    order = rng.permutation(len(data)).astype(np.int64)
    n_val = max(1, int(len(order) * config.val_fraction))
    val_idx, train_idx = order[:n_val], order[n_val:]
    channels, height, width = data.frame_stack, data.frames.shape[2], data.frames.shape[3]
    model = ConvAutoencoder(channels, height, width, config.feature_dim).to(device)
    optimizer = make_optimizer(list(model.parameters()), config.lr)
    out = Path(config.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".jsonl")
    log_path.unlink(missing_ok=True)
    eval_data = FrameStackDataset.load(Path(config.eval_frames_path)) if config.eval_frames_path else None
    eval_idx = (np.random.default_rng(1).permutation(len(eval_data))[:4096].astype(np.int64)
                if eval_data is not None else None)
    start = time.perf_counter()
    history: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        rng.shuffle(train_idx)
        total, seen = 0.0, 0
        for begin in range(0, len(train_idx), config.batch_size):
            pixels = pixels_to_device(data.batch(train_idx[begin:begin + config.batch_size]), device)
            loss = model.loss(pixels)
            optimizer.zero_grad(set_to_none=True)
            backward(loss)
            optimizer.step()
            total += float(loss.item()) * len(pixels)
            seen += len(pixels)
        val = _ae_eval(model, data, val_idx, device, config.batch_size)
        row: dict[str, Any] = {"epoch": epoch + 1, "train_bce": total / max(1, seen), "val_bce": val["bce"],
                               "val_mse": val["mse"], "val_r2": val["r2"], "seconds": time.perf_counter() - start}
        if eval_data is not None and eval_idx is not None:
            cross = _ae_eval(model, eval_data, eval_idx, device, config.batch_size)
            row.update({"cross_game_bce": cross["bce"], "cross_game_mse": cross["mse"], "cross_game_r2": cross["r2"]})
        history.append(row)
        write_jsonl(log_path, row)
    tmp = out.with_suffix(".tmp")
    torch.save({"encoder": {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()},
                "decoder": {k: v.detach().cpu() for k, v in model.decoder.state_dict().items()},
                "config": asdict(config), "shape": [channels, height, width],
                "frames": len(data)}, tmp)
    tmp.replace(out)
    return {"final": history[-1], "device": str(device), "path": str(out), "frames": len(data)}


# --------------------------------------------------------------------------- PPO + RND + episodic novelty

@dataclass
class TrainConfig:
    game: str
    out_dir: str
    total_steps: int = 250_000
    num_envs: int = 6
    rollout_len: int = 128
    epochs: int = 4
    minibatches: int = 4
    lr: float = 2.5e-4
    gamma_rnd: float = 0.99
    gamma_epi: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rnd_coef: float = 0.5
    epi_coef: float = 1.0  # 0 -> RND only (the episodic head is still fitted, for diagnostics)
    place_coef: float = 0.0  # > 0 adds the pixel place-novelty bonus (fade -> new scene) to the episodic stream
    reward_norm_decay: float = 0.9  # EMA decay per update of the return-std normalisers; >= 1 -> all history
    hash_rows: int = 5
    hash_cols: int = 7
    hash_levels: int = 8
    rnd_update_proportion: float = 0.25
    center_rnd: bool = True  # reward centering of the continuing RND stream
    target_kl: float = 0.05  # stop the remaining PPO epochs of an update once mean approx-KL exceeds it (0 = off)
    norm_init_steps: int = 64
    init: str = "scratch"  # "scratch" | "ae:<path>" | "policy:<checkpoint>"
    random_policy: bool = False
    seed: int = 0
    device: str = "auto"
    checkpoint_every: int = 10
    max_minutes: float = 0.0
    torch_threads: int = 2
    resume: bool = False
    env: NavConfig = field(default_factory=lambda: NavConfig("emerald"))


def build_actor_critic(env: NavConfig) -> ActorCritic:
    return ActorCritic(NUM_ACTIONS, env.frame_stack, env.obs_height, env.obs_width)


def load_policy(path: Path, env: NavConfig) -> ActorCritic:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_actor_critic(env)
    model.load_state_dict(checkpoint["model"])
    return model


def _initialise(model: ActorCritic, init: str) -> str:
    if init == "scratch":
        return "random init"
    kind, _, path = init.partition(":")
    if kind == "ae":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model.encoder.load_state_dict(checkpoint["encoder"])
        return f"encoder from AE {path}"
    if kind == "policy":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        return f"actor-critic from {path} (RND and normalisers reinitialised)"
    if kind == "encoder":
        # Representation transfer: the visual encoder of a trained actor-critic with fresh policy and
        # value heads (near-uniform policy, i.e. maximal entropy), to avoid the plasticity/entropy loss
        # observed when the whole pretrained policy is fine-tuned.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        prefix = "encoder."
        encoder_state = {key[len(prefix):]: value for key, value in checkpoint["model"].items()
                         if key.startswith(prefix)}
        if not encoder_state:
            raise ValueError(f"{path} has no encoder weights")
        model.encoder.load_state_dict(encoder_state)
        return f"encoder from actor-critic {path}; policy/value heads, RND and normalisers reinitialised"
    raise ValueError(f"Unknown init {init!r}")


def _config_json(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    data["env"] = config.env.to_dict()
    return data


def _newest(observations: UInt8Array) -> UInt8Array:
    """(B, C, H, W) -> (B, 1, H, W) newest frame for RND."""
    return observations[:, -1:]


class ExplorationLog:
    """Metric-only bookkeeping of RAM cells; never fed to the learner."""

    def __init__(self) -> None:
        self.global_cells: set[Cell] = set()
        self.recent_episode_cells: deque[int] = deque(maxlen=32)
        self.recent_episode_maps: deque[int] = deque(maxlen=32)
        self.episodes = 0
        self.battle_steps = 0
        self.steps = 0
        self.unconfirmed_samples = 0

    def observe(self, infos: list[dict[str, Any]]) -> list[int]:
        finished: list[int] = []
        for info in infos:
            for cell in info.get("step_cells", [info["cell"]]):
                if cell[0] >= 0:
                    self.global_cells.add((int(cell[0]), int(cell[1]), int(cell[2]), int(cell[3])))
            self.battle_steps += int(bool(info["in_battle"]))
            self.unconfirmed_samples += int(info.get("unconfirmed_samples", 0))
            self.steps += 1
            if "final_episode_cells" in info:
                self.episodes += 1
                finished.append(int(info["final_episode_cells"]))
                self.recent_episode_cells.append(int(info["final_episode_cells"]))
                self.recent_episode_maps.append(int(info["final_episode_maps"]))
        return finished

    def global_maps(self) -> int:
        return len({(g, n) for g, n, _, _ in self.global_cells})

    def state(self) -> dict[str, Any]:
        return {"global_cells": [list(c) for c in sorted(self.global_cells)],
                "recent_episode_cells": list(self.recent_episode_cells),
                "recent_episode_maps": list(self.recent_episode_maps), "episodes": self.episodes,
                "battle_steps": self.battle_steps, "steps": self.steps,
                "unconfirmed_samples": self.unconfirmed_samples}

    def load(self, state: dict[str, Any]) -> None:
        self.global_cells = {(int(c[0]), int(c[1]), int(c[2]), int(c[3])) for c in state["global_cells"]}
        self.recent_episode_cells = deque((int(v) for v in state["recent_episode_cells"]), maxlen=32)
        self.recent_episode_maps = deque((int(v) for v in state["recent_episode_maps"]), maxlen=32)
        self.episodes, self.battle_steps = int(state["episodes"]), int(state["battle_steps"])
        self.steps, self.unconfirmed_samples = int(state["steps"]), int(state["unconfirmed_samples"])


@dataclass
class LearnerState:
    """Everything a resumed run needs besides the networks and the optimiser."""

    obs_rms: RunningMeanStd
    rnd_norm: StreamNormaliser
    epi_norm: StreamNormaliser
    log: ExplorationLog
    rng: np.random.Generator
    sampler: torch.Generator
    env_steps: int = 0
    update: int = 0
    learner_seconds: float = 0.0
    wall_offset: float = 0.0
    resumes: int = 0


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    return value


def _save_checkpoint(path: Path, config: TrainConfig, model: ActorCritic, rnd: RNDModel, optimizer: Optimizer,
                     state: LearnerState, elapsed: float) -> None:
    payload = {"model": _cpu_tree(model.state_dict()), "rnd": _cpu_tree(rnd.state_dict()),
               "optimizer": _cpu_tree(optimizer.state_dict()),
               "obs_rms": state.obs_rms.state(), "rnd_norm": state.rnd_norm.state(),
               "epi_norm": state.epi_norm.state(), "log": state.log.state(),
               "rng": json.dumps(state.rng.bit_generator.state), "sampler": state.sampler.get_state(),
               "env_steps": state.env_steps, "update": state.update, "learner_seconds": state.learner_seconds,
               "wall_seconds": elapsed, "resumes": state.resumes,
               "game": config.game, "seed": config.seed, "init": config.init}
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_checkpoint(path: Path, model: ActorCritic, rnd: RNDModel, optimizer: Optimizer,
                     state: LearnerState) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    rnd.load_state_dict(payload["rnd"])
    optimizer.load_state_dict(payload["optimizer"])
    state.obs_rms.load(payload["obs_rms"])
    state.rnd_norm.load(payload["rnd_norm"])
    state.epi_norm.load(payload["epi_norm"])
    state.log.load(payload["log"])
    state.rng.bit_generator.state = json.loads(payload["rng"])
    state.sampler.set_state(payload["sampler"])
    state.env_steps, state.update = int(payload["env_steps"]), int(payload["update"])
    state.learner_seconds, state.wall_offset = float(payload["learner_seconds"]), float(payload["wall_seconds"])
    state.resumes = int(payload["resumes"]) + 1


def _policy_step(model: ActorCritic, obs: UInt8Array, device: torch.device, sampler: torch.Generator
                 ) -> tuple[NDArray[np.int64], Float64Array, Float64Array]:
    """Sample actions; returns (actions (N,), log-probs (N,), values (N, heads)) with one device sync."""
    with torch.no_grad():
        logits, values = model(pixels_to_device(obs, device))
        packed = torch.cat([torch.softmax(logits.float(), dim=-1), values.float()], dim=1).cpu()
    probs = packed[:, :NUM_ACTIONS]
    sampled = torch.multinomial(probs, 1, generator=sampler).squeeze(1)
    logp = torch.log(probs.gather(1, sampled.unsqueeze(1)).squeeze(1).clamp_min(1e-12))
    return (sampled.numpy().astype(np.int64), logp.numpy().astype(np.float64),
            packed[:, NUM_ACTIONS:].numpy().astype(np.float64))


def train_ppo_rnd(config: TrainConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    torch.set_num_threads(max(1, config.torch_threads))
    device = resolve_device(config.device)
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    env = replace(config.env, detect_transitions=True) if config.place_coef > 0 else config.env
    model = build_actor_critic(env)
    init_note = _initialise(model, config.init)
    model.to(device)
    rnd = RNDModel(1, env.obs_height, env.obs_width).to(device)
    parameters = list(model.parameters()) + list(rnd.predictor.parameters())
    optimizer = make_optimizer(parameters, config.lr)
    decay = None if config.reward_norm_decay >= 1.0 else config.reward_norm_decay
    n_envs, horizon = config.num_envs, config.rollout_len
    state = LearnerState(RunningMeanStd((1, env.obs_height, env.obs_width)),
                         StreamNormaliser(config.gamma_rnd, n_envs, decay, center=config.center_rnd),
                         StreamNormaliser(config.gamma_epi, n_envs, decay), ExplorationLog(),
                         np.random.default_rng(config.seed), torch.Generator().manual_seed(config.seed))
    checkpoint_path = out / "checkpoint.pt"
    resumed = config.resume and checkpoint_path.exists() and not (out / "done.json").exists() \
        and not config.random_policy
    if resumed:
        _load_checkpoint(checkpoint_path, model, rnd, optimizer, state)
        kept = [row for row in read_jsonl(metrics_path) if int(row["update"]) <= state.update]
        metrics_path.write_text("".join(json.dumps(row) + "\n" for row in kept), encoding="utf-8")
    else:
        metrics_path.unlink(missing_ok=True)
    (out / "config.json").write_text(json.dumps({**_config_json(config), "init_note": init_note,
                                                 "device": str(device), "resumed_from_update": state.update,
                                                 "resumes": state.resumes}, indent=2), encoding="utf-8")
    batch = n_envs * horizon
    minibatch = batch // config.minibatches
    warmup = 0 if config.random_policy else config.norm_init_steps * n_envs  # random warm-up counts in budget
    num_updates = max(1, math.ceil((config.total_steps - warmup) / batch))
    start = time.perf_counter() - state.wall_offset
    summary: dict[str, Any] = {}
    novelty = EpisodicNovelty(n_envs, env.obs_height, env.obs_width, config.hash_rows, config.hash_cols,
                              config.hash_levels)
    places = PlaceNovelty(n_envs, env.obs_height, env.obs_width)
    place_total = 0.0
    log, rng = state.log, state.rng

    with SubprocVecEnv(make_configs(env, n_envs),
                       [config.seed * 1000 + 101 * state.resumes + i for i in range(n_envs)]) as venv:
        obs = venv.reset()
        novelty.reset(obs[:, -1])
        if not config.random_policy and not resumed:  # per-pixel observation normaliser from random steps
            init_frames = []
            for _ in range(config.norm_init_steps):
                obs, dones, infos = venv.step(rng.integers(0, NUM_ACTIONS, size=n_envs))
                novelty.step(obs[:, -1], dones)
                log.observe(infos)
                init_frames.append(_newest(obs).astype(np.float32))
                state.env_steps += n_envs
            state.obs_rms.update(np.concatenate(init_frames))
        for update in range(state.update + 1, num_updates + 1):
            buf_obs = np.zeros((horizon, n_envs, *env.obs_shape), dtype=np.uint8)
            buf_next = np.zeros((horizon, n_envs, 1, env.obs_height, env.obs_width), dtype=np.uint8)
            buf_actions = np.zeros((horizon, n_envs), dtype=np.int64)
            buf_logp = np.zeros((horizon, n_envs), dtype=np.float64)
            buf_values = np.zeros((horizon, n_envs, model.value_heads), dtype=np.float64)
            buf_epi = np.zeros((horizon, n_envs), dtype=np.float64)
            buf_dones = np.zeros((horizon, n_envs), dtype=np.bool_)
            action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
            finished: list[int] = []
            for t in range(horizon):
                buf_obs[t] = obs
                if config.random_policy:
                    actions = rng.integers(0, NUM_ACTIONS, size=n_envs).astype(np.int64)
                else:
                    actions, buf_logp[t], buf_values[t] = _policy_step(model, obs, device, state.sampler)
                obs, dones, infos = venv.step(actions)
                buf_next[t] = _newest(obs)
                buf_epi[t] = novelty.step(obs[:, -1], dones)
                if config.place_coef > 0:
                    transitions = np.asarray([bool(info.get("transition", False)) for info in infos])
                    place_bonus = places.step(obs[:, -1], transitions, dones)
                    place_total += float(place_bonus.sum())
                    buf_epi[t] += config.place_coef * place_bonus
                buf_dones[t] = dones
                buf_actions[t] = actions
                np.add.at(action_counts, actions, 1)
                finished.extend(log.observe(infos))
            state.env_steps += batch
            state.update = update
            keys_finished = novelty.pop_finished()
            row: dict[str, Any] = {"update": update, "env_steps": state.env_steps,
                                   "global_cells": len(log.global_cells), "global_maps": log.global_maps(),
                                   "episodes": log.episodes,
                                   "episode_cells_finished": finished,
                                   "episode_cells_recent_mean": float(np.mean(log.recent_episode_cells))
                                   if log.recent_episode_cells else None,
                                   "episode_maps_recent_mean": float(np.mean(log.recent_episode_maps))
                                   if log.recent_episode_maps else None,
                                   "episode_frame_keys_finished": keys_finished,
                                   "epi_new_key_fraction": float(buf_epi.mean()), "place_bonus_total": place_total,
                                   "battle_fraction": log.battle_steps / max(1, log.steps),
                                   "unconfirmed_samples": log.unconfirmed_samples,
                                   "action_hist": (action_counts / batch).round(4).tolist()}
            if not config.random_policy:
                learner_start = time.perf_counter()
                row.update(_ppo_update(config, model, rnd, optimizer, parameters, device, obs, buf_obs,
                                       buf_next, buf_actions, buf_logp, buf_values, buf_epi, buf_dones, state,
                                       minibatch))
                state.learner_seconds += time.perf_counter() - learner_start
            elapsed = time.perf_counter() - start
            row.update({"wall_s": elapsed, "steps_per_s": state.env_steps / max(elapsed, 1e-9),
                        "learner_s_total": state.learner_seconds, "resumes": state.resumes})
            write_jsonl(metrics_path, row)
            summary = row
            if not config.random_policy and (update % config.checkpoint_every == 0 or update == num_updates):
                _save_checkpoint(checkpoint_path, config, model, rnd, optimizer, state, elapsed)
            if config.max_minutes > 0 and elapsed > config.max_minutes * 60:
                break
    if not config.random_policy:
        tmp = out / "final.tmp"
        torch.save({"model": _cpu_tree(model.state_dict()), "rnd": _cpu_tree(rnd.state_dict()),
                    "obs_rms": state.obs_rms.state(), "env_steps": state.env_steps, "game": config.game,
                    "seed": config.seed, "init": config.init}, tmp)
        tmp.replace(out / "final.pt")
    (out / "done.json").write_text(json.dumps({"env_steps": state.env_steps,
                                               "wall_s": time.perf_counter() - start,
                                               "global_cells": len(log.global_cells),
                                               "global_maps": log.global_maps(), "resumes": state.resumes},
                                              indent=2), encoding="utf-8")
    return summary


def _ppo_update(config: TrainConfig, model: ActorCritic, rnd: RNDModel, optimizer: Optimizer,
                parameters: list[nn.Parameter], device: torch.device, last_obs: UInt8Array,
                buf_obs: UInt8Array, buf_next: UInt8Array, buf_actions: NDArray[np.int64],
                buf_logp: Float64Array, buf_values: Float64Array, buf_epi: Float64Array,
                buf_dones: NDArray[np.bool_], state: LearnerState, minibatch: int) -> dict[str, Any]:
    horizon, n_envs = buf_actions.shape
    batch = horizon * n_envs
    rng = state.rng
    # --- RND novelty of the newest frame of each next observation (non-episodic stream)
    next_flat = buf_next.reshape(batch, *buf_next.shape[2:]).astype(np.float32)
    state.obs_rms.update(next_flat)
    mean = torch.from_numpy(state.obs_rms.mean.astype(np.float32)).to(device)
    inv_std = torch.from_numpy((1.0 / np.maximum(state.obs_rms.std, 1e-6)).astype(np.float32)).to(device)
    normalised = ((torch.from_numpy(next_flat).to(device) - mean) * inv_std).clamp_(-5.0, 5.0)
    with torch.no_grad():
        intrinsic = torch.cat([rnd(normalised[i:i + 512]) for i in range(0, batch, 512)]).float().cpu().numpy()
    raw_rnd = intrinsic.astype(np.float64).reshape(horizon, n_envs)
    rnd_rewards = state.rnd_norm(raw_rnd)
    # --- episodic frame-key novelty (episodic stream, cut at episode ends)
    epi_rewards = state.epi_norm(buf_epi, buf_dones)
    with torch.no_grad():
        _, last_values_t = model(pixels_to_device(last_obs, device))
    last_values = last_values_t.float().cpu().numpy().astype(np.float64)
    adv_rnd = gae_non_episodic(rnd_rewards, buf_values[..., 0], last_values[:, 0], config.gamma_rnd,
                               config.gae_lambda)
    adv_epi = gae_episodic(epi_rewards, buf_values[..., 1], last_values[:, 1], buf_dones, config.gamma_epi,
                           config.gae_lambda)
    returns = np.stack([adv_rnd + buf_values[..., 0], adv_epi + buf_values[..., 1]], axis=-1)
    advantages = config.rnd_coef * adv_rnd + config.epi_coef * adv_epi
    # --- flatten and move to the learner device
    obs_t = torch.from_numpy(buf_obs.reshape(batch, *buf_obs.shape[2:])).to(device)
    actions_t = torch.from_numpy(buf_actions.reshape(batch)).to(device)
    old_logp_t = torch.from_numpy(buf_logp.reshape(batch).astype(np.float32)).to(device)
    adv_t = torch.from_numpy(advantages.reshape(batch).astype(np.float32)).to(device)
    ret_t = torch.from_numpy(returns.reshape(batch, 2).astype(np.float32)).to(device)
    stats: dict[str, list[float]] = {k: [] for k in ("pg_loss", "v_loss", "entropy", "rnd_loss", "approx_kl",
                                                     "clipfrac")}
    epochs_run = 0
    for _ in range(config.epochs):
        if config.target_kl > 0 and stats["approx_kl"] and \
                float(np.mean(stats["approx_kl"][-config.minibatches:])) > config.target_kl:
            break  # early stop: protects pretrained features from one destructive update
        epochs_run += 1
        order = rng.permutation(batch)
        for begin in range(0, batch, minibatch):
            index_np = order[begin:begin + minibatch]
            index = torch.from_numpy(index_np).to(device)
            pixels = obs_t.index_select(0, index).float().div_(255.0)
            logits, values = model(pixels)
            logp, entropy = categorical_log_prob_entropy(logits, actions_t.index_select(0, index))
            old_logp = old_logp_t.index_select(0, index)
            adv = adv_t.index_select(0, index)
            centred = adv - adv.mean()
            adv = centred / (centred.pow(2).mean().sqrt() + 1e-8)
            ratio = (logp - old_logp).exp()
            pg_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - config.clip, 1 + config.clip)).mean()
            v_loss = 0.5 * (values - ret_t.index_select(0, index)).pow(2).mean(0).sum()
            ent = entropy.mean()
            mask = torch.from_numpy((rng.random(len(index_np)) < config.rnd_update_proportion)
                                    .astype(np.float32)).to(device)
            rnd_err = rnd(normalised.index_select(0, index))
            rnd_loss = (rnd_err * mask).sum() / mask.sum().clamp_min(1.0)
            loss = pg_loss + config.vf_coef * v_loss - config.ent_coef * ent + rnd_loss
            optimizer.zero_grad(set_to_none=True)
            backward(loss)
            nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, foreach=False)
            optimizer.step()
            with torch.no_grad():
                log_ratio = logp - old_logp
                stats["approx_kl"].append(float(((ratio - 1) - log_ratio).mean().item()))
                stats["clipfrac"].append(float(((ratio - 1).abs() > config.clip).float().mean().item()))
            stats["pg_loss"].append(float(pg_loss.item()))
            stats["v_loss"].append(float(v_loss.item()))
            stats["entropy"].append(float(ent.item()))
            stats["rnd_loss"].append(float(rnd_loss.item()))
    result: dict[str, Any] = {k: float(np.mean(v)) for k, v in stats.items()}
    result.update({"ppo_epochs_run": epochs_run, "rnd_raw_mean": float(raw_rnd.mean()), "rnd_raw_std": float(raw_rnd.std()),
                   "rnd_norm_mean": float(rnd_rewards.mean()), "epi_norm_mean": float(epi_rewards.mean()),
                   "adv_rnd_std": float(adv_rnd.std()), "adv_epi_std": float(adv_epi.std()),
                   "ev_rnd": explained_variance(buf_values[..., 0], returns[..., 0]),
                   "ev_epi": explained_variance(buf_values[..., 1], returns[..., 1]),
                   "value_rnd_mean": float(buf_values[..., 0].mean()),
                   "value_epi_mean": float(buf_values[..., 1].mean())})
    return result


# --------------------------------------------------------------------------- evaluation

@dataclass
class EvalConfig:
    game: str
    out_path: str
    method: str
    policy_path: str = ""  # empty -> uniform random policy
    episodes: int = 16
    num_envs: int = 6
    seed: int = 12345
    device: str = "auto"
    curve_every: int = 32
    torch_threads: int = 2
    env: NavConfig = field(default_factory=lambda: NavConfig("emerald"))


def evaluate(config: EvalConfig) -> dict[str, Any]:
    """Stochastic-policy episodes of fixed length; unique cells per episode and the per-step curve.

    No gradient is ever computed here, so DirectML inference is safe in this process."""
    torch.set_num_threads(max(1, config.torch_threads))
    device = torch.device("cpu") if config.device == "cpu" else resolve_device(config.device)
    env = config.env
    model = load_policy(Path(config.policy_path), env).to(device).eval() if config.policy_path else None
    rng = np.random.default_rng(config.seed)
    sampler = torch.Generator().manual_seed(config.seed)
    n_envs = min(config.num_envs, config.episodes)
    episodes_cells: list[int] = []
    episodes_maps: list[int] = []
    episodes_screens: list[int] = []
    curves: list[list[int]] = []
    current: list[list[int]] = [[] for _ in range(n_envs)]
    action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    battle = 0
    steps = 0
    start = time.perf_counter()
    with SubprocVecEnv(make_configs(env, n_envs), [config.seed + 7919 * i for i in range(n_envs)]) as venv:
        obs = venv.reset()
        active = [True] * n_envs
        started = n_envs
        while any(active):
            if model is None:
                actions = rng.integers(0, NUM_ACTIONS, size=n_envs).astype(np.int64)
            else:
                actions, _, _ = _policy_step(model, obs, device, sampler)
            obs, dones, infos = venv.step(actions)
            for i, info in enumerate(infos):
                if not active[i]:
                    continue
                steps += 1
                action_counts[actions[i]] += 1
                battle += int(bool(info["in_battle"]))
                if info["t"] % config.curve_every == 0:
                    current[i].append(int(info["episode_cells"]))
                if dones[i]:
                    episodes_cells.append(int(info["final_episode_cells"]))
                    episodes_maps.append(int(info["final_episode_maps"]))
                    episodes_screens.append(int(info.get("final_episode_screens", 0)))
                    curves.append(current[i])
                    current[i] = []
                    if started < config.episodes:
                        started += 1
                    else:
                        active[i] = False
    length = min(len(c) for c in curves)
    curve = np.array([c[:length] for c in curves], dtype=np.float64)
    result: dict[str, Any] = {
        "game": config.game, "method": config.method, "policy_path": config.policy_path,
        "episodes": len(episodes_cells), "episode_steps": env.episode_steps, "action_repeat": env.action_repeat,
        "cells": episodes_cells, "maps": episodes_maps,
        "cells_mean": float(np.mean(episodes_cells)), "cells_std": float(np.std(episodes_cells)),
        "maps_mean": float(np.mean(episodes_maps)), "battle_fraction": battle / max(1, steps),
        "screens": episodes_screens, "screens_mean": float(np.mean(episodes_screens)),
        "screens_std": float(np.std(episodes_screens)),
        "action_hist": (action_counts / max(1, steps)).round(4).tolist(),
        "curve_steps": [config.curve_every * (k + 1) for k in range(length)],
        "curve_mean": curve.mean(axis=0).tolist(), "curve_std": curve.std(axis=0).tolist(),
        "seconds": time.perf_counter() - start, "seed": config.seed, "device": str(device),
    }
    out = Path(config.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    tmp.replace(out)
    return result


# --------------------------------------------------------------------------- learner benchmark

def bench_learner(device_name: str = "auto", num_envs: int = 6, rollout_len: int = 128,
                  updates: int = 2) -> dict[str, float]:
    """Seconds per PPO update on synthetic pixels (same shapes as training) and inference latency."""
    config = TrainConfig(game="emerald", out_dir="", num_envs=num_envs, rollout_len=rollout_len,
                         device=device_name)
    device = resolve_device(device_name)
    env = config.env
    model = build_actor_critic(env).to(device)
    rnd = RNDModel(1, env.obs_height, env.obs_width).to(device)
    parameters = list(model.parameters()) + list(rnd.predictor.parameters())
    optimizer = make_optimizer(parameters, config.lr)
    rng = np.random.default_rng(0)
    shape = (rollout_len, num_envs, *env.obs_shape)
    buf_obs = rng.integers(0, 256, size=shape, dtype=np.uint8)
    buf_next = buf_obs[:, :, -1:].copy()
    state = LearnerState(RunningMeanStd((1, env.obs_height, env.obs_width)),
                         StreamNormaliser(config.gamma_rnd, num_envs, 0.9, center=True),
                         StreamNormaliser(config.gamma_epi, num_envs, 0.9), ExplorationLog(), rng,
                         torch.Generator().manual_seed(0))
    timings: list[float] = []
    for _ in range(updates + 1):
        begin = time.perf_counter()
        _ppo_update(config, model, rnd, optimizer, parameters, device, buf_obs[-1], buf_obs, buf_next,
                    rng.integers(0, NUM_ACTIONS, size=(rollout_len, num_envs)),
                    np.full((rollout_len, num_envs), -math.log(NUM_ACTIONS)),
                    np.zeros((rollout_len, num_envs, 2)), (rng.random((rollout_len, num_envs)) < 0.3) * 1.0,
                    np.zeros((rollout_len, num_envs), dtype=np.bool_), state,
                    rollout_len * num_envs // config.minibatches)
        timings.append(time.perf_counter() - begin)
    inference = []
    obs = buf_obs[0]
    for _ in range(20):
        begin = time.perf_counter()
        _policy_step(model, obs, device, state.sampler)
        inference.append(time.perf_counter() - begin)
    return {"update_s": float(np.mean(timings[1:])), "first_update_s": timings[0],
            "samples_per_update": float(rollout_len * num_envs),
            "inference_ms_per_step": float(np.median(inference) * 1e3)}


def with_env(config: NavConfig, **changes: Any) -> NavConfig:
    return replace(config, **changes)
