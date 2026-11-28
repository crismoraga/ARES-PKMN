"""Imitation warm-start of the entity Transformer (methods ``bc`` and the first phase of ``bc_ppo``).

Transfer/pre-training in the sense of the course: before any RL, the *same* ID-free Transformer
is behaviour-cloned from heuristic experts and only then fine-tuned with PPO.

* Demonstrations: the ARES expert prior (``MLPGenome.prior``, argmax) and Greedy play Gen1
  **training** teams (the RL sampler: Gen1 species, never a held-out species set; never Gen2/Gen3)
  against a mix of prior/Greedy/Random. Every decision of an expert side is recorded with the
  discounted return of the RL reward (terminal +-1 plus potential-based HP shaping).
* Loss: cross-entropy over the *masked legal* actions (illegal logits are already -1e9) against a
  label-smoothed target (``smoothing`` mass spread over the legal actions, so the warm-started
  policy keeps some entropy for PPO), plus ``vf_coef`` x MSE of the value head on the returns.
* Budget: every recorded demonstration decision counts as one environment step, the same unit as
  the RL learners' recorded decisions. ``bc_ppo`` spends ``demo_steps`` of its budget here and the
  rest in PPO; ``bc`` uses only the demonstrations (no RL).

The learner runs on DirectML when available; the workers only play (no ``backward``).
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from .model import EntityTransformer, Weights, build_policy, load_numpy, to_numpy
from .ppo import INPUT_KEYS, _seed_everything, make_optimizer, minibatches, save_checkpoint, set_lr, upload
from .rollout import WorkerPool
from .training import (QUICK_OPPONENTS, QUICK_SPLITS, MetricsLog, Timer, device_label, linear, quick_eval,
                       resolve_device)

DEMO_KEYS = ("field", "entities", "actions", "mask", "species", "act", "ret", "source")
DEFAULT_EXPERTS = {"prior": 0.5, "greedy": 0.5}
DEFAULT_DEMO_OPPONENTS = {"prior": 0.35, "greedy": 0.35, "random": 0.3}


@dataclass
class BCConfig:
    method: str = "bc"
    seed: int = 0
    demo_steps: int = 300_000
    experts: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_EXPERTS))
    opponents: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_DEMO_OPPONENTS))
    epochs: int = 6
    minibatch: int = 4096
    lr: float = 5e-4
    lr_end_frac: float = 0.1
    smoothing: float = 0.1
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    val_fraction: float = 0.05
    gamma: float = 0.99
    workers: int = 14
    concurrency: int = 32
    eval_battles: int = 200
    generation: int = 1              # species pool of the demonstration teams (evolution only)
    device: str = "dml"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def demo_signature(self) -> dict[str, Any]:
        """Settings that determine the BC checkpoint (two runs with equal signatures are interchangeable)."""
        keys = ("seed", "demo_steps", "experts", "opponents", "epochs", "minibatch", "lr", "lr_end_frac",
                "smoothing", "vf_coef", "max_grad_norm", "val_fraction", "gamma")
        return {key: getattr(self, key) for key in keys}


def collect_demonstrations(pool: WorkerPool, config: BCConfig, teachers: dict[str, dict[str, Any]] | None = None
                           ) -> tuple[dict[str, NDArray[Any]], list[dict[str, Any]]]:
    """``teachers``: neural experts/opponents by name (agent specs, e.g. an ``ensemble``) for distillation."""
    per_worker = max(64, config.demo_steps // pool.size)
    results = pool.broadcast([{"type": "demos", "seed": config.seed, "chunk": worker, "steps": per_worker,
                               "experts": config.experts, "opponents": config.opponents, "gamma": config.gamma,
                               "concurrency": config.concurrency, "generation": config.generation,
                               "teachers": teachers or {}} for worker in range(pool.size)])
    data = {key: np.concatenate([result[key] for result in results]) for key in DEMO_KEYS}
    return data, [item for result in results for item in result["stats"]]


def bc_loss(model: EntityTransformer, batch: dict[str, torch.Tensor], smoothing: float, vf_coef: float
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(total, policy cross-entropy, value MSE, argmax predictions) of one minibatch."""
    logits, values = model(*(batch[key] for key in INPUT_KEYS))
    log_probs = torch.log_softmax(logits, dim=-1)
    legal = batch["mask"].float()
    onehot = torch.zeros_like(log_probs).scatter_(1, batch["act"].unsqueeze(1), 1.0)
    target = (1 - smoothing) * onehot + smoothing * legal / legal.sum(dim=-1, keepdim=True).clamp(min=1.0)
    cross_entropy = -(target * log_probs.masked_fill(~batch["mask"], 0.0)).sum(dim=-1).mean()
    value_loss = ((values - batch["ret"]) ** 2).mean()
    return cross_entropy + vf_coef * value_loss, cross_entropy, value_loss, logits.argmax(dim=-1)


def _validate(model: EntityTransformer, data: dict[str, NDArray[Any]], index: NDArray[np.int64], config: BCConfig,
              device: torch.device, names: list[str]) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    hits = np.zeros(len(names))
    counts = np.zeros(len(names))
    with torch.no_grad():
        for start in range(0, len(index), config.minibatch):
            chunk = index[start:start + config.minibatch]
            batch = upload(data, chunk, (*INPUT_KEYS, "act", "ret"), device)
            _, cross_entropy, value_loss, prediction = bc_loss(model, batch, 0.0, config.vf_coef)
            predicted = prediction.cpu().numpy()
            correct = predicted == data["act"][chunk]
            for position, name in enumerate(names):
                selected = data["source"][chunk] == position
                hits[position] += float(correct[selected].sum())
                counts[position] += float(selected.sum())
            totals["val_ce"] += float(cross_entropy.cpu()) * len(chunk)
            totals["val_value_mse"] += float(value_loss.cpu()) * len(chunk)
            totals["val_acc"] += float(correct.sum())
    size = max(1, len(index))
    result = {key: value / size for key, value in totals.items()}
    result.update({f"val_acc_{name}": float(hits[i] / max(1.0, counts[i])) for i, name in enumerate(names)})
    return result


def train_bc(config: BCConfig, out: Path, pool: WorkerPool | None = None, init_weights: Weights | None = None,
             teachers: dict[str, dict[str, Any]] | None = None) -> Path:
    """Collect demonstrations, behaviour-clone, quick-evaluate; writes ``final.pt`` under ``out``.

    The checkpoint metadata records ``demo_steps`` (= environment steps consumed) for budgets. With
    ``init_weights`` and ``teachers`` this is policy distillation into an existing student.
    """
    out.mkdir(parents=True, exist_ok=True)
    _seed_everything(config.seed)
    device = resolve_device(config.device)
    network = build_policy(config.method)
    if init_weights is not None:
        load_numpy(network, init_weights)
    model = network.to(device)
    optimizer = make_optimizer(model, config.lr)
    rng = np.random.default_rng(config.seed)
    log = MetricsLog(out / "metrics.jsonl")
    if log.path.exists():
        log.path.unlink()
    (out / "config.json").write_text(json.dumps({**config.to_json(), "device_used": device_label(device)}, indent=2),
                                     encoding="utf-8")
    owned = pool is None
    workers = pool if pool is not None else WorkerPool(config.workers)
    timer = Timer()
    names = list(config.experts)
    try:
        data, stats = collect_demonstrations(workers, config, teachers)
        demo_steps = int(data["act"].shape[0])
        collect_s = timer.elapsed()
        by_pair: dict[str, list[float]] = defaultdict(list)
        for item in stats:
            by_pair[f"demo_win_{item['expert']}_vs_{item['opponent']}"].append((float(item["outcome"]) + 1) / 2)
        log.write({"kind": "demos", "steps": demo_steps, "episodes": len(stats), "wall": round(collect_s, 2),
                   **{key: float(np.mean(values)) for key, values in by_pair.items()},
                   **{f"demo_share_{name}": float(np.mean(data["source"] == i)) for i, name in enumerate(names)}})
        order = rng.permutation(demo_steps)
        holdout = max(1, int(demo_steps * config.val_fraction))
        val_index, train_index = order[:holdout].astype(np.int64), order[holdout:].astype(np.int64)
        planned = config.epochs * max(1, len(train_index) // max(1, min(config.minibatch, len(train_index))))
        updates = 0
        for epoch in range(config.epochs):
            model.train()
            sums: dict[str, float] = defaultdict(float)
            blocks = minibatches(len(train_index), config.minibatch, rng)
            for index in blocks:
                set_lr(optimizer, linear(config.lr, config.lr * config.lr_end_frac, updates / max(1, planned)))
                batch = upload(data, train_index[index], (*INPUT_KEYS, "act", "ret"), device)
                loss, cross_entropy, value_loss, prediction = bc_loss(model, batch, config.smoothing, config.vf_coef)
                accuracy = (prediction == batch["act"]).float().mean()
                optimizer.zero_grad()
                torch.autograd.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                values = torch.stack([cross_entropy.detach(), value_loss.detach(), accuracy.detach()]).cpu().numpy()
                for key, value in zip(("train_ce", "train_value_mse", "train_acc"), values):
                    sums[key] += float(value)
                updates += 1
            record = {key: value / max(1, len(blocks)) for key, value in sums.items()}
            record.update(_validate(model, data, val_index, config, device, names))
            log.write({"kind": "bc", "epoch": epoch, "steps": demo_steps, "wall": round(timer.elapsed(), 2),
                       **record})
        model.eval()
        if config.eval_battles > 0:
            evaluation = quick_eval(workers, config.method, to_numpy(model), QUICK_SPLITS, QUICK_OPPONENTS,
                                    config.eval_battles)
            log.write({"kind": "eval", "iteration": -1, "steps": demo_steps, "wall": round(timer.elapsed(), 2),
                       **evaluation})
    finally:
        if owned:
            workers.close()
    final = out / "final.pt"
    save_checkpoint(final, model, config.method, demo_steps,
                    {"demo_steps": demo_steps, "wall": timer.elapsed(), "collect_s": collect_s,
                     "device": device_label(device), "signature": config.demo_signature()})
    return final
