"""PPO self-play with a league for the entity Transformer (methods ``ppo``, ``ppo_ids`` and ``bc_ppo``).

Each iteration ships the current weights (NumPy) to the CPU rollout workers, which play Gen1
training battles against the opponent mix (``self`` = current policy on both sides, both sides
recorded; ``snapshot`` = a past frozen policy; ``greedy``; ``random``), compute GAE in the worker
and return the batch. The learner (DirectML when available) runs clipped PPO with an entropy bonus
and a value loss; it is the only process that calls ``backward``. Rewards are described in ``env``.

Stabilisation (after the first pilot showed approx-KL climbing to 0.04 with 27% clipped samples):

* linear learning-rate annealing (``lr`` -> ``lr * lr_end_frac``) and entropy-coefficient annealing
  (``ent_coef`` -> ``ent_coef_end``) over the run's own step budget;
* target-KL early stopping: the update stops at the first minibatch whose approximate KL to the
  policy at the *start of this update* exceeds ``target_kl`` (that minibatch is not applied). The KL
  is measured against the start-of-update policy, not the behaviour policy, so the pipelining lag
  does not cancel whole updates;
* a linear learning-rate warm-up over the first ``lr_warmup`` optimiser steps of the run: a fresh
  Adam moves every weight by ~lr on its first steps, which alone exceeded the KL target;
* advantages normalised per batch; value loss clipped like the policy (``clip_value``);
* fixed-shape minibatches (the remainder of each epoch's permutation is dropped), so DirectML does
  not recompile kernels for a new tail shape every iteration.

Pipelining (``pipeline=True``): the workers collect batch ``k+1`` with the pre-update weights while
the learner trains on batch ``k``; the stored behaviour log-probabilities keep the importance ratio
exact, the cost is a one-iteration policy lag (logged as ``kl_first``: the KL of the first
minibatch before any step of the update).

``bc_ppo`` starts from a behaviour-cloned checkpoint (``ares_u.battle.bc``); the demonstration
steps count toward the same environment-step budget (``step_offset``).
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.optim.optimizer import Optimizer

from .model import EntityTransformer, Weights, build_policy, load_numpy, to_numpy
from .rollout import WorkerPool
from .training import (DEFAULT_MIX, QUICK_OPPONENTS, QUICK_SPLITS, League, MetricsLog, Timer, device_label, linear,
                       outcome_stats, quick_eval, resolve_device)

TENSOR_KEYS = ("field", "entities", "actions", "mask", "species", "act", "logp", "value", "adv", "ret")
INPUT_KEYS = ("field", "entities", "actions", "mask", "species")
STAT_KEYS = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac")


@dataclass
class PPOConfig:
    method: str = "ppo"
    seed: int = 0
    total_steps: int = 2_000_000     # environment-step budget of the whole run (incl. ``step_offset``)
    step_offset: int = 0             # steps spent before PPO starts (bc_ppo: the demonstrations)
    steps_per_iter: int = 32_768
    workers: int = 14
    concurrency: int = 32
    epochs: int = 4
    minibatch: int = 4096
    lr: float = 2.5e-4
    lr_end_frac: float = 0.1
    clip: float = 0.2
    clip_value: bool = True
    ent_coef: float = 0.01
    ent_coef_end: float = 0.002
    vf_coef: float = 0.5
    target_kl: float = 0.02          # per-update KL limit (vs. the start-of-update policy); 0 disables
    lr_warmup: int = 50              # optimiser steps of linear LR warm-up at the start of the run
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    lam: float = 0.95
    mix: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MIX))
    snapshot_every: int = 5
    league_size: int = 12
    evals: int = 10
    eval_battles: int = 200
    pipeline: bool = True
    init: str = ""                   # checkpoint the policy starts from ("" = random init)
    device: str = "dml"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def progress(self, steps: int) -> float:
        """Fraction of the run's own PPO budget already used (drives the schedules)."""
        span = max(1, self.total_steps - self.step_offset)
        return (steps - self.step_offset) / span


def concat(results: list[dict[str, Any]]) -> dict[str, NDArray[Any]]:
    return {key: np.concatenate([result[key] for result in results]) for key in TENSOR_KEYS}


def minibatches(size: int, minibatch: int, rng: np.random.Generator) -> list[NDArray[np.int64]]:
    """Equal-size index blocks of a fresh permutation (the remainder is dropped this epoch)."""
    width = max(1, min(minibatch, size))
    order = rng.permutation(size)
    count = size // width
    return [order[block * width:(block + 1) * width].astype(np.int64) for block in range(count)]


def upload(data: dict[str, NDArray[Any]], index: NDArray[np.int64], keys: tuple[str, ...],
           device: torch.device) -> dict[str, torch.Tensor]:
    """CPU-side gather of one minibatch, then a single host-to-device copy per array."""
    return {key: torch.as_tensor(np.ascontiguousarray(data[key][index])).to(device) for key in keys}


def explained_variance(values: NDArray[Any], returns: NDArray[Any]) -> float:
    variance = float(np.var(returns))
    return float("nan") if variance < 1e-12 else 1.0 - float(np.var(returns - values)) / variance


def start_log_probs(model: EntityTransformer, data: dict[str, NDArray[Any]], minibatch: int,
                    device: torch.device) -> NDArray[np.float32]:
    """Log-probabilities of the taken actions under the current weights (no grad, fixed-shape chunks)."""
    size = int(data["act"].shape[0])
    width = max(1, min(minibatch, size))
    padded = np.zeros(-(-size // width) * width, dtype=np.int64)
    padded[:size] = np.arange(size)
    output = np.empty(len(padded), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(padded), width):
            index = padded[start:start + width]
            batch = upload(data, index, (*INPUT_KEYS, "act"), device)
            logits, _ = model(*(batch[key] for key in INPUT_KEYS))
            chosen = torch.log_softmax(logits, dim=-1).gather(1, batch["act"].unsqueeze(1)).squeeze(1)
            output[start:start + width] = chosen.cpu().numpy()
    return output[:size]


def optimizer_steps(optimizer: Optimizer) -> int:
    """Optimiser steps taken so far (read from the Adam state of the first parameter)."""
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            return int(state.get("step", 0))
    return 0


def ppo_update(model: EntityTransformer, optimizer: Optimizer, data: dict[str, NDArray[Any]],
               config: PPOConfig, device: torch.device, rng: np.random.Generator,
               ent_coef: float | None = None, lr: float | None = None) -> dict[str, float]:
    """Clipped PPO epochs over one batch with target-KL early stopping.

    The ratio is taken against the stored behaviour log-probabilities (exact importance weights
    under pipelining); early stopping uses the KL to the policy at the start of this update.
    Returns the mean losses, entropy, approx-KL (to the behaviour policy) and clip fraction of the
    applied minibatches, plus ``kl_first`` (behaviour -> start-of-update KL: the policy lag),
    ``kl_update`` (start-of-update -> final KL), ``updates_frac`` (applied / planned minibatches),
    ``epochs_run`` and ``explained_var`` (of the behaviour values).
    """
    coefficient = config.ent_coef if ent_coef is None else ent_coef
    base_lr = config.lr if lr is None else lr
    prepared = dict(data)
    prepared["adv"] = ((data["adv"] - data["adv"].mean()) / (data["adv"].std() + 1e-8)).astype(np.float32)
    model.eval()
    prepared["start_logp"] = start_log_probs(model, data, config.minibatch, device)
    lag = data["logp"] - prepared["start_logp"]  # log(behaviour / start-of-update)
    kl_first = float(np.mean(np.expm1(-lag) + lag))
    model.train()
    size = int(data["act"].shape[0])
    planned = config.epochs * max(1, size // max(1, min(config.minibatch, size)))
    sums = dict.fromkeys(STAT_KEYS, 0.0)
    updates = 0
    epochs_run = 0
    kl_update = 0.0
    taken = optimizer_steps(optimizer)
    keys = (*TENSOR_KEYS, "start_logp")
    stopped = False
    for _ in range(config.epochs):
        epochs_run += 1
        for index in minibatches(size, config.minibatch, rng):
            batch = upload(prepared, index, keys, device)
            logits, values = model(*(batch[key] for key in INPUT_KEYS))
            log_probs = torch.log_softmax(logits, dim=-1)
            new_logp = log_probs.gather(1, batch["act"].unsqueeze(1)).squeeze(1)
            log_ratio = new_logp - batch["logp"]
            ratio = torch.exp(log_ratio)
            adv = batch["adv"]
            surrogate = torch.min(ratio * adv, torch.clamp(ratio, 1 - config.clip, 1 + config.clip) * adv)
            policy_loss = -surrogate.mean()
            entropy = -(torch.softmax(logits, dim=-1) * log_probs).sum(dim=-1).mean()
            returns = batch["ret"]
            if config.clip_value:
                old_values = batch["value"]
                clipped_values = old_values + torch.clamp(values - old_values, -config.clip, config.clip)
                value_loss = 0.5 * torch.max((values - returns) ** 2, (clipped_values - returns) ** 2).mean()
            else:
                value_loss = 0.5 * ((values - returns) ** 2).mean()
            loss = policy_loss + config.vf_coef * value_loss - coefficient * entropy
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean()
                drift = new_logp - batch["start_logp"]
                update_kl = (torch.exp(drift) - 1 - drift).mean()
                clip_frac = ((ratio - 1).abs() > config.clip).float().mean()
                stats = torch.stack([policy_loss.detach(), value_loss.detach(), entropy.detach(), approx_kl,
                                     clip_frac, update_kl]).cpu().numpy()
            kl_update = float(stats[5])
            if config.target_kl > 0 and kl_update > config.target_kl:
                stopped = True
                break
            warm = min(1.0, (taken + updates + 1) / max(1, config.lr_warmup))
            set_lr(optimizer, base_lr * warm)
            optimizer.zero_grad()
            torch.autograd.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            for key, value in zip(STAT_KEYS, stats):
                sums[key] += float(value)
            updates += 1
        if stopped:
            break
    model.eval()
    result = {key: value / max(1, updates) for key, value in sums.items()}
    result.update({"kl_first": kl_first, "kl_update": kl_update, "updates_frac": updates / max(1, planned),
                   "epochs_run": float(epochs_run), "explained_var": explained_variance(data["value"], data["ret"])})
    return result


class PlainAdam(Optimizer):
    """Adam written with mul_/addcmul_/addcdiv_ only.

    ``torch.optim.Adam(foreach=False)`` works on DirectML but its ``lerp_`` falls back to the CPU for
    every parameter on every step; these in-place ops all run on the device. Same update rule.
    """

    def __init__(self, params: Any, lr: float = 3e-4, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8) -> None:
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps})

    @torch.no_grad()
    def step(self, closure: None = None) -> None:  # type: ignore[override]  # closure-free optimiser
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = int(state["step"])
                grad = parameter.grad
                state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                correction1 = 1 - beta1 ** step
                correction2 = 1 - beta2 ** step
                denominator = (state["exp_avg_sq"].sqrt() / math.sqrt(correction2)).add_(group["eps"])
                parameter.addcdiv_(state["exp_avg"], denominator, value=-group["lr"] / correction1)


def make_optimizer(model: torch.nn.Module, lr: float) -> Optimizer:
    return PlainAdam(model.parameters(), lr=lr, eps=1e-5)


def set_lr(optimizer: Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def save_checkpoint(path: Path, model: torch.nn.Module, method: str, steps: int, extra: dict[str, Any]) -> None:
    """Atomic write (temporary file + rename), so a killed run never leaves a truncated checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().to("cpu").clone() for name, tensor in model.state_dict().items()}
    temporary = path.with_name(path.name + ".tmp")
    torch.save({"method": method, "steps": steps, "state_dict": state, "meta": json.dumps(extra)}, temporary)
    temporary.replace(path)


def load_weights(path: Path) -> tuple[str, Weights, dict[str, Any]]:
    """(method, NumPy weights, metadata) of a checkpoint written by ``save_checkpoint``."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    weights = {name: tensor.float().numpy() for name, tensor in raw["state_dict"].items()}
    meta: dict[str, Any] = json.loads(str(raw.get("meta", "{}")))
    meta.setdefault("steps", int(raw.get("steps", 0)))
    return str(raw["method"]), weights, meta


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_ppo(config: PPOConfig, out: Path, pool: WorkerPool | None = None,
              init_weights: Weights | None = None) -> Path:
    """Train one seed; writes ``metrics.jsonl``, ``config.json`` and checkpoints under ``out``.

    Any previous ``metrics.jsonl`` in ``out`` is replaced (an unfinished run restarts from scratch).
    """
    out.mkdir(parents=True, exist_ok=True)
    _seed_everything(config.seed)
    device = resolve_device(config.device)
    network = build_policy(config.method)
    if init_weights is not None:
        load_numpy(network, init_weights)
    model = network.to(device)
    model.eval()
    optimizer = make_optimizer(model, config.lr)
    rng = np.random.default_rng(config.seed)
    log = MetricsLog(out / "metrics.jsonl")
    if log.path.exists():
        log.path.unlink()
    (out / "config.json").write_text(json.dumps({**config.to_json(), "device_used": device_label(device)}, indent=2),
                                     encoding="utf-8")
    owned = pool is None
    workers = pool if pool is not None else WorkerPool(config.workers)
    league = League(config.league_size, workers.size)
    timer = Timer()
    steps, iteration = config.step_offset, 0
    eval_every = max(1, (config.total_steps - config.step_offset) // max(1, config.evals))
    next_eval = config.step_offset + eval_every
    rollout_time = learn_time = eval_time = 0.0

    def evaluate_now(at_iteration: int) -> None:
        nonlocal eval_time
        started = timer.elapsed()
        evaluation = quick_eval(workers, config.method, to_numpy(model), QUICK_SPLITS, QUICK_OPPONENTS,
                                config.eval_battles)
        eval_time += timer.elapsed() - started
        log.write({"kind": "eval", "iteration": at_iteration, "steps": steps, "wall": round(timer.elapsed(), 2),
                   **evaluation})

    def submit(at_iteration: int) -> None:
        weights = to_numpy(model)
        if at_iteration % config.snapshot_every == 0:
            league.add(f"it{at_iteration}", weights)
        per_worker = max(64, config.steps_per_iter // workers.size)
        workers.submit([{"type": "rollout", "method": config.method, "weights": weights, "seed": config.seed,
                         "iteration": at_iteration, "worker": worker, "steps": per_worker, "mix": config.mix,
                         "snapshots": league.shipment(worker), "snapshot_keys": list(league.order),
                         "gamma": config.gamma, "lam": config.lam, "concurrency": config.concurrency}
                        for worker in range(workers.size)])

    try:
        evaluate_now(-1)  # starting point of the learning curve (random init or the BC policy)
        if steps < config.total_steps:
            submit(0)
        while workers.pending:
            started = timer.elapsed()
            results = workers.collect()
            rollout_time += timer.elapsed() - started
            data = concat(results)
            batch_steps = int(data["act"].shape[0])
            progress = config.progress(steps)
            steps += batch_steps
            done = steps >= config.total_steps
            eval_due = steps >= next_eval or done
            if config.pipeline and not done and not eval_due:
                submit(iteration + 1)  # behaviour policy = the pre-update weights (one-iteration lag)
            lr = linear(config.lr, config.lr * config.lr_end_frac, progress)
            ent_coef = linear(config.ent_coef, config.ent_coef_end, progress)
            started = timer.elapsed()
            losses = ppo_update(model, optimizer, data, config, device, rng, ent_coef, lr)
            learn_time += timer.elapsed() - started
            log.write({"kind": "train", "iteration": iteration, "steps": steps, "wall": round(timer.elapsed(), 2),
                       "batch": batch_steps, "lr": lr, "ent_coef": ent_coef, "rollout_s": round(rollout_time, 2),
                       "learn_s": round(learn_time, 2), "eval_s": round(eval_time, 2), **losses,
                       **outcome_stats([s for r in results for s in r["stats"]])})
            if eval_due:
                next_eval += eval_every
                evaluate_now(iteration)
                save_checkpoint(out / "checkpoints" / f"step{steps:09d}.pt", model, config.method, steps,
                                {"iteration": iteration})
            if not workers.pending and not done:
                submit(iteration + 1)
            iteration += 1
    finally:
        if owned:
            workers.close()
        elif workers.pending:  # a failed update must not leave a shared pool with a pending batch
            try:
                workers.collect()
            except RuntimeError:
                pass
    final = out / "final.pt"
    save_checkpoint(final, model, config.method, steps,
                    {"iterations": iteration, "wall": timer.elapsed(), "rollout_s": rollout_time,
                     "learn_s": learn_time, "eval_s": eval_time, "device": device_label(device),
                     "step_offset": config.step_offset, "init": config.init})
    return final
