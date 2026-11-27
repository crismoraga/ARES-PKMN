"""DQN baseline (Mnih et al., 2015; Double-DQN target): MLP over the flat observation.

Same training split, opponent mix (self = current greedy Q-network, past snapshots, Greedy,
Random), reward and environment-step budget as PPO. CPU workers act epsilon-greedily with the
latest weights (forward only); the learner keeps a replay buffer (float16 observations), a target
network refreshed every ``target_every`` updates and performs ``replay_ratio`` sampled transitions
per new transition. Illegal actions are masked in both the greedy choice and the bootstrap.
Rollouts are pipelined with the replay updates (``pipeline``).
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
import torch.nn.functional as F

from .model import QNetwork, to_numpy
from .obs import ACTIONS, FLAT_DIM
from .ppo import make_optimizer, save_checkpoint
from .rollout import WorkerPool
from .training import (DEFAULT_MIX, QUICK_OPPONENTS, QUICK_SPLITS, League, MetricsLog, Timer, device_label,
                       outcome_stats, quick_eval, resolve_device)

MASKED_Q = -1e9


@dataclass
class DQNConfig:
    method: str = "dqn"
    seed: int = 0
    total_steps: int = 2_000_000
    steps_per_iter: int = 16_384
    workers: int = 14
    concurrency: int = 32
    buffer: int = 200_000
    batch: int = 512
    lr: float = 2.5e-4
    gamma: float = 0.99
    replay_ratio: float = 4.0
    warmup: int = 20_000
    target_every: int = 500
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.4
    max_grad_norm: float = 10.0
    mix: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MIX))
    snapshot_every: int = 10
    league_size: int = 12
    evals: int = 10
    eval_battles: int = 200
    pipeline: bool = True
    device: str = "dml"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def epsilon(self, steps: int) -> float:
        progress = min(1.0, steps / max(1.0, self.eps_fraction * self.total_steps))
        return self.eps_start + progress * (self.eps_end - self.eps_start)


class Replay:
    """Ring buffer of one-step transitions."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.state = np.zeros((capacity, FLAT_DIM), dtype=np.float16)
        self.next = np.zeros((capacity, FLAT_DIM), dtype=np.float16)
        self.next_mask = np.zeros((capacity, ACTIONS), dtype=np.bool_)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.bool_)
        self.size = 0
        self.cursor = 0

    def add(self, data: dict[str, NDArray[Any]]) -> None:
        count = int(data["act"].shape[0])
        for start in range(0, count, self.capacity):
            chunk = slice(start, min(count, start + self.capacity))
            length = chunk.stop - chunk.start
            positions = (self.cursor + np.arange(length)) % self.capacity
            self.state[positions] = data["state"][chunk]
            self.next[positions] = data["next"][chunk]
            self.next_mask[positions] = data["next_mask"][chunk]
            self.act[positions] = data["act"][chunk]
            self.reward[positions] = data["reward"][chunk]
            self.done[positions] = data["done"][chunk]
            self.cursor = int((self.cursor + length) % self.capacity)
            self.size = min(self.capacity, self.size + length)

    def sample(self, rng: np.random.Generator, batch: int) -> dict[str, NDArray[Any]]:
        index = rng.integers(0, self.size, size=batch)
        return {"state": self.state[index].astype(np.float32), "next": self.next[index].astype(np.float32),
                "next_mask": self.next_mask[index], "act": self.act[index], "reward": self.reward[index],
                "done": self.done[index]}


def dqn_loss(online: QNetwork, target: QNetwork, batch: dict[str, NDArray[Any]], gamma: float,
             device: torch.device) -> torch.Tensor:
    """Double-DQN Huber loss with masked bootstrap."""
    state = torch.as_tensor(batch["state"]).to(device)
    following = torch.as_tensor(batch["next"]).to(device)
    mask = torch.as_tensor(batch["next_mask"]).to(device)
    actions = torch.as_tensor(batch["act"]).to(device)
    rewards = torch.as_tensor(batch["reward"]).to(device)
    not_done = torch.as_tensor((~batch["done"]).astype(np.float32)).to(device)
    q = online(state).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        best = online(following).masked_fill(~mask, MASKED_Q).argmax(dim=1, keepdim=True)
        bootstrap = target(following).gather(1, best).squeeze(1)
        goal = rewards + gamma * not_done * bootstrap
    return F.smooth_l1_loss(q, goal)


def train_dqn(config: DQNConfig, out: Path, pool: WorkerPool | None = None) -> Path:
    """Train one seed. With ``pipeline`` the workers collect the next batch (with the pre-update
    weights and the current epsilon) while the learner replays; the replay buffer makes DQN
    off-policy anyway, so the one-iteration lag needs no correction."""
    out.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    online = QNetwork().to(device)
    target = QNetwork().to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = make_optimizer(online, config.lr)
    rng = np.random.default_rng(config.seed)
    replay = Replay(config.buffer)
    log = MetricsLog(out / "metrics.jsonl")
    if log.path.exists():
        log.path.unlink()
    (out / "config.json").write_text(json.dumps({**config.to_json(), "device_used": device_label(device)}, indent=2),
                                     encoding="utf-8")
    owned = pool is None
    workers = pool if pool is not None else WorkerPool(config.workers)
    league = League(config.league_size, workers.size)
    timer = Timer()
    steps, iteration, updates = 0, 0, 0
    eval_every = max(1, config.total_steps // max(1, config.evals))
    next_eval = eval_every
    rollout_time = learn_time = eval_time = 0.0
    epsilons: dict[int, float] = {}

    def evaluate_now(at_iteration: int) -> None:
        nonlocal eval_time
        started = timer.elapsed()
        evaluation = quick_eval(workers, "dqn", to_numpy(online), QUICK_SPLITS, QUICK_OPPONENTS, config.eval_battles)
        eval_time += timer.elapsed() - started
        log.write({"kind": "eval", "iteration": at_iteration, "steps": steps, "wall": round(timer.elapsed(), 2),
                   **evaluation})

    def submit(at_iteration: int) -> None:
        weights = to_numpy(online)
        if at_iteration % config.snapshot_every == 0:
            league.add(f"it{at_iteration}", weights)
        epsilon = config.epsilon(steps)
        epsilons[at_iteration] = epsilon
        per_worker = max(64, config.steps_per_iter // workers.size)
        workers.submit([{"type": "rollout", "method": "dqn", "weights": weights, "seed": config.seed,
                         "iteration": at_iteration, "worker": worker, "steps": per_worker, "mix": config.mix,
                         "snapshots": league.shipment(worker), "snapshot_keys": list(league.order),
                         "gamma": config.gamma, "epsilon": epsilon, "concurrency": config.concurrency}
                        for worker in range(workers.size)])

    try:
        evaluate_now(-1)
        if steps < config.total_steps:
            submit(0)
        while workers.pending:
            started = timer.elapsed()
            results = workers.collect()
            rollout_time += timer.elapsed() - started
            fresh = 0
            for result in results:
                replay.add(result)
                fresh += int(result["act"].shape[0])
            steps += fresh
            done = steps >= config.total_steps
            eval_due = steps >= next_eval or done
            if config.pipeline and not done and not eval_due:
                submit(iteration + 1)
            started = timer.elapsed()
            losses: list[float] = []
            if replay.size >= config.warmup:
                online.train()
                for _ in range(max(1, int(fresh * config.replay_ratio / config.batch))):
                    loss: torch.Tensor = dqn_loss(online, target, replay.sample(rng, config.batch), config.gamma, device)
                    optimizer.zero_grad()
                    torch.autograd.backward(loss)
                    torch.nn.utils.clip_grad_norm_(online.parameters(), config.max_grad_norm)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                    updates += 1
                    if updates % config.target_every == 0:
                        target.load_state_dict(online.state_dict())
                online.eval()
            learn_time += timer.elapsed() - started
            log.write({"kind": "train", "iteration": iteration, "steps": steps, "wall": round(timer.elapsed(), 2),
                       "batch": fresh, "epsilon": epsilons.pop(iteration, config.epsilon(steps)), "updates": updates,
                       "replay": replay.size, "td_loss": float(np.mean(losses)) if losses else None,
                       "rollout_s": round(rollout_time, 2), "learn_s": round(learn_time, 2),
                       "eval_s": round(eval_time, 2), **outcome_stats([s for r in results for s in r["stats"]])})
            if eval_due:
                next_eval += eval_every
                evaluate_now(iteration)
                save_checkpoint(out / "checkpoints" / f"step{steps:09d}.pt", online, "dqn", steps,
                                {"iteration": iteration})
            if not workers.pending and not done:
                submit(iteration + 1)
            iteration += 1
    finally:
        if owned:
            workers.close()
        elif workers.pending:
            try:
                workers.collect()
            except RuntimeError:
                pass
    final = out / "final.pt"
    save_checkpoint(final, online, "dqn", steps, {"iterations": iteration, "wall": timer.elapsed(),
                                                  "rollout_s": rollout_time, "learn_s": learn_time,
                                                  "eval_s": eval_time, "updates": updates,
                                                  "device": device_label(device)})
    return final
