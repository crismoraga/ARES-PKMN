"""Shared training machinery: device selection, the opponent league, logging and quick evaluations."""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import Weights
from .rollout import WorkerPool

DEFAULT_MIX = {"self": 0.35, "snapshot": 0.25, "greedy": 0.3, "random": 0.1}


def resolve_device(name: str) -> torch.device:
    """``dml`` uses torch-directml when it is installed (falls back to CPU otherwise)."""
    if name == "dml":
        try:
            import torch_directml

            device: torch.device = torch_directml.device()
            return device
        except ImportError:
            return torch.device("cpu")
    return torch.device(name)


def device_label(device: torch.device) -> str:
    return "dml" if device.type == "privateuseone" else device.type


@dataclass
class League:
    """Frozen snapshots of the learner; each worker receives a snapshot's weights only once."""
    capacity: int
    workers: int
    snapshots: dict[str, Weights] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    sent: list[set[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.sent = [set() for _ in range(self.workers)]

    def add(self, key: str, weights: Weights) -> None:
        self.snapshots[key] = weights
        self.order.append(key)
        while len(self.order) > self.capacity:
            oldest = self.order.pop(0)
            del self.snapshots[oldest]

    def shipment(self, worker: int) -> dict[str, Weights]:
        fresh = {key: self.snapshots[key] for key in self.order if key not in self.sent[worker]}
        self.sent[worker].update(fresh)
        return fresh


class MetricsLog:
    """Append-only JSON lines (one object per iteration or evaluation)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=_json_default) + "\n")


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not serialisable: {type(value)}")


def outcome_stats(stats: list[dict[str, Any]]) -> dict[str, float]:
    """Training win rate per opponent kind ((wins + draws/2) / episodes) and mean turns."""
    by_kind: dict[str, list[float]] = defaultdict(list)
    turns: list[int] = []
    for item in stats:
        by_kind[str(item["opponent"])].append(float(item["outcome"]))
        turns.append(int(item["turns"]))
    result: dict[str, float] = {f"train_win_{kind}": float(np.mean([(o + 1) / 2 for o in outcomes]))
                                for kind, outcomes in by_kind.items()}
    result.update({f"train_episodes_{kind}": float(len(outcomes)) for kind, outcomes in by_kind.items()})
    result["mean_turns"] = float(np.mean(turns)) if turns else 0.0
    result["episodes"] = float(len(stats))
    return result


def split_indices(count: int, chunks: int) -> list[list[int]]:
    """Contiguous chunks of even size so the two battles of a team pair stay together."""
    pairs = math.ceil(count / 2)
    per = max(1, math.ceil(pairs / max(1, chunks)))
    result: list[list[int]] = []
    for start in range(0, pairs, per):
        chunk = [index for pair in range(start, min(pairs, start + per)) for index in (2 * pair, 2 * pair + 1)]
        result.append([index for index in chunk if index < count])
    return [chunk for chunk in result if chunk]


QUICK_SPLITS = ("gen1_train", "gen3")
QUICK_OPPONENTS = ("greedy", "prior")
# In-training curves use a disjoint battle-index range (team pairs >= 500k), never the final test
# battles (indices 0..N-1). No checkpoint/hyperparameter is selected on them: runs keep final.pt.
VALIDATION_OFFSET = 1_000_000


def quick_eval(pool: WorkerPool, method: str, weights: Weights, splits: tuple[str, ...],
               opponents: str | tuple[str, ...], battles: int, greedy: bool = True) -> dict[str, float]:
    """Score of the given weights against heuristic opponents on each split (keys ``eval_<opp>_<split>``).

    Validation battles: indices ``VALIDATION_OFFSET .. VALIDATION_OFFSET + battles - 1`` (the same for
    every run), disjoint from the final evaluation set.
    """
    names = (opponents,) if isinstance(opponents, str) else opponents
    tasks: list[dict[str, Any]] = []
    owners: list[str] = []
    for opponent in names:
        for split in splits:
            for raw in split_indices(battles, pool.size):
                chunk = [VALIDATION_OFFSET + index for index in raw]
                tasks.append({"type": "evaluate", "split": split, "indices": chunk, "seed": 0,
                              "agent": {"method": method, "weights": weights, "greedy": greedy},
                              "opponent": {"heuristic": opponent}})
                owners.append(f"eval_{opponent}_{split}")
    results = pool.map(tasks)
    totals: dict[str, list[float]] = defaultdict(list)
    for key, records in zip(owners, results):
        totals[key].extend((float(record["outcome"]) + 1) / 2 for record in records)
    return {key: float(np.mean(values)) for key, values in totals.items()}


def linear(start: float, end: float, progress: float) -> float:
    """Linear schedule from ``start`` (progress 0) to ``end`` (progress 1), clamped."""
    fraction = min(1.0, max(0.0, progress))
    return start + (end - start) * fraction


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start
