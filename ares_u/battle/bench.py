"""Throughput measurements: rollout env-steps/s per worker count and learner update time per device."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .model import build_policy, to_numpy
from .ppo import PPOConfig, concat, make_optimizer, ppo_update
from .rollout import WorkerPool
from .training import DEFAULT_MIX, resolve_device


def rollout_benchmark(workers: int, steps: int, method: str = "ppo") -> dict[str, Any]:
    """Env steps (recorded learner decisions) per second with ``workers`` CPU rollout processes."""
    weights = to_numpy(build_policy(method))
    with WorkerPool(workers) as pool:
        pool.broadcast([{"type": "ping"} for _ in range(workers)])
        per_worker = max(64, steps // workers)

        def tasks(iteration: int) -> list[dict[str, Any]]:
            return [{"type": "rollout", "method": method, "weights": weights, "seed": 0, "iteration": iteration,
                     "steps": per_worker, "mix": DEFAULT_MIX, "snapshots": {}, "snapshot_keys": [],
                     "concurrency": 32} for _ in range(workers)]

        pool.broadcast(tasks(0))  # warm-up (imports, caches)
        started = time.perf_counter()
        results = pool.broadcast(tasks(1))
        elapsed = time.perf_counter() - started
    collected = sum(int(result["act"].shape[0]) for result in results)
    return {"workers": workers, "steps": collected, "seconds": elapsed, "steps_per_s": collected / elapsed}


def learner_benchmark(devices: list[str], batch: int, method: str = "ppo") -> list[dict[str, Any]]:
    """Seconds per PPO epoch over ``batch`` samples. DirectML is always measured first in the process."""
    with WorkerPool(4) as pool:
        weights = to_numpy(build_policy(method))
        results = pool.broadcast([{"type": "rollout", "method": method, "weights": weights, "seed": 0,
                                   "iteration": worker, "steps": batch // 4 + 1, "mix": DEFAULT_MIX,
                                   "snapshots": {}, "snapshot_keys": []} for worker in range(4)])
    data = {key: value[:batch] for key, value in concat(results).items()}
    ordered = sorted(devices, key=lambda name: 0 if name == "dml" else 1)
    report: list[dict[str, Any]] = []
    for name in ordered:
        device = resolve_device(name)
        model = build_policy(method).to(device)
        optimizer = make_optimizer(model, 3e-4)
        config = PPOConfig(method=method, epochs=1, target_kl=0.0)  # time full epochs (no early stop)
        ppo_update(model, optimizer, data, config, device, np.random.default_rng(0))
        started = time.perf_counter()
        ppo_update(model, optimizer, data, config, device, np.random.default_rng(1))
        report.append({"device": name, "resolved": str(device), "samples": int(data["act"].shape[0]),
                       "seconds_per_epoch": time.perf_counter() - started})
    return report
