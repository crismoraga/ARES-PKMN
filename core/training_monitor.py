"""Bounded, run-scoped monitoring for real neural training processes."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from typing import cast
import uuid

from core.runtime_view import campaign_liveness, process_identity, read_jsonl_tail

TRAINING_STATUS_LIMIT = 64_000
TRAINING_SUMMARY_LIMIT = 2_000_000


def _atomic_json(path: Path, value: object) -> bool:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temporary.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
        for attempt in range(3):
            try:
                temporary.replace(path)
                return True
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.005)
        return False
    except OSError:
        return False
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _bounded_object(path: Path, maximum_bytes: int) -> dict[str, object]:
    try:
        if path.stat().st_size > maximum_bytes:
            return {}
        with path.open("rb") as stream:
            raw = stream.read(maximum_bytes + 1)
        if len(raw) > maximum_bytes:
            return {}
        value: object = json.loads(raw)
        return cast(dict[str, object], value) if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


class TrainingMonitor:
    """Publish genuine optimizer/evaluation observations without changing training.

    Publication errors are best effort and never replace a model's own error.
    Every sample carries a fresh run_id; old sessions may remain on disk but
    cannot become part of the current run's live curve.
    """

    def __init__(self, root: Path, *, total_updates: int, backend: str = "pending") -> None:
        if type(total_updates) is not int or total_updates < 1:
            raise ValueError("A positive training update budget is required")
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex
        self.status: dict[str, object] = {"run_id": self.run_id, "pid": os.getpid(),
            "process_identity": process_identity(os.getpid()), "started_at": time.time(),
            "backend": backend, "update": 0, "steps": 0, "total_updates": total_updates,
            "vicreg": None, "trajectory_balance": None, "ppo_cycle": None,
            "win_rate": None, "value_loss": None, "error": ""}
        self.publish(stage="initializing")

    def publish(self, *, stage: str, running: bool = True,
                update: int | None = None, backend: str | None = None,
                vicreg: float | None = None, trajectory_balance: float | None = None,
                ppo_cycle: int | None = None, win_rate: float | None = None,
                value_loss: float | None = None, error: str = "") -> dict[str, object]:
        if not stage or len(stage) > 128:
            raise ValueError("Invalid training stage")
        if update is not None:
            if type(update) is not int or update < 0:
                raise ValueError("Invalid training update")
            self.status.update(update=update, steps=update)
        if ppo_cycle is not None:
            self.status["ppo_cycle"] = ppo_cycle
        if backend is not None:
            self.status["backend"] = backend
        for name, metric in (("vicreg", vicreg), ("trajectory_balance", trajectory_balance),
                             ("win_rate", win_rate), ("value_loss", value_loss)):
            if metric is not None:
                self.status[name] = float(metric) if math.isfinite(metric) else None
        now = time.time()
        self.status.update(t=now, published_at=now, stage=stage, phase=stage,
                           running=running, error=error)
        sample = dict(self.status)
        _atomic_json(self.root / "live.json", sample)
        try:
            with (self.root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(sample, allow_nan=False) + "\n")
        except OSError:
            pass
        return sample

    def fail(self, error: BaseException) -> None:
        self.publish(stage="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     running=False, error=f"{type(error).__name__}: {error}")

    def finish(self) -> None:
        self.publish(stage="completed", running=False)


def read_training(root: Path, *, summary_path: Path | None = None,
                  limit: int = 1200, maximum_bytes: int = 2_000_000) -> dict[str, object]:
    """Read bounded status, the current run's samples and the latest saved summary."""
    if not 1 <= limit <= 4000 or not 1024 <= maximum_bytes <= 8_000_000:
        raise ValueError("Invalid bounded training history request")
    raw = _bounded_object(root / "live.json", TRAINING_STATUS_LIMIT)
    status: dict[str, object] = campaign_liveness(raw, stale_after=120.0) if raw else {
        "running": False, "liveness": "absent", "stage": "not_started", "run_id": None}
    run_id = raw.get("run_id")
    samples = read_jsonl_tail(root / "metrics.jsonl", limit=4000, maximum_bytes=maximum_bytes)
    samples = [row for row in samples if isinstance(run_id, str) and row.get("run_id") == run_id][-limit:]
    summary = _bounded_object(summary_path or root.parent / "checkpoints/training_metrics.json",
                              TRAINING_SUMMARY_LIMIT)
    return {"status": status, "samples": samples, "summary": summary}


def write_training_summary(path: Path, value: object) -> None:
    """Publish a complete saved-metrics generation without exposing partial JSON."""
    if not _atomic_json(path, value):
        raise OSError(f"Unable to publish training metrics: {path}")
