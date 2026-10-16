"""Real writer lifecycle and bounded read-only training API, without model runs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Any, cast
from urllib.request import urlopen

import pytest

from core import runtime_view
from core.training_monitor import TrainingMonitor, read_training, write_training_summary
from scripts import train_neural_modules
from scripts.dashboard import MonitoringStation, serve


def test_writer_publishes_current_metrics_completed_and_failure(tmp_path: Path) -> None:
    monitor = TrainingMonitor(tmp_path, total_updates=136)
    initial = json.loads((tmp_path / "live.json").read_text())
    assert initial["pid"] == os.getpid() and initial["process_identity"]
    assert initial["stage"] == "initializing" and initial["running"]
    monitor.publish(stage="world_and_team", update=16, backend="cpu", vicreg=3.2, trajectory_balance=8.1)
    monitor.publish(stage="ppo", update=24, ppo_cycle=1, win_rate=0.75, value_loss=0.3)
    monitor.finish()
    snapshot = read_training(tmp_path)
    status = cast(dict[str, Any], snapshot["status"])
    samples = cast(list[dict[str, Any]], snapshot["samples"])
    assert status["stage"] == status["phase"] == "completed" and not status["running"]
    assert status["steps"] == status["update"] == 24 and status["total_updates"] == 136
    assert status["vicreg"] == 3.2 and status["trajectory_balance"] == 8.1
    assert status["ppo_cycle"] == 1 and status["win_rate"] == 0.75 and status["value_loss"] == 0.3
    assert len(samples) == 4 and all(sample["run_id"] == monitor.run_id for sample in samples)
    assert all(isinstance(sample["t"], float) for sample in samples)
    monitor.fail(RuntimeError("optimizer failed"))
    failed = json.loads((tmp_path / "live.json").read_text())
    assert not failed["running"] and failed["stage"] == "failed"
    assert failed["error"] == "RuntimeError: optimizer failed"


def test_new_run_never_adopts_previous_sessions_samples(tmp_path: Path) -> None:
    old = TrainingMonitor(tmp_path, total_updates=10)
    old.publish(stage="ppo", update=7, win_rate=1.0)
    old.finish()
    current = TrainingMonitor(tmp_path, total_updates=20)
    current.publish(stage="world_and_team", update=1, vicreg=6.0)
    snapshot = read_training(tmp_path)
    samples = cast(list[dict[str, Any]], snapshot["samples"])
    assert len(samples) == 2 and all(sample["run_id"] == current.run_id for sample in samples)
    assert old.run_id != current.run_id
    assert not any(sample["win_rate"] == 1.0 for sample in samples)
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 5


def test_training_reader_detects_dead_pid_without_trusting_saved_running(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    TrainingMonitor(tmp_path, total_updates=8)
    monkeypatch.setattr(runtime_view, "process_identity", lambda pid: None)
    status = cast(dict[str, Any], read_training(tmp_path)["status"])
    assert status["reported_running"] is True and status["running"] is False
    assert status["liveness"] == "dead"


def test_bounded_reader_ignores_oversized_json_and_incomplete_history(tmp_path: Path) -> None:
    monitor = TrainingMonitor(tmp_path / "training", total_updates=8)
    current = json.loads((monitor.root / "live.json").read_text())
    (monitor.root / "metrics.jsonl").write_bytes(b"x" * 20_000 + b"\n" + json.dumps(current).encode() + b"\n{incomplete")
    summary = tmp_path / "huge.json"
    summary.write_bytes(b"x" * 2_000_001)
    snapshot = read_training(monitor.root, maximum_bytes=2048, summary_path=summary)
    assert len(cast(list[object], snapshot["samples"])) == 1 and snapshot["summary"] == {}
    (monitor.root / "live.json").write_bytes(b"x" * 64_001)
    absent = read_training(monitor.root)
    assert absent["samples"] == [] and cast(dict[str, Any], absent["status"])["liveness"] == "absent"


def test_monitor_windows_replace_races_do_not_interrupt_model_work(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monitor = TrainingMonitor(tmp_path, total_updates=8)
    original = (tmp_path / "live.json").read_bytes()
    attempts = 0

    def denied(self: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        raise PermissionError("reader owns previous generation")

    monkeypatch.setattr(Path, "replace", denied)
    sample = monitor.publish(stage="world_and_team", update=1, vicreg=4.5)
    assert sample["update"] == 1 and attempts == 3
    assert (tmp_path / "live.json").read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_nonfinite_diagnostic_still_produces_strict_json(tmp_path: Path) -> None:
    monitor = TrainingMonitor(tmp_path, total_updates=8)
    monitor.publish(stage="world_and_team", update=1, vicreg=float("nan"), trajectory_balance=float("inf"))
    raw = (tmp_path / "live.json").read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    assert json.loads(raw)["vicreg"] is None


def test_training_endpoint_serves_real_run_and_saved_summary(tmp_path: Path) -> None:
    root = tmp_path / "training"
    monitor = TrainingMonitor(root, total_updates=8)
    monitor.publish(stage="world_and_team", update=1, vicreg=4.5)
    summary = {"run_id": "previous_completed_run", "vicreg": {"initial": 7, "final": 2}}
    summary_path = tmp_path / "checkpoints/training_metrics.json"
    summary_path.parent.mkdir()
    write_training_summary(summary_path, summary)
    station = MonitoringStation(tmp_path / "campaigns", training_root=root)
    assert cast(dict[str, Any], station.training()["status"])["run_id"] == monitor.run_id
    server = serve(tmp_path / "campaigns", "native", "127.0.0.1", 0, training_root=root)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_address[1]}/api/training?limit=1", timeout=3) as response:
            value = json.loads(response.read())
            assert response.status == 200 and response.headers["Cache-Control"] == "no-store"
        assert value["status"]["pid"] == os.getpid() and value["status"]["running"] is True
        assert value["status"]["run_id"] == monitor.run_id and len(value["samples"]) == 1
        assert value["summary"] == summary
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
    assert not thread.is_alive()


def test_train_lifecycle_failure_and_completion_without_running_models(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_neural_modules, "ROOT", tmp_path)

    def failed(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("actual model error")

    monkeypatch.setattr(train_neural_modules, "_train", failed)
    with pytest.raises(RuntimeError, match="actual model error"):
        train_neural_modules.train(steps=8, max_cycles=1)
    root = tmp_path / "artifacts/training"
    assert json.loads((root / "live.json").read_text())["stage"] == "failed"
    monkeypatch.setattr(train_neural_modules, "_train", lambda *args: {"accepted": True})
    assert train_neural_modules.train(steps=8, max_cycles=1) == {"accepted": True}
    snapshot = read_training(root)
    assert cast(dict[str, Any], snapshot["status"])["stage"] == "completed"
    assert len(cast(list[object], snapshot["samples"])) == 2
