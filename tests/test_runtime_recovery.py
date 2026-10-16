"""Worker ownership, truthful liveness, publication races and bounded recovery."""
from __future__ import annotations

import hashlib
from collections.abc import Callable
import json
import multiprocessing
import os
from multiprocessing.connection import Connection
from pathlib import Path
import time
from typing import Any, cast

import pytest

from core import runtime_view
from core.runtime_view import LiveCampaignReader, RGB_BYTES, campaign_liveness, process_identity
from scripts import run_campaign as campaign


def test_campaign_lock_rejects_second_writer_and_releases_after_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        with campaign.CampaignWorkerLock(tmp_path):
            with pytest.raises(campaign.CampaignAlreadyRunning):
                with campaign.CampaignWorkerLock(tmp_path):
                    pytest.fail("second writer acquired campaign")
            raise ValueError("worker failed")
    with campaign.CampaignWorkerLock(tmp_path):
        assert (tmp_path / "worker.lock").is_file()


def test_current_process_identity_and_stale_heartbeat_are_distinct() -> None:
    identity = process_identity(os.getpid())
    assert identity is not None
    live: dict[str, object] = {"pid": os.getpid(), "process_identity": identity,
                               "running": True, "published_at": time.time()}
    assert campaign_liveness(live)["running"] is True
    assert campaign_liveness({**live, "process_identity": "another-process"})["liveness"] == "dead"
    assert campaign_liveness({**live, "published_at": time.time() - 120})["liveness"] == "unresponsive"
    assert campaign_liveness({**live, "running": False})["liveness"] == "stopped"


def test_reader_rechecks_dead_worker_without_any_new_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "native" / "emerald"
    directory.mkdir(parents=True)
    pixels = bytes(RGB_BYTES)
    (directory / "live.rgb").write_bytes(pixels)
    (directory / "live.json").write_text(json.dumps({"running": True, "pid": 77,
        "process_identity": "worker-77", "published_at": time.time(),
        "rgb_sha256": hashlib.sha256(pixels).hexdigest()}))
    monkeypatch.setattr(runtime_view, "process_identity", lambda pid: "worker-77")
    reader = LiveCampaignReader(tmp_path, "emerald")
    assert reader.poll().status["running"] is True
    monkeypatch.setattr(runtime_view, "process_identity", lambda pid: None)
    reader._last_liveness_poll = 0.0
    snapshot = reader.poll()
    assert snapshot.status["liveness"] == "dead" and snapshot.status["running"] is False
    assert snapshot.rgb == pixels
    assert not (directory / "stop.request").exists()


def test_reader_rejects_mixed_rgb_generation_but_keeps_last_image(tmp_path: Path) -> None:
    directory = tmp_path / "native" / "firered"
    directory.mkdir(parents=True)
    pixels = bytes(RGB_BYTES)
    (directory / "live.rgb").write_bytes(pixels)
    (directory / "live.json").write_text(json.dumps({"running": False,
        "rgb_sha256": hashlib.sha256(pixels).hexdigest()}))
    reader = LiveCampaignReader(tmp_path, "firered")
    assert reader.poll().rgb == pixels
    (directory / "live.rgb").write_bytes(bytes([1]) * RGB_BYTES)
    assert reader.poll().rgb == pixels
    assert reader.snapshot.message == "Actualización en curso"


def test_startup_error_visible_without_a_first_frame_and_newest_report_wins(tmp_path: Path) -> None:
    directory = tmp_path / "native" / "emerald"
    directory.mkdir(parents=True)
    (directory / "live.json").write_text(json.dumps({"running": True, "published_at": 10}))
    (directory / "status.json").write_text(json.dumps({"running": False, "published_at": 20,
                                                     "error": "ROM validation failed"}))
    reader = LiveCampaignReader(tmp_path, "emerald")
    snapshot = reader.poll()
    assert snapshot.rgb is None and snapshot.status["error"] == "ROM validation failed"
    assert snapshot.status["running"] is False


def test_atomic_publication_exhausts_permission_races_without_crashing_or_leaking(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def denied(self: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        raise PermissionError("viewer holds old file")

    monkeypatch.setattr(Path, "replace", denied)
    assert campaign._atomic_write(tmp_path / "live.json", b"{}") is False
    assert attempts == 3
    assert list(tmp_path.iterdir()) == []


class _Bridge:
    frame_count = 0


class _Planner:
    campaign_dag = ["one"]

    def completed_milestones(self, bridge: _Bridge, party: object) -> list[str]:
        return []


class _FailingController:
    _bootstrapped = False

    def __init__(self, directory: Path) -> None:
        self.bridge = _Bridge()
        self.planner = _Planner()
        self.directory, self.closed = directory, False

    def load_checkpoint(self, directory: Path | None = None) -> None:
        raise ValueError("checkpoint corrupt")

    def save_checkpoint(self, reason: str) -> Path:
        if reason == "error":
            raise OSError("secondary checkpoint write failed")
        checkpoint = self.directory / "stable"
        checkpoint.mkdir(exist_ok=True)
        (checkpoint / "manifest.json").write_text("{}")
        (self.directory / "latest.json").write_text('{"directory": "stable"}')
        return checkpoint

    def step_vblank_cycle(self) -> None:
        raise ValueError("original policy bug")

    def status(self) -> dict[str, object]:
        return {"frames": self.bridge.frame_count, "game": "emerald", "backend": "synthetic"}

    def close(self) -> None:
        self.closed = True


def test_worker_closes_even_when_checkpoint_cleanup_fails_without_masking_original(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "synthetic" / "emerald"
    controller = _FailingController(directory)
    # Synthetic completion inspection occurs before step_vblank_cycle.
    monkeypatch.setattr(_Bridge, "get_decrypted_player_party", lambda self: [], raising=False)
    monkeypatch.setattr(campaign, "build_laboratory", lambda *args, **kwargs: controller)
    with pytest.raises(ValueError, match="original policy bug"):
        campaign.run_campaign("emerald", "synthetic", checkpoint_root=tmp_path)
    assert controller.closed
    report = json.loads((directory / "status.json").read_text())
    assert report["failed"] and not report["running"] and not report["complete"]
    assert "original policy bug" in report["error"]
    assert any("secondary checkpoint" in error for error in report["cleanup_errors"])
    with campaign.CampaignWorkerLock(directory):
        pass


def test_failed_resume_closes_backend_without_overwriting_checkpoint(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "synthetic" / "emerald"
    controller = _FailingController(directory)
    monkeypatch.setattr(campaign, "build_laboratory", lambda *args, **kwargs: controller)
    with pytest.raises(ValueError, match="checkpoint corrupt"):
        campaign.run_campaign("emerald", "synthetic", checkpoint_root=tmp_path, resume=True)
    assert controller.closed and not (directory / "latest.json").exists()
    assert json.loads((directory / "status.json").read_text())["running"] is False


def test_transient_recovery_is_bounded_and_uses_pre_error_checkpoint(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "stable"
    calls: list[dict[str, Any]] = []

    def fail(*args: object, **kwargs: Any) -> dict[str, object]:
        calls.append(kwargs)
        raise campaign._AttemptFailure(OSError("temporary I/O"), checkpoint, 17)

    monkeypatch.setattr(campaign, "_run_attempt", fail)
    monkeypatch.setattr(time, "sleep", lambda delay: None)
    with pytest.raises(OSError, match="temporary I/O"):
        campaign._run_owned_campaign("emerald", "native", tmp_path, resume=False,
            max_frames=100, seed=42, report_interval=0, max_restarts=2)
    assert len(calls) == 3
    assert calls[1]["resume"] is True and calls[1]["checkpoint"] == checkpoint
    assert [call["max_frames"] for call in calls] == [100, 83, 66]


def test_gameplay_stall_does_not_trigger_endless_checkpoint_replay(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def stall(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise campaign._AttemptFailure(RuntimeError("CampaignStalled"), tmp_path / "stable", 10)

    monkeypatch.setattr(campaign, "_run_attempt", stall)
    with pytest.raises(RuntimeError, match="CampaignStalled"):
        campaign._run_owned_campaign("emerald", "native", tmp_path, resume=False,
            max_frames=None, seed=42, report_interval=0, max_restarts=2)
    assert calls == 1


def _fixture_checkpoint(directory: Path, game: str) -> Path:
    checkpoint = directory / "frame-000000000007-fixture"
    checkpoint.mkdir(parents=True, exist_ok=True)
    emulator, controller = b"preserved emulator bytes", b"preserved model bytes"
    (checkpoint / "emulator.state").write_bytes(emulator)
    (checkpoint / "controller.pt").write_bytes(controller)
    (checkpoint / "manifest.json").write_text(json.dumps({"game": game, "native": False, "frame": 7,
        "emulator_sha256": hashlib.sha256(emulator).hexdigest(),
        "controller_sha256": hashlib.sha256(controller).hexdigest()}))
    (directory / "latest.json").write_text(json.dumps({"directory": checkpoint.name}))
    return checkpoint


def _blocked_fixture_child(args: campaign.WorkerArgs, generation: str, sender: Connection,
                           checkpoint: str | None, behavior: str) -> None:
    """A real child blocks while holding the exact production worker OS lock."""
    game, backend, resume, _, root, _, _ = args
    assert root is not None
    directory = Path(root) / backend / game
    os.environ["ARES_WORKER_GENERATION"] = generation
    with campaign.CampaignWorkerLock(directory):
        if resume and behavior == "resume":
            assert checkpoint is not None
            assert (Path(checkpoint) / "emulator.state").read_bytes() == b"preserved emulator bytes"
            (directory / "resumed.lock.acquired").write_text("yes")
            sender.send({"game": game, "complete": True, "failed": False, "resumed_from": checkpoint})
            sender.close()
            return
        if not resume and behavior != "no-checkpoint":
            _fixture_checkpoint(directory, game)
        campaign._write_heartbeat(directory, "blocked_native_call_fixture", 7, 0)
        if behavior == "stop":
            (directory / "stop.request").write_text("stop the blocked child")
        time.sleep(60)


def _hang_then_resume_child(args: campaign.WorkerArgs, generation: str, sender: Connection,
                            checkpoint: str | None) -> None:
    _blocked_fixture_child(args, generation, sender, checkpoint, "resume")


def _always_hung_child(args: campaign.WorkerArgs, generation: str, sender: Connection,
                       checkpoint: str | None) -> None:
    _blocked_fixture_child(args, generation, sender, checkpoint, "always")


def _hang_without_checkpoint_child(args: campaign.WorkerArgs, generation: str, sender: Connection,
                                   checkpoint: str | None) -> None:
    _blocked_fixture_child(args, generation, sender, checkpoint, "no-checkpoint")


def _stop_hung_child(args: campaign.WorkerArgs, generation: str, sender: Connection,
                     checkpoint: str | None) -> None:
    _blocked_fixture_child(args, generation, sender, checkpoint, "stop")


def test_real_hung_child_is_terminated_and_resumes_exact_checkpoint_with_released_lock(tmp_path: Path) -> None:
    task: campaign.WorkerArgs = ("emerald", "synthetic", False, None, str(tmp_path), 42, 2)
    report = campaign.supervise_campaign(task, heartbeat_timeout=0.2, startup_timeout=20,
        poll_interval=0.02, child_target=_hang_then_resume_child)
    directory = tmp_path / "synthetic" / "emerald"
    assert report["complete"] and report["supervisor_restarts"] == 1
    assert report["resumed_from"] == str(directory / "frame-000000000007-fixture")
    assert (directory / "resumed.lock.acquired").read_text() == "yes"
    with campaign.CampaignWorkerLock(directory):
        pass
    events = [json.loads(line) for line in (directory / "supervisor-events.jsonl").read_text().splitlines()]
    assert sum(event["kind"] == "worker_started" for event in events) == 2
    assert any("HeartbeatTimeout" in event.get("error", "") for event in events)


def test_real_repeated_hang_exhausts_shared_restart_budget_and_has_no_orphan(tmp_path: Path) -> None:
    task: campaign.WorkerArgs = ("firered", "synthetic", False, None, str(tmp_path), 42, 2)
    report = campaign.supervise_campaign(task, heartbeat_timeout=0.2, startup_timeout=20,
        poll_interval=0.02, child_target=_always_hung_child)
    directory = tmp_path / "synthetic" / "firered"
    assert report["failed"] and not report["complete"] and not report["running"]
    assert report["supervisor_restarts"] == 2 and "HeartbeatTimeout" in str(report["error"])
    assert process_identity(int(str(report["pid"]))) is None
    with campaign.CampaignWorkerLock(directory):
        pass
    with campaign.CampaignWorkerLock(directory, "supervisor.lock"):
        pass
    events = [json.loads(line) for line in (directory / "supervisor-events.jsonl").read_text().splitlines()]
    assert sum(event["kind"] == "worker_started" for event in events) == 3


def test_new_game_hang_never_rolls_back_into_an_old_saved_campaign(tmp_path: Path) -> None:
    directory = tmp_path / "synthetic" / "emerald"
    old = _fixture_checkpoint(directory, "emerald")
    original = {path.name: path.read_bytes() for path in old.iterdir()}
    task: campaign.WorkerArgs = ("emerald", "synthetic", False, None, str(tmp_path), 42, 2)
    report = campaign.supervise_campaign(task, heartbeat_timeout=0.2, startup_timeout=20,
        poll_interval=0.02, child_target=_hang_without_checkpoint_child)
    assert report["failed"] and report["supervisor_restarts"] == 0
    assert "No complete checkpoint" in str(report["recovery_blocked"])
    assert {path.name: path.read_bytes() for path in old.iterdir()} == original


def test_stop_request_terminates_a_hung_child_without_restarting(tmp_path: Path) -> None:
    task: campaign.WorkerArgs = ("emerald", "synthetic", False, None, str(tmp_path), 42, 2)
    report = campaign.supervise_campaign(task, heartbeat_timeout=0.2, startup_timeout=20,
        poll_interval=0.02, child_target=_stop_hung_child)
    assert report["interrupted"] and not report["running"] and not report["failed"]
    assert not report["complete"] and report["supervisor_restarts"] == 0
    assert process_identity(int(str(report["pid"]))) is None


def test_supervisor_recovery_rejects_partial_or_corrupt_newer_bundle(tmp_path: Path) -> None:
    valid = _fixture_checkpoint(tmp_path, "emerald")
    corrupt = tmp_path / "frame-000000000008-newer"
    corrupt.mkdir()
    for filename in ("manifest.json", "controller.pt", "emulator.state"):
        (corrupt / filename).write_bytes((valid / filename).read_bytes())
    (corrupt / "controller.pt").write_bytes(b"corrupt replacement")
    (tmp_path / "latest.json").write_text(json.dumps({"directory": corrupt.name}))
    partial = tmp_path / "frame-000000000009-partial"
    partial.mkdir()
    (partial / "emulator.state").write_bytes(b"incomplete save")
    assert campaign._verified_checkpoint(tmp_path, "emerald", "synthetic") == valid
    assert campaign._verified_checkpoint(tmp_path, "firered", "synthetic") is None


class _SimulatedPipe:
    def __init__(self, messages: list[object]) -> None:
        self.messages = messages

    def poll(self, timeout: float = 0.0) -> bool:
        return bool(self.messages)

    def recv(self) -> object:
        if not self.messages:
            raise EOFError
        return self.messages.pop(0)

    def send(self, value: object) -> None:
        self.messages.append(value)

    def close(self) -> None:
        pass


class _SimulatedProcess:
    pid = os.getpid()
    exitcode = 0

    def __init__(self, target: Callable[..., None], args: tuple[object, ...], name: str) -> None:
        self.target, self.args = target, args

    def start(self) -> None:
        # Exercise production monitor/pipe/checkpoint logic without launching any
        # emulator or subprocess, and avoid leaking a child-only environment key.
        original = os.environ.get("ARES_WORKER_GENERATION")
        try:
            self.target(*self.args)
        finally:
            if original is None:
                os.environ.pop("ARES_WORKER_GENERATION", None)
            else:
                os.environ["ARES_WORKER_GENERATION"] = original

    def is_alive(self) -> bool:
        return False

    def join(self, timeout: float | None = None) -> None:
        pass

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass

    def close(self) -> None:
        pass


class _SimulatedContext:
    def Pipe(self, duplex: bool = False) -> tuple[_SimulatedPipe, _SimulatedPipe]:
        messages: list[object] = []
        return _SimulatedPipe(messages), _SimulatedPipe(messages)

    def Process(self, *, target: Callable[..., None], args: tuple[object, ...], name: str) -> _SimulatedProcess:
        return _SimulatedProcess(target, args, name)


def _simulate_supervision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multiprocessing, "get_context", lambda method: _SimulatedContext())
    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    def advance(seconds: float) -> None:
        clock[0] += max(seconds, 0.001)

    monkeypatch.setattr(time, "sleep", advance)


def _simulated_grind_outcome(args: campaign.WorkerArgs, generation: str, sender: Connection,
                              attempt: int, milestones: list[str], complete: bool = False) -> None:
    game, backend, _, _, root, _, _ = args
    assert root is not None
    directory = Path(root) / backend / game
    directory.mkdir(parents=True, exist_ok=True)
    frame = (attempt + 1) * 4001
    checkpoint = directory / f"frame-{frame:012d}-{1_000_000 + attempt}"
    checkpoint.mkdir()
    emulator, model = f"physical state {attempt}".encode(), f"model {attempt}".encode()
    (checkpoint / "emulator.state").write_bytes(emulator)
    (checkpoint / "controller.pt").write_bytes(model)
    report: dict[str, object] = {"game": game, "backend": backend, "pid": os.getpid(),
        "worker_generation": generation, "frames": frame, "running": False,
        "complete": complete, "failed": not complete, "interrupted": False,
        "error": "" if complete else "CampaignStalled: route loop after training",
        "published_at": time.time(), "metrics": {"completed_milestones": milestones,
            "wins": attempt * 20, "world_model_updates": attempt * 1000, "experience": frame}}
    (checkpoint / "manifest.json").write_text(json.dumps({"game": game, "native": False,
        "frame": frame, "status": report, "emulator_sha256": hashlib.sha256(emulator).hexdigest(),
        "controller_sha256": hashlib.sha256(model).hexdigest()}))
    (directory / "latest.json").write_text(json.dumps({"directory": checkpoint.name}))
    (directory / "status.json").write_text(json.dumps(report))
    os.environ["ARES_WORKER_GENERATION"] = generation
    campaign._write_heartbeat(directory, "stopped", frame, 0)
    sender.send(report)


def test_fake_frame_progress_cannot_fund_unlimited_stagnant_restarts(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _simulate_supervision(monkeypatch)
    attempts: list[str | None] = []

    def grind(args: campaign.WorkerArgs, generation: str, sender: Connection, checkpoint: str | None) -> None:
        attempts.append(checkpoint)
        _simulated_grind_outcome(args, generation, sender, len(attempts) - 1, [])

    task: campaign.WorkerArgs = ("emerald", "synthetic", False, None, str(tmp_path), 42, -1)
    report = campaign.supervise_campaign(task, child_target=grind, stagnant_restart_limit=3)
    directory = tmp_path / "synthetic" / "emerald"
    assert len(attempts) == 4 and report["supervisor_restarts"] == 3
    assert report["failed"] and "Stagnant recovery limit" in str(report["recovery_blocked"])
    assert report["stagnant_restarts"] == 3
    saved = json.loads((directory / "status.json").read_text())
    assert saved["recovery_blocked"] == report["recovery_blocked"]
    events = [json.loads(line) for line in (directory / "supervisor-events.jsonl").read_text().splitlines()]
    assert events[-1]["kind"] == "recovery_blocked"
    # Restarting the parent cannot refresh its already exhausted stagnant budget.
    resumed: campaign.WorkerArgs = ("emerald", "synthetic", True, None, str(tmp_path), 42, -1)
    again = campaign.supervise_campaign(resumed, child_target=grind, stagnant_restart_limit=3)
    assert len(attempts) == 4 and again["failed"] and again["stagnant_restarts"] == 3


def test_new_story_milestone_rearms_stagnant_budget_but_replayed_flags_do_not(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _simulate_supervision(monkeypatch)
    attempts = 0
    histories = [["START"], ["START"], ["START"], ["START", "BADGE1"], ["START", "BADGE1"]]

    def progress(args: campaign.WorkerArgs, generation: str, sender: Connection, checkpoint: str | None) -> None:
        nonlocal attempts
        index = attempts
        attempts += 1
        _simulated_grind_outcome(args, generation, sender, index, histories[index], complete=index == 4)

    task: campaign.WorkerArgs = ("emerald", "synthetic", False, None, str(tmp_path), 42, -1)
    report = campaign.supervise_campaign(task, child_target=progress, stagnant_restart_limit=2)
    assert report["complete"] and attempts == 5 and report["supervisor_restarts"] == 4
    ledger = campaign._RecoveryProgress(tmp_path / "synthetic" / "emerald", "emerald", "synthetic", resume=True)
    assert ledger.facts == {"milestone:START", "milestone:BADGE1"} and ledger.stagnant_restarts == 0
    ledger.stagnant_restarts = 2
    ledger.save()
    assert not ledger.observe({"metrics": {"completed_milestones": []}, "flags_progress": 0})
    assert not ledger.observe({"metrics": {"completed_milestones": ["START", "BADGE1"]}})
    assert ledger.stagnant_restarts == 2
    assert ledger.observe({"native_completion_evidence": {"badges": [True]}})
    assert ledger.stagnant_restarts == 0


def test_durable_progress_uses_story_facts_in_checkpoint_manifest_and_monotonic_flags(tmp_path: Path) -> None:
    progress = campaign._RecoveryProgress(tmp_path, "emerald", "synthetic", resume=False)
    assert progress.observe({"status": {"metrics": {"completed_milestones": ["POKEDEX"]},
        "native_completion_evidence": {"badges": [True, False], "game_clear": False},
        "flags_progress": 12}})
    progress.stagnant_restarts = 2
    progress.save()
    restored = campaign._RecoveryProgress(tmp_path, "emerald", "synthetic", resume=True)
    assert not restored.observe({"frames": 99_999_999, "wins": 1000, "experience": 999,
                                 "flags_progress": 8})
    assert not restored.observe({"flags_progress": 12}) and restored.stagnant_restarts == 2
    assert restored.observe({"flags_progress": 13}) and restored.stagnant_restarts == 0


def test_failed_newer_checkpoint_is_excluded_even_after_many_frames(tmp_path: Path) -> None:
    older = _fixture_checkpoint(tmp_path, "emerald")
    newer = tmp_path / "frame-000000009999-newer"
    newer.mkdir()
    for filename in ("manifest.json", "controller.pt", "emulator.state"):
        (newer / filename).write_bytes((older / filename).read_bytes())
    (tmp_path / "latest.json").write_text(json.dumps({"directory": newer.name}))
    progress = campaign._RecoveryProgress(tmp_path, "emerald", "synthetic", resume=False)
    progress.mark_failed(newer)
    assert campaign._verified_checkpoint(tmp_path, "emerald", "synthetic",
                                        excluded=frozenset(progress.failures)) == older


def _story_checkpoint(directory: Path, name: str, milestones: list[str], reason: str) -> Path:
    checkpoint = directory / name
    checkpoint.mkdir(parents=True)
    emulator, model = b"immutable emulator", b"immutable controller"
    (checkpoint / "emulator.state").write_bytes(emulator)
    (checkpoint / "controller.pt").write_bytes(model)
    (checkpoint / "manifest.json").write_text(json.dumps({"game": "emerald", "native": False,
        "frame": int(name.split("-")[1]), "reason": reason,
        "status": {"metrics": {"completed_milestones": milestones}},
        "emulator_sha256": hashlib.sha256(emulator).hexdigest(),
        "controller_sha256": hashlib.sha256(model).hexdigest()}))
    return checkpoint


def test_recovery_milestone_floor_prefers_verified_import_and_never_falls_to_six(tmp_path: Path) -> None:
    milestones = [f"STORY_{index}" for index in range(9)]
    imported = _story_checkpoint(tmp_path, "frame-000009000000-1000", milestones, "verified_recovery_import")
    error = _story_checkpoint(tmp_path, "frame-000010000000-2000", milestones, "stall_recovery")
    regressed = _story_checkpoint(tmp_path, "frame-000000000100-3000", milestones[:6], "periodic")
    (tmp_path / "latest.json").write_text(json.dumps({"directory": regressed.name}))
    required = frozenset(milestones)
    assert campaign._verified_checkpoint(tmp_path, "emerald", "synthetic", required_milestones=required) == imported
    assert campaign._verified_checkpoint(tmp_path, "emerald", "synthetic", required_milestones=required,
                                        excluded=frozenset({imported.name})) == error
    assert campaign._verified_checkpoint(tmp_path, "emerald", "synthetic", required_milestones=required,
                                        excluded=frozenset({imported.name, error.name})) is None


@pytest.mark.parametrize("reason", ["requested", "milestone", "periodic"])
def test_healthy_latest_preserves_level_xp_and_frames_after_an_older_verified_import(
        tmp_path: Path, reason: str) -> None:
    milestones = [f"STORY_{index}" for index in range(16)]
    imported = _story_checkpoint(tmp_path, "frame-000000193277-1000", milestones, "verified_recovery_import")
    healthy = _story_checkpoint(tmp_path, "frame-000000290680-2000", milestones, reason)
    for checkpoint, level, xp in ((imported, 18, 5832), (healthy, 22, 10648)):
        path = checkpoint / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["status"]["party"] = [{"species": 284, "level": level, "experience": xp}]
        path.write_text(json.dumps(manifest))
    (tmp_path / "latest.json").write_text(json.dumps({"directory": healthy.name}))
    selected = campaign._verified_checkpoint(tmp_path, "emerald", "synthetic",
                                               required_milestones=frozenset(milestones))
    assert selected == healthy
    evidence = campaign._checkpoint_report(selected)
    assert evidence["frame"] == 290680
    party = cast(dict[str, Any], evidence["status"])["party"]
    assert party[0]["level"] == 22 and party[0]["experience"] == 10648


def test_degraded_latest_uses_newest_healthy_snapshot_without_rolling_back_to_import(tmp_path: Path) -> None:
    milestones = ["POKEDEX", "BADGE1"]
    imported = _story_checkpoint(tmp_path, "frame-000000193277-1000", milestones, "verified_recovery_import")
    healthy = _story_checkpoint(tmp_path, "frame-000000290680-2000", milestones, "requested")
    error = _story_checkpoint(tmp_path, "frame-000000295000-3000", milestones, "recovery_probe_error")
    (tmp_path / "latest.json").write_text(json.dumps({"directory": error.name}))
    assert campaign._verified_checkpoint(tmp_path, "emerald", "synthetic",
        required_milestones=frozenset(milestones)) == healthy
    assert campaign._verified_checkpoint(tmp_path, "emerald", "synthetic",
        required_milestones=frozenset(milestones), excluded=frozenset({healthy.name})) == imported


def test_resume_selects_highwater_story_before_starting_worker(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _simulate_supervision(monkeypatch)
    directory = tmp_path / "synthetic" / "emerald"
    milestones = [f"STORY_{index}" for index in range(9)]
    imported = _story_checkpoint(directory, "frame-000009000000-1000", milestones, "verified_recovery_import")
    regressed = _story_checkpoint(directory, "frame-000000000100-2000", milestones[:6], "periodic")
    (directory / "latest.json").write_text(json.dumps({"directory": regressed.name}))
    ledger = campaign._RecoveryProgress(directory, "emerald", "synthetic", resume=False)
    ledger.session_started_at = time.time() - 10
    ledger.observe({"metrics": {"completed_milestones": milestones}})

    def resumed(args: campaign.WorkerArgs, generation: str, sender: Connection, checkpoint: str | None) -> None:
        assert checkpoint == str(imported)
        sender.send({"game": "emerald", "complete": True, "failed": False, "resumed_from": checkpoint})

    task: campaign.WorkerArgs = ("emerald", "synthetic", True, None, str(tmp_path), 42, -1)
    report = campaign.supervise_campaign(task, child_target=resumed)
    assert report["complete"] and report["resumed_from"] == str(imported)


def test_missing_checkpoint_at_story_floor_blocks_instead_of_rewinding_history(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _simulate_supervision(monkeypatch)
    directory = tmp_path / "synthetic" / "emerald"
    milestones = [f"STORY_{index}" for index in range(9)]
    regressed = _story_checkpoint(directory, "frame-000000000100-1000", milestones[:6], "periodic")
    (directory / "latest.json").write_text(json.dumps({"directory": regressed.name}))
    ledger = campaign._RecoveryProgress(directory, "emerald", "synthetic", resume=False)
    ledger.session_started_at = time.time() - 10
    ledger.observe({"metrics": {"completed_milestones": milestones}})

    def forbidden(args: campaign.WorkerArgs, generation: str, sender: Connection, checkpoint: str | None) -> None:
        pytest.fail("worker was launched from a history-regressing checkpoint")

    task: campaign.WorkerArgs = ("emerald", "synthetic", True, None, str(tmp_path), 42, -1)
    report = campaign.supervise_campaign(task, child_target=forbidden)
    assert report["failed"] and not report["complete"]
    assert "milestone highwater" in str(report["recovery_blocked"])


def test_new_game_session_does_not_inherit_nine_milestones_from_retained_old_saves(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _simulate_supervision(monkeypatch)
    directory = tmp_path / "synthetic" / "emerald"
    milestones = [f"OLD_STORY_{index}" for index in range(9)]
    old = _story_checkpoint(directory, "frame-000009000000-1000", milestones, "verified_recovery_import")
    old_time = time.time() - 3600
    os.utime(old / "manifest.json", (old_time, old_time))
    old_ledger = campaign._RecoveryProgress(directory, "emerald", "synthetic", resume=False)
    old_ledger.observe({"metrics": {"completed_milestones": milestones}})

    def fresh(args: campaign.WorkerArgs, generation: str, sender: Connection, checkpoint: str | None) -> None:
        _simulated_grind_outcome(args, generation, sender, 0, [], complete=True)

    task: campaign.WorkerArgs = ("emerald", "synthetic", False, None, str(tmp_path), 42, -1)
    assert campaign.supervise_campaign(task, child_target=fresh)["complete"]
    current = campaign._RecoveryProgress(directory, "emerald", "synthetic", resume=True)
    assert current.required_milestones == frozenset() and current.session_started_at is not None

    def resume_fresh(args: campaign.WorkerArgs, generation: str, sender: Connection, checkpoint: str | None) -> None:
        sender.send({"game": "emerald", "complete": True, "failed": False})

    resumed: campaign.WorkerArgs = ("emerald", "synthetic", True, None, str(tmp_path), 42, -1)
    assert campaign.supervise_campaign(resumed, child_target=resume_fresh)["complete"]
    after = campaign._RecoveryProgress(directory, "emerald", "synthetic", resume=True)
    assert after.required_milestones == frozenset()
