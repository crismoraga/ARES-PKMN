"""Monitoring station: file-backed API, safe controls, telemetry series and restart policy."""
from __future__ import annotations

import hashlib
import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
import struct
import threading
import time
from typing import Any, cast
import urllib.error
import urllib.request
import zlib

import pytest
import torch

from core.runtime_view import RGB_BYTES, list_checkpoints, read_jsonl_tail
from scripts import run_campaign as campaign
from scripts.dashboard import MonitoringStation, encode_png, serve, thin
from scripts.run_autonomous_verification_demo import build_laboratory


def _campaign_fixture(root: Path, game: str = "firered") -> Path:
    directory = root / "native" / game
    directory.mkdir(parents=True)
    pixels = bytes(range(256)) * (RGB_BYTES // 256)
    (directory / "live.rgb").write_bytes(pixels)
    (directory / "live.json").write_text(json.dumps({
        "game": game, "running": True, "pid": os.getpid(), "published_at": time.time(), "frames": 1234,
        "rgb_sha256": hashlib.sha256(pixels).hexdigest(),
        "telemetry": {"party": [{"species": 1, "name": "BULBASAUR", "level": 14, "hp": 30, "maxHP": 38}]}}))
    events = [{"kind": "movement", "frame": 1}, {"kind": "battle_decision", "frame": 2, "summary": {"move": "TACKLE"}},
              {"kind": "evolution", "frame": 3, "from_name": "MUDKIP", "to_name": "MARSHTOMP"}]
    (directory / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n{broken", encoding="utf-8")
    samples = [{"t": float(index), "frame": index * 100, "wins": index} for index in range(50)]
    (directory / "telemetry.jsonl").write_text("\n".join(json.dumps(sample) for sample in samples), encoding="utf-8")
    checkpoint = directory / "frame-000000000100-1792000000000000000"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(json.dumps({"frame": 100, "reason": "milestone"}))
    return directory


def test_png_encoder_round_trips_a_real_framebuffer() -> None:
    rgb = bytes((x * 7 + y * 3) % 256 for y in range(160) for x in range(240 * 3))
    png = encode_png(rgb)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height, depth, color = struct.unpack(">IIBB", png[16:26])
    assert (width, height, depth, color) == (240, 160, 8, 2)
    start = png.index(b"IDAT") + 4
    length = struct.unpack(">I", png[start - 8:start - 4])[0]
    raw = zlib.decompress(png[start:start + length])
    assert raw[1:1 + 720] == rgb[:720] and len(raw) == 160 * (720 + 1)
    with pytest.raises(ValueError):
        encode_png(b"short")


def test_thinning_keeps_both_ends_of_a_series() -> None:
    samples: list[dict[str, object]] = [{"i": index} for index in range(1000)]
    thinned = thin(samples, 50)
    assert len(thinned) == 50 and thinned[0]["i"] == 0 and thinned[-1]["i"] == 999
    assert thin(samples[:10], 50) == samples[:10]


def test_bounded_tail_reader_filters_kinds_and_skips_corrupt_lines(tmp_path: Path) -> None:
    directory = _campaign_fixture(tmp_path)
    rows = read_jsonl_tail(directory / "events.jsonl", limit=10)
    assert [row["kind"] for row in rows] == ["movement", "battle_decision", "evolution"]
    only = read_jsonl_tail(directory / "events.jsonl", limit=10, kinds=frozenset({"evolution"}))
    assert only == [rows[-1]]
    assert list_checkpoints(directory)[0]["reason"] == "milestone"


def test_station_api_serves_frames_series_and_guards_controls(tmp_path: Path) -> None:
    directory = _campaign_fixture(tmp_path)
    server = serve(tmp_path, "native", "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/api/overview") as response:
            overview = json.loads(response.read())
        firered = overview["games"]["firered"]
        assert firered["liveness"] == "running" and firered["telemetry"]["party"][0]["name"] == "BULBASAUR"
        assert overview["games"]["emerald"]["liveness"] == "absent"
        with urllib.request.urlopen(f"{base}/api/frame/firered.png") as response:
            assert response.headers["Content-Type"] == "image/png" and response.read()[:4] == b"\x89PNG"
        with urllib.request.urlopen(f"{base}/api/telemetry/firered?limit=10") as response:
            series = json.loads(response.read())
        assert len(series) == 10 and series[-1]["wins"] == 49
        with urllib.request.urlopen(f"{base}/api/events/firered?limit=5") as response:
            kinds = [event["kind"] for event in json.loads(response.read())]
        assert "movement" not in kinds and "evolution" in kinds  # noise is hidden by default
        with urllib.request.urlopen(f"{base}/") as response:
            assert b"Estaci" in response.read()
        request = urllib.request.Request(f"{base}/api/control/firered/stop", method="POST")
        with pytest.raises(urllib.error.HTTPError) as rejected:
            urllib.request.urlopen(request)
        assert rejected.value.code == 403 and not (directory / "stop.request").exists()
        request.add_header("X-ARES-Control", "1")
        with urllib.request.urlopen(request) as response:
            assert json.loads(response.read())["requested"] == "stop"
        assert (directory / "stop.request").is_file()
        bad = urllib.request.Request(f"{base}/api/control/firered/erase", method="POST", headers={"X-ARES-Control": "1"})
        with pytest.raises(urllib.error.HTTPError) as unknown:
            urllib.request.urlopen(bad)
        assert unknown.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_station_reports_restarting_between_supervised_generations(tmp_path: Path) -> None:
    directory = _campaign_fixture(tmp_path)
    status = json.loads((directory / "live.json").read_text())
    status["running"] = False
    (directory / "live.json").write_text(json.dumps(status))
    (directory / "supervisor-events.jsonl").write_text(json.dumps({"kind": "worker_restart", "published_at": time.time()}))
    assert MonitoringStation(tmp_path).overview()["games"]["firered"]["liveness"] == "restarting"  # type: ignore[index]


def test_telemetry_recorder_samples_and_rotates(tmp_path: Path) -> None:
    torch.set_num_threads(2)
    controller = build_laboratory(map_count=4)
    try:
        recorder = campaign.TelemetryRecorder(tmp_path, interval=3600, max_bytes=10, events_max_bytes=10)
        first = recorder.sample(controller, force=True)
        assert first is not None and first["fps"] is None  # the first sample has no rate yet
        assert recorder.sample(controller) is None  # cadence respected
        controller.run_frames(20)
        second = recorder.sample(controller, force=True)
        assert second is not None and cast(float, second["fps"]) > 0 and second["party"]
        (tmp_path / "events.jsonl").write_text("x" * 100)
        recorder._last_rotation_check = 0.0
        recorder.sample(controller, force=True)
        assert (tmp_path / "telemetry.1.jsonl").is_file() and (tmp_path / "events.1.jsonl").is_file()
    finally:
        controller.close()


def test_fatal_configuration_errors_are_never_restarted() -> None:
    assert campaign.restartable_error("CampaignStalled: No gameplay progress after 4 recoveries")
    assert campaign.restartable_error("HeartbeatTimeout: worker made no heartbeat for 120 seconds")
    assert not campaign.restartable_error("ValueError: ROM revision mismatch: abc; expected def")
    assert not campaign.restartable_error("CampaignAlreadyRunning: Campaign already owned by another worker")
    assert not campaign.restartable_error("")


def _fixture_checkpoint(directory: Path, name: str, frame: int) -> Path:
    checkpoint = directory / name
    checkpoint.mkdir(parents=True, exist_ok=True)
    emulator, controller = f"state {frame}".encode(), f"model {frame}".encode()
    (checkpoint / "emulator.state").write_bytes(emulator)
    (checkpoint / "controller.pt").write_bytes(controller)
    (checkpoint / "manifest.json").write_text(json.dumps({"game": "emerald", "native": False, "frame": frame,
        "emulator_sha256": hashlib.sha256(emulator).hexdigest(),
        "controller_sha256": hashlib.sha256(controller).hexdigest()}))
    return checkpoint


def test_verified_checkpoint_can_walk_back_past_a_failing_bundle(tmp_path: Path) -> None:
    older = _fixture_checkpoint(tmp_path, "frame-000000000100-1000", 100)
    newer = _fixture_checkpoint(tmp_path, "frame-000000000050-2000", 50)  # a rewind made a later, lower frame
    (tmp_path / "latest.json").write_text(json.dumps({"directory": newer.name}))
    assert campaign._verified_checkpoint(tmp_path, "emerald", "synthetic") == newer
    assert campaign._verified_checkpoint(tmp_path, "emerald", "synthetic", skip=1) == older
    assert campaign._verified_checkpoint(tmp_path, "emerald", "synthetic", skip=2) is None


def _fail_then_succeed_child(args: campaign.WorkerArgs, generation: str, sender: Connection,
                             checkpoint: str | None) -> None:
    """First generation finishes with a gameplay failure; the restart completes."""
    game, backend, _, _, root, seed, _ = args
    assert root is not None
    directory = Path(root) / backend / game
    os.environ["ARES_WORKER_GENERATION"] = generation
    with campaign.CampaignWorkerLock(directory):
        _fixture_checkpoint(directory, f"frame-000000000007-{time.time_ns()}", 7)
        latest = sorted(directory.glob("frame-*"))[-1]
        (directory / "latest.json").write_text(json.dumps({"directory": latest.name}))
        campaign._write_heartbeat(directory, "running", 7, 0)
        marker = directory / "first-attempt.done"
        if not marker.exists():
            marker.write_text("x")
            sender.send({"game": game, "complete": False, "failed": True, "interrupted": False,
                         "error": "CampaignStalled: no progress", "seed": seed})
        else:
            sender.send({"game": game, "complete": True, "failed": False, "seed": seed, "resumed_from": checkpoint})
        sender.close()


def _fatal_child(args: campaign.WorkerArgs, generation: str, sender: Connection, checkpoint: str | None) -> None:
    os.environ["ARES_WORKER_GENERATION"] = generation
    sender.send({"game": args[0], "complete": False, "failed": True, "error": "ValueError: ROM revision mismatch"})
    sender.close()


def test_supervisor_restarts_a_failed_worker_with_a_new_seed(tmp_path: Path) -> None:
    task: campaign.WorkerArgs = ("emerald", "synthetic", False, None, str(tmp_path), 42, -1)
    report = campaign.supervise_campaign(task, heartbeat_timeout=20, startup_timeout=20, poll_interval=0.02,
                                         child_target=_fail_then_succeed_child)
    assert report["complete"] and report["supervisor_restarts"] == 1
    assert report["seed"] == 43 and report["resumed_from"]
    events = [json.loads(line) for line in (tmp_path / "synthetic" / "emerald" / "supervisor-events.jsonl").read_text().splitlines()]
    assert any(event["kind"] == "worker_restart" and "CampaignStalled" in event["error"] for event in events)


def test_supervisor_does_not_restart_a_configuration_error(tmp_path: Path) -> None:
    task: campaign.WorkerArgs = ("emerald", "synthetic", False, None, str(tmp_path), 42, -1)
    report = campaign.supervise_campaign(task, heartbeat_timeout=20, startup_timeout=20, poll_interval=0.02,
                                         child_target=_fatal_child)
    assert report["failed"] and "ROM revision" in str(report["error"])
    assert report["supervisor_restarts"] == 0


def test_publish_live_includes_team_and_telemetry(tmp_path: Path) -> None:
    controller = build_laboratory(map_count=4)
    try:
        controller.run_frames(10)
        campaign._publish_live(controller, tmp_path)
        published: dict[str, Any] = json.loads((tmp_path / "live.json").read_text())
        assert published["party"][0]["level"] == 50 and "milestones" in published["telemetry"]
    finally:
        controller.close()
