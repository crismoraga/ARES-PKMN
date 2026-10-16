"""Run/resume genuine cartridge campaigns or explicitly separate laboratories."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Mapping
import hashlib
import json
import math
import multiprocessing
from multiprocessing.connection import Connection
import os
import sys
from threading import Event
import time
import uuid
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, cast

import torch

from core.bridge.ffi_mgba import NativeLibMGBABridge
from core.game_data import Game, ROM_NAMES, ROOT
from core.orchestrator import AresAutonomousOrchestrator
from core.runtime_view import campaign_liveness, nested_mapping, process_identity, read_campaign_status
from scripts.run_autonomous_verification_demo import build_laboratory


def _checkpoint_root(backend: str, game: Game, root: Path | None = None) -> Path:
    return (root or ROOT / "artifacts/campaigns") / backend / game


class CampaignAlreadyRunning(RuntimeError):
    """A different worker owns the emulator and checkpoints for this campaign."""


class CampaignWorkerLock:
    """OS advisory lock; stale files cannot retain ownership after a process dies."""

    def __init__(self, directory: Path, filename: str = "worker.lock") -> None:
        self.directory = directory
        self.filename = filename
        self._stream: BinaryIO | None = None

    def __enter__(self) -> CampaignWorkerLock:
        self.directory.mkdir(parents=True, exist_ok=True)
        stream = (self.directory / self.filename).open("a+b")
        try:
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            raise CampaignAlreadyRunning(f"Campaign already owned by another worker: {self.directory}") from error
        self._stream = stream
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> None:
        if self._stream is not None:
            # Closing the handle releases either OS lock, including on exceptions.
            self._stream.close()
            self._stream = None


def _atomic_write(path: Path, data: bytes) -> bool:
    """Telemetry is best effort: a reader's Windows handle must never kill a game."""
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temporary.write_bytes(data)
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


def _runtime_report(report: dict[str, object], *, running: bool, error: str = "") -> dict[str, object]:
    return {**report, "running": running, "pid": os.getpid(),
            "process_identity": process_identity(os.getpid()), "published_at": time.time(), "error": error,
            "worker_generation": os.environ.get("ARES_WORKER_GENERATION", "")}


def _write_heartbeat(directory: Path, phase: str, frames: int, restart_count: int) -> None:
    """Called only by the frame owner, so a blocked call cannot fake liveness."""
    _atomic_write(directory / "heartbeat.json", json.dumps({"pid": os.getpid(),
        "generation": os.environ.get("ARES_WORKER_GENERATION", ""), "sequence": time.monotonic_ns(),
        "published_at": time.time(), "phase": phase, "frames": frames,
        "restart_count": restart_count}).encode("utf-8"))


def _latest_checkpoint(directory: Path) -> Path | None:
    try:
        pointer = json.loads((directory / "latest.json").read_text(encoding="utf-8"))
        name = pointer["directory"]
        if not isinstance(name, str) or Path(name).name != name:
            return None
        path = directory / name
        return path if (path / "manifest.json").is_file() else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _save_screen(controller: AresAutonomousOrchestrator, path: Path) -> None:
    """Save framebuffer evidence from the actual native handle, without a window."""
    if isinstance(controller.bridge, NativeLibMGBABridge):
        import pygame
        rgb = controller.bridge.framebuffer().view("uint8").reshape(160, 240, 4)[:, :, :3]
        surface = pygame.image.frombuffer(rgb.tobytes(), (240, 160), "RGB")
        pygame.image.save(surface, str(path))


def _telemetry_fields(controller: AresAutonomousOrchestrator) -> dict[str, object]:
    """Display-only state for the viewer and dashboard; failures never stop a game."""
    try:
        telemetry = controller.telemetry()
    except Exception as error:  # Monitoring must never become a gameplay fault.
        return {"telemetry_error": f"{type(error).__name__}: {error}"}
    fields: dict[str, object] = {"telemetry": telemetry}
    if "menu" in telemetry:
        fields["menu"] = telemetry["menu"]
    if "party" in telemetry:
        fields["party"] = telemetry["party"]
    return fields


def _publish_live(controller: AresAutonomousOrchestrator, directory: Path,
                  running: bool = True, error: str = "") -> None:
    """Publish the real emulator framebuffer and telemetry for the read-only viewer."""
    report = _runtime_report(controller.status(), running=running, error=error)
    report.update(_telemetry_fields(controller))
    if isinstance(controller.bridge, NativeLibMGBABridge):
        pixels = controller.bridge.framebuffer().view("uint8").reshape(160, 240, 4)[:, :, :3]
        rgb = pixels.tobytes()
        if _atomic_write(directory / "live.rgb", rgb):
            report["rgb_sha256"] = hashlib.sha256(rgb).hexdigest()
        else:
            _atomic_write(directory / "status.json", json.dumps(report, default=str).encode("utf-8"))
            return  # Retain the last complete frame/JSON generation together.
    _atomic_write(directory / "live.json", json.dumps(report, default=str).encode("utf-8"))


class TelemetryRecorder:
    """Append compact time-series samples for charts; rotate large logs.

    Samples are written by the frame owner at a wall-clock cadence, so the
    dashboard can plot levels, wins, recoveries and speed without touching
    the emulator. Rotation keeps exactly one previous generation of each log.
    """

    def __init__(self, directory: Path, interval: float = 5.0, max_bytes: int = 32 << 20,
                 events_max_bytes: int = 64 << 20) -> None:
        self.directory, self.interval = directory, interval
        self.max_bytes, self.events_max_bytes = max_bytes, events_max_bytes
        self._last = 0.0
        self._last_frames: tuple[float, int] | None = None
        self._last_rotation_check = 0.0

    def sample(self, controller: AresAutonomousOrchestrator, *, force: bool = False) -> dict[str, object] | None:
        now = time.monotonic()
        if not force and now - self._last < self.interval:
            return None
        self._last = now
        frames = controller.bridge.frame_count
        fps: float | None = None  # Unknown for a run's first sample or across a rewind.
        if self._last_frames is not None and now > self._last_frames[0] and frames >= self._last_frames[1]:
            fps = (frames - self._last_frames[1]) / (now - self._last_frames[0])
        self._last_frames = (now, frames)
        metrics = controller.metrics
        sample: dict[str, object] = {
            "t": round(time.time(), 3), "frame": frames, "fps": None if fps is None else round(fps, 1), "scene": controller.scene,
            "battles": metrics.battles, "wins": metrics.wins, "losses": metrics.losses,
            "captures": metrics.captures, "faints": metrics.faints, "moves": metrics.moves,
            "recoveries": metrics.recoveries, "faults": metrics.faults, "rewinds": metrics.rewinds,
            "heals": metrics.heals, "evolutions": metrics.evolutions, "level_ups": metrics.level_ups,
            "milestones": len(metrics.completed_milestones), "map_transitions": metrics.map_transitions,
            "proved": metrics.proved_decisions, "unproven": metrics.unproven_decisions,
            "unproductive": controller.watchdog.unproductive_frames(frames),
            "story_unproductive": controller.watchdog.strategic_unproductive_frames}
        try:
            telemetry = controller.telemetry()
            party = telemetry.get("party")
            if isinstance(party, list):
                sample["party"] = [[mon.get("species"), mon.get("name"), mon.get("level"), mon.get("hp"), mon.get("maxHP")]
                                   for mon in party if isinstance(mon, dict)]
            trainer = telemetry.get("trainer")
            if isinstance(trainer, dict):
                for key in ("money", "dex_owned", "dex_seen", "badge_count", "play_seconds"):
                    sample[key] = trainer.get(key)
            sample["map_name"] = telemetry.get("map_name", "")
        except Exception as error:  # Monitoring must never become a gameplay fault.
            sample["telemetry_error"] = f"{type(error).__name__}: {error}"
        try:
            with (self.directory / "telemetry.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(sample, default=str) + "\n")
        except OSError:
            pass
        if now - self._last_rotation_check >= 60:
            self._last_rotation_check = now
            self._rotate("telemetry.jsonl", self.max_bytes)
            self._rotate("events.jsonl", self.events_max_bytes)
        return sample

    def _rotate(self, name: str, limit: int) -> None:
        path = self.directory / name
        try:
            if path.stat().st_size <= limit:
                return
            previous = self.directory / (path.stem + ".1" + path.suffix)
            previous.unlink(missing_ok=True)
            path.replace(previous)
        except OSError:
            pass  # A reader may hold the file; rotate on a later check.


def run_campaign(game: Game, backend: str = "native", *, resume: bool = False,
                 max_frames: int | None = None, checkpoint_root: Path | None = None,
                 seed: int = 42, report_interval: int = 5000,
                 max_restarts: int = 2, resume_checkpoint: Path | None = None) -> dict[str, object]:
    """Run until verified success or the user's optional local probe frame cap."""
    if backend not in ("native", "synthetic") or max_frames is not None and max_frames < 1 or max_restarts < 0:
        raise ValueError("Supported backend and positive frame limit required")
    torch.set_num_threads(8)
    directory = _checkpoint_root(backend, game, checkpoint_root)
    with CampaignWorkerLock(directory):
        return _run_owned_campaign(game, backend, directory, resume=resume, max_frames=max_frames,
                                   seed=seed, report_interval=report_interval, max_restarts=max_restarts,
                                   initial_checkpoint=resume_checkpoint)


def _run_owned_campaign(game: Game, backend: str, directory: Path, *, resume: bool,
                        max_frames: int | None, seed: int, report_interval: int,
                        max_restarts: int, initial_checkpoint: Path | None = None) -> dict[str, object]:
    """Retry only transient I/O, from a checkpoint made before the failing cycle."""
    if not os.environ.get("ARES_WORKER_GENERATION"):
        for request in ("stop.request", "checkpoint.request"):
            (directory / request).unlink(missing_ok=True)
    recovery = (initial_checkpoint or _latest_checkpoint(directory)) if resume else None
    consumed = 0
    for attempt in range(max_restarts + 1):
        try:
            return _run_attempt(game, backend, directory, resume=resume or attempt > 0,
                checkpoint=recovery, max_frames=None if max_frames is None else max(1, max_frames - consumed),
                seed=seed, report_interval=report_interval, restart_count=attempt)
        except _AttemptFailure as failure:
            consumed += failure.frames
            transient = isinstance(failure.original, OSError) and not isinstance(failure.original, FileNotFoundError)
            if not transient or failure.checkpoint is None or attempt >= max_restarts:
                raise failure.original from failure
            recovery = failure.checkpoint
            _atomic_write(directory / "recovery.json", json.dumps({"attempt": attempt + 1,
                "error": str(failure.original), "checkpoint": str(recovery), "published_at": time.time()}).encode())
            print(json.dumps({"game": game, "recovery_attempt": attempt + 1,
                              "error": str(failure.original), "checkpoint": str(recovery)}), flush=True)
            time.sleep(min(2.0, 0.25 * (attempt + 1)))
    raise AssertionError("bounded recovery loop exhausted without a result")


class _AttemptFailure(Exception):
    def __init__(self, original: Exception, checkpoint: Path | None, frames: int) -> None:
        super().__init__(str(original))
        self.original, self.checkpoint, self.frames = original, checkpoint, frames


def _run_attempt(game: Game, backend: str, directory: Path, *, resume: bool,
                 checkpoint: Path | None, max_frames: int | None, seed: int,
                 report_interval: int, restart_count: int) -> dict[str, object]:
    controller: AresAutonomousOrchestrator | None = None
    start = 0
    ready = False
    complete = False
    interrupted = False
    failure: Exception | None = None
    recovery: Path | None = checkpoint
    report: dict[str, object] = {"game": game, "backend": backend, "frames": 0}
    cleanup_errors: list[str] = []
    last_publish = 0.0
    recorder = TelemetryRecorder(directory)
    final_telemetry: dict[str, object] = {}
    try:
        _write_heartbeat(directory, "starting", 0, restart_count)
        if backend == "native":
            controller = AresAutonomousOrchestrator(ROOT / "native/build/ares_mgba.dll",
                ROOT.parent / ROM_NAMES[game], game == "emerald", checkpoint_dir=directory, seed=seed)
        else:
            controller = build_laboratory(game == "emerald", seed, checkpoint_dir=directory)
        if resume:
            _write_heartbeat(directory, "restoring", 0, restart_count)
            controller.load_checkpoint(checkpoint)
            if isinstance(controller, AresAutonomousOrchestrator):
                # A new process is a new attempt: the restored timeline's spent
                # recovery ladder must not end it within seconds.
                controller.rearm_recovery()
        start = controller.bridge.frame_count
        ready = True
        if not resume:
            _write_heartbeat(directory, "checkpointing", start, restart_count)
            controller.save_checkpoint("worker_start")
        _write_heartbeat(directory, "running", start, restart_count)
        while True:
            if backend == "native":
                complete = controller._bootstrapped and bool(controller.native_completion_evidence()["complete"])
            else:
                completed = controller.planner.completed_milestones(controller.bridge,
                    controller.bridge.get_decrypted_player_party())
                complete = len(completed) == len(controller.planner.campaign_dag)
            if complete or max_frames is not None and controller.bridge.frame_count - start >= max_frames:
                break
            controller.step_vblank_cycle()
            if time.monotonic() - last_publish >= 0.15:
                _write_heartbeat(directory, "running", controller.bridge.frame_count, restart_count)
                _publish_live(controller, directory)
                recorder.sample(controller)
                last_publish = time.monotonic()
                checkpoint_request = directory / "checkpoint.request"
                stop_request = directory / "stop.request"
                if checkpoint_request.is_file():
                    checkpoint_request.unlink()
                    _write_heartbeat(directory, "checkpointing", controller.bridge.frame_count, restart_count)
                    controller.save_checkpoint("requested")
                if stop_request.is_file():
                    stop_request.unlink()
                    interrupted = True
                    break
            if report_interval and controller.bridge.frame_count % report_interval == 0:
                report = _runtime_report(controller.status(), running=True)
                report["restart_count"] = restart_count
                _atomic_write(directory / "status.json", json.dumps(report, indent=2).encode("utf-8"))
                print(json.dumps(report), flush=True)
    except KeyboardInterrupt:
        interrupted = True
    except Exception as error:
        failure = error
        recovery = _latest_checkpoint(directory) if ready else checkpoint
    finally:
        _write_heartbeat(directory, "finalizing", controller.bridge.frame_count if controller else 0, restart_count)
        if controller is not None:
            if ready:
                try:
                    controller.save_checkpoint("error" if failure else ("complete" if complete else (
                        "interrupted" if interrupted else "frame_limit")))
                except Exception as error:
                    cleanup_errors.append(f"checkpoint: {type(error).__name__}: {error}")
                    if failure is None:
                        failure = error
            try:
                _save_screen(controller, directory / "latest.png")
                report = controller.status()
                final_telemetry = _telemetry_fields(controller)
                recorder.sample(controller, force=True)
                _publish_live(controller, directory, running=False,
                              error="" if failure is None else f"{type(failure).__name__}: {failure}")
            except Exception as error:
                cleanup_errors.append(f"telemetry: {type(error).__name__}: {error}")
            finally:
                try:
                    controller.close()
                except Exception as error:
                    cleanup_errors.append(f"close: {type(error).__name__}: {error}")
                    if failure is None:
                        failure = error
        report = _runtime_report(report, running=False,
                                 error="" if failure is None else f"{type(failure).__name__}: {failure}")
        report.update(complete=complete and failure is None, interrupted=interrupted, failed=failure is not None,
                      checkpoint_directory=str(directory), restart_count=restart_count,
                      cleanup_errors=cleanup_errors, **final_telemetry)
        _atomic_write(directory / "status.json", json.dumps(report, indent=2, default=str).encode("utf-8"))
        _atomic_write(directory / "live.json", json.dumps(report, default=str).encode("utf-8"))
        _write_heartbeat(directory, "stopped", int(cast(int, report.get("frames", 0))), restart_count)
    if failure is not None:
        raise _AttemptFailure(failure, recovery, max(0, int(cast(int, report.get("frames", start))) - start))
    return report


WorkerArgs = tuple[Game, str, bool, int | None, str | None, int, int]


def _worker(args: WorkerArgs, resume_checkpoint: str | None = None) -> dict[str, object]:
    game, backend, resume, frames, directory, seed, restarts = args
    try:
        return run_campaign(game, backend, resume=resume, max_frames=frames,
                            checkpoint_root=Path(directory) if directory else None, seed=seed, max_restarts=restarts,
                            resume_checkpoint=Path(resume_checkpoint) if resume_checkpoint else None)
    except Exception as error:
        report: dict[str, object] = {"game": game, "backend": backend, "complete": False,
            "failed": True, "running": False, "error": f"{type(error).__name__}: {error}"}
        print(json.dumps({"worker_failed": report}), flush=True)
        return report


def _supervised_child(args: WorkerArgs, generation: str, sender: Connection,
                      checkpoint: str | None) -> None:
    os.environ["ARES_WORKER_GENERATION"] = generation
    try:
        sender.send(_worker(args, checkpoint))
    finally:
        sender.close()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_checkpoint(directory: Path, game: Game, backend: str,
                          not_before: float | None = None, skip: int = 0,
                          excluded: frozenset[str] = frozenset(),
                          required_milestones: frozenset[str] = frozenset()) -> Path | None:
    """Recover immutable complete bundles; never rewrite a save or its pointer.

    ``skip`` passes over that many newest valid bundles, so a checkpoint that
    keeps failing the same way is replaced by an older, different one.
    """
    latest = _latest_checkpoint(directory)
    candidates = ([latest] if latest is not None else [])
    candidates.extend(sorted(directory.glob("frame-*"), key=lambda path: (_checkpoint_creation(path), path.name),
                             reverse=True)[:40])
    # Keep a healthy latest pointer first, then creation chronology. A verified
    # import is healthy recovery evidence, not permission to discard subsequent
    # training/XP from a requested, milestone or periodic save. Only explicitly
    # degraded snapshots lose priority, even if their frame number is greater.
    candidates.sort(key=lambda path: _degraded_checkpoint_reason(_checkpoint_report(path).get("reason")))
    skipped = 0
    for checkpoint in dict.fromkeys(candidates):
        if checkpoint.name in excluded:
            continue
        try:
            manifest_path = checkpoint / "manifest.json"
            if not_before is not None and manifest_path.stat().st_mtime < not_before:
                continue  # --new-game cannot silently resume a previous run.
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                continue
            if manifest.get("game") != game or manifest.get("native") is not (backend == "native"):
                continue
            if required_milestones:
                facts, _ = _durable_facts(manifest)
                if any(f"milestone:{milestone}" not in facts for milestone in required_milestones):
                    continue
            valid = all(_file_sha256(checkpoint / filename) == manifest.get(key) for filename, key in (
                ("emulator.state", "emulator_sha256"), ("controller.pt", "controller_sha256")))
            if "battery_sha256" in manifest:
                valid = valid and _file_sha256(checkpoint / "battery.sav") == manifest["battery_sha256"]
            if valid:
                if skipped < skip:
                    skipped += 1
                    continue
                return checkpoint
        except (OSError, ValueError, TypeError):
            continue
    return None


def _checkpoint_creation(path: Path) -> int:
    """Bundles are named frame-<frame>-<time_ns>; rewinds make frames non-monotonic."""
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return 0


def _degraded_checkpoint_reason(value: object) -> bool:
    reason = str(value).lower().replace("-", "_")
    return "error" in reason or reason.startswith(("stall", "fault", "failed", "crash", "timeout"))


# Configuration errors that no restart can fix; everything else is retried.
_FATAL_ERRORS = ("ROM revision mismatch", "Pinned symbol mismatch", "Checkpoint game/backend mismatch",
                 "CampaignAlreadyRunning", "already owned by another worker", "Supported backend",
                 "ares_mgba.dll", "No such file or directory", "FileNotFoundError")


def restartable_error(error: str) -> bool:
    return bool(error) and not any(marker in error for marker in _FATAL_ERRORS)


def _durable_facts(report: Mapping[str, object]) -> tuple[set[str], dict[str, int]]:
    """Only observed story facts count: frames, XP, wins and travel never do."""
    facts: set[str] = set()
    counts: dict[str, int] = {}
    sources = (report, nested_mapping(report, "status"))
    for source in sources:
        metrics = nested_mapping(source, "metrics")
        telemetry = nested_mapping(source, "telemetry")
        milestone_lists = (source.get("completed_milestones"), metrics.get("completed_milestones"),
            nested_mapping(source, "milestones").get("completed"),
            nested_mapping(telemetry, "milestones").get("completed"))
        for values in milestone_lists:
            if isinstance(values, (list, tuple)):
                facts.update(f"milestone:{value}" for value in values
                             if isinstance(value, str) and 0 < len(value) <= 200)
        evidence = nested_mapping(source, "native_completion_evidence")
        badges = evidence.get("badges")
        if isinstance(badges, list):
            facts.update(f"badge:{slot}" for slot, present in enumerate(badges[:8]) if present is True)
        if evidence.get("game_clear") is True:
            facts.add("game_clear")
        for key, value in (("hall_of_fame_entries", evidence.get("hall_of_fame_entries")),
                           ("persistent_flags", source.get("flags_progress")),
                           ("persistent_flags", metrics.get("flags_progress")),
                           ("persistent_flags", source.get("persistent_flags_count"))):
            if type(value) is int and 0 <= value <= 1_000_000:
                counts[key] = max(counts.get(key, 0), value)
        flags = source.get("persistent_flags")
        if isinstance(flags, Mapping):
            facts.update(f"flag:{key}" for key, present in flags.items()
                         if present is True and isinstance(key, str) and 0 < len(key) <= 200)
        elif isinstance(flags, (list, tuple)):
            facts.update(f"flag:{flag}" for flag in flags
                         if type(flag) is int and 0 <= flag <= 65535)
    return facts, counts


def _checkpoint_report(checkpoint: Path | None) -> dict[str, object]:
    if checkpoint is None:
        return {}
    try:
        path = checkpoint / "manifest.json"
        if path.stat().st_size > 1_000_000:
            return {}
        with path.open("rb") as stream:
            raw = stream.read(1_000_001)
        value: object = json.loads(raw) if len(raw) <= 1_000_000 else None
        return cast(dict[str, object], value) if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


class _RecoveryProgress:
    """Monotonic parent-side history survives cartridge and supervisor rewinds."""

    def __init__(self, directory: Path, game: Game, backend: str, *, resume: bool) -> None:
        self.path, self.game, self.backend = directory / "recovery-progress.json", game, backend
        self.facts: set[str] = set()
        self.counts: dict[str, int] = {}
        self.failures: dict[str, int] = {}
        self.stagnant_restarts = self.revision = 0
        self.session_started_at: float | None = None if resume else time.time()
        self.saved_at = 0.0  # when the loaded ledger was last written
        if resume and self.path.is_file():
            try:
                if self.path.stat().st_size > 1_000_000:
                    raise ValueError("recovery history exceeds bounded size")
                data = json.loads(self.path.read_bytes())
                if not isinstance(data, dict) or data.get("game") != game or data.get("backend") != backend:
                    raise ValueError("recovery history belongs to a different campaign")
                facts, counts, failures = data.get("facts"), data.get("counts"), data.get("failures", {})
                if (not isinstance(facts, list) or len(facts) > 4096 or
                        any(not isinstance(value, str) or len(value) > 256 for value in facts)
                        or not isinstance(counts, dict) or len(counts) > 16
                        or any(not isinstance(key, str) or type(value) is not int or value < 0
                               for key, value in counts.items())
                        or not isinstance(failures, dict) or len(failures) > 256
                        or any(not isinstance(key, str) or type(value) is not int or value < 0
                               for key, value in failures.items())):
                    raise ValueError("invalid recovery progress facts")
                stagnant, revision = data.get("stagnant_restarts", 0), data.get("revision", 0)
                if type(stagnant) is not int or stagnant < 0 or type(revision) is not int or revision < 0:
                    raise ValueError("invalid stagnant recovery budget")
                self.facts, self.counts, self.failures = set(facts), counts, failures
                self.stagnant_restarts, self.revision = stagnant, revision
                saved_at = data.get("published_at", 0.0)
                if isinstance(saved_at, (int, float)) and not isinstance(saved_at, bool) and math.isfinite(saved_at):
                    self.saved_at = float(saved_at)
                started = data.get("session_started_at")
                if started is not None:
                    if not isinstance(started, (int, float)) or isinstance(started, bool) or not math.isfinite(started) or started < 0:
                        raise ValueError("invalid recovery session boundary")
                    self.session_started_at = float(started)
            except (OSError, ValueError, TypeError) as error:
                raise ValueError(f"Invalid durable recovery history: {error}") from error

    def observe(self, *reports: Mapping[str, object]) -> bool:
        gained = False
        for report in reports:
            facts, counts = _durable_facts(report)
            novel = facts - self.facts
            if novel and len(self.facts) + len(novel) <= 4096:
                self.facts.update(novel)
                gained = True
            for key, value in counts.items():
                if value > self.counts.get(key, 0):
                    self.counts[key] = value
                    gained = True
        if gained:
            self.stagnant_restarts = 0
            self.revision += 1
            self.save()
        return gained

    @property
    def required_milestones(self) -> frozenset[str]:
        return frozenset(fact.removeprefix("milestone:") for fact in self.facts if fact.startswith("milestone:"))

    def mark_failed(self, checkpoint: Path | None) -> None:
        if checkpoint is not None:
            self.failures[checkpoint.name] = self.failures.get(checkpoint.name, 0) + 1
            if len(self.failures) > 256:
                self.failures.pop(next(iter(self.failures)))

    def save(self) -> None:
        value = {"game": self.game, "backend": self.backend, "facts": sorted(self.facts),
            "counts": self.counts, "failures": self.failures,
            "stagnant_restarts": self.stagnant_restarts, "revision": self.revision,
            "session_started_at": self.session_started_at,
            "published_at": time.time()}
        if not _atomic_write(self.path, json.dumps(value).encode("utf-8")):
            raise OSError(f"Unable to persist durable recovery budget: {self.path}")


def _supervisor_event(directory: Path, **values: object) -> None:
    event = {"published_at": time.time(), "supervisor_pid": os.getpid(), **values}
    try:
        with (directory / "supervisor-events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
    except OSError:
        pass
    print(json.dumps({"supervisor": event}), flush=True)


def _supervisor_report(directory: Path, game: Game, backend: str, *, pid: int,
                        generation: str, error: str, interrupted: bool,
                        restart_count: int) -> dict[str, object]:
    report = read_campaign_status(directory)
    if report.get("pid") != pid and report.get("worker_generation") != generation:
        report = {}  # Never attach previous-run success to a failed new child.
    report.update(game=game, backend=backend, running=False, failed=not interrupted,
                  complete=False, interrupted=interrupted, error=error, pid=pid,
                  worker_generation=generation, published_at=time.time(),
                  supervisor_pid=os.getpid(), supervisor_restarts=restart_count,
                  checkpoint_directory=str(directory))
    _atomic_write(directory / "status.json", json.dumps(report, indent=2).encode("utf-8"))
    _atomic_write(directory / "live.json", json.dumps(report).encode("utf-8"))
    return report


def _owned_status(directory: Path, pid: int, generation: str) -> dict[str, object]:
    report = read_campaign_status(directory)
    owner = report.get("worker_generation")
    if isinstance(owner, str) and owner:
        return report if owner == generation else {}
    return report if report.get("pid") == pid else {}


def _blocked_recovery(directory: Path, report: dict[str, object], reason: str,
                       progress: _RecoveryProgress, stagnant_limit: int) -> dict[str, object]:
    report.update(recovery_blocked=reason, stagnant_restarts=progress.stagnant_restarts,
                  stagnant_restart_limit=stagnant_limit)
    _atomic_write(directory / "status.json", json.dumps(report, indent=2).encode("utf-8"))
    _atomic_write(directory / "live.json", json.dumps(report).encode("utf-8"))
    _supervisor_event(directory, kind="recovery_blocked", game=progress.game,
                      error=reason, stagnant_restarts=progress.stagnant_restarts,
                      stagnant_restart_limit=stagnant_limit, durable_facts=len(progress.facts))
    return report


def supervise_campaign(args: WorkerArgs, *, heartbeat_timeout: float = 120.0,
                        poll_interval: float = 0.25, startup_timeout: float | None = None,
                        stop_grace: float = 15.0, stop_event: Event | None = None,
                        child_target: Callable[[WorkerArgs, str, Connection, str | None], None] = _supervised_child,
                        stagnant_restart_limit: int = 3,
                        stagnation_cooldown: float | None = None,
                        ) -> dict[str, object]:
    """Bound dead native/model calls in a disposable child, preserving saves.

    The parent owns no emulator and generates no hardware input. An independent
    monitor thread per game only watches a child-authored heartbeat and pipe.
    With ``stagnation_cooldown`` an exhausted stagnant budget pauses instead of
    ending the campaign; durable lessons (lessons.json) make each later attempt
    different from the failed ones.
    """
    deadlines = (heartbeat_timeout, poll_interval, stop_grace) + (() if startup_timeout is None else (startup_timeout,))
    if any(not math.isfinite(value) or value <= 0 for value in deadlines):
        raise ValueError("Supervisor deadlines must be positive")
    if type(stagnant_restart_limit) is not int or stagnant_restart_limit < 1:
        raise ValueError("Stagnant recovery restart limit must be a positive integer")
    if stagnation_cooldown is not None and (not math.isfinite(stagnation_cooldown) or stagnation_cooldown < 0):
        raise ValueError("Stagnation cooldown must be finite and nonnegative")
    game, backend, resume, frames, root, seed, budget = args
    directory = _checkpoint_root(backend, game, Path(root) if root else None)
    directory.mkdir(parents=True, exist_ok=True)
    with CampaignWorkerLock(directory, "supervisor.lock"):
        # Preflight also rejects an independently launched low-level worker.
        with CampaignWorkerLock(directory):
            for request in ("stop.request", "checkpoint.request"):
                (directory / request).unlink(missing_ok=True)
        return _supervise_owned(args, directory, heartbeat_timeout=heartbeat_timeout,
            poll_interval=poll_interval, startup_timeout=startup_timeout, stop_grace=stop_grace,
            stop_event=stop_event, child_target=child_target, stagnant_restart_limit=stagnant_restart_limit,
            stagnation_cooldown=stagnation_cooldown)


def _cooldown(directory: Path, game: str, progress: _RecoveryProgress, seconds: float,
              stop_event: Event | None, poll_interval: float) -> bool:
    """Pause an exhausted stagnant budget, then re-arm it. False if stopped meanwhile."""
    _supervisor_event(directory, kind="recovery_cooldown", game=game, stagnant_restarts=progress.stagnant_restarts,
                      cooldown_seconds=seconds,
                      reason="No new durable milestone; pausing, then retrying with accumulated lessons")
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if (stop_event is not None and stop_event.is_set()) or (directory / "stop.request").is_file():
            return False
        time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
    progress.stagnant_restarts = 0
    progress.failures.clear()
    progress.save()
    return True


def _supervise_owned(args: WorkerArgs, directory: Path, *, heartbeat_timeout: float,
                     poll_interval: float, startup_timeout: float | None, stop_grace: float,
                     stop_event: Event | None,
                     child_target: Callable[[WorkerArgs, str, Connection, str | None], None],
                     stagnant_restart_limit: int = 3,
                     stagnation_cooldown: float | None = None,
                     ) -> dict[str, object]:
    game, backend, resume, frames, root, seed, budget = args
    context = multiprocessing.get_context("spawn")
    checkpoint: Path | None = None
    restarts = consumed = 0
    progress = _RecoveryProgress(directory, game, backend, resume=resume)
    if resume:
        # Infer legacy ledgers only inside the latest observed fresh-game session.
        # Retained manifests from an older --new-game cannot impose its badges.
        history = [(path, _checkpoint_report(path)) for path in sorted(directory.glob("frame-*"),
                   key=lambda path: (_checkpoint_creation(path), path.name), reverse=True)[:256]]
        history = [(path, report) for path, report in history
                   if report.get("game") == game and report.get("native") is (backend == "native")]
        if progress.session_started_at is None:
            starts = [(path / "manifest.json").stat().st_mtime for path, report in history
                      if report.get("reason") == "worker_start"]
            if starts:
                progress.session_started_at = max(starts)
        manifests = [report for path, report in history if progress.session_started_at is not None
                     and (path / "manifest.json").stat().st_mtime >= progress.session_started_at]
        progress.observe(read_campaign_status(directory), _checkpoint_report(_latest_checkpoint(directory)), *manifests)
    new_game_since = progress.session_started_at
    progress.save()
    if progress.stagnant_restarts >= stagnant_restart_limit and stagnation_cooldown is not None:
        # The pause counts from when the budget was exhausted: resuming hours
        # later starts at once, resuming immediately still waits it out.
        pause = max(0.0, stagnation_cooldown - max(0.0, time.time() - progress.saved_at))
        if not _cooldown(directory, game, progress, pause, stop_event, poll_interval):
            return read_campaign_status(directory) or {"game": game, "backend": backend, "interrupted": True}
    if progress.stagnant_restarts >= stagnant_restart_limit:
        previous = read_campaign_status(directory)
        reason = "Stagnant recovery limit exhausted without a new durable milestone"
        with CampaignWorkerLock(directory):
            report = _supervisor_report(directory, game, backend, pid=int(cast(int, previous.get("pid", 0))),
                generation=str(previous.get("worker_generation", "")), error=reason, interrupted=False,
                restart_count=int(cast(int, previous.get("supervisor_restarts", 0))))
            return _blocked_recovery(directory, report, reason, progress, stagnant_restart_limit)
    if resume and progress.required_milestones:
        checkpoint = _verified_checkpoint(directory, game, backend, new_game_since,
                                           required_milestones=progress.required_milestones)
        if checkpoint is None:
            previous = read_campaign_status(directory)
            reason = "No verified checkpoint preserves the durable milestone highwater"
            with CampaignWorkerLock(directory):
                report = _supervisor_report(directory, game, backend, pid=int(cast(int, previous.get("pid", 0))),
                    generation=str(previous.get("worker_generation", "")), error=reason, interrupted=False,
                    restart_count=int(cast(int, previous.get("supervisor_restarts", 0))))
                return _blocked_recovery(directory, report, reason, progress, stagnant_restart_limit)
    unlimited = budget < 0
    while True:
        attempt_revision = progress.revision
        generation = uuid.uuid4().hex
        receiver, sender = context.Pipe(duplex=False)
        remaining = None if frames is None else max(1, frames - consumed)
        base_checkpoint = checkpoint or (_latest_checkpoint(directory) if resume else None)
        start_frame = 0
        if base_checkpoint is not None:
            try:
                start_frame = int(json.loads((base_checkpoint / "manifest.json").read_bytes()).get("frame", 0))
            except (OSError, ValueError, TypeError):
                pass
        # Each restart uses a different seed so stochastic policies take another path.
        worker_budget = 2 if unlimited else max(0, budget - restarts)
        task: WorkerArgs = (game, backend, resume or restarts > 0, remaining, root, seed + restarts, worker_budget)
        process = context.Process(target=child_target, args=(task, generation, sender,
                                  str(checkpoint) if checkpoint else None), name=f"ares-{game}")
        process.start()
        sender.close()
        pid = process.pid or 0
        last_seen = time.monotonic()
        last_sequence: object = None
        heartbeat: dict[str, object] = {}
        result: dict[str, object] | None = None
        cause = ""
        stopping = False
        stop_at = 0.0
        _supervisor_event(directory, kind="worker_started", game=game, pid=pid,
            generation=generation, restart_count=restarts, checkpoint=str(checkpoint) if checkpoint else None)
        try:
            while True:
                now = time.monotonic()
                if stop_event is not None and stop_event.is_set():
                    _atomic_write(directory / "stop.request", str(time.time()).encode("ascii"))
                if not stopping and (directory / "stop.request").is_file():
                    stopping, stop_at = True, now
                try:
                    observed = json.loads((directory / "heartbeat.json").read_bytes())
                    if isinstance(observed, dict) and observed.get("generation") == generation and observed.get("pid") == pid:
                        heartbeat = observed
                        if observed.get("sequence") != last_sequence:
                            last_sequence, last_seen = observed.get("sequence"), now
                            progress.observe(_owned_status(directory, pid, generation))
                except (OSError, ValueError):
                    pass
                if receiver.poll():
                    try:
                        received: object = receiver.recv()
                        if isinstance(received, dict):
                            result = cast(dict[str, object], received)
                    except (EOFError, OSError):
                        pass
                    if result is not None:
                        process.join(timeout=min(5.0, heartbeat_timeout))
                        if process.is_alive():
                            # Production sends only after run_campaign finally
                            # closed the backend. Slow Python/PyTorch shutdown
                            # cannot turn a valid finished run into a gameplay
                            # failure or replay a completed native campaign.
                            _supervisor_event(directory, kind="worker_exit_cleanup", game=game,
                                pid=pid, generation=generation, error="Slow interpreter exit after final result")
                        break
                if not process.is_alive():
                    # A result written just before exit can arrive on the next
                    # poll. Pipe EOF distinguishes a hard crash from slow code.
                    if receiver.poll(0.05):
                        try:
                            received = receiver.recv()
                            if isinstance(received, dict):
                                result = cast(dict[str, object], received)
                        except (EOFError, OSError):
                            pass
                    cause = "" if result is not None else f"Worker exited unexpectedly (exit code {process.exitcode})"
                    break
                timeout = heartbeat_timeout if heartbeat else (startup_timeout or heartbeat_timeout)
                if stopping and now - stop_at >= min(stop_grace, timeout):
                    cause = "Stopped unresponsive worker after graceful-stop deadline"
                    break
                if now - last_seen >= timeout:
                    cause = f"HeartbeatTimeout: worker made no heartbeat for {timeout:g} seconds"
                    break
                time.sleep(poll_interval)
        except BaseException:
            # A supervisor failure must never leave an unmonitored child alive.
            if process.is_alive():
                process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
            raise
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2.0)
            receiver.close()
            if not process.is_alive():
                process.join()
        if result is not None and not cause:
            progress.observe(result, _owned_status(directory, pid, generation))
            error = str(result.get("error", ""))
            gameplay_failure = (bool(result.get("failed")) and not result.get("complete")
                                and not result.get("interrupted") and not stopping)
            if not gameplay_failure or not restartable_error(error) or frames is not None:
                result["supervisor_restarts"] = restarts
                _supervisor_event(directory, kind="worker_finished", game=game, pid=pid,
                                  generation=generation, complete=result.get("complete", False),
                                  failed=result.get("failed", False), restart_count=restarts)
                process.close()
                return result
            # A finished-but-failed worker (e.g. an exhausted stall ladder) is
            # restarted like a crash: a campaign never stops on its own.
            cause = f"WorkerFailed: {error}"
        internal = heartbeat.get("restart_count", 0)
        spent = restarts + (max(0, int(internal)) if isinstance(internal, int) else 0)
        _supervisor_event(directory, kind="worker_terminated", game=game, pid=pid,
                          generation=generation, error=cause, exitcode=process.exitcode,
                          heartbeat=heartbeat, restart_count=spent, stop_requested=stopping)
        process.close()
        last_frame = heartbeat.get("frames", start_frame)
        if isinstance(last_frame, int):
            consumed += max(0, last_frame - start_frame)
        with CampaignWorkerLock(directory):
            # Observe the newest intact bundle first; its newly achieved story
            # facts raise the floor before any rollback candidate is selected.
            newest = _verified_checkpoint(directory, game, backend, new_game_since)
            progress.observe(_checkpoint_report(newest), _owned_status(directory, pid, generation))
            required = progress.required_milestones
            newest = _verified_checkpoint(directory, game, backend, new_game_since,
                                            required_milestones=required)
            progressed = progress.revision > attempt_revision
            if not progressed:
                progress.mark_failed(base_checkpoint)
                if base_checkpoint is not None:
                    progress.mark_failed(newest)
            excluded = frozenset(progress.failures)
            checkpoint = _verified_checkpoint(directory, game, backend, new_game_since, excluded=excluded,
                                                required_milestones=required)
            if checkpoint is None:
                skip = progress.failures.get(newest.name, 0) if newest is not None else 0
                checkpoint = _verified_checkpoint(directory, game, backend, new_game_since, skip=skip,
                                                    required_milestones=required) or newest
            report = _supervisor_report(directory, game, backend, pid=pid, generation=generation,
                error=cause, interrupted=stopping, restart_count=spent)
            report.update(stagnant_restarts=progress.stagnant_restarts,
                          stagnant_restart_limit=stagnant_restart_limit)
        exhausted = not unlimited and spent >= budget
        stagnant_exhausted = not progressed and progress.stagnant_restarts >= stagnant_restart_limit
        if (stagnant_exhausted and stagnation_cooldown is not None and not stopping and not exhausted and
                checkpoint is not None and (frames is None or consumed < frames)):
            progress.save()
            if not _cooldown(directory, game, progress, stagnation_cooldown, stop_event, poll_interval):
                return report
            stagnant_exhausted = False
            checkpoint = _verified_checkpoint(directory, game, backend, new_game_since,
                                              required_milestones=progress.required_milestones) or checkpoint
        if stopping or exhausted or stagnant_exhausted or checkpoint is None or frames is not None and consumed >= frames:
            progress.save()
            if not stopping:
                if stagnant_exhausted:
                    return _blocked_recovery(directory, report,
                        "Stagnant recovery limit exhausted without a new durable milestone",
                        progress, stagnant_restart_limit)
                if checkpoint is None:
                    return _blocked_recovery(directory, report,
                        ("No verified checkpoint preserves the durable milestone highwater" if progress.required_milestones else
                         "No complete checkpoint from this run passed integrity validation"),
                        progress, stagnant_restart_limit)
            return report
        restarts = spent + 1
        if not progressed:
            progress.stagnant_restarts += 1
        progress.save()
        consecutive = min(6, progress.stagnant_restarts)
        delay = 0.0 if progressed else min(60.0, float(2 ** consecutive) - 1.0)
        _supervisor_event(directory, kind="worker_restart", game=game, restart_count=restarts,
                          checkpoint=str(checkpoint), error=cause, backoff_seconds=delay, seed=seed + restarts)
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            if (stop_event is not None and stop_event.is_set()) or (directory / "stop.request").is_file():
                break
            time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))


def _supervisor_cli_task(args: WorkerArgs, timeout: float, stop_event: Event,
                         stagnant_restart_limit: int = 3,
                         stagnation_cooldown: float | None = None) -> dict[str, object]:
    try:
        return supervise_campaign(args, heartbeat_timeout=timeout, stop_event=stop_event,
                                  stagnant_restart_limit=stagnant_restart_limit,
                                  stagnation_cooldown=stagnation_cooldown)
    except Exception as error:
        report: dict[str, object] = {"game": args[0], "backend": args[1], "complete": False,
            "failed": True, "running": False, "error": f"{type(error).__name__}: {error}"}
        print(json.dumps({"supervisor_failed": report}), flush=True)
        return report


def main() -> None:
    """Both games use independent processes; one game cannot starve the other."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "synthetic"), default="native")
    parser.add_argument("--game", choices=("firered", "emerald", "both"), default="both")
    launch = parser.add_mutually_exclusive_group()
    launch.add_argument("--resume", action="store_true")
    launch.add_argument("--new-game", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-restarts", type=int, default=-1,
                        help="Supervisor restart budget for crashes, hangs and failed workers; -1 means unlimited")
    parser.add_argument("--stagnant-restart-limit", type=int, default=3,
                        help="Consecutive recovery restarts allowed without a new durable milestone")
    parser.add_argument("--stagnation-cooldown", type=float, default=300.0,
                        help="Seconds to pause when the stagnant budget is exhausted before retrying with the "
                             "accumulated lessons; a negative value stops the campaign instead")
    parser.add_argument("--heartbeat-timeout", type=float, default=120.0,
                        help="Wall-clock seconds without child heartbeat before bounded checkpoint recovery")
    parser.add_argument("--status", action="store_true")
    control = parser.add_mutually_exclusive_group()
    control.add_argument("--stop", action="store_true", help="Request checkpoint and graceful stop of existing workers")
    control.add_argument("--checkpoint", action="store_true", help="Request immediate checkpoint from existing workers")
    args = parser.parse_args()
    if args.max_restarts < -1:
        parser.error("--max-restarts must be -1 (unlimited) or nonnegative")
    if args.stagnant_restart_limit < 1:
        parser.error("--stagnant-restart-limit must be positive")
    if not math.isfinite(args.heartbeat_timeout) or args.heartbeat_timeout <= 0:
        parser.error("--heartbeat-timeout must be finite and positive")
    games: tuple[Game, ...] = ("firered", "emerald") if args.game == "both" else (cast(Game, args.game),)
    if args.stop or args.checkpoint:
        request = "stop.request" if args.stop else "checkpoint.request"
        for game in games:
            directory = _checkpoint_root(args.backend, game, args.checkpoint_root)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / request).write_text(str(time.time()), encoding="utf-8")
        print(json.dumps({"requested": request, "games": games}))
        return
    if args.status:
        reports: list[dict[str, object]] = []
        for game in games:
            directory = _checkpoint_root(args.backend, game, args.checkpoint_root)
            report = read_campaign_status(directory) or {"game": game, "backend": args.backend, "status": "not_started"}
            reports.append(campaign_liveness(report))
        print(json.dumps(reports, indent=2))
        return
    tasks = [(game, str(args.backend), bool(args.resume), cast(int | None, args.max_frames),
              str(args.checkpoint_root) if args.checkpoint_root else None, int(args.seed), int(args.max_restarts)) for game in games]
    stop_event = Event()
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        cooldown = None if args.stagnation_cooldown < 0 else float(args.stagnation_cooldown)
        futures = [pool.submit(_supervisor_cli_task, task, float(args.heartbeat_timeout), stop_event,
                               int(args.stagnant_restart_limit), cooldown) for task in tasks]
        results = []
        try:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps({"worker_finished": result}), flush=True)
        except KeyboardInterrupt:
            stop_event.set()
            print(json.dumps({"requested": "stop", "games": games}), flush=True)
            for future in futures:
                result = future.result()
                if result not in results:
                    results.append(result)
    print(json.dumps(results, indent=2))
    if not all(result["complete"] for result in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
