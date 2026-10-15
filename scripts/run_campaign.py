"""Run/resume genuine cartridge campaigns or explicitly separate laboratories."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import time
from pathlib import Path
from typing import Any, cast

import torch

from core.bridge.ffi_mgba import NativeLibMGBABridge
from core.game_data import Game, ROM_NAMES, ROOT
from core.orchestrator import AresAutonomousOrchestrator
from scripts.run_autonomous_verification_demo import build_laboratory


def _checkpoint_root(backend: str, game: Game, root: Path | None = None) -> Path:
    return (root or ROOT / "artifacts/campaigns") / backend / game


def _save_screen(controller: AresAutonomousOrchestrator, path: Path) -> None:
    """Save framebuffer evidence from the actual native handle, without a window."""
    if isinstance(controller.bridge, NativeLibMGBABridge):
        import pygame
        rgb = controller.bridge.framebuffer().view("uint8").reshape(160, 240, 4)[:, :, :3]
        surface = pygame.image.frombuffer(rgb.tobytes(), (240, 160), "RGB")
        pygame.image.save(surface, str(path))


def _publish_live(controller: AresAutonomousOrchestrator, directory: Path,
                  running: bool = True, error: str = "") -> None:
    """Publish the real emulator framebuffer and telemetry for the read-only viewer."""
    if not isinstance(controller.bridge, NativeLibMGBABridge):
        return
    pixels = controller.bridge.framebuffer().view("uint8").reshape(160, 240, 4)[:, :, :3]
    frame = directory / "live.rgb.tmp"
    frame.write_bytes(pixels.tobytes())
    try:
        frame.replace(directory / "live.rgb")
    except PermissionError:
        # Windows readers can briefly deny replacement. Telemetry must never
        # interrupt the emulator; the next publication retries without waiting.
        return
    report = controller.status()
    report.update(running=running, pid=os.getpid(), published_at=time.time(), error=error)
    if controller._bootstrapped:
        try:
            report["menu"] = controller.menu.menu_mode()
            report["party"] = [{"species": mon["species"], "level": mon["level"],
                                "hp": mon["hp"], "maxHP": mon["maxHP"]}
                               for mon in controller.bridge.get_decrypted_player_party()]
        except ValueError:
            report["menu"] = "snapshot_retry"
    status = directory / "live.json.tmp"
    status.write_text(json.dumps(report), encoding="utf-8")
    try:
        status.replace(directory / "live.json")
    except PermissionError:
        return


def run_campaign(game: Game, backend: str = "native", *, resume: bool = False,
                 max_frames: int | None = None, checkpoint_root: Path | None = None,
                 seed: int = 42, report_interval: int = 5000) -> dict[str, object]:
    """Run until verified success or the user's optional local probe frame cap."""
    if backend not in ("native", "synthetic") or max_frames is not None and max_frames < 1:
        raise ValueError("Supported backend and positive frame limit required")
    torch.set_num_threads(8)
    directory = _checkpoint_root(backend, game, checkpoint_root)
    directory.mkdir(parents=True, exist_ok=True)
    if backend == "native":
        controller = AresAutonomousOrchestrator(ROOT / "native/build/ares_mgba.dll",
            ROOT.parent / ROM_NAMES[game], game == "emerald", checkpoint_dir=directory, seed=seed)
    else:
        controller = build_laboratory(game == "emerald", seed, checkpoint_dir=directory)
    if resume:
        controller.load_checkpoint()
    start = controller.bridge.frame_count
    complete = False
    interrupted = False
    failed = False
    error_message = ""
    last_publish = 0.0
    try:
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
                _publish_live(controller, directory)
                last_publish = time.monotonic()
                checkpoint_request = directory / "checkpoint.request"
                stop_request = directory / "stop.request"
                if checkpoint_request.is_file():
                    checkpoint_request.unlink()
                    controller.save_checkpoint("requested")
                if stop_request.is_file():
                    stop_request.unlink()
                    interrupted = True
                    break
            if report_interval and controller.bridge.frame_count % report_interval == 0:
                report = controller.status()
                (directory / "status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(json.dumps(report), flush=True)
    except KeyboardInterrupt:
        interrupted = True
    except Exception as error:
        failed = True
        error_message = f"{type(error).__name__}: {error}"
        raise
    finally:
        controller.save_checkpoint("error" if failed else ("complete" if complete else ("interrupted" if interrupted else "frame_limit")))
        _save_screen(controller, directory / "latest.png")
        report = controller.status()
        report["complete"] = complete
        report["interrupted"] = interrupted
        report["checkpoint_directory"] = str(directory)
        (directory / "status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        _publish_live(controller, directory, running=False, error=error_message)
        controller.close()
    return report


def _worker(args: tuple[Game, str, bool, int | None, str | None, int]) -> dict[str, object]:
    game, backend, resume, frames, directory, seed = args
    return run_campaign(game, backend, resume=resume, max_frames=frames,
                        checkpoint_root=Path(directory) if directory else None, seed=seed)


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
    parser.add_argument("--status", action="store_true")
    control = parser.add_mutually_exclusive_group()
    control.add_argument("--stop", action="store_true", help="Request checkpoint and graceful stop of existing workers")
    control.add_argument("--checkpoint", action="store_true", help="Request immediate checkpoint from existing workers")
    args = parser.parse_args()
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
        reports: list[dict[str, Any]] = []
        for game in games:
            path = _checkpoint_root(args.backend, game, args.checkpoint_root) / "status.json"
            reports.append(json.loads(path.read_text(encoding="utf-8")) if path.is_file() else
                           {"game": game, "backend": args.backend, "status": "not_started"})
        print(json.dumps(reports, indent=2))
        return
    tasks = [(game, str(args.backend), bool(args.resume), cast(int | None, args.max_frames),
              str(args.checkpoint_root) if args.checkpoint_root else None, int(args.seed)) for game in games]
    if len(tasks) == 2:
        with ProcessPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(_worker, tasks))
    else:
        results = [_worker(tasks[0])]
    print(json.dumps(results, indent=2))
    if not all(result["complete"] for result in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
