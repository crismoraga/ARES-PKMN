"""Replay saved native stalls into a separate output directory and measure gains."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import cast

import torch

from core.game_data import Game, ROM_NAMES, ROOT
from core.orchestrator import AresAutonomousOrchestrator
from core.progress_watchdog import ProgressWatchdog
from scripts.run_campaign import CampaignWorkerLock, _publish_live, _save_screen


def verify(game: Game, source_root: Path, output_root: Path, max_frames: int,
           reset_watchdog: bool = False) -> dict[str, object]:
    source = source_root.resolve() / "native" / game
    output = output_root.resolve() / "native" / game
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Recovery replay must preserve the source campaign directory")
    pointer = json.loads((source / "latest.json").read_text(encoding="utf-8"))
    checkpoint = source / str(pointer["directory"])
    output.mkdir(parents=True, exist_ok=True)
    with CampaignWorkerLock(output):
        return _verify_owned(game, checkpoint, output, max_frames, reset_watchdog)


def _verify_owned(game: Game, checkpoint: Path, output: Path, max_frames: int,
                  reset_watchdog: bool) -> dict[str, object]:
    torch.set_num_threads(2)
    controller = AresAutonomousOrchestrator(ROOT / "native/build/ares_mgba.dll",
        ROOT.parent / ROM_NAMES[game], game == "emerald", checkpoint_dir=output)
    started = time.monotonic()
    try:
        controller.load_checkpoint(checkpoint)
        if reset_watchdog:
            controller.watchdog = ProgressWatchdog()
            controller._record("stall_recovery_rearmed", reason="explicit_recovery_probe")
        initial = controller.status()
        initial_mode = controller.menu.menu_mode()
        initial_party = controller.bridge.get_decrypted_player_party()
        initial_xp = sum(int(mon.get("experience", 0)) for mon in initial_party)
        initial_outcomes = controller.metrics.wins + controller.metrics.losses + controller.metrics.captures
        initial_milestones = len(controller.metrics.completed_milestones)
        frames = controller.bridge.frame_count
        last_publish = 0.0
        for _ in range(max_frames):
            controller.step_vblank_cycle()
            now = time.monotonic()
            if now - last_publish >= 0.15:
                _publish_live(controller, output)
                last_publish = now
        party = controller.bridge.get_decrypted_player_party()
        final = controller.status()
        gains = {
            "experience": sum(int(mon.get("experience", 0)) for mon in party) - initial_xp,
            "battle_outcomes": controller.metrics.wins + controller.metrics.losses + controller.metrics.captures - initial_outcomes,
            "milestones": len(controller.metrics.completed_milestones) - initial_milestones,
            "map_changed": cast(dict[str, int], initial["position"])["map_group"] != controller.bridge.get_player_coords_and_map()["map_group"] or
                           cast(dict[str, int], initial["position"])["map_num"] != controller.bridge.get_player_coords_and_map()["map_num"],
        }
        progressed = any(bool(value) for value in gains.values())
        result: dict[str, object] = {"game": game, "source_checkpoint": str(checkpoint),
            "initial_menu": initial_mode, "final_menu": controller.menu.menu_mode(),
            "frames_replayed": controller.bridge.frame_count - frames,
            "elapsed_seconds": time.monotonic() - started, "observed_gains": gains,
            "progressed": progressed, "native_campaign_certified": False,
            "initial": initial, "final": final}
        controller.save_checkpoint("recovery_probe")
        _save_screen(controller, output / "latest.png")
        (output / "recovery_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        _publish_live(controller, output, running=False)
        return result
    except Exception as error:
        _publish_live(controller, output, running=False, error=f"{type(error).__name__}: {error}")
        controller.save_checkpoint("recovery_probe_error")
        _save_screen(controller, output / "latest.png")
        raise
    finally:
        controller.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=("firered", "emerald"), required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT / "artifacts/campaigns")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/recovery_verification")
    parser.add_argument("--max-frames", type=int, default=30000)
    parser.add_argument("--reset-watchdog", action="store_true",
                        help="Rearm bounded attempts when replaying a diagnosed/fixed failed checkpoint")
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be positive")
    result = verify(cast(Game, args.game), args.source_root, args.output_root, int(args.max_frames), args.reset_watchdog)
    print(json.dumps(result, indent=2))
    if not result["progressed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
