"""Boot verified ROMs through actual mGBA and record introduction evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from core.bridge.ffi_mgba import NativeLibMGBABridge
from core.game_data import Game, ROM_NAMES, ROOT
from core.native_bootstrap import NativeUIController


def probe(game: Game, max_frames: int = 60000) -> dict[str, object]:
    """Advance one new game until field entry; never count it as campaign clearance."""
    output = ROOT / "artifacts" / game
    output.mkdir(parents=True, exist_ok=True)
    with NativeLibMGBABridge(ROOT / "native" / "build" / "ares_mgba.dll",
                             ROOT.parent / ROM_NAMES[game], game == "emerald") as bridge:
        ui = NativeUIController(bridge, game)
        with (output / "bootstrap.jsonl").open("w", encoding="utf-8") as log:
            previous = ""
            for _ in range(max_frames):
                keys = ui.choose_keys()
                if ui.last_callback != previous:
                    log.write(json.dumps({"frame": bridge.frame_count, "callback": ui.last_callback}) + "\n")
                    previous = ui.last_callback
                if keys is None:
                    break
                bridge.set_hardware_keys(keys)
                bridge.step_frame()
            bridge.set_hardware_keys(0)
            status = ui.status()
            (output / "bootstrap.state").write_bytes(bridge.save_state())
            import pygame
            rgb = bridge.framebuffer().view("uint8").reshape(160, 240, 4)[:, :, :3]
            surface = pygame.image.frombuffer(rgb.tobytes(), (240, 160), "RGB")
            pygame.image.save(surface, str(output / "bootstrap.png"))
            result: dict[str, object] = {"game": game, "frames": bridge.frame_count,
                                         "callback": status.callback, "field_entered": status.in_field,
                                         "campaign_complete": False}
            if status.in_field:
                result["position"] = bridge.get_player_coords_and_map()
                result["player_name"] = bridge.read_memory(bridge.get_save_block2_dynamic_ptr(), 8).hex()
            (output / "bootstrap_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result


def main() -> None:
    """Probe one or both supplied games with truthful nonzero failure status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=("firered", "emerald", "both"), default="both")
    parser.add_argument("--max-frames", type=int, default=60000)
    args = parser.parse_args()
    games: tuple[Game, ...] = ("firered", "emerald") if args.game == "both" else (cast(Game, args.game),)
    results = [probe(game, args.max_frames) for game in games]
    print(json.dumps(results, indent=2))
    if not all(item["field_entered"] for item in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
