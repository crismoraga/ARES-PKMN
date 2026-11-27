"""ARES-U Game Abstraction Layer tools: probe games, build start states, verify/discover RAM adapters.

    python -m scripts.ares_u_gal list
    python -m scripts.ares_u_gal intro    --game platino --frames 30000 --every 1500
    python -m scripts.ares_u_gal pick     --game platino --state artifacts/ares_u/gal/intro/platino/f021000.state
    python -m scripts.ares_u_gal verify   --game rojo
    python -m scripts.ares_u_gal discover --game platino

``intro`` drives a game from power-on with a fixed, game-agnostic button schedule (A pulses, a
periodic START and B) and saves a savestate plus a screenshot every ``--every`` frames, with a
contact sheet for visual inspection; the chosen state (the player standing in the overworld after
the intro) becomes the start state of the pixel agent. ``discover`` finds the player's x/y (and a
map-id candidate) automatically: it walks the player and keeps the RAM halfwords whose value
moves by exactly +-1 per tile in the walked direction and stays put in the other.
RAM is only ever used for measurement; the agent sees pixels.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ares_u.gal.backends import (GBA_A, GBA_B, GBA_DOWN, GBA_LEFT, GBA_RIGHT, GBA_START, GBA_UP, DeSmuMEBridge,
                                 MGBAMultiBridge, make_bridge)
from ares_u.gal.games import GAL_ARTIFACTS, GAMES, CoordSpec, Field, GameSpec, game
from ares_u.nav.env import NATIVE_GAMES, default_state_path

Bridge = MGBAMultiBridge | DeSmuMEBridge


def to_rgb(frame: NDArray[np.uint32]) -> NDArray[np.uint8]:
    rgb: NDArray[np.uint8] = np.stack([frame & 0xFF, (frame >> 8) & 0xFF, (frame >> 16) & 0xFF], axis=-1).astype(np.uint8)
    return rgb


def save_png(frame: NDArray[np.uint32], path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_rgb(frame)).save(path)


def contact_sheet(images: list[tuple[str, Path]], out: Path, columns: int = 6, scale: float = 0.75) -> Path:
    from PIL import Image, ImageDraw

    thumbs = []
    for label, path in images:
        image = Image.open(path).convert("RGB")
        image = image.resize((int(image.width * scale), int(image.height * scale)))
        canvas = Image.new("RGB", (image.width, image.height + 12), (255, 255, 255))
        canvas.paste(image, (0, 12))
        ImageDraw.Draw(canvas).text((2, 0), label, fill=(0, 0, 0))
        thumbs.append(canvas)
    if not thumbs:
        raise ValueError("no images")
    width, height = thumbs[0].size
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), (200, 200, 200))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * width, (index // columns) * height))
    sheet.save(out)
    return out


NDS_TAP_PULSE = -1  # sentinel returned by nds_pulse(): touch the screen instead of pressing a key
# Deterministic, biased pseudo-random NDS schedule: (key, weight). A random-but-biased masher escapes
# every looping menu of the Gen 4/5 tutorials (yes/no questions, 3-option "anything else?" menus,
# naming keyboards) with high probability, without any game-specific rule.
NDS_SCHEDULE = ((GBA_A, 50), (GBA_B, 15), (GBA_DOWN, 12), (GBA_UP, 5), (GBA_START, 3), (NDS_TAP_PULSE, 15))


def nds_pulse(pulse: int) -> int:
    rng = np.random.default_rng(pulse * 7919 + 13)
    keys = [key for key, _ in NDS_SCHEDULE]
    weights = np.array([weight for _, weight in NDS_SCHEDULE], dtype=np.float64)
    return int(keys[int(rng.choice(len(keys), p=weights / weights.sum()))])


def intro_keys(frame: int, period: int = 24, nds: bool = False) -> int:
    """Game-agnostic intro schedule (6-frame pulses every ``period`` frames).

    GB/GBC/GBA: A pulses; every 10th pulse is START; every 23rd is B.
    NDS: the biased pseudo-random ``nds_pulse`` schedule (taps are handled by ``run_intro``).
    """
    phase, pulse = frame % period, frame // period
    if phase >= 6:
        return 0
    if nds:
        key = nds_pulse(pulse)
        return 0 if key == NDS_TAP_PULSE else key
    if pulse % 10 == 9:
        return GBA_START
    if pulse % 23 == 22:
        return GBA_B
    return GBA_A


# Upper and lower middle of the bottom screen: where Gen 4/5 intros put their Yes/No (or continue)
# buttons. Taps alternate between them, so any binary question that loops on one answer ("do you
# want an explanation?", "anything else?") is eventually answered the other way.
NDS_TAPS = ((120, 80), (120, 121))


def run_intro(spec: GameSpec, frames: int, every: int, out: Path, period: int = 24) -> Path:
    bridge = make_bridge(spec)
    out.mkdir(parents=True, exist_ok=True)
    images: list[tuple[str, Path]] = []
    nds = isinstance(bridge, DeSmuMEBridge)
    try:
        for frame in range(frames):
            phase, pulse = frame % period, frame // period
            tapping = nds and nds_pulse(pulse) == NDS_TAP_PULSE
            bridge.set_hardware_keys(0 if tapping else intro_keys(frame, period, nds=nds))
            if tapping:  # the intro driver may touch the screen; the agent never does
                assert isinstance(bridge, DeSmuMEBridge)
                if phase == 0:
                    bridge.touch(*NDS_TAPS[(pulse // 7) % len(NDS_TAPS)])
                elif phase == 10:
                    bridge.release_touch()
            bridge.step_frame()
            if (frame + 1) % every == 0:
                tag = f"f{frame + 1:06d}"
                save_png(bridge.framebuffer(), out / f"{tag}.png")
                (out / f"{tag}.state").write_bytes(bridge.save_state())
                images.append((tag, out / f"{tag}.png"))
    finally:
        bridge.close()
    return contact_sheet(images, out / "sheet.png")


def pick_state(spec: GameSpec, state: Path) -> Path:
    target = default_state_path(spec.key)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(state, target)
    provenance = {"game": spec.key, "source_state": str(state), "procedure": "scripts.ares_u_gal intro (A/START/B "
                  "schedule from power-on), chosen by visual inspection of the contact sheet: player in control "
                  "in the overworld after the intro", "rom": spec.rom_name}
    target.with_suffix(".json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return target


def hold(bridge: Bridge, keys: int, frames: int) -> None:
    bridge.set_hardware_keys(keys)
    for _ in range(frames):
        bridge.step_frame()
    bridge.set_hardware_keys(0)
    for _ in range(12):
        bridge.step_frame()


DIRECTIONS = {"up": GBA_UP, "down": GBA_DOWN, "left": GBA_LEFT, "right": GBA_RIGHT}


def verify(spec: GameSpec, tile_frames: int = 16) -> dict[str, Any]:
    """Walk 3 tiles in each direction from the start state and report the adapter's readings."""
    state = default_state_path(spec.key).read_bytes()
    bridge = make_bridge(spec)
    report: dict[str, Any] = {"game": spec.key, "walks": {}}
    try:
        for name, keys in DIRECTIONS.items():
            bridge.load_state(state)
            for _ in range(30):
                bridge.step_frame()
            start = bridge.get_player_coords_and_map()
            readings = [start]
            for _ in range(3):
                hold(bridge, keys, tile_frames)
                readings.append(bridge.get_player_coords_and_map())
            report["walks"][name] = readings
    finally:
        bridge.close()
    return report


def _snapshot(read: Callable[[int, int], bytes], start: int, size: int, width: int) -> NDArray[np.int32]:
    data = np.frombuffer(read(start, size), dtype=np.uint8 if width == 1 else np.uint16).astype(np.int32)
    return data


def discover(spec: GameSpec, region: tuple[int, int], tile_frames: int = 16, chunk: int = 0x10000,
             width: int = 2) -> dict[str, Any]:
    """Find bytes (``width`` 1) or halfwords (2) that track the player's x and y by walking."""
    state = default_state_path(spec.key).read_bytes()
    bridge = make_bridge(spec)
    start, size = region

    def read_all() -> NDArray[np.int32]:
        parts = [_snapshot(bridge.read, address, min(chunk, start + size - address), width)
                 for address in range(start, start + size, chunk)]
        return np.concatenate(parts)

    candidates: dict[str, Any] = {}
    try:
        for axis, (plus, minus) in {"x": (GBA_RIGHT, GBA_LEFT), "y": (GBA_DOWN, GBA_UP)}.items():
            best: NDArray[np.bool_] | None = None
            for keys, sign in ((plus, 1), (minus, -1)):
                bridge.load_state(state)
                for _ in range(30):
                    bridge.step_frame()
                before = read_all()
                hold(bridge, keys, tile_frames)
                after = read_all()
                moved = (after - before) == sign
                best = moved if best is None else best | moved
            assert best is not None
            # Must stay put when walking along the other axis.
            bridge.load_state(state)
            for _ in range(30):
                bridge.step_frame()
            before = read_all()
            hold(bridge, GBA_DOWN if axis == "x" else GBA_RIGHT, tile_frames)
            after = read_all()
            steady = after == before
            hits = np.flatnonzero(best & steady)
            candidates[axis] = [start + width * int(index) for index in hits[:64]]
    finally:
        bridge.close()
    return {"game": spec.key, "region": [hex(start), hex(size)], "candidates": {k: [hex(a) for a in v]
                                                                                for k, v in candidates.items()}}


def save_coords(spec: GameSpec, coords: CoordSpec) -> Path:
    GAL_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = GAL_ARTIFACTS / f"{spec.key}_coords.json"
    path.write_text(json.dumps(coords.to_json(), indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    intro = sub.add_parser("intro")
    intro.add_argument("--game", required=True)
    intro.add_argument("--frames", type=int, default=30000)
    intro.add_argument("--every", type=int, default=1500)
    pick = sub.add_parser("pick")
    pick.add_argument("--game", required=True)
    pick.add_argument("--state", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--game", required=True)
    find = sub.add_parser("discover")
    find.add_argument("--game", required=True)
    find.add_argument("--start", type=lambda value: int(value, 0), default=0x02000000)
    find.add_argument("--size", type=lambda value: int(value, 0), default=0x400000)
    find.add_argument("--width", type=int, choices=(1, 2), default=2)
    adopt = sub.add_parser("adopt", help="save a verified adapter: --x ADDR --y ADDR --map ADDR [--width 2]")
    adopt.add_argument("--game", required=True)
    adopt.add_argument("--x", type=lambda value: int(value, 0), required=True)
    adopt.add_argument("--y", type=lambda value: int(value, 0), required=True)
    adopt.add_argument("--map", type=lambda value: int(value, 0), required=True)
    adopt.add_argument("--width", type=int, default=2)
    adopt.add_argument("--source", default="scripts.ares_u_gal discover + verify")
    args = parser.parse_args(argv)
    if args.command == "list":
        for spec in GAMES.values():
            print(f"{spec.key:10s} gen{spec.generation} {spec.console:4s} rom={'ok' if spec.rom.exists() else 'MISSING'} "
                  f"adapter={'yes' if spec.coord_spec() or spec.key in NATIVE_GAMES else 'pixel-only'}  {spec.title}")
        return
    spec = game(args.game)
    if args.command == "intro":
        print(run_intro(spec, args.frames, args.every, GAL_ARTIFACTS / "intro" / spec.key))
    elif args.command == "pick":
        print(pick_state(spec, args.state))
    elif args.command == "verify":
        print(json.dumps(verify(spec), indent=1))
    elif args.command == "discover":
        print(json.dumps(discover(spec, (args.start, args.size), width=args.width), indent=1))
    elif args.command == "adopt":
        coords = CoordSpec(map_num=Field(args.map, args.width), x=Field(args.x, args.width), y=Field(args.y, args.width),
                           source=args.source)
        print(save_coords(spec, coords))


if __name__ == "__main__":
    main()
