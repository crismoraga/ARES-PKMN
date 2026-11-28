"""Gen 4 (NDS) RAM discovery: find the player's position/map words by walking and diffing main RAM.

The NDS main RAM (4 MiB at 0x02000000) is snapshotted after every step of a scripted walk
(right x3, left x3, down x2, up x2). A word is an x candidate when it only changes on horizontal steps,
by the same amount, upward for RIGHT and downward for LEFT; y likewise for vertical steps. Gen 4 keeps
positions both as tile integers and as fx32 world coordinates (tile << 16 or << 12 with 16 px tiles), so
several step sizes are accepted. Heap structures move between sessions, so candidates are re-validated
from a second savestate and reported with their step size.

    python -m ares_u.gen4.probe --game platino --state artifacts/ares_u/nav/states/platino_start.state
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ares_u.gal.backends import make_bridge
from ares_u.gal.games import GAMES

ROOT = Path(__file__).resolve().parents[2]
RAM_START, RAM_SIZE = 0x02000000, 0x00400000
KEY = {"a": 1 << 0, "b": 1 << 1, "select": 1 << 2, "start": 1 << 3, "right": 1 << 4, "left": 1 << 5,
       "up": 1 << 6, "down": 1 << 7}
SEQUENCE = ("right", "right", "right", "left", "left", "left", "up", "up", "up", "down", "down", "down")
STEP_SIZES = (1, 16, 1 << 12, 1 << 16, 16 << 12, 1 << 20)


class Walker:
    """Drives an NDS bridge: DeSmuME (``.dst`` states) or melonDS (``backend="melonds"``, its own states)."""

    def __init__(self, game: str, state: Path | None, backend: str = "desmume") -> None:
        self.bridge: Any
        if backend == "melonds":
            from ares_u.gal.melonds import MelonDSBridge

            self.bridge = MelonDSBridge(GAMES[game])
        else:
            self.bridge = make_bridge(GAMES[game])
        if state is not None:
            self.bridge.load_state(state.read_bytes())

    def frames(self, count: int, keys: int = 0) -> None:
        self.bridge.set_hardware_keys(keys)
        for _ in range(count):
            self.bridge.step_frame()

    def step(self, direction: str, hold: int = 12, settle: int = 20) -> None:
        self.frames(hold, KEY[direction])
        self.frames(settle, 0)

    def snapshot(self) -> bytes:
        if hasattr(self.bridge, "ram"):  # melonDS: direct pointer copy (~2 ms)
            return bytes(self.bridge.ram())
        return bytes(self.bridge._emu.memory.unsigned[RAM_START:RAM_START + RAM_SIZE])


def words(blob: bytes, width: int) -> NDArray[np.int64]:
    dtype = np.dtype("<u2") if width == 2 else np.dtype("<u4")
    usable = len(blob) - len(blob) % width
    return np.frombuffer(blob[:usable], dtype=dtype).astype(np.int64)


def candidates(snaps: list[bytes], width: int, axis: str) -> list[dict[str, Any]]:
    """Words whose step-by-step changes follow the walk on ``axis`` ("x" or "y") with one step size."""
    series = np.stack([words(s, width) for s in snaps])  # (len(SEQUENCE)+1, n)
    deltas = np.diff(series, axis=0)
    plus, minus = ("right", "left") if axis == "x" else ("down", "up")
    other = ("down", "up") if axis == "x" else ("right", "left")
    found: list[dict[str, Any]] = []
    for size in STEP_SIZES:
        ok = np.ones(series.shape[1], dtype=bool)
        moved_plus = np.zeros(series.shape[1], dtype=np.int64)
        moved_minus = np.zeros(series.shape[1], dtype=np.int64)
        for index, direction in enumerate(SEQUENCE):
            d = deltas[index]
            if direction == plus:
                ok &= (d == 0) | (d == size)
                moved_plus += d == size
            elif direction == minus:
                ok &= (d == 0) | (d == -size)
                moved_minus += d == -size
            elif direction in other:
                ok &= d == 0
        ok &= (moved_plus >= 1) & (moved_minus >= 1) & (moved_plus + moved_minus >= 3)
        for offset in np.flatnonzero(ok)[:64]:
            address = RAM_START + int(offset) * width
            found.append({"address": hex(address), "width": width, "step": size,
                          "values": series[:, offset].tolist()})
    return found


def probe(game: str, state: Path) -> dict[str, Any]:
    walker = Walker(game, state)
    walker.frames(30)
    snaps = [walker.snapshot()]
    for direction in SEQUENCE:
        walker.step(direction)
        snaps.append(walker.snapshot())
    result: dict[str, Any] = {"game": game, "state": str(state), "sequence": list(SEQUENCE)}
    for axis in ("x", "y"):
        result[axis] = candidates(snaps, 2, axis) + candidates(snaps, 4, axis)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--game", default="platino")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "gen4")
    args = parser.parse_args()
    result = probe(args.game, args.state)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.game}_probe_{args.state.stem}.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for axis in ("x", "y"):
        print(axis, len(result[axis]), [(c["address"], c["width"], c["step"], c["values"]) for c in result[axis][:12]])
    print(path)


if __name__ == "__main__":
    main()
