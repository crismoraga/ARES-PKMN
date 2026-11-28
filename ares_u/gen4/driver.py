"""Assisted story driver for the Gen 4 games: a tiny command language replayed on a savestate.

Story progress (getting the starter, the first battles) is produced by *recorded* command strings: an
operator (here: the engineer, looking at the saved screenshots) chooses each command, the driver executes it
deterministically on the emulator, saves the resulting state + screenshot, and appends the step to a JSON
log. Replaying the log from the same start state reproduces the same states (the emulators are
deterministic given the inputs), so every manual decision is documented and reproducible.

Commands (space separated):
  w:<dir>:<n>     walk n tiles (stops when blocked or when the map changes)
  g:<x>,<z>[:zx]  greedy walk to tile (x, z) on the current map; presses B when stuck (closes text)
  a:<n> b:<n>     press A / B n times (6 frames held, 24 frames apart)
  st sel          press START / SELECT once
  k:<keys>:<hold>:<settle>  press a key combination ("up+b") for <hold> frames, then idle <settle>
  turn:<dir>      2-frame tap (turns in place)
  t:<x>,<y>       tap the touch screen (bottom-screen pixels)
  f:<n>           idle n frames
  ma:<n>          mash: n x (A, wait 24)
  mb:<n>          mash: n x (B, wait 24)

    python -m ares_u.gen4.driver --game platino --state S --out NAME --cmds "w:up:6 w:right:4 g:8,4"
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from ares_u.gal.games import GAMES
from ares_u.gen4.layout import layout

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "gen4" / "story"
KEY = {"a": 1, "b": 2, "select": 4, "start": 8, "right": 16, "left": 32, "up": 64, "down": 128, "r": 256, "l": 512}
STEP = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def make_nds_bridge(game: str, backend: str | None = None) -> Any:
    """DeSmuME (default, compatible with the .dst start states) or melonDS (``backend="melonds"``)."""
    choice = (backend or os.environ.get("ARES_NDS_BACKEND", "desmume")).lower()
    if choice == "melonds":
        from ares_u.gal.melonds import MelonDSBridge

        return MelonDSBridge(GAMES[game])
    from ares_u.gal.backends import DeSmuMEBridge

    return DeSmuMEBridge(GAMES[game])


def read_bytes(bridge: Any, address: int, size: int) -> bytes:
    """Fast-ish raw read for either NDS bridge."""
    emu = getattr(bridge, "_emu", None)
    if emu is not None:  # DeSmuME: slice read (one FFI call per byte, fine for small blocks)
        return bytes(emu.memory.unsigned[address:address + size])
    return bytes(bridge.read(address, size))


def read_u32(bridge: Any, address: int) -> int:
    emu = getattr(bridge, "_emu", None)
    if emu is not None:
        return int(emu.memory.unsigned.read_long(address))
    return int.from_bytes(bridge.read(address, 4), "little")


class Driver:
    def __init__(self, bridge: Any, game: str) -> None:
        self.bridge, self.game = bridge, game
        self.layout = layout(game)
        self.trace: list[Any] = []

    # ------------------------------------------------------------------ primitives
    def run(self, frames: int, keys: int = 0) -> None:
        self.bridge.set_hardware_keys(keys)
        for _ in range(frames):
            self.bridge.step_frame()

    def press(self, keys: str, hold: int = 6, settle: int = 24) -> None:
        mask = 0
        for name in keys.split("+"):
            mask |= KEY[name]
        self.run(hold, mask)
        self.run(settle, 0)

    def tap(self, x: int, y: int) -> None:
        self.bridge.touch(x, y)
        self.run(8)
        self.bridge.release_touch()
        self.run(24)

    def location(self) -> list[int]:
        """[map_id, warp_id, x, z, dir] from the save-block Location (signed)."""
        base = read_u32(self.bridge, self.layout.base_pointer) + self.layout.general_offset + self.layout.location
        raw = read_bytes(self.bridge, base, 20)
        return [int(v) for v in np.frombuffer(raw, dtype="<i4")]

    # ------------------------------------------------------------------ movement
    def walk(self, direction: str, tiles: int) -> list[int]:
        start = self.location()
        for _ in range(tiles):
            before = self.location()
            self.press(direction, 16, 16)
            after = self.location()
            if after[0] != start[0]:
                self.run(60)
                break
            if after[2:4] == before[2:4]:
                self.press(direction, 16, 16)  # first press may only turn the player
                if self.location()[2:4] == before[2:4]:
                    break
        return self.location()

    def goto(self, tx: int, tz: int, order: str = "xz", max_steps: int = 80) -> list[int]:
        start_map = self.location()[0]
        stuck = 0
        for _ in range(max_steps):
            m, _, x, z, _ = self.location()
            if m != start_map or (x, z) == (tx, tz):
                break
            options = []
            for axis in order:
                if axis == "x" and x != tx:
                    options.append("right" if tx > x else "left")
                if axis == "z" and z != tz:
                    options.append("down" if tz > z else "up")
            direction = options[stuck % len(options)]
            before = self.location()
            self.press(direction, 16, 16)
            if self.location()[2:4] == before[2:4] and self.location()[0] == before[0]:
                stuck += 1
                self.press("b", 4, 20)
                if stuck > 12:
                    break
            else:
                stuck = 0
        if self.location()[0] != start_map:
            self.run(60)
        return self.location()

    # ------------------------------------------------------------------ commands
    def execute(self, command: str) -> None:
        parts = command.split(":")
        op = parts[0]
        if op == "w":
            self.walk(parts[1], int(parts[2]))
        elif op == "g":
            x, z = (int(v) for v in parts[1].split(","))
            self.goto(x, z, parts[2] if len(parts) > 2 else "xz")
        elif op in ("a", "b"):
            for _ in range(int(parts[1]) if len(parts) > 1 else 1):
                self.press(op)
        elif op in ("ma", "mb"):
            for _ in range(int(parts[1])):
                self.press(op[1], 6, 24)
        elif op == "st":
            self.press("start")
        elif op == "sel":
            self.press("select")
        elif op == "k":
            self.press(parts[1], int(parts[2]) if len(parts) > 2 else 6, int(parts[3]) if len(parts) > 3 else 24)
        elif op == "turn":
            self.press(parts[1], 2, 20)
        elif op == "t":
            x, y = (int(v) for v in parts[1].split(","))
            self.tap(x, y)
        elif op == "f":
            self.run(int(parts[1]))
        else:
            raise ValueError(f"unknown command {command!r}")
        self.trace.append((command, self.location()))

    def party_summary(self) -> list[tuple[int, int, int, int]]:
        """[(species, level, hp, max_hp)] of the party read through the save-block pointer."""
        from ares_u.gen4.pkm import PARTY_SIZE, decrypt, parse

        general = read_u32(self.bridge, self.layout.base_pointer) + self.layout.general_offset
        count = read_u32(self.bridge, general + self.layout.party_count)
        out = []
        for slot in range(min(count, 6)):
            plain = decrypt(read_bytes(self.bridge, general + self.layout.party + slot * PARTY_SIZE, PARTY_SIZE))
            if plain is not None:
                mon = parse(plain)
                out.append((mon.species, mon.level, mon.hp, mon.max_hp))
        return out

    def screens(self) -> np.ndarray:
        if hasattr(self.bridge, "screens"):
            shot: np.ndarray = self.bridge.screens()
            return shot
        raw = np.ctypeslib.as_array(self.bridge._emu.display_buffer(), shape=(192 * 2 * 256,))
        v = raw.astype(np.uint32).reshape(384, 256)
        return np.stack([(v & 0x1F) << 3, ((v >> 5) & 0x1F) << 3, ((v >> 10) & 0x1F) << 3], axis=-1).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--game", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--out", required=True, help="name of the resulting state (artifacts/gen4/story/<game>/)")
    parser.add_argument("--cmds", required=True)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    from PIL import Image

    folder = OUT_DIR / args.game
    folder.mkdir(parents=True, exist_ok=True)
    state = args.state if args.state.exists() else folder / f"{args.state}.state"
    bridge = make_nds_bridge(args.game, args.backend)
    bridge.load_state(state.read_bytes())
    driver = Driver(bridge, args.game)
    started = time.time()
    for command in args.cmds.split():
        driver.execute(command)
    shots = driver.screens()
    Image.fromarray(shots[:192]).save(folder / f"{args.out}.png")
    Image.fromarray(shots).save(folder / f"{args.out}_both.png")
    (folder / f"{args.out}.state").write_bytes(bridge.save_state())
    log_path = folder / "story_log.json"
    log: list[Any] = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    entry = {"from": str(state.relative_to(ROOT) if state.is_relative_to(ROOT) else state), "to": args.out,
             "cmds": args.cmds, "note": args.note, "loc": driver.location(),
             "trace": [[c, loc] for c, loc in driver.trace], "seconds": round(time.time() - started, 1)}
    log = [item for item in log if item.get("to") != args.out] + [entry]
    log_path.write_text(json.dumps(log, indent=1), encoding="utf-8")
    print("@@RESULT", json.dumps({"loc": driver.location(), "party": driver.party_summary(),
                                  "trace": [[c, loc[:4]] for c, loc in driver.trace][-6:]}), flush=True)


if __name__ == "__main__":
    main()
