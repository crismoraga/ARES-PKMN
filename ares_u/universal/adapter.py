"""``GameAdapter``: the only place where a game's specifics live.

The base class owns the emulator (``ares_u.gal`` bridge) and the input primitives (press a button,
walk one cell with RAM confirmation, wait); subclasses implement perception (``snapshot``,
``map_info``, ``npcs``) for one generation. ``GBScreenParser`` reads the 20x18 background tile map
that Gen 1 and Gen 2 share (same frame/cursor tile conventions) and turns it into dialog lines and
menus, so the dialog logic of the agent is generic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ares_u.gal.backends import (GBA_A, GBA_B, GBA_DOWN, GBA_LEFT, GBA_RIGHT, GBA_SELECT, GBA_START, GBA_UP,
                                 MGBAMultiBridge)
from ares_u.gal.games import GameSpec

from .model import DELTAS, Direction, MapInfo, MapKey, MenuInfo, MoveInfo, Npc, ScreenInfo, Snapshot

BUTTONS: dict[str, int] = {"A": GBA_A, "B": GBA_B, "START": GBA_START, "SELECT": GBA_SELECT, "up": GBA_UP,
                           "down": GBA_DOWN, "left": GBA_LEFT, "right": GBA_RIGHT}

# Tile ids shared by the Gen 1 and Gen 2 fonts.
TILE_SPACE = 0x7F
FRAME_TL, FRAME_H, FRAME_TR, FRAME_V, FRAME_BL, FRAME_BR = 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E
CURSOR, CURSOR_IDLE, PROMPT_ARROW = 0xED, 0xEC, 0xEE
SCREEN_W, SCREEN_H = 20, 18


def base_charmap() -> dict[int, str]:
    """Western GB charmap (pokered/pokecrystal); adapters add their localized letters."""
    table: dict[int, str] = {TILE_SPACE: " "}
    for index in range(26):
        table[0x80 + index] = chr(ord("A") + index)
        table[0xA0 + index] = chr(ord("a") + index)
    for index in range(10):
        table[0xF6 + index] = str(index)
    table.update({0x9A: "(", 0x9B: ")", 0x9C: ":", 0x9D: ";", 0x9E: "[", 0x9F: "]", 0xE0: "'", 0xE1: "PK",
                  0xE2: "MN", 0xE3: "-", 0xE6: "?", 0xE7: "!", 0xE8: ".", 0xED: ">", 0xEF: "♂", 0xF0: "$",
                  0xF1: "x", 0xF2: ".", 0xF3: "/", 0xF4: ",", 0xF5: "♀"})
    return table


@dataclass
class GBScreenParser:
    """Dialog box, prompt arrow and active menu from a 20x18 tile map (Gen 1/Gen 2 conventions)."""

    charmap: dict[int, str]
    unknown: dict[int, int] = field(default_factory=dict)

    def decode(self, tiles: NDArray[np.uint8] | bytes | list[int]) -> str:
        out = []
        for tile in tiles:
            value = int(tile)
            char = self.charmap.get(value)
            if char is None:
                if value in (FRAME_V, FRAME_H):
                    char = " "
                else:
                    self.unknown[value] = self.unknown.get(value, 0) + 1
                    char = "·"
            out.append(char)
        return "".join(out)

    @staticmethod
    def has_textbox(tiles: NDArray[np.uint8]) -> bool:
        return bool(tiles[12, 0] == FRAME_TL and tiles[12, 19] == FRAME_TR and tiles[17, 0] == FRAME_BL
                    and tiles[17, 19] == FRAME_BR)

    def dialog_lines(self, tiles: NDArray[np.uint8]) -> tuple[str, ...]:
        if not self.has_textbox(tiles):
            return ()
        lines = (self.decode(tiles[row, 1:19]).strip() for row in (13, 14, 15, 16))
        return tuple(line for line in lines if line and set(line) != {"·"})

    def find_cursor(self, tiles: NDArray[np.uint8]) -> tuple[int, int] | None:
        hits = np.argwhere(tiles == CURSOR)
        if len(hits) != 1:
            return None
        row, col = (int(v) for v in hits[0])
        return row, col

    def enclosing_box(self, tiles: NDArray[np.uint8], row: int, col: int) -> tuple[int, int, int, int] | None:
        """(top, left, bottom, right) of the frame around (row, col), or None."""
        left = col
        while left >= 0 and tiles[row, left] != FRAME_V:
            left -= 1
        right = col
        while right < SCREEN_W and tiles[row, right] != FRAME_V:
            right += 1
        if left < 0 or right >= SCREEN_W:
            return None
        top = row
        while top >= 0 and tiles[top, left] not in (FRAME_TL,):
            top -= 1
        bottom = row
        while bottom < SCREEN_H and tiles[bottom, left] not in (FRAME_BL,):
            bottom += 1
        if top < 0 or bottom >= SCREEN_H:
            return None
        return top, left, bottom, right

    @staticmethod
    def aligned_list(tiles: NDArray[np.uint8], row: int, col: int) -> tuple[int, int, int, int] | None:
        """Rows above/below the cursor whose label starts right after the cursor column."""
        def is_item(r: int) -> bool:
            return (0 <= r < SCREEN_H and int(tiles[r, col]) in (TILE_SPACE, CURSOR, CURSOR_IDLE)
                    and col + 1 < SCREEN_W and int(tiles[r, col + 1]) != TILE_SPACE)

        top = bottom = row
        for step in (-1, 1):
            r = row
            while True:
                if is_item(r + step):
                    r += step
                elif is_item(r + 2 * step) and int(tiles[r + step, col + 1]) == TILE_SPACE:
                    r += 2 * step
                else:
                    break
            if step < 0:
                top = r
            else:
                bottom = r
        if top == bottom:
            return None
        return top - 1, col, bottom + 1, SCREEN_W

    def menu(self, tiles: NDArray[np.uint8]) -> MenuInfo | None:
        """The active menu (one ``>`` cursor on screen): its options in reading order."""
        found = self.find_cursor(tiles)
        if found is None:
            return None
        row, col = found
        box = self.enclosing_box(tiles, row, col)
        if box is None:  # frame drawn with other tiles (Gen 1 battle move list): column-aligned list
            box = self.aligned_list(tiles, row, col)
        top, left, bottom, right = box if box is not None else (row - 1, -1, row + 1, SCREEN_W)
        options = self.segments(tiles, top + 1, bottom, left + 1, right)
        # 2x2 menus (battle: 'LUCHA PKMN' one blank apart): split where another row starts a label.
        starts = {c for _, c, _ in options} | {col + 1}
        options = self.segments(tiles, top + 1, bottom, left + 1, right, split_at=starts)
        if not options:
            return None
        selected = next((i for i, (r, c, _) in enumerate(options) if r == row and c == col + 1), -1)
        if selected < 0:  # cursor drawn on a row whose label could not be segmented
            selected = min(range(len(options)), key=lambda i: abs(options[i][0] - row) * 40
                           + abs(options[i][1] - col))
        return MenuInfo(cursor_row=row, cursor_col=col, options=tuple(label for _, _, label in options),
                        selected=selected, positions=tuple((r, c) for r, c, _ in options))

    def segments(self, tiles: NDArray[np.uint8], row0: int, row1: int, col0: int, col1: int,
                 split_at: set[int] | None = None) -> list[tuple[int, int, str]]:
        """Labels inside a box: runs of text separated by >= 2 blanks or by a cursor tile."""
        found: list[tuple[int, int, str]] = []
        for r in range(max(0, row0), min(SCREEN_H, row1)):
            c = max(0, col0)
            limit = min(SCREEN_W, col1)
            while c < limit:
                tile = int(tiles[r, c])
                if tile in (TILE_SPACE, CURSOR, CURSOR_IDLE, FRAME_V, FRAME_H):
                    c += 1
                    continue
                start = c
                while c < limit:
                    tile = int(tiles[r, c])
                    if tile in (CURSOR, CURSOR_IDLE, FRAME_V):
                        break
                    if tile == TILE_SPACE and (c + 1 >= limit or int(tiles[r, c + 1]) == TILE_SPACE):
                        break
                    if tile == TILE_SPACE and split_at and c + 1 in split_at and c > start:
                        break
                    c += 1
                label = self.decode(tiles[r, start:c]).strip()
                if label and label != "·":
                    found.append((r, start, label))
        return found

    def screen(self, tiles: NDArray[np.uint8]) -> ScreenInfo:
        textbox = self.has_textbox(tiles)
        return ScreenInfo(lines=self.dialog_lines(tiles), textbox=textbox,
                          prompt=bool(tiles[16, 18] == PROMPT_ARROW), menu=self.menu(tiles),
                          rows=tuple(self.decode(tiles[r]) for r in range(SCREEN_H)))


class GameAdapter(ABC):
    """Emulator + perception of one game. Subclasses: ``Gen1Adapter``, ``Gen2Adapter``."""

    key: str = ""
    generation: int = 0
    language: str = "es"

    def __init__(self, spec: GameSpec, bridge: Any | None = None) -> None:
        self.spec = spec
        self.bridge: Any = bridge if bridge is not None else MGBAMultiBridge(spec)
        self._held = 0
        self.rom: bytes = Path(spec.rom).read_bytes() if Path(spec.rom).exists() else b""

    # ------------------------------------------------------------------ emulator
    @property
    def frame_count(self) -> int:
        return int(self.bridge.frame_count)

    def framebuffer(self) -> NDArray[np.uint32]:
        frame: NDArray[np.uint32] = self.bridge.framebuffer()
        return frame

    def save_state(self) -> bytes:
        state: bytes = self.bridge.save_state()
        return state

    def load_state(self, state: bytes) -> None:
        self.bridge.load_state(state)
        self._held = -1
        self.set_keys(0)
        self.invalidate()

    def set_keys(self, mask: int) -> None:
        if mask != self._held:
            self.bridge.set_hardware_keys(mask)
            self._held = mask

    def wait(self, frames: int) -> None:
        self.set_keys(0)
        for _ in range(frames):
            self.bridge.step_frame()

    def press(self, button: str, hold: int = 4, release: int = 8) -> None:
        self.set_keys(BUTTONS[button])
        for _ in range(hold):
            self.bridge.step_frame()
        self.set_keys(0)
        for _ in range(release):
            self.bridge.step_frame()

    def read(self, address: int, size: int, bank: int = -1) -> bytes:
        data: bytes = self.bridge.read(address, size, bank)
        return data

    def u8(self, address: int, bank: int = -1) -> int:
        return self.read(address, 1, bank)[0]

    def close(self) -> None:
        self.bridge.close()

    # ------------------------------------------------------------------ movement
    def walk(self, direction: Direction, max_frames: int = 48) -> str:
        """Hold ``direction`` until the player moves one cell, the map changes or something interrupts.

        Returns "moved", "warped", "blocked", "turned", "text", "battle" or "timeout". The RAM is the
        ground truth for the outcome (position, map, battle flag, dialog box).
        """
        start = self.position()
        self.set_keys(BUTTONS[direction])
        outcome = "blocked"
        frames = 0
        start_facing = self.facing()
        while frames < max_frames:
            self.bridge.step_frame()
            frames += 1
            if frames % 2:
                continue
            if self.in_battle():
                outcome = "battle"
                break
            pos = self.position()
            if pos[0] != start[0]:
                outcome = "warped"
                break
            if pos != start:
                outcome = "moved"
                break
            if frames >= 8 and self.dialog_open():
                outcome = "text"
                break
        self.set_keys(0)
        if outcome == "blocked" and self.facing() != start_facing:
            outcome = "turned"
        # Let the step animation finish so the next decision starts from a settled state.
        settle = 0
        while settle < 40 and self.is_walking():
            self.bridge.step_frame()
            settle += 1
        if outcome == "warped":
            self.wait(4)
            self.invalidate()
        return outcome

    def face_and_press(self, direction: Direction) -> None:
        """Turn toward ``direction`` (tap) then press A."""
        if self.facing() != direction:
            self.set_keys(BUTTONS[direction])
            for _ in range(3):
                self.bridge.step_frame()
            self.set_keys(0)
            self.wait(6)
            if self.facing() != direction:  # some games need a longer tap to turn
                self.set_keys(BUTTONS[direction])
                for _ in range(6):
                    self.bridge.step_frame()
                self.set_keys(0)
                self.wait(10)
        self.press("A", 4, 12)

    # ------------------------------------------------------------------ perception (abstract)
    @abstractmethod
    def position(self) -> tuple[MapKey, int, int]: ...

    @abstractmethod
    def facing(self) -> Direction: ...

    @abstractmethod
    def is_walking(self) -> bool: ...

    @abstractmethod
    def in_battle(self) -> bool: ...

    @abstractmethod
    def dialog_open(self) -> bool: ...

    @abstractmethod
    def snapshot(self) -> Snapshot: ...

    @abstractmethod
    def map_info(self) -> MapInfo: ...

    @abstractmethod
    def npcs(self) -> list[Npc]: ...

    @abstractmethod
    def screen(self) -> ScreenInfo: ...

    @abstractmethod
    def menu_index(self) -> int: ...

    @abstractmethod
    def map_name(self, key: MapKey) -> str: ...

    def invalidate(self) -> None:
        """Drop cached map data (called after a warp or a state load)."""

    # ------------------------------------------------------------------ localisation / measurement hooks
    def item_name(self, item_id: int) -> str:
        return f"objeto {item_id}"

    def type_words(self) -> dict[str, str]:
        """Localized type names in dialog text -> English type name."""
        return {}

    def effectiveness_phrases(self) -> list[tuple[str, float]]:
        """Folded (lowercase, no accents) battle phrases -> damage multiplier, most specific first."""
        return []

    def find_move_in_text(self, text: str) -> MoveInfo | None:
        return None

    def milestones(self, snap: Snapshot, visited: set[MapKey], flags: set[str]) -> list[str]:
        """Story milestones reached (measurement only; the agent never plans with them)."""
        return []

    def heal_signature(self) -> dict[str, Any]:
        """Game-agnostic description of the current place, stored when a heal happens."""
        info = self.map_info()
        return {"map": list(info.key), "tileset": info.tileset, "outdoor": info.outdoor, "width": info.width,
                "height": info.height}


def neighbours(x: int, y: int) -> list[tuple[Direction, int, int]]:
    return [(direction, x + dx, y + dy) for direction, (dx, dy) in DELTAS.items()]
