"""Game-agnostic state model shared by the adapters and the universal agent.

Adapters translate a game's RAM into these plain dataclasses; the agent reasons only on them.
Coordinates are in player steps ("cells"); ``MapKey`` is ``(group, number)`` (group 0 for Gen 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

MapKey = tuple[int, int]
Direction = Literal["up", "down", "left", "right"]
DIRECTIONS: tuple[Direction, ...] = ("up", "down", "left", "right")
DELTAS: dict[Direction, tuple[int, int]] = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
OPPOSITE: dict[Direction, Direction] = {"up": "down", "down": "up", "left": "right", "right": "left"}

# Cell classes of the walkability grid.
UNKNOWN, FLOOR, WALL, WATER, GRASS, LEDGE_DOWN, LEDGE_LEFT, LEDGE_RIGHT, LEDGE_UP, WARP_CELL, COUNTER = range(11)
WALKABLE_CLASSES = frozenset({FLOOR, GRASS, WARP_CELL})
LEDGE_DIRECTION: dict[int, Direction] = {LEDGE_DOWN: "down", LEDGE_LEFT: "left", LEDGE_RIGHT: "right",
                                         LEDGE_UP: "up"}

Mode = Literal["overworld", "text", "menu", "battle", "naming", "busy"]


@dataclass(frozen=True)
class MoveInfo:
    move_id: int            # equals the Gen 3 move id for every move of Gen 1-2
    name: str
    type: str               # lowercase English type name ("fire")
    power: int
    accuracy: int           # percent
    pp: int
    max_pp: int
    effect: int = 0

    @property
    def category(self) -> str:
        if self.power <= 0:
            return "status"
        return "special" if self.type in SPECIAL_TYPES else "physical"


SPECIAL_TYPES = frozenset({"fire", "water", "grass", "electric", "ice", "psychic", "dragon", "dark"})


@dataclass(frozen=True)
class MonInfo:
    species_id: int         # the game's internal species index
    national_dex: int
    name: str
    level: int
    hp: int
    max_hp: int
    types: tuple[str, ...]
    status: str             # "" | "par" | "slp" | "psn" | "brn" | "frz" | "tox"
    moves: tuple[MoveInfo, ...]
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def fainted(self) -> bool:
        return self.hp <= 0

    @property
    def hp_fraction(self) -> float:
        return self.hp / self.max_hp if self.max_hp > 0 else 0.0


@dataclass(frozen=True)
class Warp:
    x: int
    y: int
    dest_map: MapKey | None     # None: "the map the player came from" (Gen 1 $FF)
    dest_index: int


@dataclass(frozen=True)
class Connection:
    direction: Direction
    dest_map: MapKey


@dataclass(frozen=True)
class Npc:
    index: int                  # object slot in the map (stable while in this map)
    x: int
    y: int
    sprite: int
    visible: bool = True
    moving: bool = False        # wanders around (target its live position)
    is_item: bool = False       # a Poke Ball lying on the ground


@dataclass
class MapInfo:
    key: MapKey
    width: int                  # cells
    height: int
    grid: NDArray[np.int8]      # (height, width) cell classes
    warps: list[Warp]
    connections: list[Connection]
    signs: list[tuple[int, int]]
    tileset: int = 0
    outdoor: bool = False
    name: str = ""

    def inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def cell(self, x: int, y: int) -> int:
        return int(self.grid[y, x]) if self.inside(x, y) else UNKNOWN


@dataclass(frozen=True)
class MenuInfo:
    cursor_row: int             # screen tile row of the active cursor
    cursor_col: int
    options: tuple[str, ...]    # decoded option labels, top to bottom
    selected: int               # index into options
    kind: str = "generic"       # "yesno" | "battle_main" | "battle_moves" | "party" | "start" | "generic"
    positions: tuple[tuple[int, int], ...] = ()   # (row, col) of each label's first tile


@dataclass(frozen=True)
class ScreenInfo:
    lines: tuple[str, ...]      # decoded text of the dialog box (if any)
    textbox: bool               # bottom dialog frame drawn
    prompt: bool                # "continue" arrow waiting for a button
    menu: MenuInfo | None
    naming: bool = False        # naming keyboard on screen
    rows: tuple[str, ...] = ()  # every screen row decoded (classification of full-screen menus)


@dataclass(frozen=True)
class BattleInfo:
    kind: Literal["wild", "trainer"]
    me: MonInfo | None
    enemy: MonInfo | None
    active_index: int
    turn: int = 0


@dataclass(frozen=True)
class Snapshot:
    frame: int
    map: MapKey
    x: int
    y: int
    facing: Direction
    mode: Mode
    screen: ScreenInfo
    party: tuple[MonInfo, ...]
    battle: BattleInfo | None
    badges: int
    money: int
    bag: tuple[tuple[int, int], ...]        # (item id, quantity)
    event_flags: int                        # number of story event flags set (progress proxy)
    dex_owned: int
    walking: bool = False
    joy_locked: bool = False                # a script owns the controls

    @property
    def cell(self) -> tuple[MapKey, int, int]:
        return (self.map, self.x, self.y)

    @property
    def party_hp_fraction(self) -> float:
        total = sum(mon.max_hp for mon in self.party)
        return sum(mon.hp for mon in self.party) / total if total else 0.0

    @property
    def able_count(self) -> int:
        return sum(1 for mon in self.party if mon.hp > 0)
