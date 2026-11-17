"""Immutable battle data tables of one Gen-3 game.

The tables come from the hash-verified ROM when it is present, or from the
snapshot ``data/stadium_dex.json`` (exported from that same ROM) when it is
not, so the stadium runs on machines that have no ROM at all. Both sources
produce identical tables (a test compares them).

The engine never asks the ROM during a battle: every lookup is a tuple index.
Type effectiveness keeps the native table order, because Gen-3 applies each
matching modifier separately with truncation (the order changes damage).
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from core.game_data import ROM_NAMES, ROOT, RomCatalog

Game = Literal["firered", "emerald"]
TYPE_COUNT = 18
TYPE_MYSTERY = 9
TYPE_NAMES = ("Normal", "Fighting", "Flying", "Poison", "Ground", "Rock", "Bug", "Ghost", "Steel", "???",
              "Fire", "Water", "Grass", "Electric", "Psychic", "Ice", "Dragon", "Dark")


@dataclass(frozen=True, slots=True)
class MoveData:
    move_id: int
    name: str
    effect: int
    power: int
    move_type: int
    accuracy: int
    pp: int
    chance: int
    target: int
    priority: int
    flags: int

    @property
    def contact(self) -> bool:
        return bool(self.flags & 1)

    @property
    def damaging(self) -> bool:
        return self.power > 0

    @property
    def special(self) -> bool:
        return self.move_type >= 10


@dataclass(frozen=True, slots=True)
class SpeciesData:
    species_id: int
    name: str
    base_stats: tuple[int, int, int, int, int, int]  # hp, atk, def, spe, spa, spd
    types: tuple[int, int]
    abilities: tuple[int, int]


SNAPSHOT_PATH = ROOT / "data" / "stadium_dex.json"
Learnset = tuple[tuple[int, int], ...]
Evolutions = tuple[tuple[int, int, int], ...]


class Dex:
    """Species, moves, ordered type table, evolutions and level-up learnsets."""

    def __init__(self, game: str, moves: tuple[MoveData, ...], species: tuple[SpeciesData, ...],
                 type_entries: tuple[tuple[int, int, int], ...], evolutions: tuple[Evolutions, ...],
                 learnsets: tuple[Learnset, ...], source: str) -> None:
        self.game, self.moves, self.species = game, moves, species
        self.type_entries, self.evolutions, self.learnsets = type_entries, evolutions, learnsets
        self.source = source  # "rom" or "snapshot"
        self._ordered: dict[tuple[int, int, int], tuple[tuple[int, int], ...]] = {}
        self.level_up: dict[int, tuple[int, ...]] = {}

    # ------------------------------------------------------------ sources
    @classmethod
    def from_catalog(cls, catalog: RomCatalog) -> Dex:
        moves = tuple(cls._move(catalog, index) for index in range(catalog.move_count))
        species = tuple(cls._species(catalog, index) for index in range(catalog.species_count))
        symbol = catalog.symbols["gTypeEffectiveness"]
        raw = catalog.read(symbol.address, symbol.size)
        entries: list[tuple[int, int, int]] = []
        for index in range(0, len(raw) - 2, 3):
            attack, defense, factor = raw[index], raw[index + 1], raw[index + 2]
            if attack == 0xFF:
                break
            if attack == 0xFE:  # TYPE_FORESIGHT separator: Ghost immunities follow normally
                continue
            entries.append((attack, defense, factor))
        evolution = catalog.symbols["gEvolutionTable"]
        table = catalog.read(evolution.address, evolution.size)
        rows: list[Evolutions] = []
        for index in range(evolution.size // 40):
            methods = []
            for slot in range(5):
                method, param, target = struct.unpack_from("<HHH", table, index * 40 + slot * 8)
                if method:
                    methods.append((int(method), int(param), int(target)))
            rows.append(tuple(methods))
        pointers = catalog.symbols["gLevelUpLearnsets"].address
        learnsets: list[Learnset] = []
        for index in range(catalog.species_count):
            learnset: list[tuple[int, int]] = []
            if species[index].name != "?":
                try:
                    pointer = struct.unpack("<I", catalog.read(pointers + 4 * index, 4))[0]
                    for position in range(256):
                        entry = struct.unpack("<H", catalog.read(pointer + 2 * position, 2))[0]
                        if entry == 0xFFFF:
                            break
                        learnset.append((entry >> 9, entry & 0x1FF))
                except ValueError:
                    learnset = []
            learnsets.append(tuple(learnset))
        return cls(catalog.game, moves, species, tuple(entries), tuple(rows), tuple(learnsets), "rom")

    def to_snapshot(self) -> dict[str, Any]:
        """Plain tables (no ROM code or assets), enough to run the stadium without the ROM."""
        return {
            "version": 1, "game": self.game,
            "moves": [[m.name, m.effect, m.power, m.move_type, m.accuracy, m.pp, m.chance, m.target, m.priority,
                       m.flags] for m in self.moves],
            "species": [[d.name, list(d.base_stats), list(d.types), list(d.abilities)] for d in self.species],
            "types": [list(entry) for entry in self.type_entries],
            "evolutions": [[list(method) for method in row] for row in self.evolutions],
            "learnsets": [[list(entry) for entry in row] for row in self.learnsets],
        }

    @classmethod
    def from_snapshot(cls, raw: dict[str, Any]) -> Dex:
        if raw.get("version") != 1:
            raise ValueError("unsupported stadium dex snapshot")
        moves = tuple(MoveData(index, str(row[0]), *(int(value) for value in row[1:]))
                      for index, row in enumerate(raw["moves"]))
        species = []
        for index, (name, stats, types, abilities) in enumerate(raw["species"]):
            species.append(SpeciesData(index, str(name), (int(stats[0]), int(stats[1]), int(stats[2]), int(stats[3]),
                                                          int(stats[4]), int(stats[5])),
                                       (int(types[0]), int(types[1])), (int(abilities[0]), int(abilities[1]))))
        types_table = tuple((int(a), int(b), int(c)) for a, b, c in raw["types"])
        evolutions = tuple(tuple((int(a), int(b), int(c)) for a, b, c in row) for row in raw["evolutions"])
        learnsets = tuple(tuple((int(level), int(move)) for level, move in row) for row in raw["learnsets"])
        return cls(str(raw["game"]), moves, tuple(species), types_table, evolutions, learnsets, "snapshot")

    @staticmethod
    def _move(catalog: RomCatalog, move_id: int) -> MoveData:
        if move_id == 0:
            return MoveData(0, "-", 0, 0, 0, 0, 0, 0, 0, 0, 0)
        info = catalog.move(move_id)
        return MoveData(move_id, info.name, info.effect, info.power, info.move_type, info.accuracy, info.pp,
                        info.secondary_chance, info.target, info.priority, info.flags)

    @staticmethod
    def _species(catalog: RomCatalog, species_id: int) -> SpeciesData:
        try:
            info = catalog.species(species_id)
        except (ValueError, KeyError):
            return SpeciesData(species_id, "?", (1, 1, 1, 1, 1, 1), (0, 0), (0, 0))
        return SpeciesData(species_id, info.name, info.base_stats, info.types, info.abilities)

    # ------------------------------------------------------------ lookups
    def modifiers(self, attack_type: int, defender_types: tuple[int, int]) -> tuple[tuple[int, int], ...]:
        """Native-order (numerator, 10) modifiers; empty means neutral."""
        key = (attack_type, defender_types[0], defender_types[1])
        cached = self._ordered.get(key)
        if cached is not None:
            return cached
        types = {defender_types[0], defender_types[1]}
        result = tuple((factor, 10) for attack, defense, factor in self.type_entries
                       if attack == attack_type and defense in types)
        self._ordered[key] = result
        return result

    def multiplier(self, attack_type: int, defender_types: tuple[int, int]) -> float:
        value = 1.0
        for numerator, denominator in self.modifiers(attack_type, defender_types):
            value *= numerator / denominator
        return value

    def evolve(self, species_id: int, level: int, personality: int = 0, friendship: int = 70,
               stats: tuple[int, ...] | None = None) -> int:
        """Species reached by levelling to ``level`` (level/friendship methods only, as in game)."""
        current = species_id
        for _ in range(3):
            following = None
            for method, param, target in self.evolutions[current] if current < len(self.evolutions) else ():
                attack, defense = (stats[1], stats[2]) if stats is not None and len(stats) > 2 else (1, 1)
                if method == 4 and level >= param:
                    following = target
                elif method in (8, 9, 10) and level >= param and (
                        (method == 8 and attack > defense) or (method == 9 and attack == defense) or
                        (method == 10 and attack < defense)):
                    following = target
                elif method in (11, 12) and level >= param:
                    silcoon = (personality >> 16) % 10 <= 4
                    if (method == 11) == silcoon:
                        following = target
                elif method == 13 and level >= param:
                    following = target  # Nincada becomes Ninjask (Shedinja needs a free party slot)
                elif method == 1 and friendship >= 220:
                    following = target
                if following is not None:
                    break
            if following is None:
                return current
            current = following
        return current

    def learnable(self, species_id: int, level: int) -> tuple[int, ...]:
        """Level-up moves legally known by this level (TM moves come from the team itself)."""
        key = species_id * 128 + level
        cached = self.level_up.get(key)
        if cached is None:
            result: list[int] = []
            for learnt_at, move in self.learnsets[species_id] if 0 <= species_id < len(self.learnsets) else ():
                if learnt_at <= level:
                    if move in result:
                        result.remove(move)
                    result.append(move)
            cached = tuple(result)
            self.level_up[key] = cached
        return cached


def rom_game() -> Game | None:
    """Emerald is preferred (it is a superset for battle data); FireRed otherwise."""
    if (ROOT.parent / ROM_NAMES["emerald"]).is_file():
        return "emerald"
    if (ROOT.parent / ROM_NAMES["firered"]).is_file():
        return "firered"
    return None


def available_game() -> Game:
    game = rom_game()
    if game is not None:
        return game
    if SNAPSHOT_PATH.is_file():
        return "emerald"
    raise FileNotFoundError("The stadium needs data/stadium_dex.json or a FireRed/Emerald ROM next to the project")


def export_snapshot(path: Path = SNAPSHOT_PATH) -> Path:
    """Write the ROM-free data file (needs the ROM once, on the machine that prepares the demo)."""
    game = rom_game()
    if game is None:
        raise FileNotFoundError("Exporting the stadium dex needs the ROM")
    dex = Dex.from_catalog(RomCatalog(game))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dex.to_snapshot(), separators=(",", ":")), encoding="utf-8")
    return path


@lru_cache(maxsize=4)
def load_dex(game: Game | None = None, prefer_snapshot: bool = False) -> Dex:
    """The ROM when present; otherwise the shipped snapshot."""
    chosen = game if game is not None else rom_game()
    if chosen is not None and not prefer_snapshot and (ROOT.parent / ROM_NAMES[chosen]).is_file():
        try:
            return Dex.from_catalog(RomCatalog(chosen))
        except (ValueError, OSError):
            pass  # unsupported ROM revision: fall back to the snapshot
    if SNAPSHOT_PATH.is_file():
        return Dex.from_snapshot(json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8")))
    raise FileNotFoundError("The stadium needs data/stadium_dex.json or a FireRed/Emerald ROM next to the project")
