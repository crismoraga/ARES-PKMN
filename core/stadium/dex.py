"""Immutable battle data tables, read once from a hash-verified ROM.

The engine never asks the ROM during a battle: every lookup is a tuple index.
Type effectiveness keeps the native table order, because Gen-3 applies each
matching modifier separately with truncation (the order changes damage).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

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


class Dex:
    """Species, moves and the ordered type table of one Gen-3 ROM."""

    def __init__(self, catalog: RomCatalog) -> None:
        self.game = catalog.game
        self.moves: tuple[MoveData, ...] = tuple(self._move(catalog, index) for index in range(catalog.move_count))
        self.species: tuple[SpeciesData, ...] = tuple(self._species(catalog, index)
                                                      for index in range(catalog.species_count))
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
        self.type_entries: tuple[tuple[int, int, int], ...] = tuple(entries)
        self._ordered: dict[tuple[int, int, int], tuple[tuple[int, int], ...]] = {}
        evolution = catalog.symbols["gEvolutionTable"]
        table = catalog.read(evolution.address, evolution.size)
        rows: list[tuple[tuple[int, int, int], ...]] = []
        for species in range(evolution.size // 40):
            entries = []
            for slot in range(5):
                offset = species * 40 + slot * 8
                method = int.from_bytes(table[offset:offset + 2], "little")
                param = int.from_bytes(table[offset + 2:offset + 4], "little")
                target = int.from_bytes(table[offset + 4:offset + 6], "little")
                if method:
                    entries.append((method, param, target))
            rows.append(tuple(entries))
        self.evolutions: tuple[tuple[tuple[int, int, int], ...], ...] = tuple(rows)
        self.level_up: dict[int, tuple[int, ...]] = {}
        self._catalog = catalog

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
            try:
                cached = tuple(self._catalog.level_up_moves(species_id, level))
            except (ValueError, KeyError):
                cached = ()
            self.level_up[key] = cached
        return cached


def available_game() -> Game:
    """Emerald is preferred (it is a superset for battle data); FireRed otherwise."""
    for game in ("emerald", "firered"):
        if (ROOT.parent / ROM_NAMES[game]).is_file():
            return game
    raise FileNotFoundError("The stadium needs a FireRed or Emerald ROM next to the project folder")


@lru_cache(maxsize=2)
def load_dex(game: Game | None = None) -> Dex:
    return Dex(RomCatalog(game or available_game()))
