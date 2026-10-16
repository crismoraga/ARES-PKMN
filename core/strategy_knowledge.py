"""What the agent learns about Pokemon itself, kept across battles, rewinds and games.

A player learns the type chart from the game's own messages ("It's super
effective!", "It's not very effective...", "It doesn't affect..."), remembers
what a gym leader brought last time and prepares a counter. This module holds
that knowledge:

* ``TypeChartMemory``: effectiveness categories observed for (attack type,
  defender types), solved as a small constraint problem so dual-type evidence
  also teaches single-type factors. FireRed and Emerald share the chart, so a
  fact learned in one game helps the other.
* ``OpponentMemory``: species, levels and revealed move types each objective's
  trainer used, and the attack types that counter them according to the chart.

No ROM type table is consulted: facts come only from visible battle results.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
import json
from pathlib import Path
import time
from typing import Any

FACTORS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
TYPE_COUNT = 18
TYPE_MYSTERY = 9
PLAYABLE = tuple(type_id for type_id in range(TYPE_COUNT) if type_id != TYPE_MYSTERY)
# Categories as the battle engine reports them (gMoveResultFlags).
NO_EFFECT, RESISTED, NEUTRAL, SUPER = "no_effect", "resisted", "neutral", "super"
CATEGORIES = (NO_EFFECT, RESISTED, NEUTRAL, SUPER)
# Abilities that can turn a type's hit into "doesn't affect", per attack type
# (None: every type). With such an ability possible, immunity is not a type fact.
IMMUNITY_ABILITIES: dict[int, int | None] = {26: 4, 10: 13, 11: 11, 18: 10, 25: None}
# Moves whose type or damage the chart does not describe.
UNTYPED_MOVES = frozenset({165, 237, 311})  # Struggle, Hidden Power, Weather Ball


def category_of(product_value: float) -> str:
    if product_value == 0:
        return NO_EFFECT
    if product_value < 1:
        return RESISTED
    if product_value > 1:
        return SUPER
    return NEUTRAL


def result_category(result_flags: int) -> str | None:
    """Category from gMoveResultFlags; None for misses and failures (no evidence)."""
    if result_flags & 0x21:  # MISSED | FAILED
        return None
    if result_flags & 0x08:
        return NO_EFFECT
    if result_flags & 0x02:
        return SUPER
    if result_flags & 0x04:
        return RESISTED
    return NEUTRAL


def _defenders(types: Sequence[int]) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in types if 0 <= int(value) < TYPE_COUNT}))
    if not values:
        raise ValueError("A defender needs a type")
    return values


@dataclass
class TypeChartMemory:
    """Observed categories per (attack, defender types) and the factors they imply."""
    facts: dict[tuple[int, ...], set[str]] = field(default_factory=dict)
    immunity_species: dict[tuple[int, ...], set[int]] = field(default_factory=dict)
    observations: int = 0
    _solution: dict[tuple[int, int], frozenset[float]] | None = None

    def observe(self, attack_type: int, defender_types: Sequence[int], category: str,
                species: int = 0, species_abilities: Sequence[int] = (), abilities_known: bool = True) -> bool:
        """Record one visible result. Returns True when it is new knowledge."""
        if category not in CATEGORIES or not 0 <= attack_type < TYPE_COUNT or attack_type == TYPE_MYSTERY:
            return False
        defenders = _defenders(defender_types)
        key = (attack_type, *defenders)
        if category == NO_EFFECT:
            explained = not abilities_known or any(
                ability in IMMUNITY_ABILITIES and IMMUNITY_ABILITIES[ability] in (None, attack_type)
                for ability in species_abilities)
            witnesses = self.immunity_species.setdefault(key, set())
            if species:
                witnesses.add(species)
            # An ability (Levitate, Volt Absorb...) may explain one species; two
            # different species, or one with no such ability, make it a type fact.
            if explained and len(witnesses) < 2:
                return False
        known = self.facts.setdefault(key, set())
        self.observations += 1
        if category in known:
            return False
        known.add(category)
        self._solution = None
        return True

    def _solve(self) -> dict[tuple[int, int], frozenset[float]]:
        """Arc-consistent factor domains for every (attack, single defender) pair."""
        if self._solution is not None:
            return self._solution
        domains: dict[tuple[int, int], set[float]] = {}

        def domain(attack: int, defender: int) -> set[float]:
            return domains.setdefault((attack, defender), set(FACTORS))

        pairs: list[tuple[int, int, int, set[str]]] = []
        for key, categories in self.facts.items():
            attack = key[0]
            if len(key) == 2:
                allowed = {factor for factor in FACTORS if category_of(factor) in categories}
                if allowed:
                    domain(attack, key[1]).intersection_update(allowed)
            else:
                pairs.append((attack, key[1], key[2], categories))
        changed = True
        while changed:
            changed = False
            for attack, first, second, categories in pairs:
                left, right = domain(attack, first), domain(attack, second)
                consistent = [(a, b) for a, b in product(left, right) if category_of(a * b) in categories]
                if not consistent:
                    continue  # contradictory evidence (ability, glitch): keep prior domains
                new_left = {a for a, _ in consistent}
                new_right = {b for _, b in consistent}
                if new_left != left or new_right != right:
                    domains[(attack, first)], domains[(attack, second)] = new_left, new_right
                    changed = True
        self._solution = {key: frozenset(values) for key, values in domains.items() if values}
        return self._solution

    def factor(self, attack_type: int, defender: int) -> float | None:
        values = self._solve().get((attack_type, defender))
        return next(iter(values)) if values is not None and len(values) == 1 else None

    def multiplier(self, attack_type: int, defender_types: Sequence[int]) -> float | None:
        """Exact multiplier when observed directly or implied by known factors."""
        if attack_type == TYPE_MYSTERY:
            return 1.0
        try:
            defenders = _defenders(defender_types)
        except ValueError:
            return None
        key = (attack_type, *defenders)
        categories = self.facts.get(key)
        factors = [self.factor(attack_type, defender) for defender in defenders]
        if all(value is not None for value in factors):
            total = 1.0
            for value in factors:
                assert value is not None
                total *= value
            return total
        if categories is not None and len(categories) == 1:
            category = next(iter(categories))
            if category == NO_EFFECT:
                return 0.0
            if len(defenders) == 1:
                return {RESISTED: 0.5, NEUTRAL: 1.0, SUPER: 2.0}[category]
            # Dual type: the category bounds the product; use its common value.
            return {RESISTED: 0.5, NEUTRAL: 1.0, SUPER: 2.0}[category]
        return None

    def known_cells(self) -> int:
        return sum(1 for attack in PLAYABLE for defender in PLAYABLE if self.factor(attack, defender) is not None)

    def table(self) -> list[list[float | None]]:
        """17x17 learned single-type chart (None where unknown), playable types only."""
        return [[self.factor(attack, defender) for defender in PLAYABLE] for attack in PLAYABLE]

    def state(self) -> dict[str, Any]:
        return {"facts": [[list(key), sorted(values)] for key, values in sorted(self.facts.items())],
                "immunity_species": [[list(key), sorted(values)] for key, values in sorted(self.immunity_species.items())],
                "observations": self.observations}

    def merge_state(self, state: Mapping[str, Any]) -> None:
        for row in state.get("facts", [])[:4096] if isinstance(state.get("facts"), list) else []:
            try:
                key = tuple(int(value) for value in row[0])
                values = {str(value) for value in row[1] if str(value) in CATEGORIES}
            except (TypeError, ValueError, IndexError):
                continue
            if 2 <= len(key) <= 3 and all(0 <= value < TYPE_COUNT for value in key) and values:
                self.facts.setdefault(key, set()).update(values)
        for row in state.get("immunity_species", [])[:4096] if isinstance(state.get("immunity_species"), list) else []:
            try:
                key = tuple(int(value) for value in row[0])
                self.immunity_species.setdefault(key, set()).update(int(value) for value in row[1])
            except (TypeError, ValueError, IndexError):
                continue
        observations = state.get("observations")
        if isinstance(observations, int):
            self.observations = max(self.observations, observations)
        self._solution = None


@dataclass
class OpponentMemory:
    """Rosters of objective battles: what the boss brought and what it used."""
    rosters: dict[str, dict[int, int]] = field(default_factory=dict)          # objective -> species -> level
    move_types: dict[str, dict[int, set[int]]] = field(default_factory=dict)  # objective -> species -> types

    def observe(self, objective: str, species: int, level: int, move_type: int | None = None) -> bool:
        if not objective or species <= 0:
            return False
        roster = self.rosters.setdefault(objective, {})
        new = species not in roster or level > roster[species]
        roster[species] = max(level, roster.get(species, 0))
        if move_type is not None and 0 <= move_type < TYPE_COUNT:
            known = self.move_types.setdefault(objective, {}).setdefault(species, set())
            new |= move_type not in known
            known.add(move_type)
        return new

    def counters(self, objective: str, species_types: Mapping[int, Sequence[int]],
                 chart: TypeChartMemory) -> list[int]:
        """Attack types that hit most of the remembered roster super effectively."""
        roster = self.rosters.get(objective, {})
        if not roster:
            return []
        scores: dict[int, float] = {}
        for attack in PLAYABLE:
            score = 0.0
            for species in roster:
                types = species_types.get(species)
                if not types:
                    continue
                value = chart.multiplier(attack, types)
                if value is None:
                    continue
                score += 1.0 if value > 1 else (-0.5 if value < 1 else 0.0)
            if score > 0:
                scores[attack] = score
        return [attack for attack, _ in sorted(scores.items(), key=lambda row: (-row[1], row[0]))][:3]

    def threats(self, objective: str) -> list[int]:
        types: set[int] = set()
        for revealed in self.move_types.get(objective, {}).values():
            types.update(revealed)
        return sorted(types)

    def state(self) -> dict[str, Any]:
        return {"rosters": {name: {str(species): level for species, level in roster.items()}
                            for name, roster in self.rosters.items()},
                "move_types": {name: {str(species): sorted(types) for species, types in rows.items()}
                               for name, rows in self.move_types.items()}}

    def merge_state(self, state: Mapping[str, Any]) -> None:
        rosters = state.get("rosters")
        if isinstance(rosters, dict):
            for name, roster in list(rosters.items())[:256]:
                if isinstance(roster, dict):
                    for species, level in roster.items():
                        try:
                            self.observe(str(name), int(species), int(level))
                        except (TypeError, ValueError):
                            continue
        move_types = state.get("move_types")
        if isinstance(move_types, dict):
            for name, rows in list(move_types.items())[:256]:
                if isinstance(rows, dict):
                    for species, types in rows.items():
                        if isinstance(types, list):
                            for move_type in types:
                                try:
                                    self.move_types.setdefault(str(name), {}).setdefault(int(species), set()).add(
                                        int(move_type))
                                except (TypeError, ValueError):
                                    continue


@dataclass
class SharedKnowledge:
    """Cross-game knowledge file (type chart) plus per-game opponent memory."""
    path: Path | None
    chart: TypeChartMemory = field(default_factory=TypeChartMemory)
    dirty: bool = False

    def load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, dict) and data.get("version") == 1 and isinstance(data.get("type_chart"), dict):
            self.chart.merge_state(data["type_chart"])

    def save(self) -> None:
        """Merge with what the other game wrote meanwhile, then replace atomically."""
        if self.path is None or not self.dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.is_file():
                try:
                    other = json.loads(self.path.read_text(encoding="utf-8"))
                    if isinstance(other, dict) and isinstance(other.get("type_chart"), dict):
                        self.chart.merge_state(other["type_chart"])
                except (OSError, ValueError):
                    pass
            data = {"version": 1, "saved_at": round(time.time(), 3),
                    "description": "Type effectiveness learned from battle messages; shared by FireRed and Emerald.",
                    "known_single_type_cells": self.chart.known_cells(), "type_chart": self.chart.state()}
            temporary = self.path.with_name(f"{self.path.name}.{time.time_ns()}.tmp")
            temporary.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
            temporary.replace(self.path)
            self.dirty = False
        except OSError:
            pass


def counter_species(candidates: Iterable[int], counter_types: Sequence[int],
                    species_types: Mapping[int, Sequence[int]]) -> list[int]:
    """Catchable species whose own types match a counter type (STAB-likely attackers)."""
    wanted = set(counter_types)
    return [species for species in candidates if wanted & set(species_types.get(species, ()))]
