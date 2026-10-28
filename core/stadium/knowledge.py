"""What one adversary has observed, and how two adversaries exchange it.

An agent never reads the ROM type table during play: its type chart is learnt
from battle messages ("super effective", "doesn't affect", Levitate...) with
the campaign's ``TypeChartMemory`` (constraint propagation over dual types).
It also remembers every move and ability an opposing species revealed.
``exchange`` merges two agents' memories (union, idempotent, commutative):
complementary learning means each side gains the facts the other discovered.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from core.strategy_knowledge import NEUTRAL, NO_EFFECT, RESISTED, SUPER, TypeChartMemory

from .dex import Dex
from .engine import Event

_CATEGORY = {"super": SUPER, "resisted": RESISTED, "neutral": NEUTRAL, "no_effect": NO_EFFECT}


@dataclass
class SideKnowledge:
    chart: TypeChartMemory = field(default_factory=TypeChartMemory)
    revealed_moves: dict[int, set[int]] = field(default_factory=dict)
    revealed_abilities: dict[int, int] = field(default_factory=dict)
    facts_learned: int = 0
    _multiplier_cache: dict[tuple[int, int, int], float | None] = field(default_factory=dict)

    def multiplier(self, attack_type: int, defender_types: tuple[int, int]) -> float | None:
        """Learnt multiplier, or None when the agent has not established it yet."""
        key = (attack_type, defender_types[0], defender_types[1])
        if key not in self._multiplier_cache:
            if attack_type == 9:
                self._multiplier_cache[key] = 1.0
            else:
                self._multiplier_cache[key] = self.chart.multiplier(attack_type, defender_types)
        return self._multiplier_cache[key]

    def observe(self, dex: Dex, events: Iterable[Event], me: int, species_on_field: Mapping[int, int]) -> int:
        """Learn from one step's messages; returns how many new facts appeared."""
        new = 0
        for event in events:
            kind = event[0]
            if kind == "hit" and event[3] in _CATEGORY:
                _, _, _, label, _, move_type, _, defender_species, _ = event
                if move_type == 9:
                    continue
                types = dex.species[int(defender_species)].types
                if self.chart.observe(int(move_type), types, _CATEGORY[str(label)], int(defender_species),
                                      dex.species[int(defender_species)].abilities):
                    new += 1
            elif kind == "noeffect":
                _, _, move_type, _, defender_species = event
                types = dex.species[int(defender_species)].types
                if self.chart.observe(int(move_type), types, NO_EFFECT, int(defender_species),
                                      dex.species[int(defender_species)].abilities):
                    new += 1
            elif kind == "move" and event[1] != me:
                species = species_on_field.get(int(event[1]))
                if species is not None and int(event[3]) not in self.revealed_moves.setdefault(species, set()):
                    self.revealed_moves[species].add(int(event[3]))
            elif kind == "ability" and event[1] != me:
                species = species_on_field.get(int(event[1]))
                if species is not None:
                    self.revealed_abilities[species] = int(event[2])
        if new:
            self.facts_learned += new
            self._multiplier_cache.clear()
        return new

    def state(self) -> dict[str, Any]:
        return {"chart": self.chart.state(),
                "revealed_moves": {str(species): sorted(moves) for species, moves in self.revealed_moves.items()},
                "revealed_abilities": {str(species): ability for species, ability in self.revealed_abilities.items()},
                "facts_learned": self.facts_learned}

    def merge_state(self, state: Mapping[str, Any]) -> None:
        chart = state.get("chart")
        if isinstance(chart, Mapping):
            self.chart.merge_state(chart)
        for species, moves in dict(state.get("revealed_moves", {})).items():
            self.revealed_moves.setdefault(int(species), set()).update(int(move) for move in moves)
        for species, ability in dict(state.get("revealed_abilities", {})).items():
            self.revealed_abilities[int(species)] = int(ability)
        self._multiplier_cache.clear()

    def known_cells(self) -> int:
        return self.chart.known_cells()


def exchange(first: SideKnowledge, second: SideKnowledge) -> tuple[int, int]:
    """Complementary knowledge transfer; returns the cells each side gained."""
    before = first.known_cells(), second.known_cells()
    first_state, second_state = first.state(), second.state()
    first.merge_state(second_state)
    second.merge_state(first_state)
    return first.known_cells() - before[0], second.known_cells() - before[1]
