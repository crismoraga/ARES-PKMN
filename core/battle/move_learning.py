"""Explainable move-replacement policy shared by battle, evolution and field UIs.

The engine asks which move to forget when a Pokemon with four moves learns a new
one. HMs are never forgotten. A new move that is weaker than every replaceable
move is declined, so a level-up can never silently weaken the team.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from core.game_data import RomCatalog

# Generation-three EFFECT_EXPLOSION faints the user; it is not a reliable attack.
_SELF_KO_EFFECT = 7
_STATUS_VALUE = 12.0
# Power-1 moves whose damage only reflects damage received (Bide, Counter,
# Mirror Coat) are situational; level-based ones (Seismic Toss, Dragon Rage,
# Sonic Boom, Psywave, Super Fang) are reliable early-game damage.
_REACTIVE_EFFECTS = frozenset({26, 89, 144})
_DREAM_EATER_EFFECT = 8
_SNORE_EFFECT = 92


@dataclass(frozen=True)
class MoveSummary:
    """The ROM facts this policy needs, independent of catalog implementations."""
    power: int
    move_type: int
    accuracy: int
    effect: int = 0


@dataclass(frozen=True)
class MoveLearningChoice:
    forget_slot: int | None
    reason: str
    new_move_value: float
    replaced_value: float | None = None


def rom_move_summary(catalog: RomCatalog) -> Callable[[int], MoveSummary]:
    """Adapt a verified ROM catalog to the policy's minimal move facts."""
    def summary(move_id: int) -> MoveSummary:
        info = catalog.move(move_id)
        return MoveSummary(info.power, info.move_type, info.accuracy, info.effect)
    return summary


def rom_hm_moves(catalog: RomCatalog) -> frozenset[int]:
    """The eight native HM moves; field progress depends on never forgetting them."""
    return frozenset(catalog.tmhm_move(item) for item in range(339, 347))


def move_value(move_id: int, summary: Callable[[int], MoveSummary],
               species_types: Sequence[int] = ()) -> float:
    """Rough battle value: accuracy-weighted power with STAB; status is modest."""
    if move_id <= 0:
        return -1.0
    move = summary(move_id)
    if move.effect == _SELF_KO_EFFECT:
        return 5.0
    if move.power == 0 or move.effect in _REACTIVE_EFFECTS:
        return _STATUS_VALUE
    if move.effect in (_DREAM_EATER_EFFECT, _SNORE_EFFECT):
        return 10.0  # Only works while the target (or the user) sleeps.
    accuracy = (move.accuracy or 100) / 100
    # Variable-damage moves (Seismic Toss, Night Shade, Return...) store power 1.
    power = 50 if move.power == 1 else move.power
    stab = 1.5 if move.move_type in species_types else 1.0
    return power * accuracy * stab


def choose_move_to_forget(current_moves: Sequence[int], new_move: int,
                          summary: Callable[[int], MoveSummary],
                          protected: frozenset[int] = frozenset(),
                          species_types: Sequence[int] = ()) -> MoveLearningChoice:
    """Return the slot to replace, or None to keep the current four moves."""
    moves = [int(move) for move in current_moves][:4]
    new_value = move_value(new_move, summary, species_types)
    empty = next((slot for slot, move in enumerate(moves) if move == 0), None)
    if empty is not None:
        return MoveLearningChoice(empty, "free move slot", new_value, None)
    if new_move in moves:
        return MoveLearningChoice(None, "move already known", new_value, None)
    candidates = [slot for slot, move in enumerate(moves) if move not in protected]
    if not candidates:
        return MoveLearningChoice(None, "every known move is a protected HM", new_value, None)
    values = {slot: move_value(moves[slot], summary, species_types) for slot in candidates}
    worst = min(candidates, key=lambda slot: (values[slot], slot))
    if new_value <= values[worst]:
        return MoveLearningChoice(None, "new move is weaker than every replaceable move",
                                  new_value, values[worst])
    return MoveLearningChoice(worst, "replace the weakest non-HM move", new_value, values[worst])
