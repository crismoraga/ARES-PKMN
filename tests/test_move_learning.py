"""Move replacement never weakens the team and never forgets an HM."""
from __future__ import annotations

from core.battle.move_learning import MoveSummary, choose_move_to_forget, move_value

# A tiny explicit move table: (power, type, accuracy, effect).
_MOVES = {
    33: MoveSummary(35, 0, 95),      # Tackle
    117: MoveSummary(1, 0, 100, 26), # Bide: reactive, power-1 placeholder
    189: MoveSummary(20, 4, 100),    # Mud-Slap (Ground)
    55: MoveSummary(40, 11, 100),    # Water Gun
    341: MoveSummary(55, 4, 95),     # Mud Shot (Ground)
    45: MoveSummary(0, 0, 100),      # Growl
    69: MoveSummary(1, 1, 100, 87),  # Seismic Toss: reliable level damage
    120: MoveSummary(200, 0, 100, 7),  # Self-Destruct
    15: MoveSummary(50, 0, 95),      # Cut (an HM)
    57: MoveSummary(95, 11, 100),    # Surf (an HM)
}
_HMS = frozenset({15, 57})


def summary(move: int) -> MoveSummary:
    return _MOVES[move]


def test_free_slot_is_used_before_forgetting_anything() -> None:
    choice = choose_move_to_forget([33, 45, 0, 0], 55, summary, _HMS)
    assert choice.forget_slot == 2 and choice.reason == "free move slot"


def test_marshtomp_mud_shot_replaces_bide_not_its_stab_attack() -> None:
    # The real Emerald stall: Marshtomp (Water/Ground) learning Mud Shot.
    choice = choose_move_to_forget([33, 117, 189, 55], 341, summary, _HMS, species_types=(11, 4))
    assert choice.forget_slot == 1  # Bide is situational; Mud-Slap keeps STAB.


def test_a_weaker_move_is_declined_instead_of_replacing_a_stronger_one() -> None:
    choice = choose_move_to_forget([33, 189, 55, 341], 45, summary, _HMS, species_types=(11, 4))
    assert choice.forget_slot is None and "weaker" in choice.reason


def test_hms_are_never_forgotten_even_when_weak() -> None:
    assert choose_move_to_forget([15, 57, 15, 57], 341, summary, _HMS).forget_slot is None
    choice = choose_move_to_forget([15, 45, 57, 33], 341, summary, _HMS)
    assert choice.forget_slot == 1  # Growl, never Cut or Surf


def test_self_ko_and_reactive_moves_are_valued_below_real_attacks() -> None:
    assert move_value(120, summary) < move_value(33, summary)
    assert move_value(117, summary) < move_value(189, summary, (4,))
    assert move_value(69, summary) > move_value(117, summary)
    assert move_value(0, summary) < 0


def test_an_already_known_move_changes_nothing() -> None:
    assert choose_move_to_forget([33, 55, 189, 341], 55, summary, _HMS).forget_slot is None
