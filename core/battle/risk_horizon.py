"""Pure, bounded damage-burden estimates for productive battle plans.

The result is explicitly a risk proxy, not a calibrated faint probability or a
safety proof. Every option is charged until the estimated opponent knockout;
switching additionally spends a turn taking an unanswered hit. No hidden
opponent state or emulator access belongs in this module.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

RISK_MODEL = "expected_damage_until_ko_over_current_hp"


@dataclass(frozen=True)
class MoveDamageEstimate:
    index: int
    damage_on_hit: float
    accuracy: float
    pp: int

    def __post_init__(self) -> None:
        if not 0 <= self.index < 4 or self.pp < 0:
            raise ValueError("Move slot and PP must describe a legal own move")
        if (not math.isfinite(self.damage_on_hit) or self.damage_on_hit < 0 or
                not math.isfinite(self.accuracy) or not 0 <= self.accuracy <= 1):
            raise ValueError("Damage must be finite and accuracy must be a probability")

    @property
    def expected_damage(self) -> float:
        return self.damage_on_hit * self.accuracy


@dataclass(frozen=True)
class HorizonEstimate:
    risk_proxy: float
    turns_to_ko: int | None
    incoming_turns: int
    expected_damage_taken: float
    first_move_index: int | None
    can_finish: bool
    remaining_enemy_hp: float
    risk_model: str = RISK_MODEL


def estimate_attack_plan(hp: float, enemy_hp: float, incoming_damage: float,
                         moves: Sequence[MoveDamageEstimate], *,
                         first_move_index: int | None = None,
                         switching: bool = False, acts_first: bool = False,
                         max_turns: int = 64, initial_delay_turns: int = 0) -> HorizonEstimate:
    """Estimate one attack followed by best legal attacks, or switch then attack.

    ``damage_on_hit`` excludes accuracy; ``incoming_damage`` is the caller's
    visible-data/belief estimate per opposing turn. Known move PP is consumed
    once per planned attack. Zero-power, zero-accuracy and exhausted moves are
    unavailable. ``acts_first`` discounts retaliation on the final KO turn;
    use False when speed order is uncertain. Status moves can conservatively
    spend an initial delay turn without claiming an unmodelled benefit.

    ``risk_proxy`` is unbounded expected damage divided by current HP. A value
    below one only describes this approximate model. A plan that cannot finish
    has ``can_finish=False`` and an explicit finite unresolved penalty. Such a
    switch should be excluded whenever a productive attack option exists.
    """
    if (not all(math.isfinite(value) for value in (hp, enemy_hp, incoming_damage)) or
            hp <= 0 or enemy_hp < 0 or incoming_damage < 0):
        raise ValueError("HP and incoming damage require finite valid values")
    if not 1 <= max_turns <= 512 or not 0 <= initial_delay_turns <= max_turns:
        raise ValueError("The planning horizon and delay must be bounded")
    if len({move.index for move in moves}) != len(moves):
        raise ValueError("Move slots must be unique")
    legal = [move for move in moves if move.pp > 0 and move.expected_damage > 0]
    if first_move_index is not None and not any(move.index == first_move_index for move in legal):
        raise ValueError("The first move must be a legal damaging move with PP")
    if enemy_hp == 0:
        return HorizonEstimate(0.0, 0, 0, 0.0, None, True, 0.0)
    remaining_pp = {move.index: move.pp for move in legal}
    remaining_hp = enemy_hp
    first: int | None = None
    attacks = 0
    for turn in range(max_turns - initial_delay_turns):
        candidates = [move for move in legal if remaining_pp[move.index] > 0]
        if not candidates:
            break
        if turn == 0 and first_move_index is not None:
            selected = next(move for move in candidates if move.index == first_move_index)
        else:
            selected = max(candidates, key=lambda move: (move.expected_damage, -move.index))
        if first is None:
            first = selected.index
        remaining_pp[selected.index] -= 1
        remaining_hp = max(0.0, remaining_hp - selected.expected_damage)
        attacks += 1
        if remaining_hp <= 1e-9:
            remaining_hp = 0.0
            break
    can_finish = remaining_hp == 0.0
    own_turns = attacks + initial_delay_turns
    incoming_turns = own_turns + int(switching) - int(can_finish and acts_first and attacks > 0)
    if not can_finish:
        # Failing to damage or running out of PP cannot look safer simply
        # because the estimate ended before the battle did.
        incoming_turns = max(incoming_turns, max_turns + int(switching))
    expected_damage_taken = incoming_damage * incoming_turns
    proxy = expected_damage_taken / hp
    if not can_finish:
        proxy = max(1.0, proxy) + 1.0 + remaining_hp / enemy_hp
    return HorizonEstimate(proxy, own_turns + int(switching) if can_finish else None,
                           incoming_turns, expected_damage_taken, first, can_finish, remaining_hp)
