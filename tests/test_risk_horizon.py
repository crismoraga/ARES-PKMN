"""Compare productive plans under a common, explicitly approximate horizon."""
from __future__ import annotations

import math

import pytest

from core.battle.risk_horizon import MoveDamageEstimate, estimate_attack_plan


def test_healthy_weak_bench_is_worse_than_a_lead_that_can_finish_quickly() -> None:
    lead = estimate_attack_plan(21, 24, 4, [MoveDamageEstimate(0, 12, 1, 20)])
    weak = estimate_attack_plan(17, 24, 3, [MoveDamageEstimate(0, 2, 1, 35)], switching=True)
    assert lead.turns_to_ko == 2 and weak.turns_to_ko == 13
    assert lead.risk_proxy < weak.risk_proxy


def test_effective_switch_pays_free_hit_then_repeated_switching_loses_to_attack() -> None:
    lead = estimate_attack_plan(21, 24, 4, [MoveDamageEstimate(0, 6, 1, 20)])
    bench_moves = [MoveDamageEstimate(0, 12, 1, 20)]
    switch = estimate_attack_plan(30, 24, 2, bench_moves, switching=True)
    assert switch.risk_proxy < lead.risk_proxy < 1
    assert switch.incoming_turns == 3 and switch.expected_damage_taken == 6
    attack_after_switch = estimate_attack_plan(28, 24, 2, bench_moves)
    switch_back = estimate_attack_plan(21, 24, 4, [MoveDamageEstimate(0, 6, 1, 20)], switching=True)
    assert attack_after_switch.risk_proxy < switch_back.risk_proxy


def test_accuracy_and_pp_are_part_of_the_outgoing_plan() -> None:
    result = estimate_attack_plan(30, 25, 2, [MoveDamageEstimate(0, 20, 0.5, 1),
                                                   MoveDamageEstimate(1, 5, 1, 3)], first_move_index=0)
    assert result.can_finish and result.turns_to_ko == 4 and result.first_move_index == 0
    assert result.expected_damage_taken == 8


def test_zero_damage_or_exhausted_bench_cannot_offer_a_productive_switch() -> None:
    empty = estimate_attack_plan(30, 25, 2, [MoveDamageEstimate(0, 0, 1, 20),
                                                  MoveDamageEstimate(1, 25, 1, 0)], switching=True)
    attack = estimate_attack_plan(30, 25, 2, [MoveDamageEstimate(0, 10, 1, 5)])
    assert not empty.can_finish and empty.first_move_index is None
    assert empty.risk_proxy > attack.risk_proxy and math.isfinite(empty.risk_proxy)


def test_no_finish_from_exhausting_pp_or_horizon_is_explicit() -> None:
    pp_exhausted = estimate_attack_plan(30, 25, 2, [MoveDamageEstimate(0, 10, 1, 2)])
    bounded = estimate_attack_plan(30, 25, 2, [MoveDamageEstimate(0, 1, 1, 100)], max_turns=4)
    assert not pp_exhausted.can_finish and pp_exhausted.remaining_enemy_hp == 5
    assert not bounded.can_finish and bounded.turns_to_ko is None
    assert bounded.remaining_enemy_hp == 21


def test_known_faster_ko_avoids_last_retaliation_but_switch_still_costs_a_turn() -> None:
    moves = [MoveDamageEstimate(0, 30, 1, 10)]
    direct = estimate_attack_plan(30, 25, 5, moves, acts_first=True)
    switch = estimate_attack_plan(30, 25, 5, moves, switching=True, acts_first=True)
    assert direct.incoming_turns == 0 and switch.incoming_turns == 1
    assert direct.risk_proxy == 0 and switch.risk_proxy == pytest.approx(1 / 6)
    assert "probability" not in switch.risk_model


def test_status_delay_cannot_gain_an_unmodelled_safety_benefit() -> None:
    moves = [MoveDamageEstimate(0, 10, 1, 10)]
    attack = estimate_attack_plan(30, 25, 5, moves)
    status_then_attack = estimate_attack_plan(30, 25, 5, moves, initial_delay_turns=1)
    assert status_then_attack.risk_proxy > attack.risk_proxy


@pytest.mark.parametrize("hp,enemy_hp,incoming", [(0, 10, 1), (20, -1, 1), (20, 10, -1), (math.nan, 10, 1)])
def test_invalid_horizon_state_is_rejected(hp: float, enemy_hp: float, incoming: float) -> None:
    with pytest.raises(ValueError):
        estimate_attack_plan(hp, enemy_hp, incoming, [])
