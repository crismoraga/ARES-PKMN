"""Regression: a Pokemon forced out by Roar/Whirlwind before acting must not act this turn.

Found by the ARES-U E1 protocol (DQN seed 2 crashed): when both sides used a -6 priority
phazing move, the faster one forced the other out and the replacement executed the slot chosen
by its predecessor, indexing past its own move list.
"""
from __future__ import annotations

from core.stadium.dex import load_dex
from core.stadium.engine import Battle, MonSpec

WHIRLWIND, TACKLE, ROAR, SPLASH = 18, 33, 46, 150


def _teams() -> tuple[list[MonSpec], list[MonSpec]]:
    fast = MonSpec(18, 50, (160, 100, 90, 150, 80, 80), (WHIRLWIND, TACKLE))
    slow = MonSpec(59, 50, (180, 120, 90, 60, 90, 90), (TACKLE, TACKLE, TACKLE, ROAR))
    bench = MonSpec(129, 50, (120, 30, 60, 50, 20, 30), (SPLASH,))
    return [fast], [slow, bench]


def test_phazed_replacement_does_not_use_predecessors_slot() -> None:
    dex = load_dex(prefer_snapshot=True)
    dragged = 0
    for seed in range(40):
        team_a, team_b = _teams()
        battle = Battle(dex, team_a, team_b, seed=seed, record=False)
        replacement = battle.sides[1].party[1]
        pp_before = list(replacement.pp)
        battle.submit({0: 0, 1: 3})  # A: Whirlwind (faster), B: Roar in slot 3 -> no IndexError
        if battle.sides[1].fighter is replacement:
            dragged += 1
            assert replacement.pp == pp_before  # the replacement did not act this turn
    assert dragged > 0  # the scenario is actually exercised


def test_phazing_still_switches_and_battle_continues() -> None:
    dex = load_dex(prefer_snapshot=True)
    team_a, team_b = _teams()
    battle = Battle(dex, team_a, team_b, seed=3, record=False)
    battle.submit({0: 0, 1: 3})
    assert battle.sides[1].fighter.spec.species == 129
    assert not battle.over
    assert battle.legal_actions(1)  # the replacement can act next turn
