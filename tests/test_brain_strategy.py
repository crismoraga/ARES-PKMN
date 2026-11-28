"""Battle brain decisions on hand-built views: immunity avoidance, switching, staying, explanations."""
from __future__ import annotations

from functools import lru_cache

import pytest

from ares_u.brain.api import BattleView, MonView, MoveView, category_for
from ares_u.brain.brain import DEFAULT_CHECKPOINT, BattleBrain

MODES = ["strategy"] + (["brain", "policy"] if DEFAULT_CHECKPOINT.exists() else [])


def move(name: str, kind: str, power: int) -> MoveView:
    return MoveView(name, kind, power, 100, 20, category_for(kind, power, 3))


def mon(species: str, dex: int, level: int, types: tuple[str, ...], moves: list[MoveView], own: bool = True,
        stats: tuple[int, int, int, int, int] = (60, 60, 60, 60, 60), hp: float = 1.0) -> MonView:
    top = 3 * level
    named = dict(zip(("atk", "def", "spa", "spd", "spe"), stats))
    return MonView(species, dex, level, int(top * hp), top, types, "", moves, named if own else None)


@lru_cache(maxsize=4)
def brain(mode: str) -> BattleBrain:
    return BattleBrain(mode)


PIKACHU = mon("PIKACHU", 25, 30, ("electric",), [move("THUNDERBOLT", "electric", 95), move("QUICK ATTACK", "normal", 40)],
              stats=(55, 40, 60, 50, 90))
SQUIRTLE = mon("SQUIRTLE", 7, 30, ("water",), [move("WATER GUN", "water", 40), move("TACKLE", "normal", 35)],
               stats=(45, 60, 50, 60, 40))
GEODUDE = mon("GEODUDE", 74, 30, ("rock", "ground"), [], own=False)
BULBASAUR = mon("BULBASAUR", 1, 20, ("grass", "poison"), [move("VINE WHIP", "grass", 35), move("TACKLE", "normal", 35)],
                stats=(30, 30, 35, 35, 28))
BLASTOISE = mon("BLASTOISE", 9, 50, ("water",), [move("SURF", "water", 95), move("BITE", "dark", 60)],
                stats=(110, 130, 115, 135, 105))
CHARIZARD = mon("CHARIZARD", 6, 45, ("fire", "flying"), [move("FLAMETHROWER", "fire", 95)], own=False)


def view(party: list[MonView], enemy: MonView, can_switch: bool = True) -> BattleView:
    return BattleView("test", 3, party[0], enemy, party, 0, is_wild=False, can_switch=can_switch, can_run=False)


@pytest.mark.parametrize("mode", MODES)
def test_never_picks_an_immune_move(mode: str) -> None:
    choice = brain(mode).choose(view([PIKACHU], GEODUDE, can_switch=False))
    assert (choice.kind, choice.index) == ("move", 1)  # Quick Attack, never Thunderbolt into Ground
    rattata = mon("RATTATA", 19, 30, ("normal",), [move("TACKLE", "normal", 35), move("BITE", "dark", 60)])
    gastly = mon("GASTLY", 92, 30, ("ghost", "poison"), [], own=False)
    choice = brain(mode).choose(view([rattata], gastly, can_switch=False))
    assert (choice.kind, choice.index) == ("move", 1)  # Bite, never Tackle into Ghost


@pytest.mark.parametrize("mode", [mode for mode in MODES if mode != "policy"])
def test_switches_out_of_a_lost_matchup(mode: str) -> None:
    choice = brain(mode).choose(view([PIKACHU, SQUIRTLE], GEODUDE))
    assert (choice.kind, choice.index) == ("switch", 1)
    choice = brain(mode).choose(view([BULBASAUR, BLASTOISE], CHARIZARD))
    assert (choice.kind, choice.index) == ("switch", 1)
    assert "Blastoise" in choice.reason


@pytest.mark.parametrize("mode", [mode for mode in MODES if mode != "policy"])
def test_stays_in_a_won_matchup(mode: str) -> None:
    choice = brain(mode).choose(view([BLASTOISE, BULBASAUR], CHARIZARD))
    assert (choice.kind, choice.index) == ("move", 0)  # Surf


def test_legal_moves_and_live_description() -> None:
    battle = view([BLASTOISE, BULBASAUR], CHARIZARD)
    battle.legal_moves = (1,)  # Surf disabled
    choice = brain("strategy").choose(battle)
    assert choice.kind != "move" or choice.index == 1
    live = brain("strategy").describe(battle, choice)
    assert set(live) == {"me", "enemy", "choice", "reason", "effectiveness", "scores"}
    assert live["reason"] and isinstance(live["scores"], dict)


def test_policy_mode_requires_known_mode() -> None:
    with pytest.raises(ValueError):
        BattleBrain("nonsense")
