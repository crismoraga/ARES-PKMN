"""Adapters: Stadium BattleEnv -> BattleView, Gen-3 bridge structs -> BattleView -> BattleDecision."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ares_u.battle.env import BattleEnv
from ares_u.battle.tasks import battle_seed
from ares_u.battle.teams import default_dex, eval_team
from ares_u.brain import gen3_adapter as g3
from ares_u.brain.api import BrainChoice, canonical_type
from ares_u.brain.brain import DEFAULT_CHECKPOINT
from ares_u.brain.stadium_adapter import action_of, view_of
from core.bridge.gba_structs import BattlePokemonStruct

# ----------------------------------------------------------------------------- Stadium adapter


def test_stadium_view_mapping() -> None:
    dex = default_dex()
    env = BattleEnv(dex, eval_team(dex, "gen2", 0, 0), eval_team(dex, "gen2", 0, 1), battle_seed("eval", "gen2", 0, 0))
    for side in (0, 1):
        legal = env.legal(side)
        battle = view_of(env, side, legal)
        own = env.battle.sides[side]
        assert len(battle.party) == len(own.party) and battle.me is battle.party[own.active]
        assert battle.generation == 3 and not battle.is_wild
        assert battle.enemy.stats is None and battle.me.stats is not None  # honest enemy view
        assert all(canonical_type(name) == name for mon in battle.party for name in mon.types)
        assert all(0 < mon.national_dex <= 386 for mon in battle.party)
        assert set(battle.legal_moves or ()) <= set(range(len(battle.me.moves)))
        assert action_of(env, side, BrainChoice("move", 99, ""), legal) in legal  # illegal -> legal fallback
        bench = own.bench()
        if bench and any(action >= 4 for action in legal):
            assert action_of(env, side, BrainChoice("switch", bench[0], ""), legal) == 4


# ----------------------------------------------------------------------------- Gen-3 adapter

SPECIES = {25: ("PIKACHU", (13, 13)), 7: ("SQUIRTLE", (11, 11)), 74: ("GEODUDE", (5, 4))}
MOVES = {84: ("THUNDERSHOCK", 40, 13), 98: ("QUICK ATTACK", 40, 0), 55: ("WATER GUN", 40, 11), 33: ("TACKLE", 35, 0)}


class FakeCatalog:
    def species(self, species_id: int) -> Any:
        name, types = SPECIES[species_id]
        return SimpleNamespace(name=name, types=types)

    def move(self, move_id: int) -> Any:
        name, power, kind = MOVES[move_id]
        return SimpleNamespace(name=name, power=power, move_type=kind, accuracy=100, pp=30, priority=0)


def battler(species: int, level: int, hp: int, moves: tuple[int, ...], status: int = 0) -> BattlePokemonStruct:
    mon = BattlePokemonStruct()
    mon.species, mon.level, mon.hp, mon.maxHP = species, level, hp, 80
    mon.attack, mon.defense, mon.spAttack, mon.spDefense, mon.speed = 50, 40, 55, 45, 90
    mon.type1, mon.type2 = SPECIES[species][1]
    mon.status1 = status
    for slot, move in enumerate(moves):
        mon.moves[slot], mon.pp[slot] = move, 20
    for stage in range(8):
        mon.statStages[stage] = 6
    mon.statStages[1] = 8  # +2 attack
    return mon


def party_member(species: int, hp: int, moves: tuple[int, ...]) -> dict[str, Any]:
    return {"species": species, "level": 25, "hp": hp, "maxHP": 75, "attack": 45, "defense": 60, "spAttack": 50,
            "spDefense": 60, "speed": 40, "status": 0, "moves": list(moves) + [0] * (4 - len(moves)),
            "pp": [20] * len(moves) + [0] * (4 - len(moves))}


PARTY = [party_member(25, 80, (84, 98)), party_member(7, 75, (55, 33))]


def features(switch_ok: bool = True, forced: bool = False) -> Any:
    mask = np.zeros((1, 9), dtype=bool)
    mask[0, :2] = not forced
    mask[0, 4] = switch_ok
    return SimpleNamespace(action_mask=mask, switch_party_indices=(1,), target_mask=np.array([[True, False, False]]),
                           active_party_index=0, forced_switch=forced, doubles=False)


def test_status_and_stages() -> None:
    assert [g3.status_of(value) for value in (0, 3, 0x8, 0x10, 0x20, 0x40, 0x80)] == \
        ["", "slp", "psn", "brn", "frz", "par", "tox"]
    assert g3.stages_of([6, 8, 6, 4, 6, 6, 6, 7]) == {"atk": 2, "def": 0, "spe": -2, "spa": 0, "spd": 0, "acc": 0,
                                                     "eva": 1}


def test_gen3_view_is_honest_and_ordered() -> None:
    me, enemy = battler(25, 25, 60, (84, 98), status=0x40), battler(74, 24, 40, (33, 88))
    turn = g3.turn_of(features(), PARTY)
    assert turn.switchable == (1,) and turn.move_legal == (True, True, False, False)
    view = g3.build_view("emerald", me, enemy, PARTY, turn, True, FakeCatalog(), revealed=[33])
    assert view.me.species == "PIKACHU" and view.me.status == "par" and view.me.stages and view.me.stages["atk"] == 2
    assert view.me.hp == 60 and view.party[0] is view.me and view.party[1].species == "SQUIRTLE"
    assert [move.name for move in view.me.moves] == ["THUNDERSHOCK", "QUICK ATTACK"]
    assert view.me.moves[0].category == "special" and view.me.moves[1].category == "physical"
    assert view.enemy.types == ("rock", "ground") and view.enemy.stats is None
    assert [move.name for move in view.enemy.moves] == ["TACKLE"]  # only the revealed move, not RAM's move array
    assert view.legal_moves == (0, 1) and view.can_switch and not view.is_wild and not view.can_run


def test_choice_to_battle_decision() -> None:
    turn = g3.turn_of(features(), PARTY)
    view = g3.build_view("emerald", battler(25, 25, 60, (84, 98)), battler(74, 24, 40, ()), PARTY, turn, True,
                         FakeCatalog())
    move = g3.decision_from_choice(BrainChoice("move", 1, "x"), view, turn, None, 0.75)
    assert move is not None and (move.action, move.index) == ("move", 1) and move.estimated_risk == pytest.approx(0.25)
    switch = g3.decision_from_choice(BrainChoice("switch", 1, "x"), view, turn, None, 0.6)
    assert switch is not None and (switch.action, switch.index) == ("switch", 1)
    assert g3.decision_from_choice(BrainChoice("switch", 0, "x"), view, turn, None) is None  # active mon
    assert g3.decision_from_choice(BrainChoice("move", 3, "x"), view, turn, None) is None    # empty slot
    assert g3.decision_from_choice(BrainChoice("run", 0, "x"), view, turn, None) is None     # heuristic owns run


def test_brain_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(g3.ENV, raising=False)
    assert g3.brain_mode() == g3.DEFAULT_MODE == "learned"
    monkeypatch.setenv(g3.ENV, "Heuristic")
    assert g3.brain_mode() == "heuristic"
    monkeypatch.setenv(g3.ENV, "garbage")
    assert g3.brain_mode() == "learned"


class FakeController:
    def __init__(self, trainer: bool = True) -> None:
        self.bridge = SimpleNamespace(is_in_battle=lambda: (True, trainer))
        self.catalog = FakeCatalog()
        self.knowledge = SimpleNamespace(revealed_moves={})
        self.game = "emerald"
        self.desired_species: set[int] = set()
        self.owned_species: set[int] = set()
        self.native = True
        self._last_directive = {"mode": "EXECUTE_MILESTONE"}
        self._battle_escape_attempts = 0
        self.last_battle_choice: dict[str, Any] | None = None


@pytest.mark.skipif(not DEFAULT_CHECKPOINT.exists(), reason="E1 checkpoint not available")
def test_learned_decision_end_to_end_avoids_immunity() -> None:
    controller = FakeController()
    me, enemy = battler(25, 25, 60, (84, 98)), battler(74, 24, 70, ())
    decision = g3.learned_battle_decision(controller, PARTY, features(), me, enemy)
    assert decision is not None and (decision.action, decision.index) != ("move", 0)  # no Thundershock into Ground
    live = controller.last_battle_choice
    assert live is not None and {"me", "enemy", "choice", "reason", "effectiveness", "scores", "kind"} <= set(live)
    # no switch available: the best move is Quick Attack
    decision = g3.learned_battle_decision(controller, PARTY, features(switch_ok=False), me, enemy)
    assert decision is not None and (decision.action, decision.index) == ("move", 1)
    # wild battle during a story milestone: the heuristic decides (escape logic)
    assert g3.learned_battle_decision(FakeController(trainer=False), PARTY, features(), me, enemy) is None
