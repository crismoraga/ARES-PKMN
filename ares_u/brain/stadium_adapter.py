"""The battle brain inside the Stadium: views of Stadium positions and evaluation controllers.

* ``view_of(env, side)``: the ``BattleView`` a player of ``side`` may see (honest enemy: species,
  level, HP fraction, types, status, stat stages, moves it has used, the ability it revealed and how
  many members it has left).
* ``BrainController``: plays a side with ``BattleBrain`` through that view (the full path a game
  adapter uses: view -> brain -> Stadium reconstruction -> E1 policy + strategy -> choice -> action).
* ``CampaignController``: a faithful Stadium port of the campaign's analytic heuristic
  (``core.orchestrator.AresAutonomousOrchestrator.choose_battle_decision``): the same damage
  estimator (``DamageContext`` + ``GbaIntegerSafetyVerifier.compute_context_damage_reference`` at
  roll 93 with the IV 0/15/31 belief mean for the rival's stats), the same retaliation estimate
  (revealed moves or a 50-power STAB prior, max attack vs min defence), the same
  ``estimate_attack_plan`` horizon and the same rule: minimum ``(risk_proxy, score)`` over moves
  and switches, a switch that cannot finish is dropped when an attack can. Omitted (they only
  break exact ties of ``risk_proxy``): the causal-EFE term and the recurrent-policy logits (x0.05).
  The type chart is the true one (the campaign learns it from battle messages).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ares_u.battle.agents import Choice, Decision
from ares_u.battle.env import BattleEnv
from ares_u.battle.obs import resolve_move
from core.stadium.engine import BURN, FREEZE, PARALYSIS, POISON, SLEEP_MASK, TOXIC, Fighter

from .api import GEN3_INDEX_TYPE, BattleView, BrainChoice, MonView, MoveView, category_for
from .brain import BattleBrain
from .strategy import national_number

_STATUS = ((SLEEP_MASK, "slp"), (TOXIC, "tox"), (POISON, "psn"), (BURN, "brn"), (FREEZE, "frz"),
           (PARALYSIS, "par"))
_STAGES = ("atk", "def", "spe", "spa", "spd", "acc", "eva")
_ABILITY_NAMES = {5: "STURDY", 7: "LIMBER", 10: "VOLT ABSORB", 11: "WATER ABSORB", 15: "INSOMNIA", 17: "IMMUNITY",
                  18: "FLASH FIRE", 25: "WONDER GUARD", 26: "LEVITATE", 41: "WATER VEIL", 47: "THICK FAT",
                  72: "VITAL SPIRIT"}


def status_name(status: int) -> str:
    for mask, name in _STATUS:
        if status & mask:
            return name
    return ""


def _national(env: BattleEnv, species: int) -> int:
    return national_number(species)


def _moves(env: BattleEnv, fighter: Fighter, own: bool, move_ids: list[int]) -> list[MoveView]:
    dex = env.dex
    out: list[MoveView] = []
    for slot, move_id in enumerate(move_ids):
        data = dex.moves[move_id]
        if own:
            data = resolve_move(env.battle, fighter, data)
        move_type = GEN3_INDEX_TYPE.get(int(data.move_type), "normal")
        out.append(MoveView(data.name, move_type, int(data.power), int(data.accuracy),
                            int(fighter.pp[slot]) if own and slot < len(fighter.pp) else int(data.pp),
                            category_for(move_type, int(data.power), 3), gen3_move_id=int(move_id),
                            max_pp=int(data.pp), priority=int(data.priority)))
    return out


def _mon(env: BattleEnv, fighter: Fighter, own: bool, revealed: set[int] | None = None,
         ability: int = 0) -> MonView:
    dex = env.dex
    species = int(fighter.spec.species)
    types = tuple(dict.fromkeys(GEN3_INDEX_TYPE.get(int(value), "???") for value in fighter.types))
    stages = {name: int(fighter.stages[index + 1]) - 6 for index, name in enumerate(_STAGES)}
    if own:
        moves = _moves(env, fighter, True, list(fighter.move_ids))
        stats = {"atk": fighter.stat(1), "def": fighter.stat(2), "spe": fighter.stat(3), "spa": fighter.stat(4),
                 "spd": fighter.stat(5)}
        return MonView(dex.species[species].name, _national(env, species), int(fighter.spec.level),
                       int(fighter.hp), int(fighter.max_hp), types, status_name(fighter.status), moves, stats,
                       stages=stages, ability=_ABILITY_NAMES.get(int(fighter.ability), ""))
    shown = [move for move in fighter.move_ids if move in (revealed or set())]
    moves = _moves(env, fighter, False, shown)
    # the HP bar: a fraction, expressed out of 100 (the brain reads hp/max_hp as a fraction)
    fraction = fighter.hp / max(1, fighter.max_hp)
    hp = 0 if fighter.hp <= 0 else max(1, round(fraction * 100))
    return MonView(dex.species[species].name, _national(env, species), int(fighter.spec.level), hp, 100, types,
                   status_name(fighter.status), moves, None, stages=stages,
                   ability=_ABILITY_NAMES.get(ability, "") if ability else "")


def view_of(env: BattleEnv, side: int, legal: list[int] | None = None) -> BattleView:
    battle = env.battle
    own, foe = battle.sides[side], battle.sides[1 - side]
    knowledge = env.knowledge[side]
    party = [_mon(env, fighter, True) for fighter in own.party]
    enemy_fighter = foe.fighter
    revealed = knowledge.revealed_moves.get(int(enemy_fighter.spec.species), set())
    ability = knowledge.revealed_abilities.get(int(enemy_fighter.spec.species), 0)
    enemy = _mon(env, enemy_fighter, False, revealed, ability)
    actions = legal if legal is not None else battle.legal_actions(side)
    weather = battle._weather_active()
    view = BattleView("stadium", 3, party[own.active], enemy, party, own.active, is_wild=False,
                      can_switch=any(action >= 4 for action in actions), can_run=False, turn=battle.turn,
                      enemy_remaining=foe.remaining(), weather="" if weather == "none" else weather)
    view.legal_moves = tuple(action for action in actions if action < 4)
    return view


def action_of(env: BattleEnv, side: int, choice: BrainChoice, legal: list[int]) -> int:
    """Engine action of a brain choice (falls back to the first legal action if it is not legal)."""
    own = env.battle.sides[side]
    action = -1
    if choice.kind == "move":
        action = choice.index
    elif choice.kind == "switch":
        bench = own.bench()
        if choice.index in bench:
            action = 4 + bench.index(choice.index)
    if action in legal:
        return action
    moves = [item for item in legal if item < 4]
    return moves[0] if moves else legal[0]


@dataclass
class BrainController:
    """A Stadium controller that plays through ``view_of`` -> ``BattleBrain`` (batched)."""
    brain: BattleBrain
    name: str = "brain"
    wants_obs: bool = False

    def choose(self, decisions: list[Decision], rng: random.Random) -> list[Choice]:
        if not decisions:
            return []
        views = [view_of(item.env, item.side, item.legal) for item in decisions]
        choices = self.brain.choose_many(views)
        return [Choice(action_of(item.env, item.side, choice, item.legal))
                for item, choice in zip(decisions, choices)]


# ----------------------------------------------------------------------------- campaign heuristic port


@dataclass
class CampaignController:
    """Stadium port of the campaign heuristic (see the module docstring)."""
    name: str = "campaign"
    wants_obs: bool = False
    _cache: dict[tuple[int, int], tuple[float, float, float, float, float, int]] = field(default_factory=dict)

    def choose(self, decisions: list[Decision], rng: random.Random) -> list[Choice]:
        return [Choice(self.act(item.env, item.side, item.legal)) for item in decisions]

    # beliefs: mean of the IV 0/15/31 hypotheses (BeliefStateEstimator's uniform prior)
    def _belief(self, env: BattleEnv, fighter: Fighter) -> tuple[float, float, float, float, float, int]:
        key = (int(fighter.spec.species), int(fighter.spec.level))
        if key not in self._cache:
            base = env.dex.species[key[0]].base_stats  # hp, atk, def, spe, spa, spd
            level = max(1, key[1])
            def mean(value: int) -> float:
                return sum(max(1, (2 * value + iv) * level // 100 + 5) for iv in (0, 15, 31)) / 3
            max_hp = 1 if key[0] == 303 else (2 * base[0] + 15) * level // 100 + level + 10
            self._cache[key] = (mean(base[1]), mean(base[2]), mean(base[3]), mean(base[4]), mean(base[5]),
                                max_hp)
        return self._cache[key]

    @staticmethod
    def _speed(raw: float, fighter: Fighter) -> float:
        from core.stadium.engine import STAGE_RATIOS

        numerator, denominator = STAGE_RATIOS[max(0, min(12, int(fighter.stages[3])))]
        return raw * numerator / denominator / (4 if fighter.status & PARALYSIS else 1)

    def _incoming(self, env: BattleEnv, side: int, defender: Fighter, enemy: Fighter) -> float:
        from core.safety.z3_integer_verifier import (DamageContext, GbaIntegerSafetyVerifier, ResidualContext,
                                                     compute_residual_damage)

        dex = env.dex
        atk, _, _, spa, _, _ = self._belief(env, enemy)
        revealed = env.knowledge[side].revealed_moves.get(int(enemy.spec.species), set())
        powers = [dex.moves[move].power * dex.multiplier(dex.moves[move].move_type, defender.types)
                  for move in revealed]
        if not powers:
            powers = [50 * dex.multiplier(attack_type, defender.types)
                      for attack_type in dict.fromkeys(enemy.types) if attack_type != 9]
        estimated_power = int(max(powers, default=50))
        context = DamageContext(max(1, int(enemy.spec.level)), max(1, int(max(atk, spa))),
                                max(1, min(defender.stat(2), defender.stat(5))), estimated_power, stab=True,
                                attack_stage=max(int(enemy.stages[1]), int(enemy.stages[4])),
                                defense_stage=min(int(defender.stages[2]), int(defender.stages[5])),
                                attacker_status=int(enemy.status))
        residual = compute_residual_damage(ResidualContext(max(1, int(defender.max_hp)),
                                                           status=int(defender.status) & 0xFF))
        expected = GbaIntegerSafetyVerifier.compute_context_damage_reference(context, 93) + residual
        return expected / max(1, int(defender.hp))

    def _outgoing(self, env: BattleEnv, side: int, own: Fighter, enemy: Fighter) -> list[object]:
        from core.battle.risk_horizon import MoveDamageEstimate
        from core.safety.z3_integer_verifier import DamageContext, GbaIntegerSafetyVerifier

        dex = env.dex
        _, dfn, _, _, spd, _ = self._belief(env, enemy)
        estimates: list[object] = []
        for slot, move_id in enumerate(own.move_ids[:4]):
            if not move_id or not own.pp[slot]:
                continue
            move = resolve_move(env.battle, own, dex.moves[move_id])
            expected = 0.0
            if move.power > 0 and move.effect != 7:
                special = move.special
                context = DamageContext(max(1, int(own.spec.level)), max(1, own.stat(4 if special else 1)),
                                        max(1, int(spd if special else dfn)), int(move.power),
                                        move_type=int(move.move_type), category="special" if special else "physical",
                                        stab=move.move_type in own.types,
                                        attack_stage=int(own.stages[4 if special else 1]),
                                        defense_stage=int(enemy.stages[5 if special else 2]),
                                        attacker_ability=int(own.ability), attacker_item=int(own.item),
                                        attacker_species=int(own.spec.species), attacker_status=int(own.status),
                                        attacker_hp=max(1, int(own.hp)), attacker_max_hp=max(1, int(own.max_hp)),
                                        defender_species=int(enemy.spec.species), defender_status=int(enemy.status),
                                        defender_ability=env.knowledge[side].revealed_abilities.get(
                                            int(enemy.spec.species), 0))
                neutral = GbaIntegerSafetyVerifier.compute_context_damage_reference(context, 93)
                expected = neutral * dex.multiplier(move.move_type, enemy.types)
            difference = int(own.stages[6]) - int(enemy.stages[7])
            accuracy = min(1.0, (move.accuracy or 100) / 100 * (3 + max(0, difference)) / (3 + max(0, -difference)))
            estimates.append(MoveDamageEstimate(slot, float(expected), accuracy, int(own.pp[slot])))
        return estimates

    def act(self, env: BattleEnv, side: int, legal: list[int]) -> int:
        from core.battle.risk_horizon import MoveDamageEstimate, estimate_attack_plan

        if len(legal) == 1:
            return legal[0]
        battle = env.battle
        own, foe = battle.sides[side], battle.sides[1 - side]
        me, enemy = own.fighter, foe.fighter
        atk, dfn, spe, spa, spd, enemy_max = self._belief(env, enemy)
        enemy_hp = max(0, math.ceil(enemy.hp / max(1, enemy.max_hp) * enemy_max))
        forced = bool(battle._replacing)
        retaliation = self._incoming(env, side, me, enemy) if me.hp > 0 else 1.0
        outgoing = [item for item in self._outgoing(env, side, me, enemy) if isinstance(item, MoveDamageEstimate)]
        acts_first = self._speed(me.stat(3), me) > self._speed(spe, enemy)
        productive = False
        options: list[tuple[float, float, int]] = []
        for action in legal:
            if action >= 4 or forced:
                continue
            selected = next((item for item in outgoing if item.index == action), None)
            if selected is None:
                continue
            damaging = selected.expected_damage > 0
            try:
                horizon = estimate_attack_plan(max(1, me.hp), enemy_hp, retaliation * max(1, me.hp), outgoing,
                                               first_move_index=action if damaging else None, acts_first=acts_first,
                                               initial_delay_turns=0 if damaging else 1)
            except ValueError:
                continue
            productive |= damaging and horizon.can_finish
            score = -selected.expected_damage / max(1, enemy_max) * 4
            options.append((horizon.risk_proxy, score, action))
        bench = own.bench()
        for action in legal:
            if action < 4:
                continue
            replacement = own.party[bench[action - 4]]
            if replacement.hp <= 0:
                continue
            risk = self._incoming(env, side, replacement, enemy)
            replacement_out = [item for item in self._outgoing(env, side, replacement, enemy)
                               if isinstance(item, MoveDamageEstimate)]
            try:
                horizon = estimate_attack_plan(int(replacement.hp), enemy_hp, risk * int(replacement.hp),
                                               replacement_out, switching=True,
                                               acts_first=self._speed(replacement.stat(3), replacement) >
                                               self._speed(spe, enemy))
            except ValueError:
                continue
            if not horizon.can_finish and productive and not forced:
                continue
            options.append((horizon.risk_proxy, 1.0, action))
        if not options:
            moves = [action for action in legal if action < 4]
            return moves[0] if moves else legal[0]
        return min(options, key=lambda option: (option[0], option[1]))[2]


__all__ = ["BrainController", "CampaignController", "action_of", "status_name", "view_of"]
