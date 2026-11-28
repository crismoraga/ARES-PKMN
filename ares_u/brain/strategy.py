"""Explicit, explainable strategic layer of the battle brain.

The layer reads only a ``BattleView`` (public information for the enemy) and scores every option of
the turn with one analytic model, the *duel*: both active Pokemon keep attacking with their best
move and whoever completes the KO first wins.

* Hits needed: the damage formula (STAB, the generation's own type chart, stat stages, burn,
  weather, an expected critical-hit factor) with the 85-100 % roll, under a common-roll
  approximation: ``P(N <= k) = P(u >= hp / (k * top))`` with ``u ~ U[0.85, 1]``.
* Turn of the KO: each turn lands a hit with its own probability (accuracy x the chance to act:
  paralysis 0.75, sleep 0, confusion ~0.5), a Poisson-binomial dynamic program over turns.
* Win probability: ``sum_t P(my KO at t) * (P(no foe KO before t) - P(foe KO at t) * P(foe first))``;
  a stall within the horizon counts 0.4.

Every option of the turn changes the two KO-turn distributions:

* a move: its own first-turn KO chance and damage, then the best move (an immune target is vetoed:
  x0 is never chosen);
* a status move: I lose the turn, the foe's rates/damage change once it lands (sleep 1-4 lost turns
  in Gen 3+, paralysis x0.75 and quartered speed, poison/toxic/burn residual, burn halves physical
  attacks); vetoed on a statused or immune foe, discounted on a foe about to faint;
* a stat boost / heal / stat drop: the duel with the changed stats or HP after spending the turn;
* a switch: the bench member takes the foe's hit on entry, then fights its own duel; it is chosen
  only when the active is badly matched (losing duel) and the member is clearly better;
* run (wild): when the duel is dangerous (we likely lose the active fast and no bench member wins)
  or pointless (long fight against a much lower level), and escape is likely;
* catch (wild): ``kind="item"`` when the wild Pokemon adds a type the party lacks (weaken it
  first with a move that does not KO, or sleep/paralyse it).

Enemy stats are estimated from base stats at its level (IV 15, no EVs) and its moves from what it
revealed plus (Gen 3) the level-up moves it knows at its level, else a STAB prior.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

from core.stadium.dex import Dex

from .api import (STATUS_ES, TYPE_ES, BattleView, MonView, MoveView, canonical_type, category_for, display_name,
                  normalize_name, type_multiplier)
from .moves import MoveFacts, describe

T_MAX = 16
ROLL_LOW = 0.85
STALL_VALUE = 0.4
SWITCH_MARGIN = 0.15        # a switch must beat staying by this much win probability
ACTIVE_LOSING = 0.40        # ... and the active must be losing its duel
PRIOR_POWER_MIN, PRIOR_POWER_MAX = 40, 85
STAT_KEYS = ("atk", "def", "spa", "spd", "spe")
_STAT_ES = {"atk": "Ataque", "def": "Defensa", "spa": "At. Esp.", "spd": "Def. Esp.", "spe": "Velocidad",
            "acc": "Precisión", "eva": "Evasión"}
_GROUND_IMMUNE = frozenset({"LEVITATE", "LEVITACION"})
_ABSORB = {"VOLTABSORB": "electric", "ABSORBEELEC": "electric", "WATERABSORB": "water", "ABSORBEAGUA": "water",
           "FLASHFIRE": "fire", "ABSORBEFUEGO": "fire", "MOTORDRIVE": "electric", "SAPSIPPER": "grass",
           "DRYSKIN": "water"}
_WONDER_GUARD = frozenset({"WONDERGUARD", "SUPERGUARDA"})
_THICK_FAT = frozenset({"THICKFAT", "SEBO"})
_SLEEP_IMMUNE = frozenset({"INSOMNIA", "VITALSPIRIT", "ESPIRITUVITAL"})
_PAR_IMMUNE = frozenset({"LIMBER", "FLEXIBILIDAD"})
_PSN_IMMUNE = frozenset({"IMMUNITY", "INMUNIDAD"})
_BRN_IMMUNE = frozenset({"WATERVEIL", "VELOAGUA"})
# Gen-3 ability ids (``SpeciesData.abilities``) -> normalized English names used above.
_GEN3_ABILITY: dict[int, str] = {5: "STURDY", 7: "LIMBER", 10: "VOLTABSORB", 11: "WATERABSORB", 15: "INSOMNIA",
                                 17: "IMMUNITY", 18: "FLASHFIRE", 25: "WONDERGUARD", 26: "LEVITATE",
                                 31: "LIGHTNINGROD", 41: "WATERVEIL", 47: "THICKFAT", 72: "VITALSPIRIT"}
Rates = tuple[float, ...]


def stage_multiplier(stage: int) -> float:
    stage = max(-6, min(6, int(stage)))
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def accuracy_multiplier(stage: int) -> float:
    stage = max(-6, min(6, int(stage)))
    return (3 + stage) / 3 if stage >= 0 else 3 / (3 - stage)


# ----------------------------------------------------------------------------- duel model


def constant_rates(rate: float, lost: int = 0) -> Rates:
    """Per-turn landing probabilities: ``lost`` turns at 0 then ``rate``."""
    rate = max(0.0, min(1.0, rate))
    return tuple(0.0 if turn < lost else rate for turn in range(T_MAX))


@dataclass(frozen=True)
class Profile:
    """One side's attacking profile in a duel (absolute HP units of the target)."""
    top: float                 # top-roll damage of a landed hit, expected critical factor included
    rates: Rates               # P(landed hit) on turns 1..T_MAX
    residual: float = 0.0      # damage per turn to the target independent of hits (poison, burn)

    def later(self, turns: int = 1) -> Profile:
        rates = self.rates[turns:] + (self.rates[-1],) * turns
        return Profile(self.top, rates, self.residual)


def _hits_pmf(target_hp: float, top: float) -> list[float]:
    """P(N = n), n = 1..T_MAX, landed hits needed under the common-roll approximation."""
    if target_hp <= 0:
        return [1.0] + [0.0] * (T_MAX - 1)
    if top <= 0:
        return [0.0] * T_MAX
    ratio = target_hp / top
    width = 1.0 - ROLL_LOW
    pmf, previous = [], 0.0
    for hits in range(1, T_MAX + 1):
        x = ratio / hits
        cdf = 1.0 if x <= ROLL_LOW else 0.0 if x > 1.0 else (1.0 - x) / width
        cdf = max(previous, cdf)
        pmf.append(cdf - previous)
        previous = cdf
    return pmf


@lru_cache(maxsize=65536)
def _ko_cached(ratio_key: int, rates: Rates) -> tuple[float, ...]:
    pmf = _hits_pmf(ratio_key / 1000, 1.0)
    counts = [1.0] + [0.0] * T_MAX  # P(k landed hits so far)
    out = [0.0] * T_MAX
    for turn in range(T_MAX):
        rate = rates[turn]
        if rate <= 0:
            continue
        for hits in range(1, T_MAX + 1):
            if pmf[hits - 1] > 0:
                out[turn] += pmf[hits - 1] * counts[hits - 1] * rate
        for hits in range(T_MAX, 0, -1):
            counts[hits] = counts[hits] * (1 - rate) + counts[hits - 1] * rate
        counts[0] *= 1 - rate
    return tuple(out)


def ko_turns(target_hp: float, profile: Profile) -> list[float]:
    """P(the KO lands on turn t), t = 1..T_MAX (missing mass = no KO within the horizon)."""
    if target_hp <= 0:
        return [1.0] + [0.0] * (T_MAX - 1)
    top = profile.top
    mean_rate = sum(profile.rates) / len(profile.rates)
    if profile.residual > 0 and mean_rate > 0:
        top += profile.residual / (mean_rate * (1 + ROLL_LOW) / 2)
    if top <= 0 or mean_rate <= 0:
        return [0.0] * T_MAX
    ratio = min(T_MAX * 2.0, target_hp / top)
    rates = tuple(round(rate, 3) for rate in profile.rates)
    return list(_ko_cached(int(round(ratio * 1000)), rates))


def duel(mine: list[float], theirs: list[float], first_now: float, first_later: float) -> float:
    """Win probability from the two KO-turn distributions and the turn order (now / later turns)."""
    alive = 1.0  # P(the foe has not KOed me before turn t)
    win = 0.0
    for turn in range(T_MAX):
        first = first_now if turn == 0 else first_later
        win += mine[turn] * max(0.0, alive - theirs[turn] * (1 - first))
        alive -= theirs[turn]
    stall = max(0.0, 1.0 - sum(mine) - sum(theirs))
    return max(0.0, min(1.0, win + STALL_VALUE * stall))


def win_probability(mine: Profile, enemy_hp: float, theirs: Profile, my_hp: float, p_first: float) -> float:
    return duel(ko_turns(enemy_hp, mine), ko_turns(my_hp, theirs), p_first, p_first)


def expected_ko_turn(target_hp: float, profile: Profile) -> float:
    dist = ko_turns(target_hp, profile)
    mass = sum(dist)
    if mass <= 1e-9:
        return float("inf")
    return sum((index + 1) * value for index, value in enumerate(dist)) / mass


# ----------------------------------------------------------------------------- combatants


@lru_cache(maxsize=1)
def _national_to_internal() -> dict[int, int]:
    raw = json.loads((Path(__file__).with_name("national_dex.json")).read_text(encoding="utf-8"))
    mapping: dict[int, int] = {}
    for internal, national in enumerate(raw["internal_to_national"]):
        if national and int(national) not in mapping:
            mapping[int(national)] = internal
    return mapping


def internal_species(national: int) -> int | None:
    """Stadium (Gen-3 internal) species id of a national dex number, ``None`` beyond #386."""
    return _national_to_internal().get(int(national))


@lru_cache(maxsize=1)
def _internal_to_national() -> tuple[int, ...]:
    raw = json.loads((Path(__file__).with_name("national_dex.json")).read_text(encoding="utf-8"))
    return tuple(int(value) for value in raw["internal_to_national"])


def national_number(internal: int) -> int:
    """National dex number of a Stadium (Gen-3 internal) species id (0 for placeholders)."""
    table = _internal_to_national()
    return table[internal] if 0 <= internal < len(table) else 0


@dataclass
class Combatant:
    mon: MonView
    stats: dict[str, float]
    max_hp: float
    hp: float
    types: tuple[str, ...]
    moves: list[MoveFacts]
    stages: dict[str, int]
    status: str
    abilities: frozenset[str]   # normalized names it may have (the known one, else the species' set)
    ability_known: bool
    level: int

    @property
    def alive(self) -> bool:
        return self.hp > 0

    def speed(self, generation: int) -> float:
        value = self.stats["spe"] * stage_multiplier(self.stages.get("spe", 0))
        if self.status == "par":
            value *= 0.5 if generation >= 7 else 0.25
        return value


@dataclass(frozen=True)
class Hit:
    """One move of an attacker against a defender."""
    move: MoveFacts
    slot: int
    multiplier: float
    top: float          # top-roll damage of a landed hit (absolute HP, expected critical factor)
    acc: float          # hit probability
    mean: float         # mean damage of a landed hit, capped at the target's HP
    ko_now: float       # probability that this single use KOs the target (accuracy included)
    immune: bool

    @property
    def expected(self) -> float:
        return self.mean * self.acc


@dataclass
class Option:
    kind: str           # move | switch | run | item
    index: int
    label: str
    value: float
    legal: bool = True
    veto: str = ""
    multiplier: float = 1.0
    damage: float = 0.0     # expected damage this turn, fraction of the foe's current HP
    ko_now: float = 0.0
    reason: str = ""
    details: dict[str, float] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.legal and not self.veto


@dataclass
class Analysis:
    view: BattleView
    options: list[Option]
    me: Combatant | None
    enemy: Combatant
    threat: Hit | None           # the foe's best move against my active
    p_first: float
    stay_value: float            # win probability of the best move
    forced_switch: bool
    notes: list[str] = field(default_factory=list)

    def option(self, kind: str, index: int) -> Option | None:
        return next((item for item in self.options if item.kind == kind and item.index == index), None)

    def best(self) -> Option:
        usable = [item for item in self.options if item.usable]
        pool = usable or [item for item in self.options if item.legal] or self.options
        return max(pool, key=lambda item: (item.value, item.damage, -item.index))


class Strategist:
    """Analytic duel model over a ``BattleView``; ``analyze`` scores every option of the turn."""

    def __init__(self, dex: Dex | None = None) -> None:
        if dex is None:
            from ares_u.battle.teams import default_dex

            dex = default_dex()
        self.dex = dex

    # ------------------------------------------------------------------ estimates
    def base_stats(self, mon: MonView) -> tuple[int, int, int, int, int, int] | None:
        internal = internal_species(mon.national_dex) if mon.national_dex else None
        if internal is None or internal >= len(self.dex.species):
            return None
        return self.dex.species[internal].base_stats  # hp, atk, def, spe, spa, spd

    def estimate_stats(self, mon: MonView, generation: int) -> tuple[dict[str, float], float]:
        """Stats (atk, def, spa, spd, spe) and max HP; known stats win, else base stats at IV 15."""
        level = max(1, int(mon.level))
        base = self.base_stats(mon) or (70, 70, 70, 70, 70, 70)
        hp_base, atk, dfn, spe, spa, spd = base
        if generation <= 1:
            spd = spa  # one Special stat in Gen 1 (the Gen-2 Sp. Atk kept the Gen-1 Special)

        def stat(value: int) -> float:
            return float((2 * value + 15) * level // 100 + 5)

        estimated = {"atk": stat(atk), "def": stat(dfn), "spa": stat(spa), "spd": stat(spd), "spe": stat(spe)}
        max_hp = float((2 * hp_base + 15) * level // 100 + level + 10)
        if mon.stats:
            for key in STAT_KEYS:
                if mon.stats.get(key, 0) > 0:
                    estimated[key] = float(mon.stats[key])
        return estimated, max_hp

    def possible_abilities(self, mon: MonView) -> tuple[frozenset[str], bool]:
        if mon.ability:
            return frozenset({normalize_name(mon.ability)}), True
        internal = internal_species(mon.national_dex) if mon.national_dex else None
        if internal is None or internal >= len(self.dex.species):
            return frozenset(), False
        names = {_GEN3_ABILITY.get(int(ability), f"ID{int(ability)}")
                 for ability in self.dex.species[internal].abilities if int(ability)}
        return frozenset(names), False

    def predicted_moves(self, mon: MonView, generation: int) -> list[MoveFacts]:
        """Revealed moves, plus (Gen 3) the level-up moves it knows at its level, else a STAB prior."""
        revealed = [describe(move, generation) for move in mon.moves]
        known_ids = {move.gen3_id for move in revealed if move.gen3_id is not None}
        predicted: list[MoveFacts] = []
        internal = internal_species(mon.national_dex) if mon.national_dex else None
        if generation == 3 and internal is not None and len(revealed) < 4:
            for move_id in self.dex.learnable(internal, max(1, int(mon.level)))[-4:]:
                if move_id in known_ids:
                    continue
                data = self.dex.moves[move_id]
                view = MoveView(data.name, "", int(data.power), int(data.accuracy), int(data.pp), "",
                                gen3_move_id=int(move_id))
                predicted.append(describe(view, generation))
        moves = revealed + predicted
        if not any(move.damaging for move in moves):
            power = max(PRIOR_POWER_MIN, min(PRIOR_POWER_MAX, 30 + int(mon.level)))
            for name in dict.fromkeys(canonical_type(item) for item in mon.types):
                if name == "???":
                    continue
                moves.append(MoveFacts(name=f"({TYPE_ES.get(name, name)})", type=name, power=power, accuracy=100,
                                       category=category_for(name, power, generation, None), priority=0, effect=0,
                                       gen3_id=None))
        return moves

    def combatant(self, mon: MonView, generation: int, own: bool) -> Combatant:
        stats, est_max = self.estimate_stats(mon, generation)
        if own:
            max_hp, hp = float(max(1, mon.max_hp)), float(max(0, mon.hp))
            moves = [describe(move, generation) for move in mon.moves]
        else:
            max_hp = est_max
            hp = max(1.0, mon.hp_fraction * est_max) if mon.hp > 0 else 0.0
            moves = self.predicted_moves(mon, generation)
        abilities, known = self.possible_abilities(mon)
        stages = {key: int(value) for key, value in (mon.stages or {}).items()}
        types = tuple(dict.fromkeys(canonical_type(item) for item in mon.types)) or ("???",)
        return Combatant(mon, stats, max_hp, hp, types, moves, stages, mon.status, abilities, known, int(mon.level))

    # ------------------------------------------------------------------ damage
    @staticmethod
    def _ability_factor(move_type: str, defender: Combatant, multiplier: float) -> float:
        """Damage factor from the defender's possible abilities (share of them that block it)."""
        if not defender.abilities:
            return 1.0
        blocked = 0.0
        for name in defender.abilities:
            if (move_type == "ground" and name in _GROUND_IMMUNE) or _ABSORB.get(name) == move_type or (
                    name in _WONDER_GUARD and multiplier <= 1.0):
                blocked += 1.0
            elif name in _THICK_FAT and move_type in ("fire", "ice"):
                blocked += 0.5
        return (len(defender.abilities) - blocked) / len(defender.abilities)

    @staticmethod
    def accuracy(attacker: Combatant, move: MoveFacts, defender: Combatant) -> float:
        if move.accuracy <= 0 or move.accuracy > 100:
            return 1.0
        ratio = accuracy_multiplier(attacker.stages.get("acc", 0) - defender.stages.get("eva", 0))
        return max(0.0, min(1.0, move.accuracy / 100 * ratio))

    def hit(self, attacker: Combatant, slot: int, move: MoveFacts, defender: Combatant, generation: int,
            weather: str = "") -> Hit:
        multiplier = type_multiplier(generation, move.type, defender.types)
        acc = self.accuracy(attacker, move, defender)
        if not move.damaging:
            return Hit(move, slot, multiplier, 0.0, acc, 0.0, 0.0, False)
        ability = self._ability_factor(move.type, defender, multiplier)
        if multiplier == 0 or ability <= 0:
            return Hit(move, slot, 0.0, 0.0, acc, 0.0, 0.0, True)
        if move.ohko:
            if attacker.level < defender.level:
                return Hit(move, slot, multiplier, 0.0, 0.0, 0.0, 0.0, False)
            chance = min(1.0, (30 + attacker.level - defender.level) / 100) * ability
            return Hit(move, slot, multiplier, defender.hp * 1.2, chance, defender.hp, chance, False)
        if move.fixed:
            amount = {"40": 40.0, "20": 20.0, "level": float(attacker.level), "half": max(1.0, defender.hp / 2),
                      "endeavor": max(0.0, defender.hp - attacker.hp)}.get(move.fixed, 0.0) * ability
            mean = min(defender.hp, amount)
            ko = acc if amount >= defender.hp > 0 else 0.0
            # a fixed amount has no roll: express it as a top roll whose whole range equals it
            return Hit(move, slot, multiplier, amount / ((1 + ROLL_LOW) / 2), acc, mean, ko, False)
        special = move.category == "special"
        atk_key, def_key = ("spa", "spd") if special else ("atk", "def")
        attack = attacker.stats[atk_key] * stage_multiplier(attacker.stages.get(atk_key, 0))
        defense = max(1.0, defender.stats[def_key] * stage_multiplier(defender.stages.get(def_key, 0)))
        base = (2 * attacker.level / 5 + 2) * move.power * attack / defense / 50
        if attacker.status == "brn" and not special:
            base *= 0.5
        base += 2
        if move.type in attacker.types:
            base *= 1.5
        base *= multiplier * ability
        if weather in ("rain", "lluvia"):
            base *= 1.5 if move.type == "water" else 0.5 if move.type == "fire" else 1.0
        elif weather in ("sun", "sol"):
            base *= 1.5 if move.type == "fire" else 0.5 if move.type == "water" else 1.0
        crit_chance, crit_mult = (1 / 24, 1.5) if generation >= 7 else (1 / 16, 1.5 if generation >= 6 else 2.0)
        top = base * move.hits
        if move.charge and weather not in ("sun", "sol"):
            top *= 0.5  # two turns per hit
        if move.recharge:
            top *= 0.6  # the recharge turn slows it down (a KO skips it)
        if move.self_ko:
            top *= 0.5  # it costs our own Pokemon: only a sure KO justifies it
        top_crit = top * (1 + crit_chance * (crit_mult - 1))
        mean = min(defender.hp, top_crit * (1 + ROLL_LOW) / 2) if defender.hp > 0 else 0.0
        ko = 0.0
        if defender.hp > 0:
            ko = acc * ((1 - crit_chance) * _roll_share(top, defender.hp) +
                        crit_chance * _roll_share(top * crit_mult, defender.hp))
        return Hit(move, slot, multiplier, top_crit, acc, mean, ko, False)

    def best_hit(self, attacker: Combatant, defender: Combatant, generation: int, weather: str,
                 usable: list[int] | None = None) -> Hit | None:
        hits = [self.hit(attacker, slot, move, defender, generation, weather)
                for slot, move in enumerate(attacker.moves)
                if move.damaging and (usable is None or slot in usable)]
        hits = [item for item in hits if not item.immune and item.top > 0 and item.acc > 0]
        if not hits:
            return None
        return max(hits, key=lambda item: (item.expected, item.ko_now))

    # ------------------------------------------------------------------ duel helpers
    def p_first(self, mine: Combatant, theirs: Combatant, generation: int, my_priority: int = 0,
                their_priority: int = 0) -> float:
        if my_priority != their_priority:
            return 1.0 if my_priority > their_priority else 0.0
        ratio = mine.speed(generation) / max(1.0, theirs.speed(generation))
        if ratio >= 1.08:
            return 1.0
        if ratio <= 0.92:
            return 0.0
        return (ratio - 0.92) / 0.16

    @staticmethod
    def action_rate(mon: Combatant) -> float:
        return {"par": 0.75, "frz": 0.2}.get(mon.status, 1.0)

    @staticmethod
    def sleep_turns(generation: int) -> tuple[int, ...]:
        """Turns a freshly slept Pokemon cannot act (Gen 3+: 1-4; Gen 1: 1-7; Gen 2: 1-6)."""
        return (1, 2, 3, 4) if generation >= 3 else tuple(range(1, 8)) if generation == 1 else tuple(range(1, 7))

    def rates_for(self, mon: Combatant, hit: Hit | None, generation: int, lost: int = 0) -> list[tuple[Rates, float]]:
        """Mixture of per-turn landing rates of ``mon`` attacking with ``hit``."""
        if hit is None:
            return [(constant_rates(0.0), 1.0)]
        rate = hit.acc * self.action_rate(mon)
        if mon.status == "slp":
            # already asleep: the counter is hidden, assume 1-3 more lost turns
            return [(constant_rates(rate, lost + extra), 1 / 3) for extra in (1, 2, 3)]
        return [(constant_rates(rate, lost), 1.0)]

    @staticmethod
    def residual_of(mon: Combatant, generation: int) -> float:
        if mon.status in ("psn", "brn"):
            return mon.max_hp / (16 if generation == 1 else 8)
        if mon.status == "tox":
            return mon.max_hp * 2.5 / 16
        return 0.0

    def ko_mix(self, target_hp: float, top: float, mixture: list[tuple[Rates, float]], residual: float) -> list[float]:
        out = [0.0] * T_MAX
        for rates, weight in mixture:
            for index, value in enumerate(ko_turns(target_hp, Profile(top, rates, residual))):
                out[index] += weight * value
        return out

    def foe_ko(self, enemy: Combatant, me: Combatant, threat: Hit | None, generation: int, lost: int = 0,
               scale: float = 1.0, rate_scale: float = 1.0) -> list[float]:
        """KO-turn distribution of the foe attacking ``me`` with its best move."""
        if threat is None:
            return [0.0] * T_MAX
        mixture = [(tuple(rate * rate_scale for rate in rates), weight)
                   for rates, weight in self.rates_for(enemy, threat, generation, lost)]
        return self.ko_mix(me.hp, threat.top * scale, mixture, self.residual_of(me, generation))

    def my_ko(self, me: Combatant, enemy: Combatant, best: Hit | None, generation: int, lost: int = 0,
              extra_residual: float = 0.0, enemy_hp: float | None = None) -> list[float]:
        if best is None:
            residual = self.residual_of(enemy, generation) + extra_residual
            if residual <= 0:
                return [0.0] * T_MAX
            return ko_turns(enemy.hp if enemy_hp is None else enemy_hp,
                            Profile(0.0, constant_rates(1.0, lost), residual))
        mixture = self.rates_for(me, best, generation, lost)
        return self.ko_mix(enemy.hp if enemy_hp is None else enemy_hp, best.top, mixture,
                           self.residual_of(enemy, generation) + extra_residual)

    # ------------------------------------------------------------------ analysis
    def analyze(self, view: BattleView) -> Analysis:
        generation = view.generation
        weather = view.weather.lower() if view.weather else ""
        enemy = self.combatant(view.enemy, generation, own=False)
        forced = view.me.hp <= 0
        me = None if forced else self.combatant(view.me, generation, own=True)
        bench = [(index, self.combatant(mon, generation, own=True)) for index, mon in enumerate(view.party)
                 if index != view.active_index and mon.hp > 0]
        options: list[Option] = []
        notes: list[str] = []
        threat = None
        stay = 0.0
        p_first = 0.5
        if me is not None and enemy.alive:
            threat = self.best_hit(enemy, me, generation, weather)
            p_first = self.p_first(me, enemy, generation)
            options.extend(self._move_options(view, me, enemy, threat, generation, weather))
            stay = max((item.value for item in options if item.kind == "move" and item.usable), default=0.0)
        elif me is not None:
            for slot, move in enumerate(me.moves):
                options.append(Option("move", slot, display_name(move.name), 0.5 + 0.001 * move.power,
                                      legal=view.me.moves[slot].pp > 0,
                                      reason=f"{display_name(move.name)}: el rival ya está debilitado"))
        if enemy.alive and (view.can_switch or forced):
            options.extend(self._switch_options(view, me, enemy, threat, bench, stay, generation, weather, forced))
        elif forced:
            for index, member in bench:
                options.append(Option("switch", index, display_name(member.mon.species), member.hp / member.max_hp,
                                      reason=f"Sale {display_name(member.mon.species)} (relevo obligado)"))
        if me is not None and view.is_wild and enemy.alive:
            catch = self._catch_option(view, enemy, options)
            if catch is not None:
                options.append(catch)
            run = self._run_option(view, me, enemy, threat, options, stay, generation)
            if run is not None:
                options.append(run)
        if not any(item.legal for item in options):
            options.append(Option("move", 0, "Forcejeo", 0.0, reason="Sin PP: Forcejeo"))
        if threat is not None and me is not None:
            notes.append(f"{display_name(view.enemy.species)} amenaza con {display_name(threat.move.name)} "
                         f"({threat.expected / max(1.0, me.hp):.0%} de nuestros PS por turno)")
        return Analysis(view, options, me, enemy, threat, p_first, stay, forced, notes)

    # ------------------------------------------------------------------ moves
    @staticmethod
    def usable_slots(view: BattleView) -> list[int]:
        slots = [slot for slot, move in enumerate(view.me.moves) if move.pp > 0]
        legal = getattr(view, "legal_moves", None)
        if legal is not None:
            slots = [slot for slot in slots if slot in legal]
        return slots

    def _move_options(self, view: BattleView, me: Combatant, enemy: Combatant, threat: Hit | None,
                      generation: int, weather: str) -> list[Option]:
        usable = self.usable_slots(view)
        best = self.best_hit(me, enemy, generation, weather, usable)
        their_priority = threat.move.priority if threat is not None else 0
        first_later = self.p_first(me, enemy, generation, best.move.priority if best else 0, their_priority)
        theirs = self.foe_ko(enemy, me, threat, generation)
        options: list[Option] = []
        for slot, move in enumerate(me.moves):
            label = display_name(move.name)
            if slot not in usable:
                options.append(Option("move", slot, label, 0.0, legal=False, reason=f"{label}: sin PP o bloqueado"))
                continue
            hit = self.hit(me, slot, move, enemy, generation, weather)
            first = self.p_first(me, enemy, generation, move.priority, their_priority)
            if move.damaging:
                options.append(self._attack_option(view, me, enemy, hit, best, theirs, first, first_later,
                                                   generation))
            else:
                options.append(self._utility_option(view, me, enemy, slot, move, hit, best, threat, theirs, first,
                                                    first_later, generation, weather))
        return options

    def _attack_option(self, view: BattleView, me: Combatant, enemy: Combatant, hit: Hit, best: Hit | None,
                       theirs: list[float], first: float, first_later: float, generation: int) -> Option:
        label = display_name(hit.move.name)
        enemy_name = display_name(view.enemy.species)
        type_es = TYPE_ES.get(hit.move.type, hit.move.type)
        if hit.immune:
            return Option("move", hit.slot, label, 0.0, veto=f"{enemy_name} es inmune (x0)", multiplier=0.0,
                          reason=f"Evita {label}: {enemy_name} es inmune a {type_es} (x0)")
        if hit.move.effect == 8 and view.enemy.status != "slp":
            return Option("move", hit.slot, label, 0.0, veto="Comesueños necesita un rival dormido",
                          reason=f"Evita {label}: solo funciona si el rival duerme")
        rate_now = hit.acc * self.action_rate(me)
        if me.status == "slp":
            rate_now = 0.0
        p_ko = hit.ko_now * (rate_now / hit.acc if hit.acc > 0 else 0.0)
        p_land = rate_now
        follow = best if best is not None else hit
        remaining = max(1.0, enemy.hp - hit.mean)
        after_land = self.my_ko(me, enemy, follow, generation, enemy_hp=remaining)
        after_miss = self.my_ko(me, enemy, follow, generation)
        mine = [p_ko] + [0.0] * (T_MAX - 1)
        land_no_ko = max(0.0, p_land - p_ko)
        miss = max(0.0, 1.0 - p_land)
        for turn in range(1, T_MAX):
            mine[turn] = land_no_ko * after_land[turn - 1] + miss * after_miss[turn - 1]
        value = duel(mine, theirs, first, first_later)
        damage = hit.expected / max(1.0, enemy.hp)
        reason = self._attack_reason(view, me, enemy, hit, value, first, follow)
        return Option("move", hit.slot, label, value + 0.01 * min(1.0, damage), multiplier=hit.multiplier,
                      damage=damage, ko_now=p_ko, reason=reason,
                      details={"win": round(value, 4), "damage": round(damage, 4), "ko_now": round(p_ko, 4),
                               "first": round(first, 3)})

    def _attack_reason(self, view: BattleView, me: Combatant, enemy: Combatant, hit: Hit, value: float,
                       first: float, follow: Hit) -> str:
        label = display_name(hit.move.name)
        enemy_name = display_name(view.enemy.species)
        attack_type = TYPE_ES.get(hit.move.type, hit.move.type)
        defense_types = "/".join(TYPE_ES.get(name, name) for name in enemy.types)
        turns = expected_ko_turn(enemy.hp, Profile(follow.top, constant_rates(follow.acc)))
        if hit.ko_now >= 0.5:
            turns = 1.0
        rounded = max(1, round(turns)) if math.isfinite(turns) else 0
        turns_text = f" en {rounded} turno{'s' if rounded != 1 else ''}" if rounded else ""
        verdict = "gana el intercambio" if value >= 0.5 else "duelo difícil"
        if hit.ko_now >= 0.6 and first >= 0.5:
            return f"{label} debilita a {enemy_name} este turno ({hit.ko_now:.0%}) y actuamos primero"
        if hit.multiplier > 1:
            return (f"{attack_type} es súper eficaz contra {defense_types} (x{hit.multiplier:g}): {verdict}"
                    f"{turns_text} ({value:.0%})")
        share = hit.expected / max(1.0, enemy.hp)
        if 0 < hit.multiplier < 1:
            return (f"{label} ({attack_type}) es poco eficaz contra {defense_types} (x{hit.multiplier:g}), pero es "
                    f"el mejor daño disponible ({share:.0%} de sus PS); {verdict} ({value:.0%})")
        return f"{label}: {share:.0%} de sus PS por turno, KO{turns_text}; {verdict} ({value:.0%})"

    def _utility_option(self, view: BattleView, me: Combatant, enemy: Combatant, slot: int, move: MoveFacts,
                        hit: Hit, best: Hit | None, threat: Hit | None, theirs: list[float], first: float,
                        first_later: float, generation: int, weather: str) -> Option:
        label = display_name(move.name)
        enemy_name = display_name(view.enemy.species)
        land = hit.acc * (0.0 if me.status == "slp" else self.action_rate(me))
        mine_wasted = self.my_ko(me, enemy, best, generation, lost=1)
        wasted = duel(mine_wasted, theirs, first, first_later)
        status = move.inflicts
        if status in ("slp", "par", "psn", "tox", "brn", "yawn", "cnf"):
            veto = self._status_veto(view, enemy, move, generation)
            if veto:
                return Option("move", slot, label, wasted * 0.5, veto=veto, reason=f"Evita {label}: {veto}")
            success = self._status_value(me, enemy, best, threat, status, first, first_later, generation,
                                         mine_wasted)
            value = land * success + (1 - land) * wasted
            if best is not None and best.ko_now >= 0.5:
                value = min(value, wasted)  # a foe about to faint does not deserve a status turn
            status_es = {"yawn": "sueño (Bostezo)", "cnf": "confusión"}.get(status, STATUS_ES.get(status, status))
            threat_text = (f"nos quita {threat.expected / max(1.0, me.hp):.0%} de PS por turno" if threat is not None
                           else "sin amenaza conocida")
            reason = f"{label}: {status_es} a {enemy_name} ({threat_text}); duelo {wasted:.0%} -> {value:.0%}"
            return Option("move", slot, label, value, reason=reason,
                          details={"win": round(value, 4), "status_success": round(success, 4),
                                   "hit": round(land, 3)})
        if move.heal > 0:
            heal = (me.max_hp - me.hp) if status == "rest" else min(me.max_hp - me.hp, move.heal * me.max_hp)
            if heal <= 0.05 * me.max_hp:
                return Option("move", slot, label, wasted * 0.8, reason=f"{label}: los PS ya están casi al máximo")
            healed = replace(me, hp=me.hp + heal, status="slp" if status == "rest" else me.status)
            # if the foe moves first it hits before the heal; the heal then restores that too
            theirs_healed = self.foe_ko(enemy, healed, threat, generation)
            lost = 3 if status == "rest" else 1
            mine = self.my_ko(me, enemy, best, generation, lost=lost)
            q_now = theirs[0] * (1 - first)
            value = (1 - q_now) * duel(mine, [0.0] + theirs_healed[:-1], first, first_later)
            reason = f"{label}: recupera {heal / me.max_hp:.0%} de PS (duelo {wasted:.0%} -> {value:.0%})"
            return Option("move", slot, label, value, reason=reason, details={"win": round(value, 4)})
        if move.boosts or move.drops:
            if move.effect == 142 and me.hp <= me.max_hp / 2:
                return Option("move", slot, label, 0.0, veto="no quedan PS para Tambor",
                              reason=f"Evita {label}: no quedan PS suficientes")
            user, target = me, enemy
            if move.boosts:
                user = replace(me, stages=dict(me.stages))
                for stat, amount in move.boosts:
                    user.stages[stat] = max(-6, min(6, user.stages.get(stat, 0) + amount))
                if move.effect == 142:
                    user.hp = me.hp - me.max_hp / 2
            else:
                target = replace(enemy, stages=dict(enemy.stages))
                for stat, amount in move.drops:
                    target.stages[stat] = max(-6, min(6, target.stages.get(stat, 0) - amount))
            usable = self.usable_slots(view)
            new_best = self.best_hit(user, target, generation, weather, usable)
            new_threat = self.best_hit(target, user, generation, weather)
            new_first = self.p_first(user, target, generation, new_best.move.priority if new_best else 0,
                                     new_threat.move.priority if new_threat else 0)
            mine = self.my_ko(user, target, new_best, generation, lost=1)
            theirs_new = self.foe_ko(target, user, new_threat, generation)
            # the foe's attack this turn (if first) uses the old numbers
            theirs_mixed = [theirs[0] * (1 - first) + theirs_new[0] * first] + theirs_new[1:]
            success = duel(mine, theirs_mixed, first, new_first)
            value = land * success + (1 - land) * wasted
            changes = move.boosts or tuple((stat, -amount) for stat, amount in move.drops)
            stats_text = ", ".join(f"{_STAT_ES.get(stat, stat)} {'+' if amount > 0 else ''}{amount}"
                                   for stat, amount in changes)
            who = "" if move.boosts else f" de {enemy_name}"
            reason = f"{label}: {stats_text}{who}; duelo {wasted:.0%} -> {value:.0%}"
            return Option("move", slot, label, value, reason=reason, details={"win": round(value, 4)})
        return Option("move", slot, label, wasted * 0.9, reason=f"{label}: sin efecto útil modelado; mejor atacar",
                      details={"win": round(wasted * 0.9, 4)})

    def _status_veto(self, view: BattleView, enemy: Combatant, move: MoveFacts, generation: int) -> str:
        status = move.inflicts
        if status in ("slp", "par", "psn", "tox", "brn", "yawn") and view.enemy.status:
            return f"{display_name(view.enemy.species)} ya tiene {STATUS_ES.get(view.enemy.status, view.enemy.status)}"
        types = set(enemy.types)
        if status in ("psn", "tox") and types & {"poison", "steel"}:
            return "los tipos Veneno y Acero no se envenenan"
        if status == "brn" and "fire" in types:
            return "el tipo Fuego no se quema"
        if status == "par" and generation >= 6 and "electric" in types:
            return "el tipo Eléctrico no se paraliza"
        if status == "par" and move.type == "electric" and type_multiplier(generation, "electric", tuple(types)) == 0:
            return f"{display_name(move.name)} no afecta al tipo Tierra"
        if generation >= 6 and move.gen3_id in (77, 78, 79, 147) and "grass" in types:
            return "el tipo Planta ignora los polvos"
        abilities = enemy.abilities if enemy.ability_known else frozenset()
        if status in ("slp", "yawn") and abilities & _SLEEP_IMMUNE:
            return "su habilidad impide dormir"
        if status == "par" and abilities & _PAR_IMMUNE:
            return "su habilidad impide la parálisis"
        if status in ("psn", "tox") and abilities & _PSN_IMMUNE:
            return "su habilidad impide el veneno"
        if status == "brn" and abilities & _BRN_IMMUNE:
            return "su habilidad impide las quemaduras"
        return ""

    def _status_value(self, me: Combatant, enemy: Combatant, best: Hit | None, threat: Hit | None, status: str,
                      first: float, first_later: float, generation: int, mine_wasted: list[float]) -> float:
        """Duel value once this turn's status move landed (I spent the turn)."""
        if threat is None:
            return duel(mine_wasted, [0.0] * T_MAX, first, first_later)
        theirs_plain = self.foe_ko(enemy, me, threat, generation)
        if status in ("slp", "yawn"):
            outcomes = []
            for lost in self.sleep_turns(generation):
                if status == "yawn":  # drowsy: it still acts this turn and the next one
                    outcomes.append(duel(mine_wasted, self._sleep_ko(enemy, me, threat, generation, 2, lost),
                                         first, first_later))
                    continue
                # if it moved first it already attacked this turn and sleeps from the next one
                now = self._sleep_ko(enemy, me, threat, generation, 0, lost)
                later = self._sleep_ko(enemy, me, threat, generation, 1, lost)
                outcomes.append(first * duel(mine_wasted, now, first, first_later) +
                                (1 - first) * duel(mine_wasted, later, first, first_later))
            return sum(outcomes) / len(outcomes)
        if status == "par":
            slowed = replace(enemy, status="par")
            new_first = self.p_first(me, slowed, generation, best.move.priority if best else 0, threat.move.priority)
            theirs = self.foe_ko(slowed, me, threat, generation)
            mixed = [theirs_plain[0] * (1 - first) + theirs[0] * first] + theirs[1:]
            return duel(mine_wasted, mixed, first, new_first)
        if status in ("psn", "tox", "brn"):
            poisoned = replace(enemy, status=status)
            mine = self.my_ko(me, poisoned, best, generation, lost=1)
            scale = 0.5 if status == "brn" and threat.move.category == "physical" else 1.0
            theirs = self.foe_ko(enemy, me, threat, generation, scale=scale)
            return duel(mine, theirs, first, first_later)
        if status == "cnf":
            theirs = self.foe_ko(enemy, me, threat, generation, rate_scale=0.6)
            return duel(mine_wasted, theirs, first, first_later)
        return duel(mine_wasted, theirs_plain, first, first_later)

    def _sleep_ko(self, enemy: Combatant, me: Combatant, threat: Hit, generation: int, asleep_from: int,
                  lost: int) -> list[float]:
        """Foe KO turns when it sleeps on turns ``asleep_from .. asleep_from + lost - 1`` (0-based)."""
        rate = threat.acc * self.action_rate(enemy)
        rates = tuple(0.0 if asleep_from <= turn < asleep_from + lost else rate for turn in range(T_MAX))
        return ko_turns(me.hp, Profile(threat.top, rates, self.residual_of(me, generation)))

    # ------------------------------------------------------------------ switches
    def _switch_options(self, view: BattleView, me: Combatant | None, enemy: Combatant, threat: Hit | None,
                        bench: list[tuple[int, Combatant]], stay: float, generation: int, weather: str,
                        forced: bool) -> list[Option]:
        options: list[Option] = []
        enemy_name = display_name(view.enemy.species)
        active_name = display_name(view.me.species)
        for index, member in bench:
            name = display_name(member.mon.species)
            their_best = self.best_hit(enemy, member, generation, weather)
            mine_hit = self.best_hit(member, enemy, generation, weather)
            first = self.p_first(member, enemy, generation, mine_hit.move.priority if mine_hit else 0,
                                 their_best.move.priority if their_best else 0)
            mine = self.my_ko(member, enemy, mine_hit, generation)
            entry_mult = 1.0
            if forced or threat is None:
                value = duel(mine, self.foe_ko(enemy, member, their_best, generation), first, first)
                entry_ko = 0.0
            else:
                entry = self.hit(enemy, threat.slot, threat.move, member, generation, weather)
                entry_mult = 0.0 if entry.immune else entry.multiplier
                land = entry.acc * self.action_rate(enemy) if not entry.immune else 0.0
                entry_ko = entry.ko_now * self.action_rate(enemy) if not entry.immune else 0.0
                hurt = replace(member, hp=max(1.0, member.hp - entry.mean))
                after_land = duel(mine, self.foe_ko(enemy, hurt, their_best, generation), first, first)
                after_miss = duel(mine, self.foe_ko(enemy, member, their_best, generation), first, first)
                value = max(0.0, land - entry_ko) * after_land + (1 - land) * after_miss
                if me is not None:
                    value += 0.05 * (me.hp / me.max_hp) * (1 - stay)  # the active survives for later
            duel_value = value
            resist = threat is not None and entry_mult < 1
            if forced:
                reason = f"Sale {name}: mejor duelo contra {enemy_name} ({duel_value:.0%})"
                if threat is not None and their_best is not None and their_best.multiplier < 1:
                    reason += f"; resiste {TYPE_ES.get(their_best.move.type, '')} (x{their_best.multiplier:g})"
            else:
                threat_type = TYPE_ES.get(threat.move.type, threat.move.type) if threat is not None else "?"
                reason = (f"Cambio a {name}: {active_name} pierde el duelo ({stay:.0%}) contra {enemy_name}; "
                          f"{name} {'resiste' if resist else 'recibe'} {threat_type} (x{entry_mult:g}) y gana "
                          f"{duel_value:.0%}")
            option = Option("switch", index, name, value, multiplier=entry_mult, reason=reason,
                            details={"win": round(duel_value, 4), "entry_ko": round(entry_ko, 4),
                                     "entry_multiplier": entry_mult})
            if not forced and not (stay < ACTIVE_LOSING and value >= stay + SWITCH_MARGIN):
                option.details["hold"] = 1.0  # not clearly better: information only
                option.value = min(value, stay - 1e-3) if stay > 0 else value * 0.5
            options.append(option)
        return options

    # ------------------------------------------------------------------ wild battles
    @staticmethod
    def escape_probability(me: Combatant, enemy: Combatant, generation: int) -> float:
        mine, theirs = me.stats["spe"], max(1.0, enemy.stats["spe"])
        if mine >= theirs:
            return 1.0
        if generation >= 3:
            return max(0.0, min(1.0, (mine * 128 / theirs + 30) / 256))
        return max(0.0, min(1.0, (mine * 32 / (theirs / 4) + 30) / 256))

    def _run_option(self, view: BattleView, me: Combatant, enemy: Combatant, threat: Hit | None,
                    options: list[Option], stay: float, generation: int) -> Option | None:
        if not view.can_run or any(item.kind == "item" for item in options):
            return None
        if any(item.details.get("catch_plan") for item in options):
            return None
        escape = self.escape_probability(me, enemy, generation)
        best_switch = max((item.details.get("win", 0.0) for item in options
                           if item.kind == "switch" and not item.details.get("hold")), default=0.0)
        enemy_name = display_name(view.enemy.species)
        fast_loss = sum(self.foe_ko(enemy, me, threat, generation)[:2])
        moves = [item for item in options if item.kind == "move" and item.usable]
        long_fight = (all(item.ko_now < 0.2 for item in moves) and enemy.level + 10 <= me.level
                      and stay < 0.98)
        dangerous = stay < 0.3 and fast_loss > 0.5 and best_switch < 0.6
        if dangerous and escape >= 0.5:
            reason = (f"Huir: combate salvaje peligroso ({enemy_name} nos debilita en 1-2 turnos "
                      f"({fast_loss:.0%}); duelo {stay:.0%}); escapar {escape:.0%}")
        elif long_fight and escape >= 0.9:
            reason = (f"Huir: combate salvaje sin interés ({enemy_name} Nv. {enemy.level}: poca experiencia y "
                      f"combate largo); escapar {escape:.0%}")
        else:
            return None
        value = max(escape * 0.97 + (1 - escape) * stay * 0.7, stay + 0.05)
        return Option("run", 0, "Huir", value, reason=reason,
                      details={"escape": round(escape, 3), "stay": round(stay, 4), "fast_loss": round(fast_loss, 3)})

    def _catch_option(self, view: BattleView, enemy: Combatant, options: list[Option]) -> Option | None:
        if len(view.party) >= 6:
            return None
        if view.enemy.national_dex in {mon.national_dex for mon in view.party}:
            return None
        party_types = {canonical_type(name) for mon in view.party for name in mon.types}
        new_types = [name for name in enemy.types if name not in party_types and name != "???"]
        if not new_types:
            return None
        fraction = view.enemy.hp_fraction
        enemy_name = display_name(view.enemy.species)
        types_text = "/".join(TYPE_ES.get(name, name) for name in new_types)
        weakened = fraction <= 0.5 or view.enemy.status in ("slp", "frz", "par")
        moves = [item for item in options if item.kind == "move" and item.usable]
        gentle = [item for item in moves if item.ko_now < 0.15 and item.damage > 0]
        status_moves = [item for item in moves if "status_success" in item.details and item.details["hit"] > 0.5]
        if weakened or not (gentle or status_moves):
            fallback = min(moves, key=lambda item: item.ko_now, default=None)
            reason = f"Capturar a {enemy_name}: aporta el tipo {types_text} que falta en el equipo (PS al {fraction:.0%})"
            return Option("item", 0, "Ball", 1.5, reason=reason,
                          details={"catch": 1.0, "fallback_move": float(fallback.index) if fallback else 0.0})
        if status_moves and not view.enemy.status:
            pick = max(status_moves, key=lambda item: item.details["hit"])
            pick.reason = f"{pick.label}: preparar la captura de {enemy_name} ({types_text} falta en el equipo)"
        else:
            pick = max(gentle, key=lambda item: item.damage)
            pick.reason = (f"{pick.label}: debilitar a {enemy_name} sin derrotarlo para capturarlo "
                           f"({types_text} falta en el equipo)")
        pick.value = 1.4
        pick.details["catch_plan"] = 1.0
        return None


def _roll_share(top: float, hp: float) -> float:
    """Share of the 16 damage rolls (85-100 %) of a top-roll hit that reach ``hp``."""
    if top <= 0:
        return 0.0
    return sum(1 for roll in range(85, 101) if top * roll / 100 >= hp) / 16


__all__ = ["ACTIVE_LOSING", "Analysis", "Combatant", "Hit", "Option", "Profile", "SWITCH_MARGIN", "Strategist",
           "T_MAX", "accuracy_multiplier", "constant_rates", "duel", "expected_ko_turn", "internal_species",
           "ko_turns", "stage_multiplier", "win_probability"]
