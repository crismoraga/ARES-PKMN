"""Archetype rivals: hand-written, behaviourally distinct battle styles for the ARES league.

Every archetype is a stadium policy (the ``core.stadium.policies.Policy`` protocol: ``name``,
``reset()``, ``act(battle, side, knowledge, rng) -> engine action``) that decides on *public*
information only, exactly like the Greedy anchor: its own team, the foe's visible species, level,
HP bar, status and stat stages, and the moves the foe has revealed (``Perspective``). Type
effectiveness comes from the side's knowledge object (the E1 harness gives heuristic rivals the real
chart through ``OracleKnowledge``). The only exception is the ``lookahead`` planner of
``ares_u.battle.headroom`` (registered in ``ares_u.league.registry``), which reads the full engine
state on purpose: it is the "cheating genius" of the league.

Styles (what each one does that the others do not):

* ``greedy``    - highest expected damage, never switches voluntarily (``GreedyDamagePolicy``).
* ``stall``     - wall: heals when low, inflicts status (Toxic > sleep > paralysis > burn > poison >
  Yawn/Leech Seed/confusion), Protects while residual damage ticks, raises screens, switches to the
  bench member that takes the least damage when threatened, and only then attacks (draining moves
  preferred).
* ``setup``     - sweeper: boosts first (Swords Dance, Dragon Dance, Calm Mind, Agility, Belly
  Drum, ...) while it is healthy and not threatened, then attacks with the boosted stats; never
  switches voluntarily, finishes a foe it can knock out.
* ``pivot``     - switch-heavy: scores the matchup of the active and of every bench member (its
  offence against the foe minus the foe's threat to it, plus speed and HP) and switches whenever a
  bench member is clearly better.
* ``gambler``   - risk-taker: the highest raw power x type x STAB, ignoring accuracy, recharge and
  self-destruction (Explosion counts double: it halves the defence).
* ``disruptor`` - field control: Haze/Roar against boosts, Spikes, Taunt, Encore/Disable, stat drops
  (up to -2 per stat) and confusion before attacking.
* ``nash``      - game theorist: builds the one-turn zero-sum payoff matrix of its options against
  the foe's believed options (revealed moves or STAB guesses, plus a non-damaging option) from
  public damage estimates with turn order and knock-out probabilities, solves it exactly with a
  linear programme and *samples* the mixed strategy (it is unpredictable on purpose).
* ``prior``     - the ARES expert option scorer (``MLPGenome.prior``) played greedily.
* ``random``    - uniformly random legal action.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.stadium.dex import MoveData
from core.stadium.engine import (_MULTI_BOOST, _STAT_DOWN, _STAT_UP, ATK, BURN, DEF, MOVE_STRUGGLE, POISON,
                                 SPA, SPD, SPE, STAGE_RATIOS, TOXIC, Battle, Fighter)
from core.stadium.features import FEATURE_NAMES, Perspective, estimate_damage, option_features, public_stats
from core.stadium.knowledge import SideKnowledge
from core.stadium.policies import GenomePolicy, GreedyDamagePolicy, MLPGenome, Policy, RandomPolicy

from ares_u.battle.agents import Choice, Decision
from ares_u.battle.obs import resolve_move

Array = NDArray[np.float64]
F = {name: index for index, name in enumerate(FEATURE_NAMES)}
DMG, DMG_MAX, KO, ACCURACY = F["dmg_cur"], F["dmg_max"], F["ko_prob"], F["accuracy"]
THREAT, FASTER, SW_OFF, SW_HP, OWN_HP = F["threat"], F["faster"], F["switch_offense"], F["switch_hp"], F["own_hp"]

# Move effects (Gen-3 effect ids, the ones the stadium engine models).
STATUS_INFLICT = {33: 6, 1: 5, 67: 4, 167: 4, 66: 3, 187: 3}   # Toxic, sleep, paralysis, burn, poison, Yawn
CONFUSE = frozenset({49, 199, 118, 166})
LEECH_SEED = 84
HEAL = frozenset({32, 157, 132, 133, 134, 37})
REST = 37
SELF_BOOST = frozenset(set(_STAT_UP) | set(_MULTI_BOOST) | {142, 109, 47})
FOE_DROP = frozenset(set(_STAT_DOWN) | {205})
PROTECT = frozenset({111, 116})
SCREENS = {35: "light_screen", 65: "reflect", 124: "safeguard", 46: "mist"}
SPIKES, HAZE, PHAZE, TAUNT, ENCORE, DISABLE, MEAN_LOOK = 112, 25, 28, 175, 90, 86, 106
DRAIN = frozenset({3, 8})
OHKO, EXPLOSION, BELLY_DRUM, CURSE, FOCUS_ENERGY = 38, 7, 142, 109, 47
FIXED_DAMAGE = frozenset({41, 130, 87, 88, 40, 189})
OFFENSIVE_STATS = frozenset({ATK, SPA, SPE})
NOMINAL_POWER = 60
STAB_GUESS_POWER = 70


def _stage(value: float, stage: int) -> float:
    numerator, denominator = STAGE_RATIOS[max(0, min(12, stage))]
    return value * numerator / denominator


class Situation:
    """One decision point seen by an archetype: legal actions, their option features and the view."""
    __slots__ = ("battle", "side", "knowledge", "actions", "features", "view", "own", "foe_side", "me", "foe")

    def __init__(self, battle: Battle, side: int, knowledge: SideKnowledge, actions: Sequence[int]) -> None:
        self.battle, self.side, self.knowledge = battle, side, knowledge
        self.actions = list(actions)
        self.features = option_features(battle, side, knowledge, self.actions)
        self.view = Perspective(battle, side, knowledge)
        self.own, self.foe_side = battle.sides[side], battle.sides[1 - side]
        self.me, self.foe = self.own.fighter, self.foe_side.fighter

    def row(self, action: int) -> NDArray[np.float32]:
        return self.features[self.actions.index(action)]

    @property
    def moves(self) -> list[int]:
        return [action for action in self.actions if action < 4]

    @property
    def switches(self) -> list[int]:
        return [action for action in self.actions if action >= 4]

    @property
    def hp(self) -> float:
        return self.me.hp / max(1, self.me.max_hp)

    @property
    def threat_now(self) -> float:
        moves = self.moves
        return float(self.row(moves[0])[THREAT]) if moves else 0.0

    def move(self, action: int) -> MoveData:
        """The move the engine would use for slot ``action`` (Struggle, Hidden Power and Weather Ball resolved)."""
        move = self.battle._move_for(self.me, action) if action < len(self.me.move_ids) else \
            self.battle.dex.moves[MOVE_STRUGGLE]
        return resolve_move(self.battle, self.me, move)

    def candidate(self, action: int) -> Fighter:
        return self.own.party[self.own.bench()[action - 4]]

    def expected(self, action: int) -> float:
        row = self.row(action)
        return float(row[DMG] * row[ACCURACY])

    def best_attack(self, rng: random.Random, bonus: dict[int, float] | None = None) -> int:
        """Greedy's rule: the move with the highest expected damage (tiny random tie-break)."""
        moves = self.moves
        if not moves:
            return self.replacement()
        bonus = bonus or {}
        return max(moves, key=lambda action: self.expected(action) * bonus.get(self.move(action).effect, 1.0)
                   + 0.01 * rng.random())

    def any_knockout(self, probability: float = 0.85) -> bool:
        return any(self.row(action)[KO] * self.row(action)[ACCURACY] >= probability for action in self.moves
                   if self.move(action).power > 0)

    def replacement(self, score: Callable[[NDArray[np.float32]], float] | None = None) -> int:
        """Pick a bench member (forced or voluntary); default = Greedy's replacement rule."""
        rule = score or (lambda row: float(row[SW_OFF] + 0.5 * row[SW_HP] - 0.3 * row[THREAT]))
        options = self.switches or self.actions
        return max(options, key=lambda action: rule(self.row(action)))

    def foe_bench_alive(self) -> int:
        return sum(1 for index in self.foe_side.bench() if self.foe_side.party[index].alive)


class _Archetype:
    """Shared ``act``: one legal action is played at once, otherwise ``decide`` sees the situation."""
    name = "archetype"

    def reset(self) -> None:
        return None

    def act(self, battle: Battle, side: int, knowledge: SideKnowledge, rng: random.Random) -> int:
        actions = battle.legal_actions(side)
        if len(actions) == 1:
            return actions[0]
        return int(self.decide(Situation(battle, side, knowledge, actions), rng))

    def decide(self, situation: Situation, rng: random.Random) -> int:
        raise NotImplementedError


# ----------------------------------------------------------------------------- stall
def status_usable(situation: Situation, move: MoveData) -> bool:
    """Would this non-damaging status move stick on the visible foe (types, current status, screens)?"""
    foe, effect = situation.foe, move.effect
    if not foe.alive or foe.substitute or situation.foe_side.safeguard and effect in STATUS_INFLICT:
        return False
    if effect == LEECH_SEED:
        return not foe.seeded and 12 not in foe.types
    if effect in CONFUSE:
        return foe.confusion == 0
    if effect not in STATUS_INFLICT or foe.status:
        return False
    if effect in (66, 33) and (3 in foe.types or 8 in foe.types):
        return False
    if effect == 167 and 10 in foe.types:
        return False
    if effect == 67 and move.move_type == 13 and situation.knowledge.multiplier(13, foe.types) == 0:
        return False
    return not (effect == 187 and foe.yawn)


class StallPolicy(_Archetype):
    name = "stall"
    status_value = {**STATUS_INFLICT, LEECH_SEED: 3, 49: 2, 199: 2, 118: 1, 166: 1}

    def decide(self, situation: Situation, rng: random.Random) -> int:
        if not situation.moves:  # forced replacement: the wall that takes the least damage
            return situation.replacement(lambda row: float(-row[THREAT] + 0.6 * row[SW_HP] + 0.1 * row[SW_OFF]))
        me, hp, threat = situation.me, situation.hp, situation.threat_now
        moves = [(action, situation.move(action)) for action in situation.moves]
        heals = [action for action, move in moves if move.effect in HEAL and
                 (hp <= 0.4 if move.effect == REST else hp <= 0.55)]
        if heals:
            return heals[0]
        statuses = [(self.status_value.get(move.effect, 0), action) for action, move in moves
                    if move.power == 0 and status_usable(situation, move)]
        if statuses:
            return max(statuses)[1]
        foe = situation.foe
        ticking = foe.alive and (foe.status & (POISON | TOXIC | BURN) or foe.seeded or foe.cursed)
        if ticking and me.protect_streak == 0:
            protects = [action for action, move in moves if move.effect in PROTECT]
            if protects:
                return protects[0]
        if hp >= 0.6 and threat < 0.5:
            for action, move in moves:
                attribute = SCREENS.get(move.effect)
                if attribute is not None and not getattr(situation.own, attribute):
                    return action
        if threat >= 0.7 and me.turns_active >= 1 and situation.switches:
            walls = [action for action in situation.switches
                     if situation.row(action)[THREAT] <= 0.5 * threat and situation.row(action)[SW_HP] >= 0.5]
            if walls:
                return min(walls, key=lambda action: float(situation.row(action)[THREAT]))
        return situation.best_attack(rng, bonus={effect: 1.25 for effect in DRAIN})


# ----------------------------------------------------------------------------- setup sweeper
def boost_value(situation: Situation, move: MoveData) -> float:
    """How much the setup sweeper wants this self-boost now (0 = useless: capped, wrong user, ...)."""
    me, effect = situation.me, move.effect
    if effect == BELLY_DRUM:
        return 4.0 if me.stages[ATK] < 12 and situation.hp > 0.6 else 0.0
    if effect == FOCUS_ENERGY:
        return 0.5 if not me.focus_energy else 0.0
    if effect == CURSE:
        return 2.0 if 7 not in me.types and me.stages[ATK] < 10 else 0.0
    raised: tuple[tuple[int, int], ...] = ()
    if effect in _STAT_UP:
        raised = (_STAT_UP[effect],)
    elif effect in _MULTI_BOOST:
        raised = _MULTI_BOOST[effect]
    value = 0.0
    for stat, amount in raised:
        cap = 6 + 2 * max(1, amount)  # +2 for +1 moves, +4 for +2 moves
        if me.stages[stat] < cap:
            value += (1.5 if stat in OFFENSIVE_STATS else 0.8) * amount
    return value


class SetupPolicy(_Archetype):
    name = "setup"

    def decide(self, situation: Situation, rng: random.Random) -> int:
        if not situation.moves:  # bring in a member that can set up, if any
            def lead(row: NDArray[np.float32]) -> float:
                return float(row[SW_OFF] + 0.5 * row[SW_HP] - 0.3 * row[THREAT])
            boosters = {action for action in situation.switches
                        if any(situation.battle.dex.moves[m].effect in SELF_BOOST
                               for m in situation.candidate(action).move_ids)}
            return max(situation.switches, key=lambda a: lead(situation.row(a)) + (0.6 if a in boosters else 0.0))
        if situation.any_knockout(0.9):
            return situation.best_attack(rng)
        if situation.hp >= 0.5 and situation.threat_now < 0.55:
            boosts = [(boost_value(situation, situation.move(action)), action) for action in situation.moves
                      if situation.move(action).effect in SELF_BOOST]
            boosts = [item for item in boosts if item[0] > 0]
            if boosts:
                return max(boosts)[1]
        return situation.best_attack(rng)


# ----------------------------------------------------------------------------- pivot
def matchup(row: NDArray[np.float32], offence: float, threat: float) -> float:
    return offence - 0.7 * min(2.0, threat) + 0.25 * float(row[FASTER])


class PivotPolicy(_Archetype):
    name = "pivot"
    margin = 0.3

    def bench_score(self, situation: Situation, action: int) -> float:
        row = situation.row(action)
        return matchup(row, min(1.5, float(row[SW_OFF])), float(row[THREAT])) + 0.2 * float(row[SW_HP])

    def decide(self, situation: Situation, rng: random.Random) -> int:
        if not situation.moves:
            return max(situation.switches, key=lambda action: self.bench_score(situation, action))
        if situation.switches and (situation.me.turns_active >= 1 or situation.threat_now >= 1.0):
            moves = situation.moves
            offence = max(min(1.5, situation.expected(action)) for action in moves)
            active = matchup(situation.row(moves[0]), offence, situation.threat_now) + 0.2 * situation.hp
            best = max(situation.switches, key=lambda action: self.bench_score(situation, action))
            if self.bench_score(situation, best) - active > self.margin and situation.row(best)[SW_HP] > 0.25:
                return best
        return situation.best_attack(rng)


# ----------------------------------------------------------------------------- gambler
def raw_power(situation: Situation, move: MoveData) -> float:
    if move.power <= 0:
        return 0.0
    effect, power = move.effect, float(move.power)
    if effect == OHKO:
        power = 200.0 if situation.me.spec.level >= situation.foe.spec.level else 0.0
    elif effect == EXPLOSION:
        power *= 2  # the target's defence is halved
    elif move.power == 1 or effect in FIXED_DAMAGE:
        power = float(NOMINAL_POWER)
    if effect == 29:
        power *= 3
    elif effect in (44, 77):
        power *= 2
    multiplier = situation.view.believed_multiplier(move.move_type, situation.foe.types)[0]
    stab = 1.5 if move.move_type in situation.me.types else 1.0
    return power * multiplier * stab


class GamblerPolicy(_Archetype):
    name = "gambler"

    def decide(self, situation: Situation, rng: random.Random) -> int:
        if not situation.moves:
            return situation.replacement(lambda row: float(row[SW_OFF]))
        return max(situation.moves, key=lambda action: raw_power(situation, situation.move(action))
                   + 0.01 * rng.random())


# ----------------------------------------------------------------------------- disruptor
class DisruptorPolicy(_Archetype):
    name = "disruptor"

    def options(self, situation: Situation) -> list[tuple[float, int]]:
        foe, foe_side = situation.foe, situation.foe_side
        boosted = sum(max(0, stage - 6) for stage in foe.stages[1:8])
        bench = situation.foe_bench_alive()
        found: list[tuple[float, int]] = []
        for action in situation.moves:
            move = situation.move(action)
            effect = move.effect
            if move.power > 0:
                continue
            if effect == HAZE and boosted >= 1:
                found.append((6.0, action))
            elif effect == PHAZE and bench and not foe.ingrain:
                found.append((6.0 if boosted >= 1 else 3.5 if foe_side.spikes else 2.5, action))
            elif effect == SPIKES and foe_side.spikes < 3 and bench:
                found.append((5.0, action))
            elif effect == TAUNT and not foe.taunt:
                found.append((4.0, action))
            elif effect in FOE_DROP and not foe.substitute:
                targets = ((ATK, 1), (DEF, 1)) if effect == 205 else (_STAT_DOWN[effect],)
                if any(foe.stages[stat] > 4 for stat, _ in targets):
                    found.append((3.0, action))
            elif effect in CONFUSE and foe.confusion == 0 and not foe.substitute:
                found.append((3.0, action))
            elif effect == ENCORE and foe.last_slot >= 0 and not foe.encore:
                found.append((3.0, action))
            elif effect == DISABLE and foe.last_slot >= 0 and not foe.disabled_turns:
                found.append((3.0, action))
            elif effect == MEAN_LOOK and not foe.trapped_by:
                found.append((2.0, action))
        return found

    def decide(self, situation: Situation, rng: random.Random) -> int:
        if not situation.moves:
            return situation.replacement()
        if not situation.any_knockout(0.8) and situation.threat_now < 1.0:
            found = self.options(situation)
            if found:
                return max(found)[1]
        return situation.best_attack(rng)


# ----------------------------------------------------------------------------- game theorist
FoeOption = tuple[int, int, int, int] | None  # (move type, power, accuracy %, priority) or a non-damaging action


def solve_zero_sum(payoff: Array) -> Array:
    """Max-min mixed strategy of the row player (exact LP; pure maximin when the solver fails)."""
    rows, cols = payoff.shape
    pure = np.zeros(rows)
    pure[int(np.argmax(payoff.min(axis=1)))] = 1.0
    if rows == 1 or cols == 1:
        return pure if cols > 1 or rows == 1 else np.eye(rows)[int(np.argmax(payoff[:, 0]))]
    try:
        from scipy.optimize import linprog

        cost = np.zeros(rows + 1)
        cost[-1] = -1.0
        upper = np.hstack([-payoff.T, np.ones((cols, 1))])
        equal = np.hstack([np.ones((1, rows)), np.zeros((1, 1))])
        result = linprog(cost, A_ub=upper, b_ub=np.zeros(cols), A_eq=equal, b_eq=np.ones(1),
                         bounds=[(0.0, 1.0)] * rows + [(None, None)], method="highs")
        if result.status != 0:
            return pure
        mix = np.clip(np.asarray(result.x[:rows], dtype=np.float64), 0.0, 1.0)
        mix[mix < 0.02] = 0.0
        total = float(mix.sum())
        return mix / total if total > 0 else pure
    except (ImportError, ValueError):
        return pure


class NashPolicy(_Archetype):
    name = "nash"
    ko_bonus = 0.35

    def foe_options(self, situation: Situation) -> list[FoeOption]:
        dex, foe = situation.battle.dex, situation.foe
        options: list[FoeOption] = []
        for move_id in sorted(situation.knowledge.revealed_moves.get(foe.spec.species, set())):
            move = dex.moves[move_id]
            if move.power > 0:
                power = NOMINAL_POWER if move.power == 1 else move.power
                options.append((move.move_type, power, move.accuracy or 100, move.priority))
        if len(options) < 2:
            options.extend((move_type, STAB_GUESS_POWER, 100, 0) for move_type in sorted(set(foe.types)))
        options.append(None)
        return options

    def foe_damage(self, situation: Situation, defender: Fighter, option: FoeOption, active: bool) -> float:
        """Expected damage of a foe option as a fraction of ``defender``'s max HP (accuracy included)."""
        if option is None or not situation.foe.alive:
            return 0.0
        move_type, power, accuracy, _ = option
        foe = situation.foe
        stats = public_stats(situation.battle.dex, foe.spec.species, foe.spec.level)
        special = move_type >= 10
        attack = _stage(stats[4 if special else 1], foe.stages[SPA if special else ATK])
        defence_stat = defender.stat(5 if special else 2)
        defence = _stage(defence_stat, defender.stages[SPD if special else DEF]) if active else float(defence_stat)
        multiplier = situation.view.believed_multiplier(move_type, defender.types)[0]
        low, high = estimate_damage(foe.spec.level, attack, defence, power, move_type in foe.types, multiplier,
                                    bool(foe.status & BURN) and not special)
        return (low + high) / 2 / max(1, defender.max_hp) * accuracy / 100

    def utility(self, situation: Situation, move: MoveData) -> float:
        """Positional value of a non-damaging move (fraction-of-a-mon units)."""
        effect = move.effect
        if effect in HEAL:
            return min(0.5, 1.0 - situation.hp) * (0.9 if effect == REST else 1.0)
        if effect in STATUS_INFLICT or effect in CONFUSE or effect == LEECH_SEED:
            return (0.12 + 0.05 * self_value(effect)) if status_usable(situation, move) else -0.05
        if effect in SELF_BOOST:
            return 0.08 * boost_value(situation, move)
        if effect in FOE_DROP:
            return 0.06
        if effect in SCREENS:
            return 0.1 if not getattr(situation.own, SCREENS[effect]) else -0.05
        if effect == SPIKES:
            return 0.1 * situation.foe_bench_alive() if situation.foe_side.spikes < 3 else -0.05
        return 0.02

    def payoff(self, situation: Situation, action: int, option: FoeOption, offence_now: float) -> float:
        me, hp, foe_hp = situation.me, situation.hp, situation.view.foe_hp
        if action >= 4:
            candidate = situation.candidate(action)
            row = situation.row(action)
            taken = min(candidate.hp / max(1, candidate.max_hp), self.foe_damage(situation, candidate, option, False))
            return -taken + 0.25 * (min(1.0, float(row[SW_OFF])) - offence_now) - 0.03
        move, row = situation.move(action), situation.row(action)
        accuracy = float(row[ACCURACY])
        taken = min(hp, self.foe_damage(situation, me, option, True))
        foe_ko = 1.0 if option is not None and taken >= hp - 1e-9 else 0.0
        if move.power > 0:
            dealt = min(foe_hp, float(row[DMG_MAX])) * accuracy
            my_ko = float(row[KO]) * accuracy
        else:
            dealt, my_ko = self.utility(situation, move), 0.0
            if move.effect in PROTECT and me.protect_streak == 0:
                taken, foe_ko = 0.0, 0.0
        foe_priority = option[3] if option is not None else 0
        first = move.priority > foe_priority or (move.priority == foe_priority and float(row[FASTER]) >= 0.5)
        if first:
            taken *= 1.0 - my_ko
            foe_ko *= 1.0 - my_ko
        else:
            dealt *= 1.0 - foe_ko
            my_ko *= 1.0 - foe_ko
        return dealt + self.ko_bonus * my_ko - taken - self.ko_bonus * foe_ko

    def strategy(self, situation: Situation) -> Array:
        options = self.foe_options(situation)
        moves = situation.moves
        offence_now = max((min(1.0, situation.expected(action)) for action in moves), default=0.0)
        payoff = np.array([[self.payoff(situation, action, option, offence_now) for option in options]
                           for action in situation.actions], dtype=np.float64)
        return solve_zero_sum(payoff)

    def decide(self, situation: Situation, rng: random.Random) -> int:
        if not situation.moves:
            return situation.replacement()
        mix = self.strategy(situation)
        pick, cumulative = rng.random(), 0.0
        for action, weight in zip(situation.actions, mix):
            cumulative += float(weight)
            if pick < cumulative:
                return action
        return situation.actions[int(np.argmax(mix))]


def self_value(effect: int) -> float:
    return float(StallPolicy.status_value.get(effect, 1))


# ----------------------------------------------------------------------------- registry
def _prior() -> Policy:
    return GenomePolicy(MLPGenome.prior(random.Random(0)), name="prior", greedy=True)


ARCHETYPES: dict[str, Callable[[], Policy]] = {
    "greedy": GreedyDamagePolicy,
    "stall": StallPolicy,
    "setup": SetupPolicy,
    "pivot": PivotPolicy,
    "gambler": GamblerPolicy,
    "disruptor": DisruptorPolicy,
    "nash": NashPolicy,
    "prior": _prior,
    "random": RandomPolicy,
}
DESCRIPTIONS = {
    "greedy": "highest expected damage, never switches voluntarily",
    "stall": "heals, inflicts status, Protects while residual damage ticks, screens, defensive switches",
    "setup": "boosts first while healthy and safe (Swords Dance, Agility, ...), then attacks",
    "pivot": "switches whenever a bench member has a clearly better matchup",
    "gambler": "highest raw power x type x STAB ignoring accuracy, recharge and self-KO",
    "disruptor": "Haze/Roar vs boosts, Spikes, Taunt, Encore/Disable, stat drops, confusion",
    "nash": "solves the one-turn zero-sum game of public damage estimates and samples the mix",
    "prior": "ARES expert option scorer (MLPGenome.prior) played greedily",
    "random": "uniformly random legal action",
    "lookahead": "full-information Monte-Carlo planner (cheats: sees the hidden state)",
}


def make_archetype(name: str) -> Policy:
    if name not in ARCHETYPES:
        raise ValueError(f"unknown archetype {name}")
    return ARCHETYPES[name]()


@dataclass
class ArchetypeController:
    """Batch controller (``ares_u.battle.agents.Controller``) around one stadium policy."""
    name: str
    policy: Any = None
    wants_obs: bool = False

    def __post_init__(self) -> None:
        if self.policy is None:
            self.policy = make_archetype(self.name)

    def choose(self, decisions: list[Decision], rng: random.Random) -> list[Choice]:
        return [Choice(int(self.policy.act(item.env.battle, item.side, item.env.knowledge[item.side], rng)))
                for item in decisions]
