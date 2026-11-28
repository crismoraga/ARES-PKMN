"""Behaviour of a side over battles: what it chooses, how it fights, and how far apart two rivals are.

``BehaviourEnv`` is an ``ares_u.battle.env.BattleEnv`` that additionally counts, for every *voluntary*
decision of each side (at least two legal actions and not a forced replacement), the category of the
chosen action and, for damaging moves, the type effectiveness against the visible foe, the accuracy
and the base power; after each step it reads the engine log for the damage each side dealt. The
existing counters and rewards of ``BattleEnv`` are untouched.

Categories: ``attack`` (any damaging move, Struggle included), ``status`` (sleep, paralysis, poison,
Toxic, burn, Yawn, Leech Seed, confusion), ``setup`` (self-boosts, Belly Drum, Curse, Focus Energy),
``debuff`` (foe stat drops), ``disrupt`` (Haze, Roar/Whirlwind, Taunt, Encore, Disable, Mean Look),
``heal``, ``protect`` (Protect, Endure, Substitute), ``field`` (screens, Safeguard, Mist, Spikes,
weather), ``other`` and ``switch``.

``profile`` turns summed counters into rates (switch rate, status-move rate, setup-move rate,
type-effective move rate, mean accuracy/power of chosen attacks, damage per turn, mean turns), and
``distances`` gives z-scored Euclidean distances between rival profiles plus the Jensen-Shannon
divergence between their action-category distributions.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.stadium.dex import MoveData
from core.stadium.engine import MOVE_STRUGGLE

from ares_u.battle.env import BattleEnv
from ares_u.battle.obs import resolve_move

from .archetypes import (CONFUSE, DISABLE, ENCORE, FOE_DROP, HAZE, HEAL, LEECH_SEED, MEAN_LOOK, PHAZE, PROTECT,
                         SCREENS, SELF_BOOST, SPIKES, STATUS_INFLICT, TAUNT)

CATEGORIES = ("attack", "status", "setup", "debuff", "disrupt", "heal", "protect", "field", "other", "switch")
DISRUPT = frozenset({HAZE, PHAZE, TAUNT, ENCORE, DISABLE, MEAN_LOOK})
FIELD = frozenset(set(SCREENS) | {SPIKES, 115, 136, 137, 164})
SUBSTITUTE = 79
# Rates of the behaviour vector used for distances (order matters: it is the column order of reports).
PROFILE_KEYS = ("switch_rate", "attack_rate", "status_rate", "setup_rate", "debuff_rate", "disrupt_rate",
                "support_rate", "super_effective_rate", "resisted_rate", "risky_rate", "mean_power",
                "mean_accuracy", "damage_per_turn", "mean_turns")


def move_category(move: MoveData) -> str:
    if move.move_id == MOVE_STRUGGLE or move.power > 0:
        return "attack"
    effect = move.effect
    if effect in STATUS_INFLICT or effect in CONFUSE or effect == LEECH_SEED:
        return "status"
    if effect in SELF_BOOST:
        return "setup"
    if effect in FOE_DROP:
        return "debuff"
    if effect in DISRUPT:
        return "disrupt"
    if effect in HEAL:
        return "heal"
    if effect in PROTECT or effect == SUBSTITUTE:
        return "protect"
    if effect in FIELD:
        return "field"
    return "other"


@dataclass
class SideBehaviour:
    decisions: int = 0
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CATEGORIES, 0))
    damaging: int = 0
    super_effective: int = 0
    resisted: int = 0
    immune: int = 0
    accuracy: float = 0.0
    power: float = 0.0
    risky: int = 0
    damage_dealt: float = 0.0
    hits: int = 0
    knockouts: int = 0

    def to_json(self) -> dict[str, Any]:
        return {"decisions": self.decisions, "counts": dict(self.counts), "damaging": self.damaging,
                "super_effective": self.super_effective, "resisted": self.resisted, "immune": self.immune,
                "accuracy": round(self.accuracy, 4), "power": round(self.power, 2), "risky": self.risky,
                "damage_dealt": round(self.damage_dealt, 5), "hits": self.hits, "knockouts": self.knockouts}


@dataclass
class BehaviourEnv(BattleEnv):
    behaviour: tuple[SideBehaviour, SideBehaviour] = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.behaviour = (SideBehaviour(), SideBehaviour())

    def _record_choice(self, side: int, action: int) -> None:
        record = self.behaviour[side]
        record.decisions += 1
        if action >= 4:
            record.counts["switch"] += 1
            return
        battle = self.battle
        fighter = battle.sides[side].fighter
        raw = battle._move_for(fighter, action) if action < len(fighter.move_ids) else self.dex.moves[MOVE_STRUGGLE]
        move = resolve_move(battle, fighter, raw)
        category = move_category(move)
        record.counts[category] += 1
        if category != "attack" or move.move_id == MOVE_STRUGGLE:
            return
        record.damaging += 1
        foe = battle.sides[1 - side].fighter
        multiplier = self.dex.multiplier(move.move_type, foe.types)
        if multiplier == 0:
            record.immune += 1
        elif multiplier > 1:
            record.super_effective += 1
        elif multiplier < 1:
            record.resisted += 1
        record.accuracy += (move.accuracy or 100) / 100
        record.power += 60.0 if move.power == 1 else float(move.power)
        record.risky += int(0 < move.accuracy < 90)

    def step(self, actions: Mapping[int, int]) -> None:
        for side, action in actions.items():
            if self.voluntary(side):
                self._record_choice(side, action)
        super().step(actions)
        for event in self.battle.log:
            if event[0] == "hit":
                defender_side, dealt, attacker, species = int(event[1]), int(event[2]), int(event[6]), int(event[7])
                maximum = next((member.max_hp for member in self.battle.sides[defender_side].party
                                if member.spec.species == species), 0)
                if maximum and attacker != defender_side:
                    self.behaviour[attacker].damage_dealt += dealt / maximum
                    self.behaviour[attacker].hits += 1
            elif event[0] == "faint":
                self.behaviour[1 - int(event[1])].knockouts += 1


def merge(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum of ``SideBehaviour.to_json`` dictionaries (plus ``turns`` and ``battles``)."""
    total: dict[str, Any] = {"decisions": 0, "counts": dict.fromkeys(CATEGORIES, 0), "damaging": 0,
                             "super_effective": 0, "resisted": 0, "immune": 0, "accuracy": 0.0, "power": 0.0,
                             "risky": 0, "damage_dealt": 0.0, "hits": 0, "knockouts": 0, "turns": 0, "battles": 0}
    for record in records:
        for key, value in record.items():
            if key == "counts":
                for category, count in dict(value).items():
                    total["counts"][category] = total["counts"].get(category, 0) + int(count)
            elif key in total:
                total[key] += value
    return total


def profile(summed: Mapping[str, Any]) -> dict[str, float]:
    """Rates of one rival from merged counters."""
    decisions = max(1, int(summed["decisions"]))
    counts: Mapping[str, int] = summed["counts"]
    damaging = max(1, int(summed["damaging"]))
    return {
        "switch_rate": counts.get("switch", 0) / decisions,
        "attack_rate": counts.get("attack", 0) / decisions,
        "status_rate": counts.get("status", 0) / decisions,
        "setup_rate": counts.get("setup", 0) / decisions,
        "debuff_rate": counts.get("debuff", 0) / decisions,
        "disrupt_rate": counts.get("disrupt", 0) / decisions,
        "support_rate": (counts.get("heal", 0) + counts.get("protect", 0) + counts.get("field", 0)) / decisions,
        "super_effective_rate": int(summed["super_effective"]) / damaging,
        "resisted_rate": int(summed["resisted"]) / damaging,
        "risky_rate": int(summed["risky"]) / damaging,
        "mean_power": float(summed["power"]) / damaging,
        "mean_accuracy": float(summed["accuracy"]) / damaging,
        "damage_per_turn": float(summed["damage_dealt"]) / max(1, int(summed["turns"])),
        "mean_turns": int(summed["turns"]) / max(1, int(summed["battles"])),
    }


def category_distribution(summed: Mapping[str, Any]) -> NDArray[np.float64]:
    counts = np.array([float(summed["counts"].get(category, 0)) for category in CATEGORIES])
    total = counts.sum()
    return counts / total if total > 0 else np.full(len(CATEGORIES), 1.0 / len(CATEGORIES))


def jensen_shannon(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    """Jensen-Shannon divergence in bits (0 = identical distributions, 1 = disjoint)."""
    middle = (first + second) / 2

    def kl(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    return 0.5 * kl(first, middle) + 0.5 * kl(second, middle)


def distances(profiles: Mapping[str, Mapping[str, float]], keys: Sequence[str] = PROFILE_KEYS
              ) -> dict[str, Any]:
    """Pairwise z-scored Euclidean distances between behaviour profiles and each rival's nearest neighbour."""
    names = list(profiles)
    matrix = np.array([[float(profiles[name][key]) for key in keys] for name in names], dtype=np.float64)
    spread = matrix.std(axis=0)
    scaled = (matrix - matrix.mean(axis=0)) / np.where(spread > 1e-12, spread, 1.0)
    pairwise = np.sqrt(((scaled[:, None, :] - scaled[None, :, :]) ** 2).sum(axis=2))
    nearest = {}
    for index, name in enumerate(names):
        others = [(float(pairwise[index, other]), names[other]) for other in range(len(names)) if other != index]
        distance, neighbour = min(others) if others else (math.nan, "")
        nearest[name] = {"neighbour": neighbour, "distance": distance}
    return {"names": names, "keys": list(keys), "matrix": pairwise.tolist(), "nearest": nearest}
