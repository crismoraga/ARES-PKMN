"""Duels between two policies: fitness, behaviour descriptors and decision traces.

Fitness follows the paper's zero-sum convention ``f_0 + f_1 = 1``: half comes
from the result (win 1, draw 0.5, loss 0) and half from the remaining-HP
margin, so a narrow win and a sweep are distinguishable.

The behaviour descriptor of a side (all in [0, 1]) replaces the paper's CLIP
embedding with interpretable battle style: switch rate, status-move rate,
aggression (chosen vs best believed damage), risk taking at low HP, battle
length, set-up rate, roster usage and super-effective hit rate.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .dex import Dex
from .engine import Battle, Event, MonSpec
from .features import option_features
from .knowledge import SideKnowledge
from .policies import Policy

BEHAVIOR_NAMES = ("switch_rate", "status_rate", "aggression", "risk", "length", "setup_rate", "roster_usage",
                  "super_rate")
BEHAVIOR_DIM = len(BEHAVIOR_NAMES)
_SETUP = frozenset({10, 11, 12, 13, 14, 16, 50, 51, 52, 53, 54, 56, 108, 142, 156, 206, 208, 211, 212, 109})


@dataclass
class DecisionPoint:
    """A turn decision of one side, with an optional battle clone for counterfactuals."""
    side: int
    turn: int
    actions: tuple[int, ...]
    chosen: int
    features: NDArray[np.float32]
    snapshot: Battle | None = None


@dataclass
class MatchResult:
    fitness: float                      # side 0; side 1 gets 1 - fitness
    winner: int
    turns: int
    behavior: tuple[NDArray[np.float32], NDArray[np.float32]]
    hp: tuple[float, float]
    new_facts: tuple[int, int]
    unsupported: tuple[int, int]
    decisions: list[DecisionPoint] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"fitness": round(self.fitness, 4), "winner": self.winner, "turns": self.turns,
                "hp": [round(value, 3) for value in self.hp], "behavior": [b.round(3).tolist() for b in self.behavior]}


@dataclass
class _SideStats:
    decisions: int = 0
    switches: int = 0
    moves: int = 0
    status: int = 0
    setup: int = 0
    risky: int = 0
    low: int = 0
    aggression: float = 0.0
    aggression_n: int = 0
    hits: int = 0
    super: int = 0
    entered: set[int] = field(default_factory=lambda: {0})

    def descriptor(self, turns: int, team_size: int) -> NDArray[np.float32]:
        decided, moves = max(1, self.decisions), max(1, self.moves)
        return np.array([
            self.switches / decided, self.status / moves,
            self.aggression / self.aggression_n if self.aggression_n else 0.5,
            self.risky / self.low if self.low else 0.5,
            min(1.0, turns / 60), self.setup / moves, len(self.entered) / max(1, team_size),
            self.super / self.hits if self.hits else 0.0], dtype=np.float32).clip(0.0, 1.0)


def fitness_from(winner: int, hp: tuple[float, float]) -> float:
    result = 1.0 if winner == 0 else 0.0 if winner == 1 else 0.5
    return float(0.5 * result + 0.5 * (hp[0] - hp[1] + 1) / 2)


def play(dex: Dex, teams: tuple[Sequence[MonSpec], Sequence[MonSpec]], policies: tuple[Policy, Policy],
         knowledge: tuple[SideKnowledge, SideKnowledge], seed: int, *, record: bool = False,
         snapshot_every: int = 0, max_turns: int = 150, learn: bool = True) -> MatchResult:
    """Play one battle. ``snapshot_every`` > 0 keeps clones of every n-th decision (lessons)."""
    battle = Battle(dex, teams[0], teams[1], seed=seed, max_turns=max_turns, record=record)
    rng = random.Random(seed * 7919 + 17)
    for policy in policies:
        policy.reset()
    stats = [_SideStats(), _SideStats()]
    decisions: list[DecisionPoint] = []
    facts = [0, 0]
    turn_decisions = 0
    while not battle.over:
        actions: dict[int, int] = {}
        snapshot = battle.clone() if snapshot_every and turn_decisions % snapshot_every == 0 else None
        for side in battle.pending:
            legal = battle.legal_actions(side)
            chosen = policies[side].act(battle, side, knowledge[side], rng)
            actions[side] = chosen
            if len(legal) > 1 and not battle._replacing:
                features = option_features(battle, side, knowledge[side], legal) if (snapshot is not None or record) \
                    else getattr(policies[side], "last_features", None)
                own = stats[side]
                own.decisions += 1
                own.switches += int(chosen >= 4)
                if features is not None and len(features) == len(legal):
                    row = features[legal.index(chosen)]
                    best = float(features[:, 1].max())
                    if chosen < 4 and best > 0:
                        own.aggression += float(row[1]) / best
                        own.aggression_n += 1
                    if row[15] < 0.3:
                        own.low += 1
                        own.risky += int(chosen < 4)
                    if snapshot is not None:
                        decisions.append(DecisionPoint(side, battle.turn, tuple(legal), chosen, features, snapshot))
        turn_decisions += 1
        on_field = {side: battle.sides[side].fighter.spec.species for side in (0, 1)}
        events = battle.submit(actions)
        for event in events:
            kind = event[0]
            if kind == "move":
                mover = stats[int(event[1])]
                move = dex.moves[int(event[3])]
                mover.moves += 1
                mover.status += int(not move.power)
                mover.setup += int(move.effect in _SETUP)
            elif kind == "hit" and event[3] != "fixed":
                attacker = stats[int(event[6])]
                attacker.hits += 1
                attacker.super += int(event[3] == "super")
            elif kind == "switch":
                stats[int(event[1])].entered.add(int(event[2]))
        if learn:
            for side in (0, 1):
                facts[side] += knowledge[side].observe(dex, events, side, on_field)
    summary = battle.summary()
    behaviors = [stats[side].descriptor(summary.turns, len(battle.sides[side].party)) for side in (0, 1)]
    return MatchResult(fitness_from(summary.winner, summary.hp_fraction), summary.winner, summary.turns,
                       (behaviors[0], behaviors[1]), summary.hp_fraction, (facts[0], facts[1]), summary.unsupported,
                       decisions, summary.events if record else [])
