"""Skill-headroom probe: how far above Greedy can *any* policy get on this task?

``LookaheadController`` is a full-information flat Monte-Carlo planner (an analysis tool, not a
method of the protocol): for each legal action it clones the real engine state (including the
opponent's hidden moves) with a fresh RNG seed, plays that action against the opponent's Greedy
reply, finishes the battle with Greedy on both sides, and picks the action with the best mean
outcome over ``rollouts`` samples (common random numbers across actions). It cheats (it sees
everything and models the opponent as Greedy), so its score against Greedy is an optimistic
estimate of the attainable margin; if even this planner barely beats Greedy, the task itself has a
low skill ceiling (the outcome is mostly decided by the random team match-up and by luck).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from core.stadium.engine import Battle
from core.stadium.policies import GreedyDamagePolicy

from .agents import Choice, Decision
from .env import BattleEnv
from .obs import OracleKnowledge


@dataclass
class LookaheadController:
    rollouts: int = 8
    name: str = "lookahead"
    wants_obs: bool = False
    _greedy: GreedyDamagePolicy = field(default_factory=GreedyDamagePolicy)

    def choose(self, decisions: list[Decision], rng: random.Random) -> list[Choice]:
        return [Choice(self.best_action(item.env, item.side, item.legal, rng)) for item in decisions]

    def best_action(self, env: BattleEnv, side: int, legal: list[int], rng: random.Random) -> int:
        if len(legal) == 1:
            return legal[0]
        seeds = [rng.randrange(2 ** 31) for _ in range(self.rollouts)]
        best, best_value = legal[0], float("-inf")
        for action in legal:
            value = sum(self.rollout(env, env.battle.clone(seed), side, action, random.Random(seed))
                        for seed in seeds) / len(seeds)
            if value > best_value:
                best, best_value = action, value
        return best

    def rollout(self, env: BattleEnv, battle: Battle, side: int, action: int, rng: random.Random) -> float:
        """Outcome for ``side`` (+1/0/-1, HP margin as a tie-break) after ``action`` and Greedy play."""
        knowledge = (OracleKnowledge(dex=env.dex), OracleKnowledge(dex=env.dex))
        first = {side: action}
        for other in battle.pending:
            if other != side:
                first[other] = int(self._greedy.act(battle, other, knowledge[other], rng))
        battle.submit(first)
        while not battle.over:
            battle.submit({current: int(self._greedy.act(battle, current, knowledge[current], rng))
                           for current in battle.pending})
        winner = battle.winner
        margin = battle.sides[side].hp_fraction() - battle.sides[1 - side].hp_fraction()
        if winner is None or winner == 2:
            return 0.1 * margin
        return (1.0 if winner == side else -1.0) + 0.1 * margin


def pair_statistics(records: list[dict[str, object]]) -> dict[str, float]:
    """How evaluation pairs were decided. Battle ``2p`` gives the agent team A, ``2p+1`` team B.

    ``team_decided``: the same *team* won both seatings (consistent with a match-up decided by the
    teams); ``agent_swept`` / ``agent_lost_both``: the agent won / lost with both teams; ``other``:
    pairs with a draw.
    """
    outcomes: dict[int, dict[int, float]] = {}
    for record in records:
        index = int(str(record["index"]))
        outcomes.setdefault(index // 2, {})[index % 2] = float(str(record["outcome"]))
    pairs = [pair for pair in outcomes.values() if len(pair) == 2]
    total = max(1, len(pairs))
    team = sum(1 for pair in pairs if pair[0] != 0 and pair[0] == -pair[1])
    swept = sum(1 for pair in pairs if pair[0] > 0 and pair[1] > 0)
    lost = sum(1 for pair in pairs if pair[0] < 0 and pair[1] < 0)
    return {"pairs": float(len(pairs)), "team_decided": team / total, "agent_swept": swept / total,
            "agent_lost_both": lost / total, "other": (len(pairs) - team - swept - lost) / total}
