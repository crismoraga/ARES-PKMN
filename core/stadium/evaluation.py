"""Parallel duel evaluation shared by GAME, tournaments and the live league.

A job is "solution s (side of ``space``) against opponent t": the duel is
played ``repeats`` times with paired seeds, s as side 0. Workers keep the dex,
the rosters and each side's learnt knowledge; knowledge discovered while
evaluating is returned and merged, so a whole population learns the type chart
from the battles it plays (no ROM type table is ever read by an agent).
"""
from __future__ import annotations

import os
import random
from collections.abc import Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .dex import Dex, Game, load_dex
from .engine import MonSpec
from .knowledge import SideKnowledge
from .match import MatchResult, play
from .policies import GenomePolicy, GreedyDamagePolicy, MLPGenome, Policy, RandomPolicy
from .teams import RosterMon, TeamGenes

SIDES = ("red", "blue")


@dataclass
class Solution:
    """One adversarial strategy: a team drawn from a campaign roster and an option-scoring MLP."""
    uid: int
    side: str
    team: TeamGenes
    mlp: MLPGenome
    origin: str = "random"
    parents: tuple[int, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {"uid": self.uid, "side": self.side, "team": self.team.to_json(), "mlp": self.mlp.to_json(),
                "origin": self.origin, "parents": list(self.parents)}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Solution:
        return cls(int(raw["uid"]), str(raw["side"]), TeamGenes.from_json(raw["team"]), MLPGenome.from_json(raw["mlp"]),
                   str(raw.get("origin", "random")), tuple(int(value) for value in raw.get("parents", [])))


@dataclass
class SideSpec:
    """Picklable description of one side's search space."""
    name: str
    game: Game
    roster: list[RosterMon]
    parties: list[list[int]]
    level: int | None = 50
    team_size: int = 6

    def build(self, dex: Dex, team: TeamGenes) -> list[MonSpec]:
        return team.build(self.roster, dex, self.level)


@dataclass
class Outcome:
    fitness: float                  # for the evaluated solution
    behavior: NDArray[np.float32]   # of the evaluated solution
    opponent_behavior: NDArray[np.float32]
    wins: int
    battles: int
    turns: float


@dataclass
class _WorkerState:
    dex: Dex
    sides: dict[str, SideSpec]
    knowledge: dict[str, SideKnowledge] = field(default_factory=dict)


_STATE: _WorkerState | None = None


def _init(sides: dict[str, SideSpec], game: Game | None) -> None:
    global _STATE
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # workers stay NumPy-only (no Torch import)
    _STATE = _WorkerState(load_dex(game), sides)


def policy_for(solution: Solution) -> Policy:
    """Anchors (fixed reference players) or the solution's evolvable option MLP."""
    if solution.origin == "anchor:greedy":
        return GreedyDamagePolicy()
    if solution.origin == "anchor:random":
        return RandomPolicy()
    return GenomePolicy(solution.mlp)


def duel(dex: Dex, sides: dict[str, SideSpec], solution: Solution, opponent: Solution,
         knowledge: dict[str, SideKnowledge], seeds: Sequence[int], learn: bool = True) -> tuple[Outcome, list[MatchResult]]:
    own_space, foe_space = sides[solution.side], sides[opponent.side]
    teams = (own_space.build(dex, solution.team), foe_space.build(dex, opponent.team))
    policies = (policy_for(solution), policy_for(opponent))
    own_k = knowledge.setdefault(solution.side, SideKnowledge())
    # Private experience: only the side being trained learns from these battles; the
    # opponent plays with a frozen copy of its knowledge (exchange shares it later).
    foe_k = SideKnowledge()
    foe_k.merge_state(knowledge.setdefault(opponent.side, SideKnowledge()).state())
    results = [play(dex, teams, policies, (own_k, foe_k), seed, learn=learn) for seed in seeds]
    fitness = float(np.mean([result.fitness for result in results]))
    behavior = np.mean([result.behavior[0] for result in results], axis=0).astype(np.float32)
    opponent_behavior = np.mean([result.behavior[1] for result in results], axis=0).astype(np.float32)
    wins = sum(1 for result in results if result.winner == 0)
    turns = float(np.mean([result.turns for result in results]))
    return Outcome(fitness, behavior, opponent_behavior, wins, len(results), turns), results


def _run_chunk(jobs: list[tuple[dict[str, Any], dict[str, Any], list[int]]],
               knowledge_states: dict[str, dict[str, Any]]) -> tuple[list[Outcome], dict[str, dict[str, Any]]]:
    assert _STATE is not None, "worker not initialised"
    knowledge = {name: SideKnowledge() for name in _STATE.sides}
    for name, state in knowledge_states.items():
        knowledge.setdefault(name, SideKnowledge()).merge_state(state)
    outcomes = []
    for solution, opponent, seeds in jobs:
        outcome, _ = duel(_STATE.dex, _STATE.sides, Solution.from_json(solution), Solution.from_json(opponent),
                          knowledge, seeds)
        outcomes.append(outcome)
    return outcomes, {name: value.state() for name, value in knowledge.items()}


class Evaluator:
    """Runs duel batches in-process (workers=0) or on a spawn process pool."""

    def __init__(self, sides: dict[str, SideSpec], game: Game | None = None, workers: int = 0,
                 repeats: int = 2, seed: int = 0) -> None:
        self.sides, self.game, self.repeats = sides, game, repeats
        self.dex = load_dex(game)
        self.knowledge: dict[str, SideKnowledge] = {name: SideKnowledge() for name in sides}
        self.rng = random.Random(seed)
        self.evaluations = 0
        self.battles = 0
        self.workers = workers
        self._pool: ProcessPoolExecutor | None = None
        self.restarts = 0
        if workers > 0:
            self._pool = self._new_pool()

    def _new_pool(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(self.workers, mp_context=get_context("spawn"), initializer=_init,
                                   initargs=(self.sides, self.game))

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None

    def __enter__(self) -> Evaluator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _seeds(self) -> list[int]:
        base = self.rng.randrange(2 ** 30)
        return [base + offset for offset in range(self.repeats)]

    def evaluate(self, pairs: Sequence[tuple[Solution, Solution]]) -> list[Outcome]:
        """Fitness and behaviours of each (solution, opponent) pair."""
        if not pairs:
            return []
        jobs = [(solution.to_json(), opponent.to_json(), self._seeds()) for solution, opponent in pairs]
        self.evaluations += len(jobs)
        self.battles += len(jobs) * self.repeats
        if self._pool is None:
            outcomes = []
            for (solution, opponent), (_, _, seeds) in zip(pairs, jobs):
                outcome, _ = duel(self.dex, self.sides, solution, opponent, self.knowledge, seeds)
                outcomes.append(outcome)
            return outcomes
        for attempt in range(3):
            try:
                return self._evaluate_pool(jobs)
            except BrokenProcessPool:
                # A worker died (killed, out of memory): rebuild the pool and replay the batch.
                if attempt == 2:
                    raise
                self.restarts += 1
                self._pool.shutdown(cancel_futures=True)
                self._pool = self._new_pool()
        raise RuntimeError("unreachable")

    def _evaluate_pool(self, jobs: list[tuple[dict[str, Any], dict[str, Any], list[int]]]) -> list[Outcome]:
        assert self._pool is not None
        states = {name: value.state() for name, value in self.knowledge.items()}
        chunk = max(1, len(jobs) // (self.workers * 3) + 1)
        futures: list[Future[tuple[list[Outcome], dict[str, dict[str, Any]]]]] = [
            self._pool.submit(_run_chunk, jobs[start:start + chunk], states) for start in range(0, len(jobs), chunk)]
        results = [future.result() for future in futures]
        outcomes = []
        for chunk_outcomes, learnt in results:
            outcomes.extend(chunk_outcomes)
            for name, state in learnt.items():
                self.knowledge.setdefault(name, SideKnowledge()).merge_state(state)
        return outcomes

    def matrix(self, rows: Sequence[Solution], cols: Sequence[Solution]) -> tuple[NDArray[np.float64], list[Outcome]]:
        """Full tournament: fitness of every row solution against every column solution."""
        pairs = [(row, col) for row in rows for col in cols]
        outcomes = self.evaluate(pairs)
        values = np.array([outcome.fitness for outcome in outcomes], dtype=np.float64).reshape(len(rows), len(cols)) \
            if rows and cols else np.zeros((len(rows), len(cols)))
        return values, outcomes
