"""Controllers that pick actions for batches of pending decisions.

Heuristic baselines reuse the stadium policies with full type-chart knowledge (``OracleKnowledge``):

* ``random``  uniformly random legal action;
* ``greedy``  ``GreedyDamagePolicy`` (highest estimated damage, switches only when forced);
* ``prior``   the ARES expert heuristic ``MLPGenome.prior`` (seed 0) played greedily (argmax).

Neural controllers run CPU forward passes only (``torch.no_grad``): they never call backward, so
they are safe inside rollout worker processes.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from core.stadium.policies import GenomePolicy, GreedyDamagePolicy, MLPGenome, RandomPolicy

from .env import BattleEnv
from .model import EntityTransformer, QNetwork, Weights, build_policy, load_numpy, policy_numpy, q_numpy
from .obs import Observation, stack

HEURISTICS = ("random", "greedy", "prior")


@dataclass
class Decision:
    env: BattleEnv
    side: int
    legal: list[int]
    obs: Observation | None = None


@dataclass
class Choice:
    action: int
    logp: float = 0.0
    value: float = 0.0


class Controller(Protocol):
    name: str
    wants_obs: bool

    def choose(self, decisions: list[Decision], rng: random.Random) -> list[Choice]: ...


@dataclass
class HeuristicController:
    name: str
    wants_obs: bool = False
    _policy: RandomPolicy | GreedyDamagePolicy | GenomePolicy = field(init=False)

    def __post_init__(self) -> None:
        if self.name == "random":
            self._policy = RandomPolicy()
        elif self.name == "greedy":
            self._policy = GreedyDamagePolicy()
        elif self.name == "prior":
            self._policy = GenomePolicy(MLPGenome.prior(random.Random(0)), name="prior", greedy=True)
        else:
            raise ValueError(f"unknown heuristic {self.name}")

    def choose(self, decisions: list[Decision], rng: random.Random) -> list[Choice]:
        return [Choice(int(self._policy.act(item.env.battle, item.side, item.env.knowledge[item.side], rng)))
                for item in decisions]


def _sample(probs: NDArray[np.float64], rng: random.Random) -> int:
    pick = rng.random()
    cumulative = 0.0
    last = 0
    for index, value in enumerate(probs):
        if value <= 0:
            continue
        last = index
        cumulative += float(value)
        if pick < cumulative:
            return index
    return last


class PolicyController:
    """The entity Transformer: samples (training) or takes the argmax (``greedy``)."""
    wants_obs = True

    def __init__(self, kind: str, weights: Weights | None = None, greedy: bool = False, name: str = "policy",
                 model: EntityTransformer | None = None) -> None:
        self.model = model if model is not None else build_policy(kind)
        self.model.eval()
        if weights is not None:
            load_numpy(self.model, weights)
        self.greedy, self.name = greedy, name

    def load(self, weights: Weights) -> None:
        load_numpy(self.model, weights)

    def choose(self, decisions: list[Decision], rng: random.Random) -> list[Choice]:
        if not decisions:
            return []
        observations = [item.obs for item in decisions]
        if any(obs is None for obs in observations):
            raise ValueError("policy controller needs observations")
        probs, values = policy_numpy(self.model, stack([obs for obs in observations if obs is not None]))
        choices: list[Choice] = []
        for row, value in zip(probs, values):
            action = int(np.argmax(row)) if self.greedy else _sample(row, rng)
            choices.append(Choice(action, float(np.log(max(row[action], 1e-12))), float(value)))
        return choices


class QController:
    """DQN agent: epsilon-greedy over masked Q-values."""
    wants_obs = True

    def __init__(self, weights: Weights | None = None, epsilon: float = 0.0, name: str = "dqn") -> None:
        self.model = QNetwork()
        self.model.eval()
        if weights is not None:
            load_numpy(self.model, weights)
        self.epsilon, self.name = epsilon, name

    def load(self, weights: Weights) -> None:
        load_numpy(self.model, weights)

    def choose(self, decisions: list[Decision], rng: random.Random) -> list[Choice]:
        if not decisions:
            return []
        observations = [item.obs for item in decisions if item.obs is not None]
        if len(observations) != len(decisions):
            raise ValueError("Q controller needs observations")
        values = q_numpy(self.model, stack(observations))
        choices: list[Choice] = []
        for item, row in zip(decisions, values):
            if rng.random() < self.epsilon:
                action = rng.choice(item.legal)
            else:
                action = int(np.argmax(row))
            choices.append(Choice(action, 0.0, float(row[action])))
        return choices


def make_neural(method: str, weights: Weights, greedy: bool, name: str) -> PolicyController | QController:
    if method == "dqn":
        return QController(weights, 0.0, name)
    return PolicyController(method, weights, greedy, name)


def make_controller(kind: str) -> HeuristicController:
    return HeuristicController(kind)


def set_worker_threads() -> None:
    torch.set_num_threads(1)
