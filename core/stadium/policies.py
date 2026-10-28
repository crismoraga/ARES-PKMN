"""Battle policies of the stadium.

* ``MLPGenome`` - the evolvable solution of GAME: a compact option-scoring MLP
  (25 -> 16 -> 8 -> 1) shared by every legal action, plus a softmax
  temperature. Pokemon turns are simultaneous moves, so a *mixed* strategy can
  be optimal; the temperature lets evolution find how much to randomise.
* ``prior`` genomes encode a sensible expert heuristic through an exact ReLU
  pass-through, so illumination starts from competent play, not from noise.
* ``NeuralPolicy`` drives the campaign's ``AresBattlePolicyNetwork``
  (Transformer + selective SSM) with the stadium's observation tensors.
* ``HybridPolicy`` adds the deep network's logits to the MLP scores: the
  complementary channel between gradient learning and quality diversity.
"""
from __future__ import annotations

import math
import random
from random import Random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # Torch is imported lazily: evaluation workers only run NumPy MLPs.
    import torch

    from core.battle.mamba_entity_net import AresBattlePolicyNetwork, SSMCache

from .engine import Battle
from .features import FEATURE_NAMES, OPTION_FEATURES, batch_observations, net_observation, option_features
from .knowledge import SideKnowledge

HIDDEN = (16, 8)
_PRIOR = {"dmg_cur": 1.0, "ko_prob": 1.3, "accuracy": 0.35, "eff_log2": 0.25, "inflicts_status": 0.55,
          "self_boost": 0.3, "heal_value": 1.1, "priority": 0.15, "is_switch": -0.85, "switch_offense": 0.6,
          "switch_hp": 0.45, "threat": -0.35, "commitment": -0.25, "pp_frac": 0.05, "is_status": -0.15,
          "foe_drop": 0.1}
Array = NDArray[np.float32]


def _shapes(dimension: int = OPTION_FEATURES) -> list[tuple[int, int]]:
    sizes = (dimension, *HIDDEN, 1)
    return [(sizes[index], sizes[index + 1]) for index in range(len(sizes) - 1)]


def parameter_count(dimension: int = OPTION_FEATURES) -> int:
    return sum(rows * cols + cols for rows, cols in _shapes(dimension))


@dataclass
class MLPGenome:
    """Flat weights of the option scorer and its sampling temperature."""
    weights: Array
    temperature: float = 0.15

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=np.float32)
        if self.weights.shape != (parameter_count(),):
            raise ValueError("unexpected MLP genome size")
        self.temperature = float(min(1.0, max(0.02, self.temperature)))

    def layers(self) -> list[tuple[Array, Array]]:
        result: list[tuple[Array, Array]] = []
        offset = 0
        for rows, cols in _shapes():
            weight = self.weights[offset:offset + rows * cols].reshape(rows, cols)
            offset += rows * cols
            bias = self.weights[offset:offset + cols]
            offset += cols
            result.append((weight, bias))
        return result

    def scores(self, features: Array) -> Array:
        hidden = features
        layers = self.layers()
        for index, (weight, bias) in enumerate(layers):
            hidden = hidden @ weight + bias
            if index < len(layers) - 1:
                hidden = np.where(hidden > 0, hidden, 0.01 * hidden)
        return hidden[:, 0]

    def mutate(self, rng: Random, fraction: float = 0.3, sigma: float = 0.1) -> MLPGenome:
        """The paper's operator: Gaussian noise (std ``sigma``) on ``fraction`` of the weights."""
        generator = np.random.default_rng(rng.randrange(2 ** 32))
        mask = generator.random(self.weights.shape) < fraction
        noise = generator.normal(0.0, sigma, self.weights.shape).astype(np.float32)
        temperature = self.temperature * math.exp(rng.gauss(0.0, 0.25)) if rng.random() < 0.3 else self.temperature
        return MLPGenome(self.weights + mask * noise, temperature)

    def crossover(self, other: MLPGenome, rng: Random) -> MLPGenome:
        generator = np.random.default_rng(rng.randrange(2 ** 32))
        take = generator.random(self.weights.shape) < 0.5
        return MLPGenome(np.where(take, self.weights, other.weights), (self.temperature + other.temperature) / 2)

    def to_json(self) -> dict[str, Any]:
        return {"weights": [round(float(value), 6) for value in self.weights], "temperature": self.temperature}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> MLPGenome:
        return cls(np.asarray(raw["weights"], dtype=np.float32), float(raw["temperature"]))

    @classmethod
    def random(cls, rng: Random, scale: float = 0.35) -> MLPGenome:
        generator = np.random.default_rng(rng.randrange(2 ** 32))
        return cls(generator.normal(0.0, scale, parameter_count()).astype(np.float32), rng.uniform(0.05, 0.5))

    @classmethod
    def prior(cls, rng: Random, noise: float = 0.05, overrides: dict[str, float] | None = None) -> MLPGenome:
        """Expert heuristic h(x)=w.x embedded exactly: relu(w.x)-relu(-w.x), other units small noise."""
        generator = np.random.default_rng(rng.randrange(2 ** 32))
        vector = np.zeros(OPTION_FEATURES, dtype=np.float32)
        for name, value in {**_PRIOR, **(overrides or {})}.items():
            vector[FEATURE_NAMES.index(name)] = value
        shapes = _shapes()
        parts: list[Array] = []
        first = generator.normal(0.0, noise, shapes[0]).astype(np.float32)
        first[:, 0], first[:, 1] = vector, -vector
        parts += [first.ravel(), np.zeros(shapes[0][1], dtype=np.float32)]
        second = generator.normal(0.0, noise, shapes[1]).astype(np.float32)
        second[0, :], second[1, :] = 0.0, 0.0
        second[0, 0], second[1, 0], second[0, 1], second[1, 1] = 1.0, -1.0, -1.0, 1.0
        parts += [second.ravel(), np.zeros(shapes[1][1], dtype=np.float32)]
        third = generator.normal(0.0, noise, shapes[2]).astype(np.float32)
        third[0, 0], third[1, 0] = 1.0, -1.0
        parts += [third.ravel(), np.zeros(shapes[2][1], dtype=np.float32)]
        return cls(np.concatenate(parts), rng.uniform(0.05, 0.25))


class Policy(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(self, battle: Battle, side: int, knowledge: SideKnowledge, rng: Random) -> int: ...


def choose(scores: Array, actions: Sequence[int], temperature: float, rng: Random, greedy: bool) -> int:
    if greedy or temperature <= 0.021:
        return int(actions[int(np.argmax(scores))])
    logits = (scores - scores.max()) / temperature
    weights = np.exp(logits)
    pick = rng.random() * float(weights.sum())
    for action, weight in zip(actions, weights):
        pick -= float(weight)
        if pick <= 0:
            return int(action)
    return int(actions[-1])


@dataclass
class GenomePolicy:
    genome: MLPGenome
    name: str = "mlp"
    greedy: bool = False
    last_features: Array | None = field(default=None, repr=False)

    def reset(self) -> None:
        self.last_features = None

    def act(self, battle: Battle, side: int, knowledge: SideKnowledge, rng: Random) -> int:
        actions = battle.legal_actions(side)
        if len(actions) == 1:
            return actions[0]
        features = option_features(battle, side, knowledge, actions)
        self.last_features = features
        return choose(self.genome.scores(features), actions, self.genome.temperature, rng, self.greedy)


@dataclass
class RandomPolicy:
    name: str = "random"

    def reset(self) -> None:
        return None

    def act(self, battle: Battle, side: int, knowledge: SideKnowledge, rng: Random) -> int:
        return rng.choice(battle.legal_actions(side))


@dataclass
class GreedyDamagePolicy:
    """Anchor: always the move with the highest believed damage; switches only when forced."""
    name: str = "greedy"

    def reset(self) -> None:
        return None

    def act(self, battle: Battle, side: int, knowledge: SideKnowledge, rng: Random) -> int:
        actions = battle.legal_actions(side)
        moves = [action for action in actions if action < 4]
        if len(actions) == 1 or not moves:
            if not moves:
                features = option_features(battle, side, knowledge, actions)
                return int(actions[int(np.argmax(features[:, 22] + 0.5 * features[:, 23] - 0.3 * features[:, 18]))])
            return actions[0]
        features = option_features(battle, side, knowledge, moves)
        return int(moves[int(np.argmax(features[:, 1] * features[:, 4] + 0.01 * rng.random()))])


class NeuralPolicy:
    """The campaign's recurrent battle network acting in the stadium."""

    def __init__(self, network: "AresBattlePolicyNetwork", greedy: bool = True, name: str = "neural") -> None:
        self.network = network.eval()
        self.greedy, self.name = greedy, name
        self.cache: "SSMCache | None" = None

    def reset(self) -> None:
        self.cache = None

    def logits(self, battle: Battle, side: int, knowledge: SideKnowledge, actions: Sequence[int]) -> "torch.Tensor":
        import torch

        with torch.no_grad():
            entities, roles, mask, targets, present = batch_observations(
                [net_observation(battle, side, knowledge, actions)])
            logits, _, _, cache = self.network.forward_turn(entities, roles, mask, self.cache, targets, present)
        self.cache = cache.detach()
        return logits[0]

    def act(self, battle: Battle, side: int, knowledge: SideKnowledge, rng: Random) -> int:
        actions = battle.legal_actions(side)
        logits = self.logits(battle, side, knowledge, actions)
        if len(actions) == 1:
            return actions[0]
        values = np.array([float(logits[action]) for action in actions], dtype=np.float32)
        return choose(values, actions, 1.0, rng, self.greedy)


class HybridPolicy:
    """MLP option scores plus ``weight`` times the deep network's action logits."""

    def __init__(self, genome: MLPGenome, network: "AresBattlePolicyNetwork", weight: float = 0.5,
                 name: str = "hybrid") -> None:
        self.mlp = GenomePolicy(genome)
        self.neural = NeuralPolicy(network)
        self.weight, self.name = weight, name

    def reset(self) -> None:
        self.mlp.reset()
        self.neural.reset()

    def act(self, battle: Battle, side: int, knowledge: SideKnowledge, rng: Random) -> int:
        actions = battle.legal_actions(side)
        logits = self.neural.logits(battle, side, knowledge, actions)
        if len(actions) == 1:
            return actions[0]
        scores = self.mlp.genome.scores(option_features(battle, side, knowledge, actions))
        scores = scores + self.weight * np.array([float(logits[action]) for action in actions], dtype=np.float32)
        return choose(scores, actions, self.mlp.genome.temperature, rng, False)
