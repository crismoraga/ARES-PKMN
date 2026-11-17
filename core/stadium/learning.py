"""Complementary adversarial learning: what each side learns *from the other*.

**Counterfactual lessons** (``counterfactual_lessons``): at the decisions of a
duel where the two adversaries would have chosen differently, both choices are
rolled out from an exact clone of the battle with common random numbers. If the
rival's choice is clearly better, we learn it ("lo bueno del otro"); if ours is
better, the rival learns ours ("lo malo del otro", exposed). A lesson is a
supervised target on the observable option features, so it transfers to every
team that meets a similar position. ``train_mlp`` applies lessons with a small
NumPy Adam optimiser (no deep-learning framework is needed for this channel);
the league also uses it as a Lamarckian variation operator.

The deep channel (imitation of elites, recurrent PPO and distillation back to
MLP students) lives in ``core.stadium.deep`` and needs PyTorch.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .engine import Battle
from .features import OPTION_FEATURES
from .knowledge import SideKnowledge
from .match import DecisionPoint, fitness_from
from .policies import GenomePolicy, MLPGenome, Policy, parameter_count


@dataclass
class Lesson:
    learner: int                # side that should learn the target action
    features: NDArray[np.float32]
    actions: tuple[int, ...]
    target: int                 # action to imitate
    rejected: int               # the worse action
    advantage: float            # value gain of target over rejected (fitness units)
    turn: int
    kind: str                   # "imitar_rival" (learner = decider) or "corregir_rival" (learner = rival)

    def to_json(self) -> dict[str, Any]:
        return {"learner": self.learner, "target": self.target, "rejected": self.rejected,
                "advantage": round(self.advantage, 3), "turn": self.turn, "kind": self.kind}

    def to_state(self) -> dict[str, Any]:
        """Full record (with features) for the lesson bank checkpoint."""
        return {**self.to_json(), "features": np.round(self.features, 5).tolist(), "actions": list(self.actions),
                "advantage": self.advantage}

    @classmethod
    def from_state(cls, raw: dict[str, Any]) -> Lesson:
        return cls(int(raw["learner"]), np.asarray(raw["features"], dtype=np.float32),
                   tuple(int(action) for action in raw["actions"]), int(raw["target"]), int(raw["rejected"]),
                   float(raw["advantage"]), int(raw["turn"]), str(raw["kind"]))


def _rollout(snapshot: Battle, side: int, first: int, policies: tuple[Policy, Policy],
             knowledge: tuple[SideKnowledge, SideKnowledge], seed: int, horizon: int = 80) -> float:
    battle = snapshot.clone(seed=seed)
    rng = random.Random(seed)
    for policy in policies:
        policy.reset()
    first_turn = True
    steps = 0
    while not battle.over and steps < horizon:
        actions = {}
        for acting in battle.pending:
            if first_turn and acting == side:
                actions[acting] = first
            else:
                actions[acting] = policies[acting].act(battle, acting, knowledge[acting], rng)
        battle.submit(actions)
        first_turn = False
        steps += 1
    summary = battle.summary()
    if not battle.over:  # truncated rollouts are judged on HP margin
        margin = summary.hp_fraction[0] - summary.hp_fraction[1]
        value = 0.5 + 0.5 * margin
    else:
        value = fitness_from(summary.winner, summary.hp_fraction)
    return value if side == 0 else 1.0 - value


def counterfactual_lessons(decisions: Sequence[DecisionPoint], genomes: tuple[MLPGenome, MLPGenome],
                           knowledge: tuple[SideKnowledge, SideKnowledge], rng: random.Random,
                           rollouts: int = 6, margin: float = 0.04, max_points: int = 16) -> list[Lesson]:
    """Compare each side's choice with the rival policy's choice in the same position."""
    disagreements: list[tuple[DecisionPoint, int]] = []
    for point in decisions:
        if point.snapshot is None or len(point.actions) < 2:
            continue
        suggestion = int(point.actions[int(np.argmax(genomes[1 - point.side].scores(point.features)))])
        if suggestion != point.chosen:  # only positions where the two adversaries think differently
            disagreements.append((point, suggestion))
    rng.shuffle(disagreements)
    lessons: list[Lesson] = []
    policies: tuple[Policy, Policy] = (GenomePolicy(genomes[0]), GenomePolicy(genomes[1]))
    for point, suggestion in disagreements[:max_points]:
        assert point.snapshot is not None
        rival = 1 - point.side
        seeds = [rng.randrange(2 ** 30) for _ in range(rollouts)]  # common random numbers for both branches
        own_value = float(np.mean([_rollout(point.snapshot, point.side, point.chosen, policies, knowledge, seed)
                                   for seed in seeds]))
        rival_value = float(np.mean([_rollout(point.snapshot, point.side, suggestion, policies, knowledge, seed)
                                     for seed in seeds]))
        if rival_value - own_value > margin:
            lessons.append(Lesson(point.side, point.features, point.actions, suggestion, point.chosen,
                                  rival_value - own_value, point.turn, "imitar_rival"))
        elif own_value - rival_value > margin:
            lessons.append(Lesson(rival, point.features, point.actions, point.chosen, suggestion,
                                  own_value - rival_value, point.turn, "corregir_rival"))
    return lessons


# ------------------------------------------------------------ MLP training
def train_mlp(genome: MLPGenome, samples: Sequence[tuple[NDArray[np.float32], int, float]], epochs: int = 60,
              lr: float = 0.01, anchor: float = 0.02) -> tuple[MLPGenome, float]:
    """Supervised option choice (softmax over legal options), anchored to the starting weights.

    ``samples`` are (features[n_options, D], index of the target option, weight).
    Loss = weighted cross-entropy of the option scores / temperature + ``anchor`` * ||w - w0||^2,
    minimised with Adam. Returns the new genome and its final training accuracy.
    """
    if not samples:
        return genome, 0.0
    size = max(len(features) for features, _, _ in samples)
    count = len(samples)
    x = np.zeros((count, size, OPTION_FEATURES), dtype=np.float64)
    valid = np.zeros((count, size), dtype=bool)
    targets = np.zeros(count, dtype=np.int64)
    weights = np.zeros(count, dtype=np.float64)
    for row, (features, target, weight) in enumerate(samples):
        x[row, :len(features)] = features
        valid[row, :len(features)] = True
        targets[row], weights[row] = target, weight
    weights = weights / max(1e-6, float(weights.mean()))
    temperature = max(0.05, genome.temperature)
    params: list[NDArray[np.float64]] = []
    for weight_matrix, bias in genome.layers():
        params += [weight_matrix.astype(np.float64), bias.astype(np.float64)]
    start = [param.copy() for param in params]
    first = [np.zeros_like(param) for param in params]
    second = [np.zeros_like(param) for param in params]
    layers = len(params) // 2
    rows = np.arange(count)
    accuracy = 0.0
    for step in range(1, epochs + 1):
        activations = [x]
        pre: list[NDArray[np.float64]] = []
        hidden = x
        for index in range(layers):
            hidden = hidden @ params[2 * index] + params[2 * index + 1]
            pre.append(hidden)
            if index < layers - 1:
                hidden = np.where(hidden > 0, hidden, 0.01 * hidden)
                activations.append(hidden)
        logits = np.where(valid, hidden[..., 0] / temperature, -1e9)
        accuracy = float((logits.argmax(axis=1) == targets).mean())
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        delta = probabilities
        delta[rows, targets] -= 1.0
        delta = (delta * valid * weights[:, None] / (count * temperature))[..., None]
        gradients: list[NDArray[np.float64]] = [np.zeros(0)] * len(params)
        for index in range(layers - 1, -1, -1):
            gradients[2 * index] = np.einsum("bsi,bsj->ij", activations[index], delta)
            gradients[2 * index + 1] = delta.sum(axis=(0, 1))
            if index:
                delta = (delta @ params[2 * index].T) * np.where(pre[index - 1] > 0, 1.0, 0.01)
        for index, param in enumerate(params):
            gradient = gradients[index] + 2 * anchor * (param - start[index])
            first[index] = 0.9 * first[index] + 0.1 * gradient
            second[index] = 0.999 * second[index] + 0.001 * gradient * gradient
            param -= lr * (first[index] / (1 - 0.9 ** step)) / (np.sqrt(second[index] / (1 - 0.999 ** step)) + 1e-8)
    flat = np.concatenate([param.reshape(-1) for param in params]).astype(np.float32)
    assert flat.shape == (parameter_count(),)
    return MLPGenome(flat, genome.temperature), accuracy


def apply_lessons(genome: MLPGenome, lessons: Sequence[Lesson], learner: int,
                  epochs: int = 80) -> tuple[MLPGenome, float]:
    samples = [(lesson.features, lesson.actions.index(lesson.target), 0.5 + lesson.advantage * 4)
               for lesson in lessons if lesson.learner == learner and lesson.target in lesson.actions]
    return train_mlp(genome, samples, epochs=epochs)
