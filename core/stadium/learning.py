"""Complementary adversarial learning: what each side learns *from the other*.

Three gradient channels complement GAME's evolutionary illumination:

1. **Counterfactual lessons** (``counterfactual_lessons``): at sampled decisions
   of a duel, the rival's policy is asked what *it* would have done in our
   shoes. Both first actions are rolled out from an exact clone of the battle.
   If the rival's choice is clearly better, we learn it ("lo bueno del otro");
   if ours is better, the rival learns ours ("lo malo del otro", exposed). The
   lesson is a supervised target on the observable option features, so it
   transfers to every team that meets a similar position.
2. **Archive -> deep network** (``DeepLearner.behavior_clone``): the campaign's
   Transformer+SSM policy imitates the decisions of the best QD elites, then
   ``DeepLearner.ppo`` improves it against the rival's elites with recurrent PPO.
3. **Deep network -> archive** (``DeepLearner.distill_student``): the network is
   distilled into a compact option-scoring MLP that re-enters the archive as a
   new candidate, so gradient-found strategies are illuminated by QD.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F
from torch.optim.adam import Adam

from core.battle.mamba_entity_net import AresBattlePolicyNetwork
from core.battle.psro_league_trainer import BattleObservation, EnvironmentStep, PSROLeagueTrainer

from .dex import Dex
from .engine import Battle, MonSpec
from .evaluation import SideSpec, Solution
from .features import OPTION_FEATURES, batch_observations, net_observation, option_features
from .knowledge import SideKnowledge
from .match import DecisionPoint, fitness_from
from .policies import GenomePolicy, MLPGenome, NeuralPolicy, Policy, parameter_count, _shapes


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
def _torch_layers(genome: MLPGenome) -> list[torch.Tensor]:
    tensors = []
    for weight, bias in genome.layers():
        tensors += [torch.tensor(weight, requires_grad=True), torch.tensor(bias, requires_grad=True)]
    return tensors


def _forward(params: Sequence[torch.Tensor], features: torch.Tensor) -> torch.Tensor:
    hidden = features
    layers = len(params) // 2
    for index in range(layers):
        hidden = hidden @ params[2 * index] + params[2 * index + 1]
        if index < layers - 1:
            hidden = F.leaky_relu(hidden, 0.01)
    return hidden[..., 0]


def train_mlp(genome: MLPGenome, samples: Sequence[tuple[NDArray[np.float32], int, float]], epochs: int = 60,
              lr: float = 0.01, anchor: float = 0.02) -> tuple[MLPGenome, float]:
    """Supervised option choice (softmax over legal options), anchored to the starting weights.

    ``samples`` are (features[n_options, D], index of the target option, weight).
    Returns the new genome and its final training accuracy.
    """
    if not samples:
        return genome, 0.0
    size = max(len(features) for features, _, _ in samples)
    batch = np.zeros((len(samples), size, OPTION_FEATURES), dtype=np.float32)
    mask = np.zeros((len(samples), size), dtype=bool)
    targets = np.zeros(len(samples), dtype=np.int64)
    weights = np.zeros(len(samples), dtype=np.float32)
    for row, (features, target, weight) in enumerate(samples):
        batch[row, :len(features)] = features
        mask[row, :len(features)] = True
        targets[row], weights[row] = target, weight
    x, valid = torch.tensor(batch), torch.tensor(mask)
    y, w = torch.tensor(targets), torch.tensor(weights / max(1e-6, weights.mean()))
    params = _torch_layers(genome)
    start = [param.detach().clone() for param in params]
    optimizer = Adam(params, lr=lr)
    accuracy = 0.0
    for _ in range(epochs):
        logits = _forward(params, x).masked_fill(~valid, -1e9) / max(0.05, genome.temperature)
        loss = (F.cross_entropy(logits, y, reduction="none") * w).mean()
        loss = loss + anchor * sum(((param - origin) ** 2).sum() for param, origin in zip(params, start))
        optimizer.zero_grad()
        cast(Callable[[], None], loss.backward)()
        optimizer.step()
        accuracy = float((logits.argmax(dim=-1) == y).float().mean())
    flat = torch.cat([param.detach().reshape(-1) for param in params]).numpy().astype(np.float32)
    assert flat.shape == (parameter_count(),) and len(params) == 2 * len(_shapes())
    return MLPGenome(flat, genome.temperature), accuracy


def apply_lessons(genome: MLPGenome, lessons: Sequence[Lesson], learner: int,
                  epochs: int = 80) -> tuple[MLPGenome, float]:
    samples = [(lesson.features, lesson.actions.index(lesson.target), 0.5 + lesson.advantage * 4)
               for lesson in lessons if lesson.learner == learner and lesson.target in lesson.actions]
    return train_mlp(genome, samples, epochs=epochs)


# ------------------------------------------------------- deep RL learner
class StadiumEnvironment:
    """Vectorised ``BattleEnvironment``: the learner is side 0, opponents are elite MLPs."""

    def __init__(self, dex: Dex, learner_teams: Sequence[list[MonSpec]],
                 opponents: Sequence[tuple[list[MonSpec], MLPGenome]], knowledge: SideKnowledge,
                 num_envs: int = 8, seed: int = 0, max_turns: int = 120) -> None:
        if not learner_teams or not opponents:
            raise ValueError("learner teams and opponents are required")
        self.dex, self.learner_teams, self.opponents = dex, list(learner_teams), list(opponents)
        self.knowledge, self.num_envs, self.max_turns = knowledge, num_envs, max_turns
        self.rng = random.Random(seed)
        self.opponent_knowledge = SideKnowledge()
        self.opponent_knowledge.merge_state(knowledge.state())
        self.battles: list[Battle] = []
        self.policies: list[GenomePolicy] = []
        self.margins: list[float] = []
        self.episodes = self.wins = self.losses = self.draws = 0

    def _new(self) -> tuple[Battle, GenomePolicy]:
        team, (foe_team, genome) = self.rng.choice(self.learner_teams), self.rng.choice(self.opponents)
        battle = Battle(self.dex, team, foe_team, seed=self.rng.randrange(2 ** 30), max_turns=self.max_turns,
                        record=False)
        return battle, GenomePolicy(genome)

    def _advance(self, index: int) -> None:
        battle, policy = self.battles[index], self.policies[index]
        while not battle.over and 0 not in battle.pending:
            battle.submit({1: policy.act(battle, 1, self.opponent_knowledge, self.rng)})

    def _margin(self, battle: Battle) -> float:
        return battle.sides[0].hp_fraction() - battle.sides[1].hp_fraction()

    def _observe(self) -> BattleObservation:
        observations = [net_observation(battle, 0, self.knowledge, battle.legal_actions(0)) for battle in self.battles]
        entities, roles, masks, targets, present = batch_observations(observations)
        return BattleObservation(entities, roles, masks, targets, present)

    def reset(self) -> BattleObservation:
        self.battles, self.policies, self.margins = [], [], []
        for index in range(self.num_envs):
            battle, policy = self._new()
            self.battles.append(battle)
            self.policies.append(policy)
            self.margins.append(self._margin(battle))
            self._advance(index)
        return self._observe()

    def step(self, actions: torch.Tensor, targets: torch.Tensor) -> EnvironmentStep:
        rewards = torch.zeros(self.num_envs)
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        result = torch.zeros(self.num_envs)
        for index, battle in enumerate(self.battles):
            chosen = int(actions[index])
            legal = battle.legal_actions(0)
            if chosen not in legal:
                chosen = legal[0]
            submit = {0: chosen}
            if 1 in battle.pending:
                submit[1] = self.policies[index].act(battle, 1, self.opponent_knowledge, self.rng)
            events = battle.submit(submit)
            self.knowledge.observe(self.dex, events, 0, {side: battle.sides[side].fighter.spec.species for side in (0, 1)})
            self._advance(index)
            margin = self._margin(battle)
            rewards[index] = margin - self.margins[index]
            self.margins[index] = margin
            if battle.over:
                outcome = 1.0 if battle.winner == 0 else -1.0 if battle.winner == 1 else 0.0
                rewards[index] += 2 * outcome
                done[index], result[index] = True, outcome
                self.episodes += 1
                self.wins += outcome > 0
                self.losses += outcome < 0
                self.draws += outcome == 0
                fresh, policy = self._new()
                self.battles[index], self.policies[index] = fresh, policy
                self.margins[index] = self._margin(fresh)
                self._advance(index)
        return EnvironmentStep(self._observe(), rewards, done, result)


class DeepLearner:
    """One side's Transformer+SSM policy, trained by imitation of elites and recurrent PPO."""

    def __init__(self, network: AresBattlePolicyNetwork | None = None, learning_rate: float = 3e-4,
                 seed: int = 0) -> None:
        torch.manual_seed(seed)
        self.trainer = PSROLeagueTrainer(network or AresBattlePolicyNetwork(), learning_rate=learning_rate, seed=seed)
        self.updates = 0
        self.history: list[dict[str, float]] = []

    @property
    def network(self) -> AresBattlePolicyNetwork:
        return self.trainer.policy

    @classmethod
    def from_campaign(cls, path: Path, seed: int = 0) -> DeepLearner:
        """Start from the campaign's trained battle policy when ``neural.pt`` exists."""
        network = AresBattlePolicyNetwork()
        if path.is_file():
            state = torch.load(path, map_location="cpu", weights_only=False)
            weights = state.get("battle_policy") if isinstance(state, dict) else None
            if isinstance(weights, dict):
                network.load_state_dict(weights)
        return cls(network, seed=seed)

    def behavior_clone(self, samples: Sequence[tuple[tuple[Any, ...], int, float]], epochs: int = 3,
                       batch_size: int = 128) -> float:
        """Imitate elite decisions (fresh recurrent state per decision); returns accuracy."""
        if not samples:
            return 0.0
        network = self.network.train()
        optimizer = self.trainer.optimizer
        accuracy = 0.0
        order = list(range(len(samples)))
        rng = random.Random(len(samples))
        for _ in range(epochs):
            rng.shuffle(order)
            correct = 0
            for start in range(0, len(order), batch_size):
                chunk = [samples[index] for index in order[start:start + batch_size]]
                entities, roles, masks, targets, present = batch_observations([item[0] for item in chunk])
                labels = torch.tensor([item[1] for item in chunk])
                weights = torch.tensor([item[2] for item in chunk], dtype=torch.float32)
                logits, _, _, _ = network.forward_turn(entities, roles, masks, None, targets, present)
                loss = (F.cross_entropy(logits.masked_fill(~masks, -1e9), labels, reduction="none") * weights).mean()
                optimizer.zero_grad(set_to_none=True)
                cast(Callable[[], None], loss.backward)()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
                optimizer.step()
                correct += int((logits.argmax(dim=-1) == labels).sum())
            accuracy = correct / len(samples)
        network.eval()
        return accuracy

    def ppo(self, environment: StadiumEnvironment, updates: int = 2, steps: int = 32) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for _ in range(updates):
            rollout = self.trainer.collect_rollout(environment, steps)
            metrics = self.trainer.ppo_update(rollout, epochs=2, sequence_length=16)
            self.updates += 1
        metrics["episodes"] = float(environment.episodes)
        metrics["env_win_rate"] = environment.wins / max(1, environment.episodes)
        self.history.append(metrics)
        return metrics

    def distill_student(self, dex: Dex, own_teams: Sequence[list[MonSpec]],
                        opponents: Sequence[tuple[list[MonSpec], MLPGenome]], start: MLPGenome,
                        knowledge: SideKnowledge, battles: int = 12, seed: int = 0) -> tuple[MLPGenome, float]:
        """Network -> option MLP: imitate the network's choices on observable features."""
        rng = random.Random(seed)
        neural = NeuralPolicy(self.network, greedy=True)
        samples: list[tuple[NDArray[np.float32], int, float]] = []
        foe_knowledge = SideKnowledge()
        foe_knowledge.merge_state(knowledge.state())
        for _ in range(battles):
            team = rng.choice(list(own_teams))
            foe_team, genome = rng.choice(list(opponents))
            battle = Battle(dex, team, foe_team, seed=rng.randrange(2 ** 30), max_turns=100, record=False)
            foe = GenomePolicy(genome)
            neural.reset()
            while not battle.over:
                actions = {}
                for side in battle.pending:
                    legal = battle.legal_actions(side)
                    if side == 0:
                        chosen = neural.act(battle, 0, knowledge, rng)
                        if len(legal) > 1:
                            samples.append((option_features(battle, 0, knowledge, legal), legal.index(chosen), 1.0))
                        actions[0] = chosen
                    else:
                        actions[1] = foe.act(battle, 1, foe_knowledge, rng)
                battle.submit(actions)
        return train_mlp(start, samples, epochs=80, anchor=0.005)


def elite_imitation_samples(dex: Dex, decider_teams: Sequence[tuple[list[MonSpec], MLPGenome, float]],
                            opponents: Sequence[tuple[list[MonSpec], MLPGenome]], knowledge: SideKnowledge,
                            battles: int = 16, seed: int = 0) -> list[tuple[tuple[Any, ...], int, float]]:
    """Decisions of elite MLPs encoded as deep-policy observations (QD -> RL)."""
    rng = random.Random(seed)
    samples: list[tuple[tuple[Any, ...], int, float]] = []
    foe_knowledge = SideKnowledge()
    foe_knowledge.merge_state(knowledge.state())
    for _ in range(battles):
        team, genome, weight = rng.choice(list(decider_teams))
        foe_team, foe_genome = rng.choice(list(opponents))
        battle = Battle(dex, team, foe_team, seed=rng.randrange(2 ** 30), max_turns=100, record=False)
        policies = (GenomePolicy(genome, greedy=True), GenomePolicy(foe_genome))
        while not battle.over:
            actions = {}
            for side in battle.pending:
                legal = battle.legal_actions(side)
                chosen = policies[side].act(battle, side, knowledge if side == 0 else foe_knowledge, rng)
                if side == 0 and len(legal) > 1:
                    samples.append((net_observation(battle, 0, knowledge, legal), chosen, weight))
                actions[side] = chosen
            battle.submit(actions)
    return samples


def team_specs(spec: SideSpec, dex: Dex, solutions: Sequence[Solution]) -> list[list[MonSpec]]:
    return [spec.build(dex, solution.team) for solution in solutions]
