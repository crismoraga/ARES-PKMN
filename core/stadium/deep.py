"""The deep channel of complementary learning (needs PyTorch).

* **Archive -> deep network** (``DeepLearner.behavior_clone``): the campaign's
  Transformer+SSM battle policy imitates the decisions of the best QD elites,
  then ``DeepLearner.ppo`` improves it against the rival's elites with the
  campaign's recurrent PPO trainer.
* **Deep network -> archive** (``DeepLearner.distill_student``): the network is
  distilled into a compact option-scoring MLP that re-enters the archive, so
  gradient-found strategies are illuminated by quality diversity.

The league imports this module only when PyTorch is installed; everything else
in the stadium (engine, GAME, lessons, dashboard) runs without it.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F

from core.battle.mamba_entity_net import AresBattlePolicyNetwork
from core.battle.psro_league_trainer import BattleObservation, EnvironmentStep, PSROLeagueTrainer

from .dex import Dex
from .engine import Battle, MonSpec
from .evaluation import SideSpec, Solution
from .features import batch_observations, net_observation, option_features
from .knowledge import SideKnowledge
from .learning import train_mlp
from .policies import GenomePolicy, MLPGenome, NeuralPolicy


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
