"""Recurrent on-policy PPO, integer battle rollouts and empirical PSRO league.

There is no imitation target: gradients use actions actually sampled by the
behavior policy, their old log probabilities and observed environment rewards.
The arena is a damage-engine laboratory, not an emulator campaign certificate.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Callable, Protocol, cast

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim.adam import Adam

from core.safety.z3_integer_verifier import DamageContext, GbaIntegerSafetyVerifier
from .mamba_entity_net import AresBattlePolicyNetwork, SSMCache


class ActionDistribution:
    """Typed categorical operations, avoiding untyped torch2.4 distribution stubs."""
    def __init__(self, logits: Tensor) -> None:
        self.log_probs = F.log_softmax(logits, dim=-1)
        self.probs = self.log_probs.exp()

    def sample(self) -> Tensor:
        return torch.multinomial(self.probs, 1).squeeze(-1)

    def log_prob(self, actions: Tensor) -> Tensor:
        return self.log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    def entropy(self) -> Tensor:
        finite_lp = torch.where(torch.isfinite(self.log_probs), self.log_probs, torch.zeros_like(self.log_probs))
        return -(self.probs * finite_lp).sum(dim=-1)


@dataclass(frozen=True)
class BattleObservation:
    entities: Tensor
    roles: Tensor
    action_mask: Tensor
    target_mask: Tensor
    entity_mask: Tensor


@dataclass(frozen=True)
class EnvironmentStep:
    observation: BattleObservation
    reward: Tensor
    done: Tensor
    result: Tensor


class BattleEnvironment(Protocol):
    """Auto-reset vector environment; result is +1 win, -1 loss, zero otherwise."""
    num_envs: int

    def reset(self) -> BattleObservation:
        """Start fresh episodes and return only observable inputs."""

    def step(self, actions: Tensor, targets: Tensor) -> EnvironmentStep:
        """Execute a whole battle decision and reset terminated slots."""


class BatchedIntegerBattleArena:
    """Finite HP/PP battles using the independently validated Gen-3 integer engine.

    Each battle permutes four legal move powers. Own powers are observable via
    move metadata, enemy powers are hidden. This specifically trains ranking
    damaging moves; richer emulator environments can implement the same protocol.
    """

    def __init__(self, num_envs: int = 2, seed: int = 42,
                 opponent: AresBattlePolicyNetwork | None = None) -> None:
        if num_envs < 1:
            raise ValueError("positive environment count required")
        self.num_envs = num_envs
        self.rng = random.Random(seed)
        self.opponent = opponent
        self.hp = [[100, 100] for _ in range(num_envs)]
        self.powers: list[list[list[int]]] = [[[], []] for _ in range(num_envs)]
        self.pp = [[[10] * 4, [10] * 4] for _ in range(num_envs)]
        self.turns = [0] * num_envs
        self.opponent_cache: SSMCache | None = None
        self.episodes = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.reset()

    def _reset_slot(self, index: int) -> None:
        self.hp[index] = [100, 100]
        rival_powers = [20, 40, 80, 120] if self.opponent is not None else [20, 40, 70, 90]
        self.powers[index] = [self.rng.sample([20, 40, 80, 120], 4), self.rng.sample(rival_powers, 4)]
        self.pp[index] = [[10] * 4, [10] * 4]
        self.turns[index] = 0

    def reset(self) -> BattleObservation:
        for index in range(self.num_envs):
            self._reset_slot(index)
        self.opponent_cache = self.opponent.initial_state(self.num_envs) if self.opponent is not None else None
        return self._observe()

    def _observe(self, side: int = 0) -> BattleObservation:
        rows = torch.zeros(self.num_envs, 2, 48)
        masks = torch.zeros(self.num_envs, 9, dtype=torch.bool)
        for index in range(self.num_envs):
            for entity, owner in enumerate((side, 1 - side)):
                rows[index, entity, :6] = torch.tensor([1 / 411, 0.25, self.hp[index][owner] / 100, 0, 0, 0])
                if entity == 0:
                    rows[index, entity, 14:19] = 100 / 1024
                    move_ids = {20: 99, 40: 98, 70: 163, 80: 70, 90: 36, 120: 38}
                    rows[index, entity, 21:25] = torch.tensor([move_ids[power] for power in self.powers[index][owner]]) / 354
                    rows[index, entity, 25:29] = torch.tensor(self.pp[index][owner]) / 64
                    rows[index, entity, 29:33] = torch.tensor(self.powers[index][owner]) / 150
            masks[index, :4] = torch.tensor(self.pp[index][side]) > 0
        return BattleObservation(rows, torch.tensor([[0, 2]]).expand(self.num_envs, -1), masks,
            torch.tensor([[True, False, False]]).expand(self.num_envs, -1),
            torch.ones(self.num_envs, 2, dtype=torch.bool))

    def _damage(self, index: int, side: int, move: int) -> int:
        if not 0 <= move < 4 or self.pp[index][side][move] < 1:
            raise ValueError("arena action selected an invalid or exhausted move")
        self.pp[index][side][move] -= 1
        ctx = DamageContext(level=25, attack=100, defense=100,
                            power=self.powers[index][side][move], move_type=0, stab=True)
        return GbaIntegerSafetyVerifier.compute_context_damage_reference(ctx, self.rng.randint(85, 100), self.rng.randrange(16) == 0)

    @torch.no_grad()
    def step(self, actions: Tensor, targets: Tensor) -> EnvironmentStep:
        if actions.shape != (self.num_envs,) or targets.shape != actions.shape or torch.any(targets != 0):
            raise ValueError("arena actions[B] and left-opponent targets required")
        opponent_actions: list[int] = []
        if self.opponent is not None:
            observed = self._observe(1)
            logits, _, _, self.opponent_cache = self.opponent.forward_turn(observed.entities, observed.roles,
                observed.action_mask, self.opponent_cache, observed.target_mask, observed.entity_mask)
            opponent_actions = [int(action) for action in ActionDistribution(logits).sample().tolist()]
        else:
            for index in range(self.num_envs):
                legal = [a for a in range(4) if self.pp[index][1][a] > 0]
                opponent_actions.append(max(legal, key=lambda a: self.powers[index][1][a]))
        rewards = torch.zeros(self.num_envs)
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        result = torch.zeros(self.num_envs)
        for index in range(self.num_envs):
            action = int(actions[index])
            dealt = min(self.hp[index][1], self._damage(index, 0, action))
            self.hp[index][1] -= dealt
            taken = 0
            if self.hp[index][1] > 0:
                taken = min(self.hp[index][0], self._damage(index, 1, opponent_actions[index]))
                self.hp[index][0] -= taken
            self.turns[index] += 1
            rewards[index] = (dealt - taken) / 100
            if self.hp[index][1] == 0 or self.hp[index][0] == 0 or self.turns[index] >= 24:
                done[index] = True
                outcome = 1 if self.hp[index][1] == 0 else (-1 if self.hp[index][0] == 0 else 0)
                result[index] = outcome
                rewards[index] += 2 * outcome
                self.episodes += 1
                self.wins += outcome == 1
                self.losses += outcome == -1
                self.draws += outcome == 0
                self._reset_slot(index)
        if self.opponent_cache is not None:
            self.opponent_cache = self.opponent_cache.reset(done)
        return EnvironmentStep(self._observe(), rewards, done, result)


@dataclass(frozen=True)
class RecurrentRollout:
    observations: tuple[BattleObservation, ...]
    caches: tuple[SSMCache, ...]
    actions: Tensor
    targets: Tensor
    old_log_probs: Tensor
    values: Tensor
    rewards: Tensor
    done: Tensor
    advantages: Tensor
    returns: Tensor
    results: Tensor


class PSROLeagueTrainer:
    """Recurrent PPO best responses plus frozen empirical zero-sum PSRO policies."""

    def __init__(self, policy: AresBattlePolicyNetwork | None = None,
                 learning_rate: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_ratio: float = 0.2,
                 seed: int = 42) -> None:
        if learning_rate <= 0 or not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1 or clip_ratio <= 0:
            raise ValueError("invalid PPO hyperparameters")
        self.policy = policy or AresBattlePolicyNetwork()
        self.optimizer = Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma, self.gae_lambda, self.clip_ratio = gamma, gae_lambda, clip_ratio
        self.rng = random.Random(seed)
        self.frozen_policies: list[AresBattlePolicyNetwork] = []
        self.payoff_matrix = torch.empty(0, 0)
        self.nash_distribution = torch.empty(0)

    @torch.no_grad()
    def collect_rollout(self, environment: BattleEnvironment, steps: int = 32) -> RecurrentRollout:
        if steps < 1:
            raise ValueError("nonempty rollout required")
        observed = environment.reset()
        cache = self.policy.initial_state(environment.num_envs)
        observations: list[BattleObservation] = []
        caches: list[SSMCache] = []
        actions: list[Tensor] = []
        targets: list[Tensor] = []
        old_log_probs: list[Tensor] = []
        values: list[Tensor] = []
        rewards: list[Tensor] = []
        done: list[Tensor] = []
        results: list[Tensor] = []
        for _ in range(steps):
            observations.append(observed)
            caches.append(cache.detach())
            logits, target_logits, value, next_cache = self.policy.forward_turn(observed.entities, observed.roles,
                observed.action_mask, cache, observed.target_mask, observed.entity_mask)
            distribution, target_distribution = ActionDistribution(logits), ActionDistribution(target_logits)
            chosen, target = distribution.sample(), target_distribution.sample()
            transition = environment.step(chosen, target)
            actions.append(chosen)
            targets.append(target)
            old_log_probs.append(distribution.log_prob(chosen) + target_distribution.log_prob(target))
            values.append(value.squeeze(-1))
            rewards.append(transition.reward)
            done.append(transition.done)
            results.append(transition.result)
            observed = transition.observation
            cache = next_cache.reset(transition.done)
        _, _, last_value, _ = self.policy.forward_turn(observed.entities, observed.roles, observed.action_mask,
            cache, observed.target_mask, observed.entity_mask)
        reward_tensor, done_tensor, value_tensor = torch.stack(rewards), torch.stack(done), torch.stack(values)
        advantages = torch.zeros_like(reward_tensor)
        next_value = last_value.squeeze(-1)
        carry = torch.zeros(environment.num_envs)
        for step in range(steps - 1, -1, -1):
            alive = (~done_tensor[step]).float()
            delta = reward_tensor[step] + self.gamma * next_value * alive - value_tensor[step]
            carry = delta + self.gamma * self.gae_lambda * alive * carry
            advantages[step] = carry
            next_value = value_tensor[step]
        return RecurrentRollout(tuple(observations), tuple(caches), torch.stack(actions), torch.stack(targets),
            torch.stack(old_log_probs), value_tensor, reward_tensor, done_tensor, advantages,
            advantages + value_tensor, torch.stack(results))

    def ppo_update(self, rollout: RecurrentRollout, epochs: int = 2,
                   sequence_length: int = 32, entropy_coefficient: float = 0.01) -> dict[str, float]:
        """Optimize truncated recurrent sequences with fixed behavior likelihoods."""
        if epochs < 1 or sequence_length < 1:
            raise ValueError("positive epochs and sequence length required")
        normalized = (rollout.advantages - rollout.advantages.mean()) / rollout.advantages.std(unbiased=False).clamp_min(1e-5)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
                   "updates": 0.0, "gradient_norm": 0.0}
        for _ in range(epochs):
            for start in range(0, rollout.actions.shape[0], sequence_length):
                stop = min(start + sequence_length, rollout.actions.shape[0])
                cache = rollout.caches[start].detach()
                log_probs: list[Tensor] = []
                value_predictions: list[Tensor] = []
                entropies: list[Tensor] = []
                for step in range(start, stop):
                    obs = rollout.observations[step]
                    logits, tl, value, cache = self.policy.forward_turn(obs.entities, obs.roles, obs.action_mask,
                        cache, obs.target_mask, obs.entity_mask)
                    distribution, target_distribution = ActionDistribution(logits), ActionDistribution(tl)
                    log_probs.append(distribution.log_prob(rollout.actions[step]) + target_distribution.log_prob(rollout.targets[step]))
                    entropies.append(distribution.entropy() + target_distribution.entropy())
                    value_predictions.append(value.squeeze(-1))
                    cache = cache.reset(rollout.done[step])
                log_prob = torch.stack(log_probs)
                log_ratio = log_prob - rollout.old_log_probs[start:stop]
                ratio = log_ratio.exp()
                advantage = normalized[start:stop]
                actor_loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * advantage).mean()
                value_loss = (torch.stack(value_predictions) - rollout.returns[start:stop]).square().mean()
                entropy = torch.stack(entropies).mean()
                loss = actor_loss + 0.5 * value_loss - entropy_coefficient * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                cast(Callable[[], None], loss.backward)()
                norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite PPO gradients")
                self.optimizer.step()
                metrics["policy_loss"] += float(actor_loss.detach())
                metrics["value_loss"] += float(value_loss.detach())
                metrics["entropy"] += float(entropy.detach())
                metrics["approx_kl"] += float(((ratio - 1) - log_ratio).mean().detach())
                metrics["gradient_norm"] += float(norm)
                metrics["updates"] += 1
        updates = metrics["updates"]
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "gradient_norm"):
            metrics[key] /= updates
        completed = rollout.done.sum().clamp_min(1)
        metrics["win_rate"] = float((rollout.results == 1).sum() / completed)
        metrics["episode_return"] = float(rollout.rewards.sum() / completed)
        return metrics

    def freeze_current_policy(self) -> int:
        """Take a deep, immutable eval snapshot; live optimizer cannot affect it."""
        snapshot = copy.deepcopy(self.policy).eval()
        snapshot.requires_grad_(False)
        self.frozen_policies.append(snapshot)
        count = len(self.frozen_policies)
        expanded = torch.zeros(count, count)
        if count > 1:
            expanded[:-1, :-1] = self.payoff_matrix
        self.payoff_matrix = expanded
        self.nash_distribution = torch.full((count,), 1 / count)
        return count - 1

    @staticmethod
    def fictitious_play(payoffs: Tensor, iterations: int = 2000) -> Tensor:
        """Empirical best-response frequencies for a finite zero-sum matrix game."""
        if payoffs.ndim != 2 or payoffs.shape[0] != payoffs.shape[1] or not payoffs.shape[0] or iterations < 1 or not torch.isfinite(payoffs).all():
            raise ValueError("finite square payoff matrix and positive iterations required")
        row_counts = torch.ones(payoffs.shape[0], dtype=payoffs.dtype, device=payoffs.device)
        column_counts = torch.ones_like(row_counts)
        for _ in range(iterations):
            row = int((payoffs @ (column_counts / column_counts.sum())).argmax())
            column = int(((row_counts / row_counts.sum()) @ payoffs).argmin())
            row_counts[row] += 1
            column_counts[column] += 1
        mixture = (row_counts / row_counts.sum() + column_counts / column_counts.sum()) * 0.5
        return mixture / mixture.sum()

    @torch.no_grad()
    def evaluate(self, policy: AresBattlePolicyNetwork | None = None,
                 opponent: AresBattlePolicyNetwork | None = None, episodes: int = 32,
                 seed: int = 123, greedy: bool = True) -> dict[str, float]:
        """Actual completed HP battles; deterministic action evaluation by default."""
        if episodes < 1:
            raise ValueError("positive episode count required")
        active = policy or self.policy
        environment = BatchedIntegerBattleArena(2, seed, opponent)
        observed, cache = environment.reset(), active.initial_state(2)
        while environment.episodes < episodes:
            logits, target_logits, _, cache = active.forward_turn(observed.entities, observed.roles,
                observed.action_mask, cache, observed.target_mask, observed.entity_mask)
            actions = logits.argmax(dim=-1) if greedy else ActionDistribution(logits).sample()
            targets = target_logits.argmax(dim=-1)
            transition = environment.step(actions, targets)
            cache = cache.reset(transition.done)
            observed = transition.observation
        total = environment.episodes
        return {"win_rate": environment.wins / total, "loss_rate": environment.losses / total,
                "draw_rate": environment.draws / total, "payoff": (environment.wins - environment.losses) / total,
                "episodes": float(total)}

    def evaluate_league(self, episodes_per_match: int = 16) -> Tensor:
        """Estimate both starting-side orientations; never manufacture payoff entries."""
        for row in range(len(self.frozen_policies)):
            for col in range(row):
                forward = self.evaluate(self.frozen_policies[row], self.frozen_policies[col], episodes_per_match,
                                        seed=1000 + 31 * row + col)["payoff"]
                reverse = self.evaluate(self.frozen_policies[col], self.frozen_policies[row], episodes_per_match,
                                        seed=1000 + 31 * row + col)["payoff"]
                payoff = (forward - reverse) / 2
                self.payoff_matrix[row, col], self.payoff_matrix[col, row] = payoff, -payoff
        if self.frozen_policies:
            self.nash_distribution = self.fictitious_play(self.payoff_matrix)
        return self.payoff_matrix.clone()

    def train_best_response(self, updates: int = 8, rollout_steps: int = 32,
                            num_envs: int = 2) -> list[dict[str, float]]:
        """Train against the frozen empirical Nash mixture, or the integer baseline."""
        if updates < 1:
            raise ValueError("positive update count required")
        history: list[dict[str, float]] = []
        for _ in range(updates):
            opponent = None
            if self.frozen_policies:
                idx = self.rng.choices(range(len(self.frozen_policies)), weights=self.nash_distribution.tolist(), k=1)[0]
                opponent = self.frozen_policies[idx]
            environment = BatchedIntegerBattleArena(num_envs, self.rng.randrange(2**31), opponent)
            rollout = self.collect_rollout(environment, rollout_steps)
            history.append(self.ppo_update(rollout))
        return history


def train_neural_smoke(steps: int = 12, seed: int = 42) -> dict[str, float]:
    """Real PPO training/evaluation helper; callers separately train JEPA and TB."""
    torch.manual_seed(seed)
    trainer = PSROLeagueTrainer(seed=seed)
    before = trainer.evaluate(episodes=32)
    history = trainer.train_best_response(steps)
    after = trainer.evaluate(episodes=32)
    trainer.freeze_current_policy()
    trainer.evaluate_league(episodes_per_match=8)
    return {"initial_win_rate": before["win_rate"], "final_win_rate": after["win_rate"],
            "initial_value_loss": history[0]["value_loss"], "final_value_loss": history[-1]["value_loss"],
            "gradient_norm": history[-1]["gradient_norm"], "nash_mass": float(trainer.nash_distribution.sum())}
