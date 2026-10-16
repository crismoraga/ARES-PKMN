"""Causal type discovery and observable-only Bayesian opponent estimates.

The eighteen RAM type IDs include the unused ``TYPE_MYSTERY`` at index nine.
Type efficacy is a categorical latent variable, rather than a running average of
damage. Dual-type observations update a joint distribution, so a total neutral
hit does not incorrectly label both component matchups neutral. Residual damage
and status-only moves are removed before applying an intervention.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

TYPE_MYSTERY = 9
PLAYABLE_TYPES = tuple(type_id for type_id in range(18) if type_id != TYPE_MYSTERY)
EFFICACIES: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
FloatArray = NDArray[np.float64]


def _type_id(value: int) -> None:
    if not 0 <= value < 18:
        raise ValueError("A generation-three type ID must lie in [0, 17]")


def _entropy(probabilities: FloatArray) -> float:
    values = probabilities[probabilities > 0.0]
    return float(-np.sum(values * np.log(values)))


@dataclass(frozen=True)
class DamageObservation:
    """An identified direct-hit intervention, with measured residual separated.

    ``expected_neutral_damage`` must include the known critical/STAB/stage/field
    modifiers but exclude type efficacy. ``defender_hp_before`` permits censored
    observations when a successful attack knocks out its target. Unknown move
    category or unknown non-type modifier should not be submitted as identified
    evidence; set ``identified=False`` instead.
    """

    attack_type: int
    defender_types: tuple[int, ...]
    observed_delta_hp: float
    expected_neutral_damage: float
    residual_damage: float = 0.0
    engine_text_flag: str = ""
    is_status_move: bool = False
    identified: bool = True
    defender_hp_before: float | None = None
    relative_noise: float = 0.08


@dataclass(frozen=True)
class HiddenOpponentSet:
    """A hypothesized legal opponent set; no hidden RAM observation is required."""

    identifier: str
    species: int
    moves: tuple[int, ...]
    ability: int | None = None
    item: int | None = None
    attack: int = 1
    special_attack: int = 1
    defense: int = 1
    special_defense: int = 1
    speed: int = 1


class BeliefStateEstimator:
    """Finite Bayesian filtering with explicit likelihoods and normalization.

    Impossible evidence raises instead of silently restoring the prior. This is
    useful for detecting an incomplete candidate set or an incorrect observation
    decoder. Likelihoods can represent a move reveal, ability animation, item
    reveal, speed-order interval, or an independently predicted damage interval.
    """

    def __init__(
        self,
        hypotheses: Sequence[HiddenOpponentSet],
        prior: Sequence[float] | None = None,
    ) -> None:
        if not hypotheses:
            raise ValueError("At least one opponent-set hypothesis is required")
        identifiers = [hypothesis.identifier for hypothesis in hypotheses]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Opponent-set identifiers must be unique")
        self.hypotheses = tuple(hypotheses)
        weights = np.asarray(
            prior if prior is not None else [1.0] * len(hypotheses), dtype=np.float64
        )
        if weights.shape != (len(hypotheses),):
            raise ValueError("Prior dimensions do not match the candidate sets")
        self._probabilities = self._normalize(weights)
        self.evidence_count = 0

    @staticmethod
    def _normalize(weights: FloatArray) -> FloatArray:
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("Bayesian weights must be finite and nonnegative")
        total = float(np.sum(weights))
        if total <= 0.0:
            raise ValueError("Evidence has zero probability under every candidate set")
        return weights / total

    @property
    def probabilities(self) -> dict[str, float]:
        return {
            hypothesis.identifier: float(probability)
            for hypothesis, probability in zip(
                self.hypotheses, self._probabilities, strict=True
            )
        }

    @property
    def entropy(self) -> float:
        return _entropy(self._probabilities)

    def update_evidence(self, likelihoods: Mapping[str, float]) -> float:
        """Apply P(observation | candidate) and return the realized KL gain."""
        expected = {hypothesis.identifier for hypothesis in self.hypotheses}
        if set(likelihoods) != expected:
            raise ValueError("Likelihoods must contain exactly all candidate identifiers")
        likelihood = np.asarray(
            [likelihoods[hypothesis.identifier] for hypothesis in self.hypotheses],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(likelihood)) or np.any(likelihood < 0.0):
            raise ValueError("Likelihoods must be finite and nonnegative")
        previous = self._probabilities.copy()
        updated = self._normalize(previous * likelihood)
        support = updated > 0.0
        gain = float(np.sum(updated[support] * np.log(updated[support] / previous[support])))
        self._probabilities = updated
        self.evidence_count += 1
        return gain

    def condition(self, likelihood: Callable[[HiddenOpponentSet], float]) -> float:
        return self.update_evidence(
            {hypothesis.identifier: likelihood(hypothesis) for hypothesis in self.hypotheses}
        )

    def observe_move(self, move_id: int, false_positive_rate: float = 0.001) -> float:
        if move_id <= 0 or not 0.0 <= false_positive_rate < 1.0:
            raise ValueError("Move ID and observation error rate are outside their domains")
        return self.condition(
            lambda hypothesis: 1.0 if move_id in hypothesis.moves else false_positive_rate
        )

    def observe_ability(self, ability_id: int, false_positive_rate: float = 0.001) -> float:
        if ability_id < 0 or not 0.0 <= false_positive_rate < 1.0:
            raise ValueError("Ability ID and observation error rate are outside their domains")
        return self.condition(
            lambda hypothesis: 1.0
            if hypothesis.ability in (None, ability_id)
            else false_positive_rate
        )

    def observe_item(self, item_id: int, false_positive_rate: float = 0.001) -> float:
        if item_id < 0 or not 0.0 <= false_positive_rate < 1.0:
            raise ValueError("Item ID and observation error rate are outside their domains")
        return self.condition(
            lambda hypothesis: 1.0
            if hypothesis.item in (None, item_id)
            else false_positive_rate
        )

    def expected_stat(self, attribute: str) -> float:
        if attribute not in {
            "attack", "special_attack", "defense", "special_defense", "speed"
        }:
            raise ValueError("Only numeric combat stats can be averaged")
        return sum(
            float(getattr(hypothesis, attribute)) * float(probability)
            for hypothesis, probability in zip(
                self.hypotheses, self._probabilities, strict=True
            )
        )


class EpistemicCausalController:
    """A categorical structural causal model and risk-sensitive active inference.

    For every attack/defense pair, ``posterior[a,d,k]`` represents probability of
    the k-th efficacy in (0, .5, 1, 2). Pair factors retain dual-type ambiguity.
    The RAM's unused type ID nine is pinned neutral; it never denotes Fairy.
    """

    def __init__(self, prior: Sequence[float] = (0.25, 0.25, 0.25, 0.25)) -> None:
        weights = np.asarray(prior, dtype=np.float64)
        if weights.shape != (4,) or np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
            raise ValueError("The efficacy prior needs four finite nonnegative probabilities")
        if float(weights.sum()) <= 0.0:
            raise ValueError("An efficacy prior cannot have zero mass")
        self.posterior: FloatArray = np.tile(weights / weights.sum(), (18, 18, 1))
        self.posterior[TYPE_MYSTERY, :, :] = np.asarray([0.0, 0.0, 1.0, 0.0])
        self.posterior[:, TYPE_MYSTERY, :] = np.asarray([0.0, 0.0, 1.0, 0.0])
        self._pairs: dict[tuple[int, int, int], FloatArray] = {}
        self._pair_marginals: dict[tuple[int, int, int], tuple[FloatArray, FloatArray]] = {}
        self.intervention_count: NDArray[np.int64] = np.zeros((18, 18), dtype=np.int64)
        self.ignored_observations = 0

    @property
    def scm_matrix(self) -> FloatArray:
        return np.asarray(np.sum(self.posterior * np.asarray(EFFICACIES), axis=-1), dtype=np.float64)

    @property
    def type_matrix(self) -> FloatArray:
        """Posterior-mean eighteen by eighteen efficacy table (a fresh array)."""
        return self.scm_matrix

    @staticmethod
    def _defenders(defender_types: Sequence[int]) -> tuple[int, ...]:
        if not 1 <= len(defender_types) <= 2:
            raise ValueError("A defender must have one or two RAM type IDs")
        for type_id in defender_types:
            _type_id(type_id)
        return tuple(sorted(set(defender_types)))

    def _joint(self, attack_type: int, defenders: tuple[int, ...]) -> FloatArray:
        if len(defenders) == 1:
            return np.asarray(self.posterior[attack_type, defenders[0]], dtype=np.float64).copy()
        first, second = defenders
        key = (attack_type, first, second)
        current_first = self.posterior[attack_type, first]
        current_second = self.posterior[attack_type, second]
        joint = np.outer(current_first, current_second)
        if key in self._pairs:
            old_first, old_second = self._pair_marginals[key]
            old_independent = np.outer(old_first, old_second)
            correlation = np.divide(
                self._pairs[key], old_independent,
                out=np.ones((4, 4), dtype=np.float64), where=old_independent > 0.0,
            )
            joint *= correlation
        total = float(joint.sum())
        if total <= 0.0:
            raise ArithmeticError("Inconsistent joint type belief has no support")
        return joint / total

    @staticmethod
    def _products(defenders: tuple[int, ...]) -> FloatArray:
        values = np.asarray(EFFICACIES, dtype=np.float64)
        return values if len(defenders) == 1 else np.outer(values, values)

    def _propagate_joint_evidence(
        self,
        attack_type: int,
        changed_types: tuple[int, ...],
        source_pair: tuple[int, int, int] | None = None,
    ) -> None:
        """Transfer revealed component evidence through retained pair factors.

        A local calibration sweep is exact for tree-shaped pair evidence. Each
        factor is visited once per intervention, preventing cyclic observations
        from being counted repeatedly as fresh independent evidence. Larger
        cyclic networks are a calibrated pair approximation, not an asserted
        full eighteen-variable joint posterior.
        """
        frontier = list(changed_types)
        visited: set[tuple[int, int, int]] = set()
        if source_pair is not None:
            visited.add(source_pair)
        while frontier:
            changed = frontier.pop(0)
            for key in tuple(self._pairs):
                if key in visited or key[0] != attack_type or changed not in key[1:]:
                    continue
                visited.add(key)
                first, second = key[1], key[2]
                joint = self._joint(attack_type, (first, second))
                first_marginal = joint.sum(axis=1)
                second_marginal = joint.sum(axis=0)
                self.posterior[attack_type, first] = first_marginal
                self.posterior[attack_type, second] = second_marginal
                self._pairs[key] = joint
                self._pair_marginals[key] = (first_marginal.copy(), second_marginal.copy())
                frontier.extend(type_id for type_id in (first, second) if type_id != changed)

    @staticmethod
    def _text_likelihood(flag: str, products: FloatArray) -> FloatArray:
        normalized = flag.strip().upper().replace(" ", "_").replace("-", "_")
        if normalized in {"NO_EFFECT", "IMMUNE", "IMMUNITY", "DOESNT_AFFECT", "DOES_NOT_AFFECT"}:
            consistent = products == 0.0
        elif normalized in {"SUPER_EFFECTIVE", "SUPEREFFECTIVE", "SUPER"}:
            consistent = products > 1.0
        elif normalized in {"NOT_VERY_EFFECTIVE", "NOT_EFFECTIVE", "RESISTED", "RESIST"}:
            consistent = (products > 0.0) & (products < 1.0)
        elif normalized in {"NEUTRAL", "NORMAL_EFFECTIVE"}:
            consistent = products == 1.0
        else:
            return np.ones_like(products)
        return np.where(consistent, 1.0, 0.001)

    def apply_do_calculus(
        self,
        t_atk: int,
        t_def: int,
        observed_delta_hp: float,
        expected_neutral_dmg: float,
        engine_text_flag: str,
    ) -> None:
        """Compatibility intervention for a single, isolated defender type."""
        self.observe_intervention(
            DamageObservation(
                t_atk, (t_def,), observed_delta_hp, expected_neutral_dmg,
                engine_text_flag=engine_text_flag,
            )
        )

    def observe_intervention(self, observation: DamageObservation) -> float:
        """Update an identified intervention and return the posterior KL gain."""
        _type_id(observation.attack_type)
        defenders = self._defenders(observation.defender_types)
        numbers = (
            observation.observed_delta_hp, observation.expected_neutral_damage,
            observation.residual_damage, observation.relative_noise,
        )
        if not all(math.isfinite(number) for number in numbers):
            raise ValueError("Damage evidence must contain finite measurements")
        if observation.observed_delta_hp < 0.0 or observation.residual_damage < 0.0:
            raise ValueError("Damage evidence cannot contain negative HP loss")
        if observation.residual_damage > observation.observed_delta_hp:
            raise ValueError("Residual HP loss cannot exceed total observed HP loss")
        if observation.relative_noise <= 0.0:
            raise ValueError("Observation noise must be positive")
        if observation.defender_hp_before is not None and (
            not math.isfinite(observation.defender_hp_before)
            or observation.defender_hp_before <= 0.0
        ):
            raise ValueError("A censoring HP bound must be finite and positive")
        if (
            observation.is_status_move or not observation.identified
            or observation.expected_neutral_damage <= 0.0
            or observation.attack_type == TYPE_MYSTERY
        ):
            self.ignored_observations += 1
            return 0.0
        previous = self._joint(observation.attack_type, defenders)
        products = self._products(defenders)
        direct_damage = observation.observed_delta_hp - observation.residual_damage
        ratio = direct_damage / observation.expected_neutral_damage
        sigma = max(observation.relative_noise, 1.5 / observation.expected_neutral_damage)
        distance = products - ratio
        if (
            observation.defender_hp_before is not None
            and direct_damage >= observation.defender_hp_before
        ):
            # A knockout is a lower bound on damage, not an exact type multiplier.
            distance = np.minimum(distance, 0.0)
        log_likelihood = -0.5 * (distance / sigma) ** 2
        log_likelihood -= float(np.max(log_likelihood))
        likelihood = np.maximum(np.exp(log_likelihood), 1e-12)
        likelihood *= self._text_likelihood(observation.engine_text_flag, products)
        updated = previous * likelihood
        updated /= float(updated.sum())
        support = updated > 0.0
        gain = float(np.sum(updated[support] * np.log(updated[support] / previous[support])))
        if len(defenders) == 1:
            self.posterior[observation.attack_type, defenders[0]] = updated
            self._propagate_joint_evidence(observation.attack_type, defenders)
        else:
            first, second = defenders
            key = (observation.attack_type, first, second)
            first_marginal = updated.sum(axis=1)
            second_marginal = updated.sum(axis=0)
            self.posterior[observation.attack_type, first] = first_marginal
            self.posterior[observation.attack_type, second] = second_marginal
            self._pairs[key] = updated.copy()
            self._pair_marginals[key] = (first_marginal.copy(), second_marginal.copy())
            self._propagate_joint_evidence(observation.attack_type, defenders, key)
        for defender in defenders:
            self.intervention_count[observation.attack_type, defender] += 1
        return gain

    def observe_effectiveness(self, attack_type: int, defender_types: Sequence[int], flag: str) -> float:
        """Condition on a visible effectiveness message alone (no damage number).

        Used for the opponent's hits on the agent and for remembered facts that
        are replayed into a fresh posterior. Returns the posterior KL gain.
        """
        _type_id(attack_type)
        defenders = self._defenders(defender_types)
        if attack_type == TYPE_MYSTERY or TYPE_MYSTERY in defenders:
            return 0.0
        previous = self._joint(attack_type, defenders)
        products = self._products(defenders)
        likelihood = self._text_likelihood(flag, products)
        if np.all(likelihood == 1.0):
            return 0.0
        updated = previous * likelihood
        total = float(updated.sum())
        if total <= 0.0:
            return 0.0
        updated /= total
        support = updated > 0.0
        gain = float(np.sum(updated[support] * np.log(updated[support] / np.maximum(previous[support], 1e-300))))
        if len(defenders) == 1:
            self.posterior[attack_type, defenders[0]] = updated
            self._propagate_joint_evidence(attack_type, defenders)
        else:
            first, second = defenders
            key = (attack_type, first, second)
            first_marginal = updated.sum(axis=1)
            second_marginal = updated.sum(axis=0)
            self.posterior[attack_type, first] = first_marginal
            self.posterior[attack_type, second] = second_marginal
            self._pairs[key] = updated.copy()
            self._pair_marginals[key] = (first_marginal.copy(), second_marginal.copy())
            self._propagate_joint_evidence(attack_type, defenders, key)
        for defender in defenders:
            self.intervention_count[attack_type, defender] += 1
        return gain

    def efficacy_distribution(
        self, attack_type: int, defender_types: Sequence[int]
    ) -> dict[float, float]:
        """Predict total efficacy while retaining stored dual-type correlations."""
        _type_id(attack_type)
        defenders = self._defenders(defender_types)
        joint = self._joint(attack_type, defenders)
        products = self._products(defenders)
        return {
            float(value): float(joint[products == value].sum())
            for value in np.unique(products)
        }

    def expected_information_gain(
        self, move_type: int, defender_types: Sequence[int], accuracy: float = 1.0
    ) -> float:
        """Mutual information of efficacy and a landed, noiseless type observation.

        Distinct component assignments with the same product remain ambiguous;
        their entropy is not counted as learnable from this move. Misses carry
        no type information. This is the expected KL posterior/prior reduction.
        """
        if not 0.0 <= accuracy <= 1.0 or not math.isfinite(accuracy):
            raise ValueError("Accuracy must be a probability")
        distribution = self.efficacy_distribution(move_type, defender_types)
        return accuracy * _entropy(np.asarray(list(distribution.values()), dtype=np.float64))

    def evaluate_free_energy(
        self,
        move_type: int,
        defender_types: list[int],
        base_power: int,
        my_hp_ratio: float,
    ) -> tuple[float, float, float]:
        """Return G, homeostatic/completion risk, and learnable information gain.

        Risk increases for a low-damage or possibly immune move when HP is low.
        Information is measured in nats, with its value attenuated near fainting.
        Formal survival remains the safety verifier's responsibility.
        """
        if base_power < 0:
            raise ValueError("Base power cannot be negative")
        if not math.isfinite(my_hp_ratio) or not 0.0 <= my_hp_ratio <= 1.0:
            raise ValueError("HP ratio must lie in [0, 1]")
        distribution = self.efficacy_distribution(move_type, defender_types)
        expected_efficacy = sum(efficacy * probability for efficacy, probability in distribution.items())
        expected_damage = base_power * expected_efficacy
        low_hp = (1.0 - my_hp_ratio) ** 2
        pragmatic_risk = (
            10.0 * low_hp
            + (1.0 + 10.0 * low_hp) * (40.0 / max(1.0, expected_damage))
            + distribution.get(0.0, 0.0) * (1.0 + 10.0 * low_hp)
        )
        epistemic_value = (
            self.expected_information_gain(move_type, defender_types)
            * my_hp_ratio if base_power > 0 else 0.0
        )
        return pragmatic_risk - epistemic_value, pragmatic_risk, epistemic_value

    def state_dict(self) -> dict[str, object]:
        """JSON-compatible checkpoint with joint factors, counts, and marginals."""
        return {
            "posterior": self.posterior.tolist(),
            "intervention_count": self.intervention_count.tolist(),
            "ignored_observations": self.ignored_observations,
            "pairs": [
                {"key": list(key), "joint": joint.tolist(),
                 "first": self._pair_marginals[key][0].tolist(),
                 "second": self._pair_marginals[key][1].tolist()}
                for key, joint in self._pairs.items()
            ],
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore a checked posterior; malformed checkpoints never partly apply."""
        posterior = np.asarray(state["posterior"], dtype=np.float64)
        counts = np.asarray(state["intervention_count"], dtype=np.int64)
        if posterior.shape != (18, 18, 4) or counts.shape != (18, 18):
            raise ValueError("Causal checkpoint dimensions are invalid")
        if (
            not np.all(np.isfinite(posterior)) or np.any(posterior < 0.0)
            or not np.allclose(posterior.sum(axis=-1), 1.0) or np.any(counts < 0)
        ):
            raise ValueError("Causal checkpoint posterior or counts are invalid")
        neutral = np.asarray([0.0, 0.0, 1.0, 0.0])
        if not np.allclose(posterior[9], neutral) or not np.allclose(posterior[:, 9], neutral):
            raise ValueError("TYPE_MYSTERY must remain neutral")
        pair_entries = state.get("pairs", [])
        if not isinstance(pair_entries, list):
            raise ValueError("Causal checkpoint pair list is invalid")
        pairs: dict[tuple[int, int, int], FloatArray] = {}
        marginals: dict[tuple[int, int, int], tuple[FloatArray, FloatArray]] = {}
        for entry in pair_entries:
            if not isinstance(entry, dict):
                raise ValueError("Causal checkpoint pair entry is invalid")
            key_values = np.asarray(entry["key"], dtype=np.int64)
            if key_values.shape != (3,) or np.any(key_values < 0) or np.any(key_values >= 18):
                raise ValueError("Causal checkpoint pair IDs are invalid")
            key = (int(key_values[0]), int(key_values[1]), int(key_values[2]))
            joint = np.asarray(entry["joint"], dtype=np.float64)
            first = np.asarray(entry["first"], dtype=np.float64)
            second = np.asarray(entry["second"], dtype=np.float64)
            if (
                joint.shape != (4, 4) or first.shape != (4,) or second.shape != (4,)
                or not np.all(np.isfinite(joint)) or np.any(joint < 0.0)
                or not np.isclose(joint.sum(), 1.0)
                or not np.allclose(joint.sum(axis=1), first)
                or not np.allclose(joint.sum(axis=0), second)
            ):
                raise ValueError("Causal checkpoint joint factor is invalid")
            pairs[key] = joint
            marginals[key] = (first, second)
        ignored = state.get("ignored_observations", 0)
        if not isinstance(ignored, int) or ignored < 0:
            raise ValueError("Ignored observation count is invalid")
        self.posterior = posterior
        self.intervention_count = counts
        self._pairs = pairs
        self._pair_marginals = marginals
        self.ignored_observations = ignored
