"""Independent engine fixtures, causal identification, and stock/risk semantics."""

from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pytest

from core.causal.capture_and_training import (
    BallInventory,
    CaptureContext,
    CapturePreparation,
    EpistemicCaptureOptimizer,
    MetacognitiveTrainingCurriculum,
    SPECIES_CATCH_RATES,
    TrainingRoute,
)
from core.causal.epistemic_scm import (
    BeliefStateEstimator,
    DamageObservation,
    EpistemicCausalController,
    HiddenOpponentSet,
    PLAYABLE_TYPES,
    TYPE_MYSTERY,
)


def test_causal_prior_is_probabilistic_18_by_18_with_reserved_type() -> None:
    causal = EpistemicCausalController()
    assert causal.scm_matrix.shape == (18, 18)
    assert causal.posterior.shape == (18, 18, 4)
    assert len(PLAYABLE_TYPES) == 17
    assert TYPE_MYSTERY not in PLAYABLE_TYPES
    assert np.all(causal.scm_matrix[TYPE_MYSTERY] == 1.0)
    assert np.all(causal.scm_matrix[:, TYPE_MYSTERY] == 1.0)
    assert np.allclose(causal.posterior.sum(axis=-1), 1.0)


@pytest.mark.parametrize("efficacy,flag", [(0.0, "NO_EFFECT"), (0.5, "NOT_VERY_EFFECTIVE"), (1.0, "NEUTRAL"), (2.0, "SUPER_EFFECTIVE")])
def test_do_interventions_converge_for_all_type_effects(efficacy: float, flag: str) -> None:
    causal = EpistemicCausalController()
    before = causal.expected_information_gain(10, [12])
    for _ in range(12):
        causal.apply_do_calculus(10, 12, efficacy * 60, 60, flag)
    assert causal.scm_matrix[10, 12] == pytest.approx(efficacy, abs=1e-8)
    assert causal.expected_information_gain(10, [12]) < before / 1000
    assert np.allclose(causal.posterior.sum(axis=-1), 1.0)


def test_direct_and_residual_are_separated_and_status_is_not_damage() -> None:
    causal = EpistemicCausalController()
    before = causal.posterior.copy()
    causal.observe_intervention(DamageObservation(10, (12,), 12, 0, 12, is_status_move=True))
    np.testing.assert_array_equal(before, causal.posterior)
    for _ in range(4):
        causal.observe_intervention(DamageObservation(10, (12,), 72, 30, 12, "SUPER_EFFECTIVE"))
    assert causal.scm_matrix[10, 12] == pytest.approx(2.0, abs=1e-8)
    assert causal.ignored_observations == 1
    with pytest.raises(ValueError, match="Residual"):
        causal.observe_intervention(DamageObservation(10, (12,), 1, 30, 2))


def test_unidentified_unknown_modifier_is_not_assumed_to_be_type_effect() -> None:
    causal = EpistemicCausalController()
    before = causal.posterior.copy()
    causal.observe_intervention(DamageObservation(10, (12,), 80, 40, identified=False))
    np.testing.assert_array_equal(before, causal.posterior)
    assert causal.ignored_observations == 1


def test_dual_type_joint_inference_retains_ambiguity_and_resolves_by_intervention() -> None:
    causal = EpistemicCausalController()
    for _ in range(4):
        causal.observe_intervention(DamageObservation(10, (11, 12), 60, 60, engine_text_flag="NEUTRAL"))
    # Product=1 can be .5*2, 1*1, or 2*.5; neither component is known neutral.
    assert causal.posterior[10, 11, 2] < 0.5
    assert causal.posterior[10, 12, 2] < 0.5
    assert causal.efficacy_distribution(10, [11, 12])[1.0] > 0.9999
    assert causal.expected_information_gain(10, [11, 12]) < 0.001
    for _ in range(6):
        causal.apply_do_calculus(10, 11, 30, 60, "NOT_VERY_EFFECTIVE")
    distribution = causal.efficacy_distribution(10, [11, 12])
    assert distribution[1.0] > 0.9999
    assert causal.posterior[10, 12, 3] > 0.9999
    # Stored pair correlations transfer a revealed component to total prediction.
    assert sum(probability for probability in distribution.values()) == pytest.approx(1.0)


def test_duplicate_ram_types_are_applied_once_and_censored_ko_is_lower_bound() -> None:
    causal = EpistemicCausalController()
    assert causal.efficacy_distribution(10, [12, 12]) == causal.efficacy_distribution(10, [12])
    causal.observe_intervention(DamageObservation(10, (12,), 20, 40, defender_hp_before=20))
    # HP=20 cannot tell apart 20, 40, or 80 theoretical damage.
    assert causal.posterior[10, 12, 3] > 0.2
    assert causal.posterior[10, 12, 2] > 0.2


def test_free_energy_risk_and_information_have_correct_direction() -> None:
    causal = EpistemicCausalController()
    g_unknown, risk, information = causal.evaluate_free_energy(10, [12], 60, 1.0)
    assert information > 0.0 and g_unknown == pytest.approx(risk - information)
    for _ in range(5):
        causal.apply_do_calculus(10, 12, 120, 60, "SUPER_EFFECTIVE")
        causal.apply_do_calculus(11, 12, 0, 60, "NO_EFFECT")
    _, strong_risk, learned_information = causal.evaluate_free_energy(10, [12], 60, 1.0)
    _, immune_risk, _ = causal.evaluate_free_energy(11, [12], 60, 1.0)
    assert strong_risk < immune_risk
    assert learned_information < information
    assert causal.evaluate_free_energy(10, [12], 60, 0.1)[1] > strong_risk
    assert causal.evaluate_free_energy(10, [12], 0, 1.0)[2] == 0.0
    with pytest.raises(ValueError):
        causal.evaluate_free_energy(10, [18], 60, 1.0)


def test_bayesian_hidden_sets_normalize_explicit_likelihoods_and_reveals() -> None:
    first = HiddenOpponentSet("special", 65, (94, 105, 115, 347), 28, 213, special_attack=150)
    second = HiddenOpponentSet("physical", 65, (33, 105, 115, 347), 39, 0, special_attack=50)
    belief = BeliefStateEstimator([first, second], [0.75, 0.25])
    gain = belief.update_evidence({"special": 0.1, "physical": 0.9})
    assert gain > 0.0
    assert belief.probabilities == pytest.approx({"special": 0.25, "physical": 0.75})
    assert belief.expected_stat("special_attack") == pytest.approx(75.0)
    belief.observe_move(94)
    belief.observe_ability(28)
    belief.observe_item(213)
    assert belief.probabilities["special"] > 0.9999
    assert belief.entropy < 0.001
    with pytest.raises(ValueError, match="zero probability"):
        belief.update_evidence({"special": 0.0, "physical": 0.0})
    assert sum(belief.probabilities.values()) == pytest.approx(1.0)


def test_causal_checkpoint_restores_joint_factors_and_rejects_partial_corruption() -> None:
    causal = EpistemicCausalController()
    causal.observe_intervention(DamageObservation(10, (11, 12), 60, 60, engine_text_flag="NEUTRAL"))
    restored = EpistemicCausalController()
    restored.load_state_dict(causal.state_dict())
    assert restored.efficacy_distribution(10, [11, 12]) == pytest.approx(causal.efficacy_distribution(10, [11, 12]))
    invalid = causal.state_dict()
    invalid["posterior"] = [1.0]
    with pytest.raises(ValueError):
        restored.load_state_dict(invalid)
    np.testing.assert_array_equal(restored.posterior, causal.posterior)


@pytest.mark.parametrize("rate,mult,status,hp,a,b", [(45, 10, 10, 100, 15, 32767), (45, 15, 10, 100, 22, 36157), (45, 20, 20, 100, 60, 47661), (3, 10, 10, 1, 2, 19784), (255, 10, 10, 1, 253, 65535)])
def test_exact_capture_engine_integer_fixtures(rate: int, mult: int, status: int, hp: int, a: int, b: int) -> None:
    # Hand-checked integer root denominators are32,29,22,53,16 respectively.
    result = EpistemicCaptureOptimizer.compute_capture_arithmetic(100, hp, rate, mult, status)
    assert result.catch_value == a
    assert result.shake_threshold == b
    assert result.probability_fraction == Fraction(b, 65536) ** 4


def test_capture_certain_zero_hp_domains_and_first_truncation_counterexample() -> None:
    capture = EpistemicCaptureOptimizer
    assert capture.compute_capture_arithmetic(100, 1, 255, 20, 20).probability == 1.0
    assert capture.compute_capture_arithmetic(100, 100, 1, 10).probability == 0.0
    # (3 * 15 // 10) * 299 // 300 is 3; postponing first division incorrectly gives4.
    assert capture.compute_capture_arithmetic(100, 1, 3, 15).catch_value == 3
    with pytest.raises(ValueError):
        capture.compute_capture_arithmetic(100, 0, 45, 10)
    with pytest.raises(ValueError):
        capture.compute_exact_gen3_catch_prob(100, 1, 45, 1.234, 10)


def test_species_rates_use_internal_generation_three_ids_and_actual_values() -> None:
    assert len(SPECIES_CATCH_RATES) == 386
    assert SPECIES_CATCH_RATES[1] == 45
    assert SPECIES_CATCH_RATES[283] == 45  # Mudkip internal ID, not national258.
    assert SPECIES_CATCH_RATES[150] == 3
    assert 258 not in SPECIES_CATCH_RATES
    optimizer = EpistemicCaptureOptimizer()
    with pytest.raises(ValueError, match="Unknown ROM-internal"):
        optimizer.capture_arithmetic("POKE_BALL", CaptureContext(258, 100, 1))


def test_contextual_gen3_balls_and_status_bits() -> None:
    optimizer = EpistemicCaptureOptimizer()
    context = CaptureContext(129, 100, 100, (11,), 5, status=0x20, turn_count=35, already_caught=True, underwater=True)
    assert optimizer.contextual_ball_multiplier("NET_BALL", context) == 30
    assert optimizer.contextual_ball_multiplier("DIVE_BALL", context) == 35
    assert optimizer.contextual_ball_multiplier("NEST_BALL", context) == 35
    assert optimizer.contextual_ball_multiplier("REPEAT_BALL", context) == 30
    assert optimizer.contextual_ball_multiplier("TIMER_BALL", context) == 40
    assert optimizer.status_multiplier(0x20) == 20
    assert optimizer.status_multiplier(0x80) == 15
    assert optimizer.status_multiplier(0x7) == 20
    assert optimizer.capture_arithmetic("MASTER_BALL", context).probability == 1.0
    trainer = CaptureContext(129, 100, 100, is_trainer_battle=True)
    assert optimizer.capture_arithmetic("MASTER_BALL", trainer).probability == 0.0


def test_capture_inventory_is_respected_and_risk_changes_ball_choice() -> None:
    optimizer = EpistemicCaptureOptimizer()
    context = CaptureContext(1, 100, 20)
    inventory = BallInventory({"POKE_BALL": 8, "ULTRA_BALL": 1, "GREAT_BALL": 0})
    cheap = optimizer.select_ball(context, inventory)
    assert cheap is not None and cheap.ball == "POKE_BALL"
    assert cheap.finite_inventory_capture_probability == pytest.approx(1 - (1 - cheap.probability) ** 8)
    risky = CaptureContext(1, 100, 20, failure_faint_probability=0.5, failure_risk_cost=100_000)
    safer = optimizer.select_ball(risky, inventory)
    assert safer is not None and safer.ball == "ULTRA_BALL"
    assert optimizer.select_ball(context, BallInventory({"POKE_BALL": 0})) is None
    assert inventory.consume("ULTRA_BALL").quantities["ULTRA_BALL"] == 0
    assert inventory.quantities["ULTRA_BALL"] == 1
    with pytest.raises(ValueError):
        inventory.consume("GREAT_BALL")


def test_capture_preparation_compares_cost_and_does_not_invent_safe_action() -> None:
    optimizer = EpistemicCaptureOptimizer()
    context = CaptureContext(1, 100, 100)
    inventory = BallInventory({"POKE_BALL": 10})
    immediate = optimizer.plan_capture(context, inventory, set(), {12})
    assert immediate.action == "THROW_POKE_BALL"
    sleep = CapturePreparation("SLEEP", 100, 1, 0.0)
    prepared = optimizer.plan_capture(context, inventory, set(), {12}, [sleep])
    assert prepared.action == "SLEEP"
    risky = CapturePreparation("RISKY_WEAKEN", 1, 0, 0.9)
    assert optimizer.plan_capture(context, inventory, set(), {12}, [risky]).action == "THROW_POKE_BALL"
    assert optimizer.plan_capture(context, inventory, {1}, set()).action == "IGNORE_CAPTURE"
    action, _ = optimizer.decide_capture_action(1, [12], 100, 100, 45, 10, set(), {12})
    assert action == "INDUCE_STATUS_OR_WEAKEN"


def test_curriculum_accounts_for_travel_healing_and_least_risk_fallback() -> None:
    curriculum = MetacognitiveTrainingCurriculum()
    fast = TrainingRoute("fast", 200, 10, 0.5)
    safe = TrainingRoute("safe", 100, 20, 0, travel_seconds=10, healing_seconds=10)
    assert curriculum.select_route([fast, safe]).route.name == "safe"
    assert curriculum.rank_routes([safe])[0].xp_per_second == 2.5
    lower_risk = TrainingRoute("slow_risk", 10, 30, 0.01)
    assert curriculum.select_route([fast, lower_risk]).route.name == "slow_risk"
    for _ in range(20):
        curriculum.record_outcome("safe", False)
    assert curriculum.estimated_risk(safe) == pytest.approx(1 / 22)
    curriculum.record_outcome("safe", True)
    assert curriculum.estimated_risk(safe) == pytest.approx(2 / 23)
    with pytest.raises(ValueError):
        curriculum.select_route([])
    with pytest.raises(ValueError):
        TrainingRoute("invalid", 10, math.inf, 0)
