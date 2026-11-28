"""Unit tests of the battle-level paired statistics of E1 (ares_u/battle/paired.py)."""
from __future__ import annotations

import numpy as np

from ares_u.battle.paired import (analyze, bootstrap_mean_ci, holm, label_permutation_p, pair_scores, seed_t_ci,
                                  sign_flip_p)


def test_pair_scores_average_seeds_then_pairs() -> None:
    per_seed = [[1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.5]]
    np.testing.assert_allclose(pair_scores(per_seed), [0.25, 0.875])


def test_sign_flip_detects_shift_and_accepts_null() -> None:
    rng = np.random.default_rng(0)
    assert sign_flip_p(rng.normal(0.2, 0.3, 250), draws=4000) < 0.001
    assert sign_flip_p(np.zeros(10)) == 1.0
    assert sign_flip_p(rng.normal(0.0, 0.3, 250), draws=4000) > 0.01


def test_label_permutation_and_bootstrap() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.6, 0.1, 100), rng.normal(0.5, 0.1, 100)
    assert label_permutation_p(a, b, draws=2000) < 0.01
    low, high = bootstrap_mean_ci(a)
    assert low < float(a.mean()) < high


def test_seed_t_ci_and_holm() -> None:
    mean, low, high = seed_t_ci([0.5, 0.52, 0.54, 0.56, 0.58])
    assert low < mean < high and abs(mean - 0.54) < 1e-12
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_analyze_family_on_synthetic_runs() -> None:
    rng = np.random.default_rng(2)

    def outcomes(p: float) -> list[float]:
        return [float(value) for value in (rng.random(200) < p)]

    splits = ("gen1_train", "gen2", "gen3")
    data = {"battles": 200, "methods": {
        method: {f"seed{s}": {split: {"greedy": outcomes(p), "prior": outcomes(p), "random": outcomes(0.9)}
                              for split in splits} for s in range(3)}
        for method, p in (("ppo", 0.6), ("ppo_ids", 0.4), ("dqn", 0.45), ("bc_ppo", 0.6))}}
    result = analyze(data)
    by_id = {(test["id"], test["split"]): test for test in result["family"]}
    assert by_id[("H1", "gen3")]["diff"] > 0 and by_id[("H1", "gen3")]["p_holm"] < 0.05
    assert by_id[("H2", "gen2")]["diff"] > 0 and by_id[("H2", "gen2")]["p_holm"] < 0.05
    assert all("p_holm" in test for test in result["family"])
    assert {gap["split"] for gap in result["gaps"]} == {"gen2", "gen3"}
