"""Ratings and the structure of a cross-play matrix: Bradley-Terry/Elo, bootstrap CIs, cycles, dominance.

``points[i, j]`` is what player ``i`` scored against ``j`` (wins + draws / 2) and ``games[i, j]`` how
many battles they played (symmetric). Bradley-Terry strengths are fitted by the MM algorithm (Hunter
2004) with a weak symmetric prior (``prior`` virtual games split evenly per played pair) that keeps a
player who never lost finite, and reported on the Elo scale (400 log10, mean 1500). Intervals come
from a parametric bootstrap of every pair's score (binomial on its battle count).

A *cycle* is a triple (i, j, k) with i > j, j > k and k > i (score above 0.5 on each edge); it is
*significant* when the Wilson 95% lower bound of every edge exceeds 0.5. A rating system cannot
represent cycles, so their presence is the evidence that no single ordering (and no single rival)
captures the league.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Z95 = 1.959963984540054


def bradley_terry(points: Array, games: Array, prior: float = 1.0, iterations: int = 2000,
                  tolerance: float = 1e-10) -> Array:
    """Elo-scale Bradley-Terry ratings (mean 1500) of a points / games matrix."""
    points = np.asarray(points, dtype=np.float64)
    games = np.asarray(games, dtype=np.float64)
    size = points.shape[0]
    played = (games > 0).astype(np.float64)
    np.fill_diagonal(played, 0.0)
    wins = points + 0.5 * prior * played
    total_games = games + prior * played
    np.fill_diagonal(total_games, 0.0)
    numerator = wins.sum(axis=1)
    strength = np.ones(size)
    for _ in range(iterations):
        denominator = (total_games / (strength[:, None] + strength[None, :])).sum(axis=1)
        updated = numerator / np.maximum(denominator, 1e-12)
        updated = np.maximum(updated, 1e-12)
        updated /= math.exp(float(np.mean(np.log(updated))))
        converged = float(np.max(np.abs(updated - strength))) < tolerance
        strength = updated
        if converged:
            break
    elo = 400.0 * np.log10(strength)
    return np.asarray(1500.0 + elo - elo.mean(), dtype=np.float64)


def bootstrap_ratings(points: Array, games: Array, draws: int = 300, seed: int = 0) -> Array:
    """(draws, players) bootstrap sample of the ratings (each pair's score resampled binomially)."""
    generator = np.random.default_rng(seed)
    size = points.shape[0]
    rate = np.divide(points, games, out=np.full_like(points, 0.5, dtype=np.float64), where=games > 0)
    samples = np.empty((draws, size))
    upper = np.triu_indices(size, 1)
    for draw in range(draws):
        resampled = np.zeros_like(points, dtype=np.float64)
        counts = games[upper].astype(np.int64)
        # Wins + draws/2 resampled as a binomial of the pair's score (draws are rare in the stadium).
        values = generator.binomial(counts, np.clip(rate[upper], 0.0, 1.0)).astype(np.float64)
        resampled[upper] = values
        resampled[(upper[1], upper[0])] = counts - values
        samples[draw] = bradley_terry(resampled, games, iterations=400, tolerance=1e-8)
    return samples


def wilson_low(score: float, total: float, z: float = Z95) -> float:
    if total <= 0:
        return 0.0
    denominator = 1 + z * z / total
    centre = (score + z * z / (2 * total)) / denominator
    half = z * math.sqrt(score * (1 - score) / total + z * z / (4 * total * total)) / denominator
    return centre - half


def cycles(score: Array, games: Array, names: Sequence[str]) -> dict[str, Any]:
    """All 3-cycles of the 'beats' relation and the significant ones (Wilson lower bound > 0.5 on each edge)."""
    size = len(names)
    found: list[list[str]] = []
    strong: list[list[str]] = []
    for i, j, k in itertools.combinations(range(size), 3):
        for a, b, c in ((i, j, k), (i, k, j)):
            edges = ((a, b), (b, c), (c, a))
            if all(games[x, y] > 0 and score[x, y] > 0.5 for x, y in edges):
                found.append([names[a], names[b], names[c]])
                if all(wilson_low(float(score[x, y]), float(games[x, y])) > 0.5 for x, y in edges):
                    strong.append([names[a], names[b], names[c]])
    return {"count": len(found), "significant_count": len(strong), "examples": found[:12],
            "significant_examples": strong[:12]}


def dominance(score: Array, games: Array, names: Sequence[str]) -> dict[str, Any]:
    """Who beats everyone (point estimate) and, for each player, the opponents it does not beat."""
    size = len(names)
    beats_all = [names[i] for i in range(size)
                 if all(score[i, j] > 0.5 for j in range(size) if j != i and games[i, j] > 0)]
    not_beaten = {names[i]: [names[j] for j in range(size) if j != i and games[i, j] > 0 and score[i, j] <= 0.5]
                  for i in range(size)}
    return {"beats_everyone": beats_all, "does_not_beat": not_beaten}


def transitivity_violations(score: Array, ratings: Array, games: Array) -> float:
    """Share of played pairs whose observed winner contradicts the rating order (excluding exact ties)."""
    size = score.shape[0]
    total = wrong = 0
    for i, j in itertools.combinations(range(size), 2):
        if games[i, j] <= 0 or abs(score[i, j] - 0.5) < 1e-12:
            continue
        total += 1
        wrong += int((score[i, j] > 0.5) != (ratings[i] > ratings[j]))
    return wrong / total if total else 0.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def rating_residual(score: Array, ratings: Array, games: Array) -> float:
    """Mean absolute gap between observed pair scores and the Bradley-Terry prediction."""
    size = score.shape[0]
    gaps = [abs(float(score[i, j]) - expected_score(float(ratings[i]), float(ratings[j])))
            for i, j in itertools.combinations(range(size), 2) if games[i, j] > 0]
    return float(np.mean(gaps)) if gaps else 0.0
