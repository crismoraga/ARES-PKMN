"""The six adversarial QD metrics of the paper, Elo ratings and significance tests.

All metrics are computed from an inter-variant tournament: ``fitness[i, j]`` is
the fitness of red solution i against blue solution j (blue's is 1 - value).
For blue-side metrics pass ``1 - fitness.T``.

* win rate   - mean share of opponents beaten (fitness > 0.5)
* ELO score  - best normalised Elo rank of the set among all compared solutions
* robustness - best worst-case fitness ("a solution with no weakness")
* coverage   - share of k-means clusters of ranking vectors the set occupies
* expertise  - worst case over opponents of the best counter in the set
* AQD score  - size of the smallest opponent set that beats every solution once
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import mannwhitneyu

from .game_qd import _kmeans, ranking_vectors

Matrix = NDArray[np.float64]


def elo_ratings(fitness: Matrix, iterations: int = 30, k: float = 16.0, seed: int = 0) -> tuple[Matrix, Matrix]:
    """Elo of rows and columns from all pairwise results (shuffled passes, decaying K)."""
    rows, cols = fitness.shape
    red = np.full(rows, 1000.0)
    blue = np.full(cols, 1000.0)
    pairs = [(i, j) for i in range(rows) for j in range(cols)]
    rng = np.random.default_rng(seed)
    for iteration in range(iterations):
        step = k * (1 - iteration / (iterations + 1))
        for index in rng.permutation(len(pairs)):
            i, j = pairs[int(index)]
            expected = 1 / (1 + 10 ** ((blue[j] - red[i]) / 400))
            delta = step * (fitness[i, j] - expected)
            red[i] += delta
            blue[j] -= delta
    return red, blue


@dataclass
class SetMetrics:
    win_rate: float
    elo: float
    robustness: float
    coverage: float
    expertise: float
    aqd: float  # math.inf when some solution never loses

    def as_dict(self) -> dict[str, float | None]:
        return {"win_rate": self.win_rate, "elo": self.elo, "robustness": self.robustness, "coverage": self.coverage,
                "expertise": self.expertise, "aqd": None if math.isinf(self.aqd) else self.aqd}


def min_counter_set(beaten_by: Sequence[set[int]], candidates: int, limit: int = 8) -> float:
    """Smallest set of opponents such that every solution is beaten by one of them (exact search)."""
    universe = set(range(len(beaten_by)))
    if not universe:
        return 0.0
    if any(not opponents for opponents in beaten_by):
        return math.inf
    covers = [set(index for index, opponents in enumerate(beaten_by) if column in opponents) for column in range(candidates)]
    useful = [column for column in range(candidates) if covers[column]]
    for size in range(1, min(limit, len(useful)) + 1):
        for combination in itertools.combinations(useful, size):
            covered: set[int] = set()
            for column in combination:
                covered |= covers[column]
            if covered == universe:
                return float(size)
    # Greedy upper bound beyond the exact limit.
    remaining, size = set(universe), 0
    while remaining:
        best = max(useful, key=lambda column: len(covers[column] & remaining))
        remaining -= covers[best]
        size += 1
    return float(size)


def set_metrics(fitness: Matrix, members: Sequence[int], all_rows_elo: Matrix, ranking_clusters: Sequence[int],
                n_clusters: int) -> SetMetrics:
    """Metrics of the row subset ``members`` (one variant's solutions) of a full tournament."""
    sub = fitness[list(members)]
    win_rate = float((sub > 0.5).mean()) if sub.size else 0.0
    order = np.argsort(np.argsort(all_rows_elo))
    normalised = order / max(1, len(all_rows_elo) - 1)
    elo = float(max(normalised[list(members)])) if members else 0.0
    robustness = float(sub.min(axis=1).max()) if sub.size else 0.0
    expertise = float(sub.max(axis=0).min()) if sub.size else 0.0
    clusters = {int(ranking_clusters[index]) for index in members}
    coverage = len(clusters) / max(1, n_clusters)
    beaten = [set(int(column) for column in np.flatnonzero(row < 0.5)) for row in sub]
    aqd = min_counter_set(beaten, fitness.shape[1])
    return SetMetrics(win_rate, elo, robustness, coverage, expertise, aqd)


def side_report(fitness: Matrix, groups: dict[str, list[int]], n_clusters: int, seed: int = 0) -> dict[str, SetMetrics]:
    """Metrics per group (variant or variant/replication) for the row side."""
    elo_rows, _ = elo_ratings(fitness, seed=seed)
    labels = _kmeans(ranking_vectors(fitness), n_clusters, seed) if len(fitness) else np.zeros(0, dtype=np.int64)
    return {name: set_metrics(fitness, members, elo_rows, list(map(int, labels)), n_clusters)
            for name, members in groups.items() if members}


def holm_bonferroni(samples: dict[str, list[float]], higher_is_better: bool = True,
                    alpha: float = 0.05) -> dict[str, bool]:
    """Which variants are *not* significantly different from the best (as bold in the paper's tables)."""
    clean = {name: [value for value in values if math.isfinite(value)] for name, values in samples.items()}
    clean = {name: values for name, values in clean.items() if values}
    if not clean:
        return {}
    best = (max if higher_is_better else min)(clean, key=lambda name: float(np.median(clean[name])))
    tests = []
    for name, values in clean.items():
        if name == best:
            continue
        if len(set(values) | set(clean[best])) == 1:
            p = 1.0
        else:
            p = float(mannwhitneyu(clean[best], values, alternative="two-sided").pvalue)
        tests.append((p, name))
    result = {best: True}
    tests.sort()
    rejecting = True
    for rank, (p, name) in enumerate(tests):
        threshold = alpha / (len(tests) - rank)
        rejecting = rejecting and p < threshold
        result[name] = not rejecting
    return result


class OnlineElo:
    """Live ratings of every agent that fought in the league."""

    def __init__(self, k: float = 24.0) -> None:
        self.k = k
        self.ratings: dict[str, float] = {}
        self.games: dict[str, int] = {}

    def update(self, first: str, second: str, score: float) -> tuple[float, float]:
        a = self.ratings.setdefault(first, 1000.0)
        b = self.ratings.setdefault(second, 1000.0)
        expected = 1 / (1 + 10 ** ((b - a) / 400))
        self.ratings[first] = a + self.k * (score - expected)
        self.ratings[second] = b - self.k * (score - expected)
        self.games[first] = self.games.get(first, 0) + 1
        self.games[second] = self.games.get(second, 0) + 1
        return self.ratings[first], self.ratings[second]

    def top(self, count: int = 20) -> list[tuple[str, float, int]]:
        ranked = sorted(self.ratings.items(), key=lambda item: -item[1])[:count]
        return [(name, round(value, 1), self.games.get(name, 0)) for name, value in ranked]
