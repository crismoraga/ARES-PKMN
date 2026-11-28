"""Main-process evaluation helpers: duels on fixed battles, the cross-play matrix and the diversity evidence.

Battle index ranges (``eval_team`` draws team pair ``i // 2`` of a split; odd battles swap seats):

* test battles ``0 .. N-1`` (``TEST_OFFSET``) - only the final champion evaluation reads them;
* ``DIVERSITY_OFFSET``  - cross-play matrix and behaviour profiles of the rivals;
* ``SELECTION_OFFSET``  - choice of the E1 starting checkpoint and of the champion checkpoint;
* ``PROBE_OFFSET``      - reference battles of the decision-disagreement probe;
* ``EXPLOITER_OFFSET``  - the check of each exploiter against the frozen main agent.

All validation ranges start at 40 000 000, far from E1's in-training curves (1 000 000+) and from
E5's tournaments (1 050 000+, 21 050 000+), so no range is shared with another experiment.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ares_u.battle.rollout import WorkerPool
from ares_u.battle.training import split_indices

from .behaviour import category_distribution, distances, jensen_shannon, merge, profile
from .ratings import (bootstrap_ratings, bradley_terry, cycles, dominance, rating_residual,
                      transitivity_violations)
from .registry import strip_weights

Array = NDArray[np.float64]
LEAGUE_OFFSET = 40_000_000
TEST_OFFSET = 0
DIVERSITY_OFFSET = LEAGUE_OFFSET
SELECTION_OFFSET = LEAGUE_OFFSET + 1_000_000
PROBE_OFFSET = LEAGUE_OFFSET + 2_000_000
EXPLOITER_OFFSET = LEAGUE_OFFSET + 3_000_000


def duel(pool: WorkerPool, agent: dict[str, Any], opponent: dict[str, Any], split: str, battles: int,
         offset: int, seed: int = 0) -> list[dict[str, Any]]:
    """Per-battle records of ``agent`` (outcome from its side, behaviour of both sides), sorted by index."""
    tasks = [{"type": "league_evaluate", "split": split, "indices": [offset + index for index in chunk],
              "seed": seed, "agent": agent, "opponent": opponent} for chunk in split_indices(battles, pool.size)]
    records = [record for result in pool.map(tasks) for record in result]
    return sorted(records, key=lambda record: int(record["index"]))


def scores(records: Sequence[Mapping[str, Any]]) -> Array:
    return np.asarray([(float(record["outcome"]) + 1) / 2 for record in records], dtype=np.float64)


def pair_means(values: Array) -> Array:
    usable = values.shape[0] - values.shape[0] % 2
    return np.asarray(values[:usable].reshape(-1, 2).mean(axis=1), dtype=np.float64)


# ----------------------------------------------------------------------------- cross-play
def crossplay(pool: WorkerPool, specs: Mapping[str, dict[str, Any]], split: str, battles: int,
              expensive: Mapping[str, int] | None = None, cache: dict[str, Any] | None = None,
              offset: int = DIVERSITY_OFFSET, log: Any = None) -> dict[str, Any]:
    """Round robin on the same validation battles; ``expensive`` caps the battles of costly rivals.

    ``cache`` (pair key -> records) makes the round robin resumable.
    """
    names = list(specs)
    expensive = dict(expensive or {})
    results: dict[str, Any] = cache if cache is not None else {}
    for first, second in itertools.combinations(names, 2):
        key = f"{first}|{second}"
        if key in results:
            continue
        count = min([battles] + [expensive[name] for name in (first, second) if name in expensive])
        records = duel(pool, specs[first], specs[second], split, count, offset)
        results[key] = records
        if log is not None:
            log(key, records)
    return results


def matrix(names: Sequence[str], pairs: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[Array, Array]:
    """(points, games) matrices of the round robin (row player's wins + draws / 2)."""
    size = len(names)
    index = {name: position for position, name in enumerate(names)}
    points, games = np.zeros((size, size)), np.zeros((size, size))
    for key, records in pairs.items():
        first, second = key.split("|")
        if first not in index or second not in index:
            continue
        i, j = index[first], index[second]
        value = float(scores(records).sum())
        points[i, j] += value
        points[j, i] += len(records) - value
        games[i, j] += len(records)
        games[j, i] += len(records)
    return points, games


def behaviour_profiles(names: Sequence[str], pairs: Mapping[str, Sequence[Mapping[str, Any]]]
                       ) -> dict[str, dict[str, Any]]:
    """Merged behaviour counters of each rival over all its round-robin battles (both seats)."""
    collected: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    for key, records in pairs.items():
        first, second = key.split("|")
        for record in records:
            if "agent" not in record:
                continue
            for name, side in ((first, "agent"), (second, "opponent")):
                if name in collected:
                    collected[name].append({**record[side], "turns": int(record["turns"]), "battles": 1})
    result: dict[str, dict[str, Any]] = {}
    for name, rows in collected.items():
        summed = merge(rows)
        result[name] = {"counters": summed, "rates": profile(summed),
                        "categories": category_distribution(summed).tolist()}
    return result


def disagreement(probe: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pairwise share of probe states where two rivals choose different actions."""
    names: list[str] = list(probe[0]["names"]) if probe else []
    rows = [row for result in probe for row in result["choices"]]
    if not rows:
        return {"names": names, "states": 0, "matrix": []}
    choices = np.asarray(rows, dtype=np.int64)
    size = len(names)
    table = np.zeros((size, size))
    for i, j in itertools.combinations(range(size), 2):
        table[i, j] = table[j, i] = float(np.mean(choices[:, i] != choices[:, j]))
    switch_share = {name: float(np.mean(choices[:, k] >= 4)) for k, name in enumerate(names)}
    return {"names": names, "states": int(choices.shape[0]), "matrix": table.tolist(), "switch_share": switch_share}


def analyse(names: Sequence[str], pairs: Mapping[str, Sequence[Mapping[str, Any]]],
            probe: Sequence[Mapping[str, Any]] | None = None, bootstrap: int = 300) -> dict[str, Any]:
    """Ratings, cycles, dominance, behaviour profiles/distances and decision disagreement."""
    names = list(names)
    points, games = matrix(names, pairs)
    score = np.divide(points, games, out=np.full_like(points, 0.5), where=games > 0)
    ratings = bradley_terry(points, games)
    samples = bootstrap_ratings(points, games, draws=bootstrap)
    low, high = np.percentile(samples, 2.5, axis=0), np.percentile(samples, 97.5, axis=0)
    order = [names[k] for k in np.argsort(-ratings)]
    profiles = behaviour_profiles(names, pairs)
    behaviour = distances({name: profiles[name]["rates"] for name in names})
    js = np.zeros((len(names), len(names)))
    for i, j in itertools.combinations(range(len(names)), 2):
        value = jensen_shannon(np.asarray(profiles[names[i]]["categories"]),
                               np.asarray(profiles[names[j]]["categories"]))
        js[i, j] = js[j, i] = value
    probe_result = disagreement(probe) if probe else None
    return {
        "names": names, "score": score.tolist(), "games": games.tolist(),
        "elo": {name: float(value) for name, value in zip(names, ratings)},
        "elo_ci": {name: [float(a), float(b)] for name, a, b in zip(names, low, high)},
        "ranking": order,
        "cycles": cycles(score, games, names),
        "dominance": dominance(score, games, names),
        "transitivity_violations": transitivity_violations(score, ratings, games),
        "rating_residual": rating_residual(score, ratings, games),
        "behaviour": {name: profiles[name]["rates"] for name in names},
        "behaviour_counters": {name: profiles[name]["counters"] for name in names},
        "behaviour_distance": behaviour,
        "category_js": js.tolist(),
        "disagreement": probe_result,
    }


def probe_tasks(specs: Mapping[str, dict[str, Any]], reference: Sequence[dict[str, Any]], battles: int,
                workers: int, probability: float = 0.35, split: str = "gen1_train") -> list[dict[str, Any]]:
    return [{"type": "league_probe", "split": split, "rivals": dict(specs), "reference": list(reference),
             "indices": [PROBE_OFFSET + index for index in chunk], "probability": probability}
            for chunk in split_indices(battles, workers)]


def describe_specs(specs: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: strip_weights(spec) for name, spec in specs.items()}


def win_rate_table(names: Sequence[str], score: Sequence[Sequence[float]]) -> list[str]:
    """Markdown rows of a score matrix (row player's score)."""
    header = "| | " + " | ".join(names) + " |"
    lines = [header, "|---" * (len(names) + 1) + "|"]
    for i, name in enumerate(names):
        cells = ["-" if i == j else f"{score[i][j]:.2f}" for j in range(len(names))]
        lines.append(f"| **{name}** | " + " | ".join(cells) + " |")
    return lines


def finite(value: float) -> float | None:
    return None if not math.isfinite(value) else value
