"""GAME (Generational Adversarial MAP-Elites) with tournament-informed task selection.

Faithful to Anne et al., "Tournament Informed Adversarial Quality Diversity"
(GECCO'26), Algorithms 1-5:

* ``GrowingArchive``   - Algorithm 3, an unstructured archive of ``n_cell`` cells
  whose centroids move to keep the most distinct behaviours, with backups to
  repair cells whose elite drifts into another cell.
* ``mtmb_me``          - Algorithm 2, multi-task multi-behaviour MAP-Elites: one
  archive per task (opponent elite), parents drawn from the whole archive.
* ``select_tasks``     - Algorithms 4/5 plus Pareto and Random:
  ``behavior`` (k-means on behaviours), ``ranking`` (k-means on normalised
  ranking vectors from a tournament against the previous tasks - the paper's
  best method), ``pareto`` (non-dominated fronts with crowding) and ``random``.
* ``run_game``         - Algorithm 1, alternating red and blue generations, each
  bootstrapped with the evaluations already paid for during task selection.
"""
from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.vq import kmeans2

from .dex import Dex
from .evaluation import Evaluator, Outcome, SideSpec, Solution
from .policies import MLPGenome
from .teams import initial_genes, mutate_genes

Variant = Literal["ranking", "pareto", "behavior", "random"]
VARIANTS: tuple[Variant, ...] = ("ranking", "pareto", "behavior", "random")


@dataclass
class Entry:
    solution: Solution
    fitness: float
    behavior: NDArray[np.float32]


class GrowingArchive:
    """Algorithm 3: growing unstructured archive with centroid repair."""

    def __init__(self, n_cell: int) -> None:
        self.n_cell = n_cell
        self.centroids: list[NDArray[np.float32]] = []
        self.elites: list[Entry] = []
        self.backup: list[list[Entry]] = []

    def __len__(self) -> int:
        return len(self.elites)

    def _closest(self, behavior: NDArray[np.float32]) -> tuple[int, float]:
        distances = [float(np.linalg.norm(behavior - centroid)) for centroid in self.centroids]
        index = int(np.argmin(distances))
        return index, distances[index]

    def update(self, solution: Solution, fitness: float, behavior: NDArray[np.float32]) -> bool:
        """Returns True when the entry became an elite."""
        entry = Entry(solution, fitness, np.asarray(behavior, dtype=np.float32))
        if len(self.centroids) < self.n_cell:
            self.centroids.append(entry.behavior.copy())
            self.elites.append(entry)
            self.backup.append([entry])
            return True
        count = len(self.centroids)
        pair_distance = np.full((count, count), np.inf)
        for i in range(count):
            for j in range(i + 1, count):
                pair_distance[i, j] = pair_distance[j, i] = float(np.linalg.norm(self.centroids[i] - self.centroids[j]))
        flat_j, flat_k = np.unravel_index(int(np.argmin(pair_distance)), pair_distance.shape)
        j, k = int(flat_j), int(flat_k)
        d_min = float(pair_distance[j, k])
        index, distance = self._closest(entry.behavior)
        if distance > d_min:
            others = [i for i in range(count) if i not in (j, k)]
            d_j = min((pair_distance[j, i] for i in others), default=np.inf)
            d_k = min((pair_distance[k, i] for i in others), default=np.inf)
            replaced = int(j if d_j < d_k else k)
            self.centroids[replaced] = entry.behavior.copy()
            self.elites[replaced] = entry
            self.backup[replaced] = [entry]
            for cell in range(count):  # repair holes: an elite must live in its own cell
                if self._closest(self.elites[cell].behavior)[0] != cell:
                    candidates = [item for item in self.backup[cell] if self._closest(item.behavior)[0] == cell]
                    if candidates:
                        self.elites[cell] = max(candidates, key=lambda item: item.fitness)
            return True
        if fitness > self.elites[index].fitness:
            self.elites[index] = entry
            self.backup[index].append(entry)
            if len(self.backup[index]) > 16:
                self.backup[index] = sorted(self.backup[index], key=lambda item: -item.fitness)[:16]
            return True
        return False

    def qd_score(self) -> float:
        return float(sum(entry.fitness for entry in self.elites))


@dataclass
class GameConfig:
    n_gen: int = 6
    n_task: int = 8
    n_cell: int = 8
    budget: int = 1200
    n_init: int = 24
    batch: int = 64
    variant: Variant = "ranking"
    team_mutation: float = 0.3
    crossover: float = 0.1
    prior_fraction: float = 0.5


class SpaceOps:
    """Random solutions and variation for one side (the paper's S and Variation)."""

    def __init__(self, spec: SideSpec, dex: Dex, rng: random.Random, counter: Callable[[], int]) -> None:
        self.spec, self.dex, self.rng, self.counter = spec, dex, rng, counter
        # Lamarckian refinement: when set, a share of offspring is fine-tuned on the
        # lessons this side learnt from its rival before being evaluated.
        self.refine: Callable[[MLPGenome], MLPGenome] | None = None
        self.refine_rate = 0.0

    def random_solution(self, prior_fraction: float = 0.5) -> Solution:
        team = initial_genes(self.spec.roster, self.spec.parties, self.rng, self.spec.team_size, self.dex, self.spec.level)
        if self.rng.random() < prior_fraction:
            mlp, origin = MLPGenome.prior(self.rng), "prior"
        else:
            mlp, origin = MLPGenome.random(self.rng), "random"
        return Solution(self.counter(), self.spec.name, team, mlp, origin)

    def vary(self, parent: Solution, mate: Solution | None, config: GameConfig) -> Solution:
        mlp = parent.mlp
        parents: tuple[int, ...] = (parent.uid,)
        if mate is not None and self.rng.random() < config.crossover:
            mlp = mlp.crossover(mate.mlp, self.rng)
            parents = (parent.uid, mate.uid)
        mlp = mlp.mutate(self.rng)
        team = parent.team
        if self.rng.random() < config.team_mutation:
            team = mutate_genes(team, self.spec.roster, self.dex, self.rng, size=self.spec.team_size,
                                level=self.spec.level)
        origin = "variation"
        if self.refine is not None and self.rng.random() < self.refine_rate:
            mlp, origin = self.refine(mlp), "lamarck"
        return Solution(self.counter(), self.spec.name, team, mlp, origin, parents)


def all_elites(archives: Sequence[GrowingArchive]) -> list[Entry]:
    seen: dict[int, Entry] = {}
    for archive in archives:
        for entry in archive.elites:
            current = seen.get(entry.solution.uid)
            if current is None or entry.fitness > current.fitness:
                seen[entry.solution.uid] = entry
    return list(seen.values())


Bootstrap = list[tuple[int, Solution, float, NDArray[np.float32]]]


def mtmb_me(tasks: Sequence[Solution], ops: SpaceOps, bootstrap: Bootstrap, evaluator: Evaluator,
            config: GameConfig, progress: Callable[[int, list[GrowingArchive]], None] | None = None
            ) -> list[GrowingArchive]:
    """Algorithm 2 (batched): illuminate one side against fixed opponent tasks."""
    archives = [GrowingArchive(config.n_cell) for _ in tasks]
    for task_index, solution, fitness, behavior in bootstrap:
        if 0 <= task_index < len(archives):
            archives[task_index].update(solution, fitness, behavior)
    done = 0
    rng = ops.rng
    while done < config.budget:
        batch: list[tuple[int, Solution]] = []
        pool = all_elites(archives)
        for _ in range(min(config.batch, config.budget - done)):
            task_index = rng.randrange(len(tasks))
            if len(pool) < config.n_init:
                candidate = ops.random_solution(config.prior_fraction)
            else:
                parent = rng.choice(pool).solution
                mate = rng.choice(pool).solution
                candidate = ops.vary(parent, mate, config)
            batch.append((task_index, candidate))
        outcomes = evaluator.evaluate([(candidate, tasks[task_index]) for task_index, candidate in batch])
        for (task_index, candidate), outcome in zip(batch, outcomes):
            archives[task_index].update(candidate, outcome.fitness, outcome.behavior)
        done += len(batch)
        if progress is not None:
            progress(done, archives)
    return archives


def _kmeans(points: NDArray[np.float64], k: int, seed: int) -> NDArray[np.int64]:
    if len(points) <= k:
        return np.arange(len(points))
    jitter = np.random.default_rng(seed).normal(0, 1e-6, points.shape)
    _, labels = kmeans2(points + jitter, k, minit="++", seed=seed)
    return np.asarray(labels, dtype=np.int64)


def ranking_vectors(fitness: NDArray[np.float64]) -> NDArray[np.float64]:
    """Algorithm 5 lines 4-5: rank of each task in the fitness vector, normalised to [-1, 1]."""
    count = fitness.shape[1]
    ranks = np.argsort(np.argsort(fitness, axis=1, kind="stable"), axis=1, kind="stable").astype(np.float64)
    return 2 * ranks / max(1, count - 1) - 1


def _crowding(values: NDArray[np.float64]) -> NDArray[np.float64]:
    count, objectives = values.shape
    distance = np.zeros(count)
    for objective in range(objectives):
        order = np.argsort(values[:, objective])
        distance[order[0]] = distance[order[-1]] = np.inf
        span = values[order[-1], objective] - values[order[0], objective]
        if span <= 0:
            continue
        for position in range(1, count - 1):
            distance[order[position]] += (values[order[position + 1], objective] -
                                          values[order[position - 1], objective]) / span
    return distance


def pareto_select(values: NDArray[np.float64], k: int) -> list[int]:
    """Non-dominated fronts (maximisation), crowding distance inside the last front."""
    remaining = list(range(len(values)))
    chosen: list[int] = []
    while remaining and len(chosen) < k:
        front = [i for i in remaining if not any(np.all(values[j] >= values[i]) and np.any(values[j] > values[i])
                                                 for j in remaining if j != i)]
        if len(chosen) + len(front) <= k:
            chosen.extend(front)
        else:
            crowd = _crowding(values[front])
            order = np.argsort(-crowd)
            chosen.extend(front[int(i)] for i in order[:k - len(chosen)])
        remaining = [i for i in remaining if i not in front]
    return chosen


@dataclass
class Selection:
    tasks: list[Solution]
    bootstrap: Bootstrap
    fitness_matrix: NDArray[np.float64] | None = None
    ranking: NDArray[np.float64] | None = None
    clusters: list[int] = field(default_factory=list)


def _tournament_bootstrap(tasks: Sequence[Solution], previous: Sequence[Solution],
                          outcomes: Sequence[Outcome], rows: Sequence[Solution]) -> Bootstrap:
    """Evaluations of (new task e vs old task t) reused for the next side: (e as task, t as solution)."""
    index = {solution.uid: position for position, solution in enumerate(tasks)}
    bootstrap: Bootstrap = []
    outcome_iter = iter(outcomes)
    for row in rows:
        for column, old in enumerate(previous):
            outcome = next(outcome_iter)
            if row.uid in index:
                bootstrap.append((index[row.uid], old, 1.0 - outcome.fitness, outcome.opponent_behavior))
    return bootstrap


def select_tasks(archives: Sequence[GrowingArchive], previous: Sequence[Solution], evaluator: Evaluator,
                 config: GameConfig, rng: random.Random) -> Selection:
    elites = all_elites(archives)
    k = config.n_task
    if config.variant == "random":
        chosen = [entry.solution for entry in rng.sample(elites, min(k, len(elites)))]
        _, outcomes = evaluator.matrix(chosen, previous)
        return Selection(chosen, _tournament_bootstrap(chosen, previous, outcomes, chosen))
    if config.variant == "behavior":
        behaviors = np.array([entry.behavior for entry in elites], dtype=np.float64)
        labels = _kmeans(behaviors, k, rng.randrange(2 ** 31))
        chosen = []
        for cluster in sorted(set(int(label) for label in labels)):
            members = [entry for entry, label in zip(elites, labels) if label == cluster]
            chosen.append(max(members, key=lambda entry: entry.fitness).solution)
        _, outcomes = evaluator.matrix(chosen, previous)
        return Selection(chosen, _tournament_bootstrap(chosen, previous, outcomes, chosen), clusters=list(map(int, labels)))
    # Ranking and Pareto both pay a tournament: every elite against every previous task.
    rows = [entry.solution for entry in elites]
    fitness, outcomes = evaluator.matrix(rows, previous)
    if config.variant == "pareto":
        picked = pareto_select(fitness, k)
        chosen = [rows[index] for index in picked]
        return Selection(chosen, _tournament_bootstrap(chosen, previous, outcomes, rows), fitness)
    ranks = ranking_vectors(fitness)
    labels = _kmeans(ranks, k, rng.randrange(2 ** 31))
    chosen = []
    for cluster in sorted(set(int(label) for label in labels)):
        indexes = [index for index, label in enumerate(labels) if label == cluster]
        best = max(indexes, key=lambda index: float(fitness[index].mean()))
        chosen.append(rows[best])
    return Selection(chosen, _tournament_bootstrap(chosen, previous, outcomes, rows), fitness, ranks,
                     list(map(int, labels)))


@dataclass
class GenerationReport:
    generation: int
    side: str
    archives: list[GrowingArchive]
    selection: Selection
    evaluations: int


def run_game(sides: dict[str, SideSpec], evaluator: Evaluator, config: GameConfig, seed: int = 0,
             on_generation: Callable[[GenerationReport], None] | None = None,
             initial_tasks: Sequence[Solution] | None = None,
             progress: Callable[[str, int, list[GrowingArchive]], None] | None = None) -> dict[str, list[Solution]]:
    """Algorithm 1: alternate red/blue illumination; returns the last tasks of each side."""
    rng = random.Random(seed)
    counter = itertools.count(seed * 1_000_000 + 1)

    def next_uid() -> int:
        return next(counter)

    ops = {name: SpaceOps(spec, evaluator.dex, random.Random(rng.randrange(2 ** 31)), next_uid)
           for name, spec in sides.items()}
    order = ("red", "blue")
    tasks: list[Solution] = list(initial_tasks) if initial_tasks else \
        [ops["blue"].random_solution(config.prior_fraction) for _ in range(config.n_task)]
    last: dict[str, list[Solution]] = {"blue": list(tasks)}
    bootstrap: Bootstrap = []
    for generation in range(1, config.n_gen + 1):
        side = order[(generation - 1) % 2]
        callback: Callable[[int, list[GrowingArchive]], None] | None = None
        if progress is not None:
            report = progress

            def callback(done: int, archive: list[GrowingArchive], side: str = side) -> None:
                report(side, done, archive)
        archives = mtmb_me(tasks, ops[side], bootstrap, evaluator, config, callback)
        selection = select_tasks(archives, tasks, evaluator, config, rng)
        tasks, bootstrap = selection.tasks, selection.bootstrap
        last[side] = list(tasks)
        if on_generation is not None:
            on_generation(GenerationReport(generation, side, archives, selection, evaluator.evaluations))
    return last


def budget_matched(config: GameConfig) -> GameConfig:
    """Random/Behavior get the evaluations Ranking/Pareto spend on tournaments (paper, Sec. 5.2)."""
    if config.variant in ("random", "behavior"):
        extra = config.n_task * config.n_task * max(0, config.n_cell - 1)
        return GameConfig(**{**config.__dict__, "budget": config.budget + extra})
    return config


def entropy(labels: Sequence[int]) -> float:
    counts = np.bincount(np.asarray(labels, dtype=np.int64)) if labels else np.zeros(1)
    probabilities = counts[counts > 0] / max(1, counts.sum())
    return float(-(probabilities * np.log(probabilities)).sum()) if len(probabilities) else 0.0


__all__ = ["Entry", "GameConfig", "GenerationReport", "GrowingArchive", "Selection", "SpaceOps", "VARIANTS",
           "all_elites", "budget_matched", "entropy", "mtmb_me", "pareto_select", "ranking_vectors", "run_game",
           "select_tasks", "math"]
