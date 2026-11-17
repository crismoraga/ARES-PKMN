"""ARES-ADVERSARIAL: the live, never-ending two-sided league.

ROJO (the FireRed campaign lineage) and AZUL (the Emerald lineage) take turns
being illuminated by GAME against the other's elites. Every generation:

1. **MTMB-ME** illuminates the active side against the rival's tasks
   (bootstrapped with the tournament already paid and with injected lessons).
2. **Task selection** (Ranking by default) picks diverse, strong elites; they
   become the rival's next tasks.
3. **Live duel**: the two champions fight a best-of-N, narrated in Spanish.
4. **Counterfactual lessons**: each side learns, from cloned positions, the
   rival's better choices; refined champions are injected into the archives.
5. **Deep learners** (optional): the campaign's Transformer+SSM policy imitates
   the side's elites, trains with recurrent PPO against the rival's elites and
   is distilled back into an MLP student that enters the archive.
6. **Knowledge exchange**: learnt type-chart facts and revealed sets are merged
   (complementary learning), measuring what each side gained from the other.
7. **Measurement**: online Elo, the six adversarial QD metrics of the paper and
   absolute progress against fixed anchors (greedy and random players).

Everything is published atomically to ``artifacts/stadium/<run>/live.json``
(+ ``history.jsonl``) for the dashboard, and checkpointed so ``--resume``
continues exactly where it stopped.
"""
from __future__ import annotations

import itertools
import json
import math
import random
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.game_data import ROOT

from .dex import Dex, Game, load_dex
from .evaluation import Evaluator, SideSpec, Solution, policy_for
from .game_qd import (Bootstrap, GameConfig, GrowingArchive, Selection, SpaceOps, Variant, all_elites, mtmb_me,
                      select_tasks)
from .knowledge import SideKnowledge, exchange
from .deep import DeepLearner, StadiumEnvironment, elite_imitation_samples
from .learning import Lesson, apply_lessons, counterfactual_lessons
from .match import BEHAVIOR_NAMES, play
from .metrics import OnlineElo, side_report
from .narration import narrate
from .policies import MLPGenome
from .teams import TeamBank, load_bank

STADIUM_ROOT = ROOT / "artifacts" / "stadium"
LABELS = {"red": "ROJO · FireRed", "blue": "AZUL · Emerald"}
GAMES: dict[str, Game] = {"red": "firered", "blue": "emerald"}


@dataclass
class LeagueConfig:
    generations: int = 0            # 0 = run forever
    variant: Variant = "ranking"
    n_task: int = 8
    n_cell: int = 8
    budget: int = 640
    batch: int = 64
    n_init: int = 24
    repeats: int = 2
    workers: int = 12
    level: int | None = 50
    duel_battles: int = 5
    lessons: bool = True
    lesson_points: int = 16
    lamarck_rate: float = 0.25      # share of offspring fine-tuned on the lesson bank
    lesson_bank: int = 400          # lessons kept per side
    rollouts: int = 6
    deep: bool = True
    ppo_updates: int = 3
    share_knowledge: bool = True
    anchor_battles: int = 24
    seed: int = 0
    refresh_teams: bool = False

    def game_config(self) -> GameConfig:
        return GameConfig(n_task=self.n_task, n_cell=self.n_cell, budget=self.budget, n_init=self.n_init,
                          batch=self.batch, variant=self.variant)


@dataclass
class SideState:
    tasks: list[Solution] = field(default_factory=list)
    champion: Solution | None = None
    champion_fitness: float = 0.0
    injections: list[Solution] = field(default_factory=list)
    archive: list[dict[str, Any]] = field(default_factory=list)
    qd_score: float = 0.0
    anchors: dict[str, float] = field(default_factory=dict)
    deep: dict[str, float] = field(default_factory=dict)
    lessons_learnt: int = 0


def _atomic_json(path: Path, value: object, attempts: int = 40) -> None:
    """Write-then-rename; on Windows a reader (the dashboard) may hold the target briefly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, default=_json_default), encoding="utf-8")
    for attempt in range(attempts):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                temporary.unlink(missing_ok=True)
                raise
            time.sleep(0.05 * (attempt + 1))


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"not serialisable: {type(value)}")


class League:
    def __init__(self, config: LeagueConfig, run_dir: Path | None = None, resume: bool = False,
                 log: Callable[[str], None] = print) -> None:
        self.config = config
        self.log = log
        self.dex: Dex = load_dex()
        bank: TeamBank = load_bank(self.dex, refresh=config.refresh_teams)
        self.sides = {name: SideSpec(name, GAMES[name], bank.roster(GAMES[name]), bank.parties.get(GAMES[name], []),
                                     config.level) for name in ("red", "blue")}
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = run_dir or STADIUM_ROOT / f"run-{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(config.seed)
        self.uids = itertools.count(config.seed * 10_000_000 + 1)
        self.state = {name: SideState() for name in ("red", "blue")}
        self.elo = OnlineElo()
        self.generation = 0
        self.bootstrap: Bootstrap = []
        self.history: list[dict[str, Any]] = []
        self.timeline: list[dict[str, Any]] = []
        self.lesson_log: list[dict[str, Any]] = []
        self.lesson_bank: dict[str, list[Lesson]] = {"red": [], "blue": []}
        self.exchanges: list[dict[str, Any]] = []
        self.duel: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}
        self.started = time.time()
        self.phase = "iniciando"
        self.progress = {"done": 0, "budget": config.budget}
        self.duel_battles = 0
        self.evaluator = Evaluator(self.sides, workers=config.workers, repeats=config.repeats, seed=config.seed)
        self.ops = {name: SpaceOps(spec, self.dex, random.Random(self.rng.randrange(2 ** 31)), self._uid)
                    for name, spec in self.sides.items()}
        self.learners: dict[str, DeepLearner] = {}
        if config.deep:
            for index, name in enumerate(("red", "blue")):
                self.learners[name] = DeepLearner.from_campaign(ROOT / "artifacts" / "checkpoints" / "neural.pt",
                                                                seed=config.seed + index)
        if resume and (self.run_dir / "state.json").is_file():
            self._load()
        else:
            if resume:
                self.log(f"Sin estado guardado en {self.run_dir.name}: comienza desde la generación 1")
            self.state["blue"].tasks = [self.ops["blue"].random_solution(0.75) for _ in range(config.n_task)]
        self.publish(status="running")

    def _uid(self) -> int:
        return next(self.uids)

    # ------------------------------------------------------------ main loop
    def run(self, max_failures: int = 5) -> None:
        """Generations until the configured count (0 = forever); a failed generation is
        logged, the last saved state is restored and the league carries on."""
        failures = 0
        try:
            while self.config.generations <= 0 or self.generation < self.config.generations:
                try:
                    self.step()
                    failures = 0
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:  # noqa: BLE001 - the league must not die on one bad generation
                    failures += 1
                    self.log(f"  fallo en la generación {self.generation} ({failures}/{max_failures}): "
                             + traceback.format_exc())
                    if failures >= max_failures:
                        raise
                    self._recover()
        finally:
            self.phase = "detenido"
            self.publish(status="stopped")
            self.evaluator.close()

    def step(self) -> None:
        self.generation += 1
        side = "red" if self.generation % 2 else "blue"
        rival = "blue" if side == "red" else "red"
        config = self.config.game_config()
        tasks = self.state[rival].tasks
        if not tasks:
            tasks = self.state[rival].tasks = [self.ops[rival].random_solution(0.75) for _ in range(config.n_task)]
        self.log(f"[gen {self.generation}] iluminando {LABELS[side]} contra {len(tasks)} élites de {LABELS[rival]}")
        bootstrap = list(self.bootstrap) + self._inject(side, tasks)
        self._arm_lamarck()
        self.phase = f"MTMB-ME: iluminando {LABELS[side]}"
        start_evals = self.evaluator.evaluations

        def progress(done: int, archives: list[GrowingArchive]) -> None:
            self.progress = {"done": done, "budget": config.budget}
            self._archive_snapshot(side, archives)
            if done % (config.batch * 2) == 0 or done >= config.budget:
                self.publish()

        archives = mtmb_me(tasks, self.ops[side], bootstrap, self.evaluator, config, progress)
        self.phase = f"Selección de tareas ({self.config.variant}) para {LABELS[rival]}"
        self.publish()
        selection = select_tasks(archives, tasks, self.evaluator, config, self.rng)
        self.state[side].tasks = selection.tasks
        self.bootstrap = selection.bootstrap
        self._crown(side, selection, archives, tasks)
        self._archive_snapshot(side, archives)
        self.phase = "Duelo en vivo de campeones"
        self.publish()
        decisions = self.live_duel()
        if self.config.lessons and decisions is not None:
            self.phase = "Lecciones contrafactuales entre adversarios"
            self.publish()
            self.learn_from_each_other(decisions)
        if self.config.deep and side in self.learners:
            self.phase = f"Aprendiz profundo de {LABELS[side]} (imitación + PPO + destilación)"
            self.publish()
            self.train_deep(side, rival)
        if self.config.share_knowledge:
            self._exchange()
        self.phase = "Midiendo (anclas, métricas QD, Elo)"
        self.publish()
        self.measure(side)
        evaluations = self.evaluator.evaluations - start_evals
        self._record_history(side, archives, selection, evaluations)
        self.phase = "esperando siguiente generación"
        self.publish()
        self._save()

    # ------------------------------------------------------------ internals
    def _inject(self, side: str, tasks: list[Solution]) -> Bootstrap:
        """Lesson-refined champions and deep students enter the archive, evaluated on every task."""
        injections = self.state[side].injections
        self.state[side].injections = []
        if not injections:
            return []
        pairs = [(solution, task) for solution in injections for task in tasks]
        outcomes = self.evaluator.evaluate(pairs)
        entries: Bootstrap = []
        for index, ((solution, task), outcome) in enumerate(zip(pairs, outcomes)):
            entries.append((index % len(tasks), solution, outcome.fitness, outcome.behavior))
        self.log(f"  inyectadas {len(injections)} soluciones aprendidas en {LABELS[side]}")
        return entries

    def _crown(self, side: str, selection: Selection, archives: list[GrowingArchive], tasks: list[Solution]) -> None:
        state = self.state[side]
        if selection.fitness_matrix is not None:
            elites = all_elites(archives)
            means = selection.fitness_matrix.mean(axis=1)
            best = int(np.argmax(means))
            champion, value = elites[best].solution, float(means[best])
        else:
            matrix, _ = self.evaluator.matrix(selection.tasks, tasks)
            means = matrix.mean(axis=1)
            best = int(np.argmax(means))
            champion, value = selection.tasks[best], float(means[best])
        state.champion, state.champion_fitness = champion, value
        state.qd_score = float(sum(archive.qd_score() for archive in archives))
        self.log(f"  campeón {LABELS[side]}: #{champion.uid} ({champion.origin}) fitness medio {value:.3f}")

    def _archive_snapshot(self, side: str, archives: list[GrowingArchive]) -> None:
        cells = []
        for task_index, archive in enumerate(archives):
            for entry in archive.elites:
                cells.append({"task": task_index, "uid": entry.solution.uid, "fitness": round(entry.fitness, 4),
                              "behavior": [round(float(value), 3) for value in entry.behavior],
                              "origin": entry.solution.origin})
        self.state[side].archive = cells
        self.state[side].qd_score = float(sum(archive.qd_score() for archive in archives))

    def _labels(self) -> tuple[str, str]:
        return LABELS["red"], LABELS["blue"]

    def live_duel(self) -> list[Any] | None:
        red, blue = self.state["red"].champion, self.state["blue"].champion
        if red is None or blue is None:
            red = red or (self.state["red"].tasks[0] if self.state["red"].tasks else None)
            blue = blue or (self.state["blue"].tasks[0] if self.state["blue"].tasks else None)
        if red is None or blue is None:
            return None
        teams = (self.sides["red"].build(self.dex, red.team), self.sides["blue"].build(self.dex, blue.team))
        knowledge = (self.evaluator.knowledge["red"], self.evaluator.knowledge["blue"])
        battles = []
        score = [0, 0, 0]
        decisions: list[Any] = []
        for index in range(self.config.duel_battles):
            seed = self.rng.randrange(2 ** 30)
            result = play(self.dex, teams, (policy_for(red), policy_for(blue)), knowledge, seed, record=True,
                          snapshot_every=1 if self.config.lessons else 0)
            score[result.winner] += 1
            self.duel_battles += 1
            decisions.extend(result.decisions)
            self.elo.update(f"R{red.uid}", f"B{blue.uid}", result.fitness)
            battles.append({"seed": seed, "winner": result.winner, "turns": result.turns,
                            "hp": [round(value, 3) for value in result.hp],
                            "log": narrate(result.events, self.dex, self._labels())[:400]})
        self.duel = {"generation": self.generation, "score": score, "battles": battles,
                     "teams": {"red": [spec.name for spec in teams[0]], "blue": [spec.name for spec in teams[1]]},
                     "moves": {"red": [[self.dex.moves[move].name for move in spec.moves] for spec in teams[0]],
                               "blue": [[self.dex.moves[move].name for move in spec.moves] for spec in teams[1]]},
                     "champions": {"red": red.uid, "blue": blue.uid}}
        _atomic_json(self.run_dir / f"duel-{self.generation:04d}.json", self.duel)
        self.log(f"  duelo en vivo: ROJO {score[0]} - {score[1]} AZUL (empates {score[2]})")
        return decisions

    def learn_from_each_other(self, decisions: list[Any]) -> None:
        red, blue = self.state["red"].champion, self.state["blue"].champion
        if red is None or blue is None:
            return
        knowledge = (self.evaluator.knowledge["red"], self.evaluator.knowledge["blue"])
        lessons: list[Lesson] = counterfactual_lessons(decisions, (red.mlp, blue.mlp), knowledge, self.rng,
                                                       rollouts=self.config.rollouts,
                                                       max_points=self.config.lesson_points)
        for side_index, name in enumerate(("red", "blue")):
            champion = red if side_index == 0 else blue
            own = [lesson for lesson in lessons if lesson.learner == side_index]
            if not own:
                continue
            bank = self.lesson_bank[name] + own
            self.lesson_bank[name] = bank[-self.config.lesson_bank:]
            refined, accuracy = apply_lessons(champion.mlp, own, side_index)
            student = Solution(self._uid(), name, champion.team, refined, "lesson", (champion.uid,))
            self.state[name].injections.append(student)
            self.state[name].lessons_learnt += len(own)
            self.log(f"  {LABELS[name]} aprende {len(own)} lecciones del rival (acc {accuracy:.2f})")
        for lesson in lessons:
            entry = lesson.to_json()
            entry.update(generation=self.generation, learner=("red", "blue")[lesson.learner],
                         features={name: round(float(value), 3) for name, value in
                                   zip(("dmg_cur", "ko_prob", "threat", "own_hp", "foe_hp"),
                                       lesson.features[lesson.actions.index(lesson.target)][[1, 3, 18, 15, 16]])})
            self.lesson_log.append(entry)
        self.lesson_log = self.lesson_log[-300:]

    def _arm_lamarck(self) -> None:
        """Offspring of each side are fine-tuned on a sample of what it learnt from the rival."""
        for side_index, name in enumerate(("red", "blue")):
            bank = self.lesson_bank[name]
            ops = self.ops[name]
            if not self.config.lessons or self.config.lamarck_rate <= 0 or len(bank) < 8:
                ops.refine, ops.refine_rate = None, 0.0
                continue
            rng = random.Random(self.rng.randrange(2 ** 31))

            def refine(genome: MLPGenome, bank: list[Lesson] = bank, index: int = side_index,
                       rng: random.Random = rng) -> MLPGenome:
                sample = rng.sample(bank, min(24, len(bank)))
                refined, _ = apply_lessons(genome, sample, index, epochs=20)
                return refined
            ops.refine, ops.refine_rate = refine, self.config.lamarck_rate

    def train_deep(self, side: str, rival: str) -> None:
        learner = self.learners[side]
        state, spec = self.state[side], self.sides[side]
        rival_tasks = self.state[rival].tasks
        if not state.tasks or not rival_tasks:
            return
        knowledge = self.evaluator.knowledge[side]
        own = [(spec.build(self.dex, solution.team), solution.mlp, 1.0) for solution in state.tasks]
        opponents = [(self.sides[rival].build(self.dex, solution.team), solution.mlp) for solution in rival_tasks]
        samples = elite_imitation_samples(self.dex, own, opponents, knowledge, battles=12,
                                          seed=self.rng.randrange(2 ** 30))
        imitation = learner.behavior_clone(samples, epochs=2)
        environment = StadiumEnvironment(self.dex, [team for team, _, _ in own], opponents, knowledge, num_envs=8,
                                         seed=self.rng.randrange(2 ** 30))
        metrics = learner.ppo(environment, updates=self.config.ppo_updates, steps=32)
        start = state.champion.mlp if state.champion is not None else MLPGenome.prior(self.rng)
        student, accuracy = learner.distill_student(self.dex, [team for team, _, _ in own], opponents, start,
                                                    knowledge, battles=8, seed=self.rng.randrange(2 ** 30))
        team = state.champion.team if state.champion is not None else state.tasks[0].team
        state.injections.append(Solution(self._uid(), side, team, student, "deep_student"))
        state.deep = {"imitation_accuracy": round(imitation, 3), "ppo_updates": float(learner.updates),
                      "ppo_win_rate": round(metrics.get("env_win_rate", 0.0), 3),
                      "value_loss": round(metrics.get("value_loss", 0.0), 4),
                      "entropy": round(metrics.get("entropy", 0.0), 3), "student_accuracy": round(accuracy, 3),
                      "samples": float(len(samples))}
        torch.save({"battle_policy": learner.network.state_dict(), "generation": self.generation,
                    "side": side, "note": "stadium-trained; not loaded by campaigns automatically"},
                   self.run_dir / f"deep-{side}.pt")
        self.log(f"  aprendiz profundo {LABELS[side]}: imitación {imitation:.2f}, "
                 f"PPO win {state.deep['ppo_win_rate']:.2f}, estudiante {accuracy:.2f}")

    def _exchange(self) -> None:
        red, blue = self.evaluator.knowledge["red"], self.evaluator.knowledge["blue"]
        gained = exchange(red, blue)
        entry = {"generation": self.generation, "red_gained": gained[0], "blue_gained": gained[1],
                 "cells": red.known_cells()}
        self.exchanges.append(entry)
        self.exchanges = self.exchanges[-200:]
        if any(gained):
            self.log(f"  intercambio de conocimiento: ROJO +{gained[0]} celdas, AZUL +{gained[1]} celdas")

    def measure(self, side: str) -> None:
        state = self.state[side]
        champion = state.champion
        if champion is not None:
            rival = "blue" if side == "red" else "red"
            rival_teams = [task.team for task in self.state[rival].tasks] or [champion.team]
            anchors = {}
            for kind in ("greedy", "random"):
                opponents = [Solution(-1, rival, team, MLPGenome.prior(self.rng), f"anchor:{kind}") for team in rival_teams]
                pairs = [(champion, opponent) for opponent in opponents]
                repeats = max(1, self.config.anchor_battles // max(1, len(pairs) * self.config.repeats))
                outcomes = self.evaluator.evaluate(pairs * repeats)
                anchors[kind] = float(np.mean([outcome.wins / outcome.battles for outcome in outcomes]))
                for outcome in outcomes:
                    self.elo.update(f"{'R' if side == 'red' else 'B'}{champion.uid}", f"ANCLA-{kind}", outcome.fitness)
            state.anchors = anchors
        red_tasks, blue_tasks = self.state["red"].tasks, self.state["blue"].tasks
        if red_tasks and blue_tasks:
            matrix, _ = self.evaluator.matrix(red_tasks, blue_tasks)
            for i, red in enumerate(red_tasks):
                for j, blue in enumerate(blue_tasks):
                    self.elo.update(f"R{red.uid}", f"B{blue.uid}", float(matrix[i, j]))
            clusters = max(2, min(self.config.n_task, len(red_tasks)))
            red_metrics = side_report(matrix, {"red": list(range(len(red_tasks)))}, clusters)["red"]
            blue_metrics = side_report(1 - matrix.T, {"blue": list(range(len(blue_tasks)))}, clusters)["blue"]
            self.metrics = {"red": red_metrics.as_dict(), "blue": blue_metrics.as_dict(),
                            "matrix": matrix.round(3).tolist(),
                            "red_uids": [task.uid for task in red_tasks], "blue_uids": [task.uid for task in blue_tasks]}

    def _record_history(self, side: str, archives: list[GrowingArchive], selection: Selection, evaluations: int) -> None:
        elites = all_elites(archives)
        fitness = [entry.fitness for entry in elites]
        behaviors = np.array([entry.behavior for entry in elites]) if elites else np.zeros((0, len(BEHAVIOR_NAMES)))
        entry = {
            "generation": self.generation, "side": side, "time": round(time.time() - self.started, 1),
            "evaluations": evaluations, "battles": self.evaluator.battles + self.duel_battles,
            "elites": len(elites), "best": round(max(fitness, default=0.0), 4),
            "mean": round(float(np.mean(fitness)) if fitness else 0.0, 4),
            "qd_score": round(self.state[side].qd_score, 3),
            "behavior_spread": round(float(behaviors.std(axis=0).mean()) if len(behaviors) > 1 else 0.0, 4),
            "champion": getattr(self.state[side].champion, "uid", None),
            "champion_fitness": round(self.state[side].champion_fitness, 4),
            "anchors": self.state[side].anchors, "deep": self.state[side].deep,
            "metrics": {key: self.metrics.get(key) for key in ("red", "blue")},
            "knowledge_cells": {name: self.evaluator.knowledge[name].known_cells() for name in ("red", "blue")},
            "lessons": {name: self.state[name].lessons_learnt for name in ("red", "blue")},
            "duel": self.duel.get("score"),
            "elo": self._champion_elo(),
            "origins": _origin_counts(elites),
        }
        self.history.append(entry)
        with (self.run_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=_json_default) + "\n")

    def _champion_elo(self) -> dict[str, float]:
        ratings: dict[str, float] = {}
        for name in ("red", "blue"):
            champion = self.state[name].champion
            if champion is not None:
                key = f"{'R' if name == 'red' else 'B'}{champion.uid}"
                ratings[name] = round(self.elo.ratings.get(key, 1000.0), 1)
        return ratings

    # ---------------------------------------------------------- publication
    def _side_view(self, name: str) -> dict[str, Any]:
        state, spec = self.state[name], self.sides[name]
        champion = state.champion
        view: dict[str, Any] = {
            "label": LABELS[name], "game": spec.game,
            "roster": [{"name": mon.name, "level": mon.level, "origin": mon.origin,
                        "stadium": self.dex.species[mon.species_at(self.dex, spec.level)].name} for mon in spec.roster],
            "archive": state.archive[-400:], "qd_score": round(state.qd_score, 3), "anchors": state.anchors,
            "deep": state.deep, "lessons": state.lessons_learnt, "tasks": [task.uid for task in state.tasks],
            "knowledge": {"cells": self.evaluator.knowledge[name].known_cells(),
                          "facts": self.evaluator.knowledge[name].chart.observations,
                          "table": self.evaluator.knowledge[name].chart.table()},
        }
        if champion is not None:
            team = spec.build(self.dex, champion.team)
            view["champion"] = {"uid": champion.uid, "origin": champion.origin, "fitness": round(state.champion_fitness, 4),
                                "temperature": round(champion.mlp.temperature, 3),
                                "team": [{"name": member.name, "level": member.level,
                                          "moves": [self.dex.moves[move].name for move in member.moves]} for member in team]}
        return view

    def _recover(self) -> None:
        """Back to the last completed generation (or a fresh start) after a failure."""
        if (self.run_dir / "state.json").is_file():
            self._load()
        else:
            self.generation = 0
            self.bootstrap = []
            self.state = {name: SideState() for name in ("red", "blue")}
            self.state["blue"].tasks = [self.ops["blue"].random_solution(0.75) for _ in range(self.config.n_task)]
        self.phase = "recuperado tras un fallo"

    def publish(self, status: str = "running") -> None:
        elapsed = max(1e-6, time.time() - self.started)
        battles = self.evaluator.battles + self.duel_battles
        live = {
            "run": self.run_dir.name, "status": status, "generation": self.generation, "phase": self.phase,
            "progress": self.progress, "updated": time.time(), "started": self.started,
            "config": asdict(self.config), "behavior_names": list(BEHAVIOR_NAMES),
            "sides": {name: self._side_view(name) for name in ("red", "blue")},
            "metrics": self.metrics, "elo": self.elo.top(24), "duel": self.duel,
            "lessons": {"recent": self.lesson_log[-40:], "total": len(self.lesson_log)},
            "exchange": self.exchanges[-40:], "history": self.history[-400:],
            "throughput": {"battles": battles, "evaluations": self.evaluator.evaluations,
                           "battles_per_s": round(battles / elapsed, 1), "elapsed_s": round(elapsed, 1)},
        }
        try:
            _atomic_json(self.run_dir / "live.json", live)
            _atomic_json(self.run_dir.parent / "latest.json", {"run": self.run_dir.name, "path": str(self.run_dir),
                                                               "updated": time.time(), "status": status})
        except OSError as error:  # telemetry is best effort; the league goes on
            self.log(f"  aviso: no se pudo publicar live.json ({error})")

    # ---------------------------------------------------------- persistence
    def _save(self) -> None:
        state = {
            "version": 1, "generation": self.generation, "config": asdict(self.config),
            "uid_next": next(self.uids), "rng": self.rng.getstate(),
            "sides": {name: {"tasks": [task.to_json() for task in value.tasks],
                             "champion": value.champion.to_json() if value.champion else None,
                             "champion_fitness": value.champion_fitness,
                             "injections": [item.to_json() for item in value.injections],
                             "lessons_learnt": value.lessons_learnt, "anchors": value.anchors, "deep": value.deep}
                      for name, value in self.state.items()},
            "bootstrap": [[index, solution.to_json(), fitness, behavior.tolist()]
                          for index, solution, fitness, behavior in self.bootstrap],
            "knowledge": {name: value.state() for name, value in self.evaluator.knowledge.items()},
            "elo": {"ratings": self.elo.ratings, "games": self.elo.games},
            "lesson_log": self.lesson_log, "exchanges": self.exchanges, "metrics": self.metrics,
            "lesson_bank": {name: [lesson.to_state() for lesson in bank] for name, bank in self.lesson_bank.items()},
        }
        self.uids = itertools.count(int(str(state["uid_next"])))
        _atomic_json(self.run_dir / "state.json", state)
        for name, learner in self.learners.items():
            torch.save({"network": learner.network.state_dict(), "optimizer": learner.trainer.optimizer.state_dict(),
                        "updates": learner.updates}, self.run_dir / f"learner-{name}.pt")

    def _load(self) -> None:
        path = self.run_dir / "state.json"
        if not path.is_file():
            raise FileNotFoundError(f"No hay estado que reanudar en {self.run_dir}")
        state = json.loads(path.read_text(encoding="utf-8"))
        self.generation = int(state["generation"])
        self.uids = itertools.count(int(state["uid_next"]))
        rng_state = state["rng"]
        self.rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))
        for name, raw in state["sides"].items():
            side = self.state[name]
            side.tasks = [Solution.from_json(item) for item in raw["tasks"]]
            side.champion = Solution.from_json(raw["champion"]) if raw.get("champion") else None
            side.champion_fitness = float(raw.get("champion_fitness", 0.0))
            side.injections = [Solution.from_json(item) for item in raw.get("injections", [])]
            side.lessons_learnt = int(raw.get("lessons_learnt", 0))
            side.anchors, side.deep = dict(raw.get("anchors", {})), dict(raw.get("deep", {}))
        self.bootstrap = [(int(index), Solution.from_json(solution), float(fitness), np.asarray(behavior, dtype=np.float32))
                          for index, solution, fitness, behavior in state.get("bootstrap", [])]
        for name, knowledge in state.get("knowledge", {}).items():
            self.evaluator.knowledge.setdefault(name, SideKnowledge()).merge_state(knowledge)
        self.elo.ratings = {key: float(value) for key, value in state.get("elo", {}).get("ratings", {}).items()}
        self.elo.games = {key: int(value) for key, value in state.get("elo", {}).get("games", {}).items()}
        self.lesson_log = list(state.get("lesson_log", []))
        self.lesson_bank = {name: [Lesson.from_state(row) for row in rows]
                            for name, rows in dict(state.get("lesson_bank", {"red": [], "blue": []})).items()}
        for name in ("red", "blue"):
            self.lesson_bank.setdefault(name, [])
        self.exchanges = list(state.get("exchanges", []))
        self.metrics = dict(state.get("metrics", {}))
        history = self.run_dir / "history.jsonl"
        if history.is_file():
            self.history = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
        for name, learner in self.learners.items():
            checkpoint = self.run_dir / f"learner-{name}.pt"
            if checkpoint.is_file():
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                learner.network.load_state_dict(saved["network"])
                learner.trainer.optimizer.load_state_dict(saved["optimizer"])
                learner.updates = int(saved.get("updates", 0))
        self.log(f"Reanudando {self.run_dir.name} en la generación {self.generation}")


def _origin_counts(elites: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in elites:
        counts[entry.solution.origin] = counts.get(entry.solution.origin, 0) + 1
    return counts


def latest_run() -> Path | None:
    pointer = STADIUM_ROOT / "latest.json"
    if not pointer.is_file():
        return None
    try:
        path = Path(json.loads(pointer.read_text(encoding="utf-8"))["path"])
    except (ValueError, KeyError, OSError):
        return None
    return path if path.is_dir() else None
