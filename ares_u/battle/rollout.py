"""Concurrent battle runner and the multiprocessing worker pool (CPU, forward passes only).

``run_episodes`` advances many battles at once so neural controllers see whole batches. Sides
controlled by a *recording* controller keep a trajectory of their decisions (observation, action,
log-probability, value, reward); rewards follow ``env`` (terminal +-1 plus potential-based HP
shaping). Worker processes never call ``backward()``: the learner is the only process that trains.
"""
from __future__ import annotations

import multiprocessing as mp
import random
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .agents import Choice, Controller, Decision, HeuristicController
from .env import SHAPING, BattleEnv, SideCounters, hp_margin
from .obs import Observation


@dataclass
class Trajectory:
    observations: list[Observation] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    logps: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    potential: float = 0.0


@dataclass
class Episode:
    env: BattleEnv
    controllers: tuple[Controller, Controller]
    record: tuple[bool, bool]
    tag: dict[str, Any] = field(default_factory=dict)
    trajectories: tuple[Trajectory, Trajectory] = field(default_factory=lambda: (Trajectory(), Trajectory()))


@dataclass
class EpisodeResult:
    tag: dict[str, Any]
    outcome: tuple[float, float]
    turns: int
    counters: tuple[SideCounters, SideCounters]
    trajectories: tuple[Trajectory | None, Trajectory | None]


def _close(episode: Episode, gamma: float) -> EpisodeResult:
    env = episode.env
    trajectories: list[Trajectory | None] = []
    for side in (0, 1):
        trajectory = episode.trajectories[side]
        if episode.record[side] and trajectory.actions:
            # Terminal potential is 0: the shaping telescopes (potential-based, policy invariant).
            trajectory.rewards[-1] += env.outcome(side) + SHAPING * (0.0 - trajectory.potential)
            trajectories.append(trajectory)
        else:
            trajectories.append(None)
    return EpisodeResult(episode.tag, (env.outcome(0), env.outcome(1)), env.battle.turn, env.counters,
                         (trajectories[0], trajectories[1]))


def run_episodes(next_episode: Callable[[], Episode | None], concurrency: int, rng: random.Random,
                 gamma: float = 0.99) -> Iterator[EpisodeResult]:
    """Play episodes produced by ``next_episode`` (None = no more), ``concurrency`` at a time."""
    active: list[Episode] = []
    exhausted = False
    while True:
        while not exhausted and len(active) < concurrency:
            episode = next_episode()
            if episode is None:
                exhausted = True
                break
            active.append(episode)
        if not active:
            return
        groups: dict[int, tuple[Controller, list[tuple[int, Decision, bool]]]] = {}
        chosen: list[dict[int, int]] = [{} for _ in active]
        for index, episode in enumerate(active):
            for side in episode.env.pending():
                legal = episode.env.legal(side)
                if len(legal) == 1:
                    chosen[index][side] = legal[0]
                    continue
                controller = episode.controllers[side]
                recording = episode.record[side]
                obs = episode.env.observe(side, legal) if controller.wants_obs or recording else None
                groups.setdefault(id(controller), (controller, []))[1].append(
                    (index, Decision(episode.env, side, legal, obs), recording))
        for controller, items in groups.values():
            choices: list[Choice] = controller.choose([decision for _, decision, _ in items], rng)
            for (index, decision, recording), choice in zip(items, choices):
                if choice.action not in decision.legal:
                    raise RuntimeError(f"{controller.name} chose illegal action {choice.action}")
                chosen[index][decision.side] = choice.action
                if recording and decision.obs is not None:
                    trajectory = active[index].trajectories[decision.side]
                    potential = hp_margin(decision.env.battle, decision.side)
                    if trajectory.actions:
                        trajectory.rewards[-1] += SHAPING * (gamma * potential - trajectory.potential)
                    trajectory.potential = potential
                    trajectory.observations.append(decision.obs)
                    trajectory.actions.append(choice.action)
                    trajectory.logps.append(choice.logp)
                    trajectory.values.append(choice.value)
                    trajectory.rewards.append(0.0)
        remaining: list[Episode] = []
        for index, episode in enumerate(active):
            episode.env.step(chosen[index])
            if episode.env.over:
                yield _close(episode, gamma)
            else:
                remaining.append(episode)
        active = remaining


def gae(rewards: list[float], values: list[float], gamma: float, lam: float
        ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Advantages and returns of one finished trajectory (bootstrap value 0 at the end)."""
    size = len(rewards)
    advantages = np.zeros(size, dtype=np.float32)
    running = 0.0
    for step in range(size - 1, -1, -1):
        following = values[step + 1] if step + 1 < size else 0.0
        delta = rewards[step] + gamma * following - values[step]
        running = delta + gamma * lam * running
        advantages[step] = running
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns.astype(np.float32)


# ---------------------------------------------------------------- worker pool
Handler = Callable[[dict[str, Any], dict[str, Any]], Any]


def _worker_main(connection: Connection, worker_id: int, handler_path: str) -> None:
    import importlib

    import torch

    torch.set_num_threads(1)
    module_name, function_name = handler_path.rsplit(":", 1)
    handler: Handler = getattr(importlib.import_module(module_name), function_name)
    state: dict[str, Any] = {"worker_id": worker_id}
    while True:
        try:
            message = connection.recv()
        except EOFError:
            return
        if message is None:
            return
        try:
            connection.send(("ok", handler(message, state)))
        except Exception:  # surfaced to the learner with the traceback
            connection.send(("error", traceback.format_exc()))


class WorkerPool:
    """Persistent spawn-based workers; each task is a dict handled by ``handler_path`` (module:function)."""

    def __init__(self, workers: int, handler_path: str = "ares_u.battle.tasks:handle") -> None:
        context = mp.get_context("spawn")
        self.connections: list[Any] = []
        self.processes: list[Any] = []
        self._pending = False
        for worker_id in range(workers):
            parent, child = context.Pipe()
            process = context.Process(target=_worker_main, args=(child, worker_id, handler_path), daemon=True)
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)

    @property
    def size(self) -> int:
        return len(self.connections)

    def map(self, tasks: list[dict[str, Any]]) -> list[Any]:
        """Run tasks (at most one in flight per worker, round-robin) and return results in order."""
        if self._pending:
            raise RuntimeError("collect the pending submission before map()")
        results: list[Any] = [None] * len(tasks)
        queue = list(enumerate(tasks))
        busy: dict[int, int] = {}
        from multiprocessing.connection import wait

        while queue or busy:
            for worker in range(self.size):
                if worker not in busy and queue:
                    index, task = queue.pop(0)
                    self.connections[worker].send(task)
                    busy[worker] = index
            ready = wait([self.connections[worker] for worker in busy])
            for worker in list(busy):
                connection = self.connections[worker]
                if connection in ready:
                    status, payload = connection.recv()
                    if status == "error":
                        raise RuntimeError(f"worker {worker} failed:\n{payload}")
                    results[busy.pop(worker)] = payload
        return results

    def broadcast(self, tasks: list[dict[str, Any]]) -> list[Any]:
        """Exactly one task per worker (task i goes to worker i)."""
        self.submit(tasks)
        return self.collect()

    def submit(self, tasks: list[dict[str, Any]]) -> None:
        """Send one task per worker without waiting: the caller may train while the workers play.

        ``collect`` must be called before the pool is used again (``map``/``broadcast``/``submit``).
        """
        if len(tasks) != self.size:
            raise ValueError("submit needs one task per worker")
        if self._pending:
            raise RuntimeError("collect the previous submission first")
        for connection, task in zip(self.connections, tasks):
            connection.send(task)
        self._pending = True

    def collect(self) -> list[Any]:
        """Results of the last ``submit`` (worker order)."""
        if not self._pending:
            raise RuntimeError("nothing was submitted")
        self._pending = False
        results = []
        failure: str | None = None
        for worker, connection in enumerate(self.connections):
            status, payload = connection.recv()
            if status == "error" and failure is None:
                failure = f"worker {worker} failed:\n{payload}"
            results.append(payload)
        if failure is not None:
            raise RuntimeError(failure)
        return results

    @property
    def pending(self) -> bool:
        return self._pending

    def close(self) -> None:
        if self._pending:  # drain an abandoned submission so the workers can exit cleanly
            self._pending = False
            for connection in self.connections:
                try:
                    if connection.poll(30):
                        connection.recv()
                except (EOFError, OSError):
                    pass
        for connection in self.connections:
            try:
                connection.send(None)
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()

    def __enter__(self) -> WorkerPool:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def heuristic(kind: str) -> HeuristicController:
    return HeuristicController(kind)
