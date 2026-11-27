"""Tasks executed inside rollout worker processes (forward passes only, never ``backward``).

* ``rollout``  - collect PPO trajectories (with GAE) or DQN transitions against the opponent mix;
* ``demos``    - expert demonstrations (ARES prior / Greedy) on Gen1 *training* teams for imitation;
* ``evaluate`` - play fixed evaluation battles of one agent against one opponent on one split.
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .agents import Controller, HeuristicController, PolicyController, QController, make_neural
from .env import BattleEnv, SideCounters
from .model import Weights
from .obs import Observation, flat, stack
from .rollout import Episode, EpisodeResult, gae, run_episodes
from .teams import Split, TrainTeamSampler, default_dex, eval_team

OPPONENT_KINDS = ("self", "snapshot", "greedy", "random")


def battle_seed(*parts: object) -> int:
    return zlib.crc32("/".join(str(part) for part in parts).encode()) & 0x7FFFFFFF


def _heuristics(state: dict[str, Any]) -> dict[str, HeuristicController]:
    cached: dict[str, HeuristicController] | None = state.get("heuristics")
    if cached is None:
        cached = {kind: HeuristicController(kind) for kind in ("random", "greedy", "prior")}
        state["heuristics"] = cached
    return cached


def _learner(state: dict[str, Any], method: str, weights: Weights, epsilon: float) -> PolicyController | QController:
    key = f"learner/{method}"
    controller: PolicyController | QController | None = state.get(key)
    if controller is None:
        controller = QController(weights, epsilon, "learner") if method == "dqn" else \
            PolicyController(method, weights, False, "learner")
        state[key] = controller
    else:
        controller.load(weights)
    if isinstance(controller, QController):
        controller.epsilon = epsilon
    return controller


def _snapshots(state: dict[str, Any], method: str, shipped: dict[str, Weights], keep: list[str]
               ) -> dict[str, Controller]:
    """Frozen league members; a shipped key always replaces a cached one (a new run reuses names)."""
    cache: dict[str, Controller] = state.setdefault(f"snapshots/{method}", {})
    for key, weights in shipped.items():
        # Snapshots act like the learner did when frozen: sampled policy / greedy Q.
        cache[key] = make_neural(method, weights, greedy=False, name=f"snapshot:{key}")
    for key in [key for key in cache if key not in keep]:
        del cache[key]
    return cache


@dataclass
class _Budget:
    target: int
    collected: int = 0


def rollout(task: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    method: str = task["method"]
    dex = default_dex()
    rng = random.Random(f"rollout/{task['seed']}/{state['worker_id']}/{task['iteration']}")
    sampler = TrainTeamSampler(dex, random.Random(rng.random()))
    learner = _learner(state, method, task["weights"], float(task.get("epsilon", 0.0)))
    snapshots = _snapshots(state, method, task.get("snapshots", {}), list(task.get("snapshot_keys", [])))
    available = [key for key in task.get("snapshot_keys", []) if key in snapshots]
    heuristics = _heuristics(state)
    mix: dict[str, float] = task["mix"]
    budget = _Budget(int(task["steps"]))
    gamma, lam = float(task.get("gamma", 0.99)), float(task.get("lam", 0.95))

    def next_episode() -> Episode | None:
        if budget.collected >= budget.target:
            return None
        kinds = list(mix)
        kind = rng.choices(kinds, weights=[mix[name] for name in kinds], k=1)[0]
        if kind == "snapshot" and not available:
            kind = "self"
        opponent: Controller
        if kind == "self":
            opponent = learner
        elif kind == "snapshot":
            opponent = snapshots[rng.choice(available)]
        else:
            opponent = heuristics[kind]
        learner_side = rng.randrange(2)
        env = BattleEnv(dex, sampler.team(), sampler.team(), rng.randrange(2 ** 31))
        controllers: tuple[Controller, Controller] = (learner, opponent) if learner_side == 0 else (opponent, learner)
        record = (True, True) if kind == "self" else (learner_side == 0, learner_side == 1)
        return Episode(env, controllers, record, {"opponent": kind, "learner_side": learner_side})

    results: list[EpisodeResult] = []
    for result in run_episodes(next_episode, int(task.get("concurrency", 16)), rng, gamma):
        results.append(result)
        budget.collected += sum(len(t.actions) for t in result.trajectories if t is not None)
    stats = [{"opponent": r.tag["opponent"], "outcome": r.outcome[r.tag["learner_side"]], "turns": r.turns}
             for r in results]
    trajectories = [t for r in results for t in r.trajectories if t is not None]
    if method == "dqn":
        return {"stats": stats, **_transitions(trajectories, gamma)}
    observations = [obs for t in trajectories for obs in t.observations]
    batch = stack(observations)
    advantages, returns = [], []
    for trajectory in trajectories:
        advantage, value_target = gae(trajectory.rewards, trajectory.values, gamma, lam)
        advantages.append(advantage)
        returns.append(value_target)
    return {"stats": stats, "field": batch.field, "entities": batch.entities, "actions": batch.actions,
            "mask": batch.mask, "species": batch.species,
            "act": np.asarray([a for t in trajectories for a in t.actions], dtype=np.int64),
            "logp": np.asarray([p for t in trajectories for p in t.logps], dtype=np.float32),
            "value": np.asarray([v for t in trajectories for v in t.values], dtype=np.float32),
            "adv": np.concatenate(advantages).astype(np.float32),
            "ret": np.concatenate(returns).astype(np.float32)}


def demos(task: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Expert demonstrations on Gen1 training teams (the same sampler as RL; never Gen2/Gen3).

    Each episode draws an expert (``task["experts"]``: name -> weight) and an opponent
    (``task["opponents"]``); every side played by an expert is recorded. Returned per decision: the
    observation, the expert's engine action, the index of the expert in ``experts`` and the
    discounted return of the same reward the RL learners get (terminal +-1 plus HP shaping).
    """
    dex = default_dex()
    rng = random.Random(f"demos/{task['seed']}/{state['worker_id']}/{task['chunk']}")
    sampler = TrainTeamSampler(dex, random.Random(rng.random()))
    heuristics = _heuristics(state)
    experts: dict[str, float] = dict(task["experts"])
    opponents: dict[str, float] = dict(task["opponents"])
    expert_names = list(experts)
    gamma = float(task.get("gamma", 0.99))
    budget = _Budget(int(task["steps"]))

    def next_episode() -> Episode | None:
        if budget.collected >= budget.target:
            return None
        expert = rng.choices(expert_names, weights=[experts[name] for name in expert_names], k=1)[0]
        opponent = rng.choices(list(opponents), weights=list(opponents.values()), k=1)[0]
        side = rng.randrange(2)
        env = BattleEnv(dex, sampler.team(), sampler.team(), rng.randrange(2 ** 31))
        pair: tuple[Controller, Controller] = (heuristics[expert], heuristics[opponent])
        controllers = pair if side == 0 else (pair[1], pair[0])
        record = (controllers[0].name in experts, controllers[1].name in experts)
        return Episode(env, controllers, record, {"expert": expert, "opponent": opponent, "expert_side": side})

    results: list[EpisodeResult] = []
    for result in run_episodes(next_episode, int(task.get("concurrency", 32)), rng, gamma):
        results.append(result)
        budget.collected += sum(len(t.actions) for t in result.trajectories if t is not None)
    sources: list[int] = []
    returns: list[NDArray[np.float32]] = []
    observations: list[Observation] = []
    actions: list[int] = []
    for result in results:
        for side, trajectory in enumerate(result.trajectories):
            if trajectory is None:
                continue
            name = result.tag["expert"] if side == int(result.tag["expert_side"]) else result.tag["opponent"]
            observations.extend(trajectory.observations)
            actions.extend(trajectory.actions)
            sources.extend([expert_names.index(str(name))] * len(trajectory.actions))
            returns.append(gae(trajectory.rewards, [0.0] * len(trajectory.rewards), gamma, 1.0)[1])
    stats = [{"expert": r.tag["expert"], "opponent": r.tag["opponent"],
              "outcome": r.outcome[int(r.tag["expert_side"])], "turns": r.turns} for r in results]
    batch = stack(observations)
    return {"stats": stats, "field": batch.field, "entities": batch.entities, "actions": batch.actions,
            "mask": batch.mask, "species": batch.species, "act": np.asarray(actions, dtype=np.int64),
            "ret": np.concatenate(returns).astype(np.float32), "source": np.asarray(sources, dtype=np.int64)}


def _transitions(trajectories: list[Any], gamma: float) -> dict[str, Any]:
    """One-step transitions (s, a, r, s', mask', done) of each finished trajectory."""
    states, nexts, masks, actions, rewards, dones = [], [], [], [], [], []
    for trajectory in trajectories:
        vectors = [flat(obs) for obs in trajectory.observations]
        size = len(vectors)
        for step in range(size):
            states.append(vectors[step])
            actions.append(trajectory.actions[step])
            rewards.append(trajectory.rewards[step])
            last = step + 1 >= size
            nexts.append(vectors[step] if last else vectors[step + 1])
            masks.append(trajectory.observations[step].mask if last else trajectory.observations[step + 1].mask)
            dones.append(last)
    return {"state": np.asarray(states, dtype=np.float16), "next": np.asarray(nexts, dtype=np.float16),
            "next_mask": np.asarray(masks, dtype=np.bool_), "act": np.asarray(actions, dtype=np.int64),
            "reward": np.asarray(rewards, dtype=np.float32), "done": np.asarray(dones, dtype=np.bool_)}


def _agent(task_agent: dict[str, Any], state: dict[str, Any]) -> Controller:
    if "heuristic" in task_agent:
        return _heuristics(state)[str(task_agent["heuristic"])]
    if "lookahead" in task_agent:  # analysis-only full-information planner (``headroom``)
        from .headroom import LookaheadController

        return LookaheadController(int(task_agent["lookahead"]))
    return make_neural(str(task_agent["method"]), task_agent["weights"], bool(task_agent.get("greedy", True)),
                       str(task_agent.get("name", "agent")))


def counters_json(counters: SideCounters) -> dict[str, int]:
    return {"voluntary": counters.voluntary, "switches": counters.switches,
            "panic_switches": counters.panic_switches, "move_choices": counters.move_choices,
            "ineffective": counters.ineffective}


def evaluate(task: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    """Battle ``i``: team pair ``i // 2``; odd battles swap both sides and teams of the agent."""
    dex = default_dex()
    split: Split = task["split"]
    agent = _agent(task["agent"], state)
    opponent = _agent(task["opponent"], state)
    indices: list[int] = list(task["indices"])
    rng = random.Random(f"evaluate/{split}/{indices[0] if indices else 0}/{task.get('seed', 0)}")
    queue = list(indices)

    def next_episode() -> Episode | None:
        if not queue:
            return None
        index = queue.pop(0)
        pair, flip = divmod(index, 2)
        team_a, team_b = eval_team(dex, split, pair, 0), eval_team(dex, split, pair, 1)
        env = BattleEnv(dex, team_a, team_b, battle_seed("eval", split, pair, task.get("seed", 0)))
        controllers: tuple[Controller, Controller] = (agent, opponent) if flip == 0 else (opponent, agent)
        return Episode(env, controllers, (False, False), {"index": index, "agent_side": flip})

    records = []
    for result in run_episodes(next_episode, int(task.get("concurrency", 32)), rng):
        side = int(result.tag["agent_side"])
        records.append({"index": result.tag["index"], "outcome": result.outcome[side], "turns": result.turns,
                        **counters_json(result.counters[side])})
    return sorted(records, key=lambda record: int(record["index"]))


def ping(task: dict[str, Any], state: dict[str, Any]) -> int:
    return int(state["worker_id"])


def handle(task: dict[str, Any], state: dict[str, Any]) -> Any:
    kind = task["type"]
    if kind == "rollout":
        return rollout(task, state)
    if kind == "demos":
        return demos(task, state)
    if kind == "evaluate":
        return evaluate(task, state)
    if kind == "ping":
        return ping(task, state)
    raise ValueError(f"unknown task {kind}")
