"""Worker-side tasks of the league (CPU forward passes only; the learner process is the only one that trains).

* ``league_rollout``  - PPO trajectories of a learner against an opponent distribution over named league
  members (``self`` = the learner on both sides, both recorded). Members are shipped once per worker
  (``ship``) and evicted when they leave the league (``keep``); battles use Gen1 *training* teams.
* ``league_evaluate`` - fixed evaluation battles (E1's battle indices and engine seeds, so a battle
  ``i`` of a split is the same battle as in ``ares_u.battle.tasks.evaluate``) with per-side behaviour
  counters (``BehaviourEnv``).
* ``league_probe``    - reference battles where, at sampled decision points, *every* rival is asked
  what it would do in the same state (decision disagreement between rivals).

Every other task type is delegated to ``ares_u.battle.tasks.handle``.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np

from ares_u.battle.agents import Choice, Controller, Decision, PolicyController
from ares_u.battle.env import BattleEnv
from ares_u.battle.obs import stack
from ares_u.battle.rollout import Episode, EpisodeResult, gae, run_episodes
from ares_u.battle.tasks import battle_seed, counters_json
from ares_u.battle.teams import TrainTeamSampler, default_dex, eval_team

from .behaviour import BehaviourEnv
from .registry import build_controller

SELF = "self"


def controller(spec: dict[str, Any], state: dict[str, Any]) -> Controller:
    """Controller of a spec; weight-free specs (archetypes, heuristics, lookahead) are cached by spec."""
    if "weights" in spec:
        return build_controller(spec)
    key = repr(sorted((k, v) for k, v in spec.items() if k != "name"))
    cache: dict[str, Controller] = state.setdefault("league/static", {})
    if key not in cache:
        cache[key] = build_controller(spec)
    return cache[key]


def _learner(state: dict[str, Any], method: str, weights: Any) -> PolicyController:
    learner: PolicyController | None = state.get(f"league/learner/{method}")
    if learner is None:
        learner = PolicyController(method, weights, False, "learner")
        state[f"league/learner/{method}"] = learner
    else:
        learner.load(weights)
    return learner


def _members(state: dict[str, Any], ship: dict[str, dict[str, Any]], keep: list[str]) -> dict[str, Controller]:
    members: dict[str, Controller] = state.setdefault("league/members", {})
    for name, spec in ship.items():
        members[name] = controller(spec, state)
    for name in [name for name in members if name not in keep]:
        del members[name]
    return members


def ppo_batch(results: list[EpisodeResult], gamma: float, lam: float) -> dict[str, Any]:
    trajectories = [t for r in results for t in r.trajectories if t is not None]
    observations = [obs for t in trajectories for obs in t.observations]
    batch = stack(observations)
    advantages, returns = [], []
    for trajectory in trajectories:
        advantage, value_target = gae(trajectory.rewards, trajectory.values, gamma, lam)
        advantages.append(advantage)
        returns.append(value_target)
    return {"field": batch.field, "entities": batch.entities, "actions": batch.actions, "mask": batch.mask,
            "species": batch.species,
            "act": np.asarray([a for t in trajectories for a in t.actions], dtype=np.int64),
            "logp": np.asarray([p for t in trajectories for p in t.logps], dtype=np.float32),
            "value": np.asarray([v for t in trajectories for v in t.values], dtype=np.float32),
            "adv": np.concatenate(advantages).astype(np.float32),
            "ret": np.concatenate(returns).astype(np.float32)}


def league_rollout(task: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    dex = default_dex()
    rng = random.Random(f"league/{task['seed']}/{task.get('phase', 'main')}/{state['worker_id']}/{task['iteration']}")
    sampler = TrainTeamSampler(dex, random.Random(rng.random()), int(task.get("generation", 1)))
    method = str(task.get("method", "ppo"))
    learner = _learner(state, method, task["weights"])
    members = _members(state, dict(task.get("ship", {})), list(task.get("keep", [])))
    opponents: dict[str, float] = {name: float(weight) for name, weight in task["opponents"].items()
                                   if weight > 0 and (name == SELF or name in members)}
    if not opponents:
        opponents = {SELF: 1.0}
    names = list(opponents)
    weights = [opponents[name] for name in names]
    target = int(task["steps"])
    collected = 0
    gamma, lam = float(task.get("gamma", 0.99)), float(task.get("lam", 0.95))

    def next_episode() -> Episode | None:
        if collected >= target:
            return None
        kind = rng.choices(names, weights=weights, k=1)[0]
        opponent: Controller = learner if kind == SELF else members[kind]
        learner_side = rng.randrange(2)
        env = BattleEnv(dex, sampler.team(), sampler.team(), rng.randrange(2 ** 31))
        pair: tuple[Controller, Controller] = (learner, opponent) if learner_side == 0 else (opponent, learner)
        record = (True, True) if kind == SELF else (learner_side == 0, learner_side == 1)
        return Episode(env, pair, record, {"opponent": kind, "learner_side": learner_side})

    results: list[EpisodeResult] = []
    for result in run_episodes(next_episode, int(task.get("concurrency", 32)), rng, gamma):
        results.append(result)
        collected += sum(len(t.actions) for t in result.trajectories if t is not None)
    stats = [{"opponent": r.tag["opponent"], "outcome": r.outcome[r.tag["learner_side"]], "turns": r.turns}
             for r in results]
    return {"stats": stats, **ppo_batch(results, gamma, lam)}


def league_evaluate(task: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    """Battle ``i``: team pair ``i // 2``; odd battles swap both sides and teams of the agent."""
    dex = default_dex()
    split = task["split"]
    agent = controller(task["agent"], state)
    opponent = controller(task["opponent"], state)
    if agent is opponent:  # a weight-free spec against itself: two instances (cached one + a fresh one)
        opponent = build_controller(task["opponent"])
    indices: list[int] = list(task["indices"])
    seed = int(task.get("seed", 0))
    rng = random.Random(f"league-evaluate/{split}/{indices[0] if indices else 0}/{seed}")
    queue = list(indices)

    def next_episode() -> Episode | None:
        if not queue:
            return None
        index = queue.pop(0)
        pair, flip = divmod(index, 2)
        env = BehaviourEnv(dex, eval_team(dex, split, pair, 0), eval_team(dex, split, pair, 1),
                           battle_seed("eval", split, pair, seed))
        controllers: tuple[Controller, Controller] = (agent, opponent) if flip == 0 else (opponent, agent)
        return Episode(env, controllers, (False, False), {"index": index, "agent_side": flip, "_env": env})

    records = []
    for result in run_episodes(next_episode, int(task.get("concurrency", 32)), rng):
        side = int(result.tag["agent_side"])
        env = result_env(result)
        record: dict[str, Any] = {"index": result.tag["index"], "outcome": result.outcome[side], "turns": result.turns,
                                  **counters_json(result.counters[side])}
        if env is not None:
            record["agent"] = env.behaviour[side].to_json()
            record["opponent"] = env.behaviour[1 - side].to_json()
        records.append(record)
    return sorted(records, key=lambda item: int(item["index"]))


def result_env(result: EpisodeResult) -> BehaviourEnv | None:
    env = result.tag.get("_env")
    return env if isinstance(env, BehaviourEnv) else None


def league_probe(task: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Reference battles; at sampled voluntary decisions every rival answers for the same state."""
    dex = default_dex()
    split = task.get("split", "gen1_train")
    rivals = {name: controller(spec, state) for name, spec in task["rivals"].items()}
    names = list(rivals)
    reference = [controller(spec, state) for spec in task["reference"]]
    if reference[0] is reference[1]:
        reference[1] = build_controller(task["reference"][1])
    probability = float(task.get("probability", 0.5))
    rng = random.Random(f"league-probe/{task['indices'][0]}/{task.get('seed', 0)}")
    choices: list[list[int]] = []
    legal_sizes: list[int] = []
    for index in task["indices"]:
        pair, flip = divmod(int(index), 2)
        env = BattleEnv(dex, eval_team(dex, split, pair, 0), eval_team(dex, split, pair, 1),
                        battle_seed("probe", split, pair, task.get("seed", 0)))
        players = (reference[0], reference[1]) if flip == 0 else (reference[1], reference[0])
        while not env.over:
            actions: dict[int, int] = {}
            for side in env.pending():
                legal = env.legal(side)
                if len(legal) == 1:
                    actions[side] = legal[0]
                    continue
                needs_obs = players[side].wants_obs
                obs = env.observe(side, legal) if needs_obs or any(r.wants_obs for r in rivals.values()) else None
                decision = Decision(env, side, legal, obs)
                if env.voluntary(side) and rng.random() < probability:
                    row = []
                    for name in names:
                        picked: Choice = rivals[name].choose([decision], rng)[0]
                        row.append(int(picked.action))
                    choices.append(row)
                    legal_sizes.append(len(legal))
                actions[side] = int(players[side].choose([decision], rng)[0].action)
            env.step(actions)
    return {"names": names, "choices": choices, "legal": legal_sizes}


def handle(task: dict[str, Any], state: dict[str, Any]) -> Any:
    kind = task["type"]
    if kind == "league_rollout":
        return league_rollout(task, state)
    if kind == "league_evaluate":
        return league_evaluate(task, state)
    if kind == "league_probe":
        return league_probe(task, state)
    from ares_u.battle.tasks import handle as battle_handle

    return battle_handle(task, state)
