"""Worker-side tasks of the brain evaluation (``WorkerPool(handler_path="ares_u.brain.eval_tasks:handle")``).

Battle ``i`` of a split is exactly the E1 evaluation battle ``i`` (``ares_u.battle.tasks.evaluate``):
team pair ``i // 2`` (``eval_team``), engine seed ``battle_seed("eval", split, pair, seed)``, and odd
battles swap the agent's side. Every agent therefore meets the same opponents with the same teams,
so per-battle scores pair up across agents.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from ares_u.battle.agents import Controller, HeuristicController, make_neural
from ares_u.battle.env import BattleEnv
from ares_u.battle.rollout import Episode, run_episodes
from ares_u.battle.tasks import battle_seed
from ares_u.battle.teams import Split, default_dex, eval_team


def _controller(spec: dict[str, Any], state: dict[str, Any]) -> Controller:
    key = repr(sorted(spec.items()))
    cache: dict[str, Controller] = state.setdefault("controllers", {})
    if key in cache:
        return cache[key]
    controller: Controller
    if "heuristic" in spec:
        controller = HeuristicController(str(spec["heuristic"]))
    elif "lookahead" in spec:
        from ares_u.battle.headroom import LookaheadController

        controller = LookaheadController(int(spec["lookahead"]))
    elif "campaign" in spec:
        from .stadium_adapter import CampaignController

        controller = CampaignController()
    elif "native" in spec:
        from ares_u.battle.ppo import load_weights

        method, weights, _ = load_weights(Path(str(spec["native"])))
        controller = make_neural(method, weights, True, "native")
    elif "brain" in spec:
        from .brain import BattleBrain
        from .stadium_adapter import BrainController

        checkpoint = spec.get("checkpoint")
        brain = BattleBrain(str(spec["brain"]), Path(str(checkpoint)) if checkpoint else None,
                            strategy_weight=float(spec.get("weight", 4.0)))
        controller = BrainController(brain, name=str(spec.get("name", spec["brain"])))
    else:
        raise ValueError(f"unknown agent spec {spec}")
    cache[key] = controller
    return controller


def evaluate(task: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    dex = default_dex()
    split: Split = task["split"]
    agent = _controller(task["agent"], state)
    opponent = _controller(task["opponent"], state)
    indices: list[int] = list(task["indices"])
    seed = int(task.get("seed", 0))
    rng = random.Random(f"evaluate/{split}/{indices[0] if indices else 0}/{seed}")
    queue = list(indices)

    def next_episode() -> Episode | None:
        if not queue:
            return None
        index = queue.pop(0)
        pair, flip = divmod(index, 2)
        env = BattleEnv(dex, eval_team(dex, split, pair, 0), eval_team(dex, split, pair, 1),
                        battle_seed("eval", split, pair, seed))
        controllers = (agent, opponent) if flip == 0 else (opponent, agent)
        return Episode(env, controllers, (False, False), {"index": index, "agent_side": flip})

    records = []
    for result in run_episodes(next_episode, int(task.get("concurrency", 32)), rng):
        side = int(result.tag["agent_side"])
        counters = result.counters[side]
        records.append({"index": result.tag["index"], "score": (float(result.outcome[side]) + 1) / 2,
                        "turns": result.turns, "switches": counters.switches,
                        "ineffective": counters.ineffective, "voluntary": counters.voluntary})
    return sorted(records, key=lambda record: int(record["index"]))


def handle(task: dict[str, Any], state: dict[str, Any]) -> Any:
    kind = task.get("type")
    if kind == "evaluate":
        return evaluate(task, state)
    if kind == "ping":
        return int(state["worker_id"])
    raise ValueError(f"unknown task type {kind}")
