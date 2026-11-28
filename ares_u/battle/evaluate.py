"""Evaluation harness of E1.

For each split (``gen1_train`` = fresh teams from the training distribution, ``gen1_heldout``,
``gen2``, ``gen3``) and opponent (Random, Greedy, ARES prior expert) an agent plays ``N`` battles on
fixed team pairs: battle ``i`` uses team pair ``i // 2``
and odd battles swap the agent's side *and* team, so every pairing is seen from both seats.
Seeds are fixed, so two agents face exactly the same battles.

Reported per (split, opponent): score = (wins + draws/2) / N with a Wilson 95% interval, wins,
losses, draws, mean turns and anti-vice metrics of the agent: switch rate, panic-switch rate
(voluntary switches that directly follow a voluntary switch, per voluntary decision) and
ineffective-move rate (damaging moves chosen into a type immunity, per move choice).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .agents import HEURISTICS
from .ppo import load_weights
from .rollout import WorkerPool
from .teams import EVAL_SPLITS
from .training import split_indices

Z95 = 1.959963984540054


def wilson(successes: float, total: int, z: float = Z95) -> tuple[float, float, float]:
    """Wilson score interval (point estimate, low, high); ``successes`` may count draws as 0.5."""
    if total <= 0:
        return 0.0, 0.0, 1.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return p, max(0.0, centre - half), min(1.0, centre + half)


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    wins = sum(1 for record in records if record["outcome"] > 0)
    losses = sum(1 for record in records if record["outcome"] < 0)
    draws = total - wins - losses
    score, low, high = wilson(wins + 0.5 * draws, total)
    voluntary = sum(int(record["voluntary"]) for record in records)
    moves = sum(int(record["move_choices"]) for record in records)
    return {"n": total, "score": score, "ci_low": low, "ci_high": high, "wins": wins, "losses": losses,
            "draws": draws, "mean_turns": float(np.mean([record["turns"] for record in records])) if records else 0.0,
            "switch_rate": sum(int(r["switches"]) for r in records) / max(1, voluntary),
            "panic_switch_rate": sum(int(r["panic_switches"]) for r in records) / max(1, voluntary),
            "ineffective_rate": sum(int(r["ineffective"]) for r in records) / max(1, moves)}


def agent_spec(checkpoint: Path | None = None, heuristic: str | None = None, greedy: bool = True) -> dict[str, Any]:
    if heuristic is not None:
        return {"heuristic": heuristic, "name": heuristic}
    if checkpoint is None:
        raise ValueError("an agent needs a checkpoint or a heuristic")
    method, weights, _ = load_weights(checkpoint)
    return {"method": method, "weights": weights, "greedy": greedy, "name": str(checkpoint)}


def play(pool: WorkerPool, agent: dict[str, Any], opponent: dict[str, Any], split: str, battles: int,
         seed: int = 0) -> list[dict[str, Any]]:
    tasks = [{"type": "evaluate", "split": split, "indices": chunk, "seed": seed, "agent": agent,
              "opponent": opponent} for chunk in split_indices(battles, pool.size)]
    return [record for result in pool.map(tasks) for record in result]


def evaluate_agent(pool: WorkerPool, agent: dict[str, Any], battles: int, splits: tuple[str, ...] = EVAL_SPLITS,
                   opponents: tuple[str, ...] = HEURISTICS, seed: int = 0) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for split in splits:
        results[split] = {}
        for opponent in opponents:
            records = play(pool, agent, {"heuristic": opponent}, split, battles, seed)
            results[split][opponent] = summarise(records)
    return {"agent": agent.get("name", ""), "battles": battles, "seed": seed, "results": results}


def baseline_matrix(pool: WorkerPool, battles: int, splits: tuple[str, ...] = EVAL_SPLITS,
                    seed: int = 0) -> dict[str, Any]:
    """Every heuristic against every heuristic (row agent's score), per split."""
    matrix: dict[str, Any] = {}
    for split in splits:
        matrix[split] = {}
        for row in HEURISTICS:
            matrix[split][row] = {}
            for column in HEURISTICS:
                records = play(pool, {"heuristic": row}, {"heuristic": column}, split, battles, seed)
                matrix[split][row][column] = summarise(records)
    return {"battles": battles, "seed": seed, "matrix": matrix}


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Atomic write (temporary file + rename): a killed run never leaves a truncated JSON behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def evaluation_complete(path: Path, battles: int, splits: tuple[str, ...] = EVAL_SPLITS,
                        opponents: tuple[str, ...] = HEURISTICS) -> bool:
    """True when ``path`` holds an evaluation with every split x opponent at >= ``battles`` battles."""
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = payload["results"] if "results" in payload else payload["matrix"]
        for split in splits:
            for opponent in opponents:
                entry = results[split][opponent]
                cells = entry.values() if "n" not in entry else [entry]
                if any(int(cell["n"]) < battles for cell in cells):
                    return False
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True
