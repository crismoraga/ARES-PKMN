"""Paired Stadium evaluation of the battle brain (ground truth: who wins the battle).

    python -m ares_u.brain.evaluate run --splits gen1_heldout gen2 gen3 --battles 2000 \
        --opponents greedy prior --agents brain policy strategy campaign native --workers 3
    python -m ares_u.brain.evaluate run --opponents lookahead --battles 400 ...   (expensive opponent)
    python -m ares_u.brain.evaluate tune --battles 600          (strategy weight, validation split only)
    python -m ares_u.brain.evaluate report

Agents: ``brain`` (E1 policy + strategy), ``policy`` (raw E1 policy through the BattleView path),
``strategy`` (strategic layer alone), ``campaign`` (Stadium port of the campaign heuristic) and
``native`` (E1 policy on its native observation: checks that the view path loses nothing).
Score = (win + draw/2); battles ``2p`` and ``2p+1`` are the same team pair with sides swapped, so the
statistical unit is the team pair: means and paired differences get 95 % bootstrap CIs over pairs
and two-sided sign-flip p-values (``ares_u.battle.paired``), Holm-corrected over the primary family.
Results: ``artifacts/brain/eval/per_battle.json`` (resumable per agent/split/opponent),
``summary.json`` and ``report.md``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ares_u.battle.paired import bootstrap_mean_ci, holm, sign_flip_p
from ares_u.battle.rollout import WorkerPool

from .brain import DEFAULT_CHECKPOINT, STRATEGY_WEIGHT

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "brain" / "eval"
HANDLER = "ares_u.brain.eval_tasks:handle"
CHUNK = 50
TEST_SPLITS = ("gen1_heldout", "gen2", "gen3")
PRIMARY = (("brain", "policy"), ("brain", "strategy"), ("brain", "campaign"), ("strategy", "campaign"),
           ("policy", "native"))


def agent_spec(name: str, weight: float = STRATEGY_WEIGHT) -> dict[str, Any]:
    if name in ("brain", "policy", "strategy"):
        return {"brain": name, "weight": weight, "checkpoint": str(DEFAULT_CHECKPOINT), "name": name}
    if name.startswith("brain_w"):
        return {"brain": "brain", "weight": float(name[len("brain_w"):]), "checkpoint": str(DEFAULT_CHECKPOINT),
                "name": name}
    if name == "campaign":
        return {"campaign": True}
    if name == "native":
        return {"native": str(DEFAULT_CHECKPOINT)}
    if name in ("greedy", "prior", "random"):
        return {"heuristic": name}
    raise ValueError(f"unknown agent {name}")


def opponent_spec(name: str, lookahead: int) -> dict[str, Any]:
    if name == "lookahead":
        return {"lookahead": lookahead}
    return {"heuristic": name}


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    temporary.replace(path)


def play(pool: WorkerPool, agent: dict[str, Any], opponent: dict[str, Any], split: str, battles: int,
         start: int = 0) -> list[dict[str, Any]]:
    indices = list(range(start, start + battles))
    tasks = [{"type": "evaluate", "split": split, "indices": indices[offset:offset + CHUNK], "seed": 0,
              "agent": agent, "opponent": opponent, "concurrency": 32} for offset in range(0, len(indices), CHUNK)]
    return sorted((record for result in pool.map(tasks) for record in result), key=lambda row: int(row["index"]))


def run(args: argparse.Namespace) -> Path:
    out = OUT / (args.file or "per_battle.json")
    data = read(out)
    data.setdefault("results", {})
    data.setdefault("meta", {})
    with WorkerPool(args.workers, HANDLER) as pool:
        for opponent in args.opponents:
            for split in args.splits:
                for name in args.agents:
                    key = f"{name}|{split}|{opponent}"
                    have = data["results"].get(key)
                    if have is not None and len(have) >= args.battles:
                        continue
                    started = time.perf_counter()
                    records = play(pool, agent_spec(name, args.weight), opponent_spec(opponent, args.lookahead),
                                   split, args.battles)
                    data["results"][key] = [row["score"] for row in records]
                    data["meta"][key] = {"seconds": round(time.perf_counter() - started, 1),
                                         "turns": float(np.mean([row["turns"] for row in records])),
                                         "switch_rate": float(np.sum([row["switches"] for row in records]) /
                                                              max(1, np.sum([row["voluntary"] for row in records]))),
                                         "ineffective": int(np.sum([row["ineffective"] for row in records]))}
                    write(out, data)
                    score = float(np.mean(data["results"][key]))
                    print(f"[brain-eval] {key}: {score:.4f} over {len(records)} battles "
                          f"({data['meta'][key]['seconds']} s)", flush=True)
    return out


def pairs(scores: list[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    usable = values.size - values.size % 2
    return np.asarray(values[:usable].reshape(-1, 2).mean(axis=1), dtype=np.float64)


def summarize(data: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, list[float]] = data.get("results", {})
    cells: dict[str, Any] = {}
    groups: dict[tuple[str, str], dict[str, list[float]]] = {}
    for key, scores in results.items():
        agent, split, opponent = key.split("|")
        groups.setdefault((split, opponent), {})[agent] = scores
        unit = pairs(scores)
        low, high = bootstrap_mean_ci(unit)
        cells[key] = {"agent": agent, "split": split, "opponent": opponent, "battles": len(scores),
                      "score": float(np.mean(scores)), "ci": [low, high], **data.get("meta", {}).get(key, {})}
    contrasts: list[dict[str, Any]] = []
    for (split, opponent), agents in sorted(groups.items()):
        for first, second in PRIMARY:
            if first in agents and second in agents:
                size = min(len(agents[first]), len(agents[second]))
                diff = pairs(agents[first][:size]) - pairs(agents[second][:size])
                low, high = bootstrap_mean_ci(diff)
                contrasts.append({"split": split, "opponent": opponent, "first": first, "second": second,
                                  "pairs": int(diff.size), "diff": float(diff.mean()), "ci": [low, high],
                                  "p": sign_flip_p(diff)})
    # pooled over the test splits (same opponent): concatenated team pairs
    for opponent in sorted({key[1] for key in groups}):
        for first, second in PRIMARY:
            diffs = []
            for split in TEST_SPLITS:
                agents = groups.get((split, opponent), {})
                if first in agents and second in agents:
                    size = min(len(agents[first]), len(agents[second]))
                    diffs.append(pairs(agents[first][:size]) - pairs(agents[second][:size]))
            if len(diffs) == len(TEST_SPLITS):
                diff = np.concatenate(diffs)
                low, high = bootstrap_mean_ci(diff)
                contrasts.append({"split": "test_pooled", "opponent": opponent, "first": first, "second": second,
                                  "pairs": int(diff.size), "diff": float(diff.mean()), "ci": [low, high],
                                  "p": sign_flip_p(diff)})
    pooled = [item for item in contrasts if item["split"] == "test_pooled"]
    for item, adjusted in zip(pooled, holm([item["p"] for item in pooled])):
        item["p_holm"] = adjusted
    pooled_cells: dict[str, Any] = {}
    for opponent in sorted({key[1] for key in groups}):
        for agent in sorted({name for agents in groups.values() for name in agents}):
            parts = [groups.get((split, opponent), {}).get(agent) for split in TEST_SPLITS]
            if all(part is not None for part in parts):
                unit = np.concatenate([pairs(part) for part in parts if part is not None])
                low, high = bootstrap_mean_ci(unit)
                pooled_cells[f"{agent}|test_pooled|{opponent}"] = {"score": float(unit.mean()), "ci": [low, high],
                                                                   "pairs": int(unit.size)}
    return {"cells": cells, "pooled": pooled_cells, "contrasts": contrasts}


def markdown(summary: dict[str, Any]) -> str:
    lines = ["# Battle brain: paired Stadium evaluation", "",
             "Score = (wins + draws/2)/N vs the opponent; 95 % bootstrap CI over team pairs (each pair = both "
             "seatings). Test splits: gen1_heldout (Gen-1 species sets never used in E1 training), gen2, gen3 "
             "(species never seen). Agents: brain = E1 policy + strategy; policy = raw E1 policy through the "
             "BattleView path; strategy = strategic layer alone; campaign = Stadium port of the campaign "
             "heuristic; native = E1 policy on its native observation.", ""]
    cells = summary["cells"]
    opponents = sorted({cell["opponent"] for cell in cells.values()})
    agents = sorted({cell["agent"] for cell in cells.values()})
    splits = sorted({cell["split"] for cell in cells.values()})
    for opponent in opponents:
        lines += [f"## vs {opponent}", "", "| agent | " + " | ".join(splits) + " | test pooled |",
                  "|---|" + "---|" * (len(splits) + 1)]
        for agent in agents:
            row = [agent]
            for split in splits:
                cell = cells.get(f"{agent}|{split}|{opponent}")
                row.append("" if cell is None else
                           f"{cell['score']:.3f} [{cell['ci'][0]:.3f}, {cell['ci'][1]:.3f}] (n={cell['battles']})")
            pooled = summary["pooled"].get(f"{agent}|test_pooled|{opponent}")
            row.append("" if pooled is None else
                       f"{pooled['score']:.3f} [{pooled['ci'][0]:.3f}, {pooled['ci'][1]:.3f}]")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines += ["## Paired contrasts (first - second)", "",
              "| split | opponent | contrast | pairs | diff | 95% CI | p (sign-flip) | p Holm |", "|---|---|---|---|---|---|---|---|"]
    for item in summary["contrasts"]:
        lines.append(f"| {item['split']} | {item['opponent']} | {item['first']} - {item['second']} | {item['pairs']} | "
                     f"{item['diff']:+.4f} | [{item['ci'][0]:+.4f}, {item['ci'][1]:+.4f}] | {item['p']:.4f} | "
                     f"{item.get('p_holm', float('nan')):.4f} |")
    return "\n".join(lines) + "\n"


def report(args: argparse.Namespace) -> Path:
    data = read(OUT / (args.file or "per_battle.json"))
    summary = summarize(data)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    path = OUT / "report.md"
    path.write_text(markdown(summary), encoding="utf-8")
    return path


def tune(args: argparse.Namespace) -> Path:
    """Strategy weight on the validation split ``gen1_train`` only (never on the test splits)."""
    args.splits = ["gen1_train"]
    args.agents = [f"brain_w{weight:g}" for weight in args.weights] + ["policy", "strategy"]
    args.file = "tuning.json"
    return run(args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("run", "tune", "report"))
    parser.add_argument("--splits", nargs="+", default=list(TEST_SPLITS))
    parser.add_argument("--agents", nargs="+", default=["brain", "policy", "strategy", "campaign", "native"])
    parser.add_argument("--opponents", nargs="+", default=["greedy", "prior"])
    parser.add_argument("--battles", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--lookahead", type=int, default=8)
    parser.add_argument("--weight", type=float, default=STRATEGY_WEIGHT)
    parser.add_argument("--weights", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--file", default="")
    args = parser.parse_args(argv)
    if args.stage == "run":
        print(run(args))
    elif args.stage == "tune":
        print(tune(args))
    else:
        print(report(args))


if __name__ == "__main__":
    main()
