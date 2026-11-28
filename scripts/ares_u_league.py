"""ARES league: diverse rivals, measured diversity, a PFSP champion and its honest test.

Stages (each one resumable; outputs under ``artifacts/ares_u/league``):

    python -m scripts.ares_u_league select     # best E1 PPO seed on validation battles (starting point)
    python -m scripts.ares_u_league diversity  # cross-play round robin, behaviour profiles, decision probe
    python -m scripts.ares_u_league protocol   # pre-registered held-out rivals, comparisons and criterion
    python -m scripts.ares_u_league train      # PFSP league training + exploiters (live: artifacts/live/league)
    python -m scripts.ares_u_league evaluate   # test battles 0..N-1 per split (never used before)
    python -m scripts.ares_u_league report     # report.json / report.md
    python -m scripts.ares_u_league all        # every stage in order

Validation battles (selection, diversity, exploiter checks) live in index ranges from 40 000 000 on
(``ares_u.league.evaluation``); the test battles 0..N-1 are read only by ``evaluate``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ares_u.battle.rollout import WorkerPool
from ares_u.league.archetypes import ARCHETYPES
from ares_u.league.evaluation import (DIVERSITY_OFFSET, SELECTION_OFFSET, analyse, crossplay, describe_specs, duel,
                                      probe_tasks, scores, win_rate_table)
from ares_u.league.registry import NEURAL, describe, rival_spec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "ares_u" / "league"
HANDLER = "ares_u.league.tasks:handle"
SPLITS = ("gen1_train", "gen2", "gen3")
E1_PPO = tuple(f"e1_ppo_s{seed}" for seed in range(5))
ALL_ARCHETYPES = tuple(ARCHETYPES)  # greedy, stall, setup, pivot, gambler, disruptor, nash, prior, random
DIVERSITY_RIVALS = (*ALL_ARCHETYPES, "lookahead", *NEURAL)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=_default), encoding="utf-8")
    temporary.replace(path)
    return path


def _default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not serialisable: {type(value)}")


def log(message: str) -> None:
    print(f"[league {time.strftime('%H:%M:%S')}] {message}", flush=True)


# ----------------------------------------------------------------------------- select
def select(pool: WorkerPool, args: argparse.Namespace) -> dict[str, Any]:
    """The E1 PPO seed with the best validation score against Greedy (gen1_train, gen2, gen3)."""
    path = OUT / "selection.json"
    if path.exists():
        return dict(read_json(path))
    greedy = rival_spec("greedy")
    result: dict[str, Any] = {"battles_per_split": args.select_battles, "offset": SELECTION_OFFSET, "seeds": {}}
    for name in E1_PPO:
        spec = rival_spec(name)
        per_split = {split: float(scores(duel(pool, spec, greedy, split, args.select_battles, SELECTION_OFFSET)).mean())
                     for split in SPLITS}
        result["seeds"][name] = {**per_split, "mean": float(np.mean(list(per_split.values())))}
        log(f"select {name}: {result['seeds'][name]}")
    result["best"] = max(result["seeds"], key=lambda name: result["seeds"][name]["mean"])
    result["checkpoint"] = str(NEURAL[result["best"]])
    write_json(path, result)
    return result


# ----------------------------------------------------------------------------- diversity
def diversity(pool: WorkerPool, args: argparse.Namespace) -> dict[str, Any]:
    folder = OUT / "diversity"
    done = folder / "diversity.json"
    if done.exists():
        return dict(read_json(done))
    names = list(args.rivals or DIVERSITY_RIVALS)
    specs = {name: rival_spec(name) for name in names}
    cache_path = folder / "crossplay_records.json"
    cache: dict[str, Any] = read_json(cache_path) if cache_path.exists() else {}
    started = time.perf_counter()

    def progress(key: str, records: list[dict[str, Any]]) -> None:
        write_json(cache_path, cache)
        log(f"crossplay {key}: {float(scores(records).mean()):.3f} over {len(records)} "
            f"({time.perf_counter() - started:.0f}s)")

    crossplay(pool, specs, args.split, args.battles, {"lookahead": args.lookahead_battles}, cache,
              DIVERSITY_OFFSET, progress)
    probe_path = folder / "probe.json"
    if probe_path.exists():
        probe = read_json(probe_path)
    else:
        reference = [rival_spec("greedy"), rival_spec("random")]
        probe = pool.map(probe_tasks(specs, reference, args.probe_battles, pool.size))
        write_json(probe_path, probe)
        log(f"probe: {sum(len(item['choices']) for item in probe)} states")
    result = analyse(names, cache, probe)
    result.update({"split": args.split, "battles_per_pair": args.battles, "lookahead_battles": args.lookahead_battles,
                   "offset": DIVERSITY_OFFSET, "specs": describe_specs(specs),
                   "descriptions": {name: describe(name) for name in names}})
    write_json(done, result)
    (folder / "diversity.md").write_text(diversity_markdown(result), encoding="utf-8")
    return result


def diversity_markdown(result: dict[str, Any]) -> str:
    names = result["names"]
    lines = ["# ARES league - diversity of the rivals", "",
             f"Round robin on validation battles of `{result['split']}` (indices from {result['offset']}; "
             f"{result['battles_per_pair']} battles per pair, {result['lookahead_battles']} with the lookahead "
             "planner); every pair plays the same team pairs with seats swapped.", "",
             "## Elo (Bradley-Terry, 95% bootstrap CI)", "", "| # | rival | Elo | CI | style |", "|---|---|---|---|---|"]
    for rank, name in enumerate(result["ranking"], 1):
        low, high = result["elo_ci"][name]
        lines.append(f"| {rank} | {name} | {result['elo'][name]:.0f} | [{low:.0f}, {high:.0f}] | "
                     f"{result['descriptions'].get(name, '')} |")
    cyc = result["cycles"]
    lines += ["", f"Non-transitive 3-cycles: {cyc['count']} (significant on every edge: {cyc['significant_count']}). "
              f"Pairs whose winner contradicts the Elo order: {result['transitivity_violations']:.1%}; mean |observed - "
              f"predicted| score: {result['rating_residual']:.3f}. Rivals beating everyone: "
              f"{result['dominance']['beats_everyone'] or 'none'}.", ""]
    for example in cyc["significant_examples"][:8] or cyc["examples"][:8]:
        lines.append(f"- {example[0]} > {example[1]} > {example[2]} > {example[0]}")
    lines += ["", "## Behaviour per rival (all its round-robin battles)", "",
              "| rival | switch | attack | status | setup | debuff | disrupt | support | super-eff. | risky | "
              "power | accuracy | dmg/turn | turns | nearest (z-dist) |", "|" + "---|" * 15]
    for name in names:
        rates = result["behaviour"][name]
        nearest = result["behaviour_distance"]["nearest"][name]
        lines.append(f"| {name} | {rates['switch_rate']:.3f} | {rates['attack_rate']:.3f} | {rates['status_rate']:.3f} | "
                     f"{rates['setup_rate']:.3f} | {rates['debuff_rate']:.3f} | {rates['disrupt_rate']:.3f} | "
                     f"{rates['support_rate']:.3f} | {rates['super_effective_rate']:.3f} | {rates['risky_rate']:.3f} | "
                     f"{rates['mean_power']:.0f} | {rates['mean_accuracy']:.3f} | {rates['damage_per_turn']:.3f} | "
                     f"{rates['mean_turns']:.1f} | {nearest['neighbour']} ({nearest['distance']:.2f}) |")
    probe = result.get("disagreement")
    if probe and probe.get("states"):
        lines += ["", f"## Decision disagreement on {probe['states']} shared states", "",
                  "Share of states where the row and column rival choose different actions:", ""]
        lines += win_rate_table(probe["names"], probe["matrix"])
    lines += ["", "## Cross-play score matrix (row player's score)", ""]
    lines += win_rate_table(names, result["score"])
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=6)
    sub = parser.add_subparsers(dest="command", required=True)
    sel = sub.add_parser("select")
    sel.add_argument("--select-battles", type=int, default=1000)
    div = sub.add_parser("diversity")
    div.add_argument("--battles", type=int, default=400)
    div.add_argument("--lookahead-battles", type=int, default=200)
    div.add_argument("--probe-battles", type=int, default=300)
    div.add_argument("--split", default="gen1_train")
    div.add_argument("--rivals", nargs="*")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    with WorkerPool(args.workers, HANDLER) as pool:
        if args.command == "select":
            print(json.dumps(select(pool, args), indent=2))
        elif args.command == "diversity":
            result = diversity(pool, args)
            print(json.dumps({"ranking": result["ranking"], "cycles": result["cycles"]["count"]}, indent=2))


if __name__ == "__main__":
    main()
