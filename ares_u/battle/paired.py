"""Battle-level paired analysis of E1 — the primary statistics of the report.

Why. ``report.py`` compares methods with per-seed scores (5 vs 5 seeds), whose smallest attainable
two-sided Mann-Whitney p is 0.0079 and which ignores that every method plays *the same* evaluation
battles (same team pair, seat and engine seed). Here each final checkpoint is re-evaluated keeping
the outcome of every battle, and methods are compared battle by battle.

Unit of analysis. Battle ``i`` belongs to team pair ``i // 2`` (the two battles of a pair swap
seats). For a method, the score of a battle is averaged over its seeds; the two battles of a pair
are averaged into one pair score, so the *team pair* is the independent unit (N/2 pairs).

Tests (all two-sided, sign-flip permutation over pairs, 20 000 draws, bootstrap 95% CI over pairs):

* parity: ``pair score - 0.5`` (does the method beat the opponent?);
* contrast A-B: ``pair score_A - pair score_B`` on the same pairs (paired);
* generalisation gap: ``mean(gen1_train) - mean(split)`` (different battles, unpaired: label
  permutation over pairs, bootstrap of the difference of means).

Pre-registered family (fixed before the final runs; Holm over the family): opponent Greedy, splits
gen2 and gen3 (zero-shot), hypotheses H1 ppo > parity, H2 ppo > ppo_ids, H3 ppo > dqn, H4 bc_ppo vs
ppo. Everything else is reported as exploratory (unadjusted p). Seed variability is reported apart
as the t-based 95% CI of the per-seed mean scores (df = seeds - 1).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .agents import HEURISTICS
from .evaluate import agent_spec, play, write_json
from .rollout import WorkerPool
from .teams import EVAL_SPLITS

Array = NDArray[np.float64]
METHODS = ("bc", "ppo", "bc_ppo", "ppo_ids", "dqn")
PRIMARY_OPPONENT = "greedy"
PRIMARY_SPLITS = ("gen2", "gen3")
FAMILY: tuple[tuple[str, str, str | None], ...] = (
    ("H1", "ppo", None),          # parity: ppo vs Greedy > 0.5
    ("H2", "ppo", "ppo_ids"),     # species IDs hurt zero-shot transfer
    ("H3", "ppo", "dqn"),         # entity Transformer + PPO vs the course baseline (DQN-MLP)
    ("H4", "bc_ppo", "ppo"),      # effect of the imitation warm-start
)
DRAWS = 20_000


# ----------------------------------------------------------------------------- evaluation

def evaluate_run(run: Path, battles: int, workers: int, methods: tuple[str, ...] = METHODS,
                 splits: tuple[str, ...] = EVAL_SPLITS, opponents: tuple[str, ...] = HEURISTICS) -> Path:
    """Per-battle outcomes of every ``<run>/<method>/seed<k>/final.pt``; resumable per (method, seed)."""
    out = run / "per_battle.json"
    data: dict[str, Any] = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    if data.get("battles") not in (None, battles):
        raise ValueError(f"{out} was written with battles={data.get('battles')}, not {battles}")
    data["battles"] = battles
    data.setdefault("methods", {})
    with WorkerPool(workers) as pool:
        for method in methods:
            for checkpoint in sorted((run / method).glob("seed*/final.pt")):
                seed = checkpoint.parent.name
                if seed in data["methods"].get(method, {}):
                    continue
                spec = agent_spec(checkpoint, greedy=True)
                per_split: dict[str, Any] = {}
                for split in splits:
                    per_split[split] = {}
                    for opponent in opponents:
                        records = play(pool, spec, {"heuristic": opponent}, split, battles)
                        ordered = sorted(records, key=lambda record: int(record["index"]))
                        per_split[split][opponent] = [(float(record["outcome"]) + 1) / 2 for record in ordered]
                data["methods"].setdefault(method, {})[seed] = per_split
                write_json(out, data)
                print(f"[per-battle] {method} {seed}", flush=True)
        if "baselines" not in data:
            baselines: dict[str, Any] = {}
            for row in HEURISTICS:
                baselines[row] = {}
                for split in splits:
                    baselines[row][split] = {}
                    for opponent in opponents:
                        records = sorted(play(pool, {"heuristic": row}, {"heuristic": opponent}, split, battles),
                                         key=lambda record: int(record["index"]))
                        baselines[row][split][opponent] = [(float(r["outcome"]) + 1) / 2 for r in records]
            data["baselines"] = baselines
            write_json(out, data)
    return out


# ----------------------------------------------------------------------------- statistics

def pair_scores(per_seed: list[list[float]]) -> Array:
    """Seed-averaged battle scores folded into team-pair means (battles 2p and 2p+1)."""
    matrix = np.asarray(per_seed, dtype=np.float64)
    battle = matrix.mean(axis=0)
    usable = battle.shape[0] - battle.shape[0] % 2
    return np.asarray(battle[:usable].reshape(-1, 2).mean(axis=1), dtype=np.float64)


def sign_flip_p(differences: Array, draws: int = DRAWS, seed: int = 0) -> float:
    """Two-sided sign-flip permutation p-value of mean(differences) = 0 (with the +1 correction)."""
    values = np.asarray(differences, dtype=np.float64)
    if values.size == 0 or np.allclose(values, 0.0):
        return 1.0
    observed = abs(float(values.mean()))
    generator = np.random.default_rng(seed)
    signs = generator.choice(np.array([-1.0, 1.0]), size=(draws, values.size))
    null = np.abs((signs * values).mean(axis=1))
    return float((1 + int(np.sum(null >= observed - 1e-12))) / (draws + 1))


def label_permutation_p(first: Array, second: Array, draws: int = DRAWS, seed: int = 0) -> float:
    """Two-sided permutation p-value of mean(first) - mean(second) = 0 for independent samples."""
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    pooled = np.concatenate([a, b])
    observed = abs(float(a.mean() - b.mean()))
    generator = np.random.default_rng(seed)
    count = 0
    for _ in range(draws):
        generator.shuffle(pooled)
        count += int(abs(float(pooled[:a.size].mean() - pooled[a.size:].mean())) >= observed - 1e-12)
    return float((1 + count) / (draws + 1))


def bootstrap_mean_ci(values: Array, draws: int = 10_000, seed: int = 0) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size < 2:
        value = float(data.mean()) if data.size else 0.0
        return value, value
    generator = np.random.default_rng(seed)
    means = data[generator.integers(0, data.size, size=(draws, data.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_diff_ci(first: Array, second: Array, draws: int = 10_000, seed: int = 0) -> tuple[float, float]:
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    generator = np.random.default_rng(seed)
    diffs = (a[generator.integers(0, a.size, size=(draws, a.size))].mean(axis=1)
             - b[generator.integers(0, b.size, size=(draws, b.size))].mean(axis=1))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def seed_t_ci(seed_means: list[float]) -> tuple[float, float, float]:
    """(mean, low, high): t-based 95% CI of the mean of per-seed scores (df = n - 1)."""
    values = np.asarray(seed_means, dtype=np.float64)
    if values.size < 2:
        value = float(values.mean()) if values.size else float("nan")
        return value, value, value
    from scipy.stats import t as student_t

    half = float(student_t.ppf(0.975, values.size - 1)) * float(values.std(ddof=1)) / math.sqrt(values.size)
    return float(values.mean()), float(values.mean()) - half, float(values.mean()) + half


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda index: pvalues[index])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def _per_seed(data: dict[str, Any], method: str, split: str, opponent: str) -> list[list[float]]:
    seeds = data["methods"].get(method, {})
    return [seeds[seed][split][opponent] for seed in sorted(seeds) if opponent in seeds[seed].get(split, {})]


def analyze(data: dict[str, Any]) -> dict[str, Any]:
    methods = [method for method in METHODS if method in data["methods"]]
    splits: list[str] = [split for split in EVAL_SPLITS if any(split in seed for m in methods
                                                    for seed in data["methods"][m].values())]
    scores: dict[str, Any] = {}
    for method in methods:
        scores[method] = {}
        for split in splits:
            scores[method][split] = {}
            for opponent in HEURISTICS:
                rows = _per_seed(data, method, split, opponent)
                if not rows:
                    continue
                pairs = pair_scores(rows)
                low, high = bootstrap_mean_ci(pairs)
                seed_mean, seed_low, seed_high = seed_t_ci([float(np.mean(row)) for row in rows])
                scores[method][split][opponent] = {
                    "mean": float(pairs.mean()), "pair_ci": [low, high], "seeds": len(rows), "pairs": int(pairs.size),
                    "seed_means": [float(np.mean(row)) for row in rows], "seed_t_ci": [seed_low, seed_high],
                    "seed_mean": seed_mean, "p_parity": sign_flip_p(pairs - 0.5)}
    baselines: dict[str, Any] = {}
    for row, by_split in data.get("baselines", {}).items():
        baselines[row] = {split: {opp: {"mean": float(np.mean(values)),
                                        "pair_ci": list(bootstrap_mean_ci(pair_scores([values])))}
                                  for opp, values in by_opp.items()} for split, by_opp in by_split.items()}

    def contrast(first: str, second: str, split: str, opponent: str) -> dict[str, Any] | None:
        a, b = _per_seed(data, first, split, opponent), _per_seed(data, second, split, opponent)
        if not a or not b:
            return None
        diff = pair_scores(a) - pair_scores(b)
        low, high = bootstrap_mean_ci(diff)
        return {"a": first, "b": second, "split": split, "opponent": opponent, "diff": float(diff.mean()),
                "ci": [low, high], "p": sign_flip_p(diff)}

    family: list[dict[str, Any]] = []
    for hypothesis, first, second in FAMILY:
        for split in PRIMARY_SPLITS:
            if second is None:
                entry = scores.get(first, {}).get(split, {}).get(PRIMARY_OPPONENT)
                if entry:
                    family.append({"id": hypothesis, "a": first, "b": "parity 0.5", "split": split,
                                   "opponent": PRIMARY_OPPONENT, "diff": entry["mean"] - 0.5,
                                   "ci": [entry["pair_ci"][0] - 0.5, entry["pair_ci"][1] - 0.5],
                                   "p": entry["p_parity"]})
            else:
                test = contrast(first, second, split, PRIMARY_OPPONENT)
                if test:
                    family.append({"id": hypothesis, **test})
    for test, adjusted in zip(family, holm([test["p"] for test in family])):
        test["p_holm"] = adjusted

    exploratory: list[dict[str, Any]] = []
    for first, second in ((a, b) for a in methods for b in methods if a < b):
        for split in splits:
            for opponent in ("greedy", "prior"):
                test = contrast(first, second, split, opponent)
                if test:
                    exploratory.append(test)

    gaps: list[dict[str, Any]] = []
    for method in methods:
        for opponent in ("greedy", "prior"):
            base_rows = _per_seed(data, method, "gen1_train", opponent)
            if not base_rows:
                continue
            base = pair_scores(base_rows)
            for split in splits:
                if split == "gen1_train":
                    continue
                rows = _per_seed(data, method, split, opponent)
                if not rows:
                    continue
                other = pair_scores(rows)
                low, high = bootstrap_diff_ci(base, other)
                gaps.append({"method": method, "opponent": opponent, "split": split,
                             "gap": float(base.mean() - other.mean()), "ci": [low, high],
                             "p": label_permutation_p(base, other, draws=5_000)})
    return {"battles": data.get("battles"), "methods": methods, "splits": splits, "scores": scores,
            "baselines": baselines, "family": family, "exploratory": exploratory, "gaps": gaps}


def _p(value: float) -> str:
    return "<1e-4" if value < 1e-4 else f"{value:.4f}"


def markdown(result: dict[str, Any]) -> str:
    lines = ["# E1 — análisis pareado a nivel de combate", "",
             "Unidad: par de equipos (2 combates con asientos intercambiados), puntaje promediado entre semillas. "
             "Tests de permutación por cambio de signo (20 000 sorteos), IC 95% bootstrap sobre pares. "
             "Familia pre-registrada (Holm): oponente Greedy, splits gen2/gen3.", "",
             "## Familia pre-registrada", "", "| H | A | B | split | diff | IC 95% | p | p Holm |",
             "|---|---|---|---|---|---|---|---|"]
    for test in result["family"]:
        lines.append(f"| {test['id']} | {test['a']} | {test['b']} | {test['split']} | {test['diff']:+.3f} | "
                     f"[{test['ci'][0]:+.3f}, {test['ci'][1]:+.3f}] | {_p(test['p'])} | {_p(test['p_holm'])} |")
    for opponent in ("greedy", "prior", "random"):
        lines += ["", f"## Puntaje contra {opponent} (media de pares [IC 95% pares]; semillas: IC t 95%)", "",
                  "| Método | " + " | ".join(result["splits"]) + " |", "|---|" + "---|" * len(result["splits"])]
        for method in result["methods"]:
            cells = []
            for split in result["splits"]:
                entry = result["scores"][method].get(split, {}).get(opponent)
                cells.append("-" if not entry else
                             f"{entry['mean']:.3f} [{entry['pair_ci'][0]:.3f}, {entry['pair_ci'][1]:.3f}]; "
                             f"seeds [{entry['seed_t_ci'][0]:.3f}, {entry['seed_t_ci'][1]:.3f}]")
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
        for row, by_split in result["baselines"].items():
            cells = [f"{by_split[s][opponent]['mean']:.3f}" if opponent in by_split.get(s, {}) else "-"
                     for s in result["splits"]]
            lines.append(f"| {row} (heurística) | " + " | ".join(cells) + " |")
    lines += ["", "## Brecha de generalización (gen1_train - split)", "",
              "| Método | oponente | split | brecha | IC 95% | p |", "|---|---|---|---|---|---|"]
    for gap in result["gaps"]:
        lines.append(f"| {gap['method']} | {gap['opponent']} | {gap['split']} | {gap['gap']:+.3f} | "
                     f"[{gap['ci'][0]:+.3f}, {gap['ci'][1]:+.3f}] | {_p(gap['p'])} |")
    lines += ["", "## Contrastes exploratorios (p sin ajustar)", "", "| A | B | split | oponente | diff | IC 95% | p |",
              "|---|---|---|---|---|---|---|"]
    for test in result["exploratory"]:
        lines.append(f"| {test['a']} | {test['b']} | {test['split']} | {test['opponent']} | {test['diff']:+.3f} | "
                     f"[{test['ci'][0]:+.3f}, {test['ci'][1]:+.3f}] | {_p(test['p'])} |")
    return "\n".join(lines) + "\n"


def summarize(run: Path) -> Path:
    data = json.loads((run / "per_battle.json").read_text(encoding="utf-8"))
    result = analyze(data)
    write_json(run / "paired.json", result)
    path = run / "paired.md"
    path.write_text(markdown(result), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate", help="per-battle outcomes of every final checkpoint of a run")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--battles", type=int, default=500)
    evaluate.add_argument("--workers", type=int, default=12)
    evaluate.add_argument("--methods", nargs="+", default=list(METHODS))
    summary = sub.add_parser("summarize", help="paired statistics -> paired.json / paired.md")
    summary.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        print(evaluate_run(args.run, args.battles, args.workers, tuple(args.methods)))
    print(summarize(args.run))


if __name__ == "__main__":
    main()
