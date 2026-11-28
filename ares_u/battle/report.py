"""Aggregation of an E1 protocol run: tables, statistics and figures.

Expected layout (written by ``scripts/ares_u_battle.py protocol``)::

    <run>/baselines.json
    <run>/<method>/seed<k>/{config.json, metrics.jsonl, final.pt, eval.json}
    <run>/bc_ppo/seed<k>/bc/...            (only when the BC phase was not shared with <run>/bc)

Outputs ``summary.json``, ``summary.md`` and ``figures/*.png``.

Statistics. Across seeds: mean, standard deviation and a bootstrap 95% CI of the mean (10 000
resamples of seeds); the battles of all seeds are also pooled into a Wilson interval. Methods are
compared with two-sided Mann-Whitney U tests on per-seed scores. With 5 vs 5 seeds the smallest
attainable p is 2/252 = 0.0079, so a Holm correction over dozens of comparisons could never reject;
the *primary* analysis therefore uses four planned contrasts, Holm-corrected within each
(split, opponent) family: bc_ppo vs ppo (does imitation warm-start help?), ppo vs ppo_ids (do
species IDs hurt transfer?), ppo vs dqn (Transformer-PPO vs the course baseline) and bc_ppo vs bc
(does RL improve on imitation?). All pairs are also stored (exploratory, Holm within family), with
the common-language effect size A = U / (n_a n_b). Each method is also compared with the 0.5
parity line (one-sample t-test over seeds, Holm across splits). The generalisation gap of a seed is
``score(gen1_train) - score(split)`` against the same opponent.
"""
from __future__ import annotations

import itertools
import json
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .agents import HEURISTICS
from .evaluate import wilson
from .teams import EVAL_SPLITS

METHODS = ("bc", "ppo", "bc_ppo", "ppo_ids", "dqn")
PLANNED = (("bc_ppo", "ppo"), ("ppo", "ppo_ids"), ("ppo", "dqn"), ("bc_ppo", "bc"))
TEST_OPPONENTS = ("greedy", "prior")
GAP_TARGETS = ("gen1_heldout", "gen2", "gen3")
LABELS = {"bc": "BC (imitación prior+Greedy)", "ppo": "PPO-Transformer (sin IDs)", "bc_ppo": "BC → PPO (sin IDs)",
          "ppo_ids": "PPO-Transformer + IDs", "dqn": "DQN-MLP", "random": "Random", "greedy": "Greedy",
          "prior": "ARES prior"}
SPLIT_LABELS = {"gen1_train": "Gen1 train-dist.", "gen1_heldout": "Gen1 hold-out", "gen2": "Gen2 (zero-shot)",
                "gen3": "Gen3 (zero-shot)"}
COLORS = {"bc": "#a855f7", "ppo": "#2563eb", "bc_ppo": "#dc2626", "ppo_ids": "#d97706", "dqn": "#059669",
          "random": "#d1d5db", "greedy": "#9ca3af", "prior": "#4b5563"}


def bootstrap_ci(values: list[float], resamples: int = 10_000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    generator = np.random.default_rng(seed)
    data = np.asarray(values, dtype=np.float64)
    means = data[generator.integers(0, len(data), size=(resamples, len(data)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    order = sorted(range(len(pvalues)), key=lambda index: pvalues[index])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def mann_whitney(first: list[float], second: list[float]) -> tuple[float, float]:
    """(two-sided p, common-language effect size A = P(first > second) + P(tie)/2)."""
    from scipy.stats import mannwhitneyu

    if len(first) < 2 or len(second) < 2:
        return 1.0, 0.5
    if np.allclose(first + second, first[0]):
        return 1.0, 0.5
    result = mannwhitneyu(first, second, alternative="two-sided")
    return float(result.pvalue), float(result.statistic) / (len(first) * len(second))


def one_sample_t(values: list[float], reference: float) -> float:
    from scipy.stats import ttest_1samp

    if len(values) < 2 or float(np.std(values)) < 1e-12:
        return float("nan")
    return float(ttest_1samp(values, reference).pvalue)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _budget(method: str, seed_dir: Path, config: dict[str, Any], metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Environment steps (demonstrations + RL) and wall-clock seconds of one seed."""
    last_wall = max((float(r.get("wall", 0.0)) for r in metrics), default=0.0)
    steps = max((int(r.get("steps", 0)) for r in metrics), default=0)
    if method == "bc":
        demos = next((int(r["steps"]) for r in metrics if r.get("kind") == "demos"), steps)
        return {"steps": demos, "demo_steps": demos, "rl_steps": 0, "wall": last_wall, "bc_shared": False}
    demos = int(config.get("step_offset", 0))
    bc_wall = 0.0
    shared = False
    if method == "bc_ppo":
        own = seed_dir / "bc" / "metrics.jsonl"
        if own.is_file():
            bc_wall = max((float(r.get("wall", 0.0)) for r in _read_jsonl(own)), default=0.0)
        else:
            shared = True
    return {"steps": steps, "demo_steps": demos, "rl_steps": max(0, steps - demos), "wall": last_wall + bc_wall,
            "bc_shared": shared}


def collect(run: Path) -> dict[str, Any]:
    """Per method: per-seed evaluation results, metrics and run metadata."""
    methods: dict[str, Any] = {}
    for method in METHODS:
        seeds: dict[str, Any] = {}
        for seed_dir in sorted((run / method).glob("seed*")):
            evaluation = _read_json(seed_dir / "eval.json")
            if not evaluation:
                continue
            config = _read_json(seed_dir / "config.json")
            metrics = _read_jsonl(seed_dir / "metrics.jsonl")
            train = [record for record in metrics if record.get("kind") == "train"]
            seeds[seed_dir.name] = {"eval": evaluation, "config": config, "metrics": metrics,
                                    "budget": _budget(method, seed_dir, config, metrics),
                                    "learn_s": float(train[-1].get("learn_s", 0.0)) if train else 0.0,
                                    "rollout_s": float(train[-1].get("rollout_s", 0.0)) if train else 0.0}
        if seeds:
            methods[method] = seeds
    baselines = _read_json(run / "baselines.json") or None
    return {"methods": methods, "baselines": baselines}


def _splits_in(data: dict[str, Any]) -> tuple[str, ...]:
    present = {split for seeds in data["methods"].values() for seed in seeds.values()
               for split in seed["eval"].get("results", {})}
    return tuple(split for split in EVAL_SPLITS if split in present) or EVAL_SPLITS


def _table(data: dict[str, Any], splits: tuple[str, ...]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for method, seeds in data["methods"].items():
        table[method] = {}
        for split in splits:
            table[method][split] = {}
            for opponent in HEURISTICS:
                entries = [seed["eval"]["results"][split][opponent] for seed in seeds.values()
                           if opponent in seed["eval"]["results"].get(split, {})]
                if not entries:
                    continue
                scores = [float(entry["score"]) for entry in entries]
                low, high = bootstrap_ci(scores)
                pooled_n = sum(int(entry["n"]) for entry in entries)
                pooled_success = sum(float(entry["wins"]) + 0.5 * float(entry["draws"]) for entry in entries)
                _, wilson_low, wilson_high = wilson(pooled_success, pooled_n)
                table[method][split][opponent] = {
                    "scores": scores, "mean": float(np.mean(scores)),
                    "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0, "boot_low": low,
                    "boot_high": high, "pooled_wilson": [wilson_low, wilson_high], "pooled_n": pooled_n,
                    **{key: float(np.mean([float(entry[key]) for entry in entries]))
                       for key in ("mean_turns", "switch_rate", "panic_switch_rate", "ineffective_rate")}}
    return table


def _contrast(table: dict[str, Any], first: str, second: str, split: str, opponent: str) -> dict[str, Any] | None:
    a = table.get(first, {}).get(split, {}).get(opponent)
    b = table.get(second, {}).get(split, {}).get(opponent)
    if not a or not b:
        return None
    p, effect = mann_whitney(a["scores"], b["scores"])
    return {"a": first, "b": second, "split": split, "opponent": opponent, "mean_a": a["mean"], "mean_b": b["mean"],
            "diff": a["mean"] - b["mean"], "A": effect, "n_a": len(a["scores"]), "n_b": len(b["scores"]), "p": p}


def _tests(table: dict[str, Any], splits: tuple[str, ...], pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Mann-Whitney contrasts, Holm-corrected within each (split, opponent) family."""
    tests: list[dict[str, Any]] = []
    for split in splits:
        for opponent in TEST_OPPONENTS:
            family = [test for test in (_contrast(table, a, b, split, opponent) for a, b in pairs) if test]
            for test, adjusted in zip(family, holm([test["p"] for test in family])):
                test["p_holm"] = adjusted
                test["family"] = f"{split}/{opponent}"
            tests.extend(family)
    return tests


def _parity(table: dict[str, Any], splits: tuple[str, ...]) -> list[dict[str, Any]]:
    """One-sample t-test of the seed scores against 0.5 (parity with the opponent), Holm over splits."""
    rows: list[dict[str, Any]] = []
    for method, by_split in table.items():
        for opponent in TEST_OPPONENTS:
            family = []
            for split in splits:
                entry = by_split.get(split, {}).get(opponent)
                if entry:
                    family.append({"method": method, "opponent": opponent, "split": split, "mean": entry["mean"],
                                   "p": one_sample_t(entry["scores"], 0.5)})
            valid = [row for row in family if not np.isnan(row["p"])]
            for row, adjusted in zip(valid, holm([row["p"] for row in valid])):
                row["p_holm"] = adjusted
            for row in family:
                row.setdefault("p_holm", float("nan"))
            rows.extend(family)
    return rows


def _gaps(table: dict[str, Any], baselines: dict[str, Any] | None, splits: tuple[str, ...]) -> dict[str, Any]:
    gaps: dict[str, Any] = {}
    if "gen1_train" not in splits:
        return gaps
    for method, by_split in table.items():
        gaps[method] = {}
        for opponent in TEST_OPPONENTS:
            source = by_split.get("gen1_train", {}).get(opponent)
            if not source:
                continue
            gaps[method][opponent] = {}
            for target in GAP_TARGETS:
                goal = by_split.get(target, {}).get(opponent)
                if not goal:
                    continue
                values = [a - b for a, b in zip(source["scores"], goal["scores"])]
                low, high = bootstrap_ci(values)
                gaps[method][opponent][target] = {"values": values, "mean": float(np.mean(values)),
                                                  "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                                                  "boot_low": low, "boot_high": high}
    if baselines:
        matrix = baselines.get("matrix", {})
        for row in HEURISTICS:
            gaps[row] = {}
            for opponent in TEST_OPPONENTS:
                if row == opponent or opponent not in matrix.get("gen1_train", {}).get(row, {}):
                    continue
                base = float(matrix["gen1_train"][row][opponent]["score"])
                gaps[row][opponent] = {target: {"values": [base - float(matrix[target][row][opponent]["score"])],
                                                "mean": base - float(matrix[target][row][opponent]["score"]),
                                                "std": 0.0, "boot_low": float("nan"), "boot_high": float("nan")}
                                       for target in GAP_TARGETS if opponent in matrix.get(target, {}).get(row, {})}
    return gaps


def aggregate(data: dict[str, Any]) -> dict[str, Any]:
    splits = _splits_in(data)
    table = _table(data, splits)
    present = [method for method in METHODS if method in table]
    walls = {method: {"wall_mean_s": float(np.mean([s["budget"]["wall"] for s in seeds.values()])),
                      "steps_mean": float(np.mean([s["budget"]["steps"] for s in seeds.values()])),
                      "demo_steps_mean": float(np.mean([s["budget"]["demo_steps"] for s in seeds.values()])),
                      "rl_steps_mean": float(np.mean([s["budget"]["rl_steps"] for s in seeds.values()])),
                      "learn_s_mean": float(np.mean([s["learn_s"] for s in seeds.values()])),
                      "rollout_wait_s_mean": float(np.mean([s["rollout_s"] for s in seeds.values()])),
                      "bc_shared": any(s["budget"]["bc_shared"] for s in seeds.values()),
                      "seeds": len(seeds),
                      "device": sorted({str(s["config"].get("device_used", "?")) for s in seeds.values()})}
             for method, seeds in data["methods"].items()}
    return {"splits": list(splits), "table": table,
            "tests": _tests(table, splits, [pair for pair in PLANNED if pair[0] in table and pair[1] in table]),
            "tests_all": _tests(table, splits, list(itertools.combinations(present, 2))),
            "parity": _parity(table, splits), "gaps": _gaps(table, data["baselines"], splits), "walls": walls,
            "baselines": data["baselines"],
            "budget_note": ("Env steps = recorded decisions (training samples). Every RL method gets the same budget; "
                            "bc_ppo spends its demonstration decisions out of that budget (demo + PPO = budget); "
                            "bc uses only its demonstrations (no RL)."),
            "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                         "python": platform.python_version()}}


def _fmt(entry: dict[str, Any] | None) -> str:
    if not entry:
        return "-"
    return f"{entry['mean']:.3f} ± {entry['std']:.3f} [{entry['boot_low']:.3f}, {entry['boot_high']:.3f}]"


def _p(value: float) -> str:
    return "-" if np.isnan(value) else f"{value:.4f}"


def markdown(summary: dict[str, Any]) -> str:
    splits: list[str] = summary["splits"]
    lines = ["# ARES-Battle (E1): transferencia entre generaciones", "",
             "Score = (victorias + empates/2) / N, con lados y equipos alternados. Métodos: media ± desv. estándar "
             "entre semillas [IC 95% bootstrap de la media]. Líneas base: score [IC 95% Wilson]. "
             "`gen1_train` = equipos nuevos de la distribución de entrenamiento; `gen1_heldout` = conjuntos de especies "
             "Gen1 nunca usados en entrenamiento; `gen2`/`gen3` = especies nunca vistas (zero-shot).", "",
             f"Presupuesto: {summary['budget_note']}", ""]
    baselines = summary.get("baselines")
    for opponent in HEURISTICS:
        lines += [f"## Contra {LABELS[opponent]}", "", "| Agente | " + " | ".join(splits) + " |",
                  "|---|" + "---|" * len(splits)]
        for method, by_split in summary["table"].items():
            cells = [_fmt(by_split.get(split, {}).get(opponent)) for split in splits]
            lines.append(f"| {LABELS.get(method, method)} | " + " | ".join(cells) + " |")
        if baselines:
            for row in HEURISTICS:
                cells = []
                for split in splits:
                    entry = baselines["matrix"].get(split, {}).get(row, {}).get(opponent)
                    cells.append(f"{entry['score']:.3f} [{entry['ci_low']:.3f}, {entry['ci_high']:.3f}]"
                                 if entry else "-")
                lines.append(f"| {LABELS[row]} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Métricas anti-vicio (contra Greedy, media entre semillas)", "",
              "| Agente | split | turnos | switch | panic switch | mov. inútil |", "|---|---|---|---|---|---|"]
    for method, by_split in summary["table"].items():
        for split in splits:
            entry = by_split.get(split, {}).get("greedy")
            if entry:
                lines.append(f"| {LABELS.get(method, method)} | {split} | {entry['mean_turns']:.1f} | "
                             f"{entry['switch_rate']:.3f} | {entry['panic_switch_rate']:.3f} | "
                             f"{entry['ineffective_rate']:.3f} |")
    if baselines:
        for row in HEURISTICS:
            for split in splits:
                entry = baselines["matrix"].get(split, {}).get(row, {}).get("greedy")
                if entry:
                    lines.append(f"| {LABELS[row]} | {split} | {entry['mean_turns']:.1f} | {entry['switch_rate']:.3f} | "
                                 f"{entry['panic_switch_rate']:.3f} | {entry['ineffective_rate']:.3f} |")
    lines += ["", "## Brecha de generalización (score(gen1_train) − score(split), por semilla)", "",
              "| Agente | rival | " + " | ".join(GAP_TARGETS) + " |", "|---|---|" + "---|" * len(GAP_TARGETS)]
    for method, by_opponent in summary["gaps"].items():
        for opponent, by_target in by_opponent.items():
            cells = [f"{by_target[t]['mean']:+.3f} ± {by_target[t]['std']:.3f}" if t in by_target else "-"
                     for t in GAP_TARGETS]
            lines.append(f"| {LABELS.get(method, method)} | {opponent} | " + " | ".join(cells) + " |")
    lines += ["", "## Contrastes planificados (Mann-Whitney U bilateral sobre scores por semilla; Holm dentro de "
              "cada familia split/rival)", "",
              "| A | B | split | rival | media A | media B | A−B | A (efecto) | p | p Holm |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for test in summary["tests"]:
        lines.append(f"| {test['a']} | {test['b']} | {test['split']} | {test['opponent']} | {test['mean_a']:.3f} | "
                     f"{test['mean_b']:.3f} | {test['diff']:+.3f} | {test['A']:.2f} | {test['p']:.4f} | "
                     f"{test['p_holm']:.4f} |")
    lines += ["", "Con 5 vs 5 semillas el menor p alcanzable es 0.0079; todas las comparaciones por pares "
              "(exploratorias) están en `summary.json` → `tests_all`.", "",
              "## Paridad con el rival (t de una muestra sobre semillas vs 0.5; Holm entre splits)", "",
              "| Agente | rival | " + " | ".join(splits) + " |", "|---|---|" + "---|" * len(splits)]
    parity: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summary["parity"]:
        parity[(row["method"], row["opponent"])][row["split"]] = row
    for (method, opponent), by_split in parity.items():
        cells = [f"{by_split[s]['mean']:.3f} (p={_p(by_split[s]['p_holm'])})" if s in by_split else "-" for s in splits]
        lines.append(f"| {LABELS.get(method, method)} | {opponent} | " + " | ".join(cells) + " |")
    lines += ["", "## Presupuesto, tiempo de pared y hardware", "",
              "| Método | semillas | pasos totales | demos | pasos RL | pared media (s) | learner (s) | espera rollout (s) "
              "| device |", "|---|---|---|---|---|---|---|---|---|"]
    for method, wall in summary["walls"].items():
        shared = " (BC compartido con `bc`)" if wall["bc_shared"] else ""
        lines.append(f"| {LABELS.get(method, method)}{shared} | {wall['seeds']} | {wall['steps_mean']:.0f} | "
                     f"{wall['demo_steps_mean']:.0f} | {wall['rl_steps_mean']:.0f} | {wall['wall_mean_s']:.0f} | "
                     f"{wall['learn_s_mean']:.0f} | {wall['rollout_wait_s_mean']:.0f} | {', '.join(wall['device'])} |")
    hardware = summary["hardware"]
    lines += ["", f"Plataforma: {hardware['platform']}; CPU: {hardware['processor']}; Python {hardware['python']}.", ""]
    return "\n".join(lines)


def _curve(seeds: dict[str, Any], key: str, kind: str, grid: NDArray[np.float64]) -> NDArray[np.float64] | None:
    rows = []
    for seed in seeds.values():
        points = sorted((float(r["steps"]), float(r[key])) for r in seed["metrics"]
                        if r.get("kind") == kind and r.get(key) is not None)
        if len(points) < 2:
            continue
        xs, ys = zip(*points)
        if kind == "train":
            ys = tuple(np.convolve(ys, np.ones(5) / 5, mode="same")) if len(ys) >= 5 else ys
        rows.append(np.interp(grid, xs, ys, left=np.nan, right=np.nan))
    return np.vstack(rows) if rows else None


def _point(seeds: dict[str, Any], key: str) -> tuple[float, float] | None:
    """(steps, mean) of a method evaluated once (bc): drawn as a marker + dotted level."""
    values = [(float(r["steps"]), float(r[key])) for seed in seeds.values() for r in seed["metrics"]
              if r.get("kind") == "eval" and r.get(key) is not None]
    if not values:
        return None
    return float(np.mean([v[0] for v in values])), float(np.mean([v[1] for v in values]))


def figures(data: dict[str, Any], summary: dict[str, Any], out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    splits: list[str] = summary["splits"]
    max_steps = max((seed["budget"]["steps"] for seeds in data["methods"].values() for seed in seeds.values()),
                    default=0)
    if max_steps:
        grid = np.linspace(0, max_steps, 80)
        panels = (("eval_greedy_gen1_train", "eval", "vs Greedy, Gen1 (validación)"),
                  ("eval_greedy_gen3", "eval", "vs Greedy, Gen3 (validación zero-shot)"),
                  ("eval_prior_gen1_train", "eval", "vs ARES prior, Gen1 (validación)"),
                  ("eval_prior_gen3", "eval", "vs ARES prior, Gen3 (validación zero-shot)"),
                  ("train_win_greedy", "train", "entrenamiento vs Greedy (Gen1, muestreo)"),
                  ("train_win_self", "train", "entrenamiento vs sí mismo"))
        figure, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
        for axis, (key, kind, title) in zip(axes.ravel(), panels):
            for method, seeds in data["methods"].items():
                color = COLORS.get(method, "#111827")
                if method == "bc":
                    point = _point(seeds, key) if kind == "eval" else None
                    if point:
                        axis.axhline(point[1], color=color, linestyle=":", linewidth=1.2)
                        axis.plot([point[0]], [point[1]], marker="o", color=color, label=f"{LABELS[method]}")
                    continue
                curves = _curve(seeds, key, kind, grid)
                if curves is None:
                    continue
                mean = np.nanmean(curves, axis=0)
                std = np.nanstd(curves, axis=0)
                axis.plot(grid, mean, color=color, label=f"{LABELS.get(method, method)} (n={curves.shape[0]})")
                axis.fill_between(grid, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
            axis.axhline(0.5, color="#6b7280", linestyle="--", linewidth=0.8)
            axis.set_title(title, fontsize=10)
            axis.set_xlabel("pasos de entorno (incl. demostraciones)")
            axis.grid(alpha=0.3)
        axes[0, 0].set_ylabel("score (media ± desv. est. entre semillas)")
        axes[1, 0].set_ylabel("score")
        axes[0, 0].legend(fontsize=7, loc="lower right")
        figure.tight_layout()
        path = out / "learning_curves.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        written.append(path)
    baselines = summary.get("baselines")
    for opponent in HEURISTICS:
        figure, axis = plt.subplots(figsize=(10, 4))
        entries: list[tuple[str, list[float], list[float], list[float], list[list[float]]]] = []
        for method, by_split in summary["table"].items():
            means, lows, highs, points = [], [], [], []
            for split in splits:
                entry = by_split.get(split, {}).get(opponent)
                means.append(entry["mean"] if entry else np.nan)
                lows.append(entry["boot_low"] if entry else np.nan)
                highs.append(entry["boot_high"] if entry else np.nan)
                points.append(entry["scores"] if entry else [])
            entries.append((method, means, lows, highs, points))
        if baselines:
            for row in HEURISTICS:
                if row == opponent:
                    continue
                means, lows, highs = [], [], []
                for split in splits:
                    entry = baselines["matrix"].get(split, {}).get(row, {}).get(opponent)
                    means.append(entry["score"] if entry else np.nan)
                    lows.append(entry["ci_low"] if entry else np.nan)
                    highs.append(entry["ci_high"] if entry else np.nan)
                entries.append((row, means, lows, highs, [[] for _ in splits]))
        width = 0.84 / max(1, len(entries))
        positions = np.arange(len(splits))
        for offset, (name, means, lows, highs, points) in enumerate(entries):
            centre = positions + (offset - (len(entries) - 1) / 2) * width
            mean_array = np.asarray(means)
            errors = np.clip(np.vstack([mean_array - np.asarray(lows), np.asarray(highs) - mean_array]), 0.0, None)
            axis.bar(centre, mean_array, width, yerr=errors, capsize=2, color=COLORS.get(name, "#111827"),
                     label=LABELS.get(name, name), edgecolor="white", linewidth=0.4)
            for x, values in zip(centre, points):
                if values:
                    axis.scatter([x] * len(values), values, s=6, color="#111827", zorder=3)
        axis.axhline(0.5, color="#6b7280", linestyle="--", linewidth=0.8)
        axis.set_xticks(positions, [SPLIT_LABELS.get(split, split) for split in splits])
        axis.set_ylabel(f"score vs {LABELS[opponent]}")
        axis.set_ylim(0, 1)
        axis.legend(fontsize=7, ncol=4, loc="lower center")
        axis.grid(axis="y", alpha=0.3)
        figure.tight_layout()
        path = out / f"transfer_vs_{opponent}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        written.append(path)
    gaps = summary.get("gaps", {})
    if gaps:
        figure, axes_row = plt.subplots(1, len(TEST_OPPONENTS), figsize=(12, 3.8), sharey=True)
        for axis, opponent in zip(np.atleast_1d(axes_row), TEST_OPPONENTS):
            names = [name for name in gaps if opponent in gaps[name]]
            width = 0.84 / max(1, len(names))
            positions = np.arange(len(GAP_TARGETS))
            for offset, name in enumerate(names):
                by_target = gaps[name][opponent]
                centre = positions + (offset - (len(names) - 1) / 2) * width
                gap_means = np.asarray([by_target[t]["mean"] if t in by_target else np.nan for t in GAP_TARGETS])
                gap_stds = np.asarray([by_target[t]["std"] if t in by_target else 0.0 for t in GAP_TARGETS])
                axis.bar(centre, gap_means, width, yerr=gap_stds, capsize=2, color=COLORS.get(name, "#111827"),
                         label=LABELS.get(name, name), edgecolor="white", linewidth=0.4)
                for x, target in zip(centre, GAP_TARGETS):
                    if target in by_target and len(by_target[target]["values"]) > 1:
                        values = by_target[target]["values"]
                        axis.scatter([x] * len(values), values, s=6, color="#111827", zorder=3)
            axis.axhline(0.0, color="#111827", linewidth=0.8)
            axis.set_xticks(positions, [SPLIT_LABELS[t] for t in GAP_TARGETS])
            axis.set_title(f"brecha vs {LABELS[opponent]}: score(Gen1 train) − score(split)", fontsize=10)
            axis.grid(axis="y", alpha=0.3)
        np.atleast_1d(axes_row)[0].set_ylabel("brecha de generalización (media ± d.e.)")
        np.atleast_1d(axes_row)[0].legend(fontsize=7, ncol=2)
        figure.tight_layout()
        path = out / "generalisation_gap.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        written.append(path)
    return written


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def summarize_run(run: Path) -> dict[str, Any]:
    data = collect(run)
    summary = aggregate(data)
    (run / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    (run / "summary.md").write_text(markdown(summary), encoding="utf-8")
    written = figures(data, summary, run / "figures")
    summary["figures"] = [str(path) for path in written]
    return summary


def per_method_counts(run: Path) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for method in METHODS:
        counts[method] = len(list((run / method).glob("seed*/eval.json")))
    return dict(counts)
