"""ARES-ADVERSARIAL: adversarial Pokemon agents that coevolve in the stadium.

Modos:
  league   (por defecto) liga en vivo ROJO (FireRed) vs AZUL (Emerald), sin fin
           salvo --generations; publica artifacts/stadium/<run>/live.json.
  compare  réplica del experimento del paper: Ranking / Pareto / Behavior /
           Random con varias réplicas, torneo inter-variante, seis métricas y
           Holm-Bonferroni; escribe report.json y report.md.
  ablation ARES-ADVERSARIAL completo (lecciones + aprendices profundos +
           intercambio) frente a GAME-Ranking puro, misma semilla y presupuesto;
           después se cruzan sus élites finales en un torneo.
  teams    re-lee los checkpoints de las campañas y muestra los equipos.

Ejemplos:
  python -m scripts.run_stadium --smoke
  python -m scripts.run_stadium --workers 14
  python -m scripts.run_stadium --resume
  python -m scripts.run_stadium --mode compare --replications 5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.stadium.dex import load_dex
from core.stadium.evaluation import Evaluator, SideSpec, Solution
from core.stadium.game_qd import VARIANTS, GameConfig, Variant, budget_matched, run_game
from core.stadium.league import GAMES, LABELS, STADIUM_ROOT, League, LeagueConfig, latest_run
from core.stadium.metrics import holm_bonferroni, side_report
from core.stadium.teams import load_bank


def _workers(requested: int | None) -> int:
    if requested is not None:
        return max(0, requested)
    return max(1, min(14, (os.cpu_count() or 4) - 2))


def league(args: argparse.Namespace) -> None:
    config = LeagueConfig(
        generations=args.generations if args.generations is not None else (4 if args.smoke else 0),
        variant=args.variant, n_task=args.tasks if args.tasks else (4 if args.smoke else 8),
        n_cell=args.cells if args.cells else (4 if args.smoke else 8),
        budget=args.budget if args.budget else (128 if args.smoke else 640), batch=64 if not args.smoke else 32,
        n_init=12 if args.smoke else 24, repeats=args.repeats, workers=_workers(args.workers),
        level=None if args.raw_levels else args.level, duel_battles=3 if args.smoke else 5,
        lessons=not args.no_lessons, lesson_points=8 if args.smoke else 16, rollouts=4 if args.smoke else 6,
        deep=not args.no_deep, ppo_updates=1 if args.smoke else 3, share_knowledge=not args.no_exchange,
        anchor_battles=8 if args.smoke else 24, seed=args.seed, refresh_teams=args.refresh_teams)
    run_dir = None
    if args.resume:
        run_dir = Path(args.run) if args.run else latest_run()
        if run_dir is None:
            raise SystemExit("No hay una liga previa que reanudar (artifacts/stadium/latest.json).")
    elif args.run:
        run_dir = STADIUM_ROOT / args.run
    arena = League(config, run_dir=run_dir, resume=args.resume, log=lambda text: print(text, flush=True))
    print(f"ARES-ADVERSARIAL en {arena.run_dir}  (Ctrl+C para detener; --resume continúa)", flush=True)
    try:
        arena.run()
    except KeyboardInterrupt:
        print("Detenido por el usuario; estado guardado.", flush=True)


def compare(args: argparse.Namespace) -> None:
    dex = load_dex()
    bank = load_bank(dex, refresh=args.refresh_teams)
    sides = {name: SideSpec(name, GAMES[name], bank.roster(GAMES[name]), bank.parties.get(GAMES[name], []),
                            None if args.raw_levels else args.level) for name in ("red", "blue")}
    requested = set(str(args.variants).split(","))
    variants: list[Variant] = [variant for variant in VARIANTS if variant in requested]
    base = GameConfig(n_gen=args.n_gen, n_task=args.tasks or 6, n_cell=args.cells or 6,
                      budget=args.budget or 384, batch=48, n_init=18)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = STADIUM_ROOT / f"compare-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    finals: dict[str, list[tuple[str, int, Solution]]] = {"red": [], "blue": []}
    workers = _workers(args.workers)
    print(f"Comparación GAME: variantes {variants}, {args.replications} réplicas, {workers} procesos -> {out}", flush=True)
    with Evaluator(sides, workers=workers, repeats=args.repeats, seed=args.seed) as evaluator:
        for replication in range(args.replications):
            for variant in variants:
                config = budget_matched(GameConfig(**{**base.__dict__, "variant": variant}))
                started = time.time()
                last = run_game(sides, evaluator, config, seed=args.seed * 1000 + replication)
                for name in ("red", "blue"):
                    finals[name] += [(variant, replication, solution) for solution in last.get(name, [])]
                print(f"  réplica {replication + 1} {variant:<8} {time.time() - started:6.1f}s "
                      f"({evaluator.battles} combates acumulados)", flush=True)
        red = [item[2] for item in finals["red"]]
        blue = [item[2] for item in finals["blue"]]
        print(f"Torneo inter-variante {len(red)} x {len(blue)} ...", flush=True)
        matrix, _ = evaluator.matrix(red, blue)
    title = f"# Comparación de variantes GAME en el Stadium ({time.strftime('%Y-%m-%d %H:%M')})"
    intro = (f"{args.replications} réplicas; N_gen={base.n_gen}, N_task={base.n_task}, N_cell={base.n_cell}, "
             f"presupuesto={base.budget} (igualado para Random/Behavior).")
    write_report(out, finals, list(variants), args.replications, matrix, base.n_task, args.seed, title, intro,
                 {"config": base.__dict__})


METRICS = ("win_rate", "elo", "robustness", "coverage", "expertise", "aqd")


def write_report(out: Path, finals: dict[str, list[tuple[str, int, Solution]]], variants: list[str],
                 replications: int, matrix: NDArray[np.float64], clusters: int, seed: int, title: str, intro: str,
                 extra: dict[str, Any], extra_lines: list[str] | None = None) -> dict[str, Any]:
    """Six metrics per variant (median [Q1, Q3] over replications) with Holm-Bonferroni."""
    report: dict[str, Any] = {**extra, "variants": variants, "replications": replications, "sides": {},
                              "significance": {}}
    for name, fitness in (("red", matrix), ("blue", 1 - matrix.T)):
        groups = {f"{variant}#{replication}": [index for index, item in enumerate(finals[name])
                                               if item[0] == variant and item[1] == replication]
                  for variant in variants for replication in range(replications)}
        metrics = side_report(fitness, groups, clusters, seed=seed)
        per_variant: dict[str, dict[str, list[float]]] = {}
        # AQD is infinite when some solution is never beaten (the best possible value):
        # for statistics it ranks above any finite counter-set size.
        never_beaten = float(fitness.shape[1] + 1)
        for key, value in metrics.items():
            group = key.split("#")[0]
            for metric, number in value.as_dict().items():
                per_variant.setdefault(group, {}).setdefault(metric, []).append(
                    never_beaten if number is None else float(number))
        report["sides"][name] = {variant: {metric: {"median": float(np.median(values)),
                                                    "q1": float(np.percentile(values, 25)),
                                                    "q3": float(np.percentile(values, 75)), "values": values,
                                                    "infinite": never_beaten if metric == "aqd" else None}
                                           for metric, values in by_metric.items()}
                                 for variant, by_metric in per_variant.items()}
        # Every metric is "higher is better" (a larger AQD needs more counter-solutions).
        report["significance"][name] = {
            metric: holm_bonferroni({variant: per_variant[variant][metric] for variant in per_variant})
            for metric in METRICS}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    lines = [title, "", intro + " Mediana [Q1, Q3] sobre réplicas; **negrita** = no distinguible del mejor "
             "(Mann-Whitney + Holm-Bonferroni, p=0.05).", ""] + (extra_lines or [])
    for name in ("red", "blue"):
        lines += [f"## {LABELS[name]}", "", "| métrica | " + " | ".join(variants) + " |",
                  "|---|" + "---|" * len(variants)]
        for metric in METRICS:
            cells = []
            for variant in variants:
                stats = report["sides"][name][variant][metric]

                def show(number: float, infinite: float | None = stats["infinite"]) -> str:
                    return "∞" if infinite is not None and number >= infinite else f"{number:.3f}"
                text = f"{show(stats['median'])} [{show(stats['q1'])}, {show(stats['q3'])}]"
                if report["significance"][name][metric].get(variant):
                    text = f"**{text}**"
                cells.append(text)
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    return report


def ablation(args: argparse.Namespace) -> None:
    """Does complementary learning help? Same seeds and GAME budget, with and without it."""
    generations = args.generations or 10
    root = STADIUM_ROOT / f"ablation-{time.strftime('%Y%m%d-%H%M%S')}"
    finals: dict[str, list[tuple[str, int, Solution]]] = {"red": [], "blue": []}
    variants = ["completo", "game"]
    base = LeagueConfig(generations=generations, variant=args.variant, n_task=args.tasks or 6,
                        n_cell=args.cells or 6, budget=args.budget or 384, batch=48, n_init=18, repeats=args.repeats,
                        workers=_workers(args.workers), level=None if args.raw_levels else args.level,
                        duel_battles=5)
    for replication in range(args.replications):
        for variant in variants:
            config = replace(base, seed=args.seed * 1000 + replication)
            if variant == "game":
                config = replace(config, lessons=False, deep=False, share_knowledge=False)
            started = time.time()
            league_run = League(config, run_dir=root / f"{variant}-{replication}", log=lambda _: None)
            league_run.run()
            for side in ("red", "blue"):
                finals[side] += [(variant, replication, solution) for solution in league_run.state[side].tasks]
            print(f"  réplica {replication + 1} {variant:<8} {time.time() - started:6.1f}s", flush=True)
    dex = load_dex()
    bank = load_bank(dex)
    sides = {name: SideSpec(name, GAMES[name], bank.roster(GAMES[name]), bank.parties.get(GAMES[name], []),
                            base.level) for name in ("red", "blue")}
    red = [item[2] for item in finals["red"]]
    blue = [item[2] for item in finals["blue"]]
    print(f"Torneo final {len(red)} x {len(blue)} ...", flush=True)
    with Evaluator(sides, workers=_workers(args.workers), repeats=max(4, args.repeats), seed=args.seed + 1) as evaluator:
        matrix, _ = evaluator.matrix(red, blue)
    # Common opponent pool: each variant's elites are scored against *all* final rival elites.
    advantage: dict[str, list[float]] = {"red": [], "blue": []}
    for replication in range(args.replications):
        for side, fitness in (("red", matrix), ("blue", 1 - matrix.T)):
            rows = {variant: [index for index, item in enumerate(finals[side])
                              if item[0] == variant and item[1] == replication] for variant in variants}
            advantage[side].append(float(fitness[rows["completo"]].mean() - fitness[rows["game"]].mean()))
    lines = ["| lado | ventaja del sistema completo (fitness medio vs todo el pool rival) | réplicas > 0 |", "|---|---|---|"]
    for side in ("red", "blue"):
        values = advantage[side]
        lines.append(f"| {LABELS[side]} | {np.mean(values):+.3f} (mediana {np.median(values):+.3f}) | "
                     f"{sum(value > 0 for value in values)}/{len(values)} |")
    lines.append("")
    title = f"# Ablación ARES-ADVERSARIAL ({time.strftime('%Y-%m-%d %H:%M')})"
    intro = (f"{args.replications} réplicas x {generations} generaciones; «completo» = GAME-{args.variant} + lecciones "
             "contrafactuales + aprendices profundos + intercambio de conocimiento; «game» = solo GAME-"
             f"{args.variant}, mismas semillas y presupuesto. Torneo final con {max(4, args.repeats)} combates por par.")
    write_report(root, finals, variants, args.replications, matrix, base.n_task, args.seed, title, intro,
                 {"config": base.__dict__, "advantage": advantage}, lines)


def teams(args: argparse.Namespace) -> None:
    dex = load_dex()
    bank = load_bank(dex, refresh=True)
    for name, game in GAMES.items():
        print(f"{LABELS[name]}: {len(bank.roster(game))} Pokémon criados en {bank.sources.get(game, 0)} checkpoints")
        for mon in bank.roster(game):
            spec = mon.spec(dex, args.level, mon.best_moves(dex, args.level))
            print(f"  {mon.name:<11} nv{mon.level:<3} -> {spec.name:<11} nv{spec.level} "
                  f"{[dex.moves[move].name for move in spec.moves]}  ({mon.origin})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("league", "compare", "ablation", "teams"), default="league")
    parser.add_argument("--smoke", action="store_true", help="liga corta de comprobación (4 generaciones pequeñas)")
    parser.add_argument("--generations", type=int, default=None, help="0 = sin fin (por defecto en league)")
    parser.add_argument("--variant", choices=VARIANTS, default="ranking")
    parser.add_argument("--variants", default="ranking,pareto,behavior,random")
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--n-gen", type=int, default=4)
    parser.add_argument("--tasks", type=int, default=None)
    parser.add_argument("--cells", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--level", type=int, default=50, help="nivel común del Stadium (evoluciona y recalcula stats)")
    parser.add_argument("--raw-levels", action="store_true", help="usar los niveles reales de las partidas")
    parser.add_argument("--no-lessons", action="store_true")
    parser.add_argument("--no-deep", action="store_true")
    parser.add_argument("--no-exchange", action="store_true")
    parser.add_argument("--refresh-teams", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run", default=None, help="nombre/carpeta de la liga")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    random.seed(args.seed)
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    if args.mode == "compare":
        compare(args)
    elif args.mode == "ablation":
        ablation(args)
    elif args.mode == "teams":
        teams(args)
    else:
        league(args)


if __name__ == "__main__":
    main()
