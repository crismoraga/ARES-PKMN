"""E6 — novedad de lugar: atacar el vicio "explorar no es avanzar" sin leer la RAM.

Diagnosis (``scripts.ares_u_scorecard``): the screen-novelty agent sees more *screens* than the
per-game scratch baseline but not more *places*: distinct RAM map ids per episode were equal or lower
(Esmeralda: -0.6 maps). The episodic bonus pays for any new screen signature, so it is satisfied by
wandering inside one place.

Remedy (pixels only): every Pokémon generation changes place through a fade to black or white. The env
flags a step as a *transition* when a near-uniform frame is seen and the step ends on a non-blank frame
(``ares_u.nav.env.frame_is_blank``); ``PlaceNovelty`` pays 1.0 the first time in the episode that a
transition lands on a coarse scene key (4x4 cells, 4 grey levels). Graded against the RAM map id
(``scripts.ares_u_place_check``): recall 1.0 of new maps in Rojo, Cristal and Esmeralda; precision
0.13-0.64 (GBA menus also fade), and the bonus is paid once per scene, which bounds that excess.

Protocol, identical to the scratch baselines (30k steps, 4 envs, 512-step episodes, action repeat 24,
eval 8 episodes with seed 100000 + seed), only ``--place-coef 1.0`` differs. Criterion fixed before
running: distinct maps per episode (RAM, never seen by the agent) higher than scratch with the 95%
hierarchical-bootstrap interval above 0 in at least 2 of the 3 games, without losing more than 10% of
the unique screens.

    python -m scripts.ares_u_place protocol --seeds 0 1 2
    python -m scripts.ares_u_place summary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.ares_u_nav import _run_phase
from scripts.ares_u_scorecard import arms_for, baseline_path, ci_text, hboot

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "ares_u" / "e6_place"
GAMES = ("rojo", "cristal", "emerald")
ENV_FLAGS = ["--episode-steps", "512", "--action-repeat", "24", "--frame-stack", "4"]


def run_seed(seed: int, args: argparse.Namespace) -> None:
    for game in args.games:
        run = OUT / game / f"seed{seed}"
        logs = run / "logs"
        _run_phase(f"{game}_place_train_s{seed}",
                   ["train", "--game", game, "--out-dir", str(run / "place"), "--steps", str(args.steps), "--seed",
                    str(seed), "--resume", "--init", "scratch", "--place-coef", str(args.place_coef),
                    "--num-envs", str(args.num_envs), "--device", args.device, *ENV_FLAGS],
                   run / "place" / "done.json", logs)
        _run_phase(f"{game}_place_eval_s{seed}",
                   ["evaluate", "--game", game, "--method", "place", "--policy", str(run / "place" / "final.pt"),
                    "--out", str(run / "eval_place.json"), "--seed", str(100_000 + seed), "--episodes",
                    str(args.episodes), "--num-envs", str(args.num_envs), "--device", args.device, *ENV_FLAGS],
                   run / "eval_place.json", logs)


def summarize() -> Path:
    rows: list[dict[str, Any]] = []
    lines = ["# E6 — novedad de lugar (fundido → escena nueva) frente a desde cero", "",
             "| Juego | Mapas: lugar − desde cero [IC95] | Pantallas: lugar / desde cero | Semillas |", "|---|---|---|---|"]
    passed = 0
    for game in GAMES:
        paths = {"place": lambda s, g=game: OUT / g / f"seed{s}" / "eval_place.json",
                 "scratch": lambda s, g=game: baseline_path(g, "scratch", s)}
        maps = arms_for(paths, key="maps")
        screens = arms_for(paths, key="screens")
        if maps is None or screens is None:
            continue
        diff, low, high, count = hboot(lambda m: m["place"] - m["scratch"], maps)
        ratio, rlow, rhigh, _ = hboot(lambda m: m["place"] / m["scratch"], screens)
        ok = low > 0 and ratio >= 0.9
        passed += int(ok)
        rows.append({"game": game, "maps_diff": [diff, low, high], "screens_ratio": [ratio, rlow, rhigh],
                     "seeds": count, "pass": ok,
                     "maps_place": float(np.mean([v.mean() for v in maps["place"]])),
                     "maps_scratch": float(np.mean([v.mean() for v in maps["scratch"]]))})
        lines.append(f"| {game} | {ci_text(diff, low, high)} | {ci_text(ratio, rlow, rhigh)} | {count} |")
    lines += ["", f"Criterio (≥2 de 3 juegos con IC > 0 y pantallas ≥ 90%): {passed}/{len(rows)} juegos."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps({"games": rows, "passed": passed}, indent=2), encoding="utf-8")
    path = OUT / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    protocol = sub.add_parser("protocol")
    protocol.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    protocol.add_argument("--games", nargs="+", default=list(GAMES))
    protocol.add_argument("--steps", type=int, default=30_000)
    protocol.add_argument("--place-coef", type=float, default=1.0)
    protocol.add_argument("--episodes", type=int, default=8)
    protocol.add_argument("--num-envs", type=int, default=4)
    protocol.add_argument("--device", default="auto")
    sub.add_parser("summary")
    args = parser.parse_args(argv)
    if args.command == "protocol":
        for seed in args.seeds:
            run_seed(seed, args)
    summarize()


if __name__ == "__main__":
    main()
