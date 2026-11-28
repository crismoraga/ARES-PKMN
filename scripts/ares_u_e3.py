"""E3 — one pixel agent across eight Pokémon games (GB, GBC, GBA, NDS; generations 1-5).

The agent of E2 (pixels 84x84x4, 8 universal buttons + NOOP, PPO + RND + episodic novelty), trained
on FireRed then Emerald, is plugged into every game of the ARES-U Game Abstraction Layer without
any change of code or architecture. Per game and seed (the seed pairs with the E2 seed whose policy
is transferred):

* ``random``     uniform random buttons (baseline);
* ``zero_shot``  the E2 GBA policy as is (no training on the target game);
* ``finetune``   the E2 policy trained ``--steps`` more on the target game;
* ``scratch``    PPO+RND+novelty from scratch with the same ``--steps`` on the target game;
* ``encoder``    representation transfer: the E2 visual encoder with fresh policy/value heads, trained
                 with the same ``--steps`` (added after observing the entropy/plasticity loss of
                 ``finetune``).

Metrics (measurement only): unique screens per episode (universal: coarse screen signatures, see
``ares_u.nav.env.screen_key``) for every game, and unique RAM cells (map, x, y) where a verified
adapter exists (Rojo, Cristal; FireRed/Emerald in E2). The summary validates the screen metric
against the RAM cells and reports per-seed means, mean +- sd over seeds and hierarchical-bootstrap
CIs of the paired differences.

    python -m scripts.ares_u_e3 protocol --games rojo cristal --seeds 0 1 2 --steps 30000
    python -m scripts.ares_u_e3 summary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.ares_u_nav import _run_phase

ROOT = Path(__file__).resolve().parents[1]
E2 = ROOT / "artifacts" / "ares_u" / "nav" / "final"
E3 = ROOT / "artifacts" / "ares_u" / "e3"
METHODS = ("random", "zero_shot", "finetune", "scratch", "encoder")
TITLES = {"rojo": "Rojo (GB, G1)", "cristal": "Cristal (GBC, G2)", "firered": "Rojo Fuego (GBA, G3)",
          "emerald": "Esmeralda (GBA, G3)", "platino": "Platino (NDS, G4)", "heartgold": "HeartGold (NDS, G4)",
          "negro": "Negro (NDS, G5)", "negro2": "Negro 2 (NDS, G5)"}


def e2_policy(seed: int) -> Path:
    return E2 / f"seed{seed}" / "emerald_transfer" / "final.pt"


def run_protocol(args: argparse.Namespace) -> None:
    out_root: Path = args.out_dir
    env_flags = ["--episode-steps", str(args.episode_steps), "--action-repeat", "24", "--frame-stack", "4"]
    common = ["--num-envs", str(args.num_envs), "--device", args.device]
    for game in args.games:
        for seed in args.seeds:
            policy = e2_policy(seed)
            if not policy.exists():
                raise FileNotFoundError(f"E2 policy for seed {seed} missing: {policy}")
            run = out_root / game / f"seed{seed}"
            logs = run / "logs"
            s = str(seed)
            trains = {"finetune": ["--init", f"policy:{policy}"], "scratch": ["--init", "scratch"],
                      "encoder": ["--init", f"encoder:{policy}"]}
            for method, extra in trains.items():
                if method not in args.methods:
                    continue
                out_dir = run / method
                _run_phase(f"{game}_{method}_train_s{s}",
                           ["train", "--game", game, "--out-dir", str(out_dir), "--steps", str(args.steps),
                            "--seed", s, "--resume", *extra, *common, *env_flags], out_dir / "done.json", logs)
            policies = {"random": "", "zero_shot": str(policy), "finetune": str(run / "finetune" / "final.pt"),
                        "scratch": str(run / "scratch" / "final.pt"), "encoder": str(run / "encoder" / "final.pt")}
            for method, path in policies.items():
                if method not in args.methods:
                    continue
                out = run / f"eval_{method}.json"
                argv = ["evaluate", "--game", game, "--method", method, "--out", str(out),
                        "--seed", str(100_000 + seed), "--episodes", str(args.episodes), *common, *env_flags]
                if path:
                    argv += ["--policy", path]
                _run_phase(f"{game}_{method}_eval_s{s}", argv, out, logs)
    summarize(out_root)


def _hier_boot_diff(a: list[list[float]], b: list[list[float]], draws: int = 10_000, seed: int = 0
                    ) -> tuple[float, float, float]:
    """Mean difference (a - b) with a hierarchical bootstrap CI: resample seed pairs, then episodes."""
    rng = np.random.default_rng(seed)
    pairs = [(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)) for x, y in zip(a, b) if x and y]
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean([x.mean() - y.mean() for x, y in pairs]))
    stats = np.empty(draws)
    for k in range(draws):
        chosen = [pairs[i] for i in rng.integers(0, len(pairs), size=len(pairs))]
        stats[k] = np.mean([rng.choice(x, size=x.size).mean() - rng.choice(y, size=y.size).mean() for x, y in chosen])
    return point, float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def summarize(out_root: Path = E3) -> Path:
    data: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for path in sorted(out_root.glob("*/seed*/eval_*.json")):
        game, seed = path.parts[-3], int(path.parts[-2].removeprefix("seed"))
        result = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault(game, {}).setdefault(result["method"], {})[seed] = result
    summary: dict[str, Any] = {"games": {}, "validation": {}}
    lines = ["# E3 — un mismo agente de píxeles en ocho juegos", "",
             "Métrica universal: pantallas únicas por episodio (firma gruesa de la pantalla). Celdas RAM (mapa, x, y) "
             "donde hay adaptador verificado. Media ± d.e. entre semillas [valores por semilla]; diferencias con IC 95% "
             "bootstrap jerárquico (semillas, luego episodios).", "",
             "| Juego | Métrica | " + " | ".join(METHODS) + " |", "|---|---|" + "---|" * len(METHODS)]
    paired_cells: list[tuple[float, float]] = []
    for game in sorted(data, key=lambda g: list(TITLES).index(g) if g in TITLES else 99):
        by_method = data[game]
        entry: dict[str, Any] = {}
        for metric, key in (("pantallas", "screens"), ("celdas", "cells")):
            values = {m: [float(np.mean(by_method[m][s][key])) for s in sorted(by_method.get(m, {}))
                          if by_method[m][s].get(key)] for m in METHODS}
            if metric == "celdas" and not any(v and max(v) > 0 for v in values.values()):
                continue
            entry[key] = {m: {"seed_means": v, "mean": float(np.mean(v)) if v else float("nan"),
                              "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0} for m, v in values.items()}
            cells = [f"{np.mean(v):.1f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.1f} [{', '.join(f'{x:.0f}' for x in v)}]"
                     if v else "-" for v in (values[m] for m in METHODS)]
            lines.append(f"| {TITLES.get(game, game)} | {metric} | " + " | ".join(cells) + " |")
            tests = {}
            for first, second in (("zero_shot", "random"), ("finetune", "scratch"), ("finetune", "random"),
                                  ("scratch", "random"), ("encoder", "scratch"), ("encoder", "finetune")):
                seeds = sorted(set(by_method.get(first, {})) & set(by_method.get(second, {})))
                a = [by_method[first][s][key] for s in seeds]
                b = [by_method[second][s][key] for s in seeds]
                tests[f"{first}-{second}"] = _hier_boot_diff(a, b)
            entry[f"{key}_tests"] = {k: {"diff": v[0], "ci": [v[1], v[2]]} for k, v in tests.items()}
        for method in METHODS:
            for result in by_method.get(method, {}).values():
                if result.get("cells") and result.get("screens") and max(result["cells"]) > 0:
                    paired_cells += list(zip(map(float, result["screens"]), map(float, result["cells"])))
        summary["games"][game] = entry
    if len(paired_cells) > 3:
        from scipy.stats import spearmanr

        screen_values = [pair[0] for pair in paired_cells]
        cell_values = [pair[1] for pair in paired_cells]
        rho = spearmanr(screen_values, cell_values)
        summary["validation"] = {"episodes": len(paired_cells), "spearman_rho": float(rho.statistic),
                                 "p": float(rho.pvalue)}
    lines += ["", "## Diferencias (IC 95% bootstrap jerárquico)", "", "| Juego | Métrica | Contraste | Δ | IC 95% |",
              "|---|---|---|---|---|"]
    for game, entry in summary["games"].items():
        for key in ("screens", "cells"):
            for name, test in entry.get(f"{key}_tests", {}).items():
                lines.append(f"| {TITLES.get(game, game)} | {key} | {name} | {test['diff']:+.1f} | "
                             f"[{test['ci'][0]:+.1f}, {test['ci'][1]:+.1f}] |")
    if summary["validation"]:
        v = summary["validation"]
        lines += ["", f"Validación de la métrica universal: Spearman ρ = {v['spearman_rho']:.2f} entre pantallas y celdas "
                      f"RAM (n = {v['episodes']} episodios, p = {v['p']:.1e})."]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    path = out_root / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)
    return path


MONTAGE_GAMES = ("rojo", "cristal", "firered", "emerald", "platino", "heartgold", "negro", "negro2")


def _capture(game: str, policy: str, steps: int, seed: int, out: str) -> None:
    """Run the E2 policy zero-shot for ``steps`` steps in ``game`` and save the native screen."""
    import torch
    from PIL import Image

    from ares_u.nav.env import NavConfig, NavEnv
    from ares_u.nav.train import _policy_step, load_policy

    config = NavConfig(game, noop_max=0)
    env = NavEnv(config, seed=seed)
    model = load_policy(Path(policy), config).eval()
    sampler = torch.Generator().manual_seed(seed)
    obs = env.reset()
    for _ in range(steps):
        actions, _, _ = _policy_step(model, obs[None], torch.device("cpu"), sampler)
        obs, *_ = env.step(int(actions[0]))
    frame = env.bridge.framebuffer()
    rgb = np.stack([frame & 0xFF, (frame >> 8) & 0xFF, (frame >> 16) & 0xFF], axis=-1).astype(np.uint8)
    Image.fromarray(rgb).save(out)
    env.close()


def montage(out_root: Path, steps: int = 96, seed: int = 0) -> Path:
    """8-game montage of the zero-shot agent (one process per game: DeSmuME is a per-process singleton)."""
    import multiprocessing as mp

    from PIL import Image, ImageDraw

    out_root.mkdir(parents=True, exist_ok=True)
    tiles: list[Image.Image] = []
    context = mp.get_context("spawn")
    for game in MONTAGE_GAMES:
        shot = out_root / f"shot_{game}.png"
        process = context.Process(target=_capture, args=(game, str(e2_policy(seed)), steps, seed, str(shot)))
        process.start()
        process.join()
        if process.exitcode != 0 or not shot.exists():
            print(f"[warn] capture failed for {game}")
            continue
        image = Image.open(shot).convert("RGB").resize((240, 168), Image.Resampling.NEAREST)
        tile = Image.new("RGB", (240, 184), (255, 255, 255))
        tile.paste(image, (0, 16))
        ImageDraw.Draw(tile).text((4, 2), TITLES.get(game, game), fill=(0, 0, 0))
        tiles.append(tile)
    sheet = Image.new("RGB", (2 * 240 + 8, ((len(tiles) + 1) // 2) * 188), (255, 255, 255))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 2) * 248, (index // 2) * 188))
    path = out_root / "montage.png"
    sheet.save(path)
    print(path)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    shots = sub.add_parser("montage")
    shots.add_argument("--out-dir", type=Path, default=E3)
    shots.add_argument("--steps", type=int, default=96)
    protocol = sub.add_parser("protocol")
    protocol.add_argument("--games", nargs="+", required=True)
    protocol.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    protocol.add_argument("--steps", type=int, default=30_000)
    protocol.add_argument("--episodes", type=int, default=8)
    protocol.add_argument("--episode-steps", type=int, default=512)
    protocol.add_argument("--num-envs", type=int, default=4)
    protocol.add_argument("--device", default="auto")
    protocol.add_argument("--out-dir", type=Path, default=E3)
    protocol.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    summary = sub.add_parser("summary")
    summary.add_argument("--out-dir", type=Path, default=E3)
    args = parser.parse_args(argv)
    if args.command == "protocol":
        run_protocol(args)
    elif args.command == "montage":
        montage(args.out_dir, args.steps)
    else:
        summarize(args.out_dir)


if __name__ == "__main__":
    main()
