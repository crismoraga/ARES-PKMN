"""E4 — generational curriculum: one pixel agent that evolves from Gen 1 to Gen 5.

The agent starts in Pokémon Rojo (Gen 1, GB) and moves through one game per generation,
Rojo -> Cristal -> Esmeralda -> Platino -> Negro, starting each game from what it learned in the
previous one. Current practice (Pleines et al. 2025, PokeRL, PokéAgent RL entries) trains every game
from scratch; the proposal is this generational curriculum. Two variants:

* ``finetune`` (proposal): the whole actor-critic (perception + behaviour) is carried over;
* ``encoder`` (ablation): only the visual encoder is carried over and the policy/value heads are
  re-initialised; E3 showed this loses the jumpstart, i.e. what transfers is behaviour, not only
  perception.

Per seed and stage the agent is evaluated on every game seen so far (retention), on the next game
before training on it (jumpstart / forward transfer) and, after the last stage, on two games never
seen (HeartGold, Negro 2: held-out generalisation). Baselines share the E3 evaluation settings
(8 episodes x 512 steps, eval seed 100000 + seed): uniform random and PPO+RND from scratch with the
same per-game budget (E3 runs, plus Esmeralda here). Stage 0 *is* the E3 Rojo scratch run.

Continual-learning summary: matrix R[i][j] = unique screens on game j after stage i, normalised
score (R - random_j) / (scratch_j - random_j), jumpstart, gain over scratch and forgetting.

    python -m scripts.ares_u_e4 protocol --transfer encoder finetune --seeds 0 1 2
    python -m scripts.ares_u_e4 summary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.ares_u_nav import _run_phase

ROOT = Path(__file__).resolve().parents[1]
E3 = ROOT / "artifacts" / "ares_u" / "e3"
E4 = ROOT / "artifacts" / "ares_u" / "e4"
CHAIN = ("rojo", "cristal", "emerald", "platino", "negro")
HELD_OUT = ("heartgold", "negro2")
GENERATION = {"rojo": 1, "cristal": 2, "emerald": 3, "platino": 4, "negro": 5, "heartgold": 4, "negro2": 5}
TITLE = {"rojo": "Rojo", "cristal": "Cristal", "emerald": "Esmeralda", "platino": "Platino", "negro": "Negro",
         "heartgold": "HeartGold", "negro2": "Negro 2"}
STEPS = {"rojo": 30_000, "cristal": 30_000, "emerald": 30_000, "platino": 8_000, "negro": 8_000}
ENV_FLAGS = ["--episode-steps", "512", "--action-repeat", "24", "--frame-stack", "4"]


def _common(args: argparse.Namespace) -> list[str]:
    return ["--num-envs", str(args.num_envs), "--device", args.device]


def _evaluate(game: str, policy: str, method: str, out: Path, seed: int, logs: Path, args: argparse.Namespace) -> None:
    argv = ["evaluate", "--game", game, "--method", method, "--out", str(out), "--seed", str(100_000 + seed),
            "--episodes", str(args.episodes), *_common(args), *ENV_FLAGS]
    if policy:
        argv += ["--policy", policy]
    _run_phase(f"eval_{out.stem}_s{seed}", argv, out, logs)


def baselines(args: argparse.Namespace) -> None:
    """Esmeralda random + scratch with the E3 settings (the other games come from E3)."""
    for seed in args.seeds:
        run = E4 / "baselines" / "emerald" / f"seed{seed}"
        logs = run / "logs"
        _run_phase(f"emerald_scratch_train_s{seed}",
                   ["train", "--game", "emerald", "--out-dir", str(run / "scratch"), "--steps", str(STEPS["emerald"]),
                    "--seed", str(seed), "--resume", "--init", "scratch", *_common(args), *ENV_FLAGS],
                   run / "scratch" / "done.json", logs)
        _evaluate("emerald", "", "random", run / "eval_random.json", seed, logs, args)
        _evaluate("emerald", str(run / "scratch" / "final.pt"), "scratch", run / "eval_scratch.json", seed, logs, args)


def run_chain(transfer: str, seed: int, args: argparse.Namespace) -> None:
    root = E4 / transfer / f"seed{seed}"
    logs = root / "logs"
    previous = E3 / "rojo" / f"seed{seed}" / "scratch" / "final.pt"  # stage 0: Gen 1 from scratch (E3)
    if not previous.exists():
        raise FileNotFoundError(f"stage 0 (E3 Rojo scratch) missing: {previous}")
    init_kind = "policy" if transfer == "finetune" else "encoder"
    for stage, game in enumerate(CHAIN):
        if stage > 0:
            out_dir = root / f"stage{stage}_{game}"
            _run_phase(f"{transfer}_stage{stage}_{game}_s{seed}",
                       ["train", "--game", game, "--out-dir", str(out_dir), "--steps", str(STEPS[game]),
                        "--seed", str(seed), "--resume", "--init", f"{init_kind}:{previous}", *_common(args),
                        *ENV_FLAGS], out_dir / "done.json", logs)
            previous = out_dir / "final.pt"
        targets = list(CHAIN[: stage + 2]) if stage + 1 < len(CHAIN) else list(CHAIN) + list(HELD_OUT)
        for target in targets:
            _evaluate(target, str(previous), f"{transfer}_after{stage}", root / f"after{stage}_on_{target}.json",
                      seed, logs, args)


def run_protocol(args: argparse.Namespace) -> None:
    baselines(args)
    for transfer in args.transfer:
        for seed in args.seeds:
            run_chain(transfer, seed, args)
    summarize()


# ----------------------------------------------------------------------------- summary

def _screens(path: Path) -> float | None:
    if not path.exists():
        return None
    return float(np.mean(json.loads(path.read_text(encoding="utf-8"))["screens"]))


def _baseline(game: str, method: str, seed: int) -> float | None:
    if game == "emerald":
        return _screens(E4 / "baselines" / "emerald" / f"seed{seed}" / f"eval_{method}.json")
    return _screens(E3 / game / f"seed{seed}" / f"eval_{method}.json")


def summarize() -> Path:
    seeds = sorted({int(p.name.removeprefix("seed")) for p in E4.glob("*/seed*") if p.parent.name != "baselines"})
    games = list(CHAIN) + list(HELD_OUT)
    result: dict[str, Any] = {"chain": list(CHAIN), "held_out": list(HELD_OUT), "seeds": seeds, "baselines": {},
                              "transfer": {}}
    for game in games:
        result["baselines"][game] = {method: [v for v in (_baseline(game, method, s) for s in seeds) if v is not None]
                                     for method in ("random", "scratch", "zero_shot")}
    lines = ["# E4 — currículo generacional (Rojo → Cristal → Esmeralda → Platino → Negro)", "",
             "Pantallas únicas por episodio de 512 pasos (media entre semillas). Puntaje normalizado = "
             "(R − aleatorio) / (desde cero − aleatorio): 0 = azar, 1 = entrenar ese juego desde cero con el mismo "
             "presupuesto, >1 = mejor que desde cero.", ""]
    for transfer in sorted(p.name for p in E4.iterdir() if p.is_dir() and p.name in ("finetune", "encoder")):
        matrix: dict[str, dict[str, list[float]]] = {}
        for stage in range(len(CHAIN)):
            for game in games:
                values = [v for v in (_screens(E4 / transfer / f"seed{s}" / f"after{stage}_on_{game}.json")
                                      for s in seeds) if v is not None]
                if values:
                    matrix.setdefault(str(stage), {})[game] = values
        entry: dict[str, Any] = {"matrix": matrix, "normalised": {}, "jumpstart": {}, "gain": {}, "forgetting": {}}
        for stage_key, row in matrix.items():
            for game, values in row.items():
                rnd, scr = result["baselines"][game]["random"], result["baselines"][game]["scratch"]
                if rnd and scr and float(np.mean(scr)) != float(np.mean(rnd)):
                    entry["normalised"].setdefault(stage_key, {})[game] = \
                        (float(np.mean(values)) - float(np.mean(rnd))) / (float(np.mean(scr)) - float(np.mean(rnd)))
        for stage, game in enumerate(CHAIN):
            if stage > 0 and game in matrix.get(str(stage - 1), {}):
                before = matrix[str(stage - 1)][game]
                rnd = result["baselines"][game]["random"]
                entry["jumpstart"][game] = {"zero_shot": float(np.mean(before)),
                                            "random": float(np.mean(rnd)) if rnd else None}
            if game in matrix.get(str(stage), {}):
                scr = result["baselines"][game]["scratch"]
                own = matrix[str(stage)][game]
                entry["gain"][game] = {"chain": float(np.mean(own)), "scratch": float(np.mean(scr)) if scr else None,
                                       "diff": float(np.mean(own) - np.mean(scr)) if scr else None}
            last = str(len(CHAIN) - 1)
            if game in matrix.get(last, {}) and game in matrix.get(str(stage), {}):
                entry["forgetting"][game] = float(np.mean(matrix[str(stage)][game]) - np.mean(matrix[last][game]))
        result["transfer"][transfer] = entry
        lines += [f"## Transferencia: {transfer}", "", "| Tras etapa \\ juego | " +
                  " | ".join(TITLE[g] for g in games) + " |", "|---|" + "---|" * len(games)]
        for stage_key in sorted(matrix, key=int):
            cells = []
            for game in games:
                shown = matrix[stage_key].get(game)
                norm = entry["normalised"].get(stage_key, {}).get(game)
                cells.append("-" if not shown else f"{np.mean(shown):.0f}" + (f" ({norm:.2f})" if norm is not None else ""))
            lines.append(f"| {int(stage_key)} ({TITLE[CHAIN[int(stage_key)]]}) | " + " | ".join(cells) + " |")
        lines.append("")
    E4.mkdir(parents=True, exist_ok=True)
    (E4 / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    path = E4 / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    protocol = sub.add_parser("protocol")
    protocol.add_argument("--transfer", nargs="+", default=["encoder", "finetune"], choices=["encoder", "finetune"])
    protocol.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    protocol.add_argument("--episodes", type=int, default=8)
    protocol.add_argument("--num-envs", type=int, default=4)
    protocol.add_argument("--device", default="auto")
    sub.add_parser("summary")
    args = parser.parse_args(argv)
    if args.command == "protocol":
        run_protocol(args)
    else:
        summarize()


if __name__ == "__main__":
    main()
