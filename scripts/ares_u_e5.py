"""E5 — generational neuroevolution: proposals compete, merge into one, and seed the next generation.

For every game of the curriculum Rojo (G1) -> Cristal (G2) -> Esmeralda (G3) -> Platino (G4) -> Negro (G5):

1. **Birth.** K proposals are born from the unified agent of the previous generation (for Rojo, from
   one shared random initialisation). Proposal 0 is the *elite* (inherits weights and default
   hyper-parameters unchanged); the others are *mutants*: different learning rate, entropy and
   novelty coefficients, and a shrink-and-perturb of the inherited weights (lambda*theta + sigma*eps).
2. **Training.** Each proposal trains on the game (PPO + RND + episodic novelty, pixels only).
3. **Combat.** Paired duels: every proposal plays the *same* validation episodes (same emulator
   seeds); a duel between two proposals is won episode by episode by the one that explores more
   unique screens. The duel matrix gives a ranking (wins, ties count 1/2).
4. **Fusion.** Greedy model soup (Wortsman et al., 2022): starting from the winner, the next-ranked
   proposal is added to the uniform weight average only if the soup's validation score does not
   drop. Weight averaging is meaningful because all proposals descend from the same parent.
5. **Next generation.** The unified agent is tested on the game (test seeds = E3/E4), on the next
   game before training on it (jumpstart), and becomes the parent of the next generation.

Validation (selection, soup) and test (reported) use disjoint evaluation seeds. Compute: K proposals
x B/2 environment steps per game (2x the single-lineage curriculum of E4), reported explicitly.

    python -m scripts.ares_u_e5 protocol --seeds 0
    python -m scripts.ares_u_e5 summary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.ares_u_nav import _run_phase

ROOT = Path(__file__).resolve().parents[1]
E5 = ROOT / "artifacts" / "ares_u" / "e5"
CHAIN = ("rojo", "cristal", "emerald", "platino", "negro")
HELD_OUT = ("heartgold", "negro2")
BUDGET = {"rojo": 15_000, "cristal": 15_000, "emerald": 15_000, "platino": 4_000, "negro": 4_000}
ENV_FLAGS = ["--episode-steps", "512", "--action-repeat", "24", "--frame-stack", "4"]
DEFAULT_HP = {"lr": 2.5e-4, "ent_coef": 0.01, "rnd_coef": 0.5, "epi_coef": 1.0}
MUTATIONS = {"lr": (1.5e-4, 2.5e-4, 4e-4), "ent_coef": (0.005, 0.01, 0.02), "rnd_coef": (0.25, 0.5, 1.0),
             "epi_coef": (0.5, 1.0, 2.0)}


def proposal_hparams(seed: int, generation: int, index: int) -> dict[str, float]:
    if index == 0:
        return dict(DEFAULT_HP)
    rng = np.random.default_rng(10_007 * seed + 101 * generation + index)
    return {key: float(rng.choice(values)) for key, values in MUTATIONS.items()}


def perturbed_init(parent: Path | None, out: Path, seed: int, generation: int, index: int,
                   shrink: float = 0.98, sigma: float = 0.02) -> Path:
    """Elite: the parent unchanged. Mutant: shrink-and-perturb of the parent (per-tensor noise scale)."""
    import torch

    from ares_u.nav.env import NavConfig
    from ares_u.nav.train import build_actor_critic

    out.parent.mkdir(parents=True, exist_ok=True)
    if parent is None:
        torch.manual_seed(1_000 + seed)
        state = build_actor_critic(NavConfig("rojo")).state_dict()
    else:
        state = torch.load(parent, map_location="cpu", weights_only=True)["model"]
    if index > 0:
        generator = torch.Generator().manual_seed(7_919 * seed + 31 * generation + index)
        mutated = {}
        for key, tensor in state.items():
            if tensor.is_floating_point() and tensor.numel() > 1:
                scale = float(tensor.std()) or 1e-3
                noise = torch.randn(tensor.shape, generator=generator) * sigma * scale
                mutated[key] = shrink * tensor + noise
            else:
                mutated[key] = tensor
        state = mutated
    torch.save({"model": state}, out)
    return out


def soup(paths: list[Path], out: Path) -> Path:
    import torch

    states = [torch.load(path, map_location="cpu", weights_only=True)["model"] for path in paths]
    averaged = {key: (sum(state[key].float() for state in states) / len(states)).to(states[0][key].dtype)
                if states[0][key].is_floating_point() else states[0][key] for key in states[0]}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": averaged}, out)
    return out


def _screens(path: Path) -> list[float]:
    return [float(value) for value in json.loads(path.read_text(encoding="utf-8"))["screens"]]


def evaluate(game: str, policy: Path, out: Path, eval_seed: int, episodes: int, logs: Path,
             args: argparse.Namespace) -> list[float]:
    _run_phase(f"eval_{out.parent.name}_{out.stem}",
               ["evaluate", "--game", game, "--method", out.stem, "--policy", str(policy), "--out", str(out),
                "--seed", str(eval_seed), "--episodes", str(episodes), "--num-envs", str(args.num_envs),
                "--device", args.device, *ENV_FLAGS], out, logs)
    return _screens(out)


def duel_matrix(scores: list[list[float]]) -> list[list[float]]:
    """wins[i][j] = fraction of paired validation episodes where i explored more than j (ties = 1/2)."""
    k = len(scores)
    wins = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            if i != j:
                pairs = list(zip(scores[i], scores[j]))
                wins[i][j] = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in pairs) / max(1, len(pairs))
    return wins


def run_seed(seed: int, args: argparse.Namespace) -> None:
    root = Path(args.out_dir) / f"seed{seed}"
    logs = root / "logs"
    parent: Path | None = None
    validation_seed, test_seed = 200_000 + seed, 100_000 + seed
    for generation, game in enumerate(CHAIN):
        if game not in args.games:
            break
        gen_dir = root / f"gen{generation}_{game}"
        record_path = gen_dir / "generation.json"
        if record_path.exists():
            parent = Path(json.loads(record_path.read_text(encoding="utf-8"))["unified"])
            continue
        proposals: list[dict[str, Any]] = []
        for index in range(args.proposals):
            hp = proposal_hparams(seed, generation, index)
            init = perturbed_init(parent, gen_dir / f"p{index}" / "init.pt", seed, generation, index)
            out_dir = gen_dir / f"p{index}"
            _run_phase(f"gen{generation}_{game}_p{index}_s{seed}",
                       ["train", "--game", game, "--out-dir", str(out_dir), "--steps", str(int(BUDGET[game] * args.budget_scale)),
                        "--seed", str(seed * 100 + index), "--resume", "--init", f"policy:{init}",
                        "--lr", str(hp["lr"]), "--ent-coef", str(hp["ent_coef"]), "--rnd-coef", str(hp["rnd_coef"]),
                        "--epi-coef", str(hp["epi_coef"]), "--num-envs", str(args.num_envs), "--device", args.device,
                        *ENV_FLAGS], out_dir / "done.json", logs)
            scores = evaluate(game, out_dir / "final.pt", gen_dir / f"p{index}" / "val.json", validation_seed,
                              args.val_episodes, logs, args)
            proposals.append({"index": index, "role": "elite" if index == 0 else "mutant", "hparams": hp,
                              "val_screens": scores, "val_mean": float(np.mean(scores))})
        wins = duel_matrix([p["val_screens"] for p in proposals])
        for p, row in zip(proposals, wins):
            p["duel_wins"] = float(sum(row))
        ranking = sorted(proposals, key=lambda p: (p["duel_wins"], p["val_mean"]), reverse=True)
        member_ids = [int(ranking[0]["index"])]
        soup_score = float(ranking[0]["val_mean"])
        soup_trace = [{"members": list(member_ids), "val_mean": soup_score, "accepted": True}]
        for candidate in ranking[1:]:
            trial_ids = member_ids + [int(candidate["index"])]
            tag = "-".join(str(i) for i in trial_ids)
            trial = soup([gen_dir / f"p{i}" / "final.pt" for i in trial_ids], gen_dir / f"soup_{tag}" / "final.pt")
            trial_score = float(np.mean(evaluate(game, trial, trial.parent / "val.json", validation_seed,
                                                 args.val_episodes, logs, args)))
            accepted = trial_score >= soup_score
            soup_trace.append({"members": trial_ids, "val_mean": trial_score, "accepted": accepted})
            if accepted:
                member_ids, soup_score = trial_ids, trial_score
        members = [gen_dir / f"p{i}" / "final.pt" for i in member_ids]
        unified = soup(members, gen_dir / "unified" / "final.pt")
        best = gen_dir / f"p{ranking[0]['index']}" / "final.pt"
        tests = {"unified": evaluate(game, unified, gen_dir / "unified" / "test.json", test_seed, args.episodes, logs,
                                     args),
                 "best_proposal": evaluate(game, best, gen_dir / "test_best.json", test_seed, args.episodes, logs, args)}
        transfers: dict[str, list[float]] = {}
        targets = [CHAIN[generation + 1]] if generation + 1 < len(CHAIN) else list(HELD_OUT)
        for target in targets:
            transfers[target] = evaluate(target, unified, gen_dir / f"zero_shot_{target}.json", test_seed,
                                         args.episodes, logs, args)
        record = {"game": game, "generation": generation, "parent": str(parent) if parent else "shared random init",
                  "proposals": proposals, "duel_matrix": wins, "ranking": [p["index"] for p in ranking],
                  "soup_trace": soup_trace, "soup_members": [str(path) for path in members], "unified": str(unified),
                  "test": {k: {"screens": v, "mean": float(np.mean(v))} for k, v in tests.items()},
                  "zero_shot_next": {k: {"screens": v, "mean": float(np.mean(v))} for k, v in transfers.items()},
                  "env_steps": BUDGET[game] * args.proposals}
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        parent = unified


def summarize() -> Path:
    E5.mkdir(parents=True, exist_ok=True)
    lines = ["# E5 — neuroevolución generacional (propuestas que compiten y se fusionan)", "",
             "| Semilla | Gen | Juego | Propuestas (val) | Ganador | Sopa | Unificado (test) | Mejor individual (test) | Zero-shot sig. |",
             "|---|---|---|---|---|---|---|---|---|"]
    records: list[dict[str, Any]] = []
    for path in sorted(E5.glob("seed*/gen*/generation.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["seed"] = int(path.parts[-3].removeprefix("seed"))
        records.append(record)
        vals = ", ".join(f"{p['val_mean']:.0f}" for p in record["proposals"])
        nxt = "; ".join(f"{k} {v['mean']:.0f}" for k, v in record["zero_shot_next"].items())
        lines.append(f"| {record['seed']} | {record['generation']} | {record['game']} | {vals} | p{record['ranking'][0]} | "
                     f"{len(record['soup_members'])} | {record['test']['unified']['mean']:.0f} | "
                     f"{record['test']['best_proposal']['mean']:.0f} | {nxt} |")
    (E5 / "summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    path = E5 / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    protocol = sub.add_parser("protocol")
    protocol.add_argument("--seeds", nargs="+", type=int, default=[0])
    protocol.add_argument("--proposals", type=int, default=4)
    protocol.add_argument("--episodes", type=int, default=8)
    protocol.add_argument("--val-episodes", type=int, default=4)
    protocol.add_argument("--num-envs", type=int, default=4)
    protocol.add_argument("--device", default="auto")
    protocol.add_argument("--games", nargs="+", default=list(CHAIN), help="prefix of the curriculum to run")
    protocol.add_argument("--budget-scale", type=float, default=1.0, help="smoke tests only")
    protocol.add_argument("--out-dir", type=Path, default=E5)
    sub.add_parser("summary")
    args = parser.parse_args(argv)
    if args.command == "protocol":
        for seed in args.seeds:
            run_seed(seed, args)
        summarize()
    else:
        summarize()


if __name__ == "__main__":
    main()
