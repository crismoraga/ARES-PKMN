"""E5-combate — evolución generacional por torneo, fusión y herencia en el Stadium (Gen 1 -> Gen 2 -> Gen 3).

Proposals literally fight each other. Each generation g has a population of proposals (entity
Transformers, ID-free) that play a round-robin tournament on *validation* battles of that generation's
species; the ranking (Bradley-Terry ratings on the Elo scale) selects the members, and the members are
merged into one unified proposal, which is the parent of the next generation:

* **Generation 0 (Gen 1 species).** Population = the 10 E1 runs (5 seeds of PPO self-play and 5 of
  BC->PPO: two "schools" that never saw each other). They have different initialisations, so their
  weights cannot be averaged (permutation symmetry); they are merged by **policy distillation**: the
  ensemble of the top-3 of the tournament (mean of their action distributions) plays Gen1 battles and
  the champion is behaviour-cloned on its decisions. The distilled student is kept only if it scores at
  least as well as the champion against the anchor panel (Greedy, ARES prior) on validation battles.
* **Generations 1 and 2 (Gen 2, then Gen 3 species).** The unified parent is first measured zero-shot on
  the new species (jumpstart). Then K children are fine-tuned with PPO on the new generation's species:
  child 0 keeps the elite hyper-parameters, children 1..K-1 draw mutated ones (learning rate, entropy,
  share of rival battles, clip). Their league is **adversarial**: the parent and the top-3 of the
  previous generation are pinned frozen rivals. The children and the parent play the round-robin
  tournament; the children are merged by a **greedy model soup** in ranking order (they share the parent,
  so weight averaging is meaningful): an ingredient is accepted when the soup's validation score against
  the anchor panel (Greedy, prior, parent) does not decrease.
* **Control: single lineage with the same compute** (``lineal``): the same parent fine-tuned once with
  the elite hyper-parameters for K x the child budget, no tournament, no merge (what E4 does in
  exploration). The current practice (``E1``: train once on Gen 1, never adapt) is the other reference.

Test battles (indices 0..N-1; tournaments and soups use only validation indices) against Greedy on
gen1_train, gen2 and gen3, plus head-to-head duels. Planned comparisons, fixed before running:

* C1  U1 vs E1-PPO on gen2            (adaptation to the new generation)
* C2  U1 vs L1 on gen2                (evolution vs single lineage, same compute)
* C3  U1 vs L1 on gen3                (zero-shot transfer to a generation never trained on)
* C4  U2 vs L2 on gen3                (after training on Gen 3)
* C5  U2 vs E1-PPO on gen3
* C6  U2 vs U0 on gen1_train          (retention: does Gen 1 knowledge survive two generations?)

Paired per-battle sign-flip tests on the seed-averaged scores vs Greedy, Holm-corrected over C1..C6.

    python -m scripts.ares_u_e5_battle protocol --seeds 0 1 2
    python -m scripts.ares_u_e5_battle summary
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ares_u.battle.bc import BCConfig, train_bc
from ares_u.battle.model import Weights, build_policy, load_numpy
from ares_u.battle.paired import bootstrap_mean_ci, holm, sign_flip_p
from ares_u.battle.ppo import PPOConfig, load_weights, save_checkpoint, train_ppo
from ares_u.battle.rollout import WorkerPool
from ares_u.battle.training import VALIDATION_OFFSET, split_indices

ROOT = Path(__file__).resolve().parents[1]
E1 = ROOT / "artifacts" / "ares_u" / "battle" / "final"
OUT = ROOT / "artifacts" / "ares_u" / "e5_battle"
SPLIT_OF = {0: "gen1_train", 1: "gen2", 2: "gen3"}
TEST_SPLITS = ("gen1_train", "gen2", "gen3")
TOURNAMENT_OFFSET = VALIDATION_OFFSET + 50_000  # validation battles never used by E1's curves
ELITE = {"lr": 1e-4, "ent_coef": 0.005, "snapshot": 0.35, "clip": 0.2}
MUTATIONS = {"lr": (5e-5, 1e-4, 2e-4), "ent_coef": (0.002, 0.005, 0.01), "snapshot": (0.25, 0.35, 0.5),
             "clip": (0.1, 0.2)}
COMPARISONS = (("C1", "U1", "E1", "gen2"), ("C2", "U1", "L1", "gen2"), ("C3", "U1", "L1", "gen3"),
               ("C4", "U2", "L2", "gen3"), ("C5", "U2", "E1", "gen3"), ("C6", "U2", "U0", "gen1_train"))


# ----------------------------------------------------------------------------- helpers

def spec(weights: Weights, name: str) -> dict[str, Any]:
    return {"method": "ppo", "weights": weights, "greedy": True, "name": name}


def heuristic(name: str) -> dict[str, Any]:
    return {"heuristic": name}


def save_weights(path: Path, weights: Weights, meta: dict[str, Any]) -> None:
    model = build_policy("ppo")
    load_numpy(model, weights)
    save_checkpoint(path, model, "ppo", int(meta.get("steps", 0)), meta)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def duel(pool: WorkerPool, agent: dict[str, Any], opponent: dict[str, Any], split: str, battles: int, offset: int,
         seed: int) -> NDArray[np.float64]:
    """Per-battle scores ((outcome + 1) / 2) of ``agent``; odd battles swap sides and teams."""
    tasks = [{"type": "evaluate", "split": split, "indices": [offset + i for i in chunk], "seed": seed,
              "agent": agent, "opponent": opponent} for chunk in split_indices(battles, pool.size)]
    records = sorted((r for result in pool.map(tasks) for r in result), key=lambda r: int(r["index"]))
    return np.asarray([(float(r["outcome"]) + 1) / 2 for r in records], dtype=np.float64)


def bradley_terry(wins: NDArray[np.float64], games: NDArray[np.float64], iterations: int = 500) -> NDArray[np.float64]:
    """Bradley-Terry strengths (MM algorithm, Hunter 2004) on the Elo scale, mean 1500."""
    size = len(wins)
    strength = np.ones(size)
    total = wins.sum(axis=1) + 1e-3  # small prior keeps a proposal with no win finite
    for _ in range(iterations):
        denominator = np.array([sum(games[i, j] / (strength[i] + strength[j]) for j in range(size) if j != i)
                                for i in range(size)])
        strength = total / np.maximum(denominator, 1e-12)
        strength /= np.exp(np.mean(np.log(strength)))
    elo = 400 * np.log10(strength)
    return np.asarray(1500 + elo - elo.mean(), dtype=np.float64)


def tournament(pool: WorkerPool, members: dict[str, Weights], split: str, battles: int, seed: int,
               path: Path) -> dict[str, Any]:
    """Round-robin between all members on validation battles; ratings + score matrix (cached in ``path``)."""
    if path.exists():
        return dict(read_json(path))
    names = list(members)
    size = len(names)
    wins, games = np.zeros((size, size)), np.zeros((size, size))
    for i, j in itertools.combinations(range(size), 2):
        offset = TOURNAMENT_OFFSET + 10_000 * (i * size + j)
        scores = duel(pool, spec(members[names[i]], names[i]), spec(members[names[j]], names[j]), split, battles,
                      offset, seed)
        wins[i, j], wins[j, i] = scores.sum(), len(scores) - scores.sum()
        games[i, j] = games[j, i] = len(scores)
    ratings = bradley_terry(wins, games)
    order = [names[k] for k in np.argsort(-ratings)]
    result = {"split": split, "battles_per_pair": battles, "names": names, "ratings": dict(zip(names, ratings.tolist())),
              "ranking": order, "score": {names[i]: {names[j]: float(wins[i, j] / games[i, j])
                                                     for j in range(size) if j != i} for i in range(size)}}
    write_json(path, result)
    return result


def anchor_score(pool: WorkerPool, weights: Weights, anchors: dict[str, dict[str, Any]], split: str, battles: int,
                 seed: int) -> float:
    """Validation score against the anchor panel (mean over anchors of the mean battle score)."""
    return float(np.mean([duel(pool, spec(weights, "candidate"), anchor, split, battles,
                               TOURNAMENT_OFFSET + 5_000_000 + 10_000 * k, seed).mean()
                          for k, anchor in enumerate(anchors.values())]))


def average(members: list[Weights]) -> Weights:
    return {key: np.mean([m[key] for m in members], axis=0).astype(members[0][key].dtype) for key in members[0]}


def child_hparams(seed: int, generation: int, index: int) -> dict[str, float]:
    if index == 0:
        return dict(ELITE)
    rng = np.random.default_rng(10_000 * seed + 100 * generation + index)
    while True:
        drawn = {key: float(rng.choice(values)) for key, values in MUTATIONS.items()}
        if drawn != ELITE:
            return drawn


def ppo_config(hp: dict[str, float], seed: int, steps: int, generation: int, rivals: int, workers: int,
               device: str) -> PPOConfig:
    snapshot = hp["snapshot"]
    rest = 1.0 - snapshot  # self / greedy / random keep E1's proportions
    mix = {"self": rest * 0.35 / 0.75, "snapshot": snapshot, "greedy": rest * 0.3 / 0.75,
           "random": rest * 0.1 / 0.75}
    return PPOConfig(method="ppo", seed=seed, total_steps=steps, steps_per_iter=min(32_768, max(4096, steps // 8)),
                     workers=workers, lr=hp["lr"], ent_coef=hp["ent_coef"], ent_coef_end=min(0.002, hp["ent_coef"]),
                     clip=hp["clip"], mix=mix, evals=0, lr_warmup=10, generation=generation + 1,
                     rivals=tuple(f"r{i}" for i in range(rivals)), device=device)


# ----------------------------------------------------------------------------- generations

def generation0(pool: WorkerPool, seed: int, args: argparse.Namespace, root: Path) -> dict[str, Any]:
    out = root / "gen0"
    done = out / "generation.json"
    if done.exists():
        return dict(read_json(done))
    population: dict[str, Weights] = {}
    for method in ("ppo", "bc_ppo"):
        for run in range(5):
            path = E1 / method / f"seed{run}" / "final.pt"
            population[f"{method}_s{run}"] = load_weights(path)[1]
    table = tournament(pool, population, "gen1_train", args.battles, seed, out / "tournament.json")
    top = table["ranking"][:3]
    champion = top[0]
    anchors = {"greedy": heuristic("greedy"), "prior": heuristic("prior")}
    distilled_path = out / "distill" / "final.pt"
    if not distilled_path.exists():
        teacher = {"ensemble": [population[name] for name in top], "greedy": True}
        config = BCConfig(method="ppo", seed=seed, demo_steps=args.distill_steps, experts={"teacher": 1.0},
                          opponents={"teacher": 0.4, "greedy": 0.4, "random": 0.2}, epochs=3, lr=1e-4,
                          smoothing=0.05, eval_battles=0, workers=args.workers, generation=1, device=args.device)
        train_bc(config, out / "distill", pool, init_weights=population[champion], teachers={"teacher": teacher})
    distilled = load_weights(distilled_path)[1]
    champion_score = anchor_score(pool, population[champion], anchors, "gen1_train", args.val_battles, seed)
    distilled_score = anchor_score(pool, distilled, anchors, "gen1_train", args.val_battles, seed)
    accepted = distilled_score >= champion_score
    unified = distilled if accepted else population[champion]
    save_weights(out / "unified.pt", unified, {"generation": 0, "merge": "distill" if accepted else "champion"})
    result = {"generation": 0, "split": "gen1_train", "population": list(population), "ranking": table["ranking"],
              "ratings": table["ratings"], "top": top, "champion": champion, "merge": "distillation",
              "champion_val": champion_score, "distilled_val": distilled_score, "accepted": bool(accepted),
              "rivals_next": top}
    write_json(done, result)
    return result


def generation(pool: WorkerPool, g: int, seed: int, args: argparse.Namespace, root: Path,
               previous: dict[str, Any], rival_weights: dict[str, Weights]) -> dict[str, Any]:
    out = root / f"gen{g}"
    done = out / "generation.json"
    if done.exists():
        return dict(read_json(done))
    split = SPLIT_OF[g]
    parent = load_weights(root / f"gen{g - 1}" / "unified.pt")[1]
    rivals = {"parent": parent, **{f"prev_{k}": w for k, w in rival_weights.items()}}
    children: dict[str, Weights] = {}
    hparams: dict[str, dict[str, float]] = {}
    for k in range(args.proposals):
        hp = child_hparams(seed, g, k)
        hparams[f"child{k}"] = hp
        final = out / f"child{k}" / "final.pt"
        if not final.exists():
            config = ppo_config(hp, 1000 * g + 10 * seed + k, args.child_steps, g, len(rivals), args.workers,
                                args.device)
            train_ppo(config, out / f"child{k}", pool, init_weights=parent,
                      rivals={f"r{i}": w for i, w in enumerate(rivals.values())})
        children[f"child{k}"] = load_weights(final)[1]
    lineal_root = root / f"lineal_gen{g}"
    if not (lineal_root / "final.pt").exists():
        lineal_parent = parent if g == 1 else load_weights(root / f"lineal_gen{g - 1}" / "final.pt")[1]
        config = ppo_config(ELITE, 1000 * g + 10 * seed + 9, args.child_steps * args.proposals, g, 0, args.workers,
                            args.device)
        train_ppo(config, lineal_root, pool, init_weights=lineal_parent)
    table = tournament(pool, {"parent": parent, **children}, split, args.battles, seed, out / "tournament.json")
    ranked = [name for name in table["ranking"] if name != "parent"]
    anchors = {"greedy": heuristic("greedy"), "prior": heuristic("prior"), "parent": spec(parent, "parent")}
    soup_members = [ranked[0]]
    best = anchor_score(pool, children[ranked[0]], anchors, split, args.val_battles, seed)
    trials = [{"members": list(soup_members), "val": best, "accepted": True}]
    for name in ranked[1:]:
        candidate = average([children[m] for m in soup_members + [name]])
        score = anchor_score(pool, candidate, anchors, split, args.val_battles, seed)
        accepted = score >= best
        trials.append({"members": soup_members + [name], "val": score, "accepted": bool(accepted)})
        if accepted:
            soup_members.append(name)
            best = score
    unified = average([children[m] for m in soup_members])
    save_weights(out / "unified.pt", unified, {"generation": g, "soup": soup_members})
    result = {"generation": g, "split": split, "hparams": hparams, "ranking": table["ranking"],
              "ratings": table["ratings"], "soup": soup_members, "soup_trials": trials, "merge": "greedy_soup",
              "rivals": list(rivals), "rivals_next": ranked[:3]}
    write_json(done, result)
    return result


def run_seed(pool: WorkerPool, seed: int, args: argparse.Namespace) -> None:
    root = OUT / f"seed{seed}"
    g0 = generation0(pool, seed, args, root)
    population = {}
    for name in g0["rivals_next"]:
        method, run = name.rsplit("_s", 1)
        population[name] = load_weights(E1 / method / f"seed{run}" / "final.pt")[1]
    previous = g0
    for g in (1, 2):
        current = generation(pool, g, seed, args, root, previous, population)
        population = {name: load_weights(root / f"gen{g}" / name / "final.pt")[1] for name in current["rivals_next"]}
        previous = current
    test(pool, seed, args, root)


# ----------------------------------------------------------------------------- test

def test(pool: WorkerPool, seed: int, args: argparse.Namespace, root: Path) -> None:
    path = root / "test.json"
    if path.exists():
        return
    agents = {"E1": E1 / "ppo" / f"seed{seed}" / "final.pt", "U0": root / "gen0" / "unified.pt",
              "U1": root / "gen1" / "unified.pt", "U2": root / "gen2" / "unified.pt",
              "L1": root / "lineal_gen1" / "final.pt", "L2": root / "lineal_gen2" / "final.pt"}
    weights = {name: load_weights(p)[1] for name, p in agents.items()}
    result: dict[str, Any] = {"vs_greedy": {}, "head_to_head": {}, "battles": args.test_battles}
    for name, w in weights.items():
        result["vs_greedy"][name] = {split: duel(pool, spec(w, name), heuristic("greedy"), split, args.test_battles,
                                                 0, 0).tolist() for split in TEST_SPLITS}
    for a, b in (("U2", "E1"), ("U2", "L2"), ("U1", "L1"), ("U1", "E1")):
        result["head_to_head"][f"{a}_vs_{b}"] = {
            split: float(duel(pool, spec(weights[a], a), spec(weights[b], b), split, args.test_battles, 0, 0).mean())
            for split in TEST_SPLITS}
    write_json(path, result)


# ----------------------------------------------------------------------------- summary

def summarize() -> Path:
    seeds = sorted(int(p.name.removeprefix("seed")) for p in OUT.glob("seed*") if (p / "test.json").exists())
    tests = {s: read_json(OUT / f"seed{s}" / "test.json") for s in seeds}
    gens = {s: [read_json(OUT / f"seed{s}" / f"gen{g}" / "generation.json") for g in range(3)] for s in seeds}
    summary: dict[str, Any] = {"seeds": seeds, "vs_greedy": {}, "comparisons": [], "head_to_head": {},
                               "generations": gens}
    if not seeds:
        write_json(OUT / "summary.json", summary)
        return OUT / "summary.json"
    agents = list(tests[seeds[0]]["vs_greedy"])
    for name in agents:
        summary["vs_greedy"][name] = {}
        for split in TEST_SPLITS:
            per_seed = [float(np.mean(tests[s]["vs_greedy"][name][split])) for s in seeds]
            summary["vs_greedy"][name][split] = {"mean": float(np.mean(per_seed)), "per_seed": per_seed,
                                                 "sd": float(np.std(per_seed, ddof=1)) if len(seeds) > 1 else 0.0}
    raw: list[dict[str, Any]] = []
    for label, a, b, split in COMPARISONS:
        first = np.mean([tests[s]["vs_greedy"][a][split] for s in seeds], axis=0)
        second = np.mean([tests[s]["vs_greedy"][b][split] for s in seeds], axis=0)
        diff = first - second
        low, high = bootstrap_mean_ci(diff)
        raw.append({"id": label, "a": a, "b": b, "split": split, "diff": float(diff.mean()), "ci": [low, high],
                    "p": sign_flip_p(diff)})
    for item, adjusted in zip(raw, holm([r["p"] for r in raw])):
        item["p_holm"] = adjusted
    summary["comparisons"] = raw
    for key in tests[seeds[0]]["head_to_head"]:
        summary["head_to_head"][key] = {split: float(np.mean([tests[s]["head_to_head"][key][split] for s in seeds]))
                                        for split in TEST_SPLITS}
    write_json(OUT / "summary.json", summary)
    lines = ["# E5-combate — evolución generacional por torneo, fusión y herencia", "",
             f"Semillas: {seeds}. Puntaje = (victorias + empates/2) / combates de test contra Greedy.", "",
             "| Agente | " + " | ".join(TEST_SPLITS) + " |", "|---|" + "---|" * len(TEST_SPLITS)]
    for name in agents:
        lines.append(f"| {name} | " + " | ".join(f"{summary['vs_greedy'][name][s]['mean']:.3f}" for s in TEST_SPLITS)
                     + " |")
    lines += ["", "| Comparación | Split | Diferencia | IC 95% | p Holm |", "|---|---|---|---|---|"]
    for item in raw:
        lines.append(f"| {item['id']} {item['a']} − {item['b']} | {item['split']} | {item['diff']:+.3f} | "
                     f"[{item['ci'][0]:+.3f}, {item['ci'][1]:+.3f}] | {item['p_holm']:.3g} |")
    lines += ["", "Duelos directos (puntaje del primero):", ""]
    for key, row in summary["head_to_head"].items():
        lines.append(f"- {key}: " + ", ".join(f"{s} {v:.3f}" for s, v in row.items()))
    lines += ["", "Genealogía por semilla:", ""]
    for s in seeds:
        g0, g1, g2 = gens[s]
        lines.append(f"- semilla {s}: G0 campeón {g0['champion']} (top-3 {g0['top']}, destilación "
                     f"{'aceptada' if g0['accepted'] else 'rechazada'}); G1 sopa {g1['soup']}; G2 sopa {g2['soup']}")
    path = OUT / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    protocol = sub.add_parser("protocol")
    protocol.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    protocol.add_argument("--proposals", type=int, default=4)
    protocol.add_argument("--child-steps", type=int, default=500_000)
    protocol.add_argument("--distill-steps", type=int, default=200_000)
    protocol.add_argument("--battles", type=int, default=200, help="validation battles per tournament pair")
    protocol.add_argument("--val-battles", type=int, default=200, help="validation battles per anchor (soup)")
    protocol.add_argument("--test-battles", type=int, default=500)
    protocol.add_argument("--workers", type=int, default=12)
    protocol.add_argument("--device", default="dml")
    protocol.add_argument("--out-dir", default="")
    sub.add_parser("summary")
    args = parser.parse_args(argv)
    global OUT
    if args.command == "protocol":
        if args.out_dir:
            OUT = Path(args.out_dir)
        with WorkerPool(args.workers) as pool:
            for seed in args.seeds:
                run_seed(pool, seed, args)
        summarize()
    else:
        summarize()


if __name__ == "__main__":
    main()
