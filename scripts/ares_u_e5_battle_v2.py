"""E5-combate v2 — selección con potencia estadística y propuestas especializadas por nicho.

Diagnosis of v1 (``scripts.ares_u_e5_battle``), made on its *validation* data: a tournament of 200
battles per pair has a standard error of ~0.035 in score, while the proposals differ by < 0.015
(ratings within +-20 Elo), and the soup was accepted on differences of 0.001-0.003 measured on 3 x 200
battles. Selection and merging were therefore driven by noise (Miller & Goldberg 1995: noisy tournament
selection degrades to random selection), and the children of one parent were near-copies (same
opponent mix, small hyper-parameter changes), so there was little to select or merge.

v2 changes only what the diagnosis points at (generation 0, the lineal control and E1 are reused):

* **Niches (diversity).** The 4 children of a generation train against different opponent mixes:
  ``equilibrado`` (the elite mix), ``anti-heurística`` (60% Greedy), ``self-play`` (60% self) and
  ``anti-rivales`` (60% frozen rivals: parent + top-3 of the previous generation).
* **Fitness with power.** Each candidate plays the same 2000 validation battles against each member of a
  panel (Greedy, ARES prior, random, parent, rivals): ~10^4 paired battles, standard error of a
  difference ~0.003. Ranking = mean fitness. The round-robin tournament (1000 battles per pair) is kept
  for the Elo figure only.
* **Paired merge.** Greedy soup in fitness order; an ingredient is accepted when the paired mean
  difference (same battles) against the current soup is >= 0. The uniform soup of all children is also
  scored, and the better of the two becomes the unified proposal.

Fresh test battles (indices 3000..3000+N-1, never used by v1 or any selection), against Greedy.
Comparisons fixed before running (paired sign-flip on seed-averaged battle scores, Holm over D1-D6):

* D1  U1v2 vs L1 on gen2      D2  U1v2 vs L1 on gen3 (zero-shot)     D3  U2v2 vs L2 on gen3
* D4  U2v2 vs U2v1 on gen3    D5  U2v2 vs U0 on gen1_train (retention) D6  U2v2 vs E1 on gen3

Measurement of the measurement: Spearman correlation between the validation ranking of the children and
their score on the fresh test battles (v1: tournament rating; v2: panel fitness).

    python -m scripts.ares_u_e5_battle_v2 protocol --seeds 0 1 2
    python -m scripts.ares_u_e5_battle_v2 summary
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ares_u.battle.paired import bootstrap_mean_ci, holm, sign_flip_p
from ares_u.battle.ppo import load_weights, train_ppo
from ares_u.battle.rollout import WorkerPool
from scripts.ares_u_e5_battle import (E1, ELITE, OUT as V1, SPLIT_OF, TEST_SPLITS, TOURNAMENT_OFFSET, average, duel,
                                      heuristic, ppo_config, read_json, save_weights, spec, tournament, write_json)

V2 = V1 / "v2"
FITNESS_OFFSET = TOURNAMENT_OFFSET + 20_000_000
TEST_OFFSET = 3000
NICHES: dict[str, tuple[str, dict[str, float] | None]] = {
    "child0": ("equilibrado", None),
    "child1": ("anti-heurística", {"self": 0.15, "snapshot": 0.15, "greedy": 0.6, "random": 0.1}),
    "child2": ("self-play", {"self": 0.6, "snapshot": 0.2, "greedy": 0.15, "random": 0.05}),
    "child3": ("anti-rivales", {"self": 0.15, "snapshot": 0.6, "greedy": 0.2, "random": 0.05}),
}
COMPARISONS = (("D1", "U1v2", "L1", "gen2"), ("D2", "U1v2", "L1", "gen3"), ("D3", "U2v2", "L2", "gen3"),
               ("D4", "U2v2", "U2", "gen3"), ("D5", "U2v2", "U0", "gen1_train"), ("D6", "U2v2", "E1", "gen3"))


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def fitness(pool: WorkerPool, weights: dict[str, Any], panel: dict[str, dict[str, Any]], split: str, battles: int,
            seed: int) -> NDArray[np.float64]:
    """Per-battle scores against every panel member (same battles for every candidate: paired)."""
    return np.concatenate([duel(pool, spec(weights, "candidate"), opponent, split, battles,
                                FITNESS_OFFSET + 100_000 * k, seed) for k, opponent in enumerate(panel.values())])


def generation(pool: WorkerPool, g: int, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    root, v1 = V2 / f"seed{seed}", V1 / f"seed{seed}"
    out = root / f"gen{g}"
    done = out / "generation.json"
    if done.exists():
        return dict(read_json(done))
    split = SPLIT_OF[g]
    if g == 1:
        parent = load_weights(v1 / "gen0" / "unified.pt")[1]
        top = read_json(v1 / "gen0" / "generation.json")["rivals_next"]
        previous = {n: load_weights(E1 / n.rsplit("_s", 1)[0] / f"seed{n.rsplit('_s', 1)[1]}" / "final.pt")[1] for n in top}
    else:
        parent = load_weights(root / "gen1" / "unified.pt")[1]
        top = read_json(root / "gen1" / "generation.json")["rivals_next"]
        previous = {n: load_weights(root / "gen1" / n / "final.pt")[1] for n in top}
    rivals = {"parent": parent, **{f"prev_{k}": w for k, w in previous.items()}}
    children: dict[str, dict[str, Any]] = {}
    for k, (name, (niche, mix)) in enumerate(NICHES.items()):
        final = out / name / "final.pt"
        if not final.exists():
            config = ppo_config(ELITE, 2000 * g + 10 * seed + k, args.child_steps, g, len(rivals), args.workers,
                                args.device)
            if mix is not None:
                config.mix = dict(mix)
            train_ppo(config, out / name, pool, init_weights=parent,
                      rivals={f"r{i}": w for i, w in enumerate(rivals.values())})
        children[name] = load_weights(final)[1]
    panel = {"greedy": heuristic("greedy"), "prior": heuristic("prior"), "random": heuristic("random"),
             **{name: spec(w, name) for name, w in rivals.items()}}
    scores = {name: fitness(pool, w, panel, split, args.val_battles, seed) for name, w in children.items()}
    parent_fit = fitness(pool, parent, panel, split, args.val_battles, seed)
    ranked = sorted(children, key=lambda n: -scores[n].mean())
    soup, current = [ranked[0]], scores[ranked[0]]
    trials = [{"members": [ranked[0]], "fitness": float(current.mean()), "accepted": True}]
    for name in ranked[1:]:
        candidate = fitness(pool, average([children[m] for m in soup + [name]]), panel, split, args.val_battles, seed)
        diff = candidate - current
        accepted = bool(diff.mean() >= 0)
        trials.append({"members": soup + [name], "fitness": float(candidate.mean()), "paired_diff": float(diff.mean()),
                       "paired_se": float(diff.std(ddof=1) / np.sqrt(len(diff))), "accepted": accepted})
        if accepted:
            soup.append(name)
            current = candidate
    uniform = fitness(pool, average(list(children.values())), panel, split, args.val_battles, seed)
    use_uniform = bool(uniform.mean() > current.mean())
    members = list(children) if use_uniform else soup
    save_weights(out / "unified.pt", average([children[m] for m in members]), {"generation": g, "soup": members})
    table = tournament(pool, {"parent": parent, **children}, split, args.battles, seed, out / "tournament.json")
    result = {"generation": g, "split": split, "niches": {n: NICHES[n][0] for n in children},
              "fitness": {n: float(scores[n].mean()) for n in children}, "parent_fitness": float(parent_fit.mean()),
              "fitness_se": {n: float(scores[n].std(ddof=1) / np.sqrt(len(scores[n]))) for n in children},
              "ranking": ranked, "soup": members, "soup_trials": trials, "greedy_soup_fitness": float(current.mean()),
              "uniform_soup_fitness": float(uniform.mean()), "uniform_chosen": use_uniform,
              "ratings": table["ratings"], "panel": list(panel), "rivals_next": ranked[:3]}
    write_json(done, result)
    return result


def test(pool: WorkerPool, seed: int, args: argparse.Namespace) -> None:
    root, v1 = V2 / f"seed{seed}", V1 / f"seed{seed}"
    path = root / "test.json"
    if path.exists():
        return
    agents = {"E1": E1 / "ppo" / f"seed{seed}" / "final.pt", "U0": v1 / "gen0" / "unified.pt",
              "U1": v1 / "gen1" / "unified.pt", "U2": v1 / "gen2" / "unified.pt",
              "U1v2": root / "gen1" / "unified.pt", "U2v2": root / "gen2" / "unified.pt",
              "L1": v1 / "lineal_gen1" / "final.pt", "L2": v1 / "lineal_gen2" / "final.pt"}
    result: dict[str, Any] = {"battles": args.test_battles, "offset": TEST_OFFSET, "vs_greedy": {}, "children": {}}
    for name, file in agents.items():
        w = load_weights(file)[1]
        result["vs_greedy"][name] = {split: duel(pool, spec(w, name), heuristic("greedy"), split, args.test_battles,
                                                 TEST_OFFSET, 0).tolist() for split in TEST_SPLITS}
    for version, base in (("v1", v1), ("v2", root)):  # selection reliability: children on their own generation
        for g in (1, 2):
            gen = read_json(base / f"gen{g}" / "generation.json")
            names = [n for n in (gen["ratings"] if version == "v1" else gen["fitness"]) if n != "parent"]
            entry = {}
            for name in names:
                w = load_weights(base / f"gen{g}" / name / "final.pt")[1]
                entry[name] = {"test": float(duel(pool, spec(w, name), heuristic("greedy"), SPLIT_OF[g],
                                                  args.test_battles, TEST_OFFSET, 0).mean()),
                               "val": float(gen["ratings"][name] if version == "v1" else gen["fitness"][name])}
            result["children"][f"{version}_gen{g}"] = entry
    write_json(path, result)


def summarize() -> Path:
    seeds = sorted(int(p.name.removeprefix("seed")) for p in V2.glob("seed*") if (p / "test.json").exists())
    summary: dict[str, Any] = {"seeds": seeds, "vs_greedy": {}, "comparisons": [], "reliability": {},
                               "generations": {s: [read_json(V2 / f"seed{s}" / f"gen{g}" / "generation.json")
                                                   for g in (1, 2)] for s in seeds}}
    if not seeds:
        write_json(V2 / "summary.json", summary)
        return V2 / "summary.json"
    tests = {s: read_json(V2 / f"seed{s}" / "test.json") for s in seeds}
    for name in tests[seeds[0]]["vs_greedy"]:
        summary["vs_greedy"][name] = {}
        for split in TEST_SPLITS:
            battles = np.mean([tests[s]["vs_greedy"][name][split] for s in seeds], axis=0)
            per_seed = [float(np.mean(tests[s]["vs_greedy"][name][split])) for s in seeds]
            low, high = bootstrap_mean_ci(battles)
            summary["vs_greedy"][name][split] = {"mean": float(battles.mean()), "ci": [low, high], "per_seed": per_seed}
    raw: list[dict[str, Any]] = []
    for label, a, b, split in COMPARISONS:
        diff = (np.mean([tests[s]["vs_greedy"][a][split] for s in seeds], axis=0)
                - np.mean([tests[s]["vs_greedy"][b][split] for s in seeds], axis=0))
        low, high = bootstrap_mean_ci(diff)
        raw.append({"id": label, "a": a, "b": b, "split": split, "diff": float(diff.mean()), "ci": [low, high],
                    "p": sign_flip_p(diff), "differing": float(np.mean(diff != 0))})
    for item, adjusted in zip(raw, holm([r["p"] for r in raw])):
        item["p_holm"] = adjusted
    summary["comparisons"] = raw
    for version in ("v1", "v2"):
        rhos = []
        for s in seeds:
            for g in (1, 2):
                entry = tests[s]["children"][f"{version}_gen{g}"]
                rhos.append(spearman([e["val"] for e in entry.values()], [e["test"] for e in entry.values()]))
        finite = [r for r in rhos if np.isfinite(r)]
        summary["reliability"][version] = {"per_generation": rhos, "mean": float(np.mean(finite)) if finite else None}
    write_json(V2 / "summary.json", summary)
    lines = ["# E5-combate v2 — selección con potencia y nichos", "",
             f"Semillas {seeds}; {tests[seeds[0]]['battles']} combates de test nuevos por split (índices desde "
             f"{TEST_OFFSET}).", "", "| Agente | " + " | ".join(TEST_SPLITS) + " |", "|---|---|---|---|"]
    for name, row in summary["vs_greedy"].items():
        lines.append(f"| {name} | " + " | ".join(f"{row[s]['mean']:.3f}" for s in TEST_SPLITS) + " |")
    lines += ["", "| Comparación | Split | Dif. | IC 95% | p Holm | combates distintos |", "|---|---|---|---|---|---|"]
    for r in raw:
        lines.append(f"| {r['id']} {r['a']} − {r['b']} | {r['split']} | {r['diff']:+.4f} | [{r['ci'][0]:+.4f}, "
                     f"{r['ci'][1]:+.4f}] | {r['p_holm']:.3g} | {r['differing']:.2f} |")
    lines += ["", "Confiabilidad de la selección (Spearman validación vs test de los hijos):"]
    for version, rel in summary["reliability"].items():
        lines.append(f"- {version}: media {rel['mean']}, por generación {[round(r, 2) for r in rel['per_generation']]}")
    path = V2 / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    protocol = sub.add_parser("protocol")
    protocol.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    protocol.add_argument("--child-steps", type=int, default=750_000)
    protocol.add_argument("--battles", type=int, default=1000, help="tournament battles per pair (Elo figure)")
    protocol.add_argument("--val-battles", type=int, default=2000, help="validation battles per panel member")
    protocol.add_argument("--test-battles", type=int, default=2000)
    protocol.add_argument("--workers", type=int, default=12)
    protocol.add_argument("--device", default="dml")
    protocol.add_argument("--out-dir", default="", help="smoke tests only")
    sub.add_parser("summary")
    args = parser.parse_args(argv)
    global V2
    if args.command == "protocol" and args.out_dir:
        V2 = Path(args.out_dir)
    if args.command == "protocol":
        with WorkerPool(args.workers) as pool:
            for seed in args.seeds:
                for g in (1, 2):
                    generation(pool, g, seed, args)
                test(pool, seed, args)
    summarize()


if __name__ == "__main__":
    main()
