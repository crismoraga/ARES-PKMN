"""ARES-Battle (E1, IPD434): train, evaluate and run the cross-generation transfer protocol.

Métodos:
  bc       clonación conductual (imitación) del prior ARES y de Greedy en equipos Gen1 de entrenamiento; sin RL.
  ppo      PPO + self-play/liga desde cero (Transformer de entidades sin IDs).
  bc_ppo   bc como inicialización y luego PPO; las demostraciones cuentan dentro del mismo presupuesto de pasos.
  ppo_ids  ablación: el mismo PPO con embeddings de ID de especie.
  dqn      línea base del curso: DQN-MLP sobre features planas, mismo presupuesto de pasos.

Comandos:
  train      entrena un método/semilla y escribe metrics.jsonl, config.json y checkpoints.
  evaluate   evalúa un checkpoint contra Random/Greedy/prior en gen1_train, gen1_heldout, gen2 y gen3 (JSON).
  baselines  matriz línea base vs línea base por split (JSON).
  protocol   todos los métodos x semillas (secuencial o --parallel K), evaluación, baselines y resumen.
             Reanudable: se salta cada método/semilla con final.pt y eval.json completos.
  status     progreso de una corrida del protocolo (qué método/semilla está terminado).
  headroom   techo de habilidad: planificador Monte-Carlo con información completa vs Greedy (solo análisis).
  summarize  agrega una corrida del protocolo: summary.md, summary.json y figuras PNG.
  bench      mide pasos de entorno/s con N workers y el tiempo de actualización del learner (dml vs cpu).

Ejemplos:
  python -m scripts.ares_u_battle train --method bc_ppo --seed 0 --steps 3000000 --device dml --workers 10 \\
      --out artifacts/ares_u/battle/pilot2/bc_ppo/seed0
  python -m scripts.ares_u_battle evaluate --checkpoint artifacts/ares_u/battle/pilot2/ppo/seed0/final.pt --battles 500
  python -m scripts.ares_u_battle protocol --run final --steps 5000000 --seeds 0 1 2 3 4 --workers 10 --battles 500
  python -m scripts.ares_u_battle status --run final
  python -m scripts.ares_u_battle summarize --run final
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from core.game_data import ROOT

ARTIFACTS = ROOT / "artifacts" / "ares_u" / "battle"
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 2)
METHODS = ("bc", "ppo", "bc_ppo", "ppo_ids", "dqn")
RL_METHODS = ("ppo", "bc_ppo", "ppo_ids", "dqn")
PPO_KEYS = ("lr", "lr_end_frac", "ent_coef", "ent_coef_end", "epochs", "steps_per_iter", "minibatch", "target_kl",
            "evals", "eval_battles", "pipeline")
BC_KEYS = {"demo_steps": "demo_steps", "bc_epochs": "epochs", "bc_lr": "lr", "bc_smoothing": "smoothing",
           "eval_battles": "eval_battles"}


def _apply(config: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)


def _bc_config(seed: int, device: str, workers: int, overrides: dict[str, Any]) -> Any:
    from ares_u.battle.bc import BCConfig

    config = BCConfig(seed=seed, device=device, workers=workers)
    _apply(config, {field: overrides.get(key) for key, field in BC_KEYS.items()})
    return config


def _shared_bc(run: Path | None, seed: int, config: Any) -> Path | None:
    """The ``bc`` run of the same seed when its demonstrations/BC settings equal ``config``'s."""
    if run is None:
        return None
    from ares_u.battle.ppo import load_weights

    candidate = run / "bc" / f"seed{seed}" / "final.pt"
    if not candidate.is_file():
        return None
    _, _, meta = load_weights(candidate)
    signature = json.loads(json.dumps(config.demo_signature()))
    return candidate if meta.get("signature") == signature else None


def _train(method: str, seed: int, steps: int, device: str, workers: int, out: Path,
           overrides: dict[str, Any], pool: Any = None, run: Path | None = None) -> Path:
    """Train one method/seed. ``steps`` is the environment-step budget of every RL method; ``bc``
    uses only its demonstrations and ``bc_ppo`` spends its demonstrations out of ``steps``."""
    if method == "dqn":
        from ares_u.battle.dqn import DQNConfig, train_dqn

        dqn_config = DQNConfig(seed=seed, total_steps=steps, device=device, workers=workers)
        _apply(dqn_config, {key: overrides.get(key) for key in ("evals", "eval_battles", "pipeline")})
        return train_dqn(dqn_config, out, pool)
    if method == "bc":
        from ares_u.battle.bc import train_bc

        return train_bc(_bc_config(seed, device, workers, overrides), out, pool)
    from ares_u.battle.ppo import PPOConfig, load_weights, train_ppo

    ppo_config = PPOConfig(method=method, seed=seed, total_steps=steps, device=device, workers=workers)
    _apply(ppo_config, {key: overrides.get(key) for key in PPO_KEYS})
    init_weights = None
    if method == "bc_ppo":
        from ares_u.battle.bc import train_bc

        bc_config = _bc_config(seed, device, workers, overrides)
        bc_final = _shared_bc(run, seed, bc_config)
        if bc_final is None:
            bc_final = train_bc(bc_config, out / "bc", pool)
        _, init_weights, meta = load_weights(bc_final)
        ppo_config.step_offset = int(meta["demo_steps"])
        ppo_config.init = str(bc_final)
    return train_ppo(ppo_config, out, pool, init_weights)


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    keys = (*PPO_KEYS, *BC_KEYS)
    values = {key: getattr(args, key, None) for key in keys}
    if getattr(args, "no_pipeline", False):
        values["pipeline"] = False
    return values


def _evaluate(checkpoint: Path, battles: int, workers: int, out: Path, greedy: bool, pool: Any = None) -> Path:
    from ares_u.battle.evaluate import agent_spec, evaluate_agent, write_json
    from ares_u.battle.rollout import WorkerPool

    owned = pool is None
    active = pool if pool is not None else WorkerPool(workers)
    try:
        started = time.perf_counter()
        result = evaluate_agent(active, agent_spec(checkpoint, greedy=greedy), battles)
        result["eval_wall_s"] = time.perf_counter() - started
        result["policy_mode"] = "argmax" if greedy else "sample"
    finally:
        if owned:
            active.close()
    return write_json(out, result)


def _baselines(battles: int, workers: int, out: Path, pool: Any = None) -> Path:
    from ares_u.battle.evaluate import baseline_matrix, write_json
    from ares_u.battle.rollout import WorkerPool

    owned = pool is None
    active = pool if pool is not None else WorkerPool(workers)
    try:
        return write_json(out, baseline_matrix(active, battles))
    finally:
        if owned:
            active.close()


def cmd_train(args: argparse.Namespace) -> None:
    out = Path(args.out) if args.out else ARTIFACTS / "runs" / args.method / f"seed{args.seed}"
    final = _train(args.method, args.seed, args.steps, args.device, args.workers, out, _overrides(args))
    print(json.dumps({"checkpoint": str(final)}))
    if args.eval_battles_final:
        path = _evaluate(final, args.eval_battles_final, args.workers, out / "eval.json", not args.sample)
        print(json.dumps({"eval": str(path)}))


def cmd_evaluate(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    out = Path(args.out) if args.out else checkpoint.parent / "eval.json"
    path = _evaluate(checkpoint, args.battles, args.workers, out, not args.sample)
    print(path.read_text(encoding="utf-8") if args.verbose else json.dumps({"eval": str(path)}))


def cmd_baselines(args: argparse.Namespace) -> None:
    out = Path(args.out) if args.out else ARTIFACTS / "baselines.json"
    print(json.dumps({"baselines": str(_baselines(args.battles, args.workers, out))}))


def job_state(run: Path, method: str, seed: int, battles: int) -> str:
    """``done`` (final.pt + complete eval.json), ``trained`` (final.pt only) or ``todo``."""
    from ares_u.battle.evaluate import evaluation_complete

    out = run / method / f"seed{seed}"
    if not (out / "final.pt").is_file():
        return "todo"
    return "done" if evaluation_complete(out / "eval.json", battles) else "trained"


def _run_seed(run: Path, method: str, seed: int, args: argparse.Namespace, pool: Any) -> None:
    out = run / method / f"seed{seed}"
    state = job_state(run, method, seed, args.battles)
    if state == "done":
        print(f"[skip] {method} seed{seed}: final.pt and a complete eval.json exist", flush=True)
        return
    started = time.perf_counter()
    final = out / "final.pt"
    if state == "todo":  # an interrupted training restarts from scratch (metrics.jsonl is rewritten)
        final = _train(method, seed, args.steps, args.device, args.workers, out, _overrides(args), pool, run)
    _evaluate(final, args.battles, args.workers, out / "eval.json", not args.sample, pool)
    print(f"[done] {method} seed{seed} in {time.perf_counter() - started:.0f}s", flush=True)


def _write_protocol(run: Path, args: argparse.Namespace) -> None:
    record = {key: value for key, value in vars(args).items() if key != "func"}
    record["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
    record["budget_note"] = ("steps = environment-step budget (recorded decisions) of every RL method; bc_ppo spends "
                             "its demonstrations out of it; bc uses only its demonstrations")
    with (run / "protocol_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    (run / "protocol.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


def cmd_protocol(args: argparse.Namespace) -> None:
    from ares_u.battle.evaluate import evaluation_complete
    from ares_u.battle.report import summarize_run
    from ares_u.battle.rollout import WorkerPool

    run = ARTIFACTS / args.run
    run.mkdir(parents=True, exist_ok=True)
    _write_protocol(run, args)
    jobs = [(method, seed) for seed in args.seeds for method in args.methods]
    started = time.perf_counter()
    if args.parallel > 1:
        workers = max(1, args.workers // args.parallel)
        pending = [job for job in jobs if job_state(run, *job, args.battles) != "done"]
        running: list[subprocess.Popen[bytes]] = []
        while pending or running:
            while pending and len(running) < args.parallel:
                method, seed = pending.pop(0)
                command = [sys.executable, "-m", "scripts.ares_u_battle", "run-seed", "--run", args.run,
                           "--method", method, "--seed", str(seed), "--steps", str(args.steps),
                           "--device", args.device, "--workers", str(workers), "--battles", str(args.battles)]
                if args.sample:
                    command.append("--sample")
                if args.no_pipeline:
                    command.append("--no-pipeline")
                for key, value in _overrides(args).items():
                    if value is not None and key != "pipeline":
                        command += ["--" + key.replace("_", "-"), str(value)]
                log = (run / f"{method}_seed{seed}.log").open("ab")
                running.append(subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
            time.sleep(5)
            finished = [process for process in running if process.poll() is not None]
            for process in finished:
                if process.returncode:
                    print(f"[warn] job exited with {process.returncode}: {process.args!r}", flush=True)
                running.remove(process)
        if not evaluation_complete(run / "baselines.json", args.battles):
            _baselines(args.battles, args.workers, run / "baselines.json")
    else:
        with WorkerPool(args.workers) as pool:
            if not evaluation_complete(run / "baselines.json", args.battles):
                _baselines(args.battles, args.workers, run / "baselines.json", pool)
            for method, seed in jobs:
                _run_seed(run, method, seed, args, pool)
                print(f"[progress] {sum(job_state(run, *job, args.battles) == 'done' for job in jobs)}/{len(jobs)} "
                      f"jobs done, {time.perf_counter() - started:.0f}s elapsed", flush=True)
    summary = summarize_run(run)
    print(json.dumps({"summary": str(run / "summary.md"), "figures": summary["figures"]}))


def cmd_run_seed(args: argparse.Namespace) -> None:
    from ares_u.battle.rollout import WorkerPool

    run = ARTIFACTS / args.run
    with WorkerPool(args.workers) as pool:
        _run_seed(run, args.method, args.seed, args, pool)


def cmd_status(args: argparse.Namespace) -> None:
    run = Path(args.run) if Path(args.run).is_dir() else ARTIFACTS / args.run
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8")) if (run / "protocol.json").is_file() \
        else {}
    battles = int(protocol.get("battles", args.battles))
    methods = protocol.get("methods", list(METHODS))
    seeds = protocol.get("seeds", [0, 1, 2, 3, 4])
    rows = {method: {seed: job_state(run, method, int(seed), battles) for seed in seeds} for method in methods}
    done = sum(state == "done" for row in rows.values() for state in row.values())
    print(json.dumps({"run": str(run), "done": done, "total": len(methods) * len(seeds), "jobs": rows}, indent=2))


def cmd_summarize(args: argparse.Namespace) -> None:
    from ares_u.battle.report import summarize_run

    run = Path(args.run) if Path(args.run).is_dir() else ARTIFACTS / args.run
    summary = summarize_run(run)
    print((run / "summary.md").read_text(encoding="utf-8"))
    print(json.dumps({"figures": summary["figures"]}))


def cmd_headroom(args: argparse.Namespace) -> None:
    """Skill ceiling probe: full-information Monte-Carlo lookahead and Greedy self-play pair statistics."""
    from ares_u.battle.evaluate import play, summarise, write_json
    from ares_u.battle.headroom import pair_statistics
    from ares_u.battle.rollout import WorkerPool

    result: dict[str, Any] = {"battles": args.battles, "rollouts": args.rollouts, "splits": {}}
    with WorkerPool(args.workers) as pool:
        for split in args.splits:
            entry: dict[str, Any] = {}
            mirror = play(pool, {"heuristic": "greedy"}, {"heuristic": "greedy"}, split, args.battles)
            entry["greedy_vs_greedy_pairs"] = pair_statistics(mirror)
            for opponent in args.opponents:
                started = time.perf_counter()
                records = play(pool, {"lookahead": args.rollouts, "name": "lookahead"}, {"heuristic": opponent},
                               split, args.battles)
                entry[f"lookahead_vs_{opponent}"] = {**summarise(records), "pairs": pair_statistics(records),
                                                     "wall_s": time.perf_counter() - started}
            result["splits"][split] = entry
            print(json.dumps({split: {key: value.get("score", value) for key, value in entry.items()}}), flush=True)
    out = Path(args.out) if args.out else ARTIFACTS / "headroom" / f"lookahead_k{args.rollouts}_n{args.battles}.json"
    print(json.dumps({"headroom": str(write_json(out, result))}))


def cmd_bench(args: argparse.Namespace) -> None:
    from ares_u.battle.bench import learner_benchmark, rollout_benchmark

    result: dict[str, Any] = {"rollout": [rollout_benchmark(workers, args.steps) for workers in args.worker_counts]}
    if args.learner:
        result["learner"] = learner_benchmark(args.devices, args.batch)
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def hyper(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--lr", type=float)
        sub.add_argument("--lr-end-frac", type=float, help="final LR = lr * this (linear annealing)")
        sub.add_argument("--ent-coef", type=float)
        sub.add_argument("--ent-coef-end", type=float, help="entropy coefficient at the end (linear annealing)")
        sub.add_argument("--epochs", type=int)
        sub.add_argument("--steps-per-iter", type=int)
        sub.add_argument("--minibatch", type=int)
        sub.add_argument("--target-kl", type=float, help="PPO early stop when a minibatch KL exceeds it (0 = off)")
        sub.add_argument("--evals", type=int)
        sub.add_argument("--eval-battles", type=int, help="battles per learning-curve point (per split/opponent)")
        sub.add_argument("--no-pipeline", action="store_true", help="do not overlap rollouts with learning")
        sub.add_argument("--demo-steps", type=int, help="bc/bc_ppo: demonstration decisions (count toward budget)")
        sub.add_argument("--bc-epochs", type=int)
        sub.add_argument("--bc-lr", type=float)
        sub.add_argument("--bc-smoothing", type=float)

    train = commands.add_parser("train")
    train.add_argument("--method", choices=METHODS, default="ppo")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--steps", type=int, default=2_000_000)
    train.add_argument("--device", choices=("dml", "cpu"), default="dml")
    train.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    train.add_argument("--out")
    train.add_argument("--eval-battles-final", type=int, default=0)
    train.add_argument("--sample", action="store_true", help="evaluate the sampled policy instead of argmax")
    hyper(train)
    train.set_defaults(func=cmd_train)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--battles", type=int, default=500)
    evaluate.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    evaluate.add_argument("--out")
    evaluate.add_argument("--sample", action="store_true")
    evaluate.add_argument("--verbose", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)

    baselines = commands.add_parser("baselines")
    baselines.add_argument("--battles", type=int, default=500)
    baselines.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    baselines.add_argument("--out")
    baselines.set_defaults(func=cmd_baselines)

    for name, function in (("protocol", cmd_protocol), ("run-seed", cmd_run_seed)):
        sub = commands.add_parser(name)
        sub.add_argument("--run", default="main")
        sub.add_argument("--steps", type=int, default=6_000_000, help="env-step budget per RL run")
        sub.add_argument("--device", choices=("dml", "cpu"), default="dml")
        sub.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
        sub.add_argument("--battles", type=int, default=500, help="final evaluation battles per split x opponent")
        sub.add_argument("--sample", action="store_true")
        if name == "protocol":
            sub.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
            sub.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
            sub.add_argument("--parallel", type=int, default=1)
        else:
            sub.add_argument("--method", choices=METHODS, required=True)
            sub.add_argument("--seed", type=int, required=True)
        hyper(sub)
        sub.set_defaults(func=function)

    status = commands.add_parser("status")
    status.add_argument("--run", required=True)
    status.add_argument("--battles", type=int, default=500)
    status.set_defaults(func=cmd_status)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--run", required=True)
    summarize.set_defaults(func=cmd_summarize)

    headroom = commands.add_parser("headroom", help="skill-ceiling probe (full-information MC lookahead vs Greedy)")
    headroom.add_argument("--battles", type=int, default=200)
    headroom.add_argument("--rollouts", type=int, default=8)
    headroom.add_argument("--splits", nargs="+", default=["gen1_heldout", "gen3"])
    headroom.add_argument("--opponents", nargs="+", default=["greedy", "prior"])
    headroom.add_argument("--workers", type=int, default=4)
    headroom.add_argument("--out")
    headroom.set_defaults(func=cmd_headroom)

    bench = commands.add_parser("bench")
    bench.add_argument("--worker-counts", nargs="+", type=int, default=[1, 4, 8, 14])
    bench.add_argument("--steps", type=int, default=40_000)
    bench.add_argument("--learner", action="store_true")
    bench.add_argument("--devices", nargs="+", default=["dml", "cpu"])
    bench.add_argument("--batch", type=int, default=32_768)
    bench.set_defaults(func=cmd_bench)
    return parser


def main(argv: list[str] | None = None) -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):  # summaries contain non-ASCII text (Windows consoles are cp1252)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
