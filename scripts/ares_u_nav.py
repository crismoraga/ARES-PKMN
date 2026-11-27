"""ARES-Nav (IPD434 experiment E2) command line.

Commands:
  prepare-states  derive the fixed post-intro start savestates and the FireRed AE state pool (provenance JSON)
  bench           raw emulator frames/s, vectorised env steps/s and learner update time
  collect         sticky-random frame collection (for AE pretraining), optionally from a pool of start states
  pretrain        conv autoencoder pretraining on collected frames
  train           PPO + RND + episodic frame-key novelty (or a uniform-random run with --random-policy)
  evaluate        fixed-length evaluation episodes, unique (map, x, y) cells per episode (RAM = metric only)
  protocol        full multi-seed protocol, resumable: per seed FR frames -> AE -> FR PPO, then Emerald
                  scratch / transfer / random runs and evaluation of random / scratch / transfer / zero-shot
  summary         tables, statistics and matplotlib figures for a protocol directory

Heavy modules (torch) are imported lazily inside the commands so that the spawned rollout
workers, which re-import this file as ``__mp_main__``, stay torch-free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import IO, Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NAV = ROOT / "artifacts" / "ares_u" / "nav"
FINAL = NAV / "final"
EMERALD_METHODS: tuple[str, ...] = ("random", "zero_shot", "scratch", "transfer", "ae_only")
TRAIN_KINDS: tuple[str, ...] = ("emerald_scratch", "emerald_transfer", "emerald_random_run", "emerald_ae_only")


def _env_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--action-repeat", type=int, default=24)
    parser.add_argument("--episode-steps", type=int, default=1024)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--start-states", nargs="*", default=[],
                        help="pool of start savestates (default: the game's fixed start state)")


def _env_config(game: str, args: argparse.Namespace) -> Any:
    from ares_u.nav.env import NavConfig
    return NavConfig(game, action_repeat=args.action_repeat, episode_steps=args.episode_steps,
                     frame_stack=args.frame_stack, noop_max=args.noop_max,
                     start_states=tuple(str(p) for p in args.start_states))


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str), flush=True)


# --------------------------------------------------------------------------- commands

def cmd_prepare_states(args: argparse.Namespace) -> None:
    from ares_u.nav.states import prepare_ae_pool, prepare_all
    _print({"start_states": prepare_all(force=args.force), "ae_pool": prepare_ae_pool(force=args.force)})


def cmd_bench(args: argparse.Namespace) -> None:
    from ares_u.nav.env import NavConfig, bench_env_steps, bench_raw_frames
    result: dict[str, Any] = {"raw": {}, "env": [], "machine_note": args.note}
    for game in args.games:
        result["raw"][game] = bench_raw_frames(NavConfig(game), args.raw_frames)
        _print({game: result["raw"][game]})
    for game in args.games:
        for repeat in args.ks:
            for num_envs in args.ns:
                row = {"game": game, "action_repeat": repeat,
                       **bench_env_steps(NavConfig(game, action_repeat=repeat), num_envs, args.steps)}
                result["env"].append(row)
                _print(row)
    if args.learner:
        # A separate process keeps the DML learner benchmark isolated (first backward on DML).
        code = ("import json; from ares_u.nav.train import bench_learner; "
                f"print(json.dumps(bench_learner({args.device!r}, {max(args.ns)})))")
        output = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        result["learner"] = json.loads(output.stdout.strip().splitlines()[-1])
        _print({"learner": result["learner"]})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")


def cmd_collect(args: argparse.Namespace) -> None:
    from ares_u.nav.train import collect_frames
    if args.ae_pool:
        from ares_u.nav.states import ae_pool_paths
        args.start_states = [str(p) for p in ae_pool_paths()]
    _print(collect_frames(_env_config(args.game, args), args.num_envs, args.steps, args.seed, Path(args.out),
                          args.repeat_prob))


def cmd_pretrain(args: argparse.Namespace) -> None:
    from ares_u.nav.train import AEConfig, pretrain_autoencoder
    _print(pretrain_autoencoder(AEConfig(args.frames, args.out, args.eval_frames, args.epochs, args.batch_size,
                                         args.lr, seed=args.seed, device=args.device)))


def cmd_train(args: argparse.Namespace) -> None:
    from ares_u.nav.train import TrainConfig, train_ppo_rnd
    config = TrainConfig(game=args.game, out_dir=args.out_dir, total_steps=args.steps, num_envs=args.num_envs,
                         rollout_len=args.rollout_len, lr=args.lr, ent_coef=args.ent_coef,
                         rnd_coef=args.rnd_coef, epi_coef=args.epi_coef, reward_norm_decay=args.reward_norm_decay,
                         hash_rows=args.hash_rows, hash_cols=args.hash_cols, hash_levels=args.hash_levels,
                         center_rnd=args.center_rnd, target_kl=args.target_kl,
                         init=args.init, random_policy=args.random_policy, seed=args.seed, device=args.device,
                         max_minutes=args.max_minutes, resume=args.resume, checkpoint_every=args.checkpoint_every,
                         env=_env_config(args.game, args))
    _print(train_ppo_rnd(config))


def cmd_evaluate(args: argparse.Namespace) -> None:
    from ares_u.nav.train import EvalConfig, evaluate
    result = evaluate(EvalConfig(args.game, args.out, args.method, args.policy, args.episodes, args.num_envs,
                                 args.seed, args.device, env=_env_config(args.game, args)))
    _print({k: v for k, v in result.items() if k not in ("curve_mean", "curve_std", "curve_steps")})


# --------------------------------------------------------------------------- protocol

class ProtocolLock:
    """Exclusive OS-level lock on ``path`` (released automatically if the process dies)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: IO[str] | None = None

    def __enter__(self) -> ProtocolLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            if sys.platform == "win32":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            raise RuntimeError(f"Another protocol process holds {self.path}") from error
        handle.truncate(0)
        handle.write(f"pid {os.getpid()} since {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.flush()
        self._handle = handle
        return self

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None,
                 trace: TracebackType | None) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _run_phase(name: str, argv: list[str], marker: Path, log_dir: Path, retries: int = 2,
               retry_wait_s: float = 60.0) -> float:
    """Run one phase in a fresh interpreter unless ``marker`` exists; returns minutes spent.

    A failed attempt (e.g. a transient out-of-memory while other experiments share the machine) is
    retried; training phases get ``--resume`` and continue from their last checkpoint."""
    if marker.exists():
        print(f"[skip] {name}: {_rel(marker)} exists", flush=True)
        return 0.0
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    start = time.perf_counter()
    code = -1
    for attempt in range(retries + 1):
        print(f"[run ] {name} (attempt {attempt + 1}) -> {_rel(log_path)}  ({time.strftime('%H:%M:%S')})",
              flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} attempt {attempt + 1}: {' '.join(argv)}\n")
            log.flush()
            code = subprocess.call([sys.executable, str(Path(__file__).resolve()), *argv], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT)
        if code == 0 and marker.exists():
            minutes = (time.perf_counter() - start) / 60
            print(f"[done] {name} in {minutes:.1f} min", flush=True)
            return minutes
        print(f"[fail] {name} exit {code}; see {_rel(log_path)}", flush=True)
        if attempt < retries:
            time.sleep(retry_wait_s)
    raise RuntimeError(f"Phase {name} failed {retries + 1} times (last exit {code}); see {log_path}")


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


_BUDGET_KEYS = ("seeds", "steps_fr", "steps_em", "collect_steps", "collect_episode_steps", "ae_epochs",
                "eval_episodes", "fr_eval_episodes", "num_envs", "random_run", "ae_only", "rnd_coef", "epi_coef",
                "action_repeat", "episode_steps", "frame_stack", "noop_max")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _record_protocol(base: Path, args: argparse.Namespace) -> None:
    budget = {key: getattr(args, key) for key in _BUDGET_KEYS}
    digest = hashlib.sha256(json.dumps(budget, sort_keys=True).encode()).hexdigest()[:16]
    path = base / "protocol.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        old = {key: value for key, value in previous.get("budget", {}).items() if key != "seeds"}
        new = {key: value for key, value in budget.items() if key != "seeds"}
        if old != new and not args.force_config:
            raise RuntimeError(f"{path} was written with a different budget {old}; use another --out-dir "
                               "or --force-config to mix configurations deliberately")
        starts = list(previous.get("starts", []))
    else:
        starts = []
    starts.append(time.strftime("%Y-%m-%d %H:%M:%S"))
    record = {"budget": budget, "budget_digest": digest, "device": args.device, "git_commit": _git_commit(),
              "python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor(),
              "cpu_count": os.cpu_count(), "starts": starts,
              "command": " ".join([Path(sys.executable).name, *sys.argv])}
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def cmd_protocol(args: argparse.Namespace) -> None:
    base = Path(args.out_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)
    with ProtocolLock(base / "protocol.lock"):
        _record_protocol(base, args)
        _protocol_body(base, args)


def _protocol_body(base: Path, args: argparse.Namespace) -> None:
    from ares_u.nav.states import ae_pool_paths, prepare_all
    prepare_all()
    pool = [str(p) for p in ae_pool_paths()]
    env_flags = ["--action-repeat", str(args.action_repeat), "--episode-steps", str(args.episode_steps),
                 "--frame-stack", str(args.frame_stack), "--noop-max", str(args.noop_max)]
    learn = ["--num-envs", str(args.num_envs), "--device", args.device, "--rnd-coef", str(args.rnd_coef),
             "--epi-coef", str(args.epi_coef), "--resume"]
    eval_flags = ["--num-envs", str(args.num_envs), "--device", args.device, *env_flags]
    cross_frames = base / "frames_emerald_eval.npz"
    _run_phase("collect_emerald_eval", ["collect", "--game", "emerald", "--num-envs", str(args.num_envs),
                                        "--steps", "400", "--seed", "999", "--out", str(cross_frames), *env_flags],
               cross_frames, base / "logs")
    for seed in args.seeds:
        run = base / f"seed{seed}"
        logs = run / "logs"
        s = str(seed)
        frames = run / "frames_firered.npz"
        collect_env = ["--action-repeat", str(args.action_repeat), "--episode-steps", str(args.collect_episode_steps),
                       "--frame-stack", str(args.frame_stack), "--noop-max", str(args.noop_max)]
        _run_phase("collect_firered", ["collect", "--game", "firered", "--num-envs", str(args.num_envs), "--steps",
                                       str(args.collect_steps), "--seed", s, "--out", str(frames), *collect_env,
                                       "--start-states", *pool], frames, logs)
        ae = run / "ae_firered.pt"
        _run_phase("pretrain_ae", ["pretrain", "--frames", str(frames), "--out", str(ae), "--eval-frames",
                                   str(cross_frames), "--epochs", str(args.ae_epochs), "--seed", s,
                                   "--device", args.device], ae, logs)
        fr = run / "firered_ppo_ae"
        _run_phase("train_firered", ["train", "--game", "firered", "--out-dir", str(fr), "--steps", str(args.steps_fr),
                                     "--init", f"ae:{ae}", "--seed", s, *learn, *env_flags], fr / "done.json", logs)
        stages: list[tuple[str, Path, list[str]]] = [
            ("train_emerald_scratch", run / "emerald_scratch", ["--init", "scratch"]),
            ("train_emerald_transfer", run / "emerald_transfer", ["--init", f"policy:{fr / 'final.pt'}"]),
        ]
        if args.ae_only:
            stages.append(("train_emerald_ae_only", run / "emerald_ae_only", ["--init", f"ae:{ae}"]))
        if args.random_run:
            stages.append(("train_emerald_random", run / "emerald_random_run", ["--random-policy"]))
        for name, out_dir, extra in stages:
            _run_phase(name, ["train", "--game", "emerald", "--out-dir", str(out_dir), "--steps", str(args.steps_em),
                              "--seed", s, *extra, *learn, *env_flags], out_dir / "done.json", logs)
        eval_seed = str(10_000 + seed)
        evaluations: list[tuple[str, str, str, int]] = [
            ("emerald", "random", "", args.eval_episodes),
            ("emerald", "zero_shot", str(fr / "final.pt"), args.eval_episodes),
            ("emerald", "scratch", str(run / "emerald_scratch" / "final.pt"), args.eval_episodes),
            ("emerald", "transfer", str(run / "emerald_transfer" / "final.pt"), args.eval_episodes),
            ("firered", "random", "", args.fr_eval_episodes),
            ("firered", "firered_policy", str(fr / "final.pt"), args.fr_eval_episodes),
        ]
        if args.ae_only:
            evaluations.append(("emerald", "ae_only", str(run / "emerald_ae_only" / "final.pt"), args.eval_episodes))
        for game, method, policy, episodes in evaluations:
            out = run / f"eval_{game}_{method}.json"
            argv = ["evaluate", "--game", game, "--method", method, "--out", str(out), "--seed", eval_seed,
                    "--episodes", str(episodes), *eval_flags]
            if policy:
                argv += ["--policy", policy]
            _run_phase(f"eval_{game}_{method}", argv, out, logs)
        try:  # partial summary after every seed so results exist even if a later seed is cut short
            summarise(base)
        except Exception as error:  # noqa: BLE001 - a plotting failure must not stop the protocol
            print(f"[warn] summary failed: {error!r}", flush=True)
    summarise(base, verbose=True)


# --------------------------------------------------------------------------- summary

def _load_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _curve_on_grid(rows: list[dict[str, Any]], key: str, grid: Any) -> Any:
    import numpy as np
    points = [(r["env_steps"], r[key]) for r in rows if r.get(key) is not None]
    if not points:
        return None
    xs, ys = zip(*points)
    return np.interp(grid, np.array(xs, dtype=float), np.array(ys, dtype=float), left=np.nan, right=np.nan)


def steps_to_reach(rows: list[dict[str, Any]], key: str, threshold: float) -> float | None:
    """First ``env_steps`` at which ``rows[key] >= threshold`` (None if never reached)."""
    for row in rows:
        value = row.get(key)
        if value is not None and float(value) >= threshold:
            return float(row["env_steps"])
    return None


def curve_auc(rows: list[dict[str, Any]], key: str, budget: float) -> float:
    """Mean of a step-wise curve over [0, budget] (area under the curve / budget)."""
    import numpy as np
    points = [(0.0, 0.0)] + [(float(r["env_steps"]), float(r[key])) for r in rows if r.get(key) is not None]
    xs = np.array([p[0] for p in points] + [budget])
    ys = np.array([p[1] for p in points] + [points[-1][1]])
    keep = xs <= budget
    xk, yk = xs[keep], ys[keep]
    area = float(np.sum((xk[1:] - xk[:-1]) * (yk[1:] + yk[:-1]) / 2.0))
    return area / budget if budget > 0 else float("nan")


def hierarchical_bootstrap_diff(a: list[list[float]], b: list[list[float]], reps: int = 10_000,
                                seed: int = 0) -> tuple[float, float, float]:
    """Mean difference a - b and its 95% CI, resampling seeds and then episodes within each seed."""
    import numpy as np
    rng = np.random.default_rng(seed)

    def draw(groups: list[list[float]]) -> Any:
        arrays = [np.asarray(g, dtype=float) for g in groups]
        picks = rng.integers(0, len(arrays), size=(reps, len(arrays)))
        means = np.zeros(reps)
        for column in range(len(arrays)):
            for index, values in enumerate(arrays):
                rows = picks[:, column] == index
                count = int(rows.sum())
                if count:
                    sample = values[rng.integers(0, len(values), size=(count, len(values)))]
                    means[rows] += sample.mean(axis=1)
        return means / len(arrays)

    diff = draw(a) - draw(b)
    point = float(np.mean([np.mean(g) for g in a]) - np.mean([np.mean(g) for g in b]))
    low, high = np.percentile(diff, [2.5, 97.5])
    return point, float(low), float(high)


def _holm(tests: list[dict[str, Any]]) -> None:
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p"])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, tests[i]["p"] * (len(tests) - rank)))
        tests[i]["p_holm"] = running


def _mean_std(values: list[float]) -> tuple[float, float]:
    import numpy as np
    clean = [v for v in values if v is not None and v == v]
    if not clean:
        return float("nan"), float("nan")
    return float(np.mean(clean)), float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0


def summarise(base: Path, verbose: bool = False) -> dict[str, Any]:
    import numpy as np
    from scipy.stats import mannwhitneyu
    from ares_u.nav.train import read_jsonl

    seeds = sorted((p for p in base.glob("seed*") if p.is_dir()), key=lambda p: int(p.name[4:]))
    evals: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in seeds:
        for path in sorted(run.glob("eval_*.json")):
            data = _load_json(path)
            evals.setdefault((data["game"], data["method"]), []).append({**data, "run": run.name})
    table: list[dict[str, Any]] = []
    for (game, method), items in sorted(evals.items()):
        per_seed = [float(item["cells_mean"]) for item in items]
        pooled = [c for item in items for c in item["cells"]]
        mean, std = _mean_std(per_seed)
        table.append({"game": game, "method": method, "seeds": len(items), "per_seed_mean": per_seed,
                      "mean_over_seeds": mean, "std_over_seeds": std,
                      "pooled_episodes": len(pooled), "pooled_mean": float(np.mean(pooled)),
                      "pooled_std": float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0,
                      "maps_mean": float(np.mean([item["maps_mean"] for item in items])),
                      "battle_fraction": float(np.mean([item["battle_fraction"] for item in items]))})
    tests: list[dict[str, Any]] = []
    for a, b in (("transfer", "scratch"), ("scratch", "random"), ("transfer", "random"), ("zero_shot", "random"),
                 ("transfer", "zero_shot"), ("ae_only", "scratch")):
        if ("emerald", a) in evals and ("emerald", b) in evals:
            groups_a = [list(map(float, item["cells"])) for item in evals[("emerald", a)]]
            groups_b = [list(map(float, item["cells"])) for item in evals[("emerald", b)]]
            xa, xb = [c for g in groups_a for c in g], [c for g in groups_b for c in g]
            stat = mannwhitneyu(xa, xb, alternative="two-sided")
            diff, low, high = hierarchical_bootstrap_diff(groups_a, groups_b)
            row: dict[str, Any] = {"a": a, "b": b, "n_a": len(xa), "n_b": len(xb), "U": float(stat.statistic),
                                   "p": float(stat.pvalue), "effect_cles": float(stat.statistic) / (len(xa) * len(xb)),
                                   "mean_diff": diff, "ci95_low": low, "ci95_high": high,
                                   "note": "pooled evaluation episodes across seeds; CI = hierarchical bootstrap"}
            means_a = [float(np.mean(g)) for g in groups_a]
            means_b = [float(np.mean(g)) for g in groups_b]
            if len(means_a) > 1 and len(means_b) > 1:
                seed_stat = mannwhitneyu(means_a, means_b, alternative="two-sided")
                row.update({"U_seed_means": float(seed_stat.statistic), "p_seed_means": float(seed_stat.pvalue)})
            tests.append(row)
    if tests:
        _holm(tests)
    training: dict[str, dict[str, Any]] = {}
    curves: dict[str, list[list[dict[str, Any]]]] = {}
    for kind in ("firered_ppo_ae", *TRAIN_KINDS):
        runs = [read_jsonl(run / kind / "metrics.jsonl") for run in seeds if (run / kind / "done.json").exists()]
        runs = [r for r in runs if r]
        if not runs:
            continue
        curves[kind] = runs
        final = [r[-1] for r in runs]
        training[kind] = {"seeds": len(runs), "final_env_steps": [f["env_steps"] for f in final],
                          "final_global_cells": [f["global_cells"] for f in final],
                          "final_global_maps": [f["global_maps"] for f in final],
                          "final_episode_cells_recent_mean": [f.get("episode_cells_recent_mean") for f in final],
                          "wall_minutes": [f["wall_s"] / 60 for f in final],
                          "steps_per_s": [f["steps_per_s"] for f in final]}
    efficiency = _sample_efficiency(curves, evals)
    summary = {"dir": str(base), "eval": table, "tests": tests, "training": training, "sample_efficiency": efficiency}
    (base / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = _summary_markdown(base, table, tests, training, efficiency)
    (base / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _figures(base, curves, evals)
    if verbose:
        print("\n".join(lines), flush=True)
    return summary


def _sample_efficiency(curves: dict[str, list[list[dict[str, Any]]]],
                       evals: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[str, Any]:
    import numpy as np
    learners = [k for k in ("emerald_scratch", "emerald_transfer", "emerald_ae_only") if k in curves]
    if not learners:
        return {}
    budget = float(min(r[-1]["env_steps"] for k in learners for r in curves[k]))
    thresholds: dict[str, tuple[str, float]] = {}
    if "emerald_random_run" in curves:
        random_final = [float(np.interp(budget, [x["env_steps"] for x in r], [x["global_cells"] for x in r]))
                        for r in curves["emerald_random_run"]]
        thresholds["global_cells>=random_full_budget"] = ("global_cells", float(np.mean(random_final)))
    for method in ("random", "zero_shot"):
        if ("emerald", method) in evals:
            level = float(np.mean([item["cells_mean"] for item in evals[("emerald", method)]]))
            thresholds[f"episode_cells>={method}_eval"] = ("episode_cells_recent_mean", level)
    result: dict[str, Any] = {"budget_env_steps": budget,
                              "thresholds": {name: value for name, (_, value) in thresholds.items()},
                              "methods": {}}
    for kind in [*learners, *(["emerald_random_run"] if "emerald_random_run" in curves else [])]:
        rows = curves[kind]
        entry: dict[str, Any] = {"auc_global_cells": [curve_auc(r, "global_cells", budget) for r in rows]}
        entry["auc_global_cells_mean_std"] = _mean_std(entry["auc_global_cells"])
        for name, (key, value) in thresholds.items():
            reached = [steps_to_reach(r, key, value) for r in rows]
            entry[name] = {"per_seed": reached, "reached": sum(v is not None for v in reached),
                           "mean_std_of_reached": _mean_std([v for v in reached if v is not None])}
        result["methods"][kind] = entry
    if "emerald_transfer" in curves and "emerald_scratch" in curves:
        from scipy.stats import mannwhitneyu
        a = result["methods"]["emerald_transfer"]["auc_global_cells"]
        b = result["methods"]["emerald_scratch"]["auc_global_cells"]
        if len(a) > 1 and len(b) > 1:
            stat = mannwhitneyu(a, b, alternative="two-sided")
            result["auc_transfer_vs_scratch"] = {"U": float(stat.statistic), "p": float(stat.pvalue),
                                                 "note": "per-seed AUC, n is the number of seeds"}
    return result


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _summary_markdown(base: Path, table: list[dict[str, Any]], tests: list[dict[str, Any]],
                      training: dict[str, dict[str, Any]], efficiency: dict[str, Any]) -> list[str]:
    lines = [f"# ARES-Nav summary: {base.name}", "",
             "Metric: unique (map, x, y) cells read from RAM (metric only; the policy sees pixels only).", "",
             "## Evaluation (unique cells per fixed-length episode)", "",
             "| game | method | seeds | mean ± sd over seeds | per-seed means | pooled eps (mean ± sd) | maps | battle |",
             "|---|---|---|---|---|---|---|---|"]
    for row in table:
        lines.append(f"| {row['game']} | {row['method']} | {row['seeds']} | {_fmt(row['mean_over_seeds'])} ± "
                     f"{_fmt(row['std_over_seeds'])} | {', '.join(f'{v:.1f}' for v in row['per_seed_mean'])} | "
                     f"{row['pooled_episodes']} ({row['pooled_mean']:.1f} ± {row['pooled_std']:.1f}) | "
                     f"{row['maps_mean']:.2f} | {row['battle_fraction']:.3f} |")
    lines += ["", "## Tests (Emerald; Mann-Whitney U on pooled episodes, Holm-corrected; "
                  "hierarchical-bootstrap 95% CI of the mean difference)", ""]
    for t in tests:
        seed_part = (f"; per-seed means U={t['U_seed_means']:.1f}, p={t['p_seed_means']:.3g}"
                     if "p_seed_means" in t else "")
        lines.append(f"- {t['a']} vs {t['b']}: diff {t['mean_diff']:+.1f} cells [{t['ci95_low']:+.1f}, "
                     f"{t['ci95_high']:+.1f}], U={t['U']:.1f}, p={t['p']:.3g}, p_holm={t.get('p_holm', t['p']):.3g}, "
                     f"P(a>b)={t['effect_cles']:.2f} (n={t['n_a']} vs {t['n_b']}){seed_part}")
    lines += ["", "## Training runs (cumulative unique cells over the whole run)", ""]
    for kind, data in training.items():
        lines.append(f"- {kind}: global cells {data['final_global_cells']}, maps {data['final_global_maps']}, "
                     f"steps {data['final_env_steps']}, minutes {[round(m, 1) for m in data['wall_minutes']]}, "
                     f"steps/s {[round(v) for v in data['steps_per_s']]}")
    if efficiency:
        lines += ["", f"## Sample efficiency (budget {efficiency['budget_env_steps']:.0f} env steps)", "",
                  "Thresholds: " + ", ".join(f"{k} = {v:.1f}" for k, v in efficiency["thresholds"].items()), ""]
        for kind, entry in efficiency["methods"].items():
            mean, std = entry["auc_global_cells_mean_std"]
            parts = [f"AUC(global cells)/budget {mean:.1f} ± {std:.1f}"]
            for name in efficiency["thresholds"]:
                info = entry[name]
                m, s = info["mean_std_of_reached"]
                parts.append(f"steps to {name}: {_fmt(m, 0)} ± {_fmt(s, 0)} ({info['reached']}/{len(info['per_seed'])} "
                             "seeds reached)")
            lines.append(f"- {kind}: " + "; ".join(parts))
        if "auc_transfer_vs_scratch" in efficiency:
            t = efficiency["auc_transfer_vs_scratch"]
            lines.append(f"- AUC transfer vs scratch: U={t['U']:.1f}, p={t['p']:.3g} ({t['note']})")
    return lines


def _figures(base: Path, curves: dict[str, list[list[dict[str, Any]]]],
             evals: dict[tuple[str, str], list[dict[str, Any]]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"emerald_scratch": "#1f77b4", "emerald_transfer": "#d62728", "emerald_random_run": "#7f7f7f",
              "emerald_ae_only": "#2ca02c", "firered_ppo_ae": "#ff7f0e"}
    labels = {"emerald_scratch": "scratch (PPO+novelty)", "emerald_transfer": "transfer (FR AE+PPO, fine-tuned)",
              "emerald_random_run": "random", "emerald_ae_only": "AE encoder only",
              "firered_ppo_ae": "FireRed pretraining (AE+PPO)"}
    figures = base / "figures"
    figures.mkdir(exist_ok=True)

    def band(ax: Any, kinds: list[str], key: str) -> None:
        for kind in kinds:
            runs = curves.get(kind, [])
            if not runs:
                continue
            horizon = min(r[-1]["env_steps"] for r in runs)
            grid = np.linspace(0, horizon, 200)
            stack = [c for c in (_curve_on_grid(r, key, grid) for r in runs) if c is not None]
            if not stack:
                continue
            values = np.vstack(stack)
            mean, std = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
            ax.plot(grid, mean, color=colors[kind], label=f"{labels[kind]} (n={len(stack)})")
            ax.fill_between(grid, mean - std, mean + std, color=colors[kind], alpha=0.2)

    emerald_kinds = ["emerald_scratch", "emerald_transfer", "emerald_ae_only", "emerald_random_run"]
    for key, ylabel, filename in (("global_cells", "cumulative unique cells (Emerald)", "train_global_cells.png"),
                                  ("episode_cells_recent_mean", "unique cells per training episode (last 32)",
                                   "train_episode_cells.png")):
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        band(ax, emerald_kinds, key)
        if key == "episode_cells_recent_mean":
            for method, style in (("zero_shot", "--"), ("random", ":")):
                if ("emerald", method) in evals:
                    level = [item["cells_mean"] for item in evals[("emerald", method)]]
                    ax.axhline(float(np.mean(level)), color="black", linestyle=style, linewidth=1,
                               label=f"{method} (eval mean, n={len(level)})")
        ax.set_xlabel("environment steps")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=150)
        plt.close(fig)
    if "firered_ppo_ae" in curves:
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
        band(axes[0], ["firered_ppo_ae"], "global_cells")
        band(axes[1], ["firered_ppo_ae"], "episode_cells_recent_mean")
        axes[0].set_ylabel("cumulative unique cells (FireRed)")
        axes[1].set_ylabel("unique cells per episode (last 32)")
        for ax in axes:
            ax.set_xlabel("environment steps")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures / "firered_pretraining.png", dpi=150)
        plt.close(fig)
    emerald = [(m, evals[("emerald", m)]) for m in EMERALD_METHODS if ("emerald", m) in evals]
    if emerald:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for i, (method, items) in enumerate(emerald):
            per_seed = [float(item["cells_mean"]) for item in items]
            ax.bar(i, float(np.mean(per_seed)), yerr=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
                   color="#9ecae1", capsize=4)
            ax.scatter([i] * len(per_seed), per_seed, color="black", s=12, zorder=3)
        ax.set_xticks(range(len(emerald)), [m for m, _ in emerald])
        ax.set_ylabel("unique cells per episode (Emerald)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures / "eval_emerald_bars.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for method, items in emerald:
            length = min(len(item["curve_mean"]) for item in items)
            seed_curves = np.array([item["curve_mean"][:length] for item in items])
            mean = seed_curves.mean(axis=0)
            std = seed_curves.std(axis=0, ddof=1) if len(items) > 1 else np.zeros_like(mean)
            steps = items[0]["curve_steps"][:length]
            line = ax.plot(steps, mean, label=f"{method} (n={len(items)})")[0]
            ax.fill_between(steps, mean - std, mean + std, color=line.get_color(), alpha=0.15)
        ax.set_xlabel("step within evaluation episode")
        ax.set_ylabel("unique cells so far (Emerald)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "eval_emerald_within_episode.png", dpi=150)
        plt.close(fig)
    ae_logs = [(run.name, _read_rows(run / "ae_firered.jsonl")) for run in sorted(base.glob("seed*"))]
    ae_logs = [(name, rows) for name, rows in ae_logs if rows]
    if ae_logs:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for name, rows in ae_logs:
            epochs = [r["epoch"] for r in rows]
            line = ax.plot(epochs, [r.get("val_r2", float("nan")) for r in rows], label=f"{name} FireRed val")[0]
            ax.plot(epochs, [r.get("cross_game_r2", float("nan")) for r in rows], linestyle="--",
                    color=line.get_color(), label=f"{name} Emerald (unseen)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("reconstruction R^2")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "ae_reconstruction.png", dpi=150)
        plt.close(fig)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cmd_summary(args: argparse.Namespace) -> None:
    base = Path(args.dir) if args.dir else NAV / "runs" / args.name
    summarise(base, verbose=True)


# --------------------------------------------------------------------------- parser

def _add(sub: Any, name: str, func: Callable[[argparse.Namespace], None]) -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = sub.add_parser(name)
    parser.set_defaults(func=func)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = _add(sub, "prepare-states", cmd_prepare_states)
    p.add_argument("--force", action="store_true")

    p = _add(sub, "bench", cmd_bench)
    p.add_argument("--games", nargs="+", default=["firered", "emerald"])
    p.add_argument("--ks", nargs="+", type=int, default=[16, 24])
    p.add_argument("--ns", nargs="+", type=int, default=[1, 6])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--raw-frames", type=int, default=4000)
    p.add_argument("--learner", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--note", default="")
    p.add_argument("--out", default=str(NAV / "bench.json"))

    p = _add(sub, "collect", cmd_collect)
    p.add_argument("--game", default="firered")
    p.add_argument("--num-envs", type=int, default=6)
    p.add_argument("--steps", type=int, default=2000, help="steps per env")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeat-prob", type=float, default=0.75)
    p.add_argument("--ae-pool", action="store_true", help="start episodes from the FireRed AE state pool")
    p.add_argument("--out", required=True)
    _env_args(p)

    p = _add(sub, "pretrain", cmd_pretrain)
    p.add_argument("--frames", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--eval-frames", default="")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")

    p = _add(sub, "train", cmd_train)
    p.add_argument("--game", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=250_000)
    p.add_argument("--num-envs", type=int, default=6)
    p.add_argument("--rollout-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--rnd-coef", type=float, default=0.5)
    p.add_argument("--epi-coef", type=float, default=1.0, help="0 disables the episodic frame-key bonus")
    p.add_argument("--reward-norm-decay", type=float, default=0.9, help=">= 1 uses all-history statistics")
    p.add_argument("--hash-rows", type=int, default=5)
    p.add_argument("--hash-cols", type=int, default=7)
    p.add_argument("--hash-levels", type=int, default=8)
    p.add_argument("--no-center-rnd", dest="center_rnd", action="store_false",
                   help="disable reward centering of the RND stream")
    p.add_argument("--target-kl", type=float, default=0.05, help="PPO early-stop threshold (0 = off)")
    p.add_argument("--init", default="scratch")
    p.add_argument("--random-policy", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-minutes", type=float, default=0.0)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--resume", action="store_true", help="continue from out-dir/checkpoint.pt if present")
    _env_args(p)

    p = _add(sub, "evaluate", cmd_evaluate)
    p.add_argument("--game", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--policy", default="")
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--num-envs", type=int, default=6)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", required=True)
    _env_args(p)

    p = _add(sub, "protocol", cmd_protocol)
    p.add_argument("--out-dir", default=str(FINAL))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--steps-fr", type=int, default=200_000)
    p.add_argument("--steps-em", type=int, default=200_000)
    p.add_argument("--collect-steps", type=int, default=4000, help="FireRed AE frames per env")
    p.add_argument("--collect-episode-steps", type=int, default=256, help="episode length while collecting")
    p.add_argument("--ae-epochs", type=int, default=20)
    p.add_argument("--eval-episodes", type=int, default=18, help="multiple of --num-envs avoids idle envs")
    p.add_argument("--fr-eval-episodes", type=int, default=6)
    p.add_argument("--num-envs", type=int, default=6)
    p.add_argument("--device", default="auto")
    p.add_argument("--rnd-coef", type=float, default=0.5)
    p.add_argument("--epi-coef", type=float, default=1.0)
    p.add_argument("--no-random-run", dest="random_run", action="store_false",
                   help="skip the uniform-random training-length run (cumulative-cells baseline curve)")
    p.add_argument("--ae-only", action="store_true", help="ablation: AE encoder only, PPO from scratch")
    p.add_argument("--force-config", action="store_true", help="allow resuming a directory with another budget")
    p.add_argument("--action-repeat", type=int, default=24)
    p.add_argument("--episode-steps", type=int, default=1024)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--noop-max", type=int, default=30)

    p = _add(sub, "summary", cmd_summary)
    p.add_argument("--dir", default="", help="protocol directory (default: artifacts/ares_u/nav/runs/<name>)")
    p.add_argument("--name", default="main")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
