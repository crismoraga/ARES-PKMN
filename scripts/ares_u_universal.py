"""ARES-U universal player: run the game-agnostic agent on a Game Boy Pokemon game.

Examples::

    python -m scripts.ares_u_universal run --game rojo --frames 2000000
    python -m scripts.ares_u_universal run --game cristal --kb artifacts/universal/knowledge.json --sequence 2
    python -m scripts.ares_u_universal run --game cristal --kb-empty --run-id universal-cristal-nokb --no-publish
    python -m scripts.ares_u_universal run --game rojo --resume        # continue from runs/<run-id>/latest.state

Runs write ``artifacts/universal/runs/<run-id>/`` (log.txt, report.json, latest.state, m_<milestone>.state)
and publish live status as ``artifacts/live/<run-id>/``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ares_u.gal.games import game  # noqa: E402
from ares_u.universal.adapter import GameAdapter  # noqa: E402
from ares_u.universal.agent import UniversalAgent  # noqa: E402
from ares_u.universal.knowledge import DEFAULT_PATH, KnowledgeBase  # noqa: E402

UNIVERSAL = ROOT / "artifacts" / "universal"
DEFAULT_STARTS = {"rojo": UNIVERSAL / "dev" / "r1.state", "cristal": UNIVERSAL / "dev" / "c1.state"}


def make_adapter(key: str) -> GameAdapter:
    spec = game(key)
    if spec.generation == 1:
        from ares_u.universal.gen1 import Gen1Adapter

        return Gen1Adapter(spec)
    if spec.generation == 2:
        from ares_u.universal.gen2 import Gen2Adapter

        return Gen2Adapter(spec)
    raise SystemExit(f"no universal adapter for generation {spec.generation} ({key})")


def cmd_run(args: argparse.Namespace) -> int:
    run_id = args.run_id or f"universal-{args.game}"
    run_dir = UNIVERSAL / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    kb_path = Path(args.kb) if args.kb else DEFAULT_PATH
    kb = KnowledgeBase.empty(kb_path) if args.kb_empty else KnowledgeBase.load(kb_path)
    adapter = make_adapter(args.game)
    start = Path(args.start) if args.start else DEFAULT_STARTS.get(args.game)
    if args.resume and (run_dir / "latest.state").exists():
        start = run_dir / "latest.state"
    if start is None or not start.exists():
        raise SystemExit(f"start state not found: {start}")
    adapter.load_state(start.read_bytes())
    adapter.wait(2)
    log_handle = (run_dir / "log.txt").open("a", encoding="utf-8")

    def log(line: str) -> None:
        log_handle.write(line + "\n")
        log_handle.flush()
        if args.verbose:
            print(line, flush=True)

    log(f"=== run {run_id} game={args.game} start={start} kb={kb_path} empty={args.kb_empty} "
        f"sequence={args.sequence} t={time.strftime('%Y-%m-%d %H:%M:%S')}")
    agent = UniversalAgent(adapter, kb, run_id=run_id, run_dir=run_dir, sequence=args.sequence,
                           publish=not args.no_publish, seed=args.seed, use_brain=not args.no_brain, log=log)
    report = agent.run(max_frames=args.frames, max_seconds=args.seconds)
    log(f"=== end frames={report['frames']} milestones={list(report['milestones'])}")
    print(json.dumps({"milestones": {k: v["frame"] for k, v in report["milestones"].items()},
                      "stats": report["stats"], "maps_visited": report["maps_visited"]}, ensure_ascii=False))
    adapter.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="play a game with the universal agent")
    run.add_argument("--game", required=True, choices=("rojo", "cristal"))
    run.add_argument("--start", help="savestate to start from (default: artifacts/universal/dev/<game>)")
    run.add_argument("--resume", action="store_true", help="continue from runs/<run-id>/latest.state")
    run.add_argument("--frames", type=int, default=1_000_000)
    run.add_argument("--seconds", type=float, default=1e12)
    run.add_argument("--kb", help=f"knowledge base path (default {DEFAULT_PATH})")
    run.add_argument("--kb-empty", action="store_true", help="start from an empty knowledge base")
    run.add_argument("--run-id")
    run.add_argument("--sequence", type=int, default=1, help="position of this game in the curriculum")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--no-publish", action="store_true")
    run.add_argument("--no-brain", action="store_true", help="use the built-in type-chart policy only")
    run.add_argument("--verbose", action="store_true")
    run.set_defaults(func=cmd_run)
    args = parser.parse_args(argv)
    result: Any = args.func(args)
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
