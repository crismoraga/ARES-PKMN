"""Real-cartridge check of the battle brain: same checkpoint, ``learned`` vs ``heuristic``.

    python -m ares_u.brain.game_check prepare --game emerald            (copy the latest checkpoint)
    python -m ares_u.brain.game_check run --game emerald --mode learned --frames 120000
    python -m ares_u.brain.game_check run --game emerald --mode heuristic --frames 120000
    python -m ares_u.brain.game_check summary --game emerald

``prepare`` copies the newest checkpoint of ``artifacts/campaigns/native/<game>`` (read only: the live
worker is never touched, its lock is never taken) to ``artifacts/brain/campaign_copy/source/<game>``;
each ``run`` copies that source to ``campaign_copy/<mode>/<game>`` and plays ``--frames`` emulator
frames from it with ``ARES_BATTLE_BRAIN=<mode>``. Everything is read from RAM: per battle the trainer
flag, the outcome (``gBattleOutcome`` through the orchestrator metrics), party HP before/after and
faints; per decision whether the brain or the heuristic chose it, and fallback events. The live
dashboard shows the run as ``brain-check-<mode>`` (never the real campaign's feed).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts" / "campaigns" / "native"
COPY = ROOT / "artifacts" / "brain" / "campaign_copy"
OUT = ROOT / "artifacts" / "brain" / "game_check"


def prepare(game: str) -> Path:
    pointer = json.loads((SOURCE / game / "latest.json").read_text(encoding="utf-8"))
    frame_dir = SOURCE / game / str(pointer["directory"])
    target = COPY / "source" / game
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copytree(frame_dir, target / frame_dir.name)
    (target / "latest.json").write_text(json.dumps(pointer), encoding="utf-8")
    for name in ("lessons.json",):
        if (SOURCE / game / name).is_file():
            shutil.copy2(SOURCE / game / name, target / name)
    if (SOURCE / "shared_knowledge.json").is_file():
        shutil.copy2(SOURCE / "shared_knowledge.json", target.parent / "shared_knowledge.json")
    return target


def _party_hp(bridge: Any) -> tuple[int, int, int] | None:
    """(hp, max hp, fainted) of the party, ``None`` while the party block is being rewritten (bad checksum)."""
    try:
        party = bridge.get_decrypted_player_party()
    except ValueError:
        return None
    hp = sum(int(mon["hp"]) for mon in party)
    top = sum(int(mon["maxHP"]) for mon in party)
    fainted = sum(1 for mon in party if int(mon["hp"]) <= 0)
    return hp, top, fainted


def run(game: str, mode: str, frames: int, seed: int, minutes: float = 15.0) -> Path:
    os.environ["ARES_BATTLE_BRAIN"] = mode
    from core.campaign_live import CampaignLivePublisher
    from core.game_data import ROM_NAMES
    from core.orchestrator import AresAutonomousOrchestrator

    source = COPY / "source" / game
    work_root = COPY / f"{mode}_s{seed}"
    if work_root.exists():
        shutil.rmtree(work_root)
    shutil.copytree(source.parent, work_root)
    directory = work_root / game
    controller = AresAutonomousOrchestrator(ROOT / "native/build/ares_mgba.dll", ROOT.parent / ROM_NAMES[game],  # type: ignore[index]
                                            game == "emerald", checkpoint_dir=directory, seed=seed)
    controller.load_checkpoint(None)
    controller.rearm_recovery()
    counters = {"decisions": 0, "learned": 0, "fallbacks": 0}
    deferrals: dict[str, int] = {}
    original_choose = controller.choose_battle_decision
    original_record = controller._record

    def choose() -> Any:
        decision = original_choose()
        counters["decisions"] += 1
        counters["learned"] += controller.__dict__.get("last_battle_choice") is not None
        deferral = str(controller.__dict__.get("last_brain_deferral", "") or "")
        if deferral:
            deferrals[deferral] = deferrals.get(deferral, 0) + 1
        return decision

    def record(kind: str, **values: object) -> None:
        if kind == "battle_brain_fallback":
            counters["fallbacks"] += 1
            print(f"[game-check] fallback: {values.get('error')}", flush=True)
        original_record(kind, **values)

    controller.choose_battle_decision = choose  # type: ignore[method-assign]
    controller._record = record  # type: ignore[method-assign]
    live = CampaignLivePublisher(game, run_id=f"brain-check-{mode}",
                                 agent=f"ARES prueba de combate ({'cerebro aprendido' if mode == 'learned' else 'heurística'})")
    bridge = controller.bridge
    start_frame = bridge.frame_count
    metrics0 = dict(controller.metrics.__dict__)
    battles: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    started = time.perf_counter()
    last_print = started
    # wall-clock cap: a navigation stall (a few frames per minute) must not hold the check forever
    while bridge.frame_count - start_frame < frames and time.perf_counter() - started < minutes * 60:
        controller.step_vblank_cycle()
        in_battle, trainer = bridge.is_in_battle()
        reading = _party_hp(bridge) if in_battle and current is None else None
        if in_battle and current is None and reading is not None:
            hp, top, fainted = reading
            current = {"frame": bridge.frame_count, "trainer": bool(trainer), "hp_start": hp, "max_hp": top,
                       "fainted_start": fainted, "decisions_start": counters["decisions"],
                       "learned_start": counters["learned"], "wins": controller.metrics.wins,
                       "losses": controller.metrics.losses, "captures": controller.metrics.captures}
        elif not in_battle and current is not None:
            current["end_frame"] = bridge.frame_count
            current["pending"] = True
            battles.append(current)
            current = None
        # outcome bookkeeping: metrics move a few frames after the battle flag drops
        for row in battles:
            if row.get("pending") and bridge.frame_count - int(row["end_frame"]) >= 240:
                after = _party_hp(bridge)
                if after is None:
                    continue
                hp, _, fainted = after
                row.update({"hp_end": hp, "fainted_end": fainted, "pending": False,
                            "won": controller.metrics.wins - row["wins"] > 0,
                            "lost": controller.metrics.losses - row["losses"] > 0,
                            "caught": controller.metrics.captures - row["captures"] > 0,
                            "decisions": counters["decisions"] - row["decisions_start"],
                            "learned_decisions": counters["learned"] - row["learned_start"]})
                row["hp_lost"] = max(0, int(row["hp_start"]) - hp) if not row["lost"] else int(row["hp_start"])
        live.maybe_publish(controller)
        if time.perf_counter() - last_print > 60:
            last_print = time.perf_counter()
            done = [row for row in battles if not row.get("pending")]
            print(f"[game-check] {mode} frames {bridge.frame_count - start_frame}/{frames} battles {len(done)} "
                  f"won {sum(row['won'] for row in done)} lost {sum(row['lost'] for row in done)} "
                  f"decisions {counters['decisions']} learned {counters['learned']}", flush=True)
    done = [row for row in battles if not row.get("pending")]
    metrics1 = dict(controller.metrics.__dict__)
    delta = {key: metrics1[key] - metrics0[key] for key in metrics1
             if isinstance(metrics1[key], (int, float)) and isinstance(metrics0.get(key), (int, float))}
    result = {"game": game, "mode": mode, "seed": seed, "frames": bridge.frame_count - start_frame, "start_frame": start_frame,
              "seconds": round(time.perf_counter() - started, 1), "counters": counters, "deferrals": deferrals, "battles": done,
              "metrics_delta": delta, "milestones": list(getattr(controller.metrics, "completed_milestones", []))}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{game}_{mode}_s{seed}.json"
    path.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return path


def _play_battle(controller: Any, limit: int = 30000) -> dict[str, Any]:
    """Step until the battle has been over for 240 frames; RAM outcome, HP and decision counts."""
    bridge = controller.bridge
    before = _party_hp(bridge)
    wins, losses, captures = controller.metrics.wins, controller.metrics.losses, controller.metrics.captures
    start, ended, decisions, learned = bridge.frame_count, None, 0, 0
    original = controller.choose_battle_decision

    def choose() -> Any:
        nonlocal decisions, learned
        decision = original()
        decisions += 1
        learned += controller.__dict__.get("last_battle_choice") is not None
        return decision

    controller.choose_battle_decision = choose
    try:
        while bridge.frame_count - start < limit:
            controller.step_vblank_cycle()
            if not bridge.is_in_battle()[0]:
                ended = bridge.frame_count if ended is None else ended
                if bridge.frame_count - ended >= 240 and _party_hp(bridge) is not None:
                    break
            else:
                ended = None
    finally:
        del controller.choose_battle_decision
    after = _party_hp(bridge)
    lost = controller.metrics.losses > losses
    hp_start = before[0] if before else 0
    hp_end = after[0] if after else 0
    return {"frames": bridge.frame_count - start, "finished": ended is not None, "won": controller.metrics.wins > wins,
            "lost": lost, "caught": controller.metrics.captures > captures, "hp_start": hp_start, "hp_end": hp_end,
            "max_hp": before[1] if before else 0, "hp_lost": hp_start if lost else max(0, hp_start - hp_end),
            "faints": max(0, (after[2] if after else 0) - (before[2] if before else 0)),
            "decisions": decisions, "learned_decisions": learned,
            "deferral": str(controller.__dict__.get("last_brain_deferral", "") or "")}


def paired(game: str, frames: int, seed: int, minutes: float, trainer_only: bool) -> Path:
    """Counterfactual pairs: at each battle start save a checkpoint, play the battle with the brain,
    restore, play it with the heuristic, then continue the heuristic trajectory. Same start state,
    same RNG state, so the per-battle difference is the decision maker's."""
    from core.game_data import ROM_NAMES
    from core.orchestrator import AresAutonomousOrchestrator

    source = COPY / "source" / game
    work_root = COPY / f"paired_s{seed}"
    if work_root.exists():
        shutil.rmtree(work_root)
    shutil.copytree(source.parent, work_root)
    directory = work_root / game
    os.environ["ARES_BATTLE_BRAIN"] = "heuristic"
    controller = AresAutonomousOrchestrator(ROOT / "native/build/ares_mgba.dll", ROOT.parent / ROM_NAMES[game],  # type: ignore[index]
                                            game == "emerald", checkpoint_dir=directory, seed=seed)
    controller.load_checkpoint(None)
    controller.rearm_recovery()
    bridge = controller.bridge
    start_frame, started = bridge.frame_count, time.perf_counter()
    pairs: list[dict[str, Any]] = []
    out = OUT / f"{game}_paired_s{seed}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    while bridge.frame_count - start_frame < frames and time.perf_counter() - started < minutes * 60:
        controller.step_vblank_cycle()
        in_battle, trainer = bridge.is_in_battle()
        if not in_battle or (trainer_only and not trainer) or _party_hp(bridge) is None:
            continue
        snapshot = controller.save_checkpoint("brain_pair")
        frame = bridge.frame_count
        os.environ["ARES_BATTLE_BRAIN"] = "learned"
        learned = _play_battle(controller)
        controller.load_checkpoint(snapshot)
        os.environ["ARES_BATTLE_BRAIN"] = "heuristic"
        heuristic = _play_battle(controller)
        shutil.rmtree(snapshot, ignore_errors=True)
        pairs.append({"frame": frame, "trainer": bool(trainer), "learned": learned, "heuristic": heuristic})
        print(f"[paired] {game} frame {frame} trainer={bool(trainer)} learned won={learned['won']} lost={learned['lost']} "
              f"hp-{learned['hp_lost']} ({learned['learned_decisions']}/{learned['decisions']}) | heuristic "
              f"won={heuristic['won']} lost={heuristic['lost']} hp-{heuristic['hp_lost']}", flush=True)
        out.write_text(json.dumps({"game": game, "seed": seed, "pairs": pairs}, indent=1), encoding="utf-8")
    out.write_text(json.dumps({"game": game, "seed": seed, "frames": bridge.frame_count - start_frame,
                               "seconds": round(time.perf_counter() - started, 1), "pairs": pairs}, indent=1),
                   encoding="utf-8")
    return out


def summary(game: str) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for mode in ("learned", "heuristic"):
        runs = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(OUT.glob(f"{game}_{mode}_s*.json"))]
        if not runs:
            continue
        rows = [row for data in runs for row in data["battles"]]
        data = {"frames": sum(item["frames"] for item in runs), "seeds": [item.get("seed") for item in runs],
                "counters": {key: sum(item["counters"][key] for item in runs) for key in ("decisions", "learned", "fallbacks")},
                "metrics_delta": {key: sum(item["metrics_delta"].get(key, 0) for item in runs)
                                  for key in ("battles", "wins", "losses", "captures", "faints", "map_transitions")}}
        fought = [row for row in rows if row["decisions"] > 0]
        table[mode] = {
            "frames": data["frames"], "runs": len(runs), "seeds": data["seeds"], "battles": len(rows), "trainer_battles": sum(row["trainer"] for row in rows),
            "won": sum(row["won"] for row in rows), "lost": sum(row["lost"] for row in rows),
            "caught": sum(row["caught"] for row in rows),
            "hp_lost_mean": round(sum(row["hp_lost"] for row in fought) / max(1, len(fought)), 2),
            "hp_lost_frac_mean": round(sum(row["hp_lost"] / max(1, row["max_hp"]) for row in fought) / max(1, len(fought)), 4),
            "trainer_hp_lost_frac_mean": round(
                sum(row["hp_lost"] / max(1, row["max_hp"]) for row in fought if row["trainer"]) /
                max(1, sum(1 for row in fought if row["trainer"])), 4),
            "faints": sum(max(0, row["fainted_end"] - row["fainted_start"]) for row in rows),
            "decisions": data["counters"]["decisions"], "learned_decisions": data["counters"]["learned"],
            "fallbacks": data["counters"]["fallbacks"], "metrics_delta": data["metrics_delta"],
            "deferrals": {key: sum(item.get("deferrals", {}).get(key, 0) for item in runs)
                          for key in sorted({name for item in runs for name in item.get("deferrals", {})})}}
    (OUT / f"{game}_summary.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("prepare", "run", "paired", "summary"))
    parser.add_argument("--game", choices=("emerald", "firered"), default="emerald")
    parser.add_argument("--mode", choices=("learned", "heuristic"), default="learned")
    parser.add_argument("--frames", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minutes", type=float, default=15.0)
    parser.add_argument("--trainer-only", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        print(prepare(args.game))
    elif args.stage == "paired":
        print(paired(args.game, args.frames, args.seed, args.minutes, args.trainer_only))
    elif args.stage == "run":
        print(run(args.game, args.mode, args.frames, args.seed, args.minutes))
    else:
        print(json.dumps(summary(args.game), indent=1))


if __name__ == "__main__":
    main()
