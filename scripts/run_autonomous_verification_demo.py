"""Physical-input synthetic verification; never a native campaign certificate."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.bridge.synthetic_gba_silicon import (
    SyntheticEncounter, SyntheticGBASiliconBridge, SyntheticMove, SyntheticWarp,
)
from core.game_data import ROOT
from core.navigation.warp_topology import HierarchicalWarpGraph
from core.orchestrator import AresAutonomousOrchestrator, SyntheticThreatEnvelope
from core.planner.htn_planner import CampaignMilestone, HTNCampaignPlanner


def _fixture_mon(species: int, hp: int, *, level: int, attack: int,
                 defense: int, speed: int, moves: list[int],
                 types: tuple[int, int]) -> dict[str, Any]:
    """Initial laboratory inputs; runtime progress uses only physical mechanics."""
    return {"species": species, "pid": 42 + species, "otid": 1001,
            "level": level, "hp": hp, "maxHP": hp, "attack": attack,
            "defense": defense, "speed": speed, "spAttack": attack,
            "spDefense": defense, "moves": moves, "pp": [35 if move else 0 for move in moves],
            "ivs": [15] * 6, "evs": [0] * 6, "friendship": 70,
            "experience": level ** 3, "types": list(types), "status": 0}


def build_laboratory(is_emerald: bool = False, seed: int = 42,
                     map_count: int = 54,
                     checkpoint_dir: Path | None = None) -> AresAutonomousOrchestrator:
    """Build a dependency DAG with ledges, elevation changes and physical warps.

    Each completion flag is initially unset.  A is required for ordinary goals;
    bosses grant their flags through victory.  Cartridge IO/controller writes are
    never used to bypass a task.  Threat bounds are declared scenario metadata.
    """
    if not 2 <= map_count <= 200:
        raise ValueError("The laboratory needs2..200 maps")
    bridge = SyntheticGBASiliconBridge(is_emerald, seed)
    bridge.configure_move(SyntheticMove(33, 35, 0, accuracy=100, max_pp=35))
    bridge.configure_move(SyntheticMove(22, 35, 12, True, accuracy=100, max_pp=35))
    bridge.configure_move(SyntheticMove(55, 40, 11, True, accuracy=100, max_pp=35))
    bridge.configure_party([_fixture_mon(283 if is_emerald else 1, 150, level=50,
        attack=90, defense=90, speed=30, moves=[33, 55 if is_emerald else 22, 0, 0],
        types=(11, 11) if is_emerald else (12, 3))])
    bridge.configure_inventory({4: 40, 13: 5})
    graph = HierarchicalWarpGraph()
    milestones: list[CampaignMilestone] = []
    bosses = {number for number in range(6, map_count - 2, 5)}
    previous = ""
    for number in range(map_count):
        tiles = np.ones((5, 8), dtype=np.uint16)
        collision = np.ones_like(tiles)
        collision[2, 1:7] = 0
        elevation = np.ones_like(tiles)
        elevation[2, 4:7] = 2
        behavior = np.zeros_like(tiles)
        tiles[2, 3], behavior[2, 3], collision[2, 3] = 2, 0x38, 1
        warps = [] if number == map_count - 1 else [SyntheticWarp(6, 2, 0, number + 1, 1, 2)]
        if warps:
            graph.add_warp((0, number), (0, number + 1), (6, 2), (1, 2))
        flag = 0x100 + number
        boss = number in bosses
        bridge.configure_map(0, number, tiles, collision, elevation, behavior, warps=warps,
                             heal_tiles={(5, 2)}, interaction_flags={} if boss else {(5, 2): flag})
        if number == 0:
            opponent = _fixture_mon(16, 8, level=2, attack=8, defense=8,
                                    speed=60, moves=[33, 0, 0, 0], types=(0, 2))
            bridge.configure_encounter(0, number, SyntheticEncounter(opponent, catch_rate=255))
        elif boss:
            opponent = _fixture_mon(74, 45, level=5, attack=15, defense=20,
                                    speed=60, moves=[33, 0, 0, 0], types=(5, 4))
            bridge.configure_encounter(0, number, SyntheticEncounter(opponent, trainer=True,
                                       reward_flag=flag, trigger_after_steps=1))
        elif number in (map_count - 2, map_count - 1):
            opponent = _fixture_mon(19, 45, level=5, attack=15, defense=20,
                                    speed=60, moves=[33, 0, 0, 0], types=(0, 0))
            bridge.configure_encounter(0, number, SyntheticEncounter(opponent, trainer=True,
                                       reward_flag=flag, trigger_after_steps=1))
            # Champion/HOF laboratory rewards are battle-earned, not interaction flags.
            bridge.configure_map(0, number, tiles, collision, elevation, behavior, warps=warps,
                                 heal_tiles={(5, 2)}, interaction_flags={})
        name = ("LAB_CHAMPION" if number == map_count - 2 else
                ("LAB_HALL_OF_FAME" if number == map_count - 1 else
                 (f"LAB_BOSS_{number}" if boss else f"LAB_WARP_{number}")))
        milestones.append(CampaignMilestone(name, "Synthetic observed interaction/victory", flag,
            0, number, (5, 2), 0, [], dependencies=(previous,) if previous else ()))
        previous = name
    bridge.transition_map(0, 0, 1, 2, relocate=False)
    planner = HTNCampaignPlanner(is_emerald, milestones, graph)
    envelopes = {species: SyntheticThreatEnvelope(5, 20, 20, 40) for species in (16, 19, 74)}
    return AresAutonomousOrchestrator(bridge, planner=planner, topology=graph,
        checkpoint_dir=checkpoint_dir, threat_envelopes=envelopes, seed=seed)


def run_demo(is_emerald: bool = False, seed: int = 42,
             map_count: int = 54, max_frames: int = 200000,
             output: Path | None = None) -> dict[str, object]:
    """Run the actual unified frame loop and fail if any laboratory gate misses."""
    controller = build_laboratory(is_emerald, seed, map_count, output)
    try:
        controller.run_until(lambda agent: len(agent.planner.completed_milestones(
            agent.bridge, agent.bridge.get_decrypted_player_party())) == map_count,
            max_frames=max_frames)
        # Consume the final outcome, rather than omitting its earned victory.
        controller.step_vblank_cycle()
        bridge = controller.bridge
        if not isinstance(bridge, SyntheticGBASiliconBridge):
            raise AssertionError("Laboratory bridge type changed")
        completed = controller.planner.completed_milestones(bridge, bridge.get_decrypted_player_party())
        metrics = asdict(controller.metrics)
        assertions = {
            "all_observed_goals": len(completed) == map_count,
            "physical_warps": sum(event["kind"] == "map_transition" for event in bridge.events) >= map_count,
            "save_relocations": bridge.relocation_count >= map_count - 1,
            "wild_capture": controller.metrics.captures >= 1,
            "trainer_victories": controller.metrics.wins >= 2,
            "causal_observations": controller.metrics.causal_observations > 0,
            "opponent_belief_updates": controller.metrics.belief_updates > 0,
            "neural_battle_decisions": controller.metrics.neural_decisions > 0,
            "jepa_route_planning": controller.metrics.neural_plans > 0,
            "gflownet_roster_planning": controller.metrics.gflownet_samples > 0,
            "no_faints": controller.metrics.faints == 0,
            "declared_envelope_proofs": controller.metrics.proved_decisions > 0,
        }
        result: dict[str, object] = {"game": controller.game, "backend": "synthetic",
            "native_campaign_certified": False, "completed": sorted(completed),
            "metrics": metrics, "relocations": bridge.relocation_count,
            "checks": assertions, "complete": all(assertions.values())}
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            controller.save_checkpoint("laboratory_complete" if all(assertions.values()) else "laboratory_failure")
            (output / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if not all(assertions.values()):
            raise AssertionError(json.dumps(result, indent=2))
        return result
    finally:
        controller.close()


def main() -> None:
    """Verify both synthetic games with saved evidence and explicit provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maps", type=int, default=54)
    parser.add_argument("--max-frames", type=int, default=200000)
    args = parser.parse_args()
    torch.set_num_threads(8)
    results = [run_demo(emerald, args.seed, args.maps, args.max_frames,
                       ROOT / "artifacts/synthetic" / ("emerald" if emerald else "firered"))
               for emerald in (False, True)]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
