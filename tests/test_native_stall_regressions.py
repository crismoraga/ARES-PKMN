"""Replay the recorded native deadlocks; each must now resolve through ordinary keys.

These bundles are local artifacts (gitignored) marked PINNED so retention keeps
them. Every test skips when the bundle, the ROM or the native shim is absent.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from core.game_data import Game, ROM_NAMES, ROOT
from core.orchestrator import AresAutonomousOrchestrator


def _native_controller(game: Game, prefix: str) -> AresAutonomousOrchestrator:
    bundles = sorted((ROOT / "artifacts/campaigns/native" / game).glob(f"{prefix}-*"))
    dll, rom = ROOT / "native/build/ares_mgba.dll", ROOT.parent / ROM_NAMES[game]
    if not bundles or not dll.is_file() or not rom.is_file():
        pytest.skip("Recorded native stall bundle, matching ROM and shim required")
    torch.set_num_threads(2)
    controller = AresAutonomousOrchestrator(dll, rom, game == "emerald", checkpoint_dir=None, online_learning=False)
    controller.load_checkpoint(Path(bundles[0]))
    return controller


@pytest.mark.native
def test_emerald_post_battle_evolution_learns_mud_shot_and_resumes() -> None:
    """Stalled at "Delete a move to make room for MUD SHOT?" while inBattle was set."""
    controller = _native_controller("emerald", "frame-000004034730")
    try:
        assert controller.menu.menu_mode() == "evolution"
        for _ in range(1600):
            controller.step_vblank_cycle()
        marshtomp = controller.bridge.get_decrypted_player_party()[0]
        assert marshtomp["species"] == 284 and 341 in marshtomp["moves"]  # Mud Shot learned
        assert 117 not in marshtomp["moves"]  # Bide (situational) was the move forgotten
        assert controller.menu.menu_mode() != "evolution" and controller.scene != "evolution"
        assert controller.metrics.faults == 0
        assert not any(event["kind"] == "stall_detected" for event in controller.audit)
    finally:
        controller.close()


@pytest.mark.native
def test_firered_route2_takes_viridian_forest_instead_of_oscillating() -> None:
    """Stalled stepping between (10,76) and (10,77) as off-camera Cut trees flickered."""
    controller = _native_controller("firered", "frame-000009269010")
    # This regression is about the Route 2 route choice; a Viridian Mart visit
    # (healing items and balls for the forest) is a separate, valid decision.
    controller.planner.shop_cooldown_until = 1 << 40
    try:
        maps: set[tuple[int, int]] = set()
        for _ in range(2400):
            controller.step_vblank_cycle()
            coords = controller.bridge.get_player_coords_and_map()
            maps.add((coords["map_group"], coords["map_num"]))
        assert (15, 0) in maps  # Route2_ViridianForest_SouthEntrance
        assert not any(event["kind"] in ("stall_detected", "oscillation_detected") for event in controller.audit)
    finally:
        controller.close()
