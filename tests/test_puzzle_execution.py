"""Physical command feedback, bounded recovery, and exact checkpoint intent."""
from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pytest

from core.bridge.ffi_mgba import GBABridgeProtocol
from core.bridge.gba_structs import ObjectEventStruct
from core.bridge.synthetic_gba_silicon import SyntheticGBASiliconBridge, SyntheticWarp
from core.navigation.puzzle_execution import FeedbackPuzzleExecutor, can_use_surf
from core.navigation.source_data import SourceMapCatalog, load_map_catalog
from core.planner.htn_planner import HTNCampaignPlanner


class _Planner:
    def __init__(self, plans: list[list[dict[str, Any]] | None],
                 catalog: SourceMapCatalog | None = None,
                 interactions: list[dict[str, Any]] | None = None) -> None:
        self.plans, self.catalog, self.calls = plans, catalog, 0
        self.interactions = interactions or []

    def plan_puzzle_actions(self, bridge: GBABridgeProtocol, goal: tuple[int, int] | None = None,
                             max_states: int = 25_000) -> list[dict[str, Any]] | None:
        del bridge, goal, max_states
        index = min(self.calls, len(self.plans) - 1)
        self.calls += 1
        return copy.deepcopy(self.plans[index])

    def puzzle_actions(self, bridge: GBABridgeProtocol) -> list[dict[str, Any]]:
        del bridge
        return copy.deepcopy(self.interactions)


def _step(origin: tuple[int, int], destination: tuple[int, int]) -> dict[str, Any]:
    return {"mode": "EXECUTE_PUZZLE", "dest": (0, 0, destination), "action": "step",
            "expected_origin": origin, "puzzle": "fixture"}


def _floor(bridge: SyntheticGBASiliconBridge, collision: np.ndarray[Any, np.dtype[np.uint16]] | None = None) -> None:
    tiles = np.zeros((3, 8), dtype=np.uint16)
    bridge.configure_map(0, 0, tiles, tiles if collision is None else collision, tiles)
    bridge.transition_map(0, 0, 1, 1)


def test_step_plan_executes_exactly_once_and_never_advances_nested_frames() -> None:
    bridge = SyntheticGBASiliconBridge()
    try:
        _floor(bridge)
        executor = FeedbackPuzzleExecutor(bridge, _Planner([[_step((1, 1), (2, 1)), _step((2, 1), (3, 1))]]))
        for _ in range(100):
            before, frame = bridge.save_state(), bridge.frame_count
            result = executor.tick((3, 1))
            assert bridge.frame_count == frame and bridge.save_state() == before
            if result.keys is not None:
                bridge.set_hardware_keys(result.keys)
            if result.status == "complete":
                break
            bridge.step_frame()
        else:
            pytest.fail("Physical step plan never finished")
        assert bridge.get_player_coords_and_map()["x"] == 3
        assert len([event for event in bridge.events if event["kind"] == "step"]) == 2
    finally:
        bridge.close()


def test_collision_replanning_has_a_finite_limit() -> None:
    bridge = SyntheticGBASiliconBridge()
    try:
        collision = np.zeros((3, 8), dtype=np.uint16)
        collision[1, 2] = 1
        _floor(bridge, collision)
        planner = _Planner([[_step((1, 1), (2, 1))]])
        executor = FeedbackPuzzleExecutor(bridge, planner, max_replans=2, step_timeout=3)
        for _ in range(20):
            result = executor.tick((2, 1))
            bridge.set_hardware_keys(result.keys or 0)
            if result.status == "blocked":
                break
            bridge.step_frame()
        assert result.status == "blocked" and planner.calls == 2
        assert executor.tick((2, 1)).status == "blocked"
        assert bridge.get_player_coords_and_map()["x"] == 1
    finally:
        bridge.close()


def test_unexpected_origin_replans_instead_of_replaying_stale_keys() -> None:
    bridge = SyntheticGBASiliconBridge()
    try:
        _floor(bridge)
        bridge.transition_map(0, 0, 3, 1)
        executor = FeedbackPuzzleExecutor(bridge, _Planner([[_step((1, 1), (2, 1))], [_step((3, 1), (2, 1))]]))
        assert executor.tick((2, 1)).status == "replan"
        result = executor.tick((2, 1))
        assert result.status == "keys" and result.keys == bridge.KEY_LEFT
    finally:
        bridge.close()


def test_interaction_waits_for_real_script_feedback_and_replans_afterward() -> None:
    bridge = SyntheticGBASiliconBridge()
    try:
        _floor(bridge)
        interact = {"dest": (0, 0, (2, 1)), "action": "interact", "required_move": 70,
                    "script": "EventScript_StrengthBoulder", "puzzle": "strength_activation", "menu_choice": 0}
        planner = _Planner([[interact], [_step((1, 1), (2, 1))]])
        executor = FeedbackPuzzleExecutor(bridge, planner)
        first = executor.tick((3, 1))
        assert first.status == "directive" and first.required_move == 70
        bridge.write_memory(bridge.memory_map.script + 1, b"\x01")
        waiting = executor.tick((3, 1))
        assert waiting.status == "waiting" and waiting.keys is None
        bridge.write_memory(bridge.memory_map.script + 1, b"\x00")
        assert executor.tick((3, 1)).status == "replan"
        assert executor.tick((3, 1)).keys == bridge.KEY_RIGHT
        assert planner.calls == 2
    finally:
        bridge.close()


def test_same_map_portal_acknowledges_the_actual_landing() -> None:
    bridge = SyntheticGBASiliconBridge()
    try:
        collision = np.ones((3, 8), dtype=np.uint16)
        collision[1, 1:3], collision[1, 5:7] = 0, 0
        tiles = np.zeros_like(collision)
        bridge.configure_map(0, 0, tiles, collision, tiles,
                             warps=[SyntheticWarp(2, 1, 0, 0, 5, 1)])
        bridge.transition_map(0, 0, 1, 1)
        command = {"dest": (0, 0, (2, 1)), "action": "step", "expected_destination": (5, 1),
                   "puzzle": "same_map_portal"}
        executor = FeedbackPuzzleExecutor(bridge, _Planner([[command]]))
        assert executor.tick((6, 1)).status == "directive"
        bridge.set_hardware_keys(bridge.KEY_RIGHT)
        for _ in range(30):
            bridge.step_frame()
            result = executor.tick((6, 1))
            if result.status == "complete":
                break
        assert result.status == "complete" and bridge.get_player_coords_and_map()["x"] == 5
    finally:
        bridge.close()


@pytest.mark.parametrize("boulder_landing,expected", [(True, "complete"), (False, "replan")])
def test_push_requires_observed_boulder_landing_not_just_player_motion(boulder_landing: bool, expected: str) -> None:
    bridge = SyntheticGBASiliconBridge(True)
    try:
        _floor(bridge)
        source = load_map_catalog("emerald")
        command = dict(_step((1, 1), (2, 1)), action="push", pushed_boulder=(2, 1), required_move=70)
        executor = FeedbackPuzzleExecutor(bridge, _Planner([[command]], source))
        assert executor.tick((4, 1)).keys == bridge.KEY_RIGHT
        # Supply the independently observed post-animation RAM fixture. The
        # generic silicon does not implement a native Strength script.
        bridge.transition_map(0, 0, 2, 1)
        if boulder_landing:
            boulder = ObjectEventStruct()
            boulder.flags, boulder.graphicsId = 1, source.constants["OBJ_EVENT_GFX_PUSHABLE_BOULDER"]
            boulder.currentCoords_x, boulder.previousCoords_x = 10, 10
            boulder.currentCoords_y, boulder.previousCoords_y = 8, 8
            bridge.write_memory(bridge.memory_map.objects + 36, bytes(boulder))
        assert executor.tick((4, 1)).status == expected
    finally:
        bridge.close()


def test_real_source_ice_plan_runs_through_feedback_without_skipping_tiles() -> None:
    source = load_map_catalog("emerald")
    key = source.by_name["SootopolisCity_Gym_1F"]
    packed = source.grid(key)
    bridge = SyntheticGBASiliconBridge(True)
    try:
        bridge.configure_map(*key, (packed & 1023).astype(np.uint16), ((packed >> 10) & 3).astype(np.uint16),
                             (packed >> 12).astype(np.uint16), behaviors=source.behavior_grid(key))
        bridge.transition_map(*key, 8, 20)
        executor = FeedbackPuzzleExecutor(bridge, HTNCampaignPlanner(True))
        visited: list[tuple[int, int]] = []
        for _ in range(300):
            result = executor.tick((8, 17))
            if result.keys is not None:
                bridge.set_hardware_keys(result.keys)
            if result.reason == "observed physical puzzle step":
                coords = bridge.get_player_coords_and_map()
                visited.append((coords["x"], coords["y"]))
            if result.status == "complete":
                break
            bridge.step_frame()
        assert len(visited) == len(set(visited)) == 7
        assert visited[-1] == (8, 17)
    finally:
        bridge.close()


def test_puzzle_intent_checkpoint_restores_a_partial_step_without_advancing() -> None:
    bridge, restored = SyntheticGBASiliconBridge(), SyntheticGBASiliconBridge()
    try:
        _floor(bridge)
        planner = _Planner([[_step((1, 1), (2, 1))]])
        executor = FeedbackPuzzleExecutor(bridge, planner)
        bridge.set_hardware_keys(executor.tick((2, 1)).keys or 0)
        for _ in range(8):
            bridge.step_frame()
        executor.tick((2, 1))
        saved = json.loads(json.dumps(executor.state_dict()))
        restored.load_state(bridge.save_state())
        other = FeedbackPuzzleExecutor(restored, planner)
        before, frame = restored.save_state(), restored.frame_count
        other.load_state_dict(saved)
        assert restored.save_state() == before and restored.frame_count == frame
        assert other.state_dict() == executor.state_dict()
        for _ in range(20):
            result = other.tick((2, 1))
            restored.set_hardware_keys(result.keys or 0)
            if result.status == "complete":
                break
            restored.step_frame()
        assert result.status == "complete"
        bad = other.state_dict()
        before_intent = other.state_dict()
        bad["index"] = "invalid"
        with pytest.raises(ValueError):
            other.load_state_dict(bad)
        assert other.state_dict() == before_intent
    finally:
        bridge.close()
        restored.close()


def test_surf_permission_requires_badge_and_living_known_move() -> None:
    bridge = SyntheticGBASiliconBridge(True)
    try:
        party = bridge.get_decrypted_player_party()
        party[0]["moves"][0] = 57
        bridge.configure_party(party)
        assert not can_use_surf(bridge, "emerald")
        bridge.set_event_flag(load_map_catalog("emerald").constants["FLAG_BADGE05_GET"], True)
        assert can_use_surf(bridge, "emerald")
        party[0]["hp"] = 0
        bridge.configure_party(party)
        assert not can_use_surf(bridge, "emerald")
    finally:
        bridge.close()
