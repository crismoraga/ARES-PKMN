"""Verify terrain physics, physical timing, puzzle searches and source-backed goals."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.battle.menu_cursor_controller import BattleMenuCursorController
from core.bridge.gba_structs import ObjectEventStruct
from core.bridge.synthetic_gba_silicon import SyntheticGBASiliconBridge, SyntheticWarp
from core.navigation.constrained_astar import GbaConstrainedAStar
from core.navigation.movement_fsm import MovementMode, StateFeedbackMovementController
from core.navigation.puzzles import solve_ice_floor, solve_strength_puzzle
from core.navigation.puzzles import plan_sootopolis_ice_commands
from core.navigation.source_data import load_map_catalog
from core.navigation.warp_topology import HierarchicalWarpGraph, WarpTransition
from core.planner.htn_planner import CampaignMilestone, HTNCampaignPlanner


def test_elevation_zero_allows_transition_and_fifteen_preserves_layer() -> None:
    collision = np.zeros((1, 4), dtype=np.uint16)
    elevation = np.array([[3, 15, 4, 4]], dtype=np.uint16)
    assert GbaConstrainedAStar(collision, elevation).find_optimal_path((0, 0, 3), (3, 0)) is None
    elevation[0, 1] = 0
    assert GbaConstrainedAStar(collision, elevation).find_optimal_path((0, 0, 3), (3, 0)) == [(1, 0), (2, 0), (3, 0)]


def test_ledge_requires_direction_and_free_landing() -> None:
    collision = np.array([[0, 1, 0, 0]], dtype=np.uint16)
    elevation = np.array([[3, 3, 2, 2]], dtype=np.uint16)
    behavior = np.array([[0, 0x38, 0, 0]], dtype=np.uint16)
    planner = GbaConstrainedAStar(collision, elevation, behavior)
    assert planner.find_optimal_path((0, 0, 3), (3, 0)) == [(2, 0), (3, 0)]
    assert planner.find_optimal_path((2, 0, 2), (0, 0)) is None
    collision[0, 2] = 1
    assert GbaConstrainedAStar(collision, elevation, behavior).find_optimal_path((0, 0, 3), (3, 0)) is None


def test_trainer_sight_stops_at_wall_and_path_avoids_cone() -> None:
    collision = np.zeros((5, 5), dtype=np.uint16)
    elevation = np.zeros_like(collision)
    planner = GbaConstrainedAStar(collision, elevation)
    planner.register_trainer_cone(2, 0, 1, 3, penalty=250)
    path = planner.find_optimal_path((0, 2, 0), (4, 2))
    assert path is not None and (2, 2) not in path
    collision[2, 2] = 1
    planner = GbaConstrainedAStar(collision, elevation)
    planner.register_trainer_cone(2, 0, 1, 4)
    assert (2, 1) in planner.trainer_hazard_cost and (2, 3) not in planner.trainer_hazard_cost


def test_npc_occupancy_uses_unbordered_coordinates() -> None:
    planner = GbaConstrainedAStar(np.zeros((1, 3), dtype=np.uint16), np.zeros((1, 3), dtype=np.uint16))
    player, npc = ObjectEventStruct(), ObjectEventStruct()
    npc.flags, npc.currentCoords_x, npc.currentCoords_y = 1, 8, 7
    planner.register_object_events([player, npc], 0)
    assert planner.find_optimal_path((0, 0, 0), (2, 0)) is None


@pytest.mark.parametrize("mode,frames", [("walk", 16), ("run", 8), ("bike", 4)])
def test_fsm_holds_direction_until_physical_arrival(mode: MovementMode, frames: int) -> None:
    bridge = SyntheticGBASiliconBridge()
    try:
        bridge.set_movement_mode(mode)
        fsm = StateFeedbackMovementController(bridge)
        assert fsm.request_step(bridge.KEY_RIGHT, mode)
        before = bridge.frame_count
        assert fsm.tick() is None and bridge.frame_count == before
        for _ in range(frames):
            assert fsm.tick() is None
            bridge.step_frame()
        event = fsm.tick()
        assert event is not None and event.status == "arrived" and event.frames == frames
        assert bridge.get_player_coords_and_map()["x"] == event.origin[0] + 1
    finally:
        bridge.close()


def test_collision_deadline_and_persisted_obstacle(tmp_path: Path) -> None:
    bridge = SyntheticGBASiliconBridge()
    try:
        tiles = np.zeros((3, 3), dtype=np.uint16)
        collision = tiles.copy()
        collision[1, 2] = 1
        bridge.configure_map(0, 0, tiles, collision, tiles)
        bridge.transition_map(0, 0, 1, 1)
        controller = StateFeedbackMovementController(bridge)
        assert controller.request_step(bridge.KEY_RIGHT)
        for _ in range(24):
            assert controller.tick() is None
            bridge.step_frame()
        result = controller.tick()
        assert result is not None and result.status == "blocked" and result.frames == 24
        controller.save_obstacles(tmp_path / "obstacles.json")
        restored = StateFeedbackMovementController(bridge)
        restored.load_obstacles(tmp_path / "obstacles.json")
        assert restored.blocked_edges == {(0, 0, 1, 1, 2, 1)}
    finally:
        bridge.close()


def test_warp_changes_map_with_same_local_coordinates() -> None:
    bridge = SyntheticGBASiliconBridge()
    try:
        tiles = np.zeros((3, 4), dtype=np.uint16)
        bridge.configure_map(0, 0, tiles, tiles, tiles, warps=[SyntheticWarp(2, 1, 0, 1, 1, 1)])
        bridge.configure_map(0, 1, tiles, tiles, tiles)
        bridge.transition_map(0, 0, 1, 1)
        movement = StateFeedbackMovementController(bridge)
        assert movement.request_step(bridge.KEY_RIGHT)
        for _ in range(16):
            bridge.step_frame()
        result = movement.tick()
        assert result is not None and result.status == "map_changed" and result.destination == result.origin
    finally:
        bridge.close()


def test_dijkstra_respects_direction_costs_dependencies_and_blocking() -> None:
    direct = WarpTransition((0, 0), (0, 2), (1, 1), (2, 2), cost=10)
    first = WarpTransition((0, 0), (0, 1), (1, 1), (2, 2), cost=1, required_milestones=("CUT",))
    second = WarpTransition((0, 1), (0, 2), (2, 2), (3, 3), cost=1)
    graph = HierarchicalWarpGraph([direct, first, second])
    assert graph.find_route((0, 0), (0, 2)) == [direct]
    assert graph.find_route((0, 0), (0, 2), frozenset({"CUT"})) == [first, second]
    assert graph.find_route((0, 2), (0, 0)) is None
    graph.block_transition(first)
    assert graph.find_route((0, 0), (0, 2), frozenset({"CUT"})) == [direct]


def test_strength_and_ice_have_actual_legal_solution_paths() -> None:
    collision = np.ones((3, 6), dtype=np.uint16)
    collision[1, 1:5] = 0
    path = solve_strength_puzzle(collision, (1, 1), frozenset({(2, 1)}), frozenset({(4, 1)}))
    assert path is not None and len(path) == 2 and all(step.pushed_boulder is not None for step in path)
    assert solve_ice_floor(frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}), (0, 0), (1, 0)) == [(0, 1), (1, 1), (1, 0)]


@pytest.mark.parametrize("emerald", [False, True])
def test_real_dag_uses_matching_source_flags_and_valid_event_targets(emerald: bool) -> None:
    planner = HTNCampaignPlanner(is_emerald=emerald)
    game = "emerald" if emerald else "firered"
    source = load_map_catalog(game)
    assert len(source.maps) == (518 if emerald else 425)
    flags = {node.completion_flag_id for node in planner.campaign_dag}
    assert set(range(0x867, 0x86F) if emerald else range(0x820, 0x828)) <= flags
    assert all(node.map_key in source.maps for node in planner.campaign_dag)
    assert len({node.milestone_id for node in planner.campaign_dag}) == len(planner.campaign_dag)
    assert len(planner.campaign_dag) >= (50 if emerald else 38)


def test_campaign_completion_requires_observed_flag() -> None:
    bridge = SyntheticGBASiliconBridge()
    try:
        node = CampaignMilestone("OAK", "A test goal", 0x100, 0, 0, (1, 1), 1, [])
        planner = HTNCampaignPlanner(milestones=[node])
        party = bridge.get_decrypted_player_party()
        assert planner.get_active_directive(bridge, party)["mode"] == "EXECUTE_MILESTONE"
        assert planner.completed_milestones(bridge, party) == frozenset()
        bridge.set_event_flag(0x100, True)  # Fixture setup, never native progress.
        assert planner.get_active_directive(bridge, party)["mode"] == "POSTGAME_COMPLETE"
    finally:
        bridge.close()


def test_exact_movement_checkpoint_mid_step_rebases_deadline() -> None:
    bridge = SyntheticGBASiliconBridge()
    try:
        controller = StateFeedbackMovementController(bridge)
        assert controller.request_step(bridge.KEY_RIGHT)
        for _ in range(7):
            bridge.step_frame()
            assert controller.tick() is None
        emulator_state, controller_state = bridge.save_state(), controller.state_dict()
        for _ in range(9):
            bridge.step_frame()
        expected = controller.tick()
        expected_emulator_state = bridge.save_state()
        bridge.load_state(emulator_state)
        restored = StateFeedbackMovementController(bridge)
        restored.load_state_dict(controller_state)
        assert restored.pending and bridge.frame_count == 7
        for _ in range(9):
            assert restored.tick() is None
            bridge.step_frame()
        assert restored.tick() == expected
        assert bridge.save_state() == expected_emulator_state
    finally:
        bridge.close()


def test_movement_restore_does_not_erase_another_controllers_held_key() -> None:
    bridge = SyntheticGBASiliconBridge()
    try:
        movement = StateFeedbackMovementController(bridge)
        bridge.set_hardware_keys(bridge.KEY_RIGHT)
        before = bridge.save_state()
        movement.load_state_dict(movement.state_dict())
        assert bridge.save_state() == before
        with pytest.raises(ValueError, match="Invalid pending"):
            movement.load_state_dict({"pending": {"origin": [1, 1], "origin_map": [0, 0],
                                                  "target": [2, 1], "direction": 999,
                                                  "mode": "walk", "elapsed_frames": 7}})
        assert bridge.save_state() == before
    finally:
        bridge.close()


def test_blocked_two_tile_ledge_persists_the_actual_requested_edge() -> None:
    bridge = SyntheticGBASiliconBridge()
    try:
        tiles = np.arange(4, dtype=np.uint16).reshape(1, 4)
        collision = np.zeros_like(tiles)
        collision[0, 2] = 1
        behavior = np.zeros_like(tiles)
        behavior[0, 1] = 0x38
        bridge.configure_map(0, 0, tiles, collision, np.zeros_like(tiles), behaviors=behavior)
        bridge.transition_map(0, 0, 0, 0)
        movement = StateFeedbackMovementController(bridge)
        assert movement.move_toward((2, 0))
        saved = movement.state_dict()
        movement.load_state_dict(saved)
        for _ in range(24):
            bridge.step_frame()
        assert movement.tick() is not None
        assert (0, 0, 0, 0, 2, 0) in movement.blocked_edges
        assert (0, 0, 0, 0, 1, 0) not in movement.blocked_edges
    finally:
        bridge.close()


def test_emerald_portals_and_boats_have_real_activation_metadata() -> None:
    graph = HierarchicalWarpGraph.from_source("emerald")
    catalog = load_map_catalog("emerald")
    allowed = graph.warp_behavior_ids("emerald")
    assert {0x0E, 0x0F, 0x29, 0x8D} <= allowed
    mossdeep = catalog.by_name["MossdeepCity_Gym"]
    assert any(edge.behavior == 0x0E for edge in graph.adjacency[mossdeep])
    dewford, route109 = catalog.by_name["DewfordTown"], catalog.by_name["Route109"]
    boat = next(edge for edge in graph.adjacency[dewford] if edge.destination == route109 and edge.kind == "script")
    assert (boat.npc_local_id, boat.menu_choice, boat.menu_default) == (2, 1, 2)
    assert boat.activation == "interact" and boat.script == "DewfordTown_EventScript_Briney"
    assert "STEVEN_LETTER" in boat.required_milestones


def test_same_map_teleporters_join_disconnected_components() -> None:
    collision = np.ones((3, 8), dtype=np.uint16)
    collision[1, 1:3], collision[1, 5:7] = 0, 0
    grid = np.zeros_like(collision)
    portal = WarpTransition((0, 0), (0, 0), (2, 1), (5, 1))
    graph = HierarchicalWarpGraph([portal])
    assert graph.find_intra_map_route((0, 0), (1, 1, 0), (6, 1), collision, grid, grid) == [portal]
    graph.block_transition(portal)
    assert graph.find_intra_map_route((0, 0), (1, 1, 0), (6, 1), collision, grid, grid) is None


def test_strength_access_moves_a_boulder_before_walking_to_the_exit() -> None:
    collision = np.ones((3, 7), dtype=np.uint16)
    collision[1, 1:6] = 0
    solution = solve_strength_puzzle(collision, (1, 1), frozenset({(2, 1)}), frozenset(), exit_tile=(4, 1))
    assert solution is not None and solution[-1].destination == (4, 1)
    assert len(solution) == 3 and all(step.pushed_boulder is not None for step in solution)


def test_campaign_dependencies_prevent_future_flags_being_counted_early() -> None:
    bridge = SyntheticGBASiliconBridge()
    try:
        nodes = [CampaignMilestone("FIRST", "first", 0x100, 0, 0, (1, 1), 1, []),
                 CampaignMilestone("LATER", "later", 0x101, 0, 0, (1, 1), 1, [], dependencies=("FIRST",))]
        planner = HTNCampaignPlanner(milestones=nodes)
        bridge.set_event_flag(0x101, True)
        assert planner.completed_milestones(bridge, bridge.get_decrypted_player_party()) == frozenset()
        bridge.set_event_flag(0x100, True)
        assert planner.completed_milestones(bridge, bridge.get_decrypted_player_party()) == frozenset({"FIRST", "LATER"})
    finally:
        bridge.close()


def test_native_campaign_semantics_separate_hidden_objects_and_story_progress() -> None:
    firered = HTNCampaignPlanner()
    emerald = HTNCampaignPlanner(True)
    fr = {node.milestone_id: node for node in firered.milestones}
    em = {node.milestone_id: node for node in emerald.milestones}
    assert fr["ROCKET_LIFT_KEY"].completion_flag_id == 0x2A5
    assert fr["SILPH_SCOPE"].completion_flag_id == -1
    assert fr["SILPH_SCOPE"].completion_conditions[0].kind == "item"
    assert em["RAYQUAZA_AWAKEN"].target_coords == (14, 9)
    assert em["RAYQUAZA_AWAKEN"].action == "trigger" and em["RAYQUAZA_AWAKEN"].completion_conditions[0].value == 1
    assert em["CHAMPION"].completion_flag_id == 0x64F
    assert em["ELITE_FOUR"].completion_flag_id == 0x864
    assert emerald.training_map_prerequisites("Route112") == ("GYM_3_WATTSON", "HM_ROCK_SMASH")


@pytest.mark.native
@pytest.mark.parametrize("game", ["firered", "emerald"])
def test_real_fresh_rom_bootstrap_has_no_completed_campaign_objectives(game: str) -> None:
    from core.bridge.ffi_mgba import NativeLibMGBABridge
    from core.game_data import ROOT, ROM_NAMES, Game
    from typing import cast

    dll, state = ROOT / "native/build/ares_mgba.dll", ROOT / "artifacts" / game / "bootstrap.state"
    if not dll.exists() or not state.exists():
        pytest.skip("Native mGBA and initial bootstrap artifact required")
    bridge = NativeLibMGBABridge(str(dll), str(ROOT.parent / ROM_NAMES[cast(Game, game)]), game == "emerald")
    try:
        bridge.load_state(state.read_bytes())
        before = bridge.save_state()
        planner = HTNCampaignPlanner(game == "emerald")
        assert planner.completed_milestones(bridge, bridge.get_decrypted_player_party()) == frozenset()
        assert bridge.save_state() == before
    finally:
        bridge.close()


def test_route1_mart_route_pays_walking_cost_in_every_map_and_never_loops() -> None:
    graph = HierarchicalWarpGraph.from_source("firered")
    route = graph.find_route((3, 19), (5, 3), current_xy=(12, 39))
    assert route is not None
    assert [edge.destination for edge in route] == [(3, 1), (5, 3)]
    # Real local walking costs include the ledge detour north of the start.
    assert route[0].source_xy == (13, 0)
    source = load_map_catalog("firered")
    pallet = source.by_name["PalletTown"]
    assert not any(edge.destination == source.by_name["Route21_North"] for edge in graph.transitions_from(pallet))


def test_scripted_petalburg_doors_are_not_discarded_as_unused_floor_warps() -> None:
    source = load_map_catalog("emerald")
    graph = HierarchicalWarpGraph.from_source("emerald")
    key = source.by_name["PetalburgCity_Gym"]
    doors = [edge for edge in graph.adjacency[key] if edge.destination == key and edge.kind == "script"]
    assert len(doors) == 12
    speed = next(edge for edge in doors if edge.script.endswith("SpeedRoomDoor"))
    assert speed.source_xy == (1, 105) and speed.destination_xy == (7, 85)
    confusion = next(edge for edge in doors if edge.script.endswith("ConfusionRoomDoor"))
    assert confusion.required_flags == (0x500 + source.constants["TRAINER_RANDALL"],)


@pytest.mark.parametrize("start_y,steps,exit_y", [(20, 7, 17), (15, 19, 12), (10, 38, 6)])
def test_all_three_ice_tiers_produce_legal_commands_from_official_layout(start_y: int, steps: int, exit_y: int) -> None:
    source = load_map_catalog("emerald")
    key = source.by_name["SootopolisCity_Gym_1F"]
    packed = source.grid(key)
    bridge = SyntheticGBASiliconBridge(True)
    try:
        bridge.configure_map(*key, (packed & 1023).astype(np.uint16), ((packed >> 10) & 3).astype(np.uint16),
                             (packed >> 12).astype(np.uint16), behaviors=source.behavior_grid(key))
        bridge.transition_map(*key, 8, start_y)
        before = bridge.save_state()
        commands = plan_sootopolis_ice_commands(bridge, source)
        assert bridge.save_state() == before  # Planning cannot advance or modify RAM.
        assert commands is not None and len(commands) == steps
        visited: set[tuple[int, int]] = set()
        for command in commands:
            origin, destination = command["expected_origin"], command["dest"][2]
            assert abs(destination[0] - origin[0]) + abs(destination[1] - origin[1]) == 1
            assert destination not in visited and command["action"] == "step"
            visited.add(destination)
        assert commands[-1]["dest"][2] == (8, exit_y)
    finally:
        bridge.close()


def test_battle_menu_controller_pokedex_and_nickname_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = SyntheticGBASiliconBridge(True)
    try:
        monkeypatch.setattr(bridge, "is_in_battle", lambda: (True, False))
        controller = BattleMenuCursorController(bridge)
        assert "Task_HandleCaughtMonPageInput" in controller._native_symbols
        assert "Task_DisplayCaughtMonDexPage" in controller._native_symbols
        assert "Task_ExitCaughtMonPage" in controller._native_symbols

        # Verify pokedex mode pulses KEY_A
        monkeypatch.setattr(controller, "menu_mode", lambda: "pokedex")
        assert controller.tick() == bridge.KEY_A

        # Reset hold/release state between tests
        controller.cancel()

        # Verify caught_nickname mode pulses KEY_B to decline nickname
        monkeypatch.setattr(controller, "menu_mode", lambda: "caught_nickname")
        assert controller.tick() == bridge.KEY_B

        controller.cancel()

        # Verify battle_yesno mode pulses KEY_B
        monkeypatch.setattr(controller, "menu_mode", lambda: "battle_yesno")
        assert controller.tick() == bridge.KEY_B
    finally:
        bridge.close()

