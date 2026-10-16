"""Regression coverage for disconnected exits and live route replanning."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core.bridge.ffi_mgba import NativeLibMGBABridge
from core.bridge.gba_structs import ObjectEventStruct
from core.bridge.synthetic_gba_silicon import SyntheticGBASiliconBridge
from core.game_data import ROM_NAMES
from core.navigation.constrained_astar import GbaConstrainedAStar
from core.navigation.source_data import ROOT, load_map_catalog
from core.navigation.warp_topology import HierarchicalWarpGraph, WarpTransition
from core.planner.htn_planner import HTNCampaignPlanner


def test_reachability_respects_directional_ledge_costs() -> None:
    collision = np.array([[0, 1, 0, 0]], dtype=np.uint16)
    elevation = np.array([[3, 3, 2, 2]], dtype=np.uint16)
    behavior = np.array([[0, 0x38, 0, 0]], dtype=np.uint16)
    planner = GbaConstrainedAStar(collision, elevation, behavior)
    assert planner.reachable_costs((0, 0, 3)) == {(0, 0): 0.0, (2, 0): 2.0, (3, 0): 3.0}
    assert (0, 0) not in planner.reachable_costs((2, 0, 2))


@pytest.mark.parametrize("behavior,blocked_sides", [
    (0x30, (4,)), (0x31, (3,)), (0x32, (2,)), (0x33, (1,)),
    (0x34, (4, 2)), (0x35, (3, 2)), (0x36, (4, 1)), (0x37, (3, 1)),
])
@pytest.mark.parametrize("facing,dx,dy,opposite", [(1, 0, 1, 2), (2, 0, -1, 1),
                                                    (3, -1, 0, 4), (4, 1, 0, 3)])
@pytest.mark.parametrize("at_origin", [False, True])
def test_directional_wall_checks_departure_and_opposite_entry_sides(
        behavior: int, blocked_sides: tuple[int, ...], facing: int, dx: int, dy: int,
        opposite: int, at_origin: bool) -> None:
    """Independent table from pret's directional collision predicates."""
    collision = np.ones((3, 3), dtype=np.uint16)
    origin, destination = (1, 1), (1 + dx, 1 + dy)
    collision[origin[1], origin[0]] = collision[destination[1], destination[0]] = 0
    elevation, behaviors = np.zeros_like(collision), np.zeros_like(collision)
    wall = origin if at_origin else destination
    behaviors[wall[1], wall[0]] = behavior
    path = GbaConstrainedAStar(collision, elevation, behaviors).find_optimal_path((*origin, 0), destination)
    blocked = (facing if at_origin else opposite) in blocked_sides
    assert (path is None) == blocked
    if not blocked:
        assert path == [destination]


def test_mt_moon_b2f_wall_edges_require_other_floors_to_reach_miguel() -> None:
    source = load_map_catalog("firered")
    key = source.by_name["MtMoon_B2F"]
    packed = source.grid(key)
    collision, elevation = ((packed >> 10) & 3).astype(np.uint16), (packed >> 12).astype(np.uint16)
    behavior = source.behavior_grid(key)
    direct = GbaConstrainedAStar(collision, elevation, behavior)
    assert direct.find_optimal_path((40, 13, 3), (13, 12)) is None
    graph = HierarchicalWarpGraph.from_source("firered")
    route = graph.find_route_to_tile(key, key, [(13, 10), (13, 12), (12, 11), (14, 11)], current_xy=(40, 13))
    assert route is not None
    assert [edge.destination for edge in route] == [(1, 2), (1, 1), (1, 2), (1, 3)]
    assert route[0].source_xy == (31, 11) and route[-1].destination_xy == (25, 21)


@pytest.mark.native
def test_native_mt_moon_section_route_preserves_original_checkpoint() -> None:
    import json

    directory = ROOT / "artifacts/campaigns/native/firered"
    pointer = directory / "latest.json"
    dll, rom = ROOT / "native/build/ares_mgba.dll", ROOT.parent / ROM_NAMES["firered"]
    if not pointer.is_file() or not dll.is_file() or not rom.is_file():
        pytest.skip("Native MtMoon campaign checkpoint and matching shim/ROM required")
    state_path = directory / json.loads(pointer.read_text(encoding="utf-8"))["directory"] / "emulator.state"
    checkpoint = state_path.read_bytes()
    bridge = NativeLibMGBABridge(dll, rom)
    try:
        bridge.load_state(checkpoint)
        coords = bridge.get_player_coords_and_map()
        if (coords["map_group"], coords["map_num"]) != (1, 3) or coords["x"] < 30 or coords["y"] > 13:
            pytest.skip("Campaign has moved beyond the recorded east MtMoon B2F section")
        before, frame = bridge.save_state(), bridge.frame_count
        graph = HierarchicalWarpGraph.from_source("firered")
        route = graph.find_route_to_tile((1, 3), (1, 3), [(13, 10), (13, 12), (12, 11), (14, 11)],
                                         current_xy=(coords["x"], coords["y"]), bridge=bridge)
        assert route and route[0].source_xy == (31, 11)
        assert [edge.destination for edge in route] == [(1, 2), (1, 1), (1, 2), (1, 3)]
        assert bridge.save_state() == before and bridge.frame_count == frame
        assert state_path.read_bytes() == checkpoint
    finally:
        bridge.close()


def test_a_route_can_end_at_a_warp_but_cannot_walk_through_it() -> None:
    tiles = np.zeros((1, 4), dtype=np.uint16)
    planner = GbaConstrainedAStar(tiles, tiles)
    planner.terminal_tiles = {(1, 0)}
    assert planner.find_optimal_path((0, 0, 0), (1, 0)) == [(1, 0)]
    assert planner.find_optimal_path((0, 0, 0), (3, 0)) is None
    assert set(planner.reachable_costs((0, 0, 0))) == {(0, 0), (1, 0)}
    assert planner.find_optimal_path((1, 0, 0), (3, 0)) == [(2, 0), (3, 0)]


def test_route104_uses_reachable_woods_entrance_instead_of_closest_tree_pocket() -> None:
    graph = HierarchicalWarpGraph.from_source("emerald")
    source = load_map_catalog("emerald")
    route = graph.find_route(source.by_name["Route104"], source.by_name["PetalburgWoods"], current_xy=(39, 63))
    assert route is not None and len(route) == 1
    assert route[0].source_xy in {(10, 38), (11, 38)}
    assert route[0].source_xy not in {(32, 42), (33, 42)}


def test_disconnected_island_target_returns_before_any_tile_search(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = HierarchicalWarpGraph.from_source("firered")

    def forbid_tile_search(*args: object, **kwargs: object) -> None:
        raise AssertionError("Topologically unreachable maps must not enumerate local tile components")

    monkeypatch.setattr(graph, "_walking_costs", forbid_tile_search)
    source = load_map_catalog("firered")
    assert graph.find_route(source.by_name["Route1"], source.by_name["MtEmber_Exterior"], current_xy=(13, 35)) is None


def test_live_actor_changes_invalidate_exit_reachability_cache() -> None:
    source = load_map_catalog("emerald")
    key, woods = source.by_name["Route104"], source.by_name["PetalburgWoods"]
    packed = source.grid(key)
    bridge = SyntheticGBASiliconBridge(True)
    try:
        bridge.configure_map(*key, (packed & 1023).astype(np.uint16), ((packed >> 10) & 3).astype(np.uint16),
                             (packed >> 12).astype(np.uint16), behaviors=source.behavior_grid(key))
        bridge.transition_map(*key, 39, 63)
        graph = HierarchicalWarpGraph.from_source("emerald")
        first = graph.next_transition(key, woods, (39, 63), bridge=bridge)
        assert first is not None and first.source_xy == (11, 38)
        npc = ObjectEventStruct()
        npc.flags, npc.elevations, npc.localId = 1, 3, 1
        npc.currentCoords_x, npc.previousCoords_x = 18, 18
        npc.currentCoords_y, npc.previousCoords_y = 45, 45
        bridge.write_memory(bridge.memory_map.objects + 36, bytes(npc))
        second = graph.next_transition(key, woods, (39, 63), bridge=bridge)
        assert second is not None and second.source_xy == (10, 38)
    finally:
        bridge.close()


def test_travel_estimate_includes_walking_in_every_map() -> None:
    route = [WarpTransition((0, 0), (0, 1), (20, 0), (0, 0)),
             WarpTransition((0, 1), (0, 2), (30, 0), (0, 0))]
    assert HierarchicalWarpGraph.estimate_route_cost(route, (10, 0)) == 42.0


def test_briney_return_uses_explicit_house_warp_before_route104_exit() -> None:
    """The script's final warp, not its intermediate boat animation, is arrival."""
    source = load_map_catalog("emerald")
    graph = HierarchicalWarpGraph.from_source("emerald")
    dewford, house, route104 = (source.by_name[name] for name in
                                 ("DewfordTown", "Route104_MrBrineysHouse", "Route104"))
    return_boat = next(edge for edge in graph.adjacency[dewford]
                       if edge.kind == "script" and edge.destination == house)
    assert return_boat.destination_xy == (5, 4)
    assert return_boat.script == "DewfordTown_EventScript_Briney"
    assert (return_boat.npc_local_id, return_boat.menu_choice, return_boat.menu_default) == (2, 0, 2)
    assert return_boat.required_milestones == ("DEVON_GOODS",)
    assert return_boat.activation == "interact"
    assert not any(edge.kind == "script" and edge.destination == route104 for edge in graph.adjacency[dewford])
    route = graph.find_route(dewford, route104, frozenset({"DEVON_GOODS", "STEVEN_LETTER"}), (11, 10))
    assert route is not None and [edge.destination for edge in route] == [house, route104]
    assert route[0] == return_boat and route[1].kind == "warp"


def test_training_route_story_gates_cover_forests_and_rustboro() -> None:
    firered = HTNCampaignPlanner()
    emerald = HTNCampaignPlanner(True)
    assert firered.training_map_prerequisites("ViridianForest") == ("OAK_PARCEL",)
    assert firered.training_map_prerequisites("SafariZone_Center") == ("ELITE_FOUR",)
    assert firered.training_map_prerequisites("OneIsland_KindleRoad") == ("ELITE_FOUR",)
    assert firered.training_map_prerequisites("MtEmber_Exterior") == ("ELITE_FOUR",)
    assert emerald.training_map_prerequisites("PetalburgWoods") == ("WALLY_TUTORIAL",)
    assert emerald.training_map_prerequisites("Route116") == ("PETALBURG_WOODS_AQUA",)
    assert emerald.training_map_prerequisites("JaggedPass") == ("MT_CHIMNEY",)
    assert emerald.training_map_prerequisites("MtPyre_Exterior") == ("GYM_5_NORMAN", "HM_SURF")


def test_silph_lapras_gift_requires_engine_flag_and_species_in_party() -> None:
    planner = HTNCampaignPlanner()
    gift = next(task for task in planner.campaign_dag if task.milestone_id == "LAPRAS_GIFT")
    sabrina = next(task for task in planner.campaign_dag if task.milestone_id == "GYM_6_SABRINA")
    assert gift.dependencies == ("SILPH_CO",) and sabrina.dependencies == ("LAPRAS_GIFT",)
    assert gift.completion_flag_id == 0x246 and gift.map_name == "SilphCo_7F"
    bridge = SyntheticGBASiliconBridge()
    try:
        assert not gift.complete(bridge, [{"species": 131}])
        bridge.set_event_flag(0x246, True)
        assert not gift.complete(bridge, [{"species": 1}])
        assert gift.complete(bridge, [{"species": 1}, {"species": 131}])
    finally:
        bridge.close()


def _ready_party_member(level: int, hp: int, maximum: int = 30,
                        status: int = 0, offensive_pp: int = 20) -> dict[str, Any]:
    return {"species": 1, "level": level, "hp": hp, "maxHP": maximum,
            "status": status, "moves": [33, 45, 0, 0], "pp": [offensive_pp, 30, 0, 0]}


@pytest.mark.parametrize("lead,reason", [
    (_ready_party_member(14, 0), "fainted"),
    (_ready_party_member(14, 8), "low HP"),
    (_ready_party_member(14, 30, status=8), "status"),
    (_ready_party_member(14, 30, status=0x80), "status"),
    (_ready_party_member(14, 30, status=0x10), "status"),
    (_ready_party_member(14, 30, offensive_pp=0), "offensive PP"),
])
def test_unready_campaign_lead_heals_even_when_healthy_captures_hide_the_problem(
        lead: dict[str, Any], reason: str) -> None:
    planner = HTNCampaignPlanner()
    source = load_map_catalog("firered")
    key = source.by_name["ViridianCity_PokemonCenter_1F"]
    packed = source.grid(key)
    bridge = SyntheticGBASiliconBridge()
    try:
        bridge.configure_map(*key, (packed & 1023).astype(np.uint16), ((packed >> 10) & 3).astype(np.uint16),
                             (packed >> 12).astype(np.uint16), behaviors=source.behavior_grid(key))
        bridge.transition_map(*key, 7, 4)
        party = [lead, _ready_party_member(3, 30), _ready_party_member(3, 30)]
        assert sum(int(mon["hp"]) for mon in party) / sum(int(mon["maxHP"]) for mon in party) > 0.4
        before = bridge.save_state()
        directive = planner.get_active_directive(bridge, party)
        assert directive["mode"] == "HEAL_POKECENTER"
        assert tuple(directive["dest"][:2]) == key
        assert reason in directive["reason"]
        assert bridge.save_state() == before
    finally:
        bridge.close()


def test_a_healthy_lead_with_offensive_pp_does_not_heal_for_spent_status_move_pp() -> None:
    bridge = SyntheticGBASiliconBridge()
    try:
        member = _ready_party_member(14, 30)
        member["pp"] = [20, 0, 0, 0]
        assert HTNCampaignPlanner()._healing_reason(bridge, [member]) is None
    finally:
        bridge.close()


@pytest.mark.native
def test_saved_native_emerald_stall_routes_to_reachable_exit_without_mutating_state() -> None:
    directory = ROOT / "artifacts/campaigns/native/emerald"
    states = sorted(directory.glob("frame-000003881069-*/emulator.state"))
    dll, rom = ROOT / "native/build/ares_mgba.dll", ROOT.parent / ROM_NAMES["emerald"]
    if not states or not dll.is_file() or not rom.is_file():
        pytest.skip("Recorded native Route104 stall checkpoint, matching ROM and shim required")
    state_path: Path = states[0]
    checkpoint_bytes = state_path.read_bytes()
    bridge = NativeLibMGBABridge(dll, rom, True)
    try:
        bridge.load_state(checkpoint_bytes)
        coords = bridge.get_player_coords_and_map()
        assert (coords["map_group"], coords["map_num"], coords["x"], coords["y"]) == (0, 19, 39, 63)
        before, frame = bridge.save_state(), bridge.frame_count
        graph = HierarchicalWarpGraph.from_source("emerald")
        edge = graph.next_transition((0, 19), (24, 11), (39, 63), bridge=bridge)
        assert edge is not None and edge.source_xy in {(10, 38), (11, 38)}
        _, collision, elevation = bridge.extract_live_map_matrices()
        planner = GbaConstrainedAStar(collision, elevation, bridge.extract_behavior_map(), "emerald")
        planner.register_object_events(bridge.get_object_events(), bridge.read_u8(bridge.memory_map.avatar + 5))
        assert planner.find_optimal_path((39, 63, coords["z"]), edge.source_xy)
        assert bridge.frame_count == frame and bridge.save_state() == before
        assert state_path.read_bytes() == checkpoint_bytes
    finally:
        bridge.close()
