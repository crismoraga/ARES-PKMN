"""Off-camera object templates keep live and static routing consistent.

Regression for the FireRed Route 2 deadlock: Cut trees outside the camera are
absent from gObjectEvents, so the planner alternated between a phantom path
through one tree and a round trip through Viridian City that "forgot" it.
"""
from __future__ import annotations

import struct

import numpy as np

from core.bridge.gba_structs import ObjectEventStruct
from core.bridge.synthetic_gba_silicon import SyntheticGBASiliconBridge
from core.navigation.constrained_astar import GbaConstrainedAStar
from core.navigation.obstacles import (
    HM_CLEAR_COST, FlagSnapshot, obstacle_model, runtime_template_objects, template_objects,
)
from core.navigation.source_data import load_map_catalog
from core.navigation.warp_topology import HierarchicalWarpGraph

ROUTE2 = (3, 20)


def _route2_bridge(x: int, y: int) -> SyntheticGBASiliconBridge:
    source = load_map_catalog("firered")
    packed = source.grid(ROUTE2)
    bridge = SyntheticGBASiliconBridge(False)
    bridge.configure_map(*ROUTE2, (packed & 1023).astype(np.uint16), ((packed >> 10) & 3).astype(np.uint16),
                         (packed >> 12).astype(np.uint16), behaviors=source.behavior_grid(ROUTE2))
    bridge.transition_map(*ROUTE2, x, y, relocate=False)
    return bridge


def test_route2_cut_trees_are_classified_from_source_templates() -> None:
    trees = {template.xy: template for template in template_objects("firered", ROUTE2) if template.kind == "cut"}
    assert {(15, 69), (16, 62)} <= set(trees)
    assert all(template.temporary for template in trees.values())  # FLAG_TEMP_* respawn on map entry
    blocked = obstacle_model("firered", ROUTE2).blocked
    assert {(15, 69), (16, 62)} <= blocked
    usable = obstacle_model("firered", ROUTE2, hm_usable={"cut"})
    assert dict(usable.removable)[(15, 69)] == HM_CLEAR_COST and (15, 69) not in usable.blocked


def test_static_route_from_both_oscillation_tiles_uses_viridian_forest() -> None:
    graph = HierarchicalWarpGraph.from_source("firered")
    source = load_map_catalog("firered")
    gym = source.by_name["PewterCity_Gym"]
    for tile in ((10, 76), (10, 77), (10, 79)):
        route = graph.find_route(ROUTE2, gym, current_xy=tile)
        assert route is not None
        assert source.maps[route[0].destination]["name"] == "Route2_ViridianForest_SouthEntrance"
        assert "ViridianForest" in {source.maps[edge.destination]["name"] for edge in route}


def test_live_route_cannot_walk_through_an_unspawned_tree() -> None:
    bridge = _route2_bridge(10, 77)
    try:
        graph = HierarchicalWarpGraph.from_source("firered")
        walking = graph.live_walking_costs(bridge)
        assert walking is not None
        # The east building door is sealed by both Cut trees; no live actor is
        # spawned in this fixture, exactly like an off-camera tree on hardware.
        assert (19, 46) not in walking and (18, 46) not in walking
        assert (5, 51) in walking or (6, 51) in walking  # Viridian Forest gate
        first = graph.next_transition(ROUTE2, load_map_catalog("firered").by_name["PewterCity_Gym"], (10, 77),
                                      bridge=bridge)
        assert first is not None and first.source_xy in {(5, 51), (6, 51)}
    finally:
        bridge.close()


def test_runtime_templates_follow_script_moved_actors() -> None:
    bridge = _route2_bridge(10, 77)
    try:
        # setobjectxyperm edits the SaveBlock1 runtime template copy (FireRed +0x8E0).
        base = bridge.get_save_block1_dynamic_ptr() + 0x8E0
        record = struct.pack("<BBBxhhBBHHHIH2x", 2, 95, 0, 3, 70, 3, 0, 0, 0, 0, 0, 13)
        bridge.write_memory(base + 24, record)  # second slot, local id 2: the (15,69) tree
        templates = runtime_template_objects(bridge, "firered", ROUTE2)
        assert templates is not None
        moved = next(template for template in templates if template.local_id == 2)
        assert moved.xy == (3, 70) and moved.kind == "cut" and moved.temporary
        model = obstacle_model("firered", ROUTE2, bridge=bridge, current_map=True)
        assert (3, 70) in model.blocked and (15, 69) not in model.blocked
        # The temporary flag set by cutting hides the tree on the current map only.
        bridge.set_event_flag(13, True)
        assert (3, 70) not in obstacle_model("firered", ROUTE2, bridge=bridge, current_map=True).blocked
        assert (15, 69) in obstacle_model("firered", ROUTE2, bridge=FlagSnapshot(bridge)).blocked
    finally:
        bridge.close()


def test_spawned_tree_is_removable_only_with_usable_cut() -> None:
    bridge = _route2_bridge(15, 70)
    try:
        tree = ObjectEventStruct()
        tree.flags, tree.localId, tree.graphicsId = 1, 2, 95
        tree.mapGroup, tree.mapNum, tree.elevations = ROUTE2[0], ROUTE2[1], 3
        tree.currentCoords_x = tree.previousCoords_x = 15 + 7
        tree.currentCoords_y = tree.previousCoords_y = 69 + 7
        objects = bridge.get_object_events()
        objects[1] = tree
        _, collision, elevation = bridge.extract_live_map_matrices()

        def planner_for(hm_usable: set[str]) -> GbaConstrainedAStar:
            planner = GbaConstrainedAStar(collision, elevation, bridge.extract_behavior_map(), "firered")
            planner.register_object_events(objects, 0)
            model = obstacle_model("firered", ROUTE2, live_objects=objects, hm_usable=hm_usable)
            planner.add_static_obstacles(model.blocked, model.removable_costs)
            return planner

        # (16, 69) lies east of the tree; walls and a one-way ledge seal every other way.
        assert planner_for(set()).find_optimal_path((15, 70, 3), (16, 69)) is None
        path = planner_for({"cut"}).find_optimal_path((15, 70, 3), (16, 69))
        assert path == [(15, 69), (16, 69)]
    finally:
        bridge.close()
