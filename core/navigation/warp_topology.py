"""Directed Dijkstra topology of all source-defined map warps/connections."""
from __future__ import annotations

import heapq
import json
import math
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from core.bridge.ffi_mgba import GBABridgeProtocol
from core.navigation.obstacles import FlagReader, FlagSnapshot, ObstacleModel, obstacle_model, template_objects
from core.navigation.script_gates import gate_state, gates_by_map, load_elevators, map_overlay
from core.navigation.source_data import MapKey, SourceMapCatalog, load_map_catalog
from core.navigation.constrained_astar import GbaConstrainedAStar, Matrix

XY = tuple[int, int]


@dataclass(frozen=True)
class WarpTransition:
    source: MapKey
    destination: MapKey
    source_xy: XY
    destination_xy: XY
    kind: Literal["warp", "connection", "script"] = "warp"
    direction: str = ""
    source_warp_id: int = -1
    destination_warp_id: int = -1
    cost: float = 1.0
    required_milestones: tuple[str, ...] = ()
    script: str = ""
    npc_local_id: int = 0
    menu_choice: int = 0
    menu_default: int = 0
    menu_profile: str = ""
    activation: Literal["step", "interact", "directional"] = "step"
    behavior: int = 0
    required_flags: tuple[int, ...] = ()
    # Elevator exits are dynamic warps: the floor panel (sign or attendant)
    # must first set the destination with menu row ``menu_choice``.
    panel_xy: XY | None = None
    panel_script: str = ""
    panel_local_id: int = 0


class HierarchicalWarpGraph:
    """Find inter-map routes; local collision paths select the usable exit."""

    def __init__(self, transitions: Iterable[WarpTransition] = ()) -> None:
        self.adjacency: dict[MapKey, list[WarpTransition]] = {}
        self.blocked_transitions: set[tuple[int, int, int, int, int, int]] = set()
        self.catalog: SourceMapCatalog | None = None
        # Pushback gates: source-derived ones (``static_gates``) and directed
        # steps observed to be undone by a script at runtime (learned).
        self.static_gates = True
        # What-if searches treat flagged story actors as passable to find which
        # event would clear them (``story_actors=False``); never for walking.
        self.story_actors = True
        self.learned_gate_edges: dict[MapKey, frozenset[tuple[XY, XY]]] = {}
        self._source_walk_cache: dict[tuple[object, ...], dict[XY, float]] = {}
        self._obstacle_cache: dict[tuple[object, ...], ObstacleModel] = {}
        self._live_walk_cache: dict[tuple[object, ...], dict[XY, float]] = {}
        self._coarse_reachability_cache: tuple[tuple[MapKey, frozenset[str]], frozenset[MapKey]] | None = None
        for transition in transitions:
            self.add_transition(transition)

    def add_transition(self, transition: WarpTransition) -> None:
        if not math.isfinite(transition.cost) or transition.cost < 0:
            raise ValueError("Dijkstra requires finite nonnegative edge costs")
        self.adjacency.setdefault(transition.source, []).append(transition)
        self._coarse_reachability_cache = None
        self._live_walk_cache.clear()
        self._source_walk_cache.clear()
        self._obstacle_cache.clear()

    def add_warp(self, source: MapKey, destination: MapKey, source_xy: XY,
                 destination_xy: XY, cost: float = 1.0) -> None:
        self.add_transition(WarpTransition(source, destination, source_xy, destination_xy, cost=cost))

    @classmethod
    def from_source(cls, game: str) -> HierarchicalWarpGraph:
        catalog = load_map_catalog(game)
        graph = cls()
        graph.catalog = catalog
        warp_behaviors = graph.warp_behavior_ids(game)
        for key, raw in catalog.maps.items():
            behaviors = catalog.behavior_grid(key)
            for index, warp in enumerate(raw["warp_events"]):
                # Original JSON includes unused alternate warp coordinates on
                # ordinary floor. Only tiles with warp attributes are executable.
                wx, wy = int(warp["x"]), int(warp["y"])
                if (not 0 <= wy < behaviors.shape[0] or not 0 <= wx < behaviors.shape[1] or
                        int(behaviors[wy, wx]) not in warp_behaviors):
                    continue
                dest_id = str(warp["dest_map"])
                if dest_id not in catalog.by_id:
                    continue  # MAP_UNDEFINED uses the runtime saved dynamic warp.
                dest = catalog.by_id[dest_id]
                dest_index = int(warp["dest_warp_id"])
                destination_warps = catalog.maps[dest]["warp_events"]
                if not 0 <= dest_index < len(destination_warps):
                    continue
                counterpart = destination_warps[dest_index]
                behavior = int(behaviors[wy, wx])
                activation: Literal["step", "interact", "directional"] = "step"
                direction, script = "", ""
                requirements: tuple[str, ...] = ()
                arrow_directions = {"MB_EAST_ARROW_WARP": "right", "MB_WEST_ARROW_WARP": "left",
                                    "MB_NORTH_ARROW_WARP": "up", "MB_SOUTH_ARROW_WARP": "down",
                                    "MB_STAIRS_OUTSIDE_ABANDONED_SHIP": "up", "MB_SHOAL_CAVE_ENTRANCE": "down",
                                    "MB_WATER_SOUTH_ARROW_WARP": "down"}
                if behavior == catalog.constants.get("MB_ANIMATED_DOOR", catalog.constants.get("MB_WARP_DOOR", -1)):
                    activation, direction = "directional", "up"
                for name, arrow_direction in arrow_directions.items():
                    if behavior == catalog.constants.get(name, -1):
                        activation, direction = "directional", arrow_direction
                if behavior == catalog.constants.get("MB_PETALBURG_GYM_DOOR", -1):
                    activation, direction = "interact", "up"
                    scripts = [str(target["script"]) for target in catalog.targets(str(raw["name"]))
                               if (int(target["x"]), int(target["y"])) == (wx, wy)]
                    script = scripts[0] if scripts else ""
                if behavior == catalog.constants.get("MB_CLOSED_SOOTOPOLIS_DOOR", -1):
                    requirements, direction, activation = ("SOOTOPOLIS_TRUCE",), "up", "directional"
                if behavior == catalog.constants.get("MB_SKY_PILLAR_CLOSED_DOOR", -1):
                    requirements, direction, activation = ("WALLACE_ORIGIN",), "up", "directional"
                graph.add_transition(WarpTransition(key, dest, (int(warp["x"]), int(warp["y"])),
                                                    (int(counterpart["x"]), int(counterpart["y"])),
                                                    source_warp_id=index, destination_warp_id=dest_index,
                                                    required_milestones=requirements, direction=direction, script=script,
                                                    activation=activation, behavior=behavior))
            for connection in raw.get("connections") or []:
                connection_dest = catalog.by_id.get(str(connection["map"]))
                if connection_dest is None or connection["direction"] in ("dive", "emerge"):
                    continue
                other = catalog.maps[connection_dest]
                direction, offset = str(connection["direction"]), int(connection["offset"])
                width, height = int(raw["width"]), int(raw["height"])
                other_width, other_height = int(other["width"]), int(other["height"])
                # Preserve all boundary candidates: the midpoint may be a wall.
                size = width if direction in ("up", "down") else height
                limit = other_width if direction in ("up", "down") else other_height
                grid, other_grid = catalog.grid(key), catalog.grid(connection_dest)
                for value in range(size):
                    mapped = value - offset
                    if not 0 <= mapped < limit:
                        continue
                    if direction == "up":
                        origin, target = (value, 0), (mapped, other_height - 1)
                    elif direction == "down":
                        origin, target = (value, height - 1), (mapped, 0)
                    elif direction == "left":
                        origin, target = (0, value), (other_width - 1, mapped)
                    else:
                        origin, target = (width - 1, value), (0, mapped)
                    if int(grid[origin[1], origin[0]]) >> 10 & 3 or int(other_grid[target[1], target[0]]) >> 10 & 3:
                        continue
                    source_behavior = int(behaviors[origin[1], origin[0]])
                    target_behavior = int(catalog.behavior_grid(connection_dest)[target[1], target[0]])
                    requirements = ()
                    if source_behavior in (0x10, 0x11, 0x12, 0x13, 0x14, 0x15) or target_behavior in (0x10, 0x11, 0x12, 0x13, 0x14, 0x15):
                        requirements = ("HM_SURF", "GYM_5_NORMAN" if game == "emerald" else "GYM_5_KOGA")
                    graph.add_transition(WarpTransition(key, connection_dest, origin, target, kind="connection", direction=direction,
                                                        required_milestones=requirements, behavior=source_behavior))
        # These journeys are event scripts, not ordinary JSON warps. Metadata
        # names an interactable source event rather than inventing a direct walk.
        if game == "emerald":
            truck = catalog.by_name["InsideOfTruck"]
            littleroot = catalog.by_name["LittlerootTown"]
            for warp in catalog.maps[truck]["warp_events"]:
                graph.add_transition(WarpTransition(truck, littleroot, (int(warp["x"]), int(warp["y"])),
                                                    (5, 8), kind="warp", direction="right"))
            graph._add_boat(catalog, "Route104_MrBrineysHouse", "DewfordTown", "Briney")
            # DewfordTown_EventScript_SailToSlateport leaves the player on the
            # boat at (21,26), then DewfordTown_Movement_ExitBoatSlateport walks
            # up three tiles to the beach. Briney's tile + 1 is open ocean.
            graph._add_boat(catalog, "DewfordTown", "Route109", "Briney", choice=1, default=2,
                            requirements=("DEVON_GOODS", "STEVEN_LETTER"), arrival_xy=(21, 23))
            graph._add_boat(catalog, "Route109", "DewfordTown", "Briney")
            # DewfordTown_EventScript_SailToPetalburg ends with the explicit
            # warp MAP_ROUTE104_MR_BRINEYS_HOUSE, 5, 4. The boat animation first
            # crosses Route104, but the field-ready landing is inside the house.
            graph._add_boat(catalog, "DewfordTown", "Route104_MrBrineysHouse", "Briney",
                            default=2, arrival_xy=(5, 4))
            graph._add_petalburg_script_doors(catalog)
        graph._add_elevators(catalog, game, warp_behaviors)
        return graph

    def _add_elevators(self, catalog: SourceMapCatalog, game: str, warp_behaviors: frozenset[int]) -> None:
        """Dynamic-warp exits, one edge per floor its panel's menu can select."""
        for map_name, panels in load_elevators(game).items():
            key = catalog.by_name.get(map_name)
            if key is None:
                continue
            raw = catalog.maps[key]
            behaviors = catalog.behavior_grid(key)
            exits = [(index, (int(warp["x"]), int(warp["y"]))) for index, warp in enumerate(raw["warp_events"])
                     if str(warp["dest_map"]) == "MAP_DYNAMIC" and 0 <= int(warp["y"]) < behaviors.shape[0] and
                     0 <= int(warp["x"]) < behaviors.shape[1] and
                     int(behaviors[int(warp["y"]), int(warp["x"])]) in warp_behaviors]
            for panel in panels[:1]:  # duplicate panels run the same script
                flags = tuple(catalog.constants[name] for name in panel.get("required_flags", [])
                              if name in catalog.constants)
                for floor in panel["floors"]:
                    destination = catalog.by_id.get(str(floor["map"]))
                    if destination is None or destination == key:
                        continue
                    for index, xy in exits:
                        behavior = int(behaviors[xy[1], xy[0]])
                        activation, direction = self._warp_activation(catalog, behavior)
                        self.add_transition(WarpTransition(
                            key, destination, xy, (int(floor["xy"][0]), int(floor["xy"][1])), cost=4.0,
                            source_warp_id=index, direction=direction, activation=activation, behavior=behavior,
                            menu_choice=int(floor["choice"]), required_flags=flags,
                            panel_xy=(int(panel["xy"][0]), int(panel["xy"][1])), panel_script=str(panel["script"]),
                            panel_local_id=int(panel.get("npc_local_id", 0))))

    @staticmethod
    def _warp_activation(catalog: SourceMapCatalog,
                         behavior: int) -> tuple[Literal["step", "interact", "directional"], str]:
        arrows = {"MB_EAST_ARROW_WARP": "right", "MB_WEST_ARROW_WARP": "left",
                  "MB_NORTH_ARROW_WARP": "up", "MB_SOUTH_ARROW_WARP": "down"}
        for name, arrow_direction in arrows.items():
            if behavior == catalog.constants.get(name, -1):
                return "directional", arrow_direction
        return "step", ""

    def _add_petalburg_script_doors(self, catalog: SourceMapCatalog) -> None:
        """Petalburg doors are BG scripts on normal tiles, not JSON warp tiles."""
        map_name = "PetalburgCity_Gym"
        key = catalog.by_name[map_name]
        guarded_by = {"ConfusionRoomDoor": "TRAINER_RANDALL", "LeftDefenseRoomDoor": "TRAINER_RANDALL",
                      "RightDefenseRoomDoor": "TRAINER_MARY", "RecoveryRoomDoor": "TRAINER_MARY",
                      "LeftStrengthRoomDoor": "TRAINER_PARKER", "RightStrengthRoomDoor": "TRAINER_ALEXIA",
                      "LeftOHKORoomDoor": "TRAINER_ALEXIA", "RightOHKORoomDoor": "TRAINER_GEORGE",
                      "LeftGymLeadersRoomDoor": "TRAINER_JODY", "RightGymLeadersRoomDoor": "TRAINER_BERKE"}
        for target in catalog.targets(map_name):
            script = str(target["script"])
            if not script.endswith("RoomDoor"):
                continue
            assignments = dict(catalog.effects(script).set_vars)
            x_text, y_text = assignments.get("VAR_0x8008"), assignments.get("VAR_0x8009")
            if x_text is None or y_text is None or not x_text.isdecimal() or not y_text.isdecimal():
                raise ValueError(f"Scripted Petalburg door lacks literal target: {script}")
            trainer = guarded_by.get(script.rsplit("_", 1)[-1])
            flags = (catalog.constants["TRAINER_FLAGS_START"] + catalog.constants[trainer],) if trainer else ()
            self.add_transition(WarpTransition(key, key, (int(target["x"]), int(target["y"])),
                                                (int(x_text), int(y_text)), kind="script", direction="up",
                                                required_milestones=("GYM_4_FLANNERY",), script=script,
                                                activation="interact", menu_choice=0, required_flags=flags))

    @staticmethod
    def warp_behavior_ids(game: str) -> frozenset[int]:
        """Native event warp predicates, including portals outside 0x60..0x71."""
        if game == "firered":
            return frozenset(range(0x60, 0x72))
        if game != "emerald":
            raise ValueError("Unsupported warp behavior game")
        constants = load_map_catalog(game).constants
        names = ("MB_NON_ANIMATED_DOOR", "MB_ANIMATED_DOOR", "MB_LADDER", "MB_UP_ESCALATOR", "MB_DOWN_ESCALATOR",
                 "MB_LAVARIDGE_GYM_B1F_WARP", "MB_LAVARIDGE_GYM_1F_WARP", "MB_AQUA_HIDEOUT_WARP", "MB_MT_PYRE_HOLE",
                 "MB_MOSSDEEP_GYM_WARP", "MB_BRIDGE_OVER_OCEAN", "MB_EAST_ARROW_WARP", "MB_WEST_ARROW_WARP",
                 "MB_NORTH_ARROW_WARP", "MB_SOUTH_ARROW_WARP", "MB_STAIRS_OUTSIDE_ABANDONED_SHIP", "MB_SHOAL_CAVE_ENTRANCE",
                 "MB_WATER_DOOR", "MB_WATER_SOUTH_ARROW_WARP", "MB_DEEP_SOUTH_WARP", "MB_PETALBURG_GYM_DOOR",
                 "MB_CLOSED_SOOTOPOLIS_DOOR", "MB_SKY_PILLAR_CLOSED_DOOR", "MB_CRACKED_FLOOR_HOLE")
        return frozenset(constants[name] for name in names if name in constants)

    def _add_boat(self, catalog: SourceMapCatalog, source: str, dest: str, script: str,
                  choice: int = 0, default: int = 1, requirements: tuple[str, ...] = ("DEVON_GOODS",),
                  arrival_xy: XY | None = None) -> None:
        target = catalog.find_target(source, script + "$" )
        arrival_targets = [candidate for candidate in catalog.targets(dest) if str(candidate["script"]).endswith("Briney")]
        arrival = arrival_xy if arrival_xy is not None else (
            (int(arrival_targets[0]["x"]), int(arrival_targets[0]["y"]) + 1) if arrival_targets else (13, 51))
        self.add_transition(WarpTransition(catalog.by_name[source], catalog.by_name[dest],
                                           (int(target["x"]), int(target["y"])), arrival, kind="script",
                                           cost=20, required_milestones=requirements, script=str(target["script"]),
                                           npc_local_id=int(target["npc_local_id"]), menu_choice=choice, menu_default=default,
                                           menu_profile="briney_" + source + "_to_" + dest, activation="interact"))

    def transitions_from(self, map_key: MapKey,
                          completed: frozenset[str] = frozenset(),
                          bridge: GBABridgeProtocol | None = None) -> tuple[WarpTransition, ...]:
        """Expose alternatives when a chosen exit is blocked by live actors."""
        return tuple(edge for edge in self.adjacency.get(map_key, []) if self._edge_id(edge) not in self.blocked_transitions
                     and all(name in completed for name in edge.required_milestones)
                     and (bridge is None or all(bridge.get_event_flag(flag) for flag in edge.required_flags)))

    def find_intra_map_route(self, map_key: MapKey, start: tuple[int, int, int], goal: XY,
                            collision_map: Matrix | None = None, elevation_map: Matrix | None = None,
                            behavior_map: Matrix | None = None,
                            completed: frozenset[str] = frozenset(),
                            bridge: GBABridgeProtocol | None = None) -> list[WarpTransition] | None:
        """Dijkstra across local walkable components and same-map teleporters.

        Saffron, Aqua and Petalburg puzzle rooms can be disconnected while sharing
        one map ID. A plain map-level Dijkstra cannot represent those components.
        A caller can supply live matrices so opened doors override initial data.
        """
        if self.catalog is None and any(grid is None for grid in (collision_map, elevation_map, behavior_map)):
            raise ValueError("Local routing requires matrices or a source catalog")
        if self.catalog is not None:
            packed = self.catalog.grid(map_key)
            if collision_map is None:
                collision_map = cast(Matrix, (packed >> 10) & 3)
            if elevation_map is None:
                elevation_map = cast(Matrix, packed >> 12)
            if behavior_map is None:
                behavior_map = self.catalog.behavior_grid(map_key)
        assert collision_map is not None and elevation_map is not None and behavior_map is not None
        planner = GbaConstrainedAStar(collision_map, elevation_map, behavior_map,
                                      game=self.catalog.game if self.catalog else "firered", allow_surf=True)
        portals = [edge for edge in self.transitions_from(map_key, completed, bridge) if edge.destination == map_key]
        frontier: list[tuple[float, tuple[int, int, int], tuple[int, ...]]] = [(0, start, ())]
        distances = {start: 0.0}
        goal_marker = goal[0], goal[1], -1
        while frontier:
            cost, position, route = heapq.heappop(frontier)
            if cost != distances[position]:
                continue
            if position == goal_marker:
                return [portals[index] for index in route]
            final_walk = planner.find_optimal_path(position, goal)
            if final_walk is not None:
                finish_cost = cost + len(final_walk)
                if finish_cost < distances.get(goal_marker, math.inf):
                    distances[goal_marker] = finish_cost
                    heapq.heappush(frontier, (finish_cost, goal_marker, route))
            for index, edge in enumerate(portals):
                approach = edge.source_xy
                if edge.activation in ("interact", "directional") and edge.direction == "up":
                    approach = approach[0], approach[1] + 1
                path = planner.find_optimal_path(position, approach)
                if path is None:
                    continue
                x, y = edge.destination_xy
                if not 0 <= x < elevation_map.shape[1] or not 0 <= y < elevation_map.shape[0]:
                    continue
                landing = x, y, int(elevation_map[y, x])
                if edge.activation == "interact" and y + 1 < elevation_map.shape[0]:
                    landing = x, y + 1, int(elevation_map[y + 1, x])
                new_cost = cost + len(path) + edge.cost
                if new_cost < distances.get(landing, math.inf):
                    distances[landing] = new_cost
                    heapq.heappush(frontier, (new_cost, landing, route + (index,)))
        return None

    @staticmethod
    def _edge_id(edge: WarpTransition) -> tuple[int, int, int, int, int, int]:
        return (*edge.source, *edge.destination, *edge.source_xy)

    def block_transition(self, edge: WarpTransition) -> None:
        self.block_transition_id(self._edge_id(edge))

    def block_transition_id(self, edge_id: tuple[int, int, int, int, int, int]) -> None:
        self.blocked_transitions.add(edge_id)
        self._coarse_reachability_cache = None

    def unblock_transition(self, edge_id: tuple[int, int, int, int, int, int]) -> None:
        """Lift a temporary tabu; source-defined edges are never deleted."""
        self.blocked_transitions.discard(edge_id)
        self._coarse_reachability_cache = None

    def _coarsely_reachable(self, start: MapKey, completed: frozenset[str]) -> frozenset[MapKey]:
        """Cheap topology superset before any expensive tile-component search."""
        signature = start, completed
        if self._coarse_reachability_cache is not None and self._coarse_reachability_cache[0] == signature:
            return self._coarse_reachability_cache[1]
        reached = {start}
        queue = deque([start])
        while queue:
            for edge in self.transitions_from(queue.popleft(), completed):
                if edge.destination not in reached:
                    reached.add(edge.destination)
                    queue.append(edge.destination)
        result = frozenset(reached)
        self._coarse_reachability_cache = signature, result
        return result

    @staticmethod
    def _approach_tiles(edge: WarpTransition) -> tuple[XY, ...]:
        x, y = edge.source_xy
        if edge.activation in ("interact", "directional") and edge.direction == "up":
            return ((x, y + 1),)
        if edge.kind == "script" and edge.npc_local_id:
            return ((x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y))
        return (edge.source_xy,)

    def set_learned_gate_edges(self, edges: dict[MapKey, frozenset[tuple[XY, XY]]]) -> None:
        """Replace runtime-learned pushback steps; cached walks depend on them."""
        normalized = {key: frozenset(value) for key, value in edges.items() if value}
        if normalized == self.learned_gate_edges:
            return
        self.learned_gate_edges = normalized
        self._source_walk_cache.clear()
        self._obstacle_cache.clear()
        self._live_walk_cache.clear()

    def set_static_gates(self, enabled: bool) -> None:
        """Cache keys include the gate mode, so toggling never invalidates them."""
        self.static_gates = enabled

    def set_story_actors(self, enabled: bool) -> None:
        self.story_actors = enabled

    def _source_obstacles(self, key: MapKey, flags: FlagReader | None,
                          hm_usable: frozenset[str]) -> ObstacleModel:
        assert self.catalog is not None
        game = self.catalog.game
        dependency: tuple[object, ...] = ()
        if isinstance(flags, FlagSnapshot):
            # Only the flags of this map's templates and its gates' verdicts
            # matter. Keying on whole RAM arrays would discard every cached
            # walk whenever an unrelated counter variable ticked.
            dependency = (tuple(flags.get_event_flag(template.flag) for template in template_objects(game, key)
                                if template.flag is not None and not template.temporary),
                          tuple(gate_state(gate, flags) for gate in gates_by_map(game).get(key, ()))
                          if self.static_gates else ())
        learned = self.learned_gate_edges.get(key, frozenset())
        cache_key: tuple[object, ...] = (key, hm_usable, dependency, learned, self.static_gates, self.story_actors)
        cached = self._obstacle_cache.get(cache_key)
        if cached is None:
            cached = obstacle_model(self.catalog.game, key, bridge=flags, hm_usable=hm_usable,
                                    learned_edges=learned, static_gates=self.static_gates,
                                    story_actors=self.story_actors)
            if len(self._obstacle_cache) >= 2048:
                self._obstacle_cache.clear()
            self._obstacle_cache[cache_key] = cached
        return cached

    def _walking_costs(self, key: MapKey, position: XY, allow_surf: bool,
                       bridge: GBABridgeProtocol | None = None, *,
                       flags: FlagReader | None = None,
                       hm_usable: frozenset[str] = frozenset()) -> dict[XY, float] | None:
        """Reachable tile costs; ``bridge`` selects live RAM matrices and actors.

        Source planning still includes object templates (trees, rocks, story
        NPCs), with persistent flags read through ``flags`` when available, so
        a later map can never look more open than it does on arrival.
        """
        if self.catalog is None or position == (-1, -1):
            return None
        game = self.catalog.game
        if bridge is not None:
            coords = bridge.get_player_coords_and_map()
            if (coords["map_group"], coords["map_num"]) != key:
                # Live RAM describes only the map the avatar stands on.
                flags = flags if flags is not None else FlagSnapshot(bridge)
                bridge = None
        if bridge is None:
            packed = self.catalog.grid(key)
            collision, elevation = cast(Matrix, (packed >> 10) & 3), cast(Matrix, packed >> 12)
            # On-load scripts wall or open tiles depending on story flags; the
            # live map will show them, so the source plan must agree.
            overlay = map_overlay(game, str(self.catalog.maps[key]["name"]), flags) if flags is not None else ()
            if overlay:
                collision = collision.copy()
                for x, y, impassable in overlay:
                    if 0 <= y < collision.shape[0] and 0 <= x < collision.shape[1]:
                        collision[y, x] = impassable
            behavior = self.catalog.behavior_grid(key)
            if not 0 <= position[1] < elevation.shape[0] or not 0 <= position[0] < elevation.shape[1]:
                return {}
            actor_z = int(elevation[position[1], position[0]])
            obstacles = self._source_obstacles(key, flags, hm_usable)
            cache_key: tuple[object, ...] = (key, position, allow_surf, obstacles, overlay)
            cached = self._source_walk_cache.get(cache_key)
            if cached is not None:
                return cached
            objects = []
            player_id = 0
        else:
            _, collision, elevation = bridge.extract_live_map_matrices()
            behavior = bridge.extract_behavior_map()
            coords = bridge.get_player_coords_and_map()
            if not 0 <= position[1] < elevation.shape[0] or not 0 <= position[0] < elevation.shape[1]:
                return {}
            # A re-entry point is not the avatar's tile; use that tile's elevation.
            actor_z = coords["z"] if (coords["x"], coords["y"]) == position else int(elevation[position[1], position[0]])
            if actor_z == 15:
                actor_z = coords["z"]
            objects = bridge.get_object_events()
            player_id = bridge.read_u8(bridge.memory_map.avatar + 5)
            obstacles = obstacle_model(game, key, bridge=bridge, live_objects=objects,
                                       player_object_id=player_id, hm_usable=hm_usable, current_map=True,
                                       learned_edges=self.learned_gate_edges.get(key, frozenset()),
                                       static_gates=self.static_gates, story_actors=self.story_actors)
            signature: tuple[object, ...] = (key, position, actor_z, allow_surf,
                hash(collision.tobytes()), hash(elevation.tobytes()), hash(behavior.tobytes()),
                tuple(bytes(actor) for actor in objects), player_id, obstacles)
            cached = self._live_walk_cache.get(signature)
            if cached is not None:
                return cached
        planner = GbaConstrainedAStar(collision, elevation, behavior, game, allow_surf)
        if bridge is not None and self.story_actors:
            # Occupancy matters; trainer sight is a cost policy, not a proof that
            # a map connection is physically inaccessible. A what-if search for
            # removable story actors ignores live occupancy as well.
            planner.register_object_events(objects, player_id, penalty=0)
        planner.add_static_obstacles(obstacles.blocked, obstacles.removable_costs)
        planner.blocked_edges.update(obstacles.gate_edges)
        planner.terminal_tiles = {edge.source_xy for edge in self.adjacency.get(key, []) if edge.kind == "warp"}
        result = planner.reachable_costs((*position, actor_z))
        if bridge is None:
            if len(self._source_walk_cache) >= 512:
                self._source_walk_cache.pop(next(iter(self._source_walk_cache)))
            self._source_walk_cache[cache_key] = result
        else:
            # Several live entries: the avatar tile plus re-entry tiles of this map.
            if len(self._live_walk_cache) >= 16:
                self._live_walk_cache.pop(next(iter(self._live_walk_cache)))
            self._live_walk_cache[signature] = result
        return result

    def live_walking_costs(self, bridge: GBABridgeProtocol, allow_surf: bool = False,
                           hm_usable: frozenset[str] = frozenset()) -> dict[XY, float] | None:
        """Costs from the avatar's live tile, sharing the route search's cache."""
        coords = bridge.get_player_coords_and_map()
        return self._walking_costs((coords["map_group"], coords["map_num"]), (coords["x"], coords["y"]),
                                   allow_surf, bridge, hm_usable=hm_usable)

    def edge_reachable(self, edge: WarpTransition, bridge: GBABridgeProtocol, allow_surf: bool = False,
                       hm_usable: frozenset[str] = frozenset()) -> bool:
        """Whether a committed exit is still walkable from the live avatar tile."""
        walking = self.live_walking_costs(bridge, allow_surf, hm_usable)
        if walking is None:
            return True
        return any(math.isfinite(walking.get(tile, math.inf)) for tile in self._approach_tiles(edge))

    def find_route(self, start: MapKey, goal: MapKey,
                   completed: frozenset[str] = frozenset(),
                   current_xy: XY | None = None,
                   bridge: GBABridgeProtocol | None = None,
                   validate_local: bool = True,
                   hm_usable: frozenset[str] = frozenset()) -> list[WarpTransition] | None:
        if start == goal:
            return []
        if goal not in self._coarsely_reachable(start, completed):
            return None
        # Entry coordinates are part of the state. Charging the approach only
        # in the first map makes a one-tile exit followed by 18 maps seem cheaper
        # than walking 39 tiles north, and causes oscillation when replanning.
        # Every map now pays a lower-bound local walking cost, consistently.
        entry = current_xy if current_xy is not None else (-1, -1)
        surf_badge = "GYM_5_NORMAN" if self.catalog is not None and self.catalog.game == "emerald" else "GYM_5_KOGA"
        allow_surf = ("HM_SURF" in completed and surf_badge in completed) or "surf" in hm_usable
        flags: FlagReader | None = None
        if bridge is not None and self.catalog is not None and validate_local:
            try:
                flags = FlagSnapshot(bridge)
            except (ValueError, RuntimeError):
                flags = None
        initial = start, entry
        frontier: list[tuple[float, tuple[MapKey, XY]]] = [(0, initial)]
        distances = {initial: 0.0}
        previous: dict[tuple[MapKey, XY], tuple[tuple[MapKey, XY], WarpTransition]] = {}
        while frontier:
            distance, state = heapq.heappop(frontier)
            if distance != distances[state]:
                continue
            key, position = state
            if key == goal:
                route: list[WarpTransition] = []
                while state != initial:
                    parent, edge = previous[state]
                    route.append(edge)
                    state = parent
                return route[::-1]
            # Route104 and many caves have disconnected regions with one map
            # ID. Manhattan proximity to a warp does not establish reachability.
            # Re-entering the current map also uses live RAM: leaving and coming
            # back must never "forget" a live obstacle (the Route 2 Cut tree loop).
            walking = self._walking_costs(key, position, allow_surf,
                bridge if (state == initial or key == start) else None,
                flags=flags, hm_usable=hm_usable) if validate_local else None
            for edge in self.transitions_from(key, completed, bridge):
                if walking is None:
                    approach = abs(edge.source_xy[0] - position[0]) + abs(edge.source_xy[1] - position[1]) if position != (-1, -1) else 0.0
                else:
                    approach = min((walking.get(tile, math.inf) for tile in self._approach_tiles(edge)), default=math.inf)
                    if not math.isfinite(approach):
                        continue
                new = distance + edge.cost + approach
                neighbor = edge.destination, edge.destination_xy
                if new < distances.get(neighbor, math.inf):
                    distances[neighbor], previous[neighbor] = new, (state, edge)
                    heapq.heappush(frontier, (new, neighbor))
        return None

    def find_route_to_tile(self, start: MapKey, goal: MapKey, goal_tiles: Iterable[XY],
                           completed: frozenset[str] = frozenset(), current_xy: XY | None = None,
                           bridge: GBABridgeProtocol | None = None,
                           hm_usable: frozenset[str] = frozenset()) -> list[WarpTransition] | None:
        """Route whose final entry into ``goal`` can actually walk to a goal tile.

        Caves such as Mt. Moon, Rock Tunnel or Granite Cave split one map ID into
        sections joined only through other floors. Arriving in the right map is
        not enough: the search continues through other maps until some entry
        point reaches one of ``goal_tiles``. An empty route means the target is
        already reachable from the current tile.
        """
        targets = set(goal_tiles)
        if self.catalog is None or not targets or current_xy is None:
            return None
        if goal != start and goal not in self._coarsely_reachable(start, completed):
            return None
        surf_badge = "GYM_5_NORMAN" if self.catalog.game == "emerald" else "GYM_5_KOGA"
        allow_surf = ("HM_SURF" in completed and surf_badge in completed) or "surf" in hm_usable
        flags: FlagReader | None = None
        if bridge is not None:
            try:
                flags = FlagSnapshot(bridge)
            except (ValueError, RuntimeError):
                flags = None
        initial = start, current_xy
        frontier: list[tuple[float, tuple[MapKey, XY]]] = [(0, initial)]
        distances = {initial: 0.0}
        previous: dict[tuple[MapKey, XY], tuple[tuple[MapKey, XY], WarpTransition]] = {}
        expanded = 0
        while frontier and expanded < 4096:
            distance, state = heapq.heappop(frontier)
            if distance != distances[state]:
                continue
            expanded += 1
            key, position = state
            walking = self._walking_costs(key, position, allow_surf,
                                          bridge if (state == initial or key == start) else None,
                                          flags=flags, hm_usable=hm_usable)
            if walking is None:
                return None
            if key == goal and any(tile in walking for tile in targets):
                route: list[WarpTransition] = []
                while state != initial:
                    parent, edge = previous[state]
                    route.append(edge)
                    state = parent
                return route[::-1]
            for edge in self.transitions_from(key, completed, bridge):
                approach = min((walking.get(tile, math.inf) for tile in self._approach_tiles(edge)), default=math.inf)
                if not math.isfinite(approach):
                    continue
                new = distance + edge.cost + approach
                neighbor = edge.destination, edge.destination_xy
                if new < distances.get(neighbor, math.inf):
                    distances[neighbor], previous[neighbor] = new, (state, edge)
                    heapq.heappush(frontier, (new, neighbor))
        return None

    def next_transition(self, start: MapKey, goal: MapKey, current_xy: XY | None = None,
                        completed: frozenset[str] = frozenset(),
                        bridge: GBABridgeProtocol | None = None,
                        hm_usable: frozenset[str] = frozenset()) -> WarpTransition | None:
        route = self.find_route(start, goal, completed, current_xy, bridge, hm_usable=hm_usable)
        return route[0] if route else None

    @staticmethod
    def estimate_route_cost(route: Iterable[WarpTransition], current_xy: XY) -> float:
        """Walking lower bound plus transition costs, in tile-equivalent units.

        Consumers comparing hospitals or training areas must account for the
        walk to every exit instead of treating a city-wide walk as one frame.
        """
        total = 0.0
        position = current_xy
        for edge in route:
            total += edge.cost + min(abs(x - position[0]) + abs(y - position[1])
                                     for x, y in HierarchicalWarpGraph._approach_tiles(edge))
            position = edge.destination_xy
        return total

    def shortest_path(self, start: MapKey, goal: MapKey) -> list[MapKey] | None:
        route = self.find_route(start, goal)
        return [start] + [edge.destination for edge in route] if route is not None else None

    def save_obstacles(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(sorted(self.blocked_transitions)), encoding="utf-8")

    def load_obstacles(self, path: str | Path) -> None:
        raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, list) or any(not isinstance(row, list) or len(row) != 6 or
                                           not all(isinstance(v, int) for v in row) for row in raw):
            raise ValueError("Invalid persisted transition obstacles")
        self.blocked_transitions = {cast(tuple[int, int, int, int, int, int], tuple(row)) for row in raw}
        self._coarse_reachability_cache = None

