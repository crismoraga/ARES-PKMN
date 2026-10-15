"""Directed Dijkstra topology of all source-defined map warps/connections."""
from __future__ import annotations

import heapq
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from core.bridge.ffi_mgba import GBABridgeProtocol
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


class HierarchicalWarpGraph:
    """Find inter-map routes; local collision paths select the usable exit."""

    def __init__(self, transitions: Iterable[WarpTransition] = ()) -> None:
        self.adjacency: dict[MapKey, list[WarpTransition]] = {}
        self.blocked_transitions: set[tuple[int, int, int, int, int, int]] = set()
        self.catalog: SourceMapCatalog | None = None
        for transition in transitions:
            self.add_transition(transition)

    def add_transition(self, transition: WarpTransition) -> None:
        if not math.isfinite(transition.cost) or transition.cost < 0:
            raise ValueError("Dijkstra requires finite nonnegative edge costs")
        self.adjacency.setdefault(transition.source, []).append(transition)

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
            graph._add_boat(catalog, "DewfordTown", "Route109", "Briney", choice=1, default=2,
                            requirements=("DEVON_GOODS", "STEVEN_LETTER"))
            graph._add_boat(catalog, "Route109", "DewfordTown", "Briney")
            graph._add_boat(catalog, "DewfordTown", "Route104", "Briney", default=2)
            graph._add_petalburg_script_doors(catalog)
        return graph

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
                  choice: int = 0, default: int = 1, requirements: tuple[str, ...] = ("DEVON_GOODS",)) -> None:
        target = catalog.find_target(source, script + "$" )
        arrival_targets = [candidate for candidate in catalog.targets(dest) if str(candidate["script"]).endswith("Briney")]
        arrival = (int(arrival_targets[0]["x"]), int(arrival_targets[0]["y"]) + 1) if arrival_targets else (13, 51)
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
        self.blocked_transitions.add(self._edge_id(edge))

    def find_route(self, start: MapKey, goal: MapKey,
                   completed: frozenset[str] = frozenset(),
                   current_xy: XY | None = None) -> list[WarpTransition] | None:
        if start == goal:
            return []
        # Entry coordinates are part of the state. Charging the approach only
        # in the first map makes a one-tile exit followed by 18 maps seem cheaper
        # than walking 39 tiles north, and causes oscillation when replanning.
        # Every map now pays a lower-bound local walking cost, consistently.
        entry = current_xy if current_xy is not None else (-1, -1)
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
            for edge in self.adjacency.get(key, []):
                if (self._edge_id(edge) in self.blocked_transitions or
                        any(item not in completed for item in edge.required_milestones)):
                    continue
                approach = abs(edge.source_xy[0] - position[0]) + abs(edge.source_xy[1] - position[1]) if position != (-1, -1) else 0
                new = distance + edge.cost + approach
                neighbor = edge.destination, edge.destination_xy
                if new < distances.get(neighbor, math.inf):
                    distances[neighbor], previous[neighbor] = new, (state, edge)
                    heapq.heappush(frontier, (new, neighbor))
        return None

    def next_transition(self, start: MapKey, goal: MapKey, current_xy: XY | None = None,
                        completed: frozenset[str] = frozenset()) -> WarpTransition | None:
        route = self.find_route(start, goal, completed, current_xy)
        return route[0] if route else None

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

