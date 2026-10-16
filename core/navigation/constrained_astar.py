"""Generation-three tile physics and hazard-aware A* over actor elevation.

Behavior codes come from each game's metatile *attributes*, not its tile IDs.
Elevation 15 preserves an actor's current elevation; elevation 0 changes it to
zero, permitting a subsequent ramp transition. This matches pret's
IsElevationMismatchAt/ObjectEventUpdateElevation instead of a ±1 height rule.
"""
from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from core.bridge.gba_structs import ObjectEventStruct

State = tuple[int, int, int]
XY = tuple[int, int]
Edge = tuple[XY, XY]
Matrix = NDArray[np.uint16]
_DIRECTIONS = ((1, 0, 0x38, 4), (-1, 0, 0x39, 3),
               (0, -1, 0x3A, 2), (0, 1, 0x3B, 1))
_FACING = {1: (0, 1), 2: (0, -1), 3: (-1, 0), 4: (1, 0)}
_OPPOSITE_FACING = {1: 2, 2: 1, 3: 4, 4: 3}
# FireRed spin tiles (Rocket Hideout, Viridian Gym): stepping on one slides the
# player in its direction, turning on further spin tiles, until a stop tile or
# an obstruction (pret field_player_avatar.c TryUpdatePlayerSpinDirection).
_FIRERED_SPIN: dict[int, tuple[int, int]] = {0x54: (1, 0), 0x55: (-1, 0), 0x56: (0, -1), 0x57: (0, 1)}
_FIRERED_STOP_SPIN = 0x58
_DIRECTIONAL_BLOCKS: dict[int, frozenset[int]] = {
    0x30: frozenset((4,)), 0x31: frozenset((3,)),
    0x32: frozenset((2,)), 0x33: frozenset((1,)),
    0x34: frozenset((4, 2)), 0x35: frozenset((3, 2)),
    0x36: frozenset((4, 1)), 0x37: frozenset((3, 1)),
}


class GbaConstrainedAStar:
    """Shortest legal path excluding the start tile; returns None if unreachable."""

    MB_JUMP_EAST, MB_JUMP_WEST, MB_JUMP_NORTH, MB_JUMP_SOUTH = 0x38, 0x39, 0x3A, 0x3B

    def __init__(self, collision_map: Matrix, elevation_map: Matrix,
                 behavior_map: Matrix | None = None, game: str = "firered",
                 allow_surf: bool = False) -> None:
        if game not in ("firered", "emerald"):
            raise ValueError("Unknown game")
        if collision_map.ndim != 2 or elevation_map.shape != collision_map.shape:
            raise ValueError("Collision and elevation must have matching 2-D shapes")
        if not collision_map.size or np.any(elevation_map > 15) or np.any(collision_map > 3):
            raise ValueError("Invalid map dimensions or tile attributes")
        if behavior_map is not None and behavior_map.shape != collision_map.shape:
            raise ValueError("Behavior map shape mismatch")
        self.h, self.w = collision_map.shape
        self.collision = collision_map.copy()
        self.elevation = elevation_map.copy()
        self.behavior = behavior_map.copy() if behavior_map is not None else np.zeros_like(collision_map)
        self.game, self.allow_surf = game, allow_surf
        self.trainer_hazard_cost: dict[XY, float] = {}
        self.occupied: dict[XY, set[int]] = {}
        # Off-camera source objects block at every elevation; HM obstacles
        # (Cut trees, Rock Smash rocks) can instead be cleared at an extra cost.
        self.static_blocked: set[XY] = set()
        self.removable: dict[XY, float] = {}
        self.blocked_edges: set[Edge] = set()
        # Entering one of these tiles changes maps. A route may end there,
        # but cannot use a teleporter as an ordinary intermediate floor tile.
        self.terminal_tiles: set[XY] = set()
        # A slide step (origin, landing) is entered by walking onto this tile.
        self.slide_entries: dict[tuple[XY, XY], XY] = {}

    def _inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h

    def register_trainer_cone(self, npc_x: int, npc_y: int, facing_dir: int,
                              sight_range: int, penalty: float = 250.0) -> None:
        """Native normal trainers see a ray, stopped by solid tiles and actors."""
        if penalty < 0 or not math.isfinite(penalty) or sight_range < 0:
            raise ValueError("Invalid hazard cost or range")
        if facing_dir not in _FACING:
            return
        dx, dy = _FACING[facing_dir]
        for step in range(1, sight_range + 1):
            x, y = npc_x + step * dx, npc_y + step * dy
            if not self._inside(x, y) or int(self.collision[y, x]) or (x, y) in self.occupied:
                break
            self.trainer_hazard_cost[x, y] = max(self.trainer_hazard_cost.get((x, y), 0), penalty)

    def register_object_events(self, events: Iterable[ObjectEventStruct],
                               player_object_id: int = 0, penalty: float = 250.0,
                               defeated_local_ids: frozenset[int] = frozenset()) -> None:
        """Refresh occupancy including a moving actor's reserved previous tile.

        Object coordinates include the seven-tile camera border. Runtime actors
        take precedence over static templates, including cut trees and boulders.
        """
        self.occupied.clear()
        self.trainer_hazard_cost.clear()
        actors = list(events)
        for index, actor in enumerate(actors):
            if index == player_object_id or not int(actor.flags) & 1:
                continue
            z = int(actor.elevations) & 15
            for x, y in ((int(actor.currentCoords_x) - 7, int(actor.currentCoords_y) - 7),
                         (int(actor.previousCoords_x) - 7, int(actor.previousCoords_y) - 7)):
                if self._inside(x, y):
                    self.occupied.setdefault((x, y), set()).add(z)
        for index, actor in enumerate(actors):
            if (index == player_object_id or not int(actor.flags) & 1 or
                    int(actor.trainerType) == 0 or int(actor.localId) in defeated_local_ids):
                continue
            directions = (1, 2, 3, 4) if int(actor.trainerType) == 3 else (int(actor.facingDirection) & 15,)
            for direction in directions:
                self.register_trainer_cone(int(actor.currentCoords_x) - 7,
                                          int(actor.currentCoords_y) - 7, direction,
                                          int(actor.trainerRange), penalty)

    def block_edge(self, origin: XY, destination: XY) -> None:
        self.blocked_edges.add((origin, destination))

    def add_static_obstacles(self, blocked: Iterable[XY], removable: Mapping[XY, float] | None = None) -> None:
        """Add template obstacles; removable tiles override any blocking actor."""
        for x, y in blocked:
            if self._inside(x, y):
                self.static_blocked.add((x, y))
        for (x, y), cost in (removable or {}).items():
            if not math.isfinite(cost) or cost < 0:
                raise ValueError("HM clearing cost must be finite and nonnegative")
            if self._inside(x, y):
                self.removable[x, y] = float(cost)
                self.static_blocked.discard((x, y))

    def _is_occupied(self, x: int, y: int, z: int) -> bool:
        if (x, y) in self.removable:
            return False
        if (x, y) in self.static_blocked:
            return True
        return any(z == 0 or actor_z == 0 or actor_z == z for actor_z in self.occupied.get((x, y), set()))

    @staticmethod
    def _next_elevation(current: int, tile: int) -> int | None:
        if tile == 15:
            return current
        if tile == 0 or current == 0 or tile == current:
            return tile
        return None

    def _walkable(self, x: int, y: int) -> bool:
        if not self._inside(x, y) or int(self.collision[y, x]) != 0:
            return False
        behavior = int(self.behavior[y, x])
        # Ordinary shallow water remains walkable. Surf requires water capability.
        if behavior in (0x10, 0x11, 0x12, 0x13, 0x14, 0x15) and not self.allow_surf:
            return False
        return True

    def _neighbors(self, state: State) -> list[tuple[State, float]]:
        x, y, z = state
        result: list[tuple[State, float]] = []
        for dx, dy, ledge, facing in _DIRECTIONS:
            nx, ny = x + dx, y + dy
            if not self._inside(nx, ny):
                continue
            behavior = int(self.behavior[ny, nx])
            # Origin blocks departure through that side; the destination blocks
            # entry through its opposite side. pret checks separate direction
            # and opposite-direction behavior tables. Testing the same side at
            # both ends invents one-way passages through cave/gym wall edges.
            if (_OPPOSITE_FACING[facing] in _DIRECTIONAL_BLOCKS.get(behavior, frozenset()) or
                    facing in _DIRECTIONAL_BLOCKS.get(int(self.behavior[y, x]), frozenset())):
                continue
            distance = 1
            if 0x38 <= behavior <= (0x3F if self.game == "emerald" else 0x3B):
                if behavior != ledge or self._is_occupied(nx, ny, z):
                    continue
                nx, ny, distance = x + 2 * dx, y + 2 * dy, 2
                if not self._walkable(nx, ny):
                    continue
                nz = int(self.elevation[ny, nx])
                if nz == 15:
                    nz = z
            else:
                if not self._walkable(nx, ny):
                    continue
                elevation = self._next_elevation(z, int(self.elevation[ny, nx]))
                if elevation is None:
                    continue
                nz = elevation
            if self._is_occupied(nx, ny, nz) or ((x, y), (nx, ny)) in self.blocked_edges:
                continue
            spin = _FIRERED_SPIN.get(int(self.behavior[ny, nx])) if self.game == "firered" else None
            if spin is not None and distance == 1:
                end = self._slide(nx, ny, nz, spin)
                self.slide_entries[((x, y), end[:2])] = (nx, ny)
                result.append((end[:3], 1.0 + end[3] + self.trainer_hazard_cost.get(end[:2], 0)))
                continue
            result.append(((nx, ny, nz), float(distance) + self.trainer_hazard_cost.get((nx, ny), 0)
                           + self.removable.get((nx, ny), 0.0)))
        return result

    def _slide(self, x: int, y: int, z: int, direction: tuple[int, int]) -> tuple[int, int, int, int]:
        """Where a spin starting on (x, y) stops, and how many tiles it slid."""
        dx, dy = direction
        steps = 0
        while steps < 96:
            nx, ny = x + dx, y + dy
            # A removable obstacle (item ball, tree) still stops a slide; it is
            # cleared by an action from the tile before it, not slid through.
            if not self._walkable(nx, ny) or self._is_occupied(nx, ny, z) or (nx, ny) in self.removable:
                break
            elevation = self._next_elevation(z, int(self.elevation[ny, nx]))
            if elevation is None:
                break
            x, y, z, steps = nx, ny, elevation, steps + 1
            behavior = int(self.behavior[y, x])
            if behavior == _FIRERED_STOP_SPIN:
                break
            turn = _FIRERED_SPIN.get(behavior)
            if turn is not None:
                dx, dy = turn
        return x, y, z, steps

    def find_optimal_path(self, start: State, goal: XY) -> list[XY] | None:
        """Use admissible Manhattan distance, since a ledge costs two tiles."""
        if not 0 <= start[2] <= 15 or not self._inside(*start[:2]) or not self._inside(*goal):
            return None
        frontier: list[tuple[float, float, State]] = [(0.0, 0.0, start)]
        previous: dict[State, State] = {}
        costs: dict[State, float] = {start: 0.0}
        while frontier:
            _, old_cost, current = heapq.heappop(frontier)
            if old_cost != costs[current]:
                continue
            if current[:2] == goal:
                path: list[XY] = []
                while current != start:
                    parent = previous[current]
                    path.append(current[:2])
                    entry = self.slide_entries.get((parent[:2], current[:2]))
                    if entry is not None and entry != current[:2]:
                        # The executable step is onto the spin tile; the slide
                        # then carries the player to the landing tile.
                        path.append(entry)
                    current = parent
                return path[::-1]
            if current != start and current[:2] in self.terminal_tiles:
                continue
            for neighbor, step_cost in self._neighbors(current):
                cost = old_cost + step_cost
                if cost >= costs.get(neighbor, math.inf):
                    continue
                costs[neighbor], previous[neighbor] = cost, current
                heuristic = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                heapq.heappush(frontier, (cost + heuristic, cost, neighbor))
        return None

    def reachable_costs(self, start: State) -> dict[XY, float]:
        """All reachable tiles and actual walking costs from one actor state.

        A single local Dijkstra answers every exit query without repeatedly
        running A* for each warp or connection on a large map.
        """
        if not 0 <= start[2] <= 15 or not self._inside(*start[:2]):
            return {}
        frontier: list[tuple[float, State]] = [(0.0, start)]
        costs: dict[State, float] = {start: 0.0}
        result: dict[XY, float] = {}
        while frontier:
            cost, current = heapq.heappop(frontier)
            if cost != costs[current]:
                continue
            result[current[:2]] = min(result.get(current[:2], math.inf), cost)
            if current != start and current[:2] in self.terminal_tiles:
                continue
            for neighbor, step_cost in self._neighbors(current):
                candidate = cost + step_cost
                if candidate < costs.get(neighbor, math.inf):
                    costs[neighbor] = candidate
                    heapq.heappush(frontier, (candidate, neighbor))
        return result

