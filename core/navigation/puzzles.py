"""Exact bounded searches for Strength switches and single-visit ice floors."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import itertools
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from core.bridge.ffi_mgba import GBABridgeProtocol
from core.navigation.constrained_astar import GbaConstrainedAStar, Matrix
from core.navigation.source_data import SourceMapCatalog

XY = tuple[int, int]
_MOVES = ((1, 0, 16), (-1, 0, 32), (0, -1, 64), (0, 1, 128))
SokobanState = tuple[XY, frozenset[XY]]


@dataclass(frozen=True)
class PuzzleStep:
    origin: XY
    destination: XY
    direction: int
    pushed_boulder: XY | None = None


def solve_strength_puzzle(collision: NDArray[np.uint16], player: XY,
                          boulders: frozenset[XY], switches: frozenset[XY],
                          max_states: int = 250_000, exit_tile: XY | None = None) -> list[PuzzleStep] | None:
    """Dijkstra through push states; impossible puzzles return None.

    At most one boulder moves per cardinal action, a push requires a free landing
    tile, and boulders cannot be pulled. Collision is copied to prevent a caller's
    mutable live snapshot from changing a search already underway.
    """
    if collision.ndim != 2 or max_states < 1 or len(switches) > len(boulders):
        raise ValueError("Invalid Strength puzzle")
    grid = collision.copy()
    height, width = grid.shape

    def free(point: XY) -> bool:
        x, y = point
        return 0 <= x < width and 0 <= y < height and int(grid[y, x]) == 0

    if not free(player) or any(not free(point) for point in boulders | switches) or (exit_tile is not None and not free(exit_tile)):
        raise ValueError("Actor, boulder or switch outside the traversable grid")
    initial = player, boulders
    counter = itertools.count()
    frontier: list[tuple[int, int, SokobanState]] = [(0, next(counter), initial)]
    previous: dict[SokobanState, tuple[SokobanState, list[PuzzleStep]]] = {}
    distances = {initial: 0}
    best_goal: tuple[int, SokobanState, list[PuzzleStep]] | None = None
    while frontier:
        cost, _, state = heapq.heappop(frontier)
        if cost != distances[state]:
            continue
        if best_goal is not None and cost >= best_goal[0]:
            break
        current, stones = state
        reachable: dict[XY, list[PuzzleStep]] = {current: []}
        walk_frontier = deque([current])
        while walk_frontier:
            origin = walk_frontier.popleft()
            for dx, dy, direction in _MOVES:
                target = origin[0] + dx, origin[1] + dy
                if free(target) and target not in stones and target not in reachable:
                    reachable[target] = reachable[origin] + [PuzzleStep(origin, target, direction)]
                    walk_frontier.append(target)
        if switches <= stones and (exit_tile is None or exit_tile in reachable):
            tail = [] if exit_tile is None else reachable[exit_tile]
            finish_cost = cost + len(tail)
            if best_goal is None or finish_cost < best_goal[0]:
                best_goal = finish_cost, state, tail
        for stone in stones:
            for dx, dy, direction in _MOVES:
                approach = stone[0] - dx, stone[1] - dy
                landing = stone[0] + dx, stone[1] + dy
                if approach not in reachable:
                    continue
                if not free(landing) or landing in stones:
                    continue
                # A nongoal corner is dead only when switches must be filled.
                if len(switches) == len(stones) and switches and landing not in switches:
                    horizontal_wall = not free((landing[0] - 1, landing[1])) or not free((landing[0] + 1, landing[1]))
                    vertical_wall = not free((landing[0], landing[1] - 1)) or not free((landing[0], landing[1] + 1))
                    if horizontal_wall and vertical_wall:
                        continue
                segment = reachable[approach] + [PuzzleStep(approach, stone, direction, stone)]
                neighbor = stone, frozenset((stones - {stone}) | {landing})
                new_cost = cost + len(segment)
                if new_cost >= distances.get(neighbor, 1 << 60):
                    continue
                if len(distances) >= max_states:
                    return None
                distances[neighbor], previous[neighbor] = new_cost, (state, segment)
                heapq.heappush(frontier, (new_cost, next(counter), neighbor))
    if best_goal is None:
        return None
    _, state, tail = best_goal
    segments = [tail]
    while state != initial:
        state, segment = previous[state]
        segments.append(segment)
    return [step for segment in reversed(segments) for step in segment]


def solve_ice_floor(tiles: frozenset[XY], start: XY, exit_tile: XY,
                    max_states: int = 250_000) -> list[XY] | None:
    """Visit every ice tile once, then finish at the exit without revisiting ice."""
    if not tiles or start not in tiles or exit_tile not in tiles or max_states < 1:
        raise ValueError("Ice puzzle needs a valid start, exit and search bound")
    expanded = 0
    exhausted: set[tuple[XY, frozenset[XY]]] = set()

    def search(current: XY, remaining: frozenset[XY]) -> list[XY] | None:
        nonlocal expanded
        if not remaining:
            return [] if current == exit_tile else None
        if current == exit_tile or (current, remaining) in exhausted:
            return None
        expanded += 1
        if expanded > max_states:
            return None
        candidates = [(current[0] + dx, current[1] + dy) for dx, dy, _ in _MOVES
                      if (current[0] + dx, current[1] + dy) in remaining]
        # Fail quickly on isolated unvisited squares; otherwise prefer tight ends.
        def degree(point: XY) -> int:
            return sum((point[0] + dx, point[1] + dy) in remaining for dx, dy, _ in _MOVES)
        for point in sorted(candidates, key=degree):
            rest = remaining - {point}
            if any(degree(cell) == 0 and cell != point and cell != exit_tile for cell in rest):
                continue
            suffix = search(point, rest)
            if suffix is not None:
                return [point] + suffix
        exhausted.add((current, remaining))
        return None

    return search(start, tiles - {start})


def _move_commands(map_key: tuple[int, int], steps: list[PuzzleStep], puzzle: str) -> list[dict[str, Any]]:
    return [{"mode": "EXECUTE_PUZZLE", "dest": (*map_key, step.destination), "action": "push" if step.pushed_boulder else "step",
             "direction": step.direction, "expected_origin": step.origin, "pushed_boulder": step.pushed_boulder,
             "required_move": 70 if step.pushed_boulder else 0, "puzzle": puzzle} for step in steps]


def plan_strength_commands(bridge: GBABridgeProtocol, catalog: SourceMapCatalog,
                            exit_tile: XY | None = None, max_states: int = 25_000) -> list[dict[str, Any]] | None:
    """Translate current boulders and pressure plates into executable key goals."""
    coords = bridge.get_player_coords_and_map()
    key = coords["map_group"], coords["map_num"]
    _, collision, elevation = bridge.extract_live_map_matrices()
    behaviors = bridge.extract_behavior_map()
    object_id = bridge.read_u8(bridge.memory_map.avatar + 5)
    boulder_graphics = catalog.constants.get("OBJ_EVENT_GFX_PUSHABLE_BOULDER", -1)
    stones: dict[XY, int] = {}
    for index, event in enumerate(bridge.get_object_events()):
        if index == object_id or not int(event.flags) & 1:
            continue
        point = int(event.currentCoords_x) - 7, int(event.currentCoords_y) - 7
        if not 0 <= point[0] < collision.shape[1] or not 0 <= point[1] < collision.shape[0]:
            continue
        if int(event.graphicsId) == boulder_graphics:
            stones[point] = int(event.localId)
        else:
            collision[point[1], point[0]] = 1
    if not stones:
        return None
    # Native boulders cannot cross an unrelated elevation layer.
    collision[(elevation != coords["z"]) & (elevation != 0) & (elevation != 15)] = 1
    for x, y in stones:
        collision[y, x] = 0
    button = catalog.constants.get("MB_STRENGTH_BUTTON", -1)
    switches = frozenset((int(x), int(y)) for y, x in np.argwhere(behaviors == button))
    if not switches and exit_tile is None:
        return None
    strength_flag = catalog.constants.get("FLAG_SYS_USE_STRENGTH")
    if strength_flag is not None and not bridge.get_event_flag(strength_flag):
        local = GbaConstrainedAStar(collision, elevation, behaviors, game=catalog.game)
        local.occupied = {point: {coords["z"]} for point in stones}
        player = coords["x"], coords["y"], coords["z"]
        candidates = [(abs(x - player[0]) + abs(y - player[1]), (x, y), local_id)
                      for (x, y), local_id in stones.items()
                      if any(local.find_optimal_path(player, (x + dx, y + dy)) is not None for dx, dy, _ in _MOVES)]
        if not candidates:
            return None
        _, point, local_id = min(candidates)
        return [{"mode": "EXECUTE_PUZZLE", "dest": (*key, point), "action": "interact", "npc_local_id": local_id,
                 "script": "EventScript_StrengthBoulder", "required_move": 70, "puzzle": "strength_activation", "menu_choice": 0}]
    solution = solve_strength_puzzle(collision, (coords["x"], coords["y"]), frozenset(stones), switches,
                                     max_states=max_states, exit_tile=exit_tile if not switches else None)
    return None if solution is None else _move_commands(key, solution, "strength")


def plan_sootopolis_ice_commands(bridge: GBABridgeProtocol, catalog: SourceMapCatalog,
                                max_states: int = 250_000) -> list[dict[str, Any]] | None:
    """Solve the next native ice tier from unvisited/cracked RAM behaviors.

    pret's step callback tracks the three row bands 17..19, 12..14 and 6..9;
    stair cells are unlocked only after every tile in the current tier is visited.
    """
    coords = bridge.get_player_coords_and_map()
    key = coords["map_group"], coords["map_num"]
    if catalog.maps.get(key, {}).get("name") != "SootopolisCity_Gym_1F":
        return None
    _, collision, elevation = bridge.extract_live_map_matrices()
    behaviors = bridge.extract_behavior_map()
    thin, cracked = catalog.constants["MB_THIN_ICE"], catalog.constants["MB_CRACKED_ICE"]
    player = coords["x"], coords["y"]
    bands = ((17, 19), (12, 14), (6, 9))
    band = next(((low, high) for low, high in bands if player[1] >= low), None)
    if band is None:
        return []
    low, high = band
    tiles = frozenset((int(x), int(y)) for y, x in np.argwhere(behaviors == thin) if low <= y <= high)
    if not tiles:
        return []
    if low <= player[1] <= high and int(behaviors[player[1], player[0]]) == cracked:
        tiles |= {player}
    exit_tile = min((point for point in tiles if point[1] == min(y for _, y in tiles)), key=lambda xy: abs(xy[0] - 8))
    local_collision = collision.copy()
    local_collision[(behaviors == thin) | (behaviors == cracked)] = 1
    local = GbaConstrainedAStar(local_collision, elevation, behaviors, game="emerald")
    starts: list[tuple[XY, list[XY]]]
    if player in tiles:
        starts = [(player, [])]
    else:
        starts = []
        for tile in tiles:
            for dx, dy, _ in _MOVES:
                approach = tile[0] - dx, tile[1] - dy
                path = local.find_optimal_path((player[0], player[1], coords["z"]), approach)
                if path is not None:
                    starts.append((tile, path + [tile]))
    for start, approach_path in sorted(starts, key=lambda entry: len(entry[1])):
        path = solve_ice_floor(tiles, start, exit_tile, max_states=max_states)
        if path is None:
            continue
        xy_path = approach_path + path
        steps: list[PuzzleStep] = []
        origin = player
        for point in xy_path:
            dx, dy = point[0] - origin[0], point[1] - origin[1]
            direction = next((mask for x, y, mask in _MOVES if (x, y) == (dx, dy)), 0)
            if not direction:
                break
            steps.append(PuzzleStep(origin, point, direction))
            origin = point
        else:
            return _move_commands(key, steps, "sootopolis_ice")
    return None

