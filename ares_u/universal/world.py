"""World memory and navigation of the universal agent (game-agnostic).

``World`` remembers, per map, what the agent has seen and done: visited cells, warps and edge
connections (with the destination learned when crossing them), NPCs talked to, items, signs, grass,
cells that turned out to be blocked, and how much story progress happened there. ``bfs`` plans
inside one map on the adapter's walkability grid (ledges are one-way jumps, warp cells are only
entered on purpose); ``World.route`` plans across maps over the transitions learned so far.
"""
from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .model import (DELTAS, DIRECTIONS, GRASS, LEDGE_DIRECTION, OPPOSITE, WARP_CELL, Direction, MapInfo, MapKey,
                    Npc)

Cell = tuple[int, int]
PASSABLE = frozenset({1, 4})          # FLOOR, GRASS: cells a path may cross
EDGE_DIRS: dict[Direction, str] = {"up": "north", "down": "south", "left": "west", "right": "east"}


@dataclass
class Target:
    """Something worth walking to. ``goals`` maps a stand cell to the direction to press there
    (face + A when ``interact``; walk in that direction otherwise; ``None`` = just arrive)."""

    key: tuple[Any, ...]
    kind: str
    value: float
    goals: dict[Cell, Direction | None]
    interact: bool = False
    label: str = ""
    npc: int | None = None
    via: tuple[Any, ...] | None = None    # route targets: key of the first transition to take

    def describe(self) -> str:
        return self.label or f"{self.kind} {self.key[1:]}"


@dataclass
class MapMemory:
    key: MapKey
    name: str = ""
    width: int = 0
    height: int = 0
    visits: int = 0
    outdoor: bool = False
    visited: set[Cell] = field(default_factory=set)
    walkable: int = 0
    warps: dict[Cell, MapKey | None] = field(default_factory=dict)
    warps_used: dict[Cell, MapKey] = field(default_factory=dict)
    connections: dict[str, MapKey] = field(default_factory=dict)
    connections_used: dict[str, MapKey] = field(default_factory=dict)
    npcs: dict[int, tuple[int, int, int, bool]] = field(default_factory=dict)
    talked: dict[int, int] = field(default_factory=dict)       # npc -> progress counter when last talked
    talk_gain: dict[int, float] = field(default_factory=dict)  # progress that followed talking to it
    signs: set[Cell] = field(default_factory=set)
    signs_read: set[Cell] = field(default_factory=set)
    grass: set[Cell] = field(default_factory=set)
    story: float = 0.0
    blocked: dict[Cell, int] = field(default_factory=dict)     # cell -> tick until which it is avoided
    arrivals: dict[str, Cell] = field(default_factory=dict)    # how we entered -> where we stood
    crossings: dict[str, set[Cell]] = field(default_factory=dict)  # edge direction -> cells that crossed
    counter_npcs: set[int] = field(default_factory=set)        # npcs served across a counter (clerk, nurse)
    heal_probed: dict[int, int] = field(default_factory=dict)  # npc -> tick of the last 'can you heal?' talk

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "visits": self.visits, "visited": len(self.visited),
                "walkable": self.walkable, "warps": len(self.warps), "warps_used": len(self.warps_used),
                "talked": len(self.talked), "npcs": len(self.npcs), "story": round(self.story, 2)}


def bfs(info: MapInfo, start: Cell, blocked: set[Cell], goals: set[Cell] | None = None
        ) -> tuple[dict[Cell, int], dict[Cell, tuple[Cell, Direction]]]:
    """Distances (in walk actions) and parents from ``start``.

    A path crosses FLOOR/GRASS cells; it may end on any cell in ``goals`` (warp cells, NPC-adjacent
    cells) and jumps a ledge only in the ledge's direction (landing one cell further).
    """
    goals = goals or set()
    dist: dict[Cell, int] = {start: 0}
    parent: dict[Cell, tuple[Cell, Direction]] = {}
    queue: deque[Cell] = deque([start])
    grid = info.grid
    width, height = info.width, info.height
    while queue:
        cell = queue.popleft()
        if cell != start and cell in goals and int(grid[cell[1], cell[0]]) == WARP_CELL:
            continue  # never walk through a warp: stepping on it would teleport
        cx, cy = cell
        for direction in DIRECTIONS:
            dx, dy = DELTAS[direction]
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            klass = int(grid[ny, nx])
            target = (nx, ny)
            if klass in LEDGE_DIRECTION:
                if LEDGE_DIRECTION[klass] != direction:
                    continue
                target = (nx + dx, ny + dy)
                if not (0 <= target[0] < width and 0 <= target[1] < height):
                    continue
                if int(grid[target[1], target[0]]) not in PASSABLE:
                    continue
            elif klass not in PASSABLE and target not in goals:
                continue
            if target in blocked or target in dist:
                continue
            dist[target] = dist[cell] + 1
            parent[target] = (cell, direction)
            queue.append(target)
    return dist, parent


def first_step(parent: dict[Cell, tuple[Cell, Direction]], start: Cell, goal: Cell) -> Direction | None:
    if goal == start:
        return None
    cell = goal
    direction: Direction | None = None
    guard = 0
    while cell != start and guard < 10000:
        previous, direction = parent[cell]
        cell = previous
        guard += 1
    return direction


def path_to(parent: dict[Cell, tuple[Cell, Direction]], start: Cell, goal: Cell) -> list[Direction]:
    steps: list[Direction] = []
    cell = goal
    while cell != start:
        previous, direction = parent[cell]
        steps.append(direction)
        cell = previous
    return steps[::-1]


def interaction_cells(info: MapInfo, x: int, y: int, counters: bool = True) -> dict[Cell, Direction]:
    """Cells from which the object at (x, y) can be talked to, with the direction to face.

    Adjacent cells, plus cells two steps away across a counter (shop clerks, nurses)."""
    out: dict[Cell, Direction] = {}
    for direction in DIRECTIONS:
        dx, dy = DELTAS[direction]
        stand = (x - dx, y - dy)
        if info.inside(*stand) and info.cell(*stand) in PASSABLE:
            out[stand] = direction
        if counters:
            middle = (x - dx, y - dy)
            far = (x - 2 * dx, y - 2 * dy)
            if info.inside(*far) and info.cell(*middle) == 10 and info.cell(*far) in PASSABLE:
                out[far] = direction
    return out


def edge_cells(info: MapInfo, direction: Direction) -> list[Cell]:
    if direction == "up":
        cells = [(x, 0) for x in range(info.width)]
    elif direction == "down":
        cells = [(x, info.height - 1) for x in range(info.width)]
    elif direction == "left":
        cells = [(0, y) for y in range(info.height)]
    else:
        cells = [(info.width - 1, y) for y in range(info.height)]
    return [cell for cell in cells if info.cell(*cell) in PASSABLE]


class World:
    """Per-map memory plus the transition graph learned by walking around."""

    def __init__(self) -> None:
        self.maps: dict[MapKey, MapMemory] = {}
        self.tick = 0

    def mem(self, key: MapKey) -> MapMemory:
        memory = self.maps.get(key)
        if memory is None:
            memory = self.maps[key] = MapMemory(key=key)
        return memory

    @property
    def visited_maps(self) -> list[MapKey]:
        return [key for key, memory in self.maps.items() if memory.visits > 0]

    def observe(self, info: MapInfo, npcs: list[Npc], x: int, y: int) -> MapMemory:
        memory = self.mem(info.key)
        if memory.width == 0:
            memory.name, memory.width, memory.height, memory.outdoor = info.name, info.width, info.height, info.outdoor
            memory.walkable = int(((info.grid == 1) | (info.grid == GRASS)).sum())
            memory.grass = {(int(cx), int(cy)) for cy, cx in zip(*(info.grid == GRASS).nonzero())}
        memory.visited.add((x, y))
        # Rebuilt every time: right after a warp the RAM may still hold the previous map's list.
        memory.warps = {(w.x, w.y): memory.warps_used.get((w.x, w.y), w.dest_map) for w in info.warps}
        memory.connections = {c.direction: c.dest_map for c in info.connections}
        memory.signs.update(info.signs)
        for npc in npcs:
            memory.npcs[npc.index] = (npc.x, npc.y, npc.sprite, npc.is_item)
        return memory

    def record_warp(self, src: MapKey, cell: Cell, dest: MapKey, arrival: Cell) -> None:
        memory = self.mem(src)
        memory.warps_used[cell] = dest
        memory.warps[cell] = dest
        self.mem(dest).arrivals[f"warp:{src[0]}.{src[1]}"] = arrival

    def record_connection(self, src: MapKey, direction: str, dest: MapKey, arrival: Cell,
                          departure: Cell | None = None) -> None:
        memory = self.mem(src)
        memory.connections_used[direction] = dest
        memory.connections[direction] = dest
        if departure is not None:
            memory.crossings.setdefault(direction, set()).add(departure)
        dest_memory = self.mem(dest)
        dest_memory.arrivals[f"conn:{src[0]}.{src[1]}"] = arrival
        back = OPPOSITE[direction]  # type: ignore[index]
        dest_memory.crossings.setdefault(back, set()).add(arrival)

    def crossing_goals(self, info: MapInfo, direction: Direction) -> list[Cell]:
        """Edge cells to leave through ``direction``: the ones known to work, else every open one."""
        known = [c for c in self.mem(info.key).crossings.get(direction, set()) if info.inside(*c)]
        return known or edge_cells(info, direction)

    def blocked_cells(self, key: MapKey) -> set[Cell]:
        memory = self.mem(key)
        return {cell for cell, until in memory.blocked.items() if until > self.tick}

    def edges(self, key: MapKey) -> list[tuple[MapKey, tuple[str, Any]]]:
        """Known transitions out of ``key``: (destination, ("warp", cell) | ("conn", direction))."""
        memory = self.maps.get(key)
        if memory is None:
            return []
        out: list[tuple[MapKey, tuple[str, Any]]] = []
        for cell, dest in memory.warps.items():
            if dest is not None:
                out.append((dest, ("warp", cell)))
        for direction, dest in memory.connections.items():
            out.append((dest, ("conn", direction)))
        return out

    def route(self, start: MapKey, hop_cost: float = 25.0) -> dict[MapKey, tuple[float, tuple[str, Any] | None]]:
        """Dijkstra over maps: cost to reach every known map and the first transition to take."""
        best: dict[MapKey, tuple[float, tuple[str, Any] | None]] = {start: (0.0, None)}
        heap: list[tuple[float, int, MapKey, tuple[str, Any] | None]] = [(0.0, 0, start, None)]
        counter = 0
        while heap:
            cost, _, key, first = heapq.heappop(heap)
            if cost > best.get(key, (float("inf"), None))[0]:
                continue
            for dest, via in self.edges(key):
                step = hop_cost if self.mem(dest).visits > 0 or dest not in self.maps else hop_cost
                new_cost = cost + step
                if new_cost < best.get(dest, (float("inf"), None))[0]:
                    counter += 1
                    best[dest] = (new_cost, first if first is not None else via)
                    heapq.heappush(heap, (new_cost, counter, dest, first if first is not None else via))
        return best

    def summary(self) -> dict[str, Any]:
        return {f"{key[0]}.{key[1]}": memory.to_json() for key, memory in self.maps.items() if memory.visits}


def opposite(direction: Direction) -> Direction:
    return OPPOSITE[direction]
