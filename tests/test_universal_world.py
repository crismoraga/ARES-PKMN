"""Universal player: map graph (BFS, routes, learned transitions) and anti-stuck logic, no emulator."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ares_u.universal.agent import PINGPONG_HOPS, UniversalAgent
from ares_u.universal.knowledge import KnowledgeBase
from ares_u.universal.model import FLOOR, WALL, WARP_CELL, Connection, MapInfo, Warp
from ares_u.universal.world import World, bfs, edge_cells, first_step, path_to


def room(width: int = 6, height: int = 5, walls: tuple[tuple[int, int], ...] = ()) -> MapInfo:
    grid = np.full((height, width), FLOOR, dtype=np.int8)
    for x, y in walls:
        grid[y, x] = WALL
    grid[height - 1, 2] = WARP_CELL
    return MapInfo(key=(0, 1), width=width, height=height, grid=grid,
                   warps=[Warp(x=2, y=height - 1, dest_map=(0, 2), dest_index=0)],
                   connections=[Connection(direction="up", dest_map=(0, 3))], signs=[], name="test")


def test_bfs_goes_around_walls_and_ends_on_warp() -> None:
    info = room(walls=((1, 1), (2, 1), (3, 1)))
    dist, parent = bfs(info, (2, 0), set(), {(2, 4)})
    assert (2, 4) in dist
    steps = path_to(parent, (2, 0), (2, 4))
    assert len(steps) == dist[(2, 4)] == 8  # 2 sideways + 4 down + 2 back: detour around the wall row
    assert first_step(parent, (2, 0), (2, 4)) in ("left", "right")


def test_bfs_respects_blocked_cells() -> None:
    info = room(walls=((0, 2), (1, 2), (2, 2), (3, 2), (4, 2)))
    dist, _ = bfs(info, (0, 0), set(), {(2, 4)})
    assert (2, 4) in dist                   # gap at x=5
    dist, _ = bfs(info, (0, 0), {(5, 2)}, {(2, 4)})
    assert (2, 4) not in dist               # an NPC standing in the gap cuts the room


def test_edge_cells_only_open_ones() -> None:
    info = room(walls=((0, 0), (1, 0)))
    assert edge_cells(info, "up") == [(2, 0), (3, 0), (4, 0), (5, 0)]


def test_world_route_through_learned_transitions() -> None:
    world = World()
    world.observe(room(), [], 2, 2)
    world.mem((0, 1)).visits = 1
    world.record_warp((0, 1), (2, 4), (0, 2), (5, 5))
    world.mem((0, 2)).connections["left"] = (0, 4)
    routes = world.route((0, 1))
    assert routes[(0, 2)][1] == ("warp", (2, 4))
    assert routes[(0, 4)][1] == ("warp", (2, 4))       # first hop of a two-map trip
    assert routes[(0, 3)][1] == ("conn", "up")         # connection read from the header
    assert routes[(0, 4)][0] > routes[(0, 2)][0]
    # the crossing that worked is remembered on the far side as the way back
    world.record_connection((0, 1), "up", (0, 3), (2, 9), (2, 0))
    assert world.mem((0, 1)).crossings["up"] == {(2, 0)}
    assert world.mem((0, 3)).crossings["down"] == {(2, 9)}


class StubAdapter:
    key = "stub"
    generation = 1
    frame_count = 0
    settle_waits = 0
    settle_timeouts = 0

    def map_name(self, key: Any) -> str:
        return f"M{key[1]}"


def make_agent(tmp_path: Path) -> UniversalAgent:
    kb = KnowledgeBase.empty(tmp_path / "kb.json")
    return UniversalAgent(StubAdapter(), kb, run_id="t", run_dir=tmp_path / "run",  # type: ignore[arg-type]
                          publish=False, use_brain=False)


def test_pingpong_between_two_maps_is_detected(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    a, b = (0, 0), (0, 12)
    for hop in range(PINGPONG_HOPS):
        agent.tick += 10
        if hop % 2 == 0:
            agent.note_transition(("conn", a, "up"), a, b)
        else:
            agent.note_transition(("conn", b, "down"), b, a)
    assert agent.stats.get("pingpong") == 1
    assert agent.is_tabu(("conn", a, "up")) and agent.is_tabu(("conn", b, "down"))
    assert agent.is_tabu(("route", b))
    assert agent.stuck_events[-1]["kind"] == "pingpong"


def test_no_pingpong_when_progress_or_slow(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    a, b = (0, 0), (0, 12)
    for hop in range(PINGPONG_HOPS):
        agent.tick += 10
        if hop == 3:
            agent.progress_events += 1  # something new happened in between: a real errand
        agent.note_transition(("conn", a if hop % 2 == 0 else b, "x"), a if hop % 2 == 0 else b,
                              b if hop % 2 == 0 else a)
    assert agent.stats.get("pingpong") is None
    slow = make_agent(tmp_path / "slow")
    for hop in range(PINGPONG_HOPS):
        slow.tick += 400  # one hop every 400 ticks is travelling, not oscillating
        slow.note_transition(("warp", a, (1, 1)), a, b)
    assert slow.stats.get("pingpong") is None


def test_story_return_belief_is_optimistic_then_learns(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    assert agent.return_belief() == 0.5
    agent.kb.skills["story_return"] = {"tries": 3, "success": 0}
    assert agent.return_belief() == 0.2
    agent.kb.skills["story_return"] = {"tries": 3, "success": 3}
    assert agent.return_belief() == 0.8


def test_tabu_expires(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    agent.set_tabu(("warp", (0, 1), (2, 4)), 100)
    assert agent.is_tabu(("warp", (0, 1), (2, 4)))
    agent.tick += 101
    assert not agent.is_tabu(("warp", (0, 1), (2, 4)))
