"""Learn passages that a script refuses: the guard walks the player back.

Static analysis (``script_gates``) already knows the source-defined gates. This
learner covers everything else: an event sends the avatar back the way it came
right after a voluntary step. The refused step (and sibling trigger tiles with
the same scene variable) is then avoided while the *world* is unchanged; any
new milestone, badge or key item re-opens it for one more attempt, because the
missing requirement may now be met. Nothing here writes RAM.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from core.navigation.obstacles import gate_entry_edges
from core.navigation.script_gates import gate_at
from core.navigation.source_data import MapKey, load_map_catalog

XY = tuple[int, int]
Edge = tuple[XY, XY]
_MAX_BARRIERS = 256
# Learner revision. Barriers written by earlier revisions (which could mistake
# the agent's own step back for a refusal) are not trusted when loaded.
BARRIER_VERSION = 2
# A refusal with no trigger script explaining it is only believed for a while.
UNEXPLAINED_LIFETIME = 30000


@dataclass
class LearnedBarrier:
    """Directed steps a script undid, with the world in which it happened."""
    map_key: MapKey
    map_name: str
    tile: XY
    direction: XY
    edges: frozenset[Edge]
    world: str
    observations: int
    first_frame: int
    last_frame: int
    script: str = ""
    requirement: str = ""
    version: int = BARRIER_VERSION

    @property
    def explained(self) -> bool:
        """A trigger script on that tile explains the refusal."""
        return bool(self.script)

    def to_json(self) -> dict[str, Any]:
        row = asdict(self)
        row["map_key"], row["tile"], row["direction"] = list(self.map_key), list(self.tile), list(self.direction)
        row["edges"] = [[list(a), list(b)] for a, b in sorted(self.edges)]
        return row

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> LearnedBarrier:
        def pair(value: Any) -> XY:
            if not isinstance(value, (list, tuple)) or len(value) != 2 or any(type(item) is not int for item in value):
                raise ValueError("Invalid barrier coordinate")
            return int(value[0]), int(value[1])
        raw_edges = row.get("edges")
        if not isinstance(raw_edges, list) or len(raw_edges) > 64:
            raise ValueError("Invalid barrier edges")
        edges = frozenset((pair(edge[0]), pair(edge[1])) for edge in raw_edges
                          if isinstance(edge, (list, tuple)) and len(edge) == 2)
        values = [row.get(name) for name in ("observations", "first_frame", "last_frame")]
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("Invalid barrier counters")
        version = row.get("version", 1)
        return cls(pair(row.get("map_key")), str(row.get("map_name", "")), pair(row.get("tile")),
                   pair(row.get("direction")), edges, str(row.get("world", "")), int(row["observations"]),
                   int(row["first_frame"]), int(row["last_frame"]), str(row.get("script", "")),
                   str(row.get("requirement", "")), int(version) if isinstance(version, int) else 1)


@dataclass
class _Step:
    map_key: MapKey
    origin: XY
    destination: XY
    frame: int


@dataclass
class PushbackLearner:
    game: str
    barriers: dict[tuple[MapKey, XY, XY], LearnedBarrier] = field(default_factory=dict)  # (map, tile, direction)
    suspects: dict[tuple[MapKey, XY, XY, str], int] = field(default_factory=dict)
    _step: _Step | None = None
    _script_start: tuple[MapKey, XY, int] | None = None

    def note_step(self, map_key: MapKey, origin: XY, destination: XY, frame: int) -> None:
        """A voluntary step the agent itself requested arrived."""
        self._script_start = None
        if abs(origin[0] - destination[0]) + abs(origin[1] - destination[1]) not in (1, 2):
            self._step = None
            return
        self._step = _Step(map_key, origin, destination, frame)

    def reset(self) -> None:
        self._step = None
        self._script_start = None

    def observe(self, map_key: MapKey, position: XY, script_running: bool, frame: int,
                world: str) -> LearnedBarrier | None:
        """Feed every overworld frame; returns a barrier when a pushback completes.

        Only displacement that happens while an event script runs counts:
        ordinary walking (including the agent's own next step back) never does.
        """
        step = self._step
        if step is None:
            return None
        if map_key != step.map_key or frame - step.frame > 1800 or frame < step.frame:
            self.reset()
            return None
        if script_running:
            if self._script_start is None and position == step.destination:
                self._script_start = (map_key, position, frame)
            return None
        start = self._script_start
        if start is None:
            if position != step.destination:
                self._step = None  # the agent walked on by itself
            return None
        self._script_start = None
        self._step = None
        displacement = (position[0] - start[1][0], position[1] - start[1][1])
        if displacement == (0, 0):
            return None
        direction = (step.destination[0] - step.origin[0], step.destination[1] - step.origin[1])
        if displacement[0] * direction[0] + displacement[1] * direction[1] >= 0:
            return None  # carried forward or sideways: a cutscene, not a refusal
        return self._record(map_key, step, (_sign(direction[0]), _sign(direction[1])), displacement, frame, world)

    def _record(self, map_key: MapKey, step: _Step, direction: XY, displacement: XY,
                frame: int, world: str) -> LearnedBarrier | None:
        catalog = load_map_catalog(self.game)
        raw = catalog.maps.get(map_key, {})
        tile = step.destination
        triggers = [event for event in raw.get("coord_events", []) or []
                    if str(event.get("type", "trigger")) == "trigger" and
                    (int(event.get("x", -1)), int(event.get("y", -1))) == tile]
        gate = gate_at(self.game, map_key, tile)
        key = (map_key, tile, direction)
        suspect_key = (map_key, step.origin, tile, world)
        self.suspects[suspect_key] = self.suspects.get(suspect_key, 0) + 1
        if len(self.suspects) > 1024:
            self.suspects.clear()
        # A coord trigger that is still armed explains the refusal at once;
        # an unexplained displacement must repeat before it constrains routing.
        if not triggers and gate is None and self.suspects[suspect_key] < 2:
            return None
        edges: set[Edge] = {(step.origin, tile)}
        siblings: list[XY] = []
        if triggers:
            variable, value = str(triggers[0].get("var", "")), str(triggers[0].get("var_value", ""))
            for event in raw.get("coord_events", []) or []:
                if (str(event.get("type", "trigger")) == "trigger" and str(event.get("var", "")) == variable and
                        str(event.get("var_value", "")) == value):
                    siblings.append((int(event["x"]), int(event["y"])))
        pushback = (_sign(displacement[0]), _sign(displacement[1]))
        for sibling in siblings:
            if abs(sibling[0] - tile[0]) + abs(sibling[1] - tile[1]) <= 8:
                edges.update(edge for edge in gate_entry_edges(sibling, pushback)
                             if (edge[1][0] - edge[0][0]) * direction[0] + (edge[1][1] - edge[0][1]) * direction[1] > 0)
        script = gate.script if gate is not None else (str(triggers[0].get("script", "")) if triggers else "")
        requirement = gate.requirement if gate is not None else ""
        existing = self.barriers.get(key)
        if existing is not None:
            existing.edges = existing.edges | frozenset(edges)
            existing.world, existing.last_frame = world, frame
            existing.observations += 1
            return existing
        if len(self.barriers) >= _MAX_BARRIERS:
            oldest = min(self.barriers, key=lambda name: self.barriers[name].last_frame)
            del self.barriers[oldest]
        barrier = LearnedBarrier(map_key, str(raw.get("name", "")), tile, direction, frozenset(edges), world,
                                 1, frame, frame, script, requirement)
        self.barriers[key] = barrier
        return barrier

    def active_edges(self, world: str, frame: int | None = None,
                     unexplained: bool = True) -> dict[MapKey, frozenset[Edge]]:
        """Barriers learned in the current world; a changed world earns a retry.

        Refusals that no trigger script explains expire after a while (and can
        be left out entirely when they would trap the agent): a wrong lesson
        must not become a permanent wall.
        """
        grouped: dict[MapKey, set[Edge]] = {}
        for barrier in self.barriers.values():
            if barrier.world != world:
                continue
            if not barrier.explained:
                if not unexplained:
                    continue
                if frame is not None and not 0 <= frame - barrier.last_frame <= UNEXPLAINED_LIFETIME:
                    continue
            grouped.setdefault(barrier.map_key, set()).update(barrier.edges)
        return {key: frozenset(edges) for key, edges in grouped.items()}

    def forget_unexplained(self, map_key: MapKey | None = None) -> int:
        """Drop unexplained refusals (of one map), e.g. when they leave no way out."""
        doomed = [key for key, barrier in self.barriers.items()
                  if not barrier.explained and (map_key is None or barrier.map_key == map_key)]
        for key in doomed:
            del self.barriers[key]
        return len(doomed)

    def describe(self, world: str) -> list[dict[str, Any]]:
        rows = sorted(self.barriers.values(), key=lambda barrier: -barrier.last_frame)
        return [{"map": barrier.map_name, "tile": list(barrier.tile), "script": barrier.script,
                 "requirement": barrier.requirement, "observations": barrier.observations,
                 "active": barrier.world == world} for barrier in rows[:16]]

    def merge(self, barriers: Iterable[LearnedBarrier]) -> None:
        for barrier in barriers:
            if barrier.version < BARRIER_VERSION and not barrier.explained:
                continue  # possibly the old learner's mistake
            key = (barrier.map_key, barrier.tile, barrier.direction)
            current = self.barriers.get(key)
            if current is None:
                if len(self.barriers) < _MAX_BARRIERS:
                    self.barriers[key] = barrier
                continue
            current.edges = current.edges | barrier.edges
            current.observations = max(current.observations, barrier.observations)
            if barrier.last_frame > current.last_frame:
                current.world, current.last_frame = barrier.world, barrier.last_frame

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "barriers": [barrier.to_json() for barrier in self.barriers.values()]}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 1 or not isinstance(state.get("barriers"), list):
            raise ValueError("Invalid pushback learner state")
        barriers = [barrier for barrier in (LearnedBarrier.from_json(row) for row in state["barriers"][:_MAX_BARRIERS]
                                            if isinstance(row, dict))
                    if barrier.version >= BARRIER_VERSION or barrier.explained]
        self.barriers = {}
        self.merge(barriers)
        self.reset()


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)
