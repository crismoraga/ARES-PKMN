"""A one-frame-owner movement FSM: tick observes RAM and sets keys only."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from core.bridge.ffi_mgba import GBABridgeProtocol

MovementMode = Literal["walk", "run", "bike"]
MovementStatus = Literal["arrived", "blocked", "map_changed", "battle", "interrupted"]
MapKey = tuple[int, int]
BlockedEdge = tuple[int, int, int, int, int, int]
_DELTAS = {16: (1, 0), 32: (-1, 0), 64: (0, -1), 128: (0, 1)}


@dataclass(frozen=True)
class MovementEvent:
    status: MovementStatus
    origin: tuple[int, int]
    destination: tuple[int, int]
    origin_map: MapKey
    destination_map: MapKey
    frames: int
    direction: int


@dataclass
class _PendingStep:
    origin: tuple[int, int]
    origin_map: MapKey
    direction: int
    started_frame: int
    mode: MovementMode
    target: tuple[int, int]


class StateFeedbackMovementController:
    """Hold direction until coordinate feedback; detect idle collision at 24.

    Native ledges and turning can exceed an ordinary step. An actor that is still
    moving at the collision deadline is never marked blocked. Script/battle/map
    transitions release keys and produce explicit events instead of deadlocking.
    """

    WALK_FRAMES, RUN_FRAMES, BIKE_FRAMES, COLLISION_TIMEOUT = 16, 8, 4, 24

    def __init__(self, bridge: GBABridgeProtocol) -> None:
        self.bridge = bridge
        self._pending: _PendingStep | None = None
        self.blocked_edges: set[BlockedEdge] = set()
        self.events: list[MovementEvent] = []

    @property
    def pending(self) -> bool:
        return self._pending is not None

    def request_step(self, direction: int, mode: MovementMode = "walk") -> bool:
        if direction not in _DELTAS or mode not in ("walk", "run", "bike"):
            raise ValueError("One cardinal direction and a supported movement mode required")
        if self.pending or self.bridge.is_in_battle()[0] or not self.bridge.is_hardware_ready():
            return False
        coords = self.bridge.get_player_coords_and_map()
        dx, dy = _DELTAS[direction]
        self._pending = _PendingStep((coords["x"], coords["y"]),
                                    (coords["map_group"], coords["map_num"]), direction,
                                    self.bridge.frame_count, mode, (coords["x"] + dx, coords["y"] + dy))
        self.bridge.set_hardware_keys(direction | (self.bridge.KEY_B if mode == "run" else 0))
        return True

    def move_toward(self, destination: tuple[int, int], mode: MovementMode = "walk") -> bool:
        coords = self.bridge.get_player_coords_and_map()
        dx, dy = destination[0] - coords["x"], destination[1] - coords["y"]
        if dx and dy or abs(dx) + abs(dy) not in (1, 2):
            raise ValueError("A path edge must be a cardinal step or ledge jump")
        direction = 16 if dx > 0 else (32 if dx < 0 else (128 if dy > 0 else 64))
        accepted = self.request_step(direction, mode)
        if accepted and self._pending is not None:
            self._pending.target = destination
        return accepted

    def tick(self) -> MovementEvent | None:
        step = self._pending
        if step is None:
            return None
        coords = self.bridge.get_player_coords_and_map()
        elapsed = self.bridge.frame_count - step.started_frame
        destination = (coords["x"], coords["y"])
        destination_map = (coords["map_group"], coords["map_num"])
        status: MovementStatus | None = None
        if destination_map != step.origin_map:
            status = "map_changed"
        elif self.bridge.is_in_battle()[0]:
            status = "battle"
        elif destination != step.origin and coords["tile_transition_state"] != 1:
            status = "arrived"
        elif self.bridge.read_u8(self.bridge.memory_map.script + 1) != 0:
            status = "interrupted"
        elif elapsed >= self.COLLISION_TIMEOUT and coords["tile_transition_state"] != 1:
            status = "blocked"
        elif elapsed >= 240:
            status = "interrupted"
        if status is None:
            return None
        self.bridge.set_hardware_keys(0)
        self._pending = None
        event = MovementEvent(status, step.origin, destination, step.origin_map,
                              destination_map, elapsed, step.direction)
        self.events.append(event)
        if status == "blocked":
            self.blocked_edges.add((*step.origin_map, *step.origin, *step.target))
        return event

    def clear(self) -> None:
        """Cancel held input after loading a savestate; retain learned obstacles."""
        self._pending = None
        self.bridge.set_hardware_keys(0)

    def state_dict(self) -> dict[str, object]:
        pending: dict[str, object] | None = None
        if self._pending is not None:
            pending = {"origin": list(self._pending.origin), "origin_map": list(self._pending.origin_map),
                       "direction": self._pending.direction, "mode": self._pending.mode,
                       "target": list(self._pending.target),
                       "elapsed_frames": self.bridge.frame_count - self._pending.started_frame}
        return {"version": 1, "pending": pending,
                "blocked_edges": [list(edge) for edge in sorted(self.blocked_edges)],
                "events": [asdict(event) for event in self.events]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore after the matching emulator state, preserving held movement.

        An elapsed counter rebases to the restored bridge's frame count. Restoring
        the FSM never advances frames and never resets an in-progress movement's
        collision deadline. Validation is atomic: bad state cannot alter keys.
        """
        if state.get("version", 1) != 1:
            raise ValueError("Unsupported movement state version")
        edges = self._decode_edges(state.get("blocked_edges", []))
        pending = None
        raw_pending = state.get("pending")
        if raw_pending is not None:
            if not isinstance(raw_pending, dict):
                raise ValueError("Invalid pending movement")
            origin = self._decode_pair(raw_pending.get("origin"))
            origin_map = self._decode_pair(raw_pending.get("origin_map"), map_key=True)
            target = self._decode_pair(raw_pending.get("target"))
            direction, mode, elapsed = raw_pending.get("direction"), raw_pending.get("mode"), raw_pending.get("elapsed_frames")
            if (not isinstance(direction, int) or direction not in _DELTAS or mode not in ("walk", "run", "bike") or
                    not isinstance(elapsed, int) or elapsed < 0):
                raise ValueError("Invalid pending direction, mode or elapsed frames")
            pending = _PendingStep(origin, origin_map, direction, self.bridge.frame_count - elapsed,
                                   cast(MovementMode, mode), target)
        events: list[MovementEvent] = []
        raw_events = state.get("events", [])
        if not isinstance(raw_events, list):
            raise ValueError("Invalid movement events")
        for raw in raw_events:
            if not isinstance(raw, dict):
                raise ValueError("Invalid movement event")
            status, elapsed, direction = raw.get("status"), raw.get("frames"), raw.get("direction")
            if (status not in ("arrived", "blocked", "map_changed", "battle", "interrupted") or
                    not isinstance(elapsed, int) or elapsed < 0 or not isinstance(direction, int) or direction not in _DELTAS):
                raise ValueError("Invalid movement event fields")
            events.append(MovementEvent(cast(MovementStatus, status), self._decode_pair(raw.get("origin")),
                                        self._decode_pair(raw.get("destination")),
                                        self._decode_pair(raw.get("origin_map"), True),
                                        self._decode_pair(raw.get("destination_map"), True), elapsed, direction))
        self.blocked_edges, self.events, self._pending = edges, events, pending
        # No pending step means another controller may own an in-flight menu
        # pulse in the emulator snapshot; leave that restored key mask intact.
        if pending is not None:
            self.bridge.set_hardware_keys(pending.direction | (self.bridge.KEY_B if pending.mode == "run" else 0))

    @staticmethod
    def _decode_pair(raw: object, map_key: bool = False) -> tuple[int, int]:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2 or any(type(value) is not int for value in raw):
            raise ValueError("Invalid saved coordinate or map")
        pair = int(raw[0]), int(raw[1])
        if map_key and any(not 0 <= value <= 255 for value in pair):
            raise ValueError("Saved map ID outside uint8")
        return pair

    @staticmethod
    def _decode_edges(raw: object) -> set[BlockedEdge]:
        if not isinstance(raw, list):
            raise ValueError("Invalid saved obstacles")
        edges: set[BlockedEdge] = set()
        for row in raw:
            if not isinstance(row, (list, tuple)) or len(row) != 6 or any(type(value) is not int for value in row):
                raise ValueError("Invalid persisted blocked edge")
            if any(not 0 <= value <= 255 for value in row[:2]):
                raise ValueError("Saved obstacle map ID outside uint8")
            edges.add(cast(BlockedEdge, tuple(row)))
        return edges

    def save_obstacles(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state_dict(), indent=2), encoding="utf-8")

    def load_obstacles(self, path: str | Path) -> None:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.blocked_edges = self._decode_edges(raw["blocked_edges"])

