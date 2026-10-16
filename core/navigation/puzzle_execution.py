"""Feedback-driven execution of physical puzzle plans, without owning frames.

Only coordinates, actor motion, scripts and terrain changes acknowledge a
command. Finishing a local plan never certifies a campaign milestone. Dialogues
remain the field-menu controller's responsibility, including the HM yes/no UI.
"""
from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from core.bridge.ffi_mgba import GBABridgeProtocol
from core.navigation.source_data import SourceMapCatalog, load_map_catalog

XY = tuple[int, int]
MapKey = tuple[int, int]
PuzzleStatus = Literal["directive", "keys", "waiting", "replan", "complete", "unavailable", "blocked", "interrupted"]
_DIRECTIONS = {"right": 16, "left": 32, "up": 64, "down": 128}
_DELTAS = {16: (1, 0), 32: (-1, 0), 64: (0, -1), 128: (0, 1)}


class PuzzlePlannerProtocol(Protocol):
    catalog: SourceMapCatalog | None

    def plan_puzzle_actions(self, bridge: GBABridgeProtocol, goal: XY | None = None,
                             max_states: int = 25_000) -> list[dict[str, Any]] | None: ...

    def puzzle_actions(self, bridge: GBABridgeProtocol) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class PuzzleExecutionResult:
    status: PuzzleStatus
    directive: dict[str, Any] | None = None
    keys: int | None = None
    reason: str = ""
    required_move: int = 0


@dataclass(frozen=True)
class _Command:
    raw: dict[str, Any]
    map_key: MapKey
    destination: XY
    action: str
    direction: int
    expected_origin: XY | None
    expected_destination: XY | None
    pushed_boulder: XY | None
    required_move: int


@dataclass
class _Pending:
    origin: XY
    started_frame: int
    last_progress_frame: int
    previous_position: XY
    script_seen: bool
    terrain_before: str


def _json_canonical(value: object) -> object:
    """Tuples become lists recursively, exactly as a JSON round trip produces."""
    if isinstance(value, Mapping):
        return {str(key): _json_canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_canonical(item) for item in value]
    return value


def can_use_surf(bridge: GBABridgeProtocol, game: str) -> bool:
    """Badge 5 and a living Pokemon with Surf permit the normal shore prompt.

    Permission does not mean the player is already surfing; the caller must
    face water, press A and complete the real field script before walking on it.
    """
    catalog = load_map_catalog(game)
    badge = catalog.constants["FLAG_BADGE05_GET"]
    return bridge.get_event_flag(badge) and any(int(mon["hp"]) > 0 and 57 in mon["moves"]
                                              for mon in bridge.get_decrypted_player_party())


class FeedbackPuzzleExecutor:
    """Return a directive or one frame of keys; never advance the emulator.

    Call ``tick`` while ``active`` before generic hardware-not-ready handling.
    A waiting result with keys=None leaves dialogue input to FieldMenuController.
    Returned directives use the existing NPC-adjacency/cursor execution path.
    Steps and pushes with expected origins use cardinal keys directly so a
    boulder is never mistaken for an ordinary blocked destination.
    """

    def __init__(self, bridge: GBABridgeProtocol, planner: PuzzlePlannerProtocol,
                 *, max_replans: int = 4, step_timeout: int = 48,
                 push_timeout: int = 180, interaction_timeout: int = 1200) -> None:
        if max_replans < 1 or min(step_timeout, push_timeout, interaction_timeout) < 1:
            raise ValueError("Puzzle recovery bounds must be positive")
        self.bridge, self.planner = bridge, planner
        self.max_replans = max_replans
        self.step_timeout, self.push_timeout = step_timeout, push_timeout
        self.interaction_timeout = interaction_timeout
        self._map_key: MapKey | None = None
        self._goal: XY | None = None
        self._commands: list[_Command] = []
        self._index = 0
        self._pending: _Pending | None = None
        self._failures = 0
        self.replans = 0
        self.last_reason = "idle"
        self._fallback_attempts: set[tuple[int, int, int, int, str]] = set()

    @property
    def active(self) -> bool:
        return self._index < len(self._commands)

    def clear(self) -> None:
        """Forget local intent after recovery; do not alter restored RAM/keys."""
        self._commands, self._pending = [], None
        self._index, self._failures = 0, 0
        self._map_key, self._goal = None, None
        self._fallback_attempts.clear()
        self.last_reason = "cleared"

    @staticmethod
    def _pair(raw: object) -> XY:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2 or any(type(v) is not int for v in raw):
            raise ValueError("Puzzle coordinates require two integers")
        return int(raw[0]), int(raw[1])

    @classmethod
    def _command(cls, raw: Mapping[str, Any]) -> _Command:
        destination = raw.get("dest")
        if not isinstance(destination, (list, tuple)) or len(destination) != 3:
            raise ValueError("Puzzle command needs a map and destination")
        key = cls._pair(destination[:2])
        if any(not 0 <= value <= 255 for value in key):
            raise ValueError("Puzzle map outside uint8")
        target = cls._pair(destination[2])
        action = raw.get("action", "interact")
        if action not in ("step", "push", "interact", "directional", "trigger"):
            raise ValueError("Unsupported physical puzzle action")
        raw_direction = raw.get("direction", 0)
        direction = _DIRECTIONS.get(raw_direction, 0) if isinstance(raw_direction, str) else raw_direction
        if type(direction) is not int or direction not in (0, 16, 32, 64, 128):
            raise ValueError("Puzzle direction must be cardinal")
        origin = cls._pair(raw["expected_origin"]) if raw.get("expected_origin") is not None else None
        landing = cls._pair(raw["expected_destination"]) if raw.get("expected_destination") is not None else None
        boulder = cls._pair(raw["pushed_boulder"]) if raw.get("pushed_boulder") is not None else None
        required = raw.get("required_move", 0)
        if type(required) is not int or not 0 <= required <= 354:
            raise ValueError("Invalid required field move")
        if origin is not None and action in ("step", "push"):
            dx, dy = target[0] - origin[0], target[1] - origin[1]
            if (dx and dy) or abs(dx) + abs(dy) not in (1, 2):
                raise ValueError("Physical puzzle step must be cardinal")
            inferred = 16 if dx > 0 else 32 if dx < 0 else 128 if dy > 0 else 64
            if direction not in (0, inferred):
                raise ValueError("Puzzle direction disagrees with expected coordinates")
            direction = inferred
        if action == "push" and (origin is None or boulder != target):
            raise ValueError("A push requires its expected origin and boulder tile")
        # Store the JSON-canonical form so an in-memory plan and the same plan
        # restored from a checkpoint compare and serialize identically.
        return _Command(cast(dict[str, Any], _json_canonical(copy.deepcopy(dict(raw)))), key, target,
                        str(action), direction, origin, landing, boulder, required)

    def _terrain(self) -> str:
        tiles, collision, elevation = self.bridge.extract_live_map_matrices()
        digest = hashlib.sha256()
        for array in (tiles, collision, elevation, self.bridge.extract_behavior_map()):
            digest.update(array.tobytes())
        return digest.hexdigest()

    def _replan(self, reason: str) -> PuzzleExecutionResult:
        self._commands, self._pending, self._index = [], None, 0
        self._failures += 1
        self.replans += 1
        self.last_reason = reason
        status: PuzzleStatus = "blocked" if self._failures >= self.max_replans else "replan"
        return PuzzleExecutionResult(status, keys=0, reason=reason)

    def _finish_command(self, reason: str) -> PuzzleExecutionResult:
        command = self._commands[self._index]
        if command.raw.get("interaction_fallback"):
            self._fallback_attempts.add((*command.map_key, *command.destination, str(command.raw.get("script", ""))))
        self._index += 1
        self._pending = None
        self._failures = 0
        self.last_reason = reason
        return PuzzleExecutionResult("complete" if not self.active else "waiting", keys=0, reason=reason)

    def _boulder_arrived(self, command: _Command) -> bool:
        if command.pushed_boulder is None:
            return True
        dx, dy = _DELTAS[command.direction]
        landing = command.pushed_boulder[0] + dx, command.pushed_boulder[1] + dy
        graphics = self.planner.catalog.constants.get("OBJ_EVENT_GFX_PUSHABLE_BOULDER", -1) if self.planner.catalog else -1
        return any(int(actor.flags) & 1 and int(actor.graphicsId) == graphics and
                   (int(actor.currentCoords_x) - 7, int(actor.currentCoords_y) - 7) == landing
                   for actor in self.bridge.get_object_events())

    def _activation_observed(self, command: _Command) -> bool:
        catalog = self.planner.catalog
        if catalog is None or command.required_move != 70:
            return False
        flag = catalog.constants.get("FLAG_SYS_USE_STRENGTH")
        return flag is not None and self.bridge.get_event_flag(flag)

    def tick(self, goal: XY | None = None, *, allow_surf: bool = False,
             allow_interaction_fallback: bool = False, max_states: int = 25_000) -> PuzzleExecutionResult:
        """Observe acknowledgements and supply the next local action, if any."""
        coords = self.bridge.get_player_coords_and_map()
        key, position = (coords["map_group"], coords["map_num"]), (coords["x"], coords["y"])
        if self.bridge.is_in_battle()[0]:
            self._commands, self._pending, self._index = [], None, 0
            return PuzzleExecutionResult("interrupted", reason="battle interrupted physical puzzle")
        if self._map_key is not None and key != self._map_key:
            self.clear()
            self._map_key, self._goal = key, goal
            return PuzzleExecutionResult("interrupted", keys=0, reason="map changed; observe campaign before replanning")
        if self._map_key is None:
            self._map_key, self._goal = key, goal
        elif goal is not None and goal != self._goal:
            self.clear()
            self._map_key, self._goal = key, goal
        if self._failures >= self.max_replans:
            return PuzzleExecutionResult("blocked", keys=0, reason=self.last_reason)
        if not self.active:
            if not self.bridge.is_hardware_ready():
                return PuzzleExecutionResult("waiting", reason="field script owns input")
            raw_plan = self.planner.plan_puzzle_actions(self.bridge, self._goal, max_states=max_states)
            if raw_plan is None and allow_interaction_fallback:
                alternatives = [dict(action, interaction_fallback=True) for action in self.planner.puzzle_actions(self.bridge)
                                if (*key, *self._pair(action["dest"][2]), str(action.get("script", ""))) not in self._fallback_attempts]
                alternatives.sort(key=lambda action: abs(int(action["dest"][2][0]) - position[0]) +
                                                     abs(int(action["dest"][2][1]) - position[1]))
                raw_plan = alternatives[:1] or None
            if raw_plan is None:
                return PuzzleExecutionResult("unavailable", reason="No supported physical puzzle plan found")
            if not raw_plan:
                return PuzzleExecutionResult("complete", reason="No local puzzle commands remain; campaign evidence still required")
            self._commands = [self._command(command) for command in raw_plan]
            self._index = 0
        command = self._commands[self._index]
        if command.map_key != key:
            return self._replan("physical command belongs to another map")
        pending = self._pending
        stable = coords["tile_transition_state"] != 1
        ready = self.bridge.is_hardware_ready()
        if pending is not None:
            if position != pending.previous_position:
                pending.previous_position = position
                pending.last_progress_frame = self.bridge.frame_count
            if command.expected_destination is not None:
                arrivals = {command.expected_destination}
                if command.action == "interact":
                    arrivals.add((command.expected_destination[0], command.expected_destination[1] + 1))
                if stable and ready and position in arrivals and (position != pending.origin or pending.script_seen):
                    return self._finish_command("observed same-map portal landing")
            elif command.action in ("step", "push") and command.expected_origin is not None:
                if stable and position == command.destination:
                    if not self._boulder_arrived(command):
                        return self._replan("player arrived but expected boulder landing was not observed")
                    return self._finish_command("observed physical puzzle step")
            if not ready and stable:
                pending.script_seen = True
                return PuzzleExecutionResult("waiting", reason="field script owns input", required_move=command.required_move)
            if command.action not in ("step", "push") and command.expected_destination is None and ready:
                if (pending.script_seen or self._activation_observed(command) or
                        self._terrain() != pending.terrain_before):
                    # The script finished; its actual effects must determine a
                    # fresh plan instead of assuming a button opened a door.
                    self._finish_command("observed interaction feedback; replan from RAM")
                    self._commands, self._pending, self._index = [], None, 0
                    return PuzzleExecutionResult("replan", keys=0, reason=self.last_reason)
            if command.expected_origin is not None and stable and position not in (command.expected_origin, command.destination):
                return self._replan("unexpected physical puzzle position")
            timeout = self.push_timeout if command.action == "push" else self.step_timeout if command.expected_origin is not None else self.interaction_timeout
            if self.bridge.frame_count - pending.last_progress_frame >= timeout and stable:
                return self._replan("physical puzzle command produced no expected feedback")
        else:
            if command.expected_origin is not None and position != command.expected_origin:
                return self._replan("expected puzzle origin no longer matches RAM")
            if not ready:
                return PuzzleExecutionResult("waiting", reason="field script owns input")
            self._pending = _Pending(position, self.bridge.frame_count, self.bridge.frame_count, position,
                                     False, self._terrain())
        if command.expected_origin is not None and command.action in ("step", "push"):
            return PuzzleExecutionResult("keys", keys=command.direction, reason="execute verified physical step",
                                         required_move=command.required_move)
        if position == command.destination and command.action == "directional" and command.direction:
            return PuzzleExecutionResult("keys", keys=command.direction, reason="activate directional portal")
        directive = dict(command.raw)
        directive["allow_surf"] = allow_surf
        return PuzzleExecutionResult("directive", directive=directive, reason="approach and activate source puzzle event",
                                     required_move=command.required_move)

    def state_dict(self) -> dict[str, object]:
        pending = self._pending
        return {"version": 1, "game": self.bridge.memory_map.game,
                "map": self._map_key, "goal": self._goal,
                "commands": [copy.deepcopy(command.raw) for command in self._commands], "index": self._index,
                "failures": self._failures, "replans": self.replans, "last_reason": self.last_reason,
                "fallback_attempts": [list(attempt) for attempt in sorted(self._fallback_attempts)],
                "pending": None if pending is None else {"origin": pending.origin,
                    "elapsed": self.bridge.frame_count - pending.started_frame,
                    "idle_elapsed": self.bridge.frame_count - pending.last_progress_frame,
                    "previous_position": pending.previous_position, "script_seen": pending.script_seen,
                    "terrain_before": pending.terrain_before}}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Validate atomically and rebase elapsed timers to the restored frame."""
        if state.get("version") != 1 or state.get("game") != self.bridge.memory_map.game:
            raise ValueError("Incompatible physical puzzle checkpoint")
        key = self._pair(state["map"]) if state.get("map") is not None else None
        goal = self._pair(state["goal"]) if state.get("goal") is not None else None
        raw_commands = state.get("commands")
        if not isinstance(raw_commands, list) or len(raw_commands) > 10_000:
            raise ValueError("Malformed physical puzzle plan")
        commands: list[_Command] = []
        for raw in raw_commands:
            if not isinstance(raw, dict):
                raise ValueError("Malformed physical puzzle command")
            commands.append(self._command(raw))
        index, failures, replans = state.get("index"), state.get("failures"), state.get("replans")
        if (type(index) is not int or not 0 <= index <= len(commands) or
                type(failures) is not int or failures < 0 or type(replans) is not int or replans < 0):
            raise ValueError("Invalid physical puzzle counters")
        reason = state.get("last_reason")
        if not isinstance(reason, str) or len(reason) > 256:
            raise ValueError("Invalid physical puzzle result")
        pending = None
        raw_pending = state.get("pending")
        if raw_pending is not None:
            if not isinstance(raw_pending, dict) or index >= len(commands):
                raise ValueError("Malformed pending physical puzzle action")
            values = cast(dict[str, object], raw_pending)
            elapsed, idle = values.get("elapsed"), values.get("idle_elapsed")
            script_seen, terrain = values.get("script_seen"), values.get("terrain_before")
            if (type(elapsed) is not int or elapsed < 0 or type(idle) is not int or not 0 <= idle <= elapsed or
                    type(script_seen) is not bool or not isinstance(terrain, str) or len(terrain) != 64):
                raise ValueError("Invalid pending physical puzzle timer")
            pending = _Pending(self._pair(values["origin"]), self.bridge.frame_count - elapsed,
                               self.bridge.frame_count - idle, self._pair(values["previous_position"]), script_seen, terrain)
        attempts: set[tuple[int, int, int, int, str]] = set()
        raw_attempts = state.get("fallback_attempts", [])
        if not isinstance(raw_attempts, list):
            raise ValueError("Malformed puzzle interaction history")
        for attempt in raw_attempts:
            if (not isinstance(attempt, (list, tuple)) or len(attempt) != 5 or
                    any(type(v) is not int for v in attempt[:4]) or not isinstance(attempt[4], str)):
                raise ValueError("Malformed puzzle interaction attempt")
            attempts.add((int(attempt[0]), int(attempt[1]), int(attempt[2]), int(attempt[3]), str(attempt[4])))
        self._map_key, self._goal, self._commands, self._index = key, goal, commands, index
        self._failures, self.replans, self.last_reason = failures, replans, reason
        self._pending, self._fallback_attempts = pending, attempts
