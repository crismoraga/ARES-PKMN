"""Observed native start menus, script choices and HM teaching through keys.

No routine writes player state. A taught move is certified only by decrypting
the actual party after the native learning script has finished.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from collections.abc import Mapping
import struct
from typing import Literal, cast

from core.battle.menu_cursor_controller import BattleMenuCursorController
from core.battle.move_learning import MoveLearningChoice, choose_move_to_forget, rom_hm_moves, rom_move_summary
from core.bridge.ffi_mgba import GBABridgeProtocol
from core.bridge.gba_structs import ObjectEventStruct
from core.game_data import Game, RomCatalog, load_symbols
from core.navigation.source_data import load_map_catalog
from core.navigation.constrained_astar import GbaConstrainedAStar

FieldMoveName = Literal["cut", "surf", "strength", "rock_smash", "dive", "waterfall"]
EVOLUTION_TASKS = frozenset({"Task_EvolutionScene", "Task_TradeEvolutionScene"})
LEARNING_TASKS = frozenset({"Task_InputHandler_SelectOrForgetMove", "Task_HandleReplaceMoveInput",
                            "Task_HandleInputCantForgetHMsMoves"})
# Option rows -> desired value: text speed FAST, battle scene OFF, battle style SET.
# Fewer animation and prompt frames per battle; nothing about gameplay changes.
OPTION_TARGETS = {0: 2, 1: 1, 2: 1}
_OPTIONS_TIMEOUT = 4000


@dataclass(frozen=True)
class FieldMoveCapability:
    """Observed own-party HM capability, separate from possession of its item."""
    name: FieldMoveName
    move_id: int
    hm_item: int
    carrier_index: int | None
    badge_flag: int
    badge_observed: bool
    hm_in_inventory: bool
    can_teach: bool
    supported: bool = True

    @property
    def usable(self) -> bool:
        """The native script's badge and non-Egg move-carrier prerequisites hold."""
        return self.supported and self.badge_observed and self.carrier_index is not None


@dataclass(frozen=True)
class TeachMoveRequest:
    """An explicit TM/HM item, compatible party member, and replacement slot."""
    item: int
    move: int
    party_index: int
    forget_slot: int


class FieldMenuController:
    """Return one frame's keys for native field UI; caller owns frame advancement."""

    def __init__(self, bridge: GBABridgeProtocol, game: Game,
                 catalog: RomCatalog | None = None) -> None:
        self.bridge, self.game = bridge, game
        self.catalog = catalog or RomCatalog(game)
        self.symbols = load_symbols(game)
        self.functions = {symbol.address & ~1: name for name, symbol in self.symbols.items()
                          if 0x08000000 <= symbol.address < 0x09000000 and symbol.size > 0}
        self._bag = BattleMenuCursorController(bridge)
        self.pending: TeachMoveRequest | None = None
        self._started = 0
        self._closing = False
        self.menu_choice = 0
        self.last_result = "idle"
        self._evolution_party_index: int | None = None
        self._move_summary = rom_move_summary(self.catalog)
        self._hm_moves = rom_hm_moves(self.catalog)
        self._hm_cache: tuple[int, frozenset[str]] | None = None
        self.last_learning_choice: MoveLearningChoice | None = None
        self.scene = "none"
        self.options_pending = False
        self._options_started = 0
        self._options_retry_after = 0
        self._script_labels: list[tuple[int, str]] = []
        self._multichoice_label: str | None = None
        self._multichoice_memory: dict[str, tuple[int, tuple[int, int, int, int], int]] = {}
        constants = load_map_catalog(game).constants
        self._patrol_movement_types = frozenset(value for name, value in constants.items()
                                              if name.startswith("MOVEMENT_TYPE_WALK_SEQUENCE_"))
        self._patrol_sequences = {value: tuple(name.removeprefix("MOVEMENT_TYPE_WALK_SEQUENCE_").split("_"))
                                  for name, value in constants.items()
                                  if name.startswith("MOVEMENT_TYPE_WALK_SEQUENCE_")}
        self._interaction_wait: tuple[int, int, int, int, int, int] | None = None
        self._interaction_retry_after = 0

    @property
    def busy(self) -> bool:
        """Whether HM navigation, option setup or returning from a native screen is pending."""
        return self.pending is not None or self._closing or self.options_pending

    def options_values(self) -> tuple[int, int, int]:
        """Saved (text speed, battle scene off, battle style) from SaveBlock2+0x14."""
        packed = self.bridge.read_u16(self.bridge.get_save_block2_dynamic_ptr() + 0x14)
        return packed & 7, (packed >> 10) & 1, (packed >> 9) & 1

    def options_optimal(self) -> bool:
        return self.options_values() == tuple(OPTION_TARGETS[row] for row in (0, 1, 2))

    def options_due(self) -> bool:
        """Whether a one-time options setup should start now (never while busy)."""
        return (not self.busy and self.bridge.frame_count >= self._options_retry_after
                and not self.options_optimal())

    def request_options(self) -> None:
        if self.busy:
            raise RuntimeError("A previous field operation is still pending")
        self.options_pending, self._options_started = True, self.bridge.frame_count
        self.last_result = "options_pending"

    def _task_data(self, name: str) -> int | None:
        function = self._address(name) & ~1
        for slot in range(16):
            address = self.bridge.memory_map.tasks + slot * 40
            if self.bridge.read_u8(address + 4) and self.bridge.read_u32(address) & ~1 == function:
                return address + 8
        return None

    def _pokenav_key(self) -> int | None:
        """Emerald's forced Match Call tutorial, where B is deliberately disabled.

        READY: open Match Call, pick an entry and CALL. EXIT: leave the list
        and switch the PokéNav off. Normal PokéNav use returns None (B closes it).
        """
        if self.game != "emerald" or "gPokenavResources" not in self.symbols:
            return None
        resources = self.bridge.read_u32(self._address("gPokenavResources"))
        if not 0x02000000 <= resources <= 0x0203FF00:
            return None
        mode = self.bridge.read_u16(resources + 8)
        if mode not in (1, 2):  # POKENAV_MODE_FORCE_CALL_READY / _EXIT
            return None
        ready = mode == 1
        match_call = self.bridge.read_u32(resources + 16 + 4 * 5)  # POKENAV_SUBSTRUCT_MATCH_CALL_MAIN
        if 0x02000000 <= match_call <= 0x0203FF00:
            handler = self.functions.get(self.bridge.read_u32(match_call + 24) & ~1, "")
            if handler == "CB2_HandleCallExitInput":
                return self.bridge.KEY_A
            if handler == "CB2_HandleMatchCallOptionsInput":
                if not ready:
                    return self.bridge.KEY_B
                return self.bridge.KEY_A if self.bridge.read_u16(match_call) == 0 else self.bridge.KEY_UP  # CALL is option 0
            if handler == "CB2_HandleMatchCallInput":
                return self.bridge.KEY_A if ready else self.bridge.KEY_B
        menu = self.bridge.read_u32(resources + 16 + 4 * 1)  # POKENAV_SUBSTRUCT_MAIN_MENU_HANDLER
        if 0x02000000 <= menu <= 0x0203FF00:
            wanted = 2 if ready else 4  # POKENAV_MENUITEM_MATCH_CALL / _SWITCH_OFF
            return self.bridge.KEY_A if self.bridge.read_u16(menu + 4) == wanted else self.bridge.KEY_DOWN
        return self.bridge.KEY_A if self._bag._text_waiting() else 0

    def _script_label(self) -> str:
        """Nearest ROM label of the running field script, for per-question memory."""
        pointer = self.bridge.read_u32(self.bridge.memory_map.script + 8)
        if not self._script_labels:
            self._script_labels = sorted((symbol.address, name) for name, symbol in self.symbols.items()
                                         if 0x08000000 <= symbol.address < 0x0A000000)
        index = bisect.bisect_right(self._script_labels, (pointer, "￿")) - 1
        return self._script_labels[index][1] if index >= 0 else f"script_{pointer:08x}"

    def _multichoice_answer(self) -> int:
        """Directive choice first; if the same question loops back, try the next option.

        Scripts such as the PokéNav tutorial repeat a menu until the one correct
        answer is chosen. Re-asking the identical question at the same tile
        within a short window means the previous answer was wrong.
        """
        label = self._script_label()
        options = max(1, self.bridge.read_u8(self._address("sMenu") + 4) + 1)
        coords = self.bridge.get_player_coords_and_map()
        place = (coords["map_group"], coords["map_num"], coords["x"], coords["y"])
        frame = self.bridge.frame_count
        if self._multichoice_label != label:
            self._multichoice_label = label
            previous = self._multichoice_memory.get(label)
            if previous is not None and previous[1] == place and frame - previous[2] < 1800:
                choice = (previous[0] + 1) % options
                self.last_result = f"multichoice_retry:{label}:{choice}"
            else:
                choice = min(self.menu_choice, options - 1)
            self._multichoice_memory[label] = (choice, place, frame)
            if len(self._multichoice_memory) > 256:
                self._multichoice_memory.clear()
        choice, _, _ = self._multichoice_memory[label]
        self._multichoice_memory[label] = (choice, place, frame)
        return choice

    def _options_navigation(self, row: int, values: Mapping[int, int]) -> int:
        pending = [option for option in (0, 1, 2) if values[option] != OPTION_TARGETS[option]]
        if not pending:
            return self.bridge.KEY_B  # B saves and closes in both games
        if row != pending[0]:
            return self.bridge.KEY_DOWN if row < pending[0] else self.bridge.KEY_UP
        return self.bridge.KEY_RIGHT  # every target row cycles forward with RIGHT

    def _options_key(self, tasks: set[str], field: bool) -> int:
        """Start menu -> OPTION -> set rows observed from the native option screen."""
        if self.game == "emerald" and "Task_OptionMenuProcessInput" in tasks:
            data = self._task_data("Task_OptionMenuProcessInput")
            if data is not None:
                return self._options_navigation(self.bridge.read_u16(data),
                    {option: self.bridge.read_u16(data + 2 + 2 * option) for option in (0, 1, 2)})
        if self.game == "firered" and "Task_OptionMenu" in tasks:
            pointer = self.bridge.read_u32(self._address("sOptionMenuPtr"))
            if not 0x02000000 <= pointer <= 0x0203FFE0 or self.bridge.read_u8(pointer + 0x10) != 2:
                return 0  # fading in or out: no input is accepted
            return self._options_navigation(self.bridge.read_u16(pointer + 0x0E),
                {option: self.bridge.read_u16(pointer + 2 * option) for option in (0, 1, 2)})
        if any("StartMenu" in task for task in tasks):
            return self._start_key(6 if self.game == "emerald" else 5)  # MENU_ACTION_OPTION / STARTMENU_OPTION
        if field and self.bridge.is_hardware_ready():
            return self.bridge.KEY_START
        return 0

    def cancel(self, reason: str = "cancelled") -> None:
        """Abandon a stalled request and close its UI through ordinary B keys."""
        self.pending = None
        self._closing = True
        self.last_result = reason

    def state_dict(self) -> dict[str, object]:
        """Checkpoint the UI intent; actual cursors remain in the emulator state."""
        request: dict[str, int] | None = None
        if self.pending is not None:
            request = {"item": self.pending.item, "move": self.pending.move,
                       "party_index": self.pending.party_index, "forget_slot": self.pending.forget_slot}
        return {"version": 1, "game": self.game, "pending": request, "started": self._started,
                "closing": self._closing, "menu_choice": self.menu_choice, "last_result": self.last_result,
                "evolution_party_index": self._evolution_party_index,
                "options_pending": self.options_pending, "options_started": self._options_started,
                "options_retry_after": self._options_retry_after,
                "interaction_wait": list(self._interaction_wait) if self._interaction_wait is not None else None,
                "interaction_retry_after": self._interaction_retry_after}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore validated intent without restarting a partially completed menu."""
        if state.get("version") != 1 or state.get("game") != self.game:
            raise ValueError("Incompatible field controller checkpoint")
        raw = state.get("pending")
        request: TeachMoveRequest | None = None
        if raw is not None:
            if not isinstance(raw, Mapping):
                raise ValueError("Malformed pending field operation")
            values = cast(Mapping[str, object], raw)
            request = TeachMoveRequest(self._checkpoint_int(values, "item", 289, 346),
                self._checkpoint_int(values, "move", 1, 354),
                self._checkpoint_int(values, "party_index", 0, 5),
                self._checkpoint_int(values, "forget_slot", 0, 3))
            if self.catalog.tmhm_move(request.item) != request.move:
                raise ValueError("Checkpoint TM/HM move mismatch")
        started = self._checkpoint_int(state, "started", 0, 2**63 - 1)
        choice = self._checkpoint_int(state, "menu_choice", 0, 255)
        closing, result = state.get("closing"), state.get("last_result")
        if type(closing) is not bool or not isinstance(result, str) or not result or len(result) > 128:
            raise ValueError("Invalid field controller status")
        evolution_party = state.get("evolution_party_index")
        if evolution_party is not None and (type(evolution_party) is not int or not 0 <= evolution_party < 6):
            raise ValueError("Invalid evolution party index")
        options_pending = state.get("options_pending", False)
        options_started = self._checkpoint_int({"value": state.get("options_started", 0)}, "value", 0, 2**63 - 1)
        options_retry = self._checkpoint_int({"value": state.get("options_retry_after", 0)}, "value", 0, 2**63 - 1)
        if type(options_pending) is not bool:
            raise ValueError("Invalid field controller status")
        interaction_wait = state.get("interaction_wait")
        if interaction_wait is not None:
            if (not isinstance(interaction_wait, list) or len(interaction_wait) != 6 or
                    any(type(value) is not int for value in interaction_wait)):
                raise ValueError("Invalid native NPC interception checkpoint")
            values_int = cast(list[int], interaction_wait)
            if (any(not 0 <= value <= 255 for value in values_int[:2]) or not 1 <= values_int[2] < 255 or
                    any(not -32768 <= value <= 32767 for value in values_int[3:5]) or
                    not 0 <= values_int[5] <= 2**63 - 1):
                raise ValueError("Invalid native NPC interception checkpoint")
        retry_after = self._checkpoint_int({"value": state.get("interaction_retry_after", 0)}, "value", 0, 2**63 - 1)
        self.pending, self._started = request, started
        self._closing, self.menu_choice, self.last_result = closing, choice, result
        self._evolution_party_index = evolution_party
        self.options_pending, self._options_started, self._options_retry_after = options_pending, options_started, options_retry
        self._interaction_wait = cast(tuple[int, int, int, int, int, int] | None,
                                      tuple(interaction_wait) if interaction_wait is not None else None)
        self._interaction_retry_after = retry_after

    @staticmethod
    def _checkpoint_int(state: Mapping[str, object], name: str, low: int, high: int) -> int:
        value = state.get(name)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Invalid field controller integer: {name}")
        return value

    def _address(self, name: str) -> int:
        return self.symbols[name].address

    def active_tasks(self) -> set[str]:
        """Resolve task callbacks only for slots marked active by the game."""
        tasks: set[str] = set()
        for slot in range(16):
            address = self.bridge.memory_map.tasks + slot * 40
            if self.bridge.read_u8(address + 4):
                pointer = self.bridge.read_u32(address) & ~1
                tasks.add(self.functions.get(pointer, f"task_{pointer:x}"))
        return tasks

    def interaction_key(self, local_id: int) -> int | None:
        """Intercept the requested live NPC when adjacent, before another step.

        Moving NPCs can pass the player between route replans. An already queued
        step must not make the policy chase a new adjacent tile and miss every
        chance to talk. Only this actor in this map can receive the A edge;
        another actor in front (for example Peeko) never substitutes for Briney.
        """
        if not 0 < local_id < 255 or self.busy or self.bridge.is_in_battle()[0] or not self.bridge.is_hardware_ready():
            return None
        coords = self.bridge.get_player_coords_and_map()
        player_id = self.bridge.read_u8(self.bridge.memory_map.avatar + 5)
        if player_id >= 16:
            return None
        objects = self.bridge.get_object_events()
        target = next((actor for index, actor in enumerate(objects)
                       if index != player_id and int(actor.flags) & 1 and int(actor.localId) == local_id
                       and int(actor.mapGroup) == coords["map_group"] and int(actor.mapNum) == coords["map_num"]), None)
        if target is None:
            return None
        elevation = int(target.elevations) & 15
        if elevation and coords["z"] and elevation != coords["z"]:
            return None
        dx = int(target.currentCoords_x) - 7 - coords["x"]
        dy = int(target.currentCoords_y) - 7 - coords["y"]
        distance = abs(dx) + abs(dy)
        across_counter = False
        if distance == 2 and (dx == 0 or dy == 0):
            middle = coords["x"] + dx // 2, coords["y"] + dy // 2
            behaviors = self.bridge.extract_behavior_map()
            counter = load_map_catalog(self.game).constants.get("MB_COUNTER", 0x80)
            across_counter = (0 <= middle[1] < behaviors.shape[0] and 0 <= middle[0] < behaviors.shape[1]
                              and int(behaviors[middle[1], middle[0]]) == counter)
        if distance != 1 and not across_counter:
            # Equal-speed walking loops can keep the player perpetually several
            # tiles behind a patrol. Let the named actor come to this tile; never
            # chase Peeko or wait indefinitely for a stationary/wrong NPC.
            moving = (int(target.movementType) in self._patrol_movement_types and
                      (int(target.currentCoords_x), int(target.currentCoords_y)) !=
                      (int(target.previousCoords_x), int(target.previousCoords_y)))
            frame = self.bridge.frame_count
            context = (coords["map_group"], coords["map_num"], local_id, coords["x"], coords["y"])
            already_waiting = self._interaction_wait is not None and self._interaction_wait[:5] == context
            if frame < self._interaction_retry_after or (not already_waiting and (not moving or distance > 4)):
                return None
            if not already_waiting:
                step = self._patrol_intercept_step(target, objects, player_id, coords)
                if step is None:
                    return None
                origin = coords["x"], coords["y"]
                if step != origin:
                    sx, sy = step[0] - origin[0], step[1] - origin[1]
                    return self.bridge.KEY_RIGHT if sx > 0 else (self.bridge.KEY_LEFT if sx < 0 else
                           (self.bridge.KEY_DOWN if sy > 0 else self.bridge.KEY_UP))
                self._interaction_wait = (*context, frame)
            assert self._interaction_wait is not None
            if frame - self._interaction_wait[5] < 240:
                return 0
            self._interaction_wait = None
            self._interaction_retry_after = frame + 120
            return None
        direction = 4 if dx > 0 else (3 if dx < 0 else (1 if dy > 0 else 2))
        facing = int(objects[player_id].facingDirection) & 15
        if facing != direction:
            return {1: self.bridge.KEY_DOWN, 2: self.bridge.KEY_UP,
                    3: self.bridge.KEY_LEFT, 4: self.bridge.KEY_RIGHT}[direction]
        return self.bridge.KEY_A if self.bridge.frame_count % 4 < 2 else 0

    def patrol_interaction_key(self, local_id: int) -> int | None:
        """Intercept moving patrols early without reopening ordinary NPC dialogs.

        Stationary story/healing interactions belong after directive replanning:
        a nurse remains beside the player after healing and must not preempt the
        planner with another conversation on every field-ready frame.
        """
        if not 0 < local_id < 255:
            return None
        coords = self.bridge.get_player_coords_and_map()
        player_id = self.bridge.read_u8(self.bridge.memory_map.avatar + 5)
        target = next((actor for index, actor in enumerate(self.bridge.get_object_events())
                       if index != player_id and int(actor.flags) & 1 and int(actor.localId) == local_id
                       and int(actor.mapGroup) == coords["map_group"] and int(actor.mapNum) == coords["map_num"]), None)
        if target is None or int(target.movementType) not in self._patrol_movement_types:
            return None
        return self.interaction_key(local_id)

    def _patrol_intercept_step(self, target: ObjectEventStruct, objects: list[ObjectEventStruct],
                               player_id: int, coords: Mapping[str, int]) -> tuple[int, int] | None:
        """Wait beside, rather than on, the observed rectangular walking circuit.

        Standing on the circuit can block one NPC behind another indefinitely.
        The movement sequence and its ranges describe the whole circuit in RAM;
        live collision/elevation and all actors constrain the waiting location.
        """
        sequence = self._patrol_sequences.get(int(target.movementType), ())
        x, y = int(target.initialCoords_x) - 7, int(target.initialCoords_y) - 7
        rx, ry = int(target.rangeX_Y) & 15, int(target.rangeX_Y) >> 4
        if not sequence or not rx or not ry:
            return None
        points = [(x, y)]
        vectors = {"LEFT": (-1, 0), "RIGHT": (1, 0), "UP": (0, -1), "DOWN": (0, 1)}
        for direction in sequence:
            if direction not in vectors:
                return None
            dx, dy = vectors[direction]
            x, y = x + dx * rx, y + dy * ry
            points.append((x, y))
        min_x, max_x = min(point[0] for point in points), max(point[0] for point in points)
        min_y, max_y = min(point[1] for point in points), max(point[1] for point in points)
        waiting = {(x, y) for x in range(min_x, max_x + 1) for y in (min_y - 1, max_y + 1)}
        waiting.update((x, y) for y in range(min_y, max_y + 1) for x in (min_x - 1, max_x + 1))
        _, collisions, elevations = self.bridge.extract_live_map_matrices()
        astar = GbaConstrainedAStar(collisions, elevations, self.bridge.extract_behavior_map(), self.game)
        astar.register_object_events(objects, player_id)
        origin = coords["x"], coords["y"]
        candidates: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []
        for tile in waiting:
            path = astar.find_optimal_path((*origin, coords["z"]), tile)
            if path is not None:
                candidates.append((path, tile))
        if not candidates:
            return None
        path, _ = min(candidates, key=lambda candidate: (len(candidate[0]), candidate[1]))
        return path[0] if path else origin

    def capabilities(self) -> dict[FieldMoveName, FieldMoveCapability]:
        """Read real badges, stock and own moves; never infer progress from intent."""
        constants = load_map_catalog(self.game).constants
        party, inventory = self.bridge.get_decrypted_player_party(), self.bridge.get_inventory()
        definitions: tuple[tuple[FieldMoveName, int, int, int], ...] = (
            ("cut", 15, 339, 1 if self.game == "emerald" else 2),
            ("surf", 57, 341, 5), ("strength", 70, 342, 4),
            ("rock_smash", 249, 344, 3 if self.game == "emerald" else 6),
            ("dive", 291, 346, 7), ("waterfall", 127, 345, 8 if self.game == "emerald" else 7))
        result: dict[FieldMoveName, FieldMoveCapability] = {}
        for name, move, item, badge in definitions:
            carrier = next((index for index, mon in enumerate(party)
                            if not bool(mon.get("is_egg", False)) and move in mon["moves"]), None)
            flag = constants[f"FLAG_BADGE{badge:02d}_GET"]
            stocked = inventory.get(item, 0) > 0
            teachable = stocked and any(not bool(mon.get("is_egg", False)) and
                self.catalog.can_learn_tmhm(int(mon["species"]), item) for mon in party)
            result[name] = FieldMoveCapability(name, move, item, carrier, flag,
                self.bridge.get_event_flag(flag), stocked, teachable, name != "dive" or self.game == "emerald")
        return result

    def hm_usable(self) -> frozenset[str]:
        """Field moves whose badge and own-party carrier prerequisites hold now.

        Planners use this to treat Cut trees, Rock Smash rocks and water as
        traversable only when the native script would actually accept them.
        """
        frame = self.bridge.frame_count
        if self._hm_cache is not None and self._hm_cache[0] == frame:
            return self._hm_cache[1]
        usable = frozenset(name for name, capability in self.capabilities().items() if capability.usable)
        self._hm_cache = (frame, usable)
        return usable

    def obstacle_move(self, actor: ObjectEventStruct) -> FieldMoveName | None:
        """Classify a visible tree/rock/boulder by native graphic or map script."""
        source = load_map_catalog(self.game)
        candidates: tuple[tuple[FieldMoveName, str], ...] = (
            ("cut", "OBJ_EVENT_GFX_CUTTABLE_TREE" if self.game == "emerald" else "OBJ_EVENT_GFX_CUT_TREE"),
            ("rock_smash", "OBJ_EVENT_GFX_BREAKABLE_ROCK" if self.game == "emerald" else "OBJ_EVENT_GFX_ROCK_SMASH_ROCK"),
            ("strength", "OBJ_EVENT_GFX_PUSHABLE_BOULDER"))
        for name, graphic in candidates:
            if int(actor.graphicsId) == source.constants[graphic]:
                return name
        raw = source.maps.get((int(actor.mapGroup), int(actor.mapNum)))
        local_id = int(actor.localId)
        if raw is not None and 0 < local_id <= len(raw["object_events"]):
            script = str(raw["object_events"][local_id - 1].get("script", ""))
            if "CutTree" in script:
                return "cut"
            if "RockSmash" in script:
                return "rock_smash"
            if "StrengthBoulder" in script:
                return "strength"
        return None

    def choose_transition_key(self, origin: tuple[int, int], target: tuple[int, int], behavior: int,
                              actor: ObjectEventStruct | None = None, *,
                              intent: FieldMoveName | None = None) -> int | None:
        """Face and activate a legal HM edge using keys; None permits movement.

        `behavior` is the planned target tile's observed behavior. Dive is an
        explicit same-tile action: A to descend, B to emerge, as Emerald's field
        input handler specifies. Normal navigation never dives automatically.
        Native scripts retain responsibility for location, animation and success.
        """
        if self.bridge.is_in_battle()[0] or self.busy:
            return 0 if self.busy else None
        dx, dy = target[0] - origin[0], target[1] - origin[1]
        if abs(dx) + abs(dy) not in (0, 1):
            return None
        constants = load_map_catalog(self.game).constants
        surfing = bool(self.bridge.read_u8(self.bridge.memory_map.avatar) & 8)
        underwater = self.bridge.read_u8(self.bridge.memory_map.map_header + 0x17) == constants["MAP_TYPE_UNDERWATER"]
        requested = intent
        if requested is None and actor is not None:
            requested = self.obstacle_move(actor)
        waterfall = constants["MB_WATERFALL"]
        if requested is None and behavior == waterfall and surfing and dy == -1:
            requested = "waterfall"
        if requested is None and not surfing and not underwater:
            name = "sTileBitAttributes" if self.game == "emerald" else "sBehaviorSurfable"
            table = self.catalog.symbols[name]
            if 0 <= behavior < table.size:
                water = self.catalog.read(table.address + behavior, 1)[0] & (2 if self.game == "emerald" else 1)
                fast = self.game == "firered" and behavior == constants["MB_FAST_WATER"]
                if water and behavior != waterfall and not fast:
                    requested = "surf"
        if requested is None:
            return None
        capability = self.capabilities()[requested]
        if not capability.usable:
            if capability.supported and capability.badge_observed and capability.can_teach:
                return 0 if self.request_required_move(capability.move_id) else None
            return None
        if requested == "strength" and self.bridge.get_event_flag(constants["FLAG_SYS_USE_STRENGTH"]):
            return None
        if requested == "dive":
            if self.game != "emerald":
                return None
            behaviors = self.bridge.extract_behavior_map()
            x, y = origin
            if not 0 <= y < behaviors.shape[0] or not 0 <= x < behaviors.shape[1]:
                return None
            current = int(behaviors[y, x])
            forbidden = {constants["MB_NO_SURFACING"], constants["MB_SEAWEED_NO_SURFACING"]}
            deep = {constants["MB_INTERIOR_DEEP_WATER"], constants["MB_DEEP_WATER"], constants["MB_SOOTOPOLIS_DEEP_WATER"]}
            if underwater and current not in forbidden:
                return self.bridge.KEY_B if self.bridge.frame_count % 8 < 2 else 0
            if surfing and current in deep:
                return self.bridge.KEY_A if self.bridge.frame_count % 8 < 2 else 0
            return None
        if abs(dx) + abs(dy) != 1:
            return None
        player_id = self.bridge.read_u8(self.bridge.memory_map.avatar + 5)
        if player_id >= 16:
            return 0
        facing = int(self.bridge.get_object_events()[player_id].facingDirection) & 15
        direction = 4 if dx > 0 else (3 if dx < 0 else (1 if dy > 0 else 2))
        if facing != direction:
            return {1: self.bridge.KEY_DOWN, 2: self.bridge.KEY_UP,
                    3: self.bridge.KEY_LEFT, 4: self.bridge.KEY_RIGHT}[direction]
        return self.bridge.KEY_A if self.bridge.frame_count % 8 < 2 else 0

    def request_teach_move(self, item_id: int, party_index: int,
                           forget_slot: int = 0) -> None:
        """Start teaching only after checking inventory, native compatibility and HMs."""
        if self.busy:
            raise RuntimeError("A previous field operation is still pending")
        party = self.bridge.get_decrypted_player_party()
        if not 0 <= party_index < len(party) or not 0 <= forget_slot < 4:
            raise ValueError("Invalid party/replacement slot")
        move = self.catalog.tmhm_move(item_id)
        mon = party[party_index]
        if bool(mon.get("is_egg", False)):
            raise ValueError("An Egg cannot learn a native TM/HM")
        if move in mon["moves"]:
            self.last_result = "already_known"
            return
        if not self.catalog.can_learn_tmhm(int(mon["species"]), item_id):
            raise ValueError("Selected party member cannot learn this native TM/HM")
        if self.bridge.get_inventory().get(item_id, 0) < 1:
            raise ValueError("TM/HM not present in the native inventory")
        existing = int(mon["moves"][forget_slot])
        hm_moves = {self.catalog.tmhm_move(item) for item in range(339, 347)}
        if existing in hm_moves and all(mon["moves"]):
            raise ValueError("An HM cannot be forgotten by a native TM/HM learning screen")
        self.pending = TeachMoveRequest(item_id, move, party_index, forget_slot)
        self._started = self.bridge.frame_count
        self.last_result = "pending"

    def request_required_move(self, move_id: int) -> bool:
        """Choose the first compatible party carrier and lowest-power replaceable move."""
        party = self.bridge.get_decrypted_player_party()
        if any(move_id in mon["moves"] and not bool(mon.get("is_egg", False)) for mon in party):
            return True
        inventory = self.bridge.get_inventory()
        items = [item for item in range(289, 347)
                 if inventory.get(item, 0) > 0 and self.catalog.tmhm_move(item) == move_id]
        if not items:
            return False
        hm_moves = {self.catalog.tmhm_move(item) for item in range(339, 347)}
        for index, mon in enumerate(party):
            if not bool(mon.get("is_egg", False)) and self.catalog.can_learn_tmhm(int(mon["species"]), items[0]):
                slots = [slot for slot, old in enumerate(mon["moves"]) if old not in hm_moves]
                if not slots:
                    continue
                slot = min(slots, key=lambda position: (int(mon["moves"][position]) != 0,
                    self.catalog.move(int(mon["moves"][position])).power, position))
                self.request_teach_move(items[0], index, slot)
                return True
        return False

    def _vertical(self, current: int, desired: int) -> int:
        return self.bridge.KEY_A if current == desired else (self.bridge.KEY_DOWN if current < desired else self.bridge.KEY_UP)

    def _choice_key(self, desired: int) -> int:
        # The12-byte native menu structure has cursor at+2 in both games.
        cursor = self.bridge.read_u8(self._address("sMenu") + 2)
        return self._vertical(cursor, desired)

    def _start_key(self, desired_action: int) -> int:
        count_name = "sNumStartMenuActions" if self.game == "emerald" else "sNumStartMenuItems"
        order_name = "sCurrentStartMenuActions" if self.game == "emerald" else "sStartMenuOrder"
        count = self.bridge.read_u8(self._address(count_name))
        if not 1 <= count <= 9:
            return 0
        order = self.bridge.read_memory(self._address(order_name), count)
        if desired_action not in order:
            raise ValueError("Requested start-menu action is currently unavailable")
        return self._vertical(self.bridge.read_u8(self._address("sStartMenuCursorPos")), order.index(desired_action))

    def _case_key(self, item_id: int) -> int:
        _, desired_row = self._bag._item_location(item_id)
        case = self._address("sTMCaseStaticResources")
        current = self.bridge.read_u16(case + 8) + self.bridge.read_u16(case + 10)
        return self._vertical(current, desired_row)

    def _forget_move_key(self, slot: int) -> int:
        """Read each native summary ABI, including FireRed's input-ready state.

        FireRed Task_InputHandler_SelectOrForgetMove accepts input only in
        state2 at+0x3288 and uses a separate global cursor. Emerald's
        Task_HandleReplaceMoveInput uses firstMoveIndex at+0x40C6. These are
        replacement screens, distinct from the normal move-reordering task.
        """
        pointer = self.bridge.read_u32(self._address("sMonSummaryScreen"))
        span = 0x3289 if self.game == "firered" else 0x40C7
        if pointer == 0:
            return 0
        if not 0x02000000 <= pointer <= 0x02040000 - span:
            raise ValueError("Native move-summary span lies outside EWRAM")
        if self.game == "firered":
            if self.bridge.read_u8(pointer + 0x3288) != 2:
                return 0
            current = self.bridge.read_u8(self._address("sMoveSelectionCursorPos"))
        else:
            current = self.bridge.read_u8(pointer + 0x40C6)
        if current > 4:
            raise ValueError("Invalid native replacement-move cursor")
        return self._vertical(current, slot)

    def _evolution_learning_slot(self) -> int | None:
        """Slot to forget for the evolving mon, or None to keep its four moves."""
        party = self.bridge.get_decrypted_player_party()
        index = self._evolution_party_index
        if index is None or index >= len(party):
            return None
        mon = party[index]
        moves = [int(move) for move in mon["moves"]]
        new_move = self.bridge.read_u16(self._address("gMoveToLearn"))
        if not 1 <= new_move <= 354:
            # Without the engine's pending move, keep the established conservative
            # rule: never an HM, and prefer an empty or the lowest-power slot.
            slots = [slot for slot, move in enumerate(moves) if move not in self._hm_moves]
            return min(slots, key=lambda slot: (moves[slot] != 0, self.catalog.move(moves[slot]).power, slot)) if slots else None
        species = int(mon["species"])
        types = self.catalog.species(species).types if 0 < species < self.catalog.species_count else ()
        choice = choose_move_to_forget(moves, new_move, self._move_summary, self._hm_moves, types)
        self.last_learning_choice = choice
        return choice.forget_slot

    def _refresh_evolution_session(self, tasks: set[str], callback_name: str) -> None:
        """Forget the evolving party index once neither evolution nor its summary is open.

        A stale index would let a later in-battle level-up learning screen be
        driven with the wrong party member.
        """
        evolving = bool(tasks & EVOLUTION_TASKS) or "Evolution" in callback_name
        summary = bool(tasks & LEARNING_TASKS) or "Summary" in callback_name
        if not evolving and not summary:
            self._evolution_party_index = None

    def evolution_active(self) -> bool:
        """Whether an evolution scene (or its move-learning summary) owns the screen."""
        tasks = self.active_tasks()
        callback = self.bridge.read_u32(self._address("gMain") + 4) & ~1
        name = self.functions.get(callback, "")
        return bool(tasks & EVOLUTION_TASKS) or "Evolution" in name or (
            self._evolution_party_index is not None and bool(tasks & LEARNING_TASKS))

    def scene_keys(self) -> int | None:
        """Own evolution screens in every context, before battle or field policies.

        A post-battle evolution runs while gMain.inBattle is still set, so the
        battle cursor controller would otherwise classify its Yes/No prompt as an
        animation and wait forever. Returns None when no such scene is active.
        """
        tasks = self.active_tasks()
        callback = self.bridge.read_u32(self._address("gMain") + 4) & ~1
        callback_name = self.functions.get(callback, "")
        self._refresh_evolution_session(tasks, callback_name)
        key = self._evolution_key(tasks, callback_name)
        if key is None:
            if self.scene == "evolution":
                self.scene = "none"
            return None
        self.scene = "evolution"
        return key if self.bridge.frame_count % 8 < 2 else 0

    def _evolution_key(self, tasks: set[str], callback_name: str) -> int | None:
        """Never issue the evolution-canceling B input, including stale cancel intent."""
        if "Task_EvolutionScene" in tasks:
            function = self._address("Task_EvolutionScene") & ~1
            for slot in range(16):
                address = self.bridge.memory_map.tasks + slot * 40
                if self.bridge.read_u8(address + 4) and self.bridge.read_u32(address) & ~1 == function:
                    # Both pinned native tasks use data[10] for the party member,
                    # state22 for move replacement, and data[6] for its substate.
                    index = self.bridge.read_u16(address + 28)
                    if index < 6:
                        self._evolution_party_index = index
                    if self.bridge.read_u16(address + 8) == 22 and self.bridge.read_u16(address + 20) == 4:
                        desired_slot = self._evolution_learning_slot()
                        yes_state = self.bridge.read_u16(address + 22)
                        desired = int(desired_slot is None) if yes_state == 5 else int(desired_slot is not None)
                        cursor = self.bridge.read_u8(self._address("gBattleCommunication") + 1)
                        return self._vertical(cursor, desired)
                    break
            # The move-selection summary keeps the evolution task alive; it owns input.
            if self._evolution_party_index is not None and tasks & LEARNING_TASKS:
                learning_slot = self._evolution_learning_slot()
                return self._forget_move_key(4 if learning_slot is None else learning_slot)
            return self.bridge.KEY_A if self._bag._text_waiting() else 0
        if self._evolution_party_index is not None and tasks & LEARNING_TASKS:
            learning_slot = self._evolution_learning_slot()
            return self._forget_move_key(4 if learning_slot is None else learning_slot)
        if "Evolution" in callback_name or any("Evolution" in task for task in tasks):
            return self.bridge.KEY_A if self._bag._text_waiting() else 0
        return None

    @staticmethod
    def _known_menu(callback_name: str, tasks: set[str]) -> bool:
        """Only recognized dismissible menus may receive an unsolicited B edge."""
        callbacks = ("bagmenu", "partymenu", "summaryscreen", "pokedex", "dexscreen", "tmcase",
                     "berrypouch", "optionsmenu", "optionmenu", "pokenav")
        menu_tasks = ("StartMenu", "BagMenu", "ItemContext", "FieldItemContext", "ChooseMonInput",
                      "SelectionMenuInput", "Pokedex", "DexScreen", "SummaryScreen", "HandleListInput")
        # Case-insensitive: the PokéNav callbacks are spelled CB2_Pokenav and CB2_InitPokeNav.
        lowered = callback_name.lower()
        return any(name in lowered for name in callbacks) or any(
            name in task for task in tasks for name in menu_tasks)

    def poll_key_mask(self) -> int | None:
        """Navigate or return None when ordinary overworld movement may proceed."""
        tasks = self.active_tasks()
        callback = self.bridge.read_u32(self._address("gMain") + 4) & ~1
        callback_name = self.functions.get(callback, "")
        self._refresh_evolution_session(tasks, callback_name)
        evolution_key = self._evolution_key(tasks, callback_name)
        if evolution_key is not None:
            self.scene = "evolution"
            return evolution_key if self.bridge.frame_count % 8 < 2 else 0
        if self.bridge.is_in_battle()[0]:
            return None
        field = callback_name in {"CB2_Overworld", "CB2_OverworldBasic"}
        known_menu = self._known_menu(callback_name, tasks)
        if field:
            self._evolution_party_index = None
        if not any("Multichoice" in task or "MultiChoice" in task for task in tasks):
            self._multichoice_label = None  # the next appearance is a fresh question
        request = self.pending
        if request is not None:
            if self.bridge.frame_count - self._started > 6000:
                self.pending = None
                self._closing = True
                self.last_result = "timeout"
            else:
                party = self.bridge.get_decrypted_player_party()
                if request.move in party[request.party_index]["moves"]:
                    self.pending = None
                    self._closing = True
                    self.last_result = "learned"
        if self.options_pending:
            # Success is only what SaveBlock2 holds after the native screen saved it.
            if self.options_optimal():
                self.options_pending, self._closing = False, True
                self.last_result = "options_set"
            elif self.bridge.frame_count - self._options_started > _OPTIONS_TIMEOUT:
                self.options_pending, self._closing = False, True
                self._options_retry_after = self.bridge.frame_count + 200_000
                self.last_result = "options_timeout"
        if self.bridge.frame_count % 8 >= 2:
            return 0 if self.busy or not field else None
        pokenav = self._pokenav_key() if "okenav" in callback_name.lower() else None
        if pokenav is not None:
            return pokenav
        if self._closing:
            if field and not known_menu:
                self._closing = False
                return None
            return self.bridge.KEY_B if known_menu else 0
        if self.options_pending:
            return self._options_key(tasks, field)
        if callback_name == "CB2_NamingScreen":
            pointer = self.bridge.read_u32(self._address("sNamingScreen"))
            if not 0x02000000 <= pointer <= 0x0203DFFF or self.bridge.read_u8(pointer + 0x1E10) != 2:
                return 0
            sprite_id = self.bridge.read_u8(pointer + 0x1E23)
            if sprite_id >= 64:
                raise ValueError("Invalid native naming cursor")
            x, y = struct.unpack("<hh", self.bridge.read_memory(self._address("gSprites") + 68 * sprite_id + 0x2E, 4))
            return self.bridge.KEY_A if (x, y) == (8, 2) else self.bridge.KEY_START
        if "Task_HandleStarterChooseInput" in tasks:
            callback_ptr = self._address("Task_HandleStarterChooseInput") & ~1
            slot = next(index for index in range(16)
                        if self.bridge.read_u8(self.bridge.memory_map.tasks + index * 40 + 4) and
                        self.bridge.read_u32(self.bridge.memory_map.tasks + index * 40) & ~1 == callback_ptr)
            selection = self.bridge.read_u16(self.bridge.memory_map.tasks + slot * 40 + 8)
            return self.bridge.KEY_A if selection == 2 else self.bridge.KEY_RIGHT
        if "Task_HandleConfirmStarterInput" in tasks:
            return self._choice_key(0)
        if "StarterChoose" in callback_name:
            return 0
        if request is not None and self.pending is not None:
            if any("StartMenu" in task for task in tasks):
                return self._start_key(2)  # BAG is native action2 in both games.
            if "Task_BagMenu_HandleInput" in tasks:
                item = 364 if self.game == "firered" else request.item
                return self._bag._native_bag_key(item)
            if "Task_HandleListInput" in tasks and self.game == "firered":
                return self._case_key(request.item)
            if any("ContextMenu" in task or "ItemContext" in task for task in tasks):
                return self._choice_key(0)
            if "Task_HandleChooseMonInput" in tasks:
                current = self.bridge.read_u8(self._address("gPartyMenu") + 9)
                return self._vertical(current, request.party_index)
            if tasks & {"Task_InputHandler_SelectOrForgetMove", "Task_HandleReplaceMoveInput",
                        "Task_HandleInputCantForgetHMsMoves"}:
                return self._forget_move_key(request.forget_slot)
            if "Task_HandleStopLearningMoveYesNoInput" in tasks:
                # No keeps the requested teaching alive after a canceled summary.
                return self._choice_key(1)
            if any("YesNo" in task for task in tasks):
                return self._choice_key(0)
            if field and self.bridge.is_hardware_ready():
                return self.bridge.KEY_START
            return self.bridge.KEY_A
        if "Task_SetClock_HandleConfirmInput" in tasks:
            return self._choice_key(0)
        if any("YesNo" in task for task in tasks):
            return self._choice_key(0)
        if any("Multichoice" in task or "MultiChoice" in task for task in tasks):
            return self._choice_key(self._multichoice_answer())
        if not field and ("Clock" in callback_name or any("Clock" in task for task in tasks)):
            return self.bridge.KEY_A
        if known_menu:
            return self.bridge.KEY_B
        if not field:
            return 0
        return None
