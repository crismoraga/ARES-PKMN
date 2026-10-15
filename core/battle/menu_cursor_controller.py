"""RAM-guided battle cursor controller with one-frame-owner key output.

This controller never steps an emulator. tick() returns hardware keys for its
caller to apply and advance exactly one frame. Native menu readiness is resolved
from the pinned ARM controller callbacks rather than a timed guessed key chain.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast

from core.bridge.ffi_mgba import GBABridgeProtocol
from core.game_data import Game, RomCatalog

CommandKind = Literal["move", "switch", "ball", "run", "cancel"]


@lru_cache(maxsize=2)
def native_player_input_callbacks(game: str) -> dict[str, frozenset[int]]:
    """All legitimate input callbacks, including FireRed's first Oak-guided battle.

    Local names are reused by normal, Oak, Safari and recorded controllers. A
    name-to-address symbol dictionary loses those earlier occurrences. Simulated
    tutorial-input callbacks are deliberately excluded from this input binding.
    """
    if game not in ("firered", "emerald"):
        raise ValueError("unsupported game for native menu callbacks")
    path = Path(__file__).resolve().parents[2] / "data" / ("pokeemerald.sym" if game == "emerald" else "pokefirered.sym")
    aliases = {"HandleInputChooseAction": "action", "HandleInputChooseMove": "move",
               "OakOldManHandleInputChooseMove": "move", "HandleInputChooseTarget": "target",
               "WaitForMonSelection": "replacement", "PlayerHandleChoosePokemon": "replacement",
               "OakOldManHandleChoosePokemon": "replacement"}
    callbacks: dict[str, set[int]] = {"action": set(), "move": set(), "target": set(), "replacement": set()}
    pattern = re.compile(r"^([0-9a-fA-F]{8})\s+\w\s+[0-9a-fA-F]{8}\s+(\S+)$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match and match[2] in aliases:
            callbacks[aliases[match[2]]].add(int(match[1], 16) & ~1)
    return {mode: frozenset(addresses) for mode, addresses in callbacks.items()}


@dataclass(frozen=True)
class BattleCommand:
    kind: CommandKind
    index: int = 0
    target: int = 0
    item: int = 4


@dataclass(frozen=True)
class BattleMenuObservation:
    """RAM menu identity and currently accepting player battler."""
    mode: str
    battler: int
    action_cursor: int
    move_cursor: int
    native: bool


class BattleMenuCursorController:
    """Navigate 2x2 cursors from current RAM, preserving persistent cursor state."""

    def __init__(self, bridge: GBABridgeProtocol, pulse_frames: int = 2,
                 release_frames: int = 2, timeout_frames: int = 1800) -> None:
        if min(pulse_frames, release_frames, timeout_frames) < 1:
            raise ValueError("positive pulse/release/timeout frame counts required")
        self.bridge = bridge
        self.pulse_frames, self.release_frames, self.timeout_frames = pulse_frames, release_frames, timeout_frames
        self.pending: BattleCommand | None = None
        self._held = 0
        self._hold_left = 0
        self._release_left = 0
        self._started = 0
        self._phase = "initial"
        self._initial_item_count = 0
        self._input_battler = 0
        self._learning_catalog: RomCatalog | None = None
        self._learning_move = 0
        self._learning_slot: int | None = None
        self._learning_party_index: int | None = None
        self._input_callbacks = native_player_input_callbacks(bridge.memory_map.game)
        self._native_symbols = self._load_menu_symbols()

    def _load_menu_symbols(self) -> dict[str, int]:
        game = self.bridge.memory_map.game
        path = Path(__file__).resolve().parents[2] / "data" / ("pokeemerald.sym" if game == "emerald" else "pokefirered.sym")
        wanted = {"gBattlerControllerFuncs", "HandleInputChooseAction", "HandleInputChooseMove",
                  "HandleInputChooseTarget", "gPartyMenu", "Task_HandleChooseMonInput", "gMultiUsePlayerCursor",
                  "Task_HandleSelectionMenuInput", "Task_BagMenu_HandleInput", "Task_FieldItemContextMenuHandleInput",
                  "Task_ItemContext_SingleRow", "Task_ItemContext_MultipleRows", "gBagMenuState", "gBagPosition", "sMenu",
                  "gBattlerPositions", "gBattlerPartyIndexes", "sTextPrinters", "gBattlescriptCurrInstr",
                  "gBattleScripting", "gBattleCommunication", "gBattleStruct", "gMoveToLearn", "sMonSummaryScreen",
                  "sMoveSelectionCursorPos", "Task_InputHandler_SelectOrForgetMove", "Task_HandleReplaceMoveInput",
                  "Task_HandleInputCantForgetHMsMoves"}
        result: dict[str, int] = {}
        pattern = re.compile(r"^([0-9a-fA-F]{8})\s+\w\s+([0-9a-fA-F]{8})\s+(\S+)$")
        for line in path.read_text(encoding="utf-8").splitlines():
            match = pattern.match(line)
            # Several controllers have local callbacks with the same name;
            # the first emitted one belongs to the normal player controller.
            if match and match[3] in wanted and match[3] not in result:
                if match[3] != "sMenu" or int(match[2], 16) == 12:
                    result[match[3]] = int(match[1], 16)
        return result

    @property
    def busy(self) -> bool:
        return self.pending is not None or self._hold_left > 0 or self._release_left > 0

    def state_dict(self) -> dict[str, object]:
        """Persist intent and an in-flight hardware pulse without replaying inputs.

        Restore alongside the matching emulator savestate. ROM addresses and
        cursor contents belong to that emulator state, not the controller file.
        """
        command: dict[str, object] | None = None
        if self.pending is not None:
            command = {"kind": self.pending.kind, "index": self.pending.index,
                       "target": self.pending.target, "item": self.pending.item}
        return {"version": 1, "game": self.bridge.memory_map.game, "pending": command,
                "pulse_frames": self.pulse_frames, "release_frames": self.release_frames,
                "timeout_frames": self.timeout_frames, "held": self._held,
                "hold_left": self._hold_left, "release_left": self._release_left,
                "started": self._started, "phase": self._phase,
                "initial_item_count": self._initial_item_count, "input_battler": self._input_battler,
                "learning_move": self._learning_move, "learning_slot": self._learning_slot,
                "learning_party_index": self._learning_party_index}

    @staticmethod
    def _checkpoint_int(state: Mapping[str, object], name: str, minimum: int,
                        maximum: int) -> int:
        value = state.get(name)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"invalid battle checkpoint integer: {name}")
        return value

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Validate the complete checkpoint before atomically replacing FSM state."""
        if state.get("version") != 1 or state.get("game") != self.bridge.memory_map.game:
            raise ValueError("incompatible battle controller checkpoint")
        pulse = self._checkpoint_int(state, "pulse_frames", 1, 1_000_000)
        release = self._checkpoint_int(state, "release_frames", 1, 1_000_000)
        timeout = self._checkpoint_int(state, "timeout_frames", 1, 2**63 - 1)
        held = self._checkpoint_int(state, "held", 0, 0x3FF)
        hold_left = self._checkpoint_int(state, "hold_left", 0, pulse - 1)
        release_left = self._checkpoint_int(state, "release_left", 0, release)
        started = self._checkpoint_int(state, "started", 0, 2**63 - 1)
        initial_count = self._checkpoint_int(state, "initial_item_count", 0, 65535)
        battler = self._checkpoint_int(state, "input_battler", 0, 3)
        if battler not in (0, 2):
            raise ValueError("input battler must belong to the player side")
        phase = state.get("phase")
        if phase not in ("initial", "move_submitted", "party_confirm", "item_confirm", "item_used"):
            raise ValueError("invalid battle controller phase")
        pending: BattleCommand | None = None
        raw = state.get("pending")
        if raw is not None:
            if not isinstance(raw, Mapping):
                raise ValueError("invalid pending battle command")
            values = cast(Mapping[str, object], raw)
            kind = values.get("kind")
            if kind not in ("move", "switch", "ball", "run", "cancel"):
                raise ValueError("invalid pending battle command kind")
            index = self._checkpoint_int(values, "index", 0, 5)
            target = self._checkpoint_int(values, "target", 0, 2)
            item = self._checkpoint_int(values, "item", 1, 65535)
            if kind == "move" and index > 3:
                raise ValueError("invalid pending move slot")
            pending = BattleCommand(kind, index, target, item)
        learning_move = self._checkpoint_int({"learning_move": state.get("learning_move", 0)}, "learning_move", 0, 354)
        learning_slot = state.get("learning_slot")
        if learning_slot is not None and (type(learning_slot) is not int or not 0 <= learning_slot < 4):
            raise ValueError("invalid checkpoint learning slot")
        learning_party = state.get("learning_party_index")
        if learning_party is not None and (type(learning_party) is not int or not 0 <= learning_party < 6):
            raise ValueError("invalid checkpoint learning party index")
        # No live party decode, input injection or frame advancement occurs here;
        # an emulator savestate can be inside the game's temporary plaintext window.
        self.pending = pending
        self.pulse_frames, self.release_frames, self.timeout_frames = pulse, release, timeout
        self._held, self._hold_left, self._release_left = held, hold_left, release_left
        self._started, self._phase = started, phase
        self._initial_item_count, self._input_battler = initial_count, battler
        self._learning_move, self._learning_slot = learning_move, learning_slot
        self._learning_party_index = learning_party

    def request_command(self, kind: CommandKind, index: int = 0,
                        target: int = 0, item: int = 4) -> None:
        if self.busy:
            raise RuntimeError("previous battle command has not finished")
        if kind not in ("move", "switch", "ball", "run", "cancel"):
            raise ValueError("unknown battle command")
        if kind == "move" and not 0 <= index < 4:
            raise ValueError("move cursor must be 0..3")
        if kind == "switch" and not 0 <= index < 6:
            raise ValueError("party index must be 0..5")
        if kind == "switch":
            party = self.bridge.get_decrypted_player_party()
            if index >= len(party) or int(party[index]["hp"]) < 1:
                raise ValueError("replacement must be a healthy party member")
        if not 0 <= target < 3 or item < 1:
            raise ValueError("invalid target/item")
        self.pending = BattleCommand(kind, index, target, item)
        self._started = self.bridge.frame_count
        self._phase = "initial"
        self._initial_item_count = self.bridge.get_inventory().get(item, 0)
        if kind == "ball" and self._initial_item_count < 1:
            self.pending = None
            raise ValueError("requested item is unavailable in the bag")
        if kind == "ball" and 1 <= item <= 12 and self.bridge.is_in_battle()[1]:
            self.pending = None
            raise ValueError("capture is unavailable during trainer battles")

    def queue_move(self, move: int, target: int = 0) -> None:
        self.request_command("move", move, target)

    def request_move(self, index: int, target: int = 0) -> None:
        self.request_command("move", index, target)

    def request_switch(self, party_index: int) -> None:
        self.request_command("switch", party_index)

    def request_capture(self, item: int = 4) -> None:
        self.request_command("ball", item=item)

    def request_item(self, item: int, party_index: int = 0) -> None:
        self.request_command("ball", index=party_index, item=item)

    def cancel(self) -> None:
        self.pending = None
        self._held = self._hold_left = self._release_left = 0
        self._phase = "initial"

    def _pulse(self, key: int, complete: bool = False) -> int:
        self._held = key
        self._hold_left = self.pulse_frames - 1
        self._release_left = self.release_frames
        if complete:
            self.pending = None
        return key

    def _native_mode(self) -> str:
        callbacks = self._native_symbols["gBattlerControllerFuncs"]
        pointers = {slot: self.bridge.read_u32(callbacks + slot * 4) & ~1 for slot in (0, 2)}
        for mode in ("action", "move", "target"):
            addresses = self._input_callbacks[mode]
            for battler, pointer in pointers.items():
                if pointer in addresses:
                    self._input_battler = battler
                    return mode
        # A fainted right-side actor can enter Party without ever exposing its
        # action callback. Preserve the requesting controller's battler identity.
        for battler, pointer in pointers.items():
            if pointer in self._input_callbacks["replacement"]:
                self._input_battler = battler
                break
        task_modes = (("party", "Task_HandleChooseMonInput"), ("party_context", "Task_HandleSelectionMenuInput"),
                      ("bag", "Task_BagMenu_HandleInput"), ("bag_context", "Task_FieldItemContextMenuHandleInput"),
                      ("bag_context", "Task_ItemContext_SingleRow"), ("bag_context", "Task_ItemContext_MultipleRows"),
                      ("learn_forget", "Task_InputHandler_SelectOrForgetMove"), ("learn_forget", "Task_HandleReplaceMoveInput"),
                      ("learn_forget", "Task_HandleInputCantForgetHMsMoves"))
        active_tasks = {self.bridge.read_u32(self.bridge.memory_map.tasks + slot * 40) & ~1
                        for slot in range(16) if self.bridge.read_u8(self.bridge.memory_map.tasks + slot * 40 + 4)}
        for mode, symbol in task_modes:
            if self._native_symbols.get(symbol) in active_tasks:
                return mode
        script = self.bridge.read_u32(self._native_symbols["gBattlescriptCurrInstr"])
        if 0x08000000 <= script < 0x0A000000:
            opcode = self.bridge.read_u8(script)
            learn_state = self.bridge.read_u8(self._native_symbols["gBattleScripting"] + 0x1F)
            stats_state = self.bridge.read_u8(self._native_symbols["gBattleScripting"] + 0x1E)
            if opcode == 0x6C and stats_state in (6, 8):
                return "level_stats"
            if learn_state == 1 and opcode in (0x5A, 0x5B):
                return "learn_yesno" if opcode == 0x5A else "learn_stop"
        if self._text_waiting():
            return "text"
        return "animation"

    def _text_waiting(self) -> bool:
        # The pinned TextPrinter is 36 bytes: active+0x1B, state+0x1C.
        # WAIT/CLEAR/SCROLL_START states 1/2/3 request a new A/B edge;
        # timed pauses, audio waits and ordinary character rendering do not.
        base = self._native_symbols["sTextPrinters"]
        return any(self.bridge.read_u8(base + 36 * window + 0x1B) != 0
                   and self.bridge.read_u8(base + 36 * window + 0x1C) in (1, 2, 3)
                   for window in range(32))

    def menu_mode(self) -> str:
        observed = self.bridge.menu_state()
        return self._native_mode() if observed.mode == "battle" else observed.mode

    def observe_menu(self) -> BattleMenuObservation:
        """Expose accepting cursors for double battles and orchestration readiness."""
        native = self.bridge.menu_state().mode == "battle"
        mode = self.menu_mode()
        action, move = self._cursors(native)
        return BattleMenuObservation(mode, self._input_battler, action, move, native)

    def _cursors(self, native: bool) -> tuple[int, int]:
        if native:
            return (self.bridge.read_u8(self.bridge.memory_map.action_cursor + self._input_battler),
                    self.bridge.read_u8(self.bridge.memory_map.move_cursor + self._input_battler))
        return self.bridge.get_battle_cursors()

    def _grid_key(self, current: int, target: int) -> int:
        if current & 1 != target & 1:
            return self.bridge.KEY_RIGHT if target & 1 else self.bridge.KEY_LEFT
        if current & 2 != target & 2:
            return self.bridge.KEY_DOWN if target & 2 else self.bridge.KEY_UP
        return 0

    def _item_location(self, item: int) -> tuple[int, int]:
        """Locate an item in the native pocket slot order, excluding empty slots."""
        if self.bridge.get_inventory().get(item, 0) < 1:
            raise ValueError("requested item disappeared before selection")
        for pocket in range(5):
            descriptor = self.bridge.memory_map.bag_pockets + pocket * 8
            pointer = self.bridge.read_u32(descriptor)
            capacity = self.bridge.read_u8(descriptor + 4)
            if not pointer or not capacity:
                continue
            if capacity > 100:
                raise ValueError("corrupt bag pocket capacity")
            visible_index = 0
            for slot in range(capacity):
                slot_item = self.bridge.read_u16(pointer + 4 * slot)
                if slot_item == item:
                    return pocket, visible_index
                visible_index += slot_item != 0
        raise ValueError("item is not present in a native pocket descriptor")

    def _native_bag_key(self, item: int) -> int:
        desired_pocket, desired_row = self._item_location(item)
        if self.bridge.memory_map.game == "emerald":
            state = self._native_symbols["gBagPosition"]
            current_pocket = self.bridge.read_u8(state + 5)
            if current_pocket >= 5:
                raise ValueError("corrupt Emerald pocket index")
            row = self.bridge.read_u16(state + 8 + 2 * current_pocket)
            scroll = self.bridge.read_u16(state + 18 + 2 * current_pocket)
        else:
            state = self._native_symbols["gBagMenuState"]
            current_pocket = self.bridge.read_u16(state + 6)
            if current_pocket >= 3 or desired_pocket >= 3:
                raise ValueError("FireRed case items require a separate case screen")
            scroll = self.bridge.read_u16(state + 8 + 2 * current_pocket)
            row = self.bridge.read_u16(state + 14 + 2 * current_pocket)
        if current_pocket != desired_pocket:
            return self.bridge.KEY_RIGHT if current_pocket < desired_pocket else self.bridge.KEY_LEFT
        current_row = row + scroll
        if current_row != desired_row:
            return self.bridge.KEY_DOWN if current_row < desired_row else self.bridge.KEY_UP
        return self.bridge.KEY_A

    def _prepare_learning_choice(self) -> None:
        """Select a replaceable low-power move, preserving all eight native HMs."""
        new_move = self.bridge.read_u16(self._native_symbols["gMoveToLearn"])
        if not 1 <= new_move <= 354:
            raise ValueError("invalid level-up move identity")
        pointer = self.bridge.read_u32(self._native_symbols["gBattleStruct"])
        if not 0x02000000 <= pointer <= 0x0203FFEF:
            raise ValueError("invalid native battle structure for move learning")
        party_index = self.bridge.read_u8(pointer + 0x10)
        party = self.bridge.get_decrypted_player_party()
        if party_index >= len(party):
            raise ValueError("level-up recipient is outside the current party")
        if self._learning_catalog is None:
            game: Game = "emerald" if self.bridge.memory_map.game == "emerald" else "firered"
            self._learning_catalog = RomCatalog(game)
        catalog = self._learning_catalog
        hms = {catalog.tmhm_move(item) for item in range(339, 347)}
        old_moves = [int(move) for move in party[party_index]["moves"]]
        replaceable = [slot for slot, move in enumerate(old_moves) if move not in hms]
        chosen = min(replaceable, key=lambda slot: (catalog.move(old_moves[slot]).power, slot)) if replaceable else None
        self._learning_move, self._learning_slot = new_move, chosen
        self._learning_party_index = party_index

    def _learning_key(self, mode: str) -> int:
        self._prepare_learning_choice()
        if mode in ("learn_yesno", "learn_stop"):
            cursor = self.bridge.read_u8(self._native_symbols["gBattleCommunication"] + 1)
            # Decline only when every retained move is an HM; confirm stopping in
            # that case, otherwise return to learning if a cancel prompt appears.
            desired = int(self._learning_slot is None) if mode == "learn_yesno" else int(self._learning_slot is not None)
            key = self.bridge.KEY_A if cursor == desired else (self.bridge.KEY_DOWN if cursor < desired else self.bridge.KEY_UP)
            return self._pulse(key)
        pointer = self.bridge.read_u32(self._native_symbols["sMonSummaryScreen"])
        maximum_offset = 0x40C6 if self.bridge.memory_map.game == "emerald" else 0x3288
        if not 0x02000000 <= pointer <= 0x02040000 - maximum_offset - 1:
            raise ValueError("invalid native move-learning summary allocation")
        if self.bridge.memory_map.game == "firered":
            # Fade/setup/redraw/HM-error states accept no selection input.
            if self.bridge.read_u8(pointer + 0x3288) != 2:
                return 0
            cursor = self.bridge.read_u8(self._native_symbols["sMoveSelectionCursorPos"])
        else:
            cursor = self.bridge.read_u8(pointer + 0x40C6)
        if self._learning_slot is None:
            return self._pulse(self.bridge.KEY_B)
        desired_slot = self._learning_slot
        key = self.bridge.KEY_A if cursor == desired_slot else (self.bridge.KEY_DOWN if cursor < desired_slot else self.bridge.KEY_UP)
        return self._pulse(key)

    def tick(self) -> int:
        """Return keys for one frame; release between pulses to produce rising edges."""
        if self._hold_left:
            self._hold_left -= 1
            return self._held
        if self._release_left:
            self._release_left -= 1
            return 0
        if self.bridge.is_in_battle()[0] and self.bridge.menu_state().mode == "battle":
            mode = self.menu_mode()
            if mode == "level_stats":
                return self._pulse(self.bridge.KEY_A)
            if mode in ("learn_yesno", "learn_stop", "learn_forget"):
                return self._learning_key(mode)
        command = self.pending
        if command is None:
            if self.bridge.is_in_battle()[0] and self.menu_mode() == "text":
                return self._pulse(self.bridge.KEY_A)
            return 0
        if self.bridge.frame_count - self._started > self.timeout_frames:
            self.cancel()
            raise TimeoutError("battle command did not reach a recognized RAM menu")
        if not self.bridge.is_in_battle()[0]:
            self.cancel()
            return 0
        mode = self.menu_mode()
        native = self.bridge.menu_state().mode == "battle"
        if command.kind == "ball" and self.bridge.get_inventory().get(command.item, 0) < self._initial_item_count:
            self.cancel()
            return 0
        if command.kind == "cancel":
            return self._pulse(self.bridge.KEY_B, complete=True)
        if mode == "action":
            if self._phase in ("move_submitted", "party_confirm", "item_used"):
                self.cancel()
                return 0
            desired = {"move": 0, "ball": 1, "switch": 2, "run": 3}[command.kind]
            cursor, _ = self._cursors(native)
            direction = self._grid_key(cursor, desired)
            return self._pulse(direction or self.bridge.KEY_A, complete=not direction and command.kind == "run")
        if mode == "move" and command.kind == "move":
            _, cursor = self._cursors(native)
            direction = self._grid_key(cursor, command.index)
            # Singles need no target screen; doubles leave pending until target.
            doubles = bool(self.bridge.read_u32(self.bridge.memory_map.battle_flags) & 1)
            if not direction:
                self._phase = "move_submitted"
            return self._pulse(direction or self.bridge.KEY_A, complete=not direction and not doubles)
        if mode == "target" and command.kind == "move":
            current = self.bridge.read_u8(self._native_symbols["gMultiUsePlayerCursor"])
            desired_position = (1, 3, 2 if self._input_battler == 0 else 0)[command.target]
            positions = self.bridge.read_memory(self._native_symbols["gBattlerPositions"], 4)
            if desired_position not in positions:
                raise ValueError("requested target is absent from battler positions")
            desired_target = positions.index(desired_position)
            return self._pulse(self.bridge.KEY_A if current == desired_target else self.bridge.KEY_RIGHT,
                               complete=current == desired_target)
        if mode == "party" and command.kind in ("switch", "ball"):
            observed = self.bridge.menu_state()
            if observed.mode == "party":
                cursor, _ = self.bridge.get_battle_cursors()
                direction = self._grid_key(cursor, command.index)
            else:
                # gPartyMenu.slotId is +9 in both pinned generation-three structs.
                cursor = self.bridge.read_u8(self._native_symbols["gPartyMenu"] + 9)
                direction = self.bridge.KEY_DOWN if cursor < command.index else (self.bridge.KEY_UP if cursor > command.index else 0)
            if not direction:
                self._phase = "party_confirm"
            return self._pulse(direction or self.bridge.KEY_A, complete=not direction and not native)
        if mode == "party_context" and command.kind == "switch":
            # Shift / Send Out is the first action in the in-battle context.
            cursor = self.bridge.read_u8(self._native_symbols["sMenu"] + 2)
            return self._pulse(self.bridge.KEY_UP if cursor else self.bridge.KEY_A, complete=cursor == 0)
        if mode == "bag" and command.kind == "ball":
            if not native:
                return self._pulse(self.bridge.KEY_A, complete=True)
            key = self._native_bag_key(command.item)
            if key == self.bridge.KEY_A:
                self._phase = "item_confirm"
            return self._pulse(key)
        if mode == "bag_context" and command.kind == "ball":
            # Battle Use is index zero in both native context-menu action lists.
            cursor = self.bridge.read_u8(self._native_symbols["sMenu"] + 2)
            if not cursor:
                self._phase = "item_used"
            return self._pulse(self.bridge.KEY_UP if cursor else self.bridge.KEY_A,
                               complete=not cursor and 1 <= command.item <= 12)
        if mode in ("script", "text"):
            return self._pulse(self.bridge.KEY_A)
        return 0
