"""Observed native start menus, script choices and HM teaching through keys.

No routine writes player state. A taught move is certified only by decrypting
the actual party after the native learning script has finished.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import struct
from typing import cast

from core.battle.menu_cursor_controller import BattleMenuCursorController
from core.bridge.ffi_mgba import GBABridgeProtocol
from core.game_data import Game, RomCatalog, load_symbols


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

    @property
    def busy(self) -> bool:
        """Whether HM navigation or returning from its native screen is pending."""
        return self.pending is not None or self._closing

    def state_dict(self) -> dict[str, object]:
        """Checkpoint the UI intent; actual cursors remain in the emulator state."""
        request: dict[str, int] | None = None
        if self.pending is not None:
            request = {"item": self.pending.item, "move": self.pending.move,
                       "party_index": self.pending.party_index, "forget_slot": self.pending.forget_slot}
        return {"version": 1, "game": self.game, "pending": request, "started": self._started,
                "closing": self._closing, "menu_choice": self.menu_choice, "last_result": self.last_result}

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
        self.pending, self._started = request, started
        self._closing, self.menu_choice, self.last_result = closing, choice, result

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

    def poll_key_mask(self) -> int | None:
        """Navigate or return None when ordinary overworld movement may proceed."""
        if self.bridge.is_in_battle()[0]:
            return None
        tasks = self.active_tasks()
        callback = self.bridge.read_u32(self._address("gMain") + 4) & ~1
        callback_name = self.functions.get(callback, "")
        field = callback_name in {"CB2_Overworld", "CB2_OverworldBasic"}
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
        if self.bridge.frame_count % 8 >= 2:
            return 0 if self.busy or not field else None
        if self._closing:
            if field and not any("StartMenu" in task for task in tasks):
                self._closing = False
                return None
            return self.bridge.KEY_B
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
            return self._choice_key(self.menu_choice)
        if not field and ("Clock" in callback_name or any("Clock" in task for task in tasks)):
            return self.bridge.KEY_A
        if not field:
            return self.bridge.KEY_B
        return None
