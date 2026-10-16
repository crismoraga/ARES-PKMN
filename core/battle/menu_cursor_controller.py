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
from typing import Any, Literal, cast

from core.battle.move_learning import MoveLearningChoice, choose_move_to_forget, rom_hm_moves, rom_move_summary
from core.bridge.ffi_mgba import GBABridgeProtocol
from core.game_data import Game, RomCatalog, load_symbols

CommandKind = Literal["move", "switch", "ball", "run", "cancel"]
# Screens owned by another controller; battle input must stay neutral there.
FOREIGN_SCENES = frozenset({"evolution"})


@lru_cache(maxsize=2)
def _native_party_bindings(game: str) -> tuple[int, int, frozenset[int]]:
    symbols = load_symbols(cast(Game, game))
    tasks = {"Task_HandleChooseMonInput", "Task_HandleSelectionMenuInput", "Task_ReturnToChooseMonAfterText",
             "Task_PrintAndWaitForText", "Task_ClosePartyMenu", "Task_ClosePartyMenuAndSetCB2"}
    return (symbols["gBattlePartyCurrentOrder"].address, symbols["CB2_UpdatePartyMenu"].address & ~1,
            frozenset(symbols[name].address & ~1 for name in tasks if name in symbols))


def native_party_slot_order(bridge: GBABridgeProtocol) -> tuple[int, ...] | None:
    """Map visual Party slots to stable battle party IDs while its UI is open.

    Native Party temporarily reorders gPlayerParty. Its three packed bytes use
    the high nibble for even visual slots and low nibble for odd slots. No map is
    inferred outside this UI, or from an uninitialized/non-permutation buffer.
    """
    if not bridge.is_in_battle()[0]:
        return None
    order_address, callback, task_functions = _native_party_bindings(bridge.memory_map.game)
    party_screen = bridge.read_u32(bridge.memory_map.main + 4) & ~1 == callback
    if not party_screen:
        party_screen = any(bridge.read_u8(bridge.memory_map.tasks + 40 * slot + 4) and
            bridge.read_u32(bridge.memory_map.tasks + 40 * slot) & ~1 in task_functions for slot in range(16))
    if not party_screen:
        return None
    order = tuple(nibble for value in bridge.read_memory(order_address, 3) for nibble in (value >> 4, value & 15))
    return order if sorted(order) == list(range(6)) else None


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
        self.last_learning_choice: MoveLearningChoice | None = None
        self._switch_identity: tuple[int, int] | None = None
        self._input_callbacks = native_player_input_callbacks(bridge.memory_map.game)
        self._native_symbols = self._load_menu_symbols()

    def _load_menu_symbols(self) -> dict[str, int]:
        game = self.bridge.memory_map.game
        path = Path(__file__).resolve().parents[2] / "data" / ("pokeemerald.sym" if game == "emerald" else "pokefirered.sym")
        wanted = {"gBattlerControllerFuncs", "HandleInputChooseAction", "HandleInputChooseMove",
                  "HandleInputChooseTarget", "gPartyMenu", "Task_HandleChooseMonInput", "gMultiUsePlayerCursor",
                  "Task_HandleSelectionMenuInput", "Task_BagMenu_HandleInput", "Task_FieldItemContextMenuHandleInput",
                  "Task_ItemContext_SingleRow", "Task_ItemContext_MultipleRows", "gBagMenuState", "gBagPosition", "sMenu",
                  "gBattlerPositions", "gBattlerPartyIndexes", "gBattlersCount", "gBattlePartyCurrentOrder",
                  "sTextPrinters", "gBattlescriptCurrInstr",
                  "gBattleScripting", "gBattleCommunication", "gBattleStruct", "gMoveToLearn", "sMonSummaryScreen",
                  "sMoveSelectionCursorPos", "Task_InputHandler_SelectOrForgetMove", "Task_HandleReplaceMoveInput",
                  "Task_HandleInputCantForgetHMsMoves", "Task_ReturnToChooseMonAfterText", "Task_PrintAndWaitForText",
                  "Task_DisplayCaughtMonDexPage", "Task_HandleCaughtMonPageInput", "Task_ExitCaughtMonPage",
                  "Task_HandlePokedexInput", "Task_ShowPokedexAreaScreen", "Task_HandlePokedexAreaScreenInput",
                  "Task_DexScreen_RegisterMonToPokedex", "Task_PokedexScreen", "Task_DexScreen_ShowMonPage",
                  "Task_DexScreen_CategorySubmenu", "Task_DexScreen_NumericalOrder", "Task_DexScreen_CharacteristicOrder",
                  "Task_EvolutionScene", "Task_TradeEvolutionScene", "gBattlerAttacker"}
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
                "learning_party_index": self._learning_party_index,
                "switch_identity": list(self._switch_identity) if self._switch_identity is not None else None}

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
        switch_identity: tuple[int, int] | None = None
        identity = state.get("switch_identity")
        if identity is not None:
            if (not isinstance(identity, list) or len(identity) != 2 or
                    any(type(value) is not int or not 0 <= value <= 0xFFFFFFFF for value in identity)):
                raise ValueError("invalid checkpoint replacement identity")
            switch_identity = (cast(int, identity[0]), cast(int, identity[1]))
        # No live party decode, input injection or frame advancement occurs here;
        # an emulator savestate can be inside the game's temporary plaintext window.
        self.pending = pending
        self.pulse_frames, self.release_frames, self.timeout_frames = pulse, release, timeout
        self._held, self._hold_left, self._release_left = held, hold_left, release_left
        self._started, self._phase = started, phase
        self._initial_item_count, self._input_battler = initial_count, battler
        self._learning_move, self._learning_slot = learning_move, learning_slot
        self._learning_party_index = learning_party
        self._switch_identity = switch_identity

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
            order = native_party_slot_order(self.bridge)
            party_slot = order.index(index) if order is not None else index
            if party_slot >= len(party) or int(party[party_slot]["hp"]) < 1:
                raise ValueError("replacement must be a healthy party member")
            if bool(party[party_slot].get("is_egg", False)):
                raise ValueError("an Egg cannot be a battle replacement")
            if self.bridge.menu_state().mode == "battle":
                count = self.bridge.read_u8(self._native_symbols["gBattlersCount"])
                if count in (2, 4):
                    positions = self.bridge.read_memory(self._native_symbols["gBattlerPositions"], count)
                    active = {self.bridge.read_u16(self._native_symbols["gBattlerPartyIndexes"] + 2 * slot)
                              for slot, position in enumerate(positions) if position in (0, 2)}
                    if index in active:
                        raise ValueError("replacement is already active in battle")
            self._switch_identity = (int(party[party_slot]["pid"]), int(party[party_slot]["otid"]))
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
        self._switch_identity = None

    def _pulse(self, key: int, complete: bool = False) -> int:
        self._held = key
        self._hold_left = self.pulse_frames - 1
        self._release_left = self.release_frames
        if complete:
            self.pending = None
        return key

    def _native_task_mode(self) -> str | None:
        """Resolve active screens before inspecting suspended battle callbacks.

        Opening Bag/Party leaves the battler controller pointer in memory while
        another main callback owns input. Outside battle those pointers and the
        battle script are stale, so they must never classify the field screen.
        """
        task_modes = (("evolution", "Task_EvolutionScene"), ("evolution", "Task_TradeEvolutionScene"),
                      ("party_error", "Task_ReturnToChooseMonAfterText"),
                      ("party_error", "Task_PrintAndWaitForText"),
                      ("party", "Task_HandleChooseMonInput"), ("party_context", "Task_HandleSelectionMenuInput"),
                      ("bag", "Task_BagMenu_HandleInput"), ("bag_context", "Task_FieldItemContextMenuHandleInput"),
                      ("bag_context", "Task_ItemContext_SingleRow"), ("bag_context", "Task_ItemContext_MultipleRows"),
                      ("learn_forget", "Task_InputHandler_SelectOrForgetMove"), ("learn_forget", "Task_HandleReplaceMoveInput"),
                      ("learn_forget", "Task_HandleInputCantForgetHMsMoves"),
                      ("pokedex", "Task_DisplayCaughtMonDexPage"), ("pokedex", "Task_HandleCaughtMonPageInput"),
                      ("pokedex", "Task_ExitCaughtMonPage"), ("pokedex", "Task_HandlePokedexInput"),
                      ("pokedex", "Task_ShowPokedexAreaScreen"), ("pokedex", "Task_HandlePokedexAreaScreenInput"),
                      ("pokedex", "Task_DexScreen_RegisterMonToPokedex"), ("pokedex", "Task_PokedexScreen"),
                      ("pokedex", "Task_DexScreen_ShowMonPage"), ("pokedex", "Task_DexScreen_CategorySubmenu"),
                      ("pokedex", "Task_DexScreen_NumericalOrder"), ("pokedex", "Task_DexScreen_CharacteristicOrder"))
        active_tasks = {self.bridge.read_u32(self.bridge.memory_map.tasks + slot * 40) & ~1
                        for slot in range(16) if self.bridge.read_u8(self.bridge.memory_map.tasks + slot * 40 + 4)}
        for mode, symbol in task_modes:
            if self._native_symbols.get(symbol) in active_tasks:
                return mode
        return None

    def _native_mode(self) -> str:
        callbacks = self._native_symbols["gBattlerControllerFuncs"]
        pointers = {slot: self.bridge.read_u32(callbacks + slot * 4) & ~1 for slot in (0, 2)}
        # The screen task takes input priority, but its requesting player actor
        # still comes from the replacement callback (including right-side faints).
        for battler, pointer in pointers.items():
            if pointer in self._input_callbacks["replacement"]:
                self._input_battler = battler
                break
        task_mode = self._native_task_mode()
        if task_mode is not None:
            return task_mode
        for mode in ("action", "move", "target"):
            addresses = self._input_callbacks[mode]
            for battler, pointer in pointers.items():
                if pointer in addresses:
                    self._input_battler = battler
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
            if opcode == 0xF2:
                return "pokedex"
            if opcode == 0xF3:
                return "caught_nickname"
            if opcode == 0x67:
                return "battle_yesno"
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
        if observed.mode == "battle":
            return self._native_mode()
        if observed.mode in ("overworld", "script"):
            return self._native_task_mode() or observed.mode
        return observed.mode

    def recovery_key(self, mode: str | None = None) -> int | None:
        """Return one safe dismissal edge for a screen with no active intent.

        A successful key submission is not evidence that a native menu consumed
        it. This path lets an interrupted/checkpointed Bag return to battle, and
        closes an accidental field menu without using or discarding an item.
        None leaves ordinary field navigation and script choices to their owner.
        """
        mode = self.menu_mode() if mode is None else mode
        battle = self.bridge.is_in_battle()[0]
        if mode in FOREIGN_SCENES:
            return None
        if mode in ("bag", "bag_context", "party", "party_context", "party_error"):
            key = self.bridge.KEY_B
        elif mode == "pokedex":
            key = self.bridge.KEY_A if battle else self.bridge.KEY_B
        elif battle and mode in ("caught_nickname", "battle_yesno"):
            key = self.bridge.KEY_B
        elif battle and mode in ("level_stats", "text"):
            key = self.bridge.KEY_A
        else:
            return None
        if self._hold_left:
            self._hold_left -= 1
            return self._held
        if self._release_left:
            self._release_left -= 1
            return 0
        return self._pulse(key)

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

    def _learning_recipient(self, party: list[dict[str, Any]]) -> int:
        """gBattleStruct->expGetterMonId names the recipient; fall back to the active mon."""
        pointer = self.bridge.read_u32(self._native_symbols["gBattleStruct"])
        if 0x02000000 <= pointer <= 0x0203FFEF:
            index = self.bridge.read_u8(pointer + 0x10)
            if index < len(party):
                return index
        # A freed/reallocated struct must not stall learning: use the actor in
        # battler slot 0, which is the mon gaining experience in singles.
        index = self.bridge.read_u16(self._native_symbols["gBattlerPartyIndexes"])
        if index < len(party):
            return index
        raise ValueError("level-up recipient is outside the current party")

    def _prepare_learning_choice(self) -> None:
        """Keep strong moves: replace the weakest non-HM only if the new move is better."""
        new_move = self.bridge.read_u16(self._native_symbols["gMoveToLearn"])
        if not 1 <= new_move <= 354:
            raise ValueError("invalid level-up move identity")
        party = self.bridge.get_decrypted_player_party()
        party_index = self._learning_recipient(party)
        if self._learning_catalog is None:
            game: Game = "emerald" if self.bridge.memory_map.game == "emerald" else "firered"
            self._learning_catalog = RomCatalog(game)
        catalog = self._learning_catalog
        mon = party[party_index]
        species = int(mon["species"])
        types = catalog.species(species).types if 0 < species < catalog.species_count else ()
        choice = choose_move_to_forget([int(move) for move in mon["moves"]], new_move,
                                       rom_move_summary(catalog), rom_hm_moves(catalog), types)
        self.last_learning_choice = choice
        self._learning_move, self._learning_slot = new_move, choice.forget_slot
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
            if mode in FOREIGN_SCENES:
                # Post-battle evolution keeps inBattle set; its controller owns input.
                self.cancel()
                return 0
            if mode == "party_error":
                # A native rejection (already active, trapped, fainted, Egg) is
                # not a completed switch. Dismiss it and let the policy replan.
                self.cancel()
                return self._pulse(self.bridge.KEY_B)
            if mode in ("level_stats", "pokedex"):
                return self._pulse(self.bridge.KEY_A)
            if mode in ("caught_nickname", "battle_yesno"):
                return self._pulse(self.bridge.KEY_B)
            if mode in ("learn_yesno", "learn_stop", "learn_forget"):
                return self._learning_key(mode)
        command = self.pending
        if command is None:
            if self.bridge.is_in_battle()[0]:
                return self.recovery_key() or 0
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
            desired_index = command.index
            if command.kind == "switch" and self._switch_identity is not None:
                # Opening Party can reorder gPlayerParty to display/battle order.
                # Track the requested individual instead of replaying a stale row.
                party = self.bridge.get_decrypted_player_party()
                matches = [index for index, mon in enumerate(party)
                           if (int(mon["pid"]), int(mon["otid"])) == self._switch_identity]
                if len(matches) != 1:
                    self.cancel()
                    return self._pulse(self.bridge.KEY_B)
                desired_index = matches[0]
            observed = self.bridge.menu_state()
            if observed.mode == "party":
                cursor, _ = self.bridge.get_battle_cursors()
                direction = self._grid_key(cursor, desired_index)
            else:
                # gPartyMenu.slotId is +9 in both pinned generation-three structs.
                cursor = self.bridge.read_u8(self._native_symbols["gPartyMenu"] + 9)
                direction = self.bridge.KEY_DOWN if cursor < desired_index else (self.bridge.KEY_UP if cursor > desired_index else 0)
            if not direction:
                self._phase = "party_confirm"
            return self._pulse(direction or self.bridge.KEY_A, complete=not direction and not native)
        if mode == "party_context" and command.kind == "switch":
            # Shift / Send Out is the first action in the in-battle context.
            cursor = self.bridge.read_u8(self._native_symbols["sMenu"] + 2)
            if not cursor:
                self._phase = "party_confirm"
            # Keep the chosen replacement until the game returns to its battle
            # action screen. A can be ignored during the native palette fade.
            return self._pulse(self.bridge.KEY_UP if cursor else self.bridge.KEY_A)
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
            # Retain intent until RAM confirms consumption. The first A edge can
            # arrive while the context task is still initializing and be ignored.
            # Clearing pending here strands the battle inside Bag indefinitely.
            return self._pulse(self.bridge.KEY_UP if cursor else self.bridge.KEY_A)
        if mode in ("script", "text"):
            return self._pulse(self.bridge.KEY_A)
        return 0
