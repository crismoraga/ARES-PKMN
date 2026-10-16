"""Native capture UI regressions using actual task pointers and encrypted RAM."""
from __future__ import annotations

import struct

import pytest

from core.battle.menu_cursor_controller import BattleMenuCursorController, native_party_slot_order
from core.bridge.ffi_mgba import MenuState
from core.bridge.synthetic_gba_silicon import SyntheticGBASiliconBridge


class NativeTaskMemory(SyntheticGBASiliconBridge):
    """Use native screen discovery over bounded, real ctypes RAM buffers."""

    def menu_state(self) -> MenuState:
        battle, _ = self.is_in_battle()
        action, move = self.get_battle_cursors()
        return MenuState(battle, action, move, 0, False, "battle" if battle else "overworld")


def fixture(emerald: bool) -> tuple[NativeTaskMemory, BattleMenuCursorController]:
    bridge = NativeTaskMemory(is_emerald=emerald)
    bridge.write_memory(bridge.memory_map.main + 0x439, b"\x02")
    controller = BattleMenuCursorController(bridge, pulse_frames=1, release_frames=1)
    bridge.write_memory(controller._native_symbols["gBattlerControllerFuncs"], bytes(16))
    bridge.write_memory(controller._native_symbols["sTextPrinters"], bytes(36 * 32))
    bridge.write_memory(bridge.memory_map.tasks, bytes(40 * 16))
    return bridge, controller


def task(bridge: NativeTaskMemory, controller: BattleMenuCursorController, symbol: str) -> None:
    bridge.write_memory(bridge.memory_map.tasks, bytes(40 * 16))
    bridge.write_memory(bridge.memory_map.tasks,
                        struct.pack("<IB3x", controller._native_symbols[symbol] | 1, 1))


def release(bridge: NativeTaskMemory, controller: BattleMenuCursorController) -> None:
    bridge.step_frame()
    assert controller.tick() == 0
    bridge.step_frame()


@pytest.mark.parametrize("emerald", [False, True])
def test_capture_intent_survives_unaccepted_use_until_inventory_consumes(emerald: bool) -> None:
    bridge, controller = fixture(emerald)
    try:
        context = "Task_ItemContext_SingleRow" if emerald else "Task_FieldItemContextMenuHandleInput"
        task(bridge, controller, context)
        bridge.write_memory(controller._native_symbols["sMenu"] + 2, b"\x00")
        initial_count = bridge.get_inventory()[4]
        controller.request_capture(4)
        original_frame = bridge.frame_count
        assert controller.tick() == bridge.KEY_A
        assert bridge.frame_count == original_frame
        release(bridge, controller)
        # Native setup/fade can ignore an A edge. Keep retrying the same intent
        # instead of telling the orchestrator that a ball was already thrown.
        assert controller.pending is not None
        assert controller.tick() == bridge.KEY_A
        release(bridge, controller)
        assert controller.pending is not None
        bridge.configure_inventory({4: initial_count - 1})
        assert controller.tick() == 0
        assert not controller.busy
    finally:
        bridge.close()


@pytest.mark.parametrize("emerald", [False, True])
def test_orphaned_capture_bag_cancels_instead_of_waiting_forever(emerald: bool) -> None:
    bridge, controller = fixture(emerald)
    try:
        context = "Task_ItemContext_SingleRow" if emerald else "Task_FieldItemContextMenuHandleInput"
        task(bridge, controller, context)
        state = controller.state_dict()
        state.update({"phase": "item_used", "pending": None})
        controller.load_state_dict(state)
        original_frame = bridge.frame_count
        assert controller.menu_mode() == "bag_context"
        assert controller.tick() == bridge.KEY_B
        assert bridge.frame_count == original_frame
        release(bridge, controller)
        task(bridge, controller, "Task_BagMenu_HandleInput")
        assert controller.tick() == bridge.KEY_B
    finally:
        bridge.close()


@pytest.mark.parametrize("emerald", [False, True])
def test_active_bag_precedes_suspended_controller_and_field_ignores_stale_callback(emerald: bool) -> None:
    bridge, controller = fixture(emerald)
    try:
        callbacks = controller._native_symbols["gBattlerControllerFuncs"]
        callback = next(iter(controller._input_callbacks["action"]))
        bridge.write_memory(callbacks, struct.pack("<I", callback | 1))
        task(bridge, controller, "Task_BagMenu_HandleInput")
        assert controller.menu_mode() == "bag"
        bridge.write_memory(bridge.memory_map.main + 0x439, b"\x00")
        assert controller.menu_mode() == "bag"
        assert controller.recovery_key() == bridge.KEY_B
        controller.cancel()
        bridge.write_memory(bridge.memory_map.tasks, bytes(40 * 16))
        assert controller.menu_mode() == "overworld"
        assert controller.recovery_key() is None
    finally:
        bridge.close()


@pytest.mark.parametrize("emerald", [False, True])
def test_caught_dex_dismissal_works_without_pending_capture_and_field_dex_closes(emerald: bool) -> None:
    bridge, controller = fixture(emerald)
    try:
        symbol = "Task_HandleCaughtMonPageInput" if emerald else "Task_DexScreen_RegisterMonToPokedex"
        task(bridge, controller, symbol)
        assert controller.pending is None
        assert controller.menu_mode() == "pokedex"
        assert controller.tick() == bridge.KEY_A
        controller.cancel()
        bridge.write_memory(bridge.memory_map.main + 0x439, b"\x00")
        assert controller.recovery_key() == bridge.KEY_B
        assert controller.recovery_key() == 0
    finally:
        bridge.close()


@pytest.mark.parametrize("emerald", [False, True])
def test_native_switch_rejects_active_and_waits_for_battle_screen(emerald: bool) -> None:
    bridge, controller = fixture(emerald)
    try:
        party = bridge.get_decrypted_player_party()
        reserve = dict(party[0], pid=43, species=19)
        bridge.configure_party([party[0], reserve])
        bridge.write_memory(controller._native_symbols["gBattlersCount"], b"\x02")
        bridge.write_memory(controller._native_symbols["gBattlerPositions"], bytes((0, 1, 255, 255)))
        bridge.write_memory(controller._native_symbols["gBattlerPartyIndexes"], bytes(8))
        with pytest.raises(ValueError, match="already active"):
            controller.request_switch(0)
        controller.request_switch(1)
        task(bridge, controller, "Task_HandleSelectionMenuInput")
        bridge.write_memory(controller._native_symbols["sMenu"] + 2, b"\x00")
        assert controller.tick() == bridge.KEY_A
        release(bridge, controller)
        assert controller.pending is not None
        assert controller.tick() == bridge.KEY_A
        release(bridge, controller)
        assert controller.pending is not None
        bridge.write_memory(bridge.memory_map.tasks, bytes(40 * 16))
        callback = next(iter(controller._input_callbacks["action"]))
        bridge.write_memory(controller._native_symbols["gBattlerControllerFuncs"], struct.pack("<I", callback | 1))
        assert controller.tick() == 0
        assert not controller.busy
    finally:
        bridge.close()


@pytest.mark.parametrize("emerald", [False, True])
def test_native_rejected_party_selection_is_identified_and_dismissed(emerald: bool) -> None:
    bridge, controller = fixture(emerald)
    try:
        party = bridge.get_decrypted_player_party()
        bridge.configure_party([party[0], dict(party[0], pid=43, species=19)])
        controller.request_switch(1)
        task(bridge, controller, "Task_ReturnToChooseMonAfterText")
        assert controller.menu_mode() == "party_error"
        assert controller.tick() == bridge.KEY_B
        assert controller.pending is None
        release(bridge, controller)
        assert controller.tick() == bridge.KEY_B
    finally:
        bridge.close()


@pytest.mark.parametrize("emerald", [False, True])
def test_party_order_maps_stable_target_and_checkpoint_identity(emerald: bool) -> None:
    bridge, controller = fixture(emerald)
    try:
        first = bridge.get_decrypted_player_party()[0]
        second = dict(first, pid=43, species=19)
        # Party UI has moved stable mon1 to row0 and stable mon0 to row1.
        bridge.configure_party([second, first])
        task(bridge, controller, "Task_HandleChooseMonInput")
        bridge.write_memory(controller._native_symbols["gBattlePartyCurrentOrder"], bytes((0x10, 0x23, 0x45)))
        assert native_party_slot_order(bridge) == (1, 0, 2, 3, 4, 5)
        controller.request_switch(1)
        restored = BattleMenuCursorController(bridge)
        restored.load_state_dict(controller.state_dict())
        bridge.write_memory(controller._native_symbols["gPartyMenu"] + 9, b"\x01")
        assert restored.tick() == bridge.KEY_UP
        assert restored.state_dict()["switch_identity"] == [43, int(first["otid"])]
        bridge.write_memory(controller._native_symbols["gBattlePartyCurrentOrder"], bytes(3))
        assert native_party_slot_order(bridge) is None  # invalid nibbles are never treated as a permutation
        bridge.write_memory(controller._native_symbols["gBattlePartyCurrentOrder"], bytes((0x10, 0x23, 0x45)))
        bridge.write_memory(bridge.memory_map.tasks, bytes(40 * 16))
        assert native_party_slot_order(bridge) is None  # stale order must not reorder action-screen observations
    finally:
        bridge.close()
