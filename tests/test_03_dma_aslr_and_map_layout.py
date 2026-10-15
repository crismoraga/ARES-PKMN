"""Physical memory, DMA relocation, real synthetic transitions and native ABI."""
from __future__ import annotations

import ctypes
import struct
from pathlib import Path

import numpy as np
import pytest

from core.bridge.ffi_mgba import GBABridgeProtocol, NativeLibMGBABridge
from core.bridge.synthetic_gba_silicon import (
    SyntheticEncounter, SyntheticGBASiliconBridge, SyntheticWarp,
)


@pytest.mark.parametrize("emerald", [False, True])
def test_fifty_relocations_preserve_save_and_bag(emerald: bool) -> None:
    bridge = SyntheticGBASiliconBridge(emerald)
    assert isinstance(bridge, GBABridgeProtocol)
    assert ctypes.sizeof(bridge.ewram) == 256 * 1024
    assert ctypes.sizeof(bridge.iwram) == 32 * 1024
    bridge.set_event_flag(17, True)
    bridge.write_memory(bridge.get_save_block1_dynamic_ptr() + bridge.memory_map.vars_offset, b"\x34\x12")
    expected_bag = bridge.get_inventory()
    for _ in range(50):
        old1, old2 = bridge.get_save_block1_dynamic_ptr(), bridge.get_save_block2_dynamic_ptr()
        new1, new2 = bridge.relocate_save_blocks()
        assert old1 != new1 and old2 != new2
        assert bridge.read_memory(old1, 6) == b"\xa5" * 6
        assert bridge.get_event_flag(17)
        assert bridge.read_u16(new1 + bridge.memory_map.vars_offset) == 0x1234
        assert bridge.get_inventory() == expected_bag
    assert bridge.relocation_count == 50


@pytest.mark.parametrize("emerald", [False, True])
def test_cropped_metatiles_behaviors_and_warp(emerald: bool) -> None:
    bridge = SyntheticGBASiliconBridge(emerald)
    ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.uint16)
    collision = np.zeros_like(ids)
    collision[0, 0] = 3
    elevations = np.ones_like(ids)
    behaviors = np.zeros_like(ids)
    behaviors[2, 0] = 2
    bridge.configure_map(2, 3, ids, collision, elevations, behaviors,
                         [SyntheticWarp(2, 1, 0, 0, 3, 3)])
    bridge.transition_map(2, 3, 1, 1)
    matrices = bridge.extract_live_map_matrices()
    for actual, expected in zip(matrices, (ids, collision, elevations), strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(bridge.extract_behavior_map(), behaviors)
    bridge.inject_key_pulse(bridge.KEY_RIGHT, 15)
    assert bridge.get_player_coords_and_map()["x"] == 1
    bridge.inject_key_pulse(bridge.KEY_RIGHT, 1)
    coords = bridge.get_player_coords_and_map()
    assert (coords["map_group"], coords["map_num"], coords["x"], coords["y"]) == (0, 0, 3, 3)
    assert bridge.relocation_count == 2


def test_invalid_pointers_spans_and_restoration() -> None:
    bridge = SyntheticGBASiliconBridge()
    initial = bridge.save_state()
    for pointer in (0, 0x03000000, 0x02000001, 0x0203F000):
        bridge.write_memory(bridge.memory_map.save1_ptr, struct.pack("<I", pointer))
        with pytest.raises(ValueError):
            bridge.get_player_coords_and_map()
    bridge.load_state(initial)
    bridge.inject_key_pulse(bridge.KEY_RIGHT, 16)
    bridge.load_state(initial)
    assert bridge.get_player_coords_and_map()["x"] == 1
    for address, size in ((0x0203FFFF, 2), (0x03007FFF, 2), (-1, 4), (0x02000000, -1)):
        with pytest.raises(ValueError):
            bridge.read_memory(address, size)
    with pytest.raises(ValueError, match="corrupted"):
        bridge.load_state(initial[:-1] + bytes([initial[-1] ^ 1]))
    bridge.close()
    with pytest.raises(RuntimeError, match="closed"):
        bridge.step_frame()


def _pulse(bridge: SyntheticGBASiliconBridge, key: int) -> None:
    bridge.inject_key_pulse(key, 1)
    bridge.inject_key_pulse(0, 10)


def test_battle_victory_flag_and_capture_require_actions() -> None:
    bridge = SyntheticGBASiliconBridge()
    opponent = {"species": 19, "level": 2, "hp": 4, "maxHP": 4, "attack": 3,
                "defense": 3, "speed": 3, "spAttack": 3, "spDefense": 3,
                "moves": [33, 0, 0, 0], "pp": [35, 0, 0, 0], "types": [0, 0]}
    bridge.start_battle(SyntheticEncounter(opponent, trainer=True, reward_flag=100))
    assert not bridge.get_event_flag(100)
    _pulse(bridge, bridge.KEY_A)
    _pulse(bridge, bridge.KEY_A)
    assert not bridge.is_in_battle()[0]
    assert bridge.get_event_flag(100)
    assert any(e["kind"] == "damage" and e["damage"] > 0 for e in bridge.events)
    opponent.update({"hp": 1, "maxHP": 100, "status": 1})
    bridge.start_battle(SyntheticEncounter(opponent, catch_rate=255))
    _pulse(bridge, bridge.KEY_RIGHT)
    _pulse(bridge, bridge.KEY_A)
    _pulse(bridge, bridge.KEY_A)
    assert not bridge.is_in_battle()[0]
    assert len(bridge.get_decrypted_player_party()) == 2
    assert bridge.get_inventory()[4] == 19
    assert any(e["kind"] == "capture" and e["caught"] for e in bridge.events)


def test_synthetic_state_replays_rng_and_movement() -> None:
    bridge = SyntheticGBASiliconBridge()
    bridge.inject_key_pulse(bridge.KEY_RIGHT, 5)
    state = bridge.save_state()
    bridge.inject_key_pulse(bridge.KEY_RIGHT, 11)
    expected = bridge.save_state()
    bridge.load_state(state)
    bridge.inject_key_pulse(bridge.KEY_RIGHT, 11)
    assert bridge.save_state() == expected


@pytest.mark.native
@pytest.mark.parametrize("emerald", [False, True])
def test_native_real_core_state_and_framebuffer(emerald: bool) -> None:
    root = Path(__file__).resolve().parents[1]
    lib = root / "native/build/ares_mgba.dll"
    rom = root.parent / ("Pokemon - Emerald Version (USA, Europe).gba" if emerald else "Pokemon - FireRed Version (USA).gba")
    if not lib.exists() or not rom.exists():
        pytest.skip("Local licensed ROM and compiled mGBA shim are required")
    with NativeLibMGBABridge(lib, rom, emerald) as bridge:
        bridge.inject_key_pulse(0, 600)
        pixels = bridge.framebuffer()
        assert pixels.shape == (160, 240)
        assert np.unique(pixels).size > 2
        state = bridge.save_state()
        bridge.inject_key_pulse(bridge.KEY_START, 15)
        bridge.load_state(state)
        assert bridge.frame_count == 600
        np.testing.assert_array_equal(bridge.framebuffer(), pixels)
        battery = bridge.save_battery()
        assert len(battery) == 0x20000
        bridge.load_battery(battery)
