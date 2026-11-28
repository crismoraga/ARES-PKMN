"""Tests of the ARES-U Game Abstraction Layer (ares_u/gal) and the universal screen metric."""
from __future__ import annotations

import json

import numpy as np
import pytest

from ares_u.gal.backends import (GBA_A, GBA_B, GBA_DOWN, GBA_LEFT, GBA_RIGHT, GBA_SELECT, GBA_START, GBA_UP,
                                 MULTI_DLL, nds_keymask, read_cell, rgb555_to_xbgr)
from ares_u.gal.games import GAMES, CoordSpec, Field, game
from ares_u.nav.env import GAMES as NAV_GAMES
from ares_u.nav.env import NATIVE_GAMES, FramePreprocessor, default_state_path, screen_key


def test_registry_covers_five_generations_and_four_consoles() -> None:
    assert {spec.generation for spec in GAMES.values()} == {1, 2, 3, 4, 5}
    assert {spec.console for spec in GAMES.values()} == {"gb", "gbc", "gba", "nds"}
    assert set(NAV_GAMES) == set(GAMES)
    assert NATIVE_GAMES == ("firered", "emerald")
    with pytest.raises(ValueError):
        game("pokemon-x")


def test_nds_keymask_translates_every_universal_button() -> None:
    # desmume.controls.Keys: A=1, B=2, SELECT=3, START=4, RIGHT=5, LEFT=6, UP=7, DOWN=8 -> bit (k-1)
    expected = {GBA_A: 1 << 0, GBA_B: 1 << 1, GBA_SELECT: 1 << 2, GBA_START: 1 << 3, GBA_RIGHT: 1 << 4,
                GBA_LEFT: 1 << 5, GBA_UP: 1 << 6, GBA_DOWN: 1 << 7}
    for gba, nds in expected.items():
        assert nds_keymask(gba) == nds
    assert nds_keymask(GBA_A | GBA_DOWN) == (1 << 0) | (1 << 7)
    assert nds_keymask(0) == 0


def test_rgb555_to_xbgr_channels() -> None:
    raw = np.array([[0x001F, 0x03E0, 0x7C00, 0x7FFF]], dtype=np.uint16)  # pure R, G, B, white
    packed = rgb555_to_xbgr(raw)
    assert packed.dtype == np.uint32
    assert int(packed[0, 0]) == 0xF8 and int(packed[0, 1]) == 0xF8 << 8 and int(packed[0, 2]) == 0xF8 << 16
    assert int(packed[0, 3]) == 0xF8F8F8


def test_screen_key_is_coarse_and_stable() -> None:
    frame = np.zeros((84, 84), dtype=np.uint8)
    noisy = frame.copy()
    noisy[0, 0] = 255  # one pixel changes: same coarse key
    assert screen_key(frame) == screen_key(noisy)
    shifted = frame.copy()
    shifted[:, :42] = 200  # half the screen changes: new key
    assert screen_key(frame) != screen_key(shifted)
    assert len(screen_key(frame)) == 49


def test_preprocessor_maps_every_console_to_the_same_shape() -> None:
    pre = FramePreprocessor(84, 84)
    for shape in ((144, 160), (160, 240), (192, 256)):
        assert pre(np.zeros(shape, dtype=np.uint32)).shape == (84, 84)


def test_coord_spec_json_roundtrip_and_read_cell() -> None:
    spec = CoordSpec(map_num=Field(0xD363), x=Field(0xD367), y=Field(0xD366), map_group=Field(0xDCB5, 1, 1),
                     source="test")
    again = CoordSpec.from_json(json.loads(json.dumps(spec.to_json())))
    assert again == spec
    memory = {0xD363: 38, 0xD367: 3, 0xD366: 6, 0xDCB5: 24}
    cell = read_cell(spec, lambda item: memory[item.address])
    assert cell == {"map_group": 24, "map_num": 38, "x": 3, "y": 6}


@pytest.mark.skipif(not MULTI_DLL.exists() or not game("rojo").rom.exists(), reason="multi-core shim or ROM missing")
def test_multicore_shim_runs_gb_and_reads_the_verified_adapter() -> None:
    from ares_u.gal.backends import MGBAMultiBridge

    bridge = MGBAMultiBridge(game("rojo"))
    try:
        assert (bridge.width, bridge.height) == (160, 144)
        state_path = default_state_path("rojo")
        if state_path.exists():
            bridge.load_state(state_path.read_bytes())
            for _ in range(30):
                bridge.step_frame()
            cell = bridge.get_player_coords_and_map()
            assert cell["map_num"] == 38  # Red's house 2F (pokered map id 0x26)
        frame = bridge.framebuffer()
        assert frame.shape == (144, 160) and frame.dtype == np.uint32
    finally:
        bridge.close()
