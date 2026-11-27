"""Emulator bridges for the ARES-U Game Abstraction Layer.

Both bridges implement the ``BridgeLike`` protocol of ``ares_u.nav.env`` (step a frame, set the
GBA-style key mask, return a packed XBGR framebuffer, save/load state, read the player cell for
measurement), so the pixel agent of E2 runs unchanged on every console:

* ``MGBAMultiBridge``: GB, GBC and GBA through ``native/build_multi/ares_mgba_multi.dll`` (mGBA 0.10.5
  built with the GB and GBA cores).
* ``DeSmuMEBridge``: NDS through py-desmume (libdesmume). The observation is the top screen, where
  the overworld of Gen 4/5 is drawn; the key mask is translated to DeSmuME's keypad.
"""
from __future__ import annotations

import ctypes
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .games import ROOT, CoordSpec, Field, GameSpec

MULTI_DLL = ROOT / "native" / "build_multi" / "ares_mgba_multi.dll"
MAX_W, MAX_H = 256, 224
NDS_W, NDS_H = 256, 192
# GBA key bits used by the universal action space (ares_u.nav.env.ACTION_KEYS).
GBA_A, GBA_B, GBA_SELECT, GBA_START, GBA_RIGHT, GBA_LEFT, GBA_UP, GBA_DOWN = (1 << i for i in range(8))


class MGBAMultiBridge:
    """GB/GBC/GBA bridge over the multi-core mGBA shim."""

    def __init__(self, spec: GameSpec, dll_path: Path = MULTI_DLL) -> None:
        if not dll_path.exists():
            raise FileNotFoundError(f"{dll_path} missing: build it with cmake -S native/multi -B native/build_multi")
        self.spec = spec
        self.coords = spec.coord_spec()
        lib = ctypes.CDLL(str(dll_path))
        lib.ares_create.restype, lib.ares_create.argtypes = ctypes.c_void_p, [ctypes.c_char_p]
        for name in ("ares_step", "ares_destroy", "ares_platform"):
            getattr(lib, name).argtypes = [ctypes.c_void_p]
        lib.ares_set_keys.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.ares_video_size.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
        lib.ares_pixels.restype, lib.ares_pixels.argtypes = ctypes.POINTER(ctypes.c_uint32), [ctypes.c_void_p]
        lib.ares_read.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t]
        lib.ares_read_raw.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        lib.ares_state_size.restype, lib.ares_state_size.argtypes = ctypes.c_size_t, [ctypes.c_void_p]
        for name in ("ares_save_state", "ares_load_state"):
            getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.ares_last_error.restype = ctypes.c_char_p
        self._lib = lib
        handle = lib.ares_create(str(spec.rom).encode("mbcs" if os.name == "nt" else "utf-8"))
        if not handle:
            raise RuntimeError(f"mGBA could not open {spec.rom}: {lib.ares_last_error()!r}")
        self._handle = ctypes.c_void_p(handle)
        width, height = ctypes.c_uint(), ctypes.c_uint()
        lib.ares_video_size(self._handle, ctypes.byref(width), ctypes.byref(height))
        self.width, self.height = int(width.value), int(height.value)
        self._pixels = np.ctypeslib.as_array(lib.ares_pixels(self._handle), shape=(MAX_H * MAX_W,))
        self.frame_count = 0

    def step_frame(self) -> None:
        self._lib.ares_step(self._handle)
        self.frame_count += 1

    def set_hardware_keys(self, key_mask: int) -> None:
        self._lib.ares_set_keys(self._handle, key_mask & 0x3FF)

    def framebuffer(self) -> NDArray[np.uint32]:
        return np.array(self._pixels.reshape(MAX_H, MAX_W)[: self.height, : self.width], dtype=np.uint32)

    def save_state(self) -> bytes:
        size = int(self._lib.ares_state_size(self._handle))
        buffer = ctypes.create_string_buffer(size)
        if not self._lib.ares_save_state(self._handle, buffer, size):
            raise RuntimeError(f"save_state failed: {self._lib.ares_last_error()!r}")
        return buffer.raw

    def load_state(self, state: bytes) -> None:
        size = int(self._lib.ares_state_size(self._handle))
        if len(state) != size:
            raise ValueError(f"state has {len(state)} bytes, the core expects {size}")
        buffer = ctypes.create_string_buffer(state, size)
        if not self._lib.ares_load_state(self._handle, buffer, size):
            raise RuntimeError(f"load_state failed: {self._lib.ares_last_error()!r}")

    def read(self, address: int, size: int, bank: int = -1) -> bytes:
        buffer = ctypes.create_string_buffer(size)
        ok = (self._lib.ares_read(self._handle, address, buffer, size) if bank < 0 else
              self._lib.ares_read_raw(self._handle, address, bank, buffer, size))
        if not ok:
            raise ValueError(f"invalid read at {address:#x}: {self._lib.ares_last_error()!r}")
        return buffer.raw

    def _field(self, item: Field) -> int:
        return int.from_bytes(self.read(item.address, item.width, item.bank), "little")

    def get_player_coords_and_map(self) -> dict[str, int]:
        if self.coords is None:
            raise RuntimeError(f"no measurement adapter for {self.spec.key}")
        return read_cell(self.coords, self._field)

    def is_in_battle(self) -> tuple[bool, bool]:
        return (False, False)

    def close(self) -> None:
        if self._handle:
            self._lib.ares_destroy(self._handle)
            self._handle = ctypes.c_void_p(0)


# DeSmuME keypad indices (desmume.controls.Keys) for the GBA-style key bits.
_NDS_KEYS = ((GBA_A, 1), (GBA_B, 2), (GBA_SELECT, 3), (GBA_START, 4), (GBA_RIGHT, 5), (GBA_LEFT, 6),
             (GBA_UP, 7), (GBA_DOWN, 8))


def nds_keymask(gba_mask: int) -> int:
    """Translate the universal (GBA-bit) key mask into DeSmuME's keypad mask."""
    mask = 0
    for bit, key in _NDS_KEYS:
        if gba_mask & bit:
            mask |= 1 << (key - 1)
    return mask


def rgb555_to_xbgr(raw: NDArray[np.uint16]) -> NDArray[np.uint32]:
    """DeSmuME's 15-bit display buffer (R in the low bits) to the packed XBGR used by the agent."""
    value = raw.astype(np.uint32)
    red = (value & 0x1F) << 3
    green = ((value >> 5) & 0x1F) << 3
    blue = ((value >> 10) & 0x1F) << 3
    packed: NDArray[np.uint32] = (red | (green << 8) | (blue << 16)).astype(np.uint32)
    return packed


class DeSmuMEBridge:
    """NDS bridge over py-desmume; observation = top screen (256x192)."""

    def __init__(self, spec: GameSpec) -> None:
        from desmume.emulator import DeSmuME

        self.spec = spec
        self.coords = spec.coord_spec()
        self._emu: Any = DeSmuME()
        self._emu.open(str(spec.rom))
        self._emu.volume_set(0)
        self.width, self.height = NDS_W, NDS_H
        self._tmp = Path(tempfile.mkdtemp(prefix="ares_nds_"))
        self._loaded_state: bytes | None = None
        self.frame_count = 0

    def step_frame(self) -> None:
        self._emu.cycle(with_joystick=False)
        self.frame_count += 1

    def set_hardware_keys(self, key_mask: int) -> None:
        self._emu.input.keypad_update(nds_keymask(key_mask))

    def touch(self, x: int, y: int) -> None:
        """Press the touch screen at bottom-screen pixel (x, y). Used only by the intro driver."""
        self._emu.input.touch_set_pos(x, y)

    def release_touch(self) -> None:
        self._emu.input.touch_release()

    def framebuffer(self) -> NDArray[np.uint32]:
        raw = np.ctypeslib.as_array(self._emu.display_buffer(), shape=(NDS_H * 2 * NDS_W,))
        return rgb555_to_xbgr(raw[: NDS_H * NDS_W].reshape(NDS_H, NDS_W))

    def save_state(self) -> bytes:
        path = self._tmp / "save.dst"
        self._emu.savestate.save_file(str(path))
        return path.read_bytes()

    def load_state(self, state: bytes) -> None:
        path = self._tmp / "load.dst"
        if state != self._loaded_state:
            path.write_bytes(state)
            self._loaded_state = state
        self._emu.savestate.load_file(str(path))

    def read(self, address: int, size: int, bank: int = -1) -> bytes:
        del bank
        reader = self._emu.memory.unsigned
        return bytes(reader[address + offset] for offset in range(size))

    def _field(self, item: Field) -> int:
        return int.from_bytes(self.read(item.address, item.width), "little")

    def get_player_coords_and_map(self) -> dict[str, int]:
        if self.coords is None:
            raise RuntimeError(f"no measurement adapter for {self.spec.key}")
        return read_cell(self.coords, self._field)

    def is_in_battle(self) -> tuple[bool, bool]:
        return (False, False)

    def close(self) -> None:
        try:
            self._emu.destroy()
        finally:
            for item in self._tmp.glob("*"):
                item.unlink(missing_ok=True)
            self._tmp.rmdir()


def read_cell(coords: CoordSpec, read_field: Any) -> dict[str, int]:
    """(map_group, map_num, x, y) through ``read_field(Field) -> int``; map_group 0 when absent."""
    return {"map_group": int(read_field(coords.map_group)) if coords.map_group is not None else 0,
            "map_num": int(read_field(coords.map_num)), "x": int(read_field(coords.x)),
            "y": int(read_field(coords.y))}


def make_bridge(spec: GameSpec) -> MGBAMultiBridge | DeSmuMEBridge:
    if spec.console == "nds":
        return DeSmuMEBridge(spec)
    return MGBAMultiBridge(spec)
