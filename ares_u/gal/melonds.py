"""Nintendo DS bridge over melonDS (libretro core "melonDS DS", driven in-process by ``libretro.py``).

Why: DeSmuME's Python binding (py-desmume) runs the ARM cores in interpreter mode (its JIT is compiled
in but cannot be enabled through the binding) and reads memory byte by byte through ctypes calls
(~2 s to copy the 4 MiB main RAM). melonDS DS has a working JIT on x86-64 and exposes the main RAM as a
plain pointer, so on the same loaded machine it runs ~3x faster (~300 emulated frames/s) and RAM reads
are memcpy-speed. Same interface as ``ares_u.gal.backends.DeSmuMEBridge`` (step_frame, set_hardware_keys,
touch/release_touch, framebuffer = top screen as packed XBGR, save_state/load_state, read), plus
``ram()`` for fast scans.

Setup (outside git): the core DLL at G:\\ARES\\emulators\\melonds\\melondsds_libretro.dll (official
libretro buildbot, override with env ARES_MELONDS_CORE) and ``libretro.py`` in the venv. melonDS uses its
built-in FreeBIOS/firmware, so no Nintendo BIOS dump is needed. Savestates are melonDS's own format:
they are NOT interchangeable with DeSmuME ``.dst`` states. One instance per process (libretro cores keep
global state).
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

NDS_W, NDS_H = 256, 192
MAIN_RAM_BASE, MAIN_RAM_SIZE = 0x02000000, 0x00400000
RETRO_MEMORY_SYSTEM_RAM = 2
DEFAULT_CORE = Path(os.environ.get("ARES_MELONDS_CORE", r"G:\ARES\emulators\melonds\melondsds_libretro.dll"))
OPTIONS = {
    b"melonds_jit_enable": b"enabled",
    b"melonds_console_mode": b"ds",
    b"melonds_sysfile_mode": b"builtin",
    b"melonds_render_mode": b"software",
    b"melonds_threaded_renderer": b"enabled",
    b"melonds_screen_layout1": b"top-bottom",
}
# universal GBA-bit mask -> libretro joypad field
_KEYS = ((1 << 0, "a"), (1 << 1, "b"), (1 << 2, "select"), (1 << 3, "start"), (1 << 4, "right"),
         (1 << 5, "left"), (1 << 6, "up"), (1 << 7, "down"), (1 << 8, "r"), (1 << 9, "l"))


def _silent_audio() -> Any:
    from libretro import ArrayAudioDriver

    class SilentAudioDriver(ArrayAudioDriver):  # type: ignore[misc]
        """Discards samples (the default array driver keeps every sample: unbounded memory)."""

        def sample(self, left: int, right: int) -> None:
            return None

        def sample_batch(self, frames: memoryview) -> int:
            return len(frames) // 2

    return SilentAudioDriver()


class MelonDSBridge:
    """NDS bridge over melonDS DS; observation = top screen (256x192)."""

    def __init__(self, spec: Any, core: Path | None = None) -> None:
        from libretro import IterableInputDriver, Session
        from libretro.api import JoypadState

        self.spec = spec
        self.coords = spec.coord_spec() if hasattr(spec, "coord_spec") else None
        self.width, self.height = NDS_W, NDS_H
        self.frame_count = 0
        self._joypad_type = JoypadState
        self._pad: Any = JoypadState()
        self._touch: tuple[int, int] | None = None
        core_path = core or DEFAULT_CORE
        if not core_path.exists():
            raise FileNotFoundError(f"melonDS DS core not found: {core_path}")
        rom = Path(spec.rom) if hasattr(spec, "rom") else Path(spec)
        bridge = self

        def inputs() -> Any:
            from libretro.api import Pointer, PointerState
            from libretro.drivers.input.iterable import PortState

            while True:
                pointer = None
                if bridge._touch is not None:
                    pointer = PointerState((Pointer(*bridge._touch, pressed=True),))
                yield PortState(joypad=bridge._pad, pointer=pointer)

        self._session: Any = Session(str(core_path), str(rom), options=dict(OPTIONS),
                                     input=lambda: IterableInputDriver(inputs), audio=_silent_audio)
        self._session.__enter__()
        size = int(self._session.core.get_memory_size(RETRO_MEMORY_SYSTEM_RAM))
        if size != MAIN_RAM_SIZE:
            raise RuntimeError(f"unexpected melonDS main RAM size {size:#x}")
        self._ram_address = self._session.core.get_memory_data(RETRO_MEMORY_SYSTEM_RAM).value

    # ------------------------------------------------------------------ emulation
    def step_frame(self) -> None:
        self._session.run()
        self.frame_count += 1

    def set_hardware_keys(self, key_mask: int) -> None:
        self._pad = self._joypad_type(**{name: bool(key_mask & bit) for bit, name in _KEYS})

    def touch(self, x: int, y: int) -> None:
        """Press the bottom screen at pixel (x, y): libretro pointer space spans both screens."""
        px = int((x + 0.5) / NDS_W * 2 * 0x7FFF - 0x7FFF)
        py = int((NDS_H + y + 0.5) / (2 * NDS_H) * 2 * 0x7FFF - 0x7FFF)
        self._touch = (px, py)

    def release_touch(self) -> None:
        self._touch = None

    def screens(self) -> NDArray[np.uint8]:
        """Both screens as RGB (384, 256, 3) (top above bottom)."""
        shot = self._session.video.screenshot()
        if shot is None:
            return np.zeros((2 * NDS_H, NDS_W, 3), dtype=np.uint8)
        bgrx = np.frombuffer(bytes(shot.data), dtype=np.uint8).reshape(shot.height, shot.width, 4)
        return np.ascontiguousarray(bgrx[..., 2::-1])

    def framebuffer(self) -> NDArray[np.uint32]:
        rgb = self.screens()[:NDS_H].astype(np.uint32)
        packed: NDArray[np.uint32] = (rgb[..., 0] | (rgb[..., 1] << 8) | (rgb[..., 2] << 16)).astype(np.uint32)
        return packed

    def save_state(self) -> bytes:
        buffer = bytearray(int(self._session.core.serialize_size()))
        if not self._session.core.serialize(buffer):
            raise RuntimeError("melonDS serialize failed")
        return bytes(buffer)

    def load_state(self, state: bytes) -> None:
        if not self._session.core.unserialize(state):
            raise RuntimeError("melonDS unserialize failed (states are not DeSmuME-compatible)")

    # ------------------------------------------------------------------ memory
    def ram(self) -> NDArray[np.uint8]:
        """Copy of the 4 MiB main RAM (0x02000000-0x023FFFFF) as a uint8 array."""
        raw = (ctypes.c_uint8 * MAIN_RAM_SIZE).from_address(self._ram_address)
        return np.frombuffer(bytes(raw), dtype=np.uint8)

    def read(self, address: int, size: int, bank: int = -1) -> bytes:
        del bank
        offset = (address - MAIN_RAM_BASE) & (MAIN_RAM_SIZE - 1)  # main RAM is mirrored every 4 MiB
        if offset + size > MAIN_RAM_SIZE:
            raise ValueError(f"read crosses main RAM end: {address:#x}+{size}")
        return ctypes.string_at(self._ram_address + offset, size)

    def _field(self, item: Any) -> int:
        return int.from_bytes(self.read(item.address, item.width), "little")

    def get_player_coords_and_map(self) -> dict[str, int]:
        if self.coords is None:
            raise RuntimeError(f"no measurement adapter for {getattr(self.spec, 'key', self.spec)}")
        from ares_u.gal.backends import read_cell

        return read_cell(self.coords, self._field)

    def is_in_battle(self) -> tuple[bool, bool]:
        return (False, False)

    def close(self) -> None:
        self._session.__exit__(None, None, None)
