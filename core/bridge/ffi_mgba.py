"""Bounded GBA memory access and the opaque Windows mGBA C adapter."""
from __future__ import annotations

import ctypes
import hashlib
import json
import struct
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from core.bridge.gba_structs import (
    BackupMapLayoutStruct, BattlePokemonStruct, ObjectEventStruct,
    PartyPokemonStruct, decrypt_party_pokemon,
)

UInt16Matrix = NDArray[np.uint16]
KEY_A, KEY_B, KEY_SELECT, KEY_START = 1, 2, 4, 8
KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN = 16, 32, 64, 128
KEY_R, KEY_L = 256, 512


@dataclass(frozen=True)
class GameMemoryMap:
    game: str
    rom_sha1: str
    save1_ptr: int
    save2_ptr: int
    party: int
    party_count: int
    battle_mons: int
    battle_flags: int
    action_cursor: int
    move_cursor: int
    layout: int
    avatar: int
    objects: int
    script: int
    tasks: int
    main: int
    map_header: int
    bag_pockets: int
    flags_offset: int
    vars_offset: int
    encryption_key_offset: int


FIRERED_MEMORY = GameMemoryMap(
    "firered", "41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc", 0x03005008,
    0x0300500C, 0x02024284, 0x02024029, 0x02023BE4, 0x02022B4C,
    0x02023FF8, 0x02023FFC, 0x03005040, 0x02037078, 0x02036E38,
    0x03000EB0, 0x03005090, 0x030030F0, 0x02036DFC, 0x0203988C,
    0xEE0, 0x1000, 0xF20,
)
EMERALD_MEMORY = GameMemoryMap(
    "emerald", "f3ae088181bf583e55daf962a92bb46f4f1d07b7", 0x03005D8C,
    0x03005D90, 0x020244EC, 0x020244E9, 0x02024084, 0x02022FEC,
    0x020244AC, 0x020244B0, 0x03005DC0, 0x02037590, 0x02037350,
    0x03000E40, 0x03005E00, 0x030022C0, 0x02037318, 0x02039DD8,
    0x1270, 0x139C, 0xAC,
)


def verify_symbol_map(memory: GameMemoryMap, symbol_path: str | Path) -> dict[str, int]:
    """Verify addresses against the pinned official symbol artifact, not a guess."""
    symbols: dict[str, int] = {}
    for line in Path(symbol_path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 4:
            symbols[parts[3]] = int(parts[0], 16)
    expected = {
        "gSaveBlock1Ptr": memory.save1_ptr, "gSaveBlock2Ptr": memory.save2_ptr,
        "gPlayerParty": memory.party, "gPlayerPartyCount": memory.party_count,
        "gBattleMons": memory.battle_mons, "gBattleTypeFlags": memory.battle_flags,
        "gActionSelectionCursor": memory.action_cursor, "gMoveSelectionCursor": memory.move_cursor,
        "gBackupMapLayout" if memory.game == "emerald" else "VMap": memory.layout,
        "gPlayerAvatar": memory.avatar, "gObjectEvents": memory.objects,
        "sGlobalScriptContext": memory.script, "gTasks": memory.tasks, "gMain": memory.main,
        "gMapHeader": memory.map_header, "gBagPockets": memory.bag_pockets,
    }
    for name, address in expected.items():
        if symbols.get(name) != address:
            raise ValueError(f"Pinned symbol mismatch for {name}: {symbols.get(name)} != {address:#x}")
    return expected


@dataclass(frozen=True)
class MenuState:
    in_battle: bool
    action_cursor: int
    move_cursor: int
    script_mode: int
    text_active: bool
    mode: str


@dataclass(frozen=True)
class BridgeSnapshot:
    frame: int
    game: str
    coords: dict[str, int]
    party: list[dict[str, Any]]
    in_battle: bool
    trainer_battle: bool
    menu: MenuState
    inventory: dict[int, int]


@runtime_checkable
class GBABridgeProtocol(Protocol):
    """Controller-facing API. Coordinates exclude the camera border."""
    KEY_A: int
    KEY_B: int
    KEY_SELECT: int
    KEY_START: int
    KEY_RIGHT: int
    KEY_LEFT: int
    KEY_UP: int
    KEY_DOWN: int
    KEY_R: int
    KEY_L: int
    memory_map: GameMemoryMap

    @property
    def frame_count(self) -> int:
        """Frames advanced by this instance."""
    @property
    def hardware_keys(self) -> int:
        """Current hardware input mask."""
    def step_frame(self) -> None:
        """Advance one VBLANK."""
    def set_hardware_keys(self, key_mask: int) -> None:
        """Set the ten hardware key bits."""
    def inject_key_pulse(self, key_mask: int, duration_frames: int = 4) -> None:
        """Synchronous standalone pulse; the orchestrator normally uses set keys."""
    def is_hardware_ready(self) -> bool:
        """Whether avatar and script engine permit an action."""
    def is_in_battle(self) -> tuple[bool, bool]:
        """Battle-active and trainer-battle state."""
    def get_save_block1_dynamic_ptr(self) -> int:
        """Freshly dereferenced GBA virtual pointer."""
    def get_save_block2_dynamic_ptr(self) -> int:
        """Freshly dereferenced GBA virtual pointer."""
    def read_memory(self, address: int, size: int) -> bytes:
        """Read a validated span."""
    def write_memory(self, address: int, data: bytes) -> None:
        """Write a validated RAM span."""
    def read_u8(self, address: int) -> int:
        """Read uint8."""
    def read_u16(self, address: int) -> int:
        """Read little-endian uint16."""
    def read_u32(self, address: int) -> int:
        """Read little-endian uint32."""
    def get_player_coords_and_map(self) -> dict[str, int]:
        """Read player state with no cached SaveBlock pointer."""
    def get_event_flag(self, flag_id: int) -> bool:
        """Read a persistent event flag."""
    def set_event_flag(self, flag_id: int, value: bool) -> None:
        """Write a persistent flag; never used for native campaign progress."""
    def extract_live_map_matrices(self) -> tuple[UInt16Matrix, UInt16Matrix, UInt16Matrix]:
        """Return IDs, collision and elevation."""
    def extract_behavior_map(self) -> UInt16Matrix:
        """Return decoded tileset behaviors."""
    def get_object_events(self) -> list[ObjectEventStruct]:
        """Copy sixteen object events."""
    def get_battle_pokemon(self, slot_idx: int) -> BattlePokemonStruct:
        """Copy a battle battler."""
    def get_decrypted_player_party(self) -> list[dict[str, Any]]:
        """Decode encrypted party."""
    def get_battle_cursors(self) -> tuple[int, int]:
        """Read active battler's cursors."""
    def get_inventory(self) -> dict[int, int]:
        """Read all bag pockets."""
    def menu_state(self) -> MenuState:
        """Read menu observations."""
    def snapshot(self) -> BridgeSnapshot:
        """Take a copied observation."""
    def save_state(self) -> bytes:
        """Serialize backend state."""
    def load_state(self, state: bytes) -> None:
        """Restore backend state."""
    def close(self) -> None:
        """Release owned resources."""


class GBAMemoryReader:
    """Shared decoding layer using supplied physical read/write operations."""
    KEY_A, KEY_B, KEY_SELECT, KEY_START = KEY_A, KEY_B, KEY_SELECT, KEY_START
    KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN = KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
    KEY_R, KEY_L = KEY_R, KEY_L

    def __init__(self, is_emerald: bool, read: Callable[[int, int], bytes],
                 write: Callable[[int, bytes], None]) -> None:
        self.memory_map = EMERALD_MEMORY if is_emerald else FIRERED_MEMORY
        self.is_emerald = is_emerald
        self._read = read
        self._write = write
        self._frame_count = 0
        self._keys = 0
        self._closed = False

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def hardware_keys(self) -> int:
        """Current key mask for exact recovery of a pulse between frames."""
        return self._keys

    @staticmethod
    def validate_span(address: int, size: int, writing: bool = False) -> None:
        if size < 0 or address < 0 or address + size > 0x100000000:
            raise ValueError("Invalid memory span")
        regions = [(0x02000000, 0x02040000), (0x03000000, 0x03008000)]
        if not writing:
            regions.append((0x08000000, 0x0A000000))
        if not any(start <= address and address + size <= end for start, end in regions):
            raise ValueError(f"Memory span outside canonical RAM/ROM: {address:#x}+{size}")

    def read_memory(self, address: int, size: int) -> bytes:
        if self._closed:
            raise RuntimeError("Bridge is closed")
        self.validate_span(address, size)
        return self._read(address, size)

    def write_memory(self, address: int, data: bytes) -> None:
        if self._closed:
            raise RuntimeError("Bridge is closed")
        self.validate_span(address, len(data), True)
        self._write(address, data)

    def read_u8(self, address: int) -> int:
        return self.read_memory(address, 1)[0]

    def read_u16(self, address: int) -> int:
        return int(struct.unpack("<H", self.read_memory(address, 2))[0])

    def read_u32(self, address: int) -> int:
        return int(struct.unpack("<I", self.read_memory(address, 4))[0])

    def _save_pointer(self, symbol: int, span: int) -> int:
        pointer = self.read_u32(symbol)
        if pointer & 3 or not 0x02000000 <= pointer < 0x02040000:
            raise ValueError(f"Uninitialized or invalid SaveBlock pointer {pointer:#x}")
        self.validate_span(pointer, span, True)
        return pointer

    def get_save_block1_dynamic_ptr(self) -> int:
        return self._save_pointer(self.memory_map.save1_ptr, 0x3D68 if not self.is_emerald else 0x3D88)

    def get_save_block2_dynamic_ptr(self) -> int:
        return self._save_pointer(self.memory_map.save2_ptr, 0xF24 if not self.is_emerald else 0xF2C)

    def get_player_coords_and_map(self) -> dict[str, int]:
        pointer = self.get_save_block1_dynamic_ptr()
        x, y, group, number = struct.unpack("<hhBB", self.read_memory(pointer, 6))
        index = self.read_u8(self.memory_map.avatar + 5)
        if index >= 16:
            raise ValueError("Invalid avatar object event index")
        z = self.read_u8(self.memory_map.objects + 36 * index + 11) & 15
        return {"x": x, "y": y, "z": z, "map_group": group, "map_num": number,
                "tile_transition_state": self.read_u8(self.memory_map.avatar + 3)}

    def get_event_flag(self, flag_id: int) -> bool:
        if not 0 <= flag_id < (self.memory_map.vars_offset - self.memory_map.flags_offset) * 8:
            raise ValueError("Flag id outside persistent SaveBlock flag array")
        address = self.get_save_block1_dynamic_ptr() + self.memory_map.flags_offset + flag_id // 8
        return bool(self.read_u8(address) & (1 << (flag_id & 7)))

    def set_event_flag(self, flag_id: int, value: bool) -> None:
        self.get_event_flag(flag_id)
        address = self.get_save_block1_dynamic_ptr() + self.memory_map.flags_offset + flag_id // 8
        mask = 1 << (flag_id & 7)
        old = self.read_u8(address)
        self.write_memory(address, bytes([(old | mask) if value else (old & ~mask)]))

    def _layout_grid(self) -> UInt16Matrix:
        layout = BackupMapLayoutStruct.from_buffer_copy(self.read_memory(self.memory_map.layout, 12))
        width, height = int(layout.width), int(layout.height)
        if width <= 15 or height <= 14 or width * height > 32768:
            raise ValueError("Invalid backup map dimensions")
        grid = np.frombuffer(self.read_memory(int(layout.map_ptr), width * height * 2), dtype="<u2")
        # Gen III stores width+15 and height+14: right border is eight tiles.
        return grid.reshape(height, width)[7:height - 7, 7:width - 8].copy()

    def extract_live_map_matrices(self) -> tuple[UInt16Matrix, UInt16Matrix, UInt16Matrix]:
        grid = self._layout_grid()
        return ((grid & np.uint16(0x3FF)).astype(np.uint16),
                ((grid >> 10) & np.uint16(3)).astype(np.uint16),
                ((grid >> 12) & np.uint16(15)).astype(np.uint16))

    def extract_behavior_map(self) -> UInt16Matrix:
        ids = self.extract_live_map_matrices()[0]
        layout_ptr = self.read_u32(self.memory_map.map_header)
        self.validate_span(layout_ptr, 24)
        primary = self.read_u32(layout_ptr + 16)
        secondary = self.read_u32(layout_ptr + 20)
        attr_offset, item_size, split = (16, 2, 512) if self.is_emerald else (20, 4, 640)
        self.validate_span(primary, attr_offset + 4)
        self.validate_span(secondary, attr_offset + 4)
        attrs = (self.read_u32(primary + attr_offset), self.read_u32(secondary + attr_offset))
        result = np.zeros(ids.shape, dtype=np.uint16)
        for tile in np.unique(ids):
            tile_id = int(tile)
            bank = int(tile_id >= split)
            offset = tile_id - split if bank else tile_id
            value = self.read_u16(attrs[bank] + offset * item_size) if item_size == 2 else self.read_u32(attrs[bank] + offset * item_size)
            result[ids == tile] = value & (0xFF if self.is_emerald else 0x1FF)
        return result

    def get_object_events(self) -> list[ObjectEventStruct]:
        return [ObjectEventStruct.from_buffer_copy(self.read_memory(self.memory_map.objects + i * 36, 36))
                for i in range(16)]

    def get_battle_pokemon(self, slot_idx: int) -> BattlePokemonStruct:
        if not 0 <= slot_idx < 4:
            raise ValueError("Battle slot must be 0..3")
        return BattlePokemonStruct.from_buffer_copy(self.read_memory(self.memory_map.battle_mons + 88 * slot_idx, 88))

    def get_decrypted_player_party(self) -> list[dict[str, Any]]:
        count = self.read_u8(self.memory_map.party_count)
        if count > 6:
            raise ValueError("Corrupt party count")
        result = [decrypt_party_pokemon(PartyPokemonStruct.from_buffer_copy(
            self.read_memory(self.memory_map.party + i * 100, 100))) for i in range(count)]
        if any(not mon["valid_checksum"] for mon in result):
            raise ValueError("Party contains an invalid checksum")
        return result

    def get_battle_cursors(self) -> tuple[int, int]:
        return self.read_u8(self.memory_map.action_cursor), self.read_u8(self.memory_map.move_cursor)

    def is_in_battle(self) -> tuple[bool, bool]:
        active = bool(self.read_u8(self.memory_map.main + 0x439) & 2)
        return active, active and bool(self.read_u32(self.memory_map.battle_flags) & 8)

    def is_hardware_ready(self) -> bool:
        mode = self.read_u8(self.memory_map.script + 1)
        transition = self.read_u8(self.memory_map.avatar + 3)
        player_id = self.read_u8(self.memory_map.avatar + 5)
        if player_id >= 16:
            return False
        frozen = bool(self.read_u32(self.memory_map.objects + 36 * player_id) & (1 << 8))
        lock_address = 0x03000F2C if self.is_emerald else 0x03000F9C
        return (mode == 0 and transition != 1 and not frozen and
                self.read_u8(self.memory_map.avatar + 6) == 0 and self.read_u8(lock_address) == 0)

    def get_inventory(self) -> dict[int, int]:
        key = self.read_u32(self.get_save_block2_dynamic_ptr() + self.memory_map.encryption_key_offset) & 65535
        result: dict[int, int] = {}
        for pocket in range(5):
            descriptor = self.memory_map.bag_pockets + pocket * 8
            pointer, capacity = self.read_u32(descriptor), self.read_u8(descriptor + 4)
            if capacity > 64:
                raise ValueError("Corrupt bag pocket capacity")
            if capacity == 0:
                continue
            self.validate_span(pointer, capacity * 4, True)
            for i in range(capacity):
                item, encrypted = struct.unpack("<HH", self.read_memory(pointer + i * 4, 4))
                if item:
                    result[item] = result.get(item, 0) + (encrypted ^ key)
        return result

    def menu_state(self) -> MenuState:
        battle, _ = self.is_in_battle()
        action, move = self.get_battle_cursors()
        script = self.read_u8(self.memory_map.script + 1)
        return MenuState(battle, action, move, script, script != 0,
                         "battle" if battle else ("script" if script else "overworld"))

    def snapshot(self) -> BridgeSnapshot:
        battle, trainer = self.is_in_battle()
        return BridgeSnapshot(self.frame_count, self.memory_map.game,
                              self.get_player_coords_and_map(), self.get_decrypted_player_party(),
                              battle, trainer, self.menu_state(), self.get_inventory())


class NativeLibMGBABridge(GBAMemoryReader):
    """Own one headless mGBA handle; verify ROM revision before dereferencing."""

    def __init__(self, libmgba_path: str | Path, rom_path: str | Path,
                 is_emerald: bool = False) -> None:
        rom = Path(rom_path).resolve(strict=True)
        expected = EMERALD_MEMORY if is_emerald else FIRERED_MEMORY
        root = Path(__file__).resolve().parents[2]
        symbols = root / "data" / ("pokeemerald.sym" if is_emerald else "pokefirered.sym")
        verify_symbol_map(expected, symbols)
        digest = hashlib.sha1(rom.read_bytes()).hexdigest()
        if digest != expected.rom_sha1:
            raise ValueError(f"ROM revision mismatch: {digest}; expected {expected.rom_sha1}")
        self._library = ctypes.CDLL(str(Path(libmgba_path).resolve(strict=True)))
        self._lock = threading.RLock()
        self._bind_abi()
        handle = self._library.ares_create(str(rom).encode("utf-8"))
        if not handle:
            raise RuntimeError(self._error())
        self._handle = ctypes.c_void_p(handle)
        super().__init__(is_emerald, self._native_read, self._native_write)

    def _bind_abi(self) -> None:
        lib = self._library
        signatures: dict[str, tuple[list[Any], Any]] = {
            "ares_create": ([ctypes.c_char_p], ctypes.c_void_p),
            "ares_destroy": ([ctypes.c_void_p], None),
            "ares_step": ([ctypes.c_void_p], None),
            "ares_set_keys": ([ctypes.c_void_p, ctypes.c_uint32], None),
            "ares_read": ([ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
            "ares_write": ([ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
            "ares_state_size": ([ctypes.c_void_p], ctypes.c_size_t),
            "ares_save_state": ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
            "ares_load_state": ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
            "ares_pixels": ([ctypes.c_void_p], ctypes.POINTER(ctypes.c_uint32)),
            "ares_last_error": ([], ctypes.c_char_p),
            "ares_battery_size": ([ctypes.c_void_p], ctypes.c_size_t),
            "ares_save_battery": ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
            "ares_load_battery": ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
        }
        for name, (args, result) in signatures.items():
            function = getattr(lib, name)
            function.argtypes, function.restype = args, result

    def _error(self) -> str:
        result: bytes | None = self._library.ares_last_error()
        return result.decode("utf-8", errors="replace") if result else "mGBA operation failed"

    def _native_read(self, address: int, size: int) -> bytes:
        with self._lock:
            buffer = ctypes.create_string_buffer(size)
            if not self._library.ares_read(self._handle, address, buffer, size):
                raise RuntimeError(self._error())
            return buffer.raw

    def _native_write(self, address: int, data: bytes) -> None:
        with self._lock:
            buffer = ctypes.create_string_buffer(data)
            if not self._library.ares_write(self._handle, address, buffer, len(data)):
                raise RuntimeError(self._error())

    def step_frame(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Bridge is closed")
            self._library.ares_step(self._handle)
            self._frame_count += 1

    def set_hardware_keys(self, key_mask: int) -> None:
        if not 0 <= key_mask <= 0x3FF:
            raise ValueError("Key mask outside ten hardware bits")
        with self._lock:
            if self._closed:
                raise RuntimeError("Bridge is closed")
            self._library.ares_set_keys(self._handle, key_mask)
            self._keys = key_mask

    def inject_key_pulse(self, key_mask: int, duration_frames: int = 4) -> None:
        if duration_frames < 1:
            raise ValueError("Pulse must last at least one frame")
        self.set_hardware_keys(key_mask)
        try:
            for _ in range(duration_frames):
                self.step_frame()
        finally:
            self.set_hardware_keys(0)

    def framebuffer(self) -> NDArray[np.uint32]:
        with self._lock:
            if self._closed:
                raise RuntimeError("Bridge is closed")
            return np.ctypeslib.as_array(self._library.ares_pixels(self._handle), shape=(160 * 240,)).reshape(160, 240).copy()

    def save_state(self) -> bytes:
        with self._lock:
            if self._closed:
                raise RuntimeError("Bridge is closed")
            size = int(self._library.ares_state_size(self._handle))
            buffer = ctypes.create_string_buffer(size)
            if not self._library.ares_save_state(self._handle, buffer, size):
                raise RuntimeError(self._error())
            pixels = self.framebuffer().tobytes()
            header = json.dumps({"frame": self.frame_count, "sha1": self.memory_map.rom_sha1,
                                 "pixel_bytes": len(pixels), "keys": self._keys}).encode()
            return b"ARESN1" + struct.pack("<I", len(header)) + header + buffer.raw + pixels

    def load_state(self, state: bytes) -> None:
        if state[:6] != b"ARESN1" or len(state) < 10:
            raise ValueError("Invalid ARES native state")
        size = struct.unpack_from("<I", state, 6)[0]
        metadata = json.loads(state[10:10 + size])
        if metadata["sha1"] != self.memory_map.rom_sha1:
            raise ValueError("State belongs to another ROM")
        payload = state[10 + size:]
        with self._lock:
            if self._closed:
                raise RuntimeError("Bridge is closed")
            core_size = int(self._library.ares_state_size(self._handle))
            pixel_bytes = int(metadata.get("pixel_bytes", 0))
            if pixel_bytes not in (0, 240 * 160 * 4) or len(payload) != core_size + pixel_bytes:
                raise ValueError("State size differs from this mGBA build")
            buffer = ctypes.create_string_buffer(payload[:core_size])
            if not self._library.ares_load_state(self._handle, buffer, core_size):
                raise ValueError(self._error())
            if pixel_bytes:
                ctypes.memmove(self._library.ares_pixels(self._handle), payload[core_size:], pixel_bytes)
            self._frame_count = int(metadata["frame"])
            self.set_hardware_keys(int(metadata.get("keys", 0)))

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._library.ares_destroy(self._handle)
                self._closed = True

    def save_battery(self) -> bytes:
        """Export cartridge SRAM/Flash bytes, separate from emulator savestate."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Bridge is closed")
            size = int(self._library.ares_battery_size(self._handle))
            if size == 0:
                raise RuntimeError("Cartridge save type has not initialized yet")
            buffer = ctypes.create_string_buffer(size)
            if not self._library.ares_save_battery(self._handle, buffer, size):
                raise RuntimeError(self._error())
            return buffer.raw

    def load_battery(self, data: bytes) -> None:
        """Restore cartridge data; reset or load a compatible state afterwards."""
        if not data or len(data) > 0x20000:
            raise ValueError("Invalid cartridge save size")
        with self._lock:
            if self._closed:
                raise RuntimeError("Bridge is closed")
            buffer = ctypes.create_string_buffer(data)
            if not self._library.ares_load_battery(self._handle, buffer, len(data)):
                raise ValueError(self._error())

    def __enter__(self) -> NativeLibMGBABridge:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


LowLevelMGBABridge = NativeLibMGBABridge
