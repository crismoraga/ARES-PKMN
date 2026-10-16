"""Deterministic physical-RAM laboratory with frame-driven game mechanics.

This is a configurable test silicon, not a ROM emulator or a campaign certificate.
Progress flags are consequences of configured interactions/battle victories; inputs
still traverse collision, movement, menus, damage, capture and party encryption.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.bridge.ffi_mgba import (
    GBAMemoryReader, KEY_A, KEY_B, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP,
    MenuState, UInt16Matrix,
)
from core.bridge.gba_structs import (
    BackupMapLayoutStruct, BattlePokemonStruct, ObjectEventStruct,
    encrypt_party_pokemon,
)
from core.safety.z3_integer_verifier import DamageContext, GbaIntegerSafetyVerifier


def _json_encode(value: object) -> object:
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    raise TypeError(f"Unsupported scenario state value {type(value).__name__}")


def _json_decode(value: dict[str, Any]) -> Any:
    if set(value) == {"__bytes__"}:
        return bytes.fromhex(value["__bytes__"])
    return value


@dataclass(frozen=True)
class SyntheticMove:
    move_id: int
    power: int
    move_type: int
    special: bool = False
    accuracy: int = 100
    max_pp: int = 35


@dataclass(frozen=True)
class SyntheticWarp:
    x: int
    y: int
    target_group: int
    target_map: int
    target_x: int
    target_y: int


@dataclass
class SyntheticEncounter:
    opponent: dict[str, Any]
    trainer: bool = False
    reward_flag: int | None = None
    catch_rate: int = 255
    trigger_after_steps: int = 1
    remaining: int = 1


@dataclass
class SyntheticMap:
    group: int
    number: int
    metatiles: UInt16Matrix
    collision: UInt16Matrix
    elevation: UInt16Matrix
    behaviors: UInt16Matrix
    warps: list[SyntheticWarp] = field(default_factory=list)
    npcs: list[ObjectEventStruct] = field(default_factory=list)
    encounters: list[SyntheticEncounter] = field(default_factory=list)
    heal_tiles: set[tuple[int, int]] = field(default_factory=set)
    interaction_flags: dict[tuple[int, int], int] = field(default_factory=dict)


# Exact Gen III chart: unlisted matchups are neutral; Mystery index 9 is reserved.
_SUPER: dict[int, tuple[int, ...]] = {
    1: (0, 5, 8, 15, 17), 2: (1, 6, 12), 3: (12,), 4: (3, 5, 8, 10, 13),
    5: (2, 6, 10, 15), 6: (12, 14, 17), 7: (7, 14), 8: (5, 15),
    10: (6, 8, 12, 15), 11: (4, 5, 10), 12: (4, 5, 11), 13: (2, 11),
    14: (1, 3), 15: (2, 4, 12, 16), 16: (16,), 17: (7, 14),
}
_RESIST: dict[int, tuple[int, ...]] = {
    0: (5, 8), 1: (2, 3, 6, 14), 2: (5, 8, 13), 3: (3, 4, 5, 7),
    4: (6, 12), 5: (1, 4, 8), 6: (1, 2, 3, 7, 8, 10), 7: (8, 17),
    8: (8, 10, 11, 13), 10: (5, 10, 11, 16), 11: (11, 12, 16),
    12: (2, 3, 6, 8, 10, 12, 16), 13: (12, 13, 16), 14: (8, 14),
    15: (8, 10, 11, 15), 16: (8,), 17: (1, 8, 17),
}
_IMMUNE = {0: (7,), 1: (7,), 3: (8,), 4: (2,), 7: (0,), 13: (4,), 14: (17,)}


class SyntheticGBASiliconBridge(GBAMemoryReader):
    """ctypes EWRAM/IWRAM, relocatable saves, and seeded scenario engine."""

    def __init__(self, is_emerald: bool = False, seed: int = 42) -> None:
        self.ewram = (ctypes.c_uint8 * 0x40000)()
        self.iwram = (ctypes.c_uint8 * 0x8000)()
        self._rng = random.Random(seed)
        self._rng_seed = seed
        self._keys = 0
        self._previous_keys = 0
        self._menu = "overworld"
        self._action_delay = 0
        self._movement: tuple[int, int, int] | None = None
        self._movement_left = 0
        self._mode = "walk"
        self._relocations = 0
        self._map_steps = 0
        self._maps: dict[tuple[int, int], SyntheticMap] = {}
        self._active_encounter: SyntheticEncounter | None = None
        self.events: list[dict[str, Any]] = []
        self.moves: dict[int, SyntheticMove] = {
            33: SyntheticMove(33, 35, 0, accuracy=95),
            22: SyntheticMove(22, 35, 12, True, max_pp=10),
            55: SyntheticMove(55, 40, 11, True, max_pp=25),
            52: SyntheticMove(52, 40, 10, True, max_pp=25),
            10: SyntheticMove(10, 40, 0),
            98: SyntheticMove(98, 40, 0, max_pp=30),
        }
        super().__init__(is_emerald, self._physical_read, self._physical_write)
        self._write_u32(self.memory_map.save1_ptr, 0x02008000)
        self._write_u32(self.memory_map.save2_ptr, 0x02018000)
        self.write_memory(self.get_save_block2_dynamic_ptr() + self.memory_map.encryption_key_offset,
                          struct.pack("<I", 0x2468ACE0))
        self.configure_inventory({4: 20, 13: 5})
        starter = {"species": 283 if is_emerald else 1, "pid": 42, "otid": 12345,
                   "level": 8, "hp": 30, "maxHP": 30, "attack": 20, "defense": 20,
                   "speed": 20, "spAttack": 24, "spDefense": 20,
                   "moves": [33, 55 if is_emerald else 22, 0, 0], "pp": [35, 25 if is_emerald else 10, 0, 0],
                   "ivs": [15] * 6, "evs": [0] * 6, "friendship": 70}
        self.configure_party([starter])
        ground = np.ones((10, 12), dtype=np.uint16)
        self.configure_map(0, 0, ground, np.zeros_like(ground), np.ones_like(ground))
        self.transition_map(0, 0, 1, 1, relocate=False)

    @property
    def relocation_count(self) -> int:
        return self._relocations

    def _physical_read(self, address: int, size: int) -> bytes:
        if 0x02000000 <= address < 0x02040000:
            return ctypes.string_at(ctypes.addressof(self.ewram) + address - 0x02000000, size)
        if 0x03000000 <= address < 0x03008000:
            return ctypes.string_at(ctypes.addressof(self.iwram) + address - 0x03000000, size)
        raise ValueError("Synthetic silicon has no cartridge ROM")

    def _physical_write(self, address: int, data: bytes) -> None:
        buffer = self.ewram if address < 0x03000000 else self.iwram
        base = 0x02000000 if address < 0x03000000 else 0x03000000
        ctypes.memmove(ctypes.addressof(buffer) + address - base, data, len(data))

    def _write_u32(self, address: int, value: int) -> None:
        self.write_memory(address, struct.pack("<I", value))

    def configure_inventory(self, inventory: dict[int, int]) -> None:
        """Install seeded bag quantities, encrypted using the current save key."""
        if len(inventory) > 64 or any(not 1 <= k <= 65535 or not 0 <= v <= 65535 for k, v in inventory.items()):
            raise ValueError("Invalid scenario inventory")
        key = self.read_u32(self.get_save_block2_dynamic_ptr() + self.memory_map.encryption_key_offset) & 65535
        for pocket in range(5):
            self.write_memory(self.memory_map.bag_pockets + pocket * 8, bytes(8))
        self.write_memory(self.memory_map.bag_pockets, struct.pack("<IB3x", 0x0201E000, len(inventory)))
        slots = b"".join(struct.pack("<HH", item, quantity ^ key) for item, quantity in sorted(inventory.items()))
        self.write_memory(0x0201E000, slots)

    def configure_party(self, party: list[dict[str, Any]]) -> None:
        """Place actual encrypted party records into RAM for a scenario."""
        if not 1 <= len(party) <= 6:
            raise ValueError("Party must contain one to six Pokemon")
        self.write_memory(self.memory_map.party_count, bytes([len(party)]))
        self.write_memory(self.memory_map.party, b"".join(bytes(encrypt_party_pokemon(mon)) for mon in party).ljust(600, b"\0"))

    def configure_move(self, move: SyntheticMove) -> None:
        if not 1 <= move.move_id <= 65535 or not 0 <= move.power <= 255 or not 0 <= move.move_type < 18:
            raise ValueError("Invalid move definition")
        if not 1 <= move.accuracy <= 100 or not 1 <= move.max_pp <= 255:
            raise ValueError("Invalid move accuracy or PP")
        self.moves[move.move_id] = move

    def configure_map(self, map_group: int, map_num: int, metatiles: UInt16Matrix,
                      collision: UInt16Matrix, elevation: UInt16Matrix,
                      behaviors: UInt16Matrix | None = None,
                      warps: list[SyntheticWarp] | None = None,
                      npcs: list[ObjectEventStruct] | None = None,
                      heal_tiles: set[tuple[int, int]] | None = None,
                      interaction_flags: dict[tuple[int, int], int] | None = None) -> None:
        """Register an unbordered scenario map; loading writes its real layouts."""
        if not 0 <= map_group <= 255 or not 0 <= map_num <= 255:
            raise ValueError("Map group/number must fit uint8")
        if metatiles.ndim != 2 or any(a.shape != metatiles.shape for a in (collision, elevation)):
            raise ValueError("Map matrices must have the same two-dimensional shape")
        height, width = metatiles.shape
        if height < 1 or width < 1 or (height + 14) * (width + 15) * 2 > 0x8000:
            raise ValueError("Map exceeds synthetic grid region")
        if np.any(metatiles > 1023) or np.any(collision > 3) or np.any(elevation > 15):
            raise ValueError("Packed metatile component out of bounds")
        actual_behaviors = np.zeros_like(metatiles) if behaviors is None else behaviors
        if actual_behaviors.shape != metatiles.shape or np.any(actual_behaviors > (255 if self.is_emerald else 511)):
            raise ValueError("Invalid behavior matrix")
        objects = npcs or []
        if len(objects) > 15:
            raise ValueError("At most fifteen non-player objects")
        old = self._maps.get((map_group, map_num))
        self._maps[(map_group, map_num)] = SyntheticMap(
            map_group, map_num, metatiles.astype(np.uint16).copy(), collision.astype(np.uint16).copy(),
            elevation.astype(np.uint16).copy(), actual_behaviors.astype(np.uint16).copy(),
            list(warps or []), [ObjectEventStruct.from_buffer_copy(bytes(n)) for n in objects],
            old.encounters if old else [], set(heal_tiles or set()), dict(interaction_flags or {}))

    def configure_encounter(self, map_group: int, map_num: int,
                            encounter: SyntheticEncounter) -> None:
        if not 1 <= encounter.catch_rate <= 255 or encounter.trigger_after_steps < 0 or encounter.remaining < 0:
            raise ValueError("Invalid encounter configuration")
        self._maps[(map_group, map_num)].encounters.append(encounter)

    def set_movement_mode(self, mode: str) -> None:
        if mode not in ("walk", "run", "bike"):
            raise ValueError("Movement mode must be walk, run or bike")
        self._mode = mode

    def relocate_save_blocks(self) -> tuple[int, int]:
        """Relocate both saves atomically, poison old bytes and preserve contents."""
        old1, old2 = self.get_save_block1_dynamic_ptr(), self.get_save_block2_dynamic_ptr()
        size1, size2 = (0x3D88, 0xF2C) if self.is_emerald else (0x3D68, 0xF24)
        data1, data2 = self.read_memory(old1, size1), self.read_memory(old2, size2)
        self._relocations += 1
        new1 = 0x02008000 + (self._relocations % 4) * 0x4000
        new2 = 0x02018000 + (self._relocations % 4) * 0x1000
        self.write_memory(old1, b"\xa5" * size1)
        self.write_memory(old2, b"\xa5" * size2)
        self.write_memory(new1, data1)
        self.write_memory(new2, data2)
        self._write_u32(self.memory_map.save1_ptr, new1)
        self._write_u32(self.memory_map.save2_ptr, new2)
        self.events.append({"kind": "dma_relocation", "frame": self.frame_count, "save1": new1, "save2": new2})
        return new1, new2

    def transition_map(self, map_group: int, map_num: int, x: int, y: int,
                       relocate: bool = True) -> None:
        """Scenario setup or completed warp; updates bordered RAM and saves."""
        scenario = self._maps[(map_group, map_num)]
        height, width = scenario.metatiles.shape
        if not 0 <= x < width or not 0 <= y < height:
            raise ValueError("Destination outside map")
        if relocate:
            self.relocate_save_blocks()
        self.write_memory(self.get_save_block1_dynamic_ptr(), struct.pack("<hhBB", x, y, map_group, map_num))
        grid = np.full((height + 14, width + 15), 0xC00, dtype=np.uint16)
        grid[7:height + 7, 7:width + 7] = scenario.metatiles | (scenario.collision << 10) | (scenario.elevation << 12)
        self.write_memory(0x02000000, grid.astype("<u2").tobytes())
        layout = BackupMapLayoutStruct(width + 15, height + 14, 0x02000000)
        self.write_memory(self.memory_map.layout, bytes(layout))
        self._write_u32(self.memory_map.map_header, 0x0201BFC0)
        self.write_memory(0x0201BFC0, struct.pack("<6I", width, height, 0, 0, 0x0201C000, 0x0201C040))
        attr_offset, split, item_size = (16, 512, 2) if self.is_emerald else (20, 640, 4)
        self._write_u32(0x0201C000 + attr_offset, 0x0201C080)
        self._write_u32(0x0201C040 + attr_offset, 0x0201CB00)
        for tile in np.unique(scenario.metatiles):
            values = np.unique(scenario.behaviors[scenario.metatiles == tile])
            if len(values) != 1:
                raise ValueError("A metatile ID cannot represent two different behaviors")
            tile_id = int(tile)
            bank, index = (0x0201CB00, tile_id - split) if tile_id >= split else (0x0201C080, tile_id)
            self.write_memory(bank + index * item_size, int(values[0]).to_bytes(item_size, "little"))
        self.write_memory(self.memory_map.objects, bytes(16 * 36))
        for index, npc in enumerate(scenario.npcs, start=1):
            self.write_memory(self.memory_map.objects + index * 36, bytes(npc))
        self.write_memory(self.memory_map.avatar + 3, bytes([0, 0, 0]))
        self._sync_player_object(x, y, int(scenario.elevation[y, x]))
        self._movement = None
        self._movement_left = 0
        self._map_steps = 0
        self.events.append({"kind": "map_transition", "frame": self.frame_count, "map": [map_group, map_num]})

    def _scenario(self) -> SyntheticMap:
        coords = self.get_player_coords_and_map()
        return self._maps[(coords["map_group"], coords["map_num"])]

    def _sync_player_object(self, x: int, y: int, z: int) -> None:
        old = ObjectEventStruct.from_buffer_copy(self.read_memory(self.memory_map.objects, 36))
        old.flags = 0x10001
        old.previousCoords_x, old.previousCoords_y = old.currentCoords_x, old.currentCoords_y
        old.currentCoords_x, old.currentCoords_y = x + 7, y + 7
        old.initialCoords_x, old.initialCoords_y = x + 7, y + 7
        old.elevations = ((int(old.elevations) & 15) << 4) | z
        self.write_memory(self.memory_map.objects, bytes(old))

    def set_hardware_keys(self, key_mask: int) -> None:
        if not 0 <= key_mask <= 0x3FF:
            raise ValueError("Key mask outside ten hardware bits")
        if self._closed:
            raise RuntimeError("Bridge is closed")
        self._keys = key_mask

    def inject_key_pulse(self, key_mask: int, duration_frames: int = 4) -> None:
        if duration_frames < 1:
            raise ValueError("Pulse duration must be positive")
        self.set_hardware_keys(key_mask)
        try:
            for _ in range(duration_frames):
                self.step_frame()
        finally:
            self.set_hardware_keys(0)
            self._previous_keys = 0

    def step_frame(self) -> None:
        if self._closed:
            raise RuntimeError("Bridge is closed")
        self._frame_count += 1
        rising = self._keys & ~self._previous_keys
        self._previous_keys = self._keys
        if self._action_delay:
            self._action_delay -= 1
        if self.is_in_battle()[0]:
            if self._action_delay == 0:
                self._battle_input(rising)
            return
        if self._movement is not None:
            self._movement_left -= 1
            if self._movement_left <= 0:
                x, y, z = self._movement
                self.write_memory(self.get_save_block1_dynamic_ptr(), struct.pack("<hh", x, y))
                self._sync_player_object(x, y, z)
                self.write_memory(self.memory_map.avatar + 3, b"\x00")
                self._movement = None
                self._map_steps += 1
                self.events.append({"kind": "step", "frame": self.frame_count, "x": x, "y": y, "z": z})
                self._after_step(x, y)
            return
        if rising & KEY_A:
            self._interact()
        if self._keys & (KEY_RIGHT | KEY_LEFT | KEY_UP | KEY_DOWN):
            self._begin_step()
            if self._movement is not None:
                self._movement_left -= 1

    def _begin_step(self) -> None:
        direction = next((k for k in (KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN) if self._keys & k), 0)
        dx, dy, facing = {KEY_RIGHT: (1, 0, 4), KEY_LEFT: (-1, 0, 3), KEY_UP: (0, -1, 2), KEY_DOWN: (0, 1, 1)}[direction]
        coords = self.get_player_coords_and_map()
        x, y, z = coords["x"], coords["y"], coords["z"]
        npc = ObjectEventStruct.from_buffer_copy(self.read_memory(self.memory_map.objects, 36))
        npc.facingDirection = facing
        self.write_memory(self.memory_map.objects, bytes(npc))
        scenario = self._scenario()
        tx, ty = x + dx, y + dy
        height, width = scenario.metatiles.shape
        if not 0 <= tx < width or not 0 <= ty < height:
            return
        behavior = int(scenario.behaviors[ty, tx])
        ledge_direction = {0x38: KEY_RIGHT, 0x39: KEY_LEFT, 0x3A: KEY_UP, 0x3B: KEY_DOWN}
        jumping = behavior in ledge_direction
        if jumping:
            if ledge_direction[behavior] != direction:
                return
            tx, ty = x + 2 * dx, y + 2 * dy
            if not 0 <= tx < width or not 0 <= ty < height:
                return
        if int(scenario.collision[ty, tx]) != 0:
            return
        tz = int(scenario.elevation[ty, tx])
        if not jumping and tz not in (0, 15, z) and z not in (0, 15):
            return
        if any(int(o.flags) & 1 and int(o.currentCoords_x) - 7 == tx and int(o.currentCoords_y) - 7 == ty for o in scenario.npcs):
            return
        frames = 4 if self._mode == "bike" else (8 if self._mode == "run" or self._keys & KEY_B else 16)
        self._movement = (tx, ty, z if tz == 15 or z == 15 else tz)
        self._movement_left = frames
        self.write_memory(self.memory_map.avatar + 3, b"\x01")

    def _after_step(self, x: int, y: int) -> None:
        scenario = self._scenario()
        warp = next((w for w in scenario.warps if w.x == x and w.y == y), None)
        if warp:
            self.transition_map(warp.target_group, warp.target_map, warp.target_x, warp.target_y)
            return
        encounter = next((e for e in scenario.encounters if e.remaining and self._map_steps >= e.trigger_after_steps), None)
        if encounter:
            encounter.remaining -= 1
            self.start_battle(encounter)

    def _interact(self) -> None:
        scenario = self._scenario()
        coords = self.get_player_coords_and_map()
        position = (coords["x"], coords["y"])
        if position in scenario.heal_tiles:
            party = self.get_decrypted_player_party()
            for mon in party:
                mon["hp"], mon["status"] = mon["maxHP"], 0
                mon["pp"] = [self.moves[m].max_pp if m in self.moves else 0 for m in mon["moves"]]
            self.configure_party(party)
            self.events.append({"kind": "heal", "frame": self.frame_count})
        if position in scenario.interaction_flags:
            flag = scenario.interaction_flags[position]
            self.set_event_flag(flag, True)
            self.events.append({"kind": "interaction", "frame": self.frame_count, "flag": flag})

    @staticmethod
    def _battle_struct(mon: dict[str, Any], types: tuple[int, int]) -> BattlePokemonStruct:
        result = BattlePokemonStruct()
        for name in ("species", "hp", "maxHP", "level", "attack", "defense", "speed", "spAttack", "spDefense", "friendship", "experience"):
            setattr(result, name, int(mon.get(name, 0)))
        result.personality = int(mon.get("pid", 0))
        result.otId = int(mon.get("otid", 0))
        result.type1, result.type2 = types
        result.item, result.status1 = int(mon.get("held_item", 0)), int(mon.get("status", 0))
        for i, value in enumerate(mon.get("moves", [33, 0, 0, 0])):
            result.moves[i] = value
        for i, value in enumerate(mon.get("pp", [35, 0, 0, 0])):
            result.pp[i] = value
        for i in range(8):
            result.statStages[i] = 6
        return result

    def start_battle(self, encounter: SyntheticEncounter) -> None:
        """Install a new fight; winner/reward follows only from action mechanics."""
        if self.is_in_battle()[0]:
            raise RuntimeError("A battle is already active")
        party = self.get_decrypted_player_party()
        player = next((m for m in party if int(m["hp"]) > 0), None)
        if player is None:
            raise RuntimeError("All party Pokemon are fainted")
        self._active_party_index = party.index(player)
        player_types = (11, 11) if player["species"] == 283 else ((10, 10) if player["species"] == 280 else ((12, 3) if player["species"] == 1 else (0, 0)))
        opponent_types = tuple(encounter.opponent.get("types", [0, 0]))
        if len(opponent_types) != 2:
            raise ValueError("Opponent types require two indices")
        self.write_memory(self.memory_map.battle_mons, bytes(self._battle_struct(player, player_types)))
        self.write_memory(self.memory_map.battle_mons + 88, bytes(self._battle_struct(encounter.opponent, (int(opponent_types[0]), int(opponent_types[1])))))
        self.write_memory(self.memory_map.main + 0x439, b"\x02")
        self._write_u32(self.memory_map.battle_flags, 8 if encounter.trainer else 0)
        self.write_memory(self.memory_map.action_cursor, bytes(4))
        self.write_memory(self.memory_map.move_cursor, bytes(4))
        self._menu = "action"
        self._action_delay = 0
        self._active_encounter = encounter
        self.events.append({"kind": "battle_start", "frame": self.frame_count, "trainer": encounter.trainer})

    def menu_state(self) -> MenuState:
        battle, _ = self.is_in_battle()
        action, move = self.get_battle_cursors()
        return MenuState(battle, action, move, self.read_u8(self.memory_map.script + 1),
                         self._action_delay > 0, self._menu)

    def _battle_input(self, rising: int) -> None:
        if not rising:
            return
        cursor_address = self.memory_map.move_cursor if self._menu == "move" else self.memory_map.action_cursor
        cursor = self.read_u8(cursor_address)
        if rising & KEY_RIGHT:
            cursor = cursor | 1
        elif rising & KEY_LEFT:
            cursor = cursor & ~1
        elif rising & KEY_DOWN:
            cursor = cursor | 2
        elif rising & KEY_UP:
            cursor = cursor & ~2
        self.write_memory(cursor_address, bytes([cursor]))
        if rising & KEY_B:
            self._menu = "action"
            return
        if not rising & KEY_A:
            return
        if self._menu == "move":
            self._fight_turn(cursor)
        elif self._menu == "bag":
            self._throw_ball(4)
        elif self._menu == "party":
            self._switch_party(cursor)
        elif cursor == 0:
            self._menu = "move"
        elif cursor == 1:
            self._menu = "bag"
        elif cursor == 2:
            self._menu = "party"
        elif self._active_encounter and not self._active_encounter.trainer:
            self._end_battle("fled")
        else:
            self._action_delay = 8

    def _damage(self, attacker: BattlePokemonStruct, defender: BattlePokemonStruct,
                move: SyntheticMove) -> tuple[int, str]:
        if move.power == 0:
            return 0, "status"
        if self._rng.randrange(100) >= move.accuracy:
            return 0, "miss"
        atk = int(attacker.spAttack if move.special else attacker.attack)
        defense = max(1, int(defender.spDefense if move.special else defender.defense))
        modifiers: list[tuple[int, int]] = []
        efficacy_num, efficacy_den = 1, 1
        defender_types = [int(defender.type1)]
        if defender.type2 != defender.type1:
            defender_types.append(int(defender.type2))
        for t in defender_types:
            if t in _IMMUNE.get(move.move_type, ()):
                modifiers.append((0, 1))
                efficacy_num = 0
                break
            if t in _SUPER.get(move.move_type, ()):
                modifiers.append((2, 1))
                efficacy_num *= 2
            elif t in _RESIST.get(move.move_type, ()):
                modifiers.append((1, 2))
                efficacy_den *= 2
        context = DamageContext(level=int(attacker.level), attack=atk, defense=defense,
            power=move.power, move_type=move.move_type, type_modifiers=tuple(modifiers) or ((1, 1),),
            category="special" if move.move_type >= 10 else "physical",
            stab=move.move_type in (int(attacker.type1), int(attacker.type2)),
            attack_stage=int(attacker.statStages[4 if move.move_type >= 10 else 1]),
            defense_stage=int(defender.statStages[5 if move.move_type >= 10 else 2]),
            attacker_status=int(attacker.status1), defender_status=int(defender.status1),
            attacker_ability=int(attacker.ability), defender_ability=int(defender.ability),
            attacker_species=int(attacker.species), defender_species=int(defender.species),
            attacker_item=int(attacker.item), defender_item=int(defender.item),
            attacker_hp=int(attacker.hp), attacker_max_hp=max(1, int(attacker.maxHP)))
        damage = GbaIntegerSafetyVerifier.compute_context_damage_reference(context, self._rng.randint(85, 100))
        text = "immune" if efficacy_num == 0 else ("super_effective" if efficacy_num > efficacy_den else ("not_very_effective" if efficacy_num < efficacy_den else "neutral"))
        return damage, text

    def _attack(self, attacker_idx: int, defender_idx: int, move_idx: int) -> None:
        attacker, defender = self.get_battle_pokemon(attacker_idx), self.get_battle_pokemon(defender_idx)
        move_id = int(attacker.moves[move_idx])
        if move_id == 0 or attacker.pp[move_idx] == 0:
            return
        if move_id not in self.moves:
            raise ValueError(f"Scenario move {move_id} lacks a definition")
        attacker.pp[move_idx] -= 1
        damage, text = self._damage(attacker, defender, self.moves[move_id])
        actual = min(int(defender.hp), damage)
        defender.hp -= actual
        self.write_memory(self.memory_map.battle_mons + attacker_idx * 88, bytes(attacker))
        self.write_memory(self.memory_map.battle_mons + defender_idx * 88, bytes(defender))
        self.events.append({"kind": "damage", "frame": self.frame_count, "attacker": attacker_idx,
                            "move": move_id, "type": self.moves[move_id].move_type, "damage": actual, "text": text})

    def _fight_turn(self, move_index: int) -> None:
        player, enemy = self.get_battle_pokemon(0), self.get_battle_pokemon(1)
        if not player.moves[move_index] or not player.pp[move_index]:
            return
        legal = [i for i in range(4) if enemy.moves[i] and enemy.pp[i]]
        opponent_move = self._rng.choice(legal) if legal else 0
        order = [(0, 1, move_index), (1, 0, opponent_move)] if player.speed >= enemy.speed else [(1, 0, opponent_move), (0, 1, move_index)]
        for attacker, defender, index in order:
            if self.get_battle_pokemon(attacker).hp and self.get_battle_pokemon(defender).hp:
                self._attack(attacker, defender, index)
        self._sync_battle_party()
        if self.get_battle_pokemon(1).hp == 0:
            self._end_battle("won")
        elif self.get_battle_pokemon(0).hp == 0:
            self.events.append({"kind": "faint", "frame": self.frame_count, "party_index": self._active_party_index})
            remaining = [i for i, m in enumerate(self.get_decrypted_player_party()) if int(m["hp"]) > 0]
            if remaining:
                self._switch_party(remaining[0], retaliation=False)
            else:
                self._end_battle("lost")
        else:
            self._menu = "action"
            self._action_delay = 8

    def _sync_battle_party(self) -> None:
        party = self.get_decrypted_player_party()
        battler = self.get_battle_pokemon(0)
        mon = party[self._active_party_index]
        mon["hp"], mon["status"], mon["pp"] = int(battler.hp), int(battler.status1), list(battler.pp)
        self.configure_party(party)

    def _switch_party(self, index: int, retaliation: bool = True) -> None:
        party = self.get_decrypted_player_party()
        if index >= len(party) or int(party[index]["hp"]) == 0:
            return
        self._active_party_index = index
        self.write_memory(self.memory_map.battle_mons, bytes(self._battle_struct(party[index], (0, 0))))
        self._menu = "action"
        if retaliation:
            enemy = self.get_battle_pokemon(1)
            legal = [i for i in range(4) if enemy.moves[i] and enemy.pp[i]]
            if legal:
                self._attack(1, 0, legal[0])
                self._sync_battle_party()
        self._action_delay = 8

    def _throw_ball(self, item: int) -> None:
        encounter = self._active_encounter
        if encounter is None or encounter.trainer:
            self._menu = "action"
            return
        inventory = self.get_inventory()
        if inventory.get(item, 0) == 0:
            self._menu = "action"
            return
        inventory[item] -= 1
        self.configure_inventory(inventory)
        enemy = self.get_battle_pokemon(1)
        maximum, hp = max(1, int(enemy.maxHP)), int(enemy.hp)
        multiplier = {2: 20, 3: 15, 4: 10}.get(item, 10)
        rate_with_ball = encounter.catch_rate * multiplier // 10
        a = (3 * maximum - 2 * hp) * rate_with_ball // (3 * maximum)
        if enemy.status1 & 7 or enemy.status1 & 32:
            a *= 2
        elif enemy.status1 & (8 | 16 | 64):
            a = a * 3 // 2
        threshold = 65536 if a >= 255 else (0 if a <= 0 else 1048560 // math.isqrt(math.isqrt(16711680 // a)))
        caught = all(self._rng.randrange(65536) < threshold for _ in range(4))
        self.events.append({"kind": "capture", "frame": self.frame_count, "caught": caught, "a": a, "threshold": threshold})
        if caught:
            party = self.get_decrypted_player_party()
            mon = dict(encounter.opponent)
            mon.update({"pid": self._rng.getrandbits(32), "otid": party[0]["otid"], "hp": int(enemy.hp),
                        "moves": list(enemy.moves), "pp": list(enemy.pp), "status": int(enemy.status1)})
            if len(party) < 6:
                party.append(mon)
                self.configure_party(party)
            self._end_battle("caught")
        else:
            legal = [i for i in range(4) if enemy.moves[i] and enemy.pp[i]]
            if legal:
                self._attack(1, 0, legal[0])
                self._sync_battle_party()
            if self.get_battle_pokemon(0).hp == 0:
                self.events.append({"kind": "faint", "frame": self.frame_count, "party_index": self._active_party_index})
                self._end_battle("lost")
            else:
                self._menu, self._action_delay = "action", 8

    def _end_battle(self, result: str) -> None:
        encounter = self._active_encounter
        self._sync_battle_party()
        if result == "won":
            party = self.get_decrypted_player_party()
            mon = party[self._active_party_index]
            enemy = self.get_battle_pokemon(1)
            xp = max(1, int(enemy.level) * 64 // 7)
            mon["experience"] += xp
            self.configure_party(party)
            if encounter and encounter.reward_flag is not None:
                self.set_event_flag(encounter.reward_flag, True)
        self.write_memory(self.memory_map.main + 0x439, b"\x00")
        self._menu = "overworld"
        self.events.append({"kind": "battle_end", "frame": self.frame_count, "result": result})
        self._active_encounter = None
        self.relocate_save_blocks()

    def save_state(self) -> bytes:
        maps: list[dict[str, Any]] = []
        for m in self._maps.values():
            maps.append({"group": m.group, "number": m.number, "metatiles": m.metatiles.tolist(),
                         "collision": m.collision.tolist(), "elevation": m.elevation.tolist(),
                         "behaviors": m.behaviors.tolist(), "warps": [vars(w) for w in m.warps],
                         "npcs": [bytes(n).hex() for n in m.npcs],
                         "encounters": [vars(e) for e in m.encounters],
                         "heal_tiles": list(m.heal_tiles),
                         "interaction_flags": [[x, y, f] for (x, y), f in m.interaction_flags.items()]})
        state = {"game": self.memory_map.game, "ewram": bytes(self.ewram).hex(), "iwram": bytes(self.iwram).hex(),
                 "frame": self.frame_count, "keys": self._keys, "previous_keys": self._previous_keys,
                 "menu": self._menu, "action_delay": self._action_delay, "movement": self._movement,
                 "movement_left": self._movement_left, "mode": self._mode, "relocations": self._relocations,
                 "map_steps": self._map_steps, "maps": maps, "rng": self._rng.getstate(),
                 "events": self.events, "moves": [vars(m) for m in self.moves.values()],
                 "active_encounter": vars(self._active_encounter) if self._active_encounter else None,
                 "active_party_index": getattr(self, "_active_party_index", 0)}
        payload = zlib.compress(json.dumps(state, separators=(",", ":"), default=_json_encode).encode())
        return b"ARESS1" + hashlib.sha256(payload).digest() + payload

    def load_state(self, state: bytes) -> None:
        if state[:6] != b"ARESS1" or hashlib.sha256(state[38:]).digest() != state[6:38]:
            raise ValueError("Invalid or corrupted synthetic state")
        data = json.loads(zlib.decompress(state[38:]), object_hook=_json_decode)
        if data["game"] != self.memory_map.game:
            raise ValueError("State belongs to another game")
        ewram, iwram = bytes.fromhex(data["ewram"]), bytes.fromhex(data["iwram"])
        if len(ewram) != 0x40000 or len(iwram) != 0x8000:
            raise ValueError("State RAM size mismatch")
        self._maps.clear()
        for m in data["maps"]:
            self.configure_map(m["group"], m["number"], np.array(m["metatiles"], dtype=np.uint16),
                               np.array(m["collision"], dtype=np.uint16), np.array(m["elevation"], dtype=np.uint16),
                               np.array(m["behaviors"], dtype=np.uint16),
                               [SyntheticWarp(**w) for w in m["warps"]],
                               [ObjectEventStruct.from_buffer_copy(bytes.fromhex(n)) for n in m["npcs"]],
                               {tuple(p) for p in m["heal_tiles"]},
                               {(v[0], v[1]): v[2] for v in m["interaction_flags"]})
            for e in m["encounters"]:
                self.configure_encounter(m["group"], m["number"], SyntheticEncounter(**e))
        self.write_memory(0x02000000, ewram)
        self.write_memory(0x03000000, iwram)
        self._frame_count, self._keys, self._previous_keys = int(data["frame"]), int(data["keys"]), int(data["previous_keys"])
        self._menu, self._action_delay = data["menu"], int(data["action_delay"])
        movement = data["movement"]
        self._movement = (int(movement[0]), int(movement[1]), int(movement[2])) if movement else None
        self._movement_left, self._mode = int(data["movement_left"]), data["mode"]
        self._relocations, self._map_steps = int(data["relocations"]), int(data["map_steps"])
        rng = data["rng"]
        self._rng.setstate((rng[0], tuple(rng[1]), rng[2]))
        self.events = data["events"]
        self.moves = {m["move_id"]: SyntheticMove(**m) for m in data["moves"]}
        self._active_encounter = SyntheticEncounter(**data["active_encounter"]) if data["active_encounter"] else None
        self._active_party_index = int(data["active_party_index"])
        self.get_save_block1_dynamic_ptr()
        self.get_save_block2_dynamic_ptr()

    def close(self) -> None:
        self._closed = True
        self._keys = 0
