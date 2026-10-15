"""Canonical little-endian Generation III memory layouts and party encryption.

The boxed portion is preserved byte for byte when decrypting and re-encrypting:
unknown header bytes and packed ribbons, origins and ability flags are retained.
"""
from __future__ import annotations

import ctypes
import struct
from typing import Any


class BattlePokemonStruct(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("species", ctypes.c_uint16), ("attack", ctypes.c_uint16),
        ("defense", ctypes.c_uint16), ("speed", ctypes.c_uint16),
        ("spAttack", ctypes.c_uint16), ("spDefense", ctypes.c_uint16),
        ("moves", ctypes.c_uint16 * 4), ("ivs_and_flags", ctypes.c_uint32),
        ("statStages", ctypes.c_uint8 * 8), ("ability", ctypes.c_uint8),
        ("type1", ctypes.c_uint8), ("type2", ctypes.c_uint8),
        ("unknown_23", ctypes.c_uint8), ("pp", ctypes.c_uint8 * 4),
        ("hp", ctypes.c_uint16), ("level", ctypes.c_uint8),
        ("friendship", ctypes.c_uint8), ("maxHP", ctypes.c_uint16),
        ("item", ctypes.c_uint16), ("nickname", ctypes.c_uint8 * 11),
        ("ppBonuses", ctypes.c_uint8), ("otName", ctypes.c_uint8 * 8),
        ("experience", ctypes.c_uint32), ("personality", ctypes.c_uint32),
        ("status1", ctypes.c_uint32), ("status2", ctypes.c_uint32),
        ("otId", ctypes.c_uint32),
    ]


class PartyPokemonStruct(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("personality", ctypes.c_uint32), ("otId", ctypes.c_uint32),
        ("nickname", ctypes.c_uint8 * 10), ("language", ctypes.c_uint8),
        ("egg_bad_flags", ctypes.c_uint8), ("otName", ctypes.c_uint8 * 7),
        ("markings", ctypes.c_uint8), ("checksum", ctypes.c_uint16),
        ("unknown_1e", ctypes.c_uint16), ("encrypted_data", ctypes.c_uint32 * 12),
        ("status", ctypes.c_uint32), ("level", ctypes.c_uint8),
        ("mail", ctypes.c_uint8), ("hp", ctypes.c_uint16),
        ("maxHP", ctypes.c_uint16), ("attack", ctypes.c_uint16),
        ("defense", ctypes.c_uint16), ("speed", ctypes.c_uint16),
        ("spAttack", ctypes.c_uint16), ("spDefense", ctypes.c_uint16),
    ]


class BackupMapLayoutStruct(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [("width", ctypes.c_int32), ("height", ctypes.c_int32),
                ("map_ptr", ctypes.c_uint32)]


class ObjectEventStruct(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("flags", ctypes.c_uint32), ("spriteId", ctypes.c_uint8),
        ("graphicsId", ctypes.c_uint8), ("movementType", ctypes.c_uint8),
        ("trainerType", ctypes.c_uint8), ("localId", ctypes.c_uint8),
        ("mapNum", ctypes.c_uint8), ("mapGroup", ctypes.c_uint8),
        ("elevations", ctypes.c_uint8), ("initialCoords_x", ctypes.c_int16),
        ("initialCoords_y", ctypes.c_int16), ("currentCoords_x", ctypes.c_int16),
        ("currentCoords_y", ctypes.c_int16), ("previousCoords_x", ctypes.c_int16),
        ("previousCoords_y", ctypes.c_int16), ("facingDirection", ctypes.c_uint8),
        ("rangeX_Y", ctypes.c_uint8), ("field_1A", ctypes.c_uint8),
        ("field_1B", ctypes.c_uint8), ("movementActionId", ctypes.c_uint8),
        ("trainerRange", ctypes.c_uint8), ("field_1E", ctypes.c_uint8),
        ("field_1F", ctypes.c_uint8), ("field_20", ctypes.c_uint8),
        ("field_21", ctypes.c_uint8), ("field_22", ctypes.c_uint8),
        ("field_23", ctypes.c_uint8),
    ]


# Order is fixed by GetSubstruct(personality % 24), not lexicographic sorting.
SUBSTRUCT_ORDER_TABLE = (
    "GAEM", "GAME", "GEAM", "GEMA", "GMAE", "GMEA",
    "AGEM", "AGME", "AEGM", "AEMG", "AMGE", "AMEG",
    "EGAM", "EGMA", "EAGM", "EAMG", "EMGA", "EMAG",
    "MGAE", "MGEA", "MAGE", "MAEG", "MEGA", "MEAG",
)


def decrypt_party_pokemon(mon: PartyPokemonStruct) -> dict[str, Any]:
    """Decode a party record; report bad checksums without hiding corruption."""
    raw = bytes(mon)
    pid, otid = struct.unpack_from("<II", raw)
    key = pid ^ otid
    plain = b"".join(struct.pack("<I", word ^ key)
                     for word in struct.unpack("<12I", raw[32:80]))
    blocks = {kind: plain[i * 12:(i + 1) * 12]
              for i, kind in enumerate(SUBSTRUCT_ORDER_TABLE[pid % 24])}
    g, a, e, m = (blocks[k] for k in "GAEM")
    species, item, xp, pp_bonus, friendship = struct.unpack_from("<HHIBB", g)
    moves = list(struct.unpack_from("<4H", a))
    iv_word = struct.unpack_from("<I", m, 4)[0]
    return {
        "valid_checksum": sum(struct.unpack("<24H", plain)) & 65535 == int(mon.checksum),
        "pid": pid, "otid": otid, "species": species, "held_item": item,
        "experience": xp, "friendship": friendship, "level": int(mon.level),
        "hp": int(mon.hp), "maxHP": int(mon.maxHP), "attack": int(mon.attack),
        "defense": int(mon.defense), "speed": int(mon.speed),
        "spAttack": int(mon.spAttack), "spDefense": int(mon.spDefense),
        "status": int(mon.status), "moves": moves, "pp": list(a[8:12]),
        "evs": list(e[:6]), "ivs": [(iv_word >> (i * 5)) & 31 for i in range(6)],
        "pp_bonuses": pp_bonus, "nickname": raw[8:18], "language": raw[18],
        "flags": raw[19], "ot_name": raw[20:27], "markings": raw[27],
        "mail": int(mon.mail), "unknown": int(mon.unknown_1e),
        "is_egg": bool(iv_word & (1 << 30)), "ability_num": iv_word >> 31,
        "_blocks": blocks,
    }


def _bounded(value: Any, maximum: int, name: str) -> int:
    n = int(value)
    if not 0 <= n <= maximum:
        raise ValueError(f"{name} outside 0..{maximum}: {n}")
    return n


def _vector(data: dict[str, Any], name: str, count: int, maximum: int) -> list[int]:
    raw_values = data.get(name, [0] * count)
    if isinstance(raw_values, dict) and name in {"evs", "ivs"}:
        values = [raw_values.get(key, 0) for key in ("hp", "atk", "def", "spd", "spa", "spd_def")]
    else:
        values = list(raw_values)
    if len(values) != count:
        raise ValueError(f"{name} requires {count} values")
    return [_bounded(v, maximum, name) for v in values]


def encrypt_party_pokemon(data: dict[str, Any]) -> PartyPokemonStruct:
    """Encode GAEM with validation, preserving opaque metadata when supplied."""
    pid = _bounded(data.get("pid", 0), 0xFFFFFFFF, "pid")
    otid = _bounded(data.get("otid", 0), 0xFFFFFFFF, "otid")
    source = data.get("_blocks", {})
    blocks = {k: bytearray(source.get(k, bytes(12))) for k in "GAEM"}
    if any(len(v) != 12 for v in blocks.values()):
        raise ValueError("each GAEM block must contain 12 bytes")
    struct.pack_into("<HHIBB", blocks["G"], 0,
                     _bounded(data.get("species", 0), 65535, "species"),
                     _bounded(data.get("held_item", 0), 65535, "held_item"),
                     _bounded(data.get("experience", 0), 0xFFFFFFFF, "experience"),
                     _bounded(data.get("pp_bonuses", 0), 255, "pp_bonuses"),
                     _bounded(data.get("friendship", 70), 255, "friendship"))
    struct.pack_into("<4H4B", blocks["A"], 0, *_vector(data, "moves", 4, 65535),
                     *_vector(data, "pp", 4, 255))
    blocks["E"][:6] = bytes(_vector(data, "evs", 6, 255))
    old_flags = struct.unpack_from("<I", blocks["M"], 4)[0] & 0xC0000000
    egg = int(bool(data.get("is_egg", old_flags & (1 << 30))))
    ability = _bounded(data.get("ability_num", old_flags >> 31), 1, "ability_num")
    ivs = sum(v << (i * 5) for i, v in enumerate(_vector(data, "ivs", 6, 31)))
    struct.pack_into("<I", blocks["M"], 4, ivs | egg << 30 | ability << 31)
    plain = b"".join(blocks[k] for k in SUBSTRUCT_ORDER_TABLE[pid % 24])
    raw = bytearray(100)
    struct.pack_into("<II", raw, 0, pid, otid)
    for name, offset, size in (("nickname", 8, 10), ("ot_name", 20, 7)):
        value = bytes(data.get(name, b""))
        if len(value) > size:
            raise ValueError(f"{name} too long")
        raw[offset:offset + size] = value.ljust(size, b"\xff")
    raw[18] = _bounded(data.get("language", 2), 255, "language")
    raw[19] = _bounded(data.get("flags", 2), 255, "flags")
    raw[27] = _bounded(data.get("markings", 0), 255, "markings")
    struct.pack_into("<HH", raw, 28, sum(struct.unpack("<24H", plain)) & 65535,
                     _bounded(data.get("unknown", 0), 65535, "unknown"))
    key = pid ^ otid
    raw[32:80] = b"".join(struct.pack("<I", word ^ key)
                          for word in struct.unpack("<12I", plain))
    struct.pack_into("<IBB7H", raw, 80,
                     _bounded(data.get("status", 0), 0xFFFFFFFF, "status"),
                     _bounded(data.get("level", 1), 100, "level"),
                     _bounded(data.get("mail", 255), 255, "mail"),
                     *[_bounded(data.get(k, 1), 65535, k) for k in
                       ("hp", "maxHP", "attack", "defense", "speed", "spAttack", "spDefense")])
    return PartyPokemonStruct.from_buffer_copy(raw)
