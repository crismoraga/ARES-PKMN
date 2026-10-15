"""All GAEM permutations, a manually constructed oracle and corruption checks."""
from __future__ import annotations

import random
import struct
from typing import Any

import pytest

from core.bridge.gba_structs import (
    PartyPokemonStruct, SUBSTRUCT_ORDER_TABLE, decrypt_party_pokemon, encrypt_party_pokemon,
)


@pytest.mark.parametrize("permutation", range(24))
def test_all_permutations_and_opaque_metadata(permutation: int) -> None:
    rng = random.Random(permutation)
    blocks = {kind: rng.randbytes(12) for kind in "GAEM"}
    data: dict[str, Any] = {
        "pid": 24 * 9831 + permutation, "otid": 0x98765432,
        "species": 277, "held_item": 234, "experience": 54321,
        "friendship": 151, "level": 27, "hp": 54, "maxHP": 93,
        "attack": 65, "defense": 66, "speed": 67, "spAttack": 68, "spDefense": 69,
        "status": 64, "moves": [33, 55, 98, 237], "pp": [3, 7, 11, 13],
        "evs": [1, 2, 3, 4, 5, 6], "ivs": [31, 0, 17, 9, 21, 4],
        "nickname": bytes(range(10)), "ot_name": bytes(range(7)), "language": 2,
        "flags": 7, "markings": 14, "mail": 19, "unknown": 0xABCD,
        "is_egg": True, "ability_num": 1, "pp_bonuses": 0xA5, "_blocks": blocks,
    }
    encoded = encrypt_party_pokemon(data)
    decoded = decrypt_party_pokemon(encoded)
    assert decoded["valid_checksum"]
    for key in ("pid", "otid", "species", "held_item", "experience", "friendship", "level",
                "hp", "maxHP", "attack", "defense", "speed", "spAttack", "spDefense",
                "status", "moves", "pp", "evs", "ivs", "nickname", "ot_name", "language",
                "flags", "markings", "mail", "unknown", "is_egg", "ability_num", "pp_bonuses"):
        assert decoded[key] == data[key]
    assert bytes(encrypt_party_pokemon(decoded)) == bytes(encoded)
    assert decoded["_blocks"]["G"][10:] == blocks["G"][10:]
    assert decoded["_blocks"]["E"][6:] == blocks["E"][6:]
    assert decoded["_blocks"]["M"][:4] == blocks["M"][:4]
    assert decoded["_blocks"]["M"][8:] == blocks["M"][8:]


def test_independent_cipher_fixture() -> None:
    # PID=1 selects GAME. Construct ciphertext without calling our encoder.
    pid, otid = 1, 0x12345678
    g = struct.pack("<HHIBBH", 25, 13, 10000, 0x55, 100, 0xABCD)
    a = struct.pack("<4H4B", 85, 98, 33, 45, 15, 20, 35, 40)
    e = bytes(range(12))
    m = struct.pack("<BBHII", 7, 3, 0x5555, 0xFFFFFFFF, 0x6A987654)
    plain = g + a + m + e
    raw = bytearray(100)
    struct.pack_into("<II", raw, 0, pid, otid)
    struct.pack_into("<H", raw, 28, sum(struct.unpack("<24H", plain)) & 65535)
    for index in range(12):
        word = int.from_bytes(plain[index * 4:index * 4 + 4], "little")
        raw[32 + index * 4:36 + index * 4] = (word ^ pid ^ otid).to_bytes(4, "little")
    struct.pack_into("<IBB7H", raw, 80, 8, 30, 255, 71, 80, 55, 42, 90, 70, 65)
    decoded = decrypt_party_pokemon(PartyPokemonStruct.from_buffer_copy(raw))
    assert decoded["species"] == 25
    assert decoded["held_item"] == 13
    assert decoded["moves"] == [85, 98, 33, 45]
    assert decoded["evs"] == [0, 1, 2, 3, 4, 5]
    assert decoded["ivs"] == [31] * 6
    assert decoded["valid_checksum"]
    assert bytes(encrypt_party_pokemon(decoded)) == bytes(raw)


def test_corruption_and_bounds() -> None:
    encoded = encrypt_party_pokemon({"species": 1, "pid": 24, "otid": 1234})
    raw = bytearray(bytes(encoded))
    raw[40] ^= 1
    assert not decrypt_party_pokemon(PartyPokemonStruct.from_buffer_copy(raw))["valid_checksum"]
    assert len(set(SUBSTRUCT_ORDER_TABLE)) == 24
    for data in ({"ivs": [32] * 6}, {"pp": [1]}, {"pid": -1}, {"level": 101}):
        with pytest.raises(ValueError):
            encrypt_party_pokemon(data)
