"""Layout verification against the packed Gen III ABI, including every offset."""
from __future__ import annotations

import ctypes
from pathlib import Path

import pytest
import yaml

from core.bridge.ffi_mgba import EMERALD_MEMORY, FIRERED_MEMORY, verify_symbol_map
from core.bridge.gba_structs import (
    BackupMapLayoutStruct, BattlePokemonStruct, ObjectEventStruct, PartyPokemonStruct,
)


def test_battle_layout_all_offsets() -> None:
    expected = {"species": 0, "attack": 2, "defense": 4, "speed": 6,
                "spAttack": 8, "spDefense": 10, "moves": 12, "ivs_and_flags": 20,
                "statStages": 24, "ability": 32, "type1": 33, "type2": 34,
                "unknown_23": 35, "pp": 36, "hp": 40, "level": 42, "friendship": 43,
                "maxHP": 44, "item": 46, "nickname": 48, "ppBonuses": 59,
                "otName": 60, "experience": 68, "personality": 72, "status1": 76,
                "status2": 80, "otId": 84}
    assert ctypes.sizeof(BattlePokemonStruct) == 88
    for name, offset in expected.items():
        assert getattr(BattlePokemonStruct, name).offset == offset


def test_party_map_object_layout() -> None:
    assert ctypes.sizeof(PartyPokemonStruct) == 100
    expected = {"personality": 0, "otId": 4, "nickname": 8, "language": 18,
                "egg_bad_flags": 19, "otName": 20, "markings": 27, "checksum": 28,
                "unknown_1e": 30, "encrypted_data": 32, "status": 80, "level": 84,
                "mail": 85, "hp": 86, "maxHP": 88, "attack": 90, "defense": 92,
                "speed": 94, "spAttack": 96, "spDefense": 98}
    for name, offset in expected.items():
        assert getattr(PartyPokemonStruct, name).offset == offset
    assert ctypes.sizeof(BackupMapLayoutStruct) == 12
    assert BackupMapLayoutStruct.map_ptr.offset == 8
    assert ctypes.sizeof(ObjectEventStruct) == 36
    assert ObjectEventStruct.currentCoords_x.offset == 16
    assert ObjectEventStruct.currentCoords_y.offset == 18
    assert ObjectEventStruct.elevations.offset == 11
    assert ObjectEventStruct.movementActionId.offset == 28
    assert ObjectEventStruct.trainerRange.offset == 29


@pytest.mark.parametrize("emerald", [False, True])
def test_pinned_symbol_and_yaml_consistency(emerald: bool) -> None:
    root = Path(__file__).resolve().parents[1]
    memory = EMERALD_MEMORY if emerald else FIRERED_MEMORY
    name = "emerald_usa" if emerald else "firered_v1_usa"
    symbols = verify_symbol_map(memory, root / "data" / ("pokeemerald.sym" if emerald else "pokefirered.sym"))
    config = yaml.safe_load((root / "config" / f"{name}.yaml").read_text())
    assert config["rom_sha1"] == memory.rom_sha1
    assert config["official_symbols"] == symbols
    assert config["map_border"] == {"left": 7, "top": 7, "right": 8, "bottom": 7}


def test_invalid_symbol_artifact_rejected(tmp_path: Path) -> None:
    fake = tmp_path / "wrong.sym"
    fake.write_text("0300500c g 00000004 gSaveBlock1Ptr\n")
    with pytest.raises(ValueError, match="Pinned symbol mismatch"):
        verify_symbol_map(FIRERED_MEMORY, fake)
