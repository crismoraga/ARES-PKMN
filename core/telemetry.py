"""Read-only, display-oriented decoding of public player state for monitoring.

Everything here is visible to a human player (party summary, trainer card,
Pokedex counters, the opponent's species, level and HP bar). Nothing is written
to RAM and nothing here feeds a policy decision.
"""
from __future__ import annotations

import struct
from typing import Any

from core.bridge.ffi_mgba import GBABridgeProtocol
from core.game_data import TYPE_NAMES, RomCatalog, decode_gba_text

# SaveBlock offsets from the pinned pret sources (include/global.h).
_MONEY_OFFSET = {"emerald": 0x490, "firered": 0x290}
_DEX_OWNED, _DEX_SEEN, _DEX_BYTES = 0x28, 0x5C, 52
_PLAY_TIME = 0x0E
_BADGE_FLAGS = {"emerald": 0x867, "firered": 0x820}
_STATUS_NAMES = ((0x08, "PSN"), (0x10, "BRN"), (0x20, "FRZ"), (0x40, "PAR"), (0x80, "TOX"))


def status_name(status: int) -> str:
    """Primary non-volatile status as the summary screen abbreviates it."""
    if status & 0x7:
        return "SLP"
    for mask, name in _STATUS_NAMES:
        if status & mask:
            return name
    return ""


def _type_names(types: tuple[int, int]) -> list[str]:
    return [TYPE_NAMES[value] if 0 <= value < len(TYPE_NAMES) else str(value) for value in dict.fromkeys(types)]


def species_name(catalog: RomCatalog | None, species: int) -> str:
    if catalog is not None and 0 < species < catalog.species_count:
        try:
            return catalog.species(species).name
        except ValueError:
            pass
    return f"#{species}"


def move_name(catalog: RomCatalog | None, move: int) -> str:
    if catalog is not None and 0 < move < catalog.move_count:
        try:
            return catalog.move(move).name
        except ValueError:
            pass
    return f"#{move}"


def describe_mon(mon: dict[str, Any], catalog: RomCatalog | None) -> dict[str, object]:
    """One party member for dashboards: names, HP, status, moves and EXP progress."""
    species = int(mon["species"])
    level = int(mon["level"])
    nickname = mon.get("nickname", b"")
    result: dict[str, object] = {
        "pid": int(mon["pid"]), "species": species, "name": species_name(catalog, species),
        "nickname": decode_gba_text(bytes(nickname)) if isinstance(nickname, (bytes, bytearray)) else "",
        "level": level, "hp": int(mon["hp"]), "maxHP": int(mon["maxHP"]),
        "status": status_name(int(mon.get("status", 0))), "experience": int(mon.get("experience", 0)),
        "held_item": int(mon.get("held_item", 0)), "is_egg": bool(mon.get("is_egg", False)),
        "stats": {key: int(mon.get(key, 0)) for key in ("attack", "defense", "speed", "spAttack", "spDefense")},
    }
    moves: list[dict[str, object]] = []
    for move, pp in zip(mon.get("moves", []), mon.get("pp", [])):
        move_id = int(move)
        if not move_id:
            continue
        entry: dict[str, object] = {"id": move_id, "name": move_name(catalog, move_id), "pp": int(pp)}
        if catalog is not None and move_id < catalog.move_count:
            info = catalog.move(move_id)
            entry.update(max_pp=info.pp, power=info.power, accuracy=info.accuracy,
                         type=TYPE_NAMES[info.move_type] if info.move_type < len(TYPE_NAMES) else str(info.move_type),
                         category=info.category)
        moves.append(entry)
    result["moves"] = moves
    if catalog is not None and 0 < species < catalog.species_count:
        try:
            species_info = catalog.species(species)
            result["types"] = _type_names(species_info.types)
            low = catalog.experience_at_level(species, max(1, level))
            high = catalog.experience_at_level(species, min(100, level + 1)) if level < 100 else low
            result["exp_level_floor"], result["exp_next_level"] = low, high
        except ValueError:
            pass
    return result


def trainer_card(bridge: GBABridgeProtocol, game: str) -> dict[str, object]:
    """Money, badges, Pokedex counters and in-game clock from the save blocks."""
    memory = bridge.memory_map
    save1, save2 = bridge.get_save_block1_dynamic_ptr(), bridge.get_save_block2_dynamic_ptr()
    key = bridge.read_u32(save2 + memory.encryption_key_offset)
    money = bridge.read_u32(save1 + _MONEY_OFFSET[game]) ^ key
    owned = bridge.read_memory(save2 + _DEX_OWNED, _DEX_BYTES)
    seen = bridge.read_memory(save2 + _DEX_SEEN, _DEX_BYTES)
    hours, minutes, seconds = struct.unpack("<HBB", bridge.read_memory(save2 + _PLAY_TIME, 4))
    base = _BADGE_FLAGS[game]
    badges = [bridge.get_event_flag(base + index) for index in range(8)]
    return {"money": money if money <= 999999 else None,
            "dex_owned": sum(bin(value).count("1") for value in owned),
            "dex_seen": sum(bin(value).count("1") for value in seen),
            "play_time": f"{hours}:{minutes:02d}:{seconds:02d}", "play_seconds": hours * 3600 + minutes * 60 + seconds,
            "badges": badges, "badge_count": sum(badges)}


def describe_battle(bridge: GBABridgeProtocol, catalog: RomCatalog | None) -> dict[str, object] | None:
    """Both visible active battlers. The rival's HP is shown as its on-screen bar fraction."""
    battle, trainer = bridge.is_in_battle()
    if not battle:
        return None
    own, enemy = bridge.get_battle_pokemon(0), bridge.get_battle_pokemon(1)
    if not own.species or not enemy.species:
        return {"trainer": trainer, "own": None, "enemy": None}
    return {"trainer": trainer,
            "own": {"species": int(own.species), "name": species_name(catalog, int(own.species)),
                    "level": int(own.level), "hp": int(own.hp), "maxHP": int(own.maxHP),
                    "status": status_name(int(own.status1))},
            "enemy": {"species": int(enemy.species), "name": species_name(catalog, int(enemy.species)),
                      "level": int(enemy.level), "hp_fraction": round(int(enemy.hp) / max(1, int(enemy.maxHP)), 4),
                      "status": status_name(int(enemy.status1)),
                      "types": _type_names((int(enemy.type1), int(enemy.type2)))}}
