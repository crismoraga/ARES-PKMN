"""Read the Gen 4 battle tables straight from the user's NDS ROM (NitroFS + NARC + encrypted messages).

Used once to export ``ares_u/gen4/gen4_dex.json`` (``python -m ares_u.gen4.romdata``); the adapter only reads
the JSON, so the ROM is not needed at run time. Sources inside the ROM:

* move data (16 bytes per move: effect, category, power, type, accuracy, PP, chance, target, priority):
  Platino ``poketool/waza/pl_waza_tbl.narc``, HeartGold ``a/0/1/1``;
* personal data (44 bytes per species: base stats, type 1/2, ...): Platino ``poketool/personal/pl_personal.narc``,
  HeartGold ``a/0/0/2``;
* species and move names: the message archive (``msgdata/pl_msg.narc`` / ``a/0/2/7``), the file whose
  strings 387/33 decode to TURTWIG/PLACAJE etc. Names are Spanish (the ROMs are Spanish); English move names
  come from the Gen 3 table (``data/stadium_dex.json``, moves 1-354) plus a hand-entered list for 355-467.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from ares_u.gal.games import GAMES
from ares_u.gen4.pkm import decode_text

ROOT = Path(__file__).resolve().parents[2]
DEX_PATH = Path(__file__).resolve().parent / "gen4_dex.json"
PATHS = {"platino": {"moves": "poketool/waza/pl_waza_tbl.narc", "personal": "poketool/personal/pl_personal.narc",
                     "msg": "msgdata/pl_msg.narc"},
         "heartgold": {"moves": "a/0/1/1", "personal": "a/0/0/2", "msg": "a/0/2/7"}}
TYPE_NAMES = ("normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel", "???",
              "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark")
CATEGORIES = ("physical", "special", "status")
# English names of the Gen 4 moves (ids 355-467), in id order.
GEN4_MOVES_EN = (
    "ROOST", "GRAVITY", "MIRACLE EYE", "WAKE-UP SLAP", "HAMMER ARM", "GYRO BALL", "HEALING WISH", "BRINE",
    "NATURAL GIFT", "FEINT", "PLUCK", "TAILWIND", "ACUPRESSURE", "METAL BURST", "U-TURN", "CLOSE COMBAT",
    "PAYBACK", "ASSURANCE", "EMBARGO", "FLING", "PSYCHO SHIFT", "TRUMP CARD", "HEAL BLOCK", "WRING OUT",
    "POWER TRICK", "GASTRO ACID", "LUCKY CHANT", "ME FIRST", "COPYCAT", "POWER SWAP", "GUARD SWAP", "PUNISHMENT",
    "LAST RESORT", "WORRY SEED", "SUCKER PUNCH", "TOXIC SPIKES", "HEART SWAP", "AQUA RING", "MAGNET RISE",
    "FLARE BLITZ", "FORCE PALM", "AURA SPHERE", "ROCK POLISH", "POISON JAB", "DARK PULSE", "NIGHT SLASH",
    "AQUA TAIL", "SEED BOMB", "AIR SLASH", "X-SCISSOR", "BUG BUZZ", "DRAGON PULSE", "DRAGON RUSH", "POWER GEM",
    "DRAIN PUNCH", "VACUUM WAVE", "FOCUS BLAST", "ENERGY BALL", "BRAVE BIRD", "EARTH POWER", "SWITCHEROO",
    "GIGA IMPACT", "NASTY PLOT", "BULLET PUNCH", "AVALANCHE", "ICE SHARD", "SHADOW CLAW", "THUNDER FANG",
    "ICE FANG", "FIRE FANG", "SHADOW SNEAK", "MUD BOMB", "PSYCHO CUT", "ZEN HEADBUTT", "MIRROR SHOT",
    "FLASH CANNON", "ROCK CLIMB", "DEFOG", "TRICK ROOM", "DRACO METEOR", "DISCHARGE", "LAVA PLUME", "LEAF STORM",
    "POWER WHIP", "ROCK WRECKER", "CROSS POISON", "GUNK SHOT", "IRON HEAD", "MAGNET BOMB", "STONE EDGE",
    "CAPTIVATE", "STEALTH ROCK", "GRASS KNOT", "CHATTER", "JUDGMENT", "BUG BITE", "CHARGE BEAM", "WOOD HAMMER",
    "AQUA JET", "ATTACK ORDER", "DEFEND ORDER", "HEAL ORDER", "HEAD SMASH", "DOUBLE HIT", "ROAR OF TIME",
    "SPACIAL REND", "LUNAR DANCE", "CRUSH GRIP", "MAGMA STORM", "DARK VOID", "SEED FLARE", "OMINOUS WIND",
    "SHADOW FORCE")
assert len(GEN4_MOVES_EN) == 467 - 355 + 1


class NitroFS:
    """Minimal NDS file system reader (FNT + FAT)."""

    def __init__(self, rom: bytes) -> None:
        self.rom = rom
        fnt_off, _fnt_size, fat_off, fat_size = struct.unpack_from("<IIII", rom, 0x40)
        self.fat = [struct.unpack_from("<II", rom, fat_off + 8 * i) for i in range(fat_size // 8)]
        self.paths: dict[str, int] = {}
        self._walk(fnt_off, 0xF000, "")

    def _walk(self, fnt: int, dir_id: int, prefix: str) -> None:
        sub_off, first_id, _ = struct.unpack_from("<IHH", self.rom, fnt + (dir_id & 0xFFF) * 8)
        pos, file_id = fnt + sub_off, first_id
        while True:
            head = self.rom[pos]
            pos += 1
            if head == 0:
                break
            name = self.rom[pos:pos + (head & 0x7F)].decode("latin-1")
            pos += head & 0x7F
            if head & 0x80:
                (child,) = struct.unpack_from("<H", self.rom, pos)
                pos += 2
                self._walk(fnt, child, prefix + name + "/")
            else:
                self.paths[prefix + name] = file_id
                file_id += 1

    def read(self, path: str) -> bytes:
        start, end = self.fat[self.paths[path]]
        return self.rom[start:end]


def narc_files(blob: bytes) -> list[bytes]:
    if blob[:4] != b"NARC":
        raise ValueError("not a NARC")
    header_size = struct.unpack_from("<H", blob, 12)[0]
    pos = header_size
    if blob[pos:pos + 4] != b"BTAF":
        raise ValueError("missing BTAF")
    btaf_size, count = struct.unpack_from("<IH", blob, pos + 4)
    entries = [struct.unpack_from("<II", blob, pos + 12 + 8 * i) for i in range(count)]
    pos += btaf_size
    btnf_size = struct.unpack_from("<I", blob, pos + 4)[0]
    pos += btnf_size
    if blob[pos:pos + 4] != b"GMIF":
        raise ValueError("missing GMIF")
    data = pos + 8
    return [blob[data + start:data + end] for start, end in entries]


def decrypt_messages(blob: bytes) -> list[list[int]]:
    """Gen 4 message file -> list of strings as lists of u16 character codes."""
    count, seed = struct.unpack_from("<HH", blob, 0)
    key = (seed * 0x2FD) & 0xFFFF
    strings = []
    for index in range(count):
        k = (key * (index + 1)) & 0xFFFF
        k32 = k | (k << 16)
        offset, length = struct.unpack_from("<II", blob, 4 + 8 * index)
        offset, length = offset ^ k32, length ^ k32
        char_key = (0x91BD3 * (index + 1)) & 0xFFFF
        chars = []
        for position in range(length):
            if offset + 2 * position + 2 > len(blob):
                break
            (code,) = struct.unpack_from("<H", blob, offset + 2 * position)
            chars.append(code ^ char_key)
            char_key = (char_key + 0x493D) & 0xFFFF
        strings.append(chars)
    return strings


def _text(chars: list[int]) -> str:
    special = {0x01BB: "♂", 0x01BC: "♀", 0x01B3: "'", 0x01B4: "'"}
    out = []
    for code in chars:
        if code == 0xFFFF:
            break
        out.append(special.get(code) or decode_text(code.to_bytes(2, "little")))
    return "".join(out)


def extract(game: str) -> dict[str, Any]:
    fs = NitroFS(GAMES[game].rom.read_bytes())
    paths = PATHS[game]
    moves = []
    for move_id, raw in enumerate(narc_files(fs.read(paths["moves"]))[:468]):
        effect, category, power, mtype, accuracy, pp, chance, target, priority, flags = struct.unpack_from(
            "<HBBBBBBHbB", raw, 0)
        moves.append({"id": move_id, "effect": effect, "category": CATEGORIES[category] if category < 3 else "?",
                      "power": power, "type": TYPE_NAMES[mtype] if mtype < 18 else "?", "accuracy": accuracy,
                      "pp": pp, "chance": chance, "target": target, "priority": priority, "flags": flags})
    species = []
    for dex, raw in enumerate(narc_files(fs.read(paths["personal"]))[:494]):
        stats = list(raw[0:6])
        species.append({"id": dex, "base_stats": stats, "types": [TYPE_NAMES[raw[6]], TYPE_NAMES[raw[7]]],
                        "abilities": [raw[0x16], raw[0x17]]})
    names: dict[str, list[str]] = {}
    for blob in narc_files(fs.read(paths["msg"])):
        try:
            strings = decrypt_messages(blob)
        except struct.error:
            continue
        if len(strings) > 494 and _text(strings[387]).upper() == "TURTWIG" and "species" not in names:
            names["species"] = [_text(s) for s in strings[:494]]
        if 468 <= len(strings) <= 480 and _text(strings[33]).upper() == "PLACAJE" and "moves" not in names:
            names["moves"] = [_text(s) for s in strings[:468]]
    return {"moves": moves, "species": species, "names_es": names}


def build_dex() -> dict[str, Any]:
    """Platino tables + English move names; cross-checked against HeartGold and the Gen 3 snapshot."""
    platino = extract("platino")
    heartgold = extract("heartgold")
    stadium = json.loads((ROOT / "data" / "stadium_dex.json").read_text(encoding="utf-8"))
    english = [str(row[0]) for row in stadium["moves"]] + list(GEN4_MOVES_EN)
    moves = []
    for move in platino["moves"]:
        name_es = platino["names_es"].get("moves", [""] * 468)[move["id"]]
        moves.append([english[move["id"]], name_es, move["type"], move["power"], move["accuracy"], move["pp"],
                      move["category"], move["priority"], move["effect"]])
    species = []
    for entry in platino["species"]:
        name = platino["names_es"].get("species", [""] * 494)[entry["id"]]
        species.append([name, entry["types"][0], entry["types"][1], entry["base_stats"]])
    mismatched_moves = [m["id"] for m, h in zip(platino["moves"], heartgold["moves"])
                        if (m["power"], m["type"], m["accuracy"], m["pp"], m["category"]) !=
                        (h["power"], h["type"], h["accuracy"], h["pp"], h["category"])]
    mismatched_species = [s["id"] for s, h in zip(platino["species"], heartgold["species"]) if s["types"] != h["types"]]
    return {"version": 1, "source": "Platino ES ROM (pl_waza_tbl.narc, pl_personal.narc, pl_msg.narc); English move "
            "names: data/stadium_dex.json (1-354) + GEN4_MOVES_EN (355-467, hand-entered)",
            "move_fields": ["name_en", "name_es", "type", "power", "accuracy", "pp", "category", "priority", "effect"],
            "species_fields": ["name", "type1", "type2", "base_stats(hp,atk,def,spe,spa,spd)"],
            "moves": moves, "species": species,
            "crosscheck": {"heartgold_move_mismatches": mismatched_moves,
                           "heartgold_species_type_mismatches": mismatched_species}}


def main() -> None:
    dex = build_dex()
    DEX_PATH.write_text(json.dumps(dex, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(DEX_PATH, len(dex["moves"]), len(dex["species"]), dex["crosscheck"])
    for move_id in (1, 33, 110, 355, 394, 467):
        print(move_id, dex["moves"][move_id])
    for dex_id in (1, 252, 387, 390, 493):
        print(dex_id, dex["species"][dex_id])


if __name__ == "__main__":
    main()
