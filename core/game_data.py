"""Read verified generation-three gameplay metadata directly from user ROMs.

The catalog contains species, moves and learnsets, never hidden runtime opponent
information. Type-effectiveness ground truth is explicitly an oracle for test and
synthetic scenario generation, not an input to the epistemic controller.
"""
from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

Game = Literal["firered", "emerald"]
ROOT = Path(__file__).resolve().parents[1]
ROM_SHA1: dict[Game, str] = {
    "firered": "41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc",
    "emerald": "f3ae088181bf583e55daf962a92bb46f4f1d07b7",
}
ROM_NAMES: dict[Game, str] = {
    "firered": "Pokemon - FireRed Version (USA).gba",
    "emerald": "Pokemon - Emerald Version (USA, Europe).gba",
}
TYPE_NAMES = ("Normal", "Fighting", "Flying", "Poison", "Ground", "Rock", "Bug",
              "Ghost", "Steel", "Mystery", "Fire", "Water", "Grass", "Electric",
              "Psychic", "Ice", "Dragon", "Dark")


@dataclass(frozen=True)
class RomSymbol:
    """Absolute target address and emitted size from an official matching build."""
    address: int
    size: int


def load_symbols(game: Game) -> dict[str, RomSymbol]:
    """Parse defined GNU ARM globals and local callback symbols."""
    name = "pokefirered.sym" if game == "firered" else "pokeemerald.sym"
    result: dict[str, RomSymbol] = {}
    pattern = re.compile(r"^([0-9a-fA-F]{8})\s+\w\s+([0-9a-fA-F]{8})\s+(\S+)$")
    for line in (ROOT / "data" / name).read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            result[match[3]] = RomSymbol(int(match[1], 16), int(match[2], 16))
    if "gSpeciesInfo" not in result or "gBattleMoves" not in result:
        raise ValueError(f"Missing canonical game symbols in {name}")
    return result


def decode_gba_text(data: bytes) -> str:
    """Decode English generation-three text and stop at the native terminator."""
    punctuation = {0: " ", 0xAB: "!", 0xAC: "?", 0xAD: ".", 0xAE: "-", 0xB4: "'",
                   0xB5: "♂", 0xB6: "♀", 0xB8: ",", 0xBA: "/", 0xF0: ":", 0x1B: "é"}
    output: list[str] = []
    for value in data:
        if value == 0xFF:
            break
        if 0xA1 <= value <= 0xAA:
            output.append(chr(ord("0") + value - 0xA1))
        elif 0xBB <= value <= 0xD4:
            output.append(chr(ord("A") + value - 0xBB))
        elif 0xD5 <= value <= 0xEE:
            output.append(chr(ord("a") + value - 0xD5))
        else:
            output.append(punctuation.get(value, "�"))
    return "".join(output)


@dataclass(frozen=True)
class SpeciesInfo:
    """Native internal species identity, base statistics and capture parameters."""
    species_id: int
    name: str
    base_stats: tuple[int, int, int, int, int, int]
    types: tuple[int, int]
    catch_rate: int
    experience_yield: int
    friendship: int
    growth_rate: int
    abilities: tuple[int, int]


@dataclass(frozen=True)
class MoveInfo:
    """A ROM move record; Gen3 damage category is determined by its type."""
    move_id: int
    name: str
    effect: int
    power: int
    move_type: int
    accuracy: int
    pp: int
    secondary_chance: int
    target: int
    priority: int
    flags: int

    @property
    def category(self) -> Literal["physical", "special", "status"]:
        """Return pre-generation-four category without later-generation rules."""
        if self.power == 0:
            return "status"
        return "special" if self.move_type >= 10 else "physical"


class RomCatalog:
    """Bounds-checked immutable gameplay catalog of one hash-identified ROM."""

    def __init__(self, game: Game, rom_path: Path | None = None) -> None:
        self.game = game
        self.path = rom_path or ROOT.parent / ROM_NAMES[game]
        self.rom = self.path.read_bytes()
        digest = hashlib.sha1(self.rom).hexdigest()
        if digest != ROM_SHA1[game]:
            raise ValueError(f"Unsupported {game} ROM SHA1: {digest}")
        self.symbols = load_symbols(game)
        self.species_count = self.symbols["gSpeciesInfo"].size // 28
        self.move_count = self.symbols["gBattleMoves"].size // 12

    def read(self, address: int, size: int) -> bytes:
        """Read a complete ROM address span; never allow wraparound/truncation."""
        offset = address - 0x08000000
        if size < 0 or offset < 0 or offset + size > len(self.rom):
            raise ValueError(f"ROM span outside image: {address:#x}+{size}")
        return self.rom[offset:offset + size]

    @lru_cache(maxsize=512)
    def species(self, species_id: int) -> SpeciesInfo:
        """Decode the exact 28-byte SpeciesInfo and its 11-byte name."""
        if not 0 <= species_id < self.species_count:
            raise ValueError(f"Invalid species id {species_id}")
        data = self.read(self.symbols["gSpeciesInfo"].address + 28 * species_id, 28)
        name = decode_gba_text(self.read(self.symbols["gSpeciesNames"].address + 11 * species_id, 11))
        return SpeciesInfo(species_id, name, (data[0], data[1], data[2], data[3], data[4], data[5]),
                           (data[6], data[7]), data[8], data[9], data[18], data[19], (data[22], data[23]))

    @lru_cache(maxsize=512)
    def move(self, move_id: int) -> MoveInfo:
        """Decode the 12-byte native move record including signed priority."""
        if not 0 <= move_id < self.move_count:
            raise ValueError(f"Invalid move id {move_id}")
        data = self.read(self.symbols["gBattleMoves"].address + 12 * move_id, 12)
        name = decode_gba_text(self.read(self.symbols["gMoveNames"].address + 13 * move_id, 13))
        return MoveInfo(move_id, name, data[0], data[1], data[2], data[3], data[4], data[5],
                        data[6], int.from_bytes(data[7:8], "little", signed=True), data[8])

    def experience_at_level(self, species_id: int, level: int) -> int:
        """Use the ROM experience table, including non-cubic growth curves."""
        if not 1 <= level <= 100:
            raise ValueError("level must be 1..100")
        growth = self.species(species_id).growth_rate
        return int(struct.unpack("<I", self.read(self.symbols["gExperienceTables"].address +
                                                  4 * (101 * growth + level), 4))[0])

    def level_up_moves(self, species_id: int, level: int = 100) -> list[int]:
        """Return the chronological legal level-up moves up to a given level."""
        self.species(species_id)
        pointer = struct.unpack("<I", self.read(self.symbols["gLevelUpLearnsets"].address + 4 * species_id, 4))[0]
        result: list[int] = []
        for index in range(256):
            entry = struct.unpack("<H", self.read(pointer + 2 * index, 2))[0]
            if entry == 0xFFFF:
                return result
            if entry >> 9 <= level:
                move_id = entry & 0x1FF
                if move_id in result:
                    result.remove(move_id)
                result.append(move_id)
        raise ValueError(f"Unterminated learnset for {species_id}")

    def tmhm_move(self, item_id: int) -> int:
        """Resolve the native TM/HM item range into its taught move."""
        if not 289 <= item_id <= 346:
            raise ValueError("TM/HM item must be289..346")
        return int(struct.unpack("<H", self.read(self.symbols["sTMHMMoves"].address +
                                                  2 * (item_id - 289), 2))[0])

    def can_learn_tmhm(self, species_id: int, item_id: int) -> bool:
        """Read the native64-bit compatibility bitset, not a guessed HM carrier list."""
        self.species(species_id)
        self.tmhm_move(item_id)
        symbol = "gTMHMLearnsets" if self.game == "emerald" else "sTMHMLearnsets"
        bitset = int.from_bytes(self.read(self.symbols[symbol].address + species_id * 8, 8), "little")
        return bool(bitset & (1 << (item_id - 289)))

    def type_multiplier_oracle(self, attacking_type: int, defending_types: tuple[int, ...]) -> tuple[int, int]:
        """Read original efficacy for independent tests/scenario generation only."""
        if not 0 <= attacking_type < 18 or any(not 0 <= t < 18 for t in defending_types):
            raise ValueError("type index outside18x18")
        table = self.read(self.symbols["gTypeEffectiveness"].address, self.symbols["gTypeEffectiveness"].size)
        numerator, denominator = 1, 1
        for defending_type in dict.fromkeys(defending_types):
            for index in range(0, len(table), 3):
                atk, defense, efficacy = table[index:index + 3]
                if atk == 0xFF:
                    break
                if atk == attacking_type and defense == defending_type:
                    numerator *= efficacy
                    denominator *= 10
                    break
        return numerator, denominator

    def make_party_data(self, species_id: int, level: int, pid: int = 42, otid: int = 1001) -> dict[str, Any]:
        """Construct legal synthetic fixture stats from species and nature data."""
        species = self.species(species_id)
        stats = [((2 * base + 15) * level) // 100 + (level + 10 if i == 0 else 5)
                 for i, base in enumerate(species.base_stats)]
        nature = pid % 25
        for i in range(5):
            if nature // 5 != nature % 5:
                if i == nature // 5:
                    stats[i + 1] = stats[i + 1] * 110 // 100
                if i == nature % 5:
                    stats[i + 1] = stats[i + 1] * 90 // 100
        moves = self.level_up_moves(species_id, level)[-4:]
        moves += [0] * (4 - len(moves))
        return {"pid": pid, "otid": otid, "species": species_id, "held_item": 0,
                "experience": self.experience_at_level(species_id, level), "friendship": species.friendship,
                "level": level, "hp": stats[0], "maxHP": stats[0], "attack": stats[1],
                "defense": stats[2], "speed": stats[3], "spAttack": stats[4], "spDefense": stats[5],
                "status": 0, "moves": moves, "pp": [self.move(m).pp if m else 0 for m in moves],
                "ivs": {"hp": 15, "atk": 15, "def": 15, "spd": 15, "spa": 15, "spd_def": 15},
                "evs": {"hp": 0, "atk": 0, "def": 0, "spd": 0, "spa": 0, "spd_def": 0}}
