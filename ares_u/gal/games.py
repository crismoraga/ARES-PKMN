"""Registry of the games ARES-U can be plugged into.

ROMs are user-supplied dumps that live next to the repository (``G:\\ARES`` in the reference
setup); they are never distributed. ``CoordSpec`` describes where the measurement adapter reads
the player's map and tile position. GB/GBC addresses come from the pret disassemblies (pokered,
pokecrystal WRAM symbols) and are verified on the localized ROMs by ``scripts/ares_u_gal.py
verify``; NDS addresses are found automatically by ``scripts/ares_u_gal.py discover`` and stored in
``artifacts/ares_u/gal/<game>_coords.json``. A game without a verified adapter is measured with the
universal pixel-coverage metric only.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[2]
ROM_DIR = ROOT.parent
GAL_ARTIFACTS = ROOT / "artifacts" / "ares_u" / "gal"
Console = Literal["gb", "gbc", "gba", "nds"]


@dataclass(frozen=True)
class Field:
    """One RAM field read for measurement: address, width in bytes and (GBC) WRAM bank."""
    address: int
    width: int = 1
    bank: int = -1  # -1: whatever is mapped (GB/GBA/NDS); >=0: explicit GBC WRAM bank

    def to_json(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class CoordSpec:
    """Where the player's (map_group, map_num, x, y) live; ``map_group`` is optional."""
    map_num: Field
    x: Field
    y: Field
    map_group: Field | None = None
    source: str = ""

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"map_num": self.map_num.to_json(), "x": self.x.to_json(), "y": self.y.to_json(),
                               "source": self.source}
        if self.map_group is not None:
            out["map_group"] = self.map_group.to_json()
        return out

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> CoordSpec:
        group = raw.get("map_group")
        return cls(Field(**raw["map_num"]), Field(**raw["x"]), Field(**raw["y"]),
                   Field(**group) if group else None, str(raw.get("source", "")))


@dataclass(frozen=True)
class GameSpec:
    key: str
    title: str
    generation: int
    console: Console
    rom_name: str
    coords: CoordSpec | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def rom(self) -> Path:
        return ROM_DIR / self.rom_name

    def coord_spec(self) -> CoordSpec | None:
        """Built-in adapter, or the one discovered/verified and saved under ``GAL_ARTIFACTS``."""
        saved = GAL_ARTIFACTS / f"{self.key}_coords.json"
        if saved.exists():
            return CoordSpec.from_json(json.loads(saved.read_text(encoding="utf-8")))
        return self.coords


# pret symbols (international releases share the WRAM layout of these fields; verified per ROM).
RED_COORDS = CoordSpec(map_num=Field(0xD35E), y=Field(0xD361), x=Field(0xD362),
                       source="pokered wCurMap/wYCoord/wXCoord")
CRYSTAL_COORDS = CoordSpec(map_group=Field(0xDCB5, bank=1), map_num=Field(0xDCB6, bank=1),
                           y=Field(0xDCB7, bank=1), x=Field(0xDCB8, bank=1),
                           source="pokecrystal wMapGroup/wMapNumber/wYCoord/wXCoord")

GAMES: dict[str, GameSpec] = {spec.key: spec for spec in (
    GameSpec("rojo", "Pokémon Rojo", 1, "gb", "Pokemon Rojo (ESP).gb", RED_COORDS),
    GameSpec("cristal", "Pokémon Cristal", 2, "gbc", "Pokemon - Edicion Cristal (Spain).gbc", CRYSTAL_COORDS),
    GameSpec("firered", "Pokémon Rojo Fuego", 3, "gba", "Pokemon - FireRed Version (USA).gba",
             notes="RAM metric via the native ARES bridge (pret pokefirered symbols)"),
    GameSpec("emerald", "Pokémon Esmeralda", 3, "gba", "Pokemon - Emerald Version (USA, Europe).gba",
             notes="RAM metric via the native ARES bridge (pret pokeemerald symbols)"),
    GameSpec("platino", "Pokémon Platino", 4, "nds", "3787 - Pokemon - Edicion Platino (Spain).nds"),
    GameSpec("heartgold", "Pokémon Oro HeartGold", 4, "nds", "4832 - Pokemon - Edicion Oro HeartGold (Spain) [b].nds"),
    GameSpec("negro", "Pokémon Negro", 5, "nds", "5597 - Pokemon - Edicion Negra (Spain) (NDSi Enhanced) [b].nds"),
    GameSpec("negro2", "Pokémon Negro 2", 5, "nds",
             "z120 - Pokemon - Edicion Negra 2 (Spain) (NDSi Enhanced) [b].nds"),
)}


def game(key: str) -> GameSpec:
    try:
        return GAMES[key]
    except KeyError:
        raise ValueError(f"Unknown game {key!r}; expected one of {tuple(GAMES)}") from None
