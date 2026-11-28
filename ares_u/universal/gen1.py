"""Gen 1 adapter (Pokemon Rojo, Spanish; the same code reads pokered US with ``shift=0``).

RAM layout: pret/pokered WRAM symbols. In the Spanish ROM every symbol from the battle/overworld
block on (``wWalkCounter`` $CFC5 and everything after it) sits **5 bytes later** than in pokered;
the screen buffers and menu variables before it are not shifted. Each address below was checked on
the Spanish ROM by observing the game (``scripts/ares_u_universal.py verify --game rojo`` re-runs
the checks; see ``docs`` in ``ares_u/universal/README.md``):

==========================  =======  =======  ==================================================
symbol                      pokered  Rojo ES  how it was verified
==========================  =======  =======  ==================================================
wTileMap (20x18)            $C3A0    $C3A0    START menu text decoded at the expected cells
wSpriteStateData1/2         $C100    $C100    player facing (C109) follows the pressed direction
wOverworldMap               $C6E8    $C6E8    blocks of Red's room match the drawn room
wCurrentMenuItem            $CC26    $CC26    increments with DOWN in the START menu
wWalkCounter                $CFC5    $CFCA    counts 7..0 during one step
wPlayerName                 $D158    $D15D    "AA" written by the intro schedule
wPartyCount / wPartySpecies $D163    $D168    0 + $FF terminator before the starter
wNumBagItems                $D31D    $D322    0 + $FF terminator at the start
wPlayerMoney (BCD)          $D347    $D34C    00 30 00 = 3000 at the start
wCurMap / wYCoord / wXCoord $D35E    $D363    38 (REDS_HOUSE_2F), walking changes x/y
wNumberOfWarps / entries    $D3AE    $D3B3    1 warp (7,1) -> map 37 in Red's room
wTileset bank/blocks/coll   $D52B    $D530    collision list $1749 = pokered RedsHouse_Coll
==========================  =======  =======  ==================================================
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ares_u.gal.games import GameSpec

from .adapter import TILE_SPACE, GameAdapter, GBScreenParser, base_charmap
from .model import (Mode, COUNTER, FLOOR, GRASS, LEDGE_DOWN, LEDGE_LEFT, LEDGE_RIGHT, WALL, WARP_CELL, BattleInfo,
                    Connection, Direction, MapInfo, MapKey, MonInfo, MoveInfo, Npc, ScreenInfo, Snapshot, Warp)
from .typechart import GB_TYPE_IDS

SHIFT_FROM = 0xCF00  # pokered symbols at or above this address are shifted in the localized ROM

# pokered (US) addresses; ``Gen1Adapter.addr`` applies the localisation shift.
SYM: dict[str, int] = {
    "wTileMap": 0xC3A0, "wSpriteStateData1": 0xC100, "wSpriteStateData2": 0xC200, "wOverworldMap": 0xC6E8,
    "wTopMenuItemY": 0xCC24, "wTopMenuItemX": 0xCC25, "wCurrentMenuItem": 0xCC26, "wMaxMenuItem": 0xCC28,
    "wPlayerMonNumber": 0xCC2F,
    "wWalkCounter": 0xCFC5,
    "wEnemyMon": 0xCFE5, "wBattleMon": 0xD014, "wIsInBattle": 0xD057, "wCurOpponent": 0xD059,
    "wPlayerName": 0xD158, "wPartyCount": 0xD163, "wPartySpecies": 0xD164, "wPartyMons": 0xD16B,
    "wPartyMonNicks": 0xD2B5, "wPokedexOwned": 0xD2F7, "wNumBagItems": 0xD31D, "wBagItems": 0xD31E,
    "wPlayerMoney": 0xD347, "wObtainedBadges": 0xD356, "wCurMap": 0xD35E, "wYCoord": 0xD361,
    "wXCoord": 0xD362, "wLastMap": 0xD365, "wCurMapTileset": 0xD367, "wCurMapHeight": 0xD368,
    "wCurMapWidth": 0xD369, "wCurMapConnections": 0xD370, "wNorthConnectionHeader": 0xD371,
    "wNumberOfWarps": 0xD3AE, "wWarpEntries": 0xD3AF, "wNumSigns": 0xD4B0, "wSignCoords": 0xD4B1,
    "wNumSprites": 0xD4E1, "wMapSpriteData": 0xD4E4, "wTilesetBank": 0xD52B, "wTilesetBlocksPtr": 0xD52C,
    "wTilesetCollisionPtr": 0xD530, "wTilesetTalkingOverTiles": 0xD532, "wGrassTile": 0xD535,
    "wMissableObjectFlags": 0xD5A6, "wMissableObjectList": 0xD5CE, "wTownVisitedFlag": 0xD70B,
    "wLastBlackoutMap": 0xD719, "wEventFlags": 0xD747,
    "wJoyIgnore": 0xCD6B, "wSimulatedJoypadStatesIndex": 0xCD38,
}

# Spanish type words in dialog (starter questions) and battle effectiveness messages (folded).
TYPE_WORDS_ES = {"fuego": "fire", "agua": "water", "acuatico": "water", "planta": "grass", "hierba": "grass", "electrico": "electric", "roca": "rock",
                 "volador": "flying", "bicho": "bug", "veneno": "poison", "tierra": "ground", "lucha": "fighting",
                 "psiquico": "psychic", "hielo": "ice", "fantasma": "ghost", "dragon": "dragon"}
EFFECT_PHRASES_ES = [("no es muy eficaz", 0.5), ("no es muy efectivo", 0.5), ("poco eficaz", 0.5),
                     ("muy eficaz", 2.0), ("super eficaz", 2.0), ("muy efectivo", 2.0), ("no afecta", 0.0),
                     ("no le afecta", 0.0), ("no tiene efecto", 0.0)]

PARTY_MON_SIZE = 44
BATTLE_MON_SIZE = 29
NUM_EVENT_BYTES = 320
FACING = {0x0: "down", 0x4: "up", 0x8: "left", 0xC: "right"}
SPRITE_POKE_BALL = 0x3D
OVERWORLD_TILESET = 0
# Overworld ledges (pokered data/tilesets/ledge_tiles.asm): ledge tile -> jump direction.
LEDGE_TILES: dict[int, int] = {0x37: LEDGE_DOWN, 0x36: LEDGE_DOWN, 0x27: LEDGE_LEFT, 0x0D: LEDGE_RIGHT,
                               0x1D: LEDGE_RIGHT}
# Tilesets drawn outdoors (pokered tileset constants OVERWORLD, FOREST, PLATEAU).
OUTDOOR_TILESETS = frozenset({0, 3, 23})

# Spanish letters of the Gen 1 font (decoded from the ROM move/species name tables).
SPANISH_GEN1 = {0xBA: "à", 0xBB: "è", 0xBC: "é", 0xBD: "ù", 0xBE: "ß", 0xBF: "Á", 0xC0: "Ä", 0xC1: "Ö",
                0xC2: "Ü", 0xC3: "ä", 0xC4: "ö", 0xC5: "ü", 0xC6: "È", 0xC7: "É", 0xC8: "Ì", 0xC9: "Í",
                0xCA: "Ñ", 0xCB: "Ò", 0xCC: "Ó", 0xCD: "Ù", 0xCE: "Ú", 0xCF: "á", 0xD0: "ì", 0xD1: "Í",
                0xD2: "Ñ", 0xD3: "ò", 0xD4: "ó", 0xD5: "ú", 0xD6: "º", 0xD7: "&", 0xD8: "'d", 0xD9: "'l",
                0xDA: "'m", 0xDB: "'r", 0xDC: "'s", 0xDD: "'t", 0xDE: "'v", 0xE4: "'", 0xE5: "'",
                0xE9: "¿", 0xEA: "¡", 0xEB: "í"}

MAP_NAMES_GEN1: dict[int, str] = {
    0x00: "Pueblo Paleta", 0x01: "Ciudad Verde", 0x02: "Ciudad Plateada", 0x03: "Ciudad Celeste",
    0x0C: "Ruta 1", 0x0D: "Ruta 2", 0x0E: "Ruta 3", 0x0F: "Ruta 4", 0x21: "Ruta 22", 0x25: "Casa (1F)",
    0x26: "Casa (2F)", 0x27: "Casa del rival", 0x28: "Laboratorio de Oak", 0x29: "Centro Pokemon Verde",
    0x2A: "Tienda de Ciudad Verde", 0x2B: "Escuela de Verde", 0x2C: "Casa de Verde", 0x2D: "Gimnasio Verde",
    0x2E: "Casa de Ruta 2", 0x2F: "Puerta norte del Bosque", 0x30: "Casa de intercambio Ruta 2",
    0x31: "Puerta de Ruta 2", 0x32: "Puerta sur del Bosque", 0x33: "Bosque Verde", 0x34: "Museo (1F)",
    0x35: "Museo (2F)", 0x36: "Gimnasio Plateada", 0x37: "Casa de Plateada", 0x38: "Tienda de Plateada",
    0x39: "Casa de Plateada 2", 0x3A: "Centro Pokemon Plateada", 0x3B: "Mt. Moon 1F", 0x3C: "Mt. Moon B1F",
    0x3D: "Mt. Moon B2F", 0x44: "Centro Pokemon Ruta 3", 0xC1: "Puerta Ruta 22",
}

OAKS_PARCEL = 0x46
TOWN_MAP = 0x05


def _encode(text: str) -> bytes:
    out = bytearray()
    for char in text:
        if char == " ":
            out.append(0x7F)
        elif "A" <= char <= "Z":
            out.append(0x80 + ord(char) - ord("A"))
        else:
            raise ValueError(char)
    return bytes(out)


def _bcd(data: bytes) -> int:
    value = 0
    for byte in data:
        value = value * 100 + (byte >> 4) * 10 + (byte & 0xF)
    return value


def _status(byte: int) -> str:
    if byte & 0x07:
        return "slp"
    if byte & 0x08:
        return "psn"
    if byte & 0x10:
        return "brn"
    if byte & 0x20:
        return "frz"
    if byte & 0x40:
        return "par"
    return ""


def find_table(rom: bytes, signature: bytes) -> int:
    index = rom.find(signature)
    if index < 0:
        raise ValueError(f"signature {signature[:6].hex()}.. not found in the ROM")
    return index


class Gen1Adapter(GameAdapter):
    generation = 1

    def __init__(self, spec: GameSpec, bridge: Any | None = None, shift: int = 5) -> None:
        super().__init__(spec, bridge)
        self.key = spec.key
        self.shift = shift
        charmap = base_charmap()
        charmap.update(SPANISH_GEN1)
        self.parser = GBScreenParser(charmap)
        self._load_rom_tables()
        self._map_cache: MapInfo | None = None
        self._map_cache_key: tuple[Any, ...] = ()

    # ------------------------------------------------------------------ ROM data
    def _load_rom_tables(self) -> None:
        rom = self.rom
        # IndexToPokedex (data/pokemon/dex_order.asm): internal index 1.. -> national dex.
        dex_at = find_table(rom, bytes([112, 115, 32, 35, 21, 100, 34, 80, 2, 103]))
        self.dex_of = {index + 1: int(rom[dex_at + index]) for index in range(190)}
        # Moves (data/moves/moves.asm): anim, effect, power, type, accuracy, pp (6 bytes, POUND first).
        moves_at = find_table(rom, bytes([0x01, 0x00, 0x28, 0x00, 0xFF, 0x23, 0x02, 0x00, 0x32, 0x00, 0xFF, 0x19]))
        self.move_data: dict[int, tuple[int, int, str, int, int]] = {}
        for index in range(165):
            anim, effect, power, type_id, accuracy, pp = rom[moves_at + 6 * index: moves_at + 6 * index + 6]
            self.move_data[index + 1] = (effect, power, GB_TYPE_IDS.get(type_id, "normal"),
                                         round(accuracy * 100 / 255), pp)
            del anim
        # Move names (0x50-terminated) and fixed 10-byte species names, both found by their first entry.
        names_at = rom.find(bytes([0x83, 0x84, 0x92, 0x93, 0x91, 0x94, 0x82, 0x93, 0x8E, 0x91, 0x50]))  # DESTRUCTOR
        if names_at < 0:
            names_at = rom.find(bytes([0x8F, 0x8E, 0x94, 0x8D, 0x83, 0x50]))  # POUND (pokered US)
        self.move_names: dict[int, str] = {}
        if names_at >= 0:
            cursor = names_at
            for index in range(165):
                end = rom.index(0x50, cursor)
                self.move_names[index + 1] = self.parser.decode(rom[cursor:end])
                cursor = end + 1
        self._move_lookup = sorted(((name, move_id) for move_id, name in self.move_names.items() if len(name) >= 3),
                                   key=lambda item: -len(item[0]))
        # Item names: 0x50-terminated list starting with "MASTER BALL@ULTRA BALL@".
        self.item_names: dict[int, str] = {}
        sig = _encode("MASTER BALL") + b"P" + _encode("ULTRA BALL") + b"P"
        items_at = rom.find(sig)
        if items_at >= 0:
            cursor = items_at
            for index in range(1, 98):
                end = rom.index(0x50, cursor)
                self.item_names[index] = self.parser.decode(rom[cursor:end])
                cursor = end + 1
        species_at = rom.find(bytes([0x91, 0x87, 0x98, 0x83, 0x8E, 0x8D, 0x50]))  # RHYDON@
        self.species_names: dict[int, str] = {}
        if species_at >= 0:
            for index in range(190):
                raw = rom[species_at + 10 * index: species_at + 10 * index + 10].split(b"\x50")[0]
                self.species_names[index + 1] = self.parser.decode(raw)

    # ------------------------------------------------------------------ RAM helpers
    def addr(self, symbol: str) -> int:
        address = SYM[symbol]
        return address + self.shift if address >= SHIFT_FROM else address

    def ram(self, symbol: str, size: int = 1, offset: int = 0) -> bytes:
        return self.read(self.addr(symbol) + offset, size)

    def ram8(self, symbol: str, offset: int = 0) -> int:
        return self.read(self.addr(symbol) + offset, 1)[0]

    def rom_bytes(self, bank: int, pointer: int, size: int) -> bytes:
        """ROM bytes at a GB address (bank 0 below $4000, ``bank`` in $4000-$7FFF)."""
        offset = pointer if pointer < 0x4000 else bank * 0x4000 + (pointer - 0x4000)
        return self.rom[offset:offset + size]

    def tilemap(self) -> NDArray[np.uint8]:
        return np.frombuffer(self.ram("wTileMap", 360), dtype=np.uint8).reshape(18, 20)

    # ------------------------------------------------------------------ basic state
    def position(self) -> tuple[MapKey, int, int]:
        data = self.read(self.addr("wCurMap"), 5)
        return (0, data[0]), data[4], data[3]

    def facing(self) -> Direction:
        value = FACING.get(self.read(0xC109, 1)[0] & 0xC, "down")
        return value  # type: ignore[return-value]

    def is_walking(self) -> bool:
        return self.ram8("wWalkCounter") != 0

    def in_battle(self) -> bool:
        return self.ram8("wIsInBattle") not in (0, 0xFF)

    def dialog_open(self) -> bool:
        corners = self.read(self.addr("wTileMap") + 12 * 20, 1)[0], self.read(self.addr("wTileMap") + 17 * 20 + 19, 1)[0]
        return corners == (0x79, 0x7E)

    def menu_index(self) -> int:
        return self.ram8("wCurrentMenuItem")

    def map_name(self, key: MapKey) -> str:
        return MAP_NAMES_GEN1.get(key[1], f"Mapa {key[1]}")

    def invalidate(self) -> None:
        self._map_cache = None
        self._map_cache_key = ()

    # ------------------------------------------------------------------ party / battle
    def _move(self, move_id: int, pp_byte: int) -> MoveInfo:
        effect, power, type_name, accuracy, max_pp = self.move_data.get(move_id, (0, 0, "normal", 100, 0))
        return MoveInfo(move_id=move_id, name=self.move_names.get(move_id, f"MOVE{move_id}"), type=type_name,
                        power=power, accuracy=accuracy, pp=pp_byte & 0x3F, max_pp=max_pp, effect=effect)

    def _mon(self, data: bytes, battle: bool, name: str = "") -> MonInfo:
        species = data[0]
        if battle:
            level, max_hp = data[14], int.from_bytes(data[15:17], "big")
            stats = {"atk": int.from_bytes(data[17:19], "big"), "def": int.from_bytes(data[19:21], "big"),
                     "spe": int.from_bytes(data[21:23], "big"), "spa": int.from_bytes(data[23:25], "big")}
            pps = data[25:29]
        else:
            level, max_hp = data[33], int.from_bytes(data[34:36], "big")
            stats = {"atk": int.from_bytes(data[36:38], "big"), "def": int.from_bytes(data[38:40], "big"),
                     "spe": int.from_bytes(data[40:42], "big"), "spa": int.from_bytes(data[42:44], "big")}
            pps = data[29:33]
        stats["spd"] = stats["spa"]
        types = tuple(dict.fromkeys(GB_TYPE_IDS.get(t, "normal") for t in (data[5], data[6])))
        moves = tuple(self._move(move_id, pps[slot]) for slot, move_id in enumerate(data[8:12]) if move_id)
        return MonInfo(species_id=species, national_dex=self.dex_of.get(species, 0),
                       name=name or self.species_names.get(species, f"#{species}"), level=level,
                       hp=int.from_bytes(data[1:3], "big"), max_hp=max_hp, types=types, status=_status(data[4]),
                       moves=moves, stats=stats)

    def party(self) -> tuple[MonInfo, ...]:
        count = self.ram8("wPartyCount")
        if not 0 <= count <= 6:
            return ()
        raw = self.ram("wPartyMons", PARTY_MON_SIZE * count)
        mons = []
        for i in range(count):
            chunk = raw[i * PARTY_MON_SIZE:(i + 1) * PARTY_MON_SIZE]
            if chunk[0] == 0 or chunk[33] == 0:
                break  # slot still being written (the game fills it over several frames)
            mons.append(self._mon(chunk, battle=False))
        return tuple(mons)

    def battle(self) -> BattleInfo | None:
        kind = self.ram8("wIsInBattle")
        if kind not in (1, 2):
            return None
        me_raw = self.ram("wBattleMon", BATTLE_MON_SIZE)
        enemy_raw = self.ram("wEnemyMon", BATTLE_MON_SIZE)
        me = self._mon(me_raw, battle=True) if me_raw[0] else None
        enemy = self._mon(enemy_raw, battle=True) if enemy_raw[0] else None
        return BattleInfo(kind="wild" if kind == 1 else "trainer", me=me, enemy=enemy,
                          active_index=self.read(SYM["wPlayerMonNumber"], 1)[0])

    def bag(self) -> tuple[tuple[int, int], ...]:
        count = self.ram8("wNumBagItems")
        if not 0 <= count <= 20:
            return ()
        raw = self.ram("wBagItems", 2 * count)
        return tuple((raw[2 * i], raw[2 * i + 1]) for i in range(count))

    def event_flag_count(self) -> int:
        raw = np.frombuffer(self.ram("wEventFlags", NUM_EVENT_BYTES), dtype=np.uint8)
        return int(np.unpackbits(raw).sum())

    def badges(self) -> int:
        return bin(self.ram8("wObtainedBadges")).count("1")

    def money(self) -> int:
        return _bcd(self.ram("wPlayerMoney", 3))

    def dex_owned(self) -> int:
        raw = np.frombuffer(self.ram("wPokedexOwned", 19), dtype=np.uint8)
        return int(np.unpackbits(raw).sum())

    def joy_locked(self) -> bool:
        # wJoyIgnore ($CD6B, below the shifted block): a script ignores the D-pad / A / B.
        return bool(self.read(SYM["wJoyIgnore"], 1)[0] & 0xF3) or self.read(SYM["wSimulatedJoypadStatesIndex"], 1)[0] > 0

    # ------------------------------------------------------------------ localisation / milestones
    def item_name(self, item_id: int) -> str:
        return self.item_names.get(item_id, f"objeto {item_id}")

    def type_words(self) -> dict[str, str]:
        return dict(TYPE_WORDS_ES)

    def effectiveness_phrases(self) -> list[tuple[str, float]]:
        return list(EFFECT_PHRASES_ES)

    def find_move_in_text(self, text: str) -> MoveInfo | None:
        upper = f" {text} "
        for name, move_id in self._move_lookup:
            index = upper.find(name)
            if index > 0 and not upper[index - 1].isalpha() and not upper[index + len(name)].isalpha():
                return self._move(move_id, self.move_data.get(move_id, (0, 0, "normal", 100, 0))[4])
        return None

    def event_bits(self) -> set[int]:
        raw = np.frombuffer(self.ram("wEventFlags", NUM_EVENT_BYTES), dtype=np.uint8)
        bits = np.unpackbits(raw, bitorder="little")
        return {int(i) for i in np.nonzero(bits)[0]}

    def milestones(self, snap: Snapshot, visited: set[MapKey], flags: set[str]) -> list[str]:
        items = {item for item, _ in snap.bag}
        if OAKS_PARCEL in items:
            flags.add("had_parcel")
        out = []
        if snap.party:
            out.append("starter")
        if "first_battle_won" in flags:
            out.append("first_battle_won")
        if "trainer_won@0.40" in flags:
            out.append("rival_lab_beaten")
        if (0, 0x0C) in visited:
            out.append("left_pallet")
        if (0, 0x01) in visited:
            out.append("viridian")
        if "had_parcel" in flags:
            out.append("oaks_parcel")
        if "had_parcel" in flags and OAKS_PARCEL not in items:
            out.append("parcel_delivered_pokedex")
        if (0, 0x33) in visited:
            out.append("viridian_forest")
        if (0, 0x02) in visited:
            out.append("pewter")
        if self.ram8("wObtainedBadges") & 1:
            out.append("boulder_badge")
        return out

    # ------------------------------------------------------------------ screen
    def screen(self) -> ScreenInfo:
        tiles = self.tilemap()
        info = self.parser.screen(tiles)
        if info.menu is not None:
            labels = info.menu.options
            kind = "generic"
            if len(labels) == 2 and all(len(label) <= 3 for label in labels):
                kind = "yesno"
            elif self.in_battle() and len(labels) == 4 and info.menu.cursor_row >= 12:
                kind = "battle_main" if any(label in ("LUCHA", "FIGHT") for label in labels) else "battle"
            info = ScreenInfo(lines=info.lines, textbox=info.textbox, prompt=info.prompt,
                              menu=type(info.menu)(cursor_row=info.menu.cursor_row,
                                                   cursor_col=info.menu.cursor_col,
                                                   options=info.menu.options, selected=info.menu.selected,
                                                   kind=kind, positions=info.menu.positions),
                              naming=self._naming(tiles), rows=info.rows)
        elif self._naming(tiles):
            info = ScreenInfo(lines=info.lines, textbox=info.textbox, prompt=info.prompt, menu=None, naming=True,
                              rows=info.rows)
        return info

    def overlay_screen(self) -> bool:
        """A full-screen UI covers the map: Gen 1 map tiles are < $60, font/frame tiles are >= $60."""
        tiles = self.tilemap()
        return int(np.sum(tiles[:12] >= 0x60)) >= 60

    @staticmethod
    def _naming(tiles: NDArray[np.uint8]) -> bool:
        # The naming keyboard is a grid of single letters separated by blanks on rows 5..13.
        letters = 0
        for row in range(2, 16):
            line = tiles[row, 1:19]
            letters += int(np.sum((line >= 0x80) & (line <= 0x99) & (np.roll(line, 1) == TILE_SPACE)
                                  & (np.roll(line, -1) == TILE_SPACE)))
        return letters >= 20

    # ------------------------------------------------------------------ map
    def map_signature(self) -> bytes:
        # Header from the tileset byte on (no coordinates), the first warps and the block map.
        # (Pointers after the size bytes are rewritten earlier than the rest: they must stay out.)
        return (self.read(self.addr("wCurMapTileset"), 3) + self.read(self.addr("wCurMapConnections"), 1)
                + self.read(self.addr("wNumberOfWarps"), 33) + self.read(SYM["wOverworldMap"], 0x200))

    def map_info(self) -> MapInfo:
        self.ensure_settled()
        header = self.read(self.addr("wCurMap"), 0x10)
        warps = self.read(self.addr("wNumberOfWarps"), 1 + 4 * 32)
        cache_key = (header[0], header[9], header[10], header[11], warps[:1 + 4 * min(warps[0], 32)])
        if self._map_cache is not None and self._map_cache_key == cache_key:
            return self._map_cache
        info = self._build_map()
        self._map_cache, self._map_cache_key = info, cache_key
        return info

    def _build_map(self) -> MapInfo:
        map_id = self.ram8("wCurMap")
        height_blocks, width_blocks = self.ram8("wCurMapHeight"), self.ram8("wCurMapWidth")
        tileset = self.ram8("wCurMapTileset")
        width, height = 2 * width_blocks, 2 * height_blocks
        stride = width_blocks + 6
        blocks = self.read(SYM["wOverworldMap"], stride * (height_blocks + 6))
        bank = self.ram8("wTilesetBank")
        blocks_ptr = int.from_bytes(self.ram("wTilesetBlocksPtr", 2), "little")
        coll_ptr = int.from_bytes(self.ram("wTilesetCollisionPtr", 2), "little")
        passable = self._collision_list(coll_ptr, bank)
        talk_over = {t for t in self.ram("wTilesetTalkingOverTiles", 3) if t != 0xFF}
        grass_tile = self.ram8("wGrassTile")
        used = sorted(set(blocks))
        block_tiles = {b: self.rom_bytes(bank, blocks_ptr + 16 * b, 16) for b in used}
        grid = np.full((height, width), WALL, dtype=np.int8)
        for cy in range(height):
            for cx in range(width):
                block = blocks[(cy // 2 + 3) * stride + cx // 2 + 3]
                tile = block_tiles[block][((cy % 2) * 2 + 1) * 4 + (cx % 2) * 2]
                if tile in passable:
                    grid[cy, cx] = GRASS if tile == grass_tile else FLOOR
                elif tileset == OVERWORLD_TILESET and tile in LEDGE_TILES:
                    grid[cy, cx] = LEDGE_TILES[tile]
                elif tile in talk_over:
                    grid[cy, cx] = COUNTER
        warps = []
        count = self.ram8("wNumberOfWarps")
        raw = self.ram("wWarpEntries", 4 * min(count, 32))
        last_map = self.ram8("wLastMap")  # warp destination 0xFF = "the map we came from" (house/shop exits)
        for index in range(min(count, 32)):
            y, x, dest_index, dest_map = raw[4 * index: 4 * index + 4]
            if dest_map == 0xFF:
                dest_map = last_map
            warps.append(Warp(x=x, y=y, dest_map=None if dest_map == 0xFF else (0, dest_map), dest_index=dest_index))
            if 0 <= x < width and 0 <= y < height:
                grid[y, x] = WARP_CELL
        connections = []
        mask = self.ram8("wCurMapConnections")
        for bit, direction, slot in ((0x08, "north", 0), (0x04, "south", 1), (0x02, "west", 2), (0x01, "east", 3)):
            if mask & bit:
                dest = self.read(self.addr("wNorthConnectionHeader") + 11 * slot, 1)[0]
                name = {"north": "up", "south": "down", "west": "left", "east": "right"}[direction]
                connections.append(Connection(direction=name, dest_map=(0, dest)))  # type: ignore[arg-type]
        signs = []
        sign_count = min(self.ram8("wNumSigns"), 16)
        sign_raw = self.ram("wSignCoords", 2 * sign_count)
        for index in range(sign_count):
            signs.append((sign_raw[2 * index + 1], sign_raw[2 * index]))
        return MapInfo(key=(0, map_id), width=width, height=height, grid=grid, warps=warps, connections=connections,
                       signs=signs, tileset=tileset, outdoor=tileset in OUTDOOR_TILESETS,
                       name=self.map_name((0, map_id)))

    def _collision_list(self, pointer: int, bank: int) -> set[int]:
        raw = self.rom_bytes(bank, pointer, 64)
        out = set()
        for value in raw:
            if value == 0xFF:
                break
            out.add(value)
        return out

    def npcs(self) -> list[Npc]:
        count = min(self.ram8("wNumSprites"), 15)
        if count == 0:
            return []
        data1 = self.read(SYM["wSpriteStateData1"], 16 * (count + 1))
        data2 = self.read(SYM["wSpriteStateData2"], 16 * (count + 1))
        hidden = self._hidden_sprites()
        movement = self.ram("wMapSpriteData", 2 * count)
        out = []
        for slot in range(1, count + 1):
            picture = data1[16 * slot]
            if picture == 0 or slot in hidden:
                continue
            y, x = data2[16 * slot + 4] - 4, data2[16 * slot + 5] - 4
            moving = movement[2 * (slot - 1)] == 0xFE
            out.append(Npc(index=slot, x=x, y=y, sprite=picture, visible=True, moving=moving,
                           is_item=picture == SPRITE_POKE_BALL))
        return out

    def _hidden_sprites(self) -> set[int]:
        flags = self.ram("wMissableObjectFlags", 32)
        listing = self.ram("wMissableObjectList", 34)
        hidden = set()
        for index in range(0, 34, 2):
            slot, global_index = listing[index], listing[index + 1]
            if slot == 0xFF:
                break
            if flags[global_index // 8] & (1 << (global_index % 8)):
                hidden.add(slot)
        return hidden

    # ------------------------------------------------------------------ snapshot
    def snapshot(self) -> Snapshot:
        key, x, y = self.position()
        screen = self.screen()
        battle = self.battle()
        mode: Mode
        if battle is not None:
            mode = "battle"
        elif screen.naming:
            mode = "naming"
        elif screen.menu is None and not screen.textbox and self.overlay_screen():
            mode = "text"  # full-screen UI (Pokedex page, town map...): waits for a button
        elif screen.menu is not None:
            mode = "menu"
        elif screen.textbox:
            mode = "text"
        else:
            mode = "overworld"
        return Snapshot(frame=self.frame_count, map=key, x=x, y=y, facing=self.facing(),
                        mode=mode, screen=screen, party=self.party(), battle=battle, badges=self.badges(),
                        money=self.money(), bag=self.bag(), event_flags=self.event_flag_count(),
                        dex_owned=self.dex_owned(), walking=self.is_walking(), joy_locked=self.joy_locked())
