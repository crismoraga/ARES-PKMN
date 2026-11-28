"""Per-game RAM layout of the Gen 4 games (Spanish releases), verified empirically.

Every address below was found and checked on the Spanish ROMs (``python -m ares_u.gen4.verify``), not
copied from the US cheat-code lore. The anchor is a static ARM9 pointer whose target holds the save data:

* Platino ES: ``[0x02101F40]`` (US Platinum uses 0x02101D2C); general save block = target + 0x14.
* HeartGold ES: ``[0x021118A0]`` (US HGSS uses 0x0211186C); general save block = target + 0x10.

Offsets inside the general block follow the cartridge save layout (PKHeX ``SAV4Pt``/``SAV4HGSS``): the
trainer block (name, TID, SID, money, gender, language, badges), the party (count + six 236-byte encrypted
party structs) and the player's ``Location`` {map_id, warp_id, x, z, dir}, which the game keeps up to date
on every step and warp (the map id is the zone/map-header id).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Gen4Layout:
    game: str
    base_pointer: int        # static u32 (ARM9 .bss) -> save-data holder in the heap
    general_offset: int      # general save block = [base_pointer] + general_offset
    trainer: int             # trainer name (8 x u16, 0xFFFF-terminated) in the general block
    tid: int
    sid: int
    money: int
    gender: int
    language: int
    badges: int              # bit per badge (Sinnoh / Johto)
    kanto_badges: int | None  # HGSS only
    party_count: int
    party: int               # six 236-byte party structs
    location: int            # Location {map_id, warp_id, x, z, dir} (5 x s32)
    map_object_facing: int | None = None  # documented, not used (heap, no static path)


PLATINO = Gen4Layout(game="platino", base_pointer=0x02101F40, general_offset=0x14, trainer=0x68, tid=0x78,
                     sid=0x7A, money=0x7C, gender=0x80, language=0x81, badges=0x82, kanto_badges=None,
                     party_count=0x9C, party=0xA0, location=0x1280)
HEARTGOLD = Gen4Layout(game="heartgold", base_pointer=0x021118A0, general_offset=0x10, trainer=0x64, tid=0x74,
                       sid=0x76, money=0x78, gender=0x7C, language=0x7D, badges=0x7E, kanto_badges=0x83,
                       party_count=0x94, party=0x98, location=0x1234)
LAYOUTS: dict[str, Gen4Layout] = {"platino": PLATINO, "heartgold": HEARTGOLD}

PARTY_SIZE = 236
BOX_SIZE = 136
RAM_START, RAM_END = 0x02000000, 0x02400000


def layout(game: str) -> Gen4Layout:
    try:
        return LAYOUTS[game]
    except KeyError:
        raise ValueError(f"no Gen 4 layout for {game!r}; expected one of {tuple(LAYOUTS)}") from None
