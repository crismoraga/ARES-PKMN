"""Gen 4 Pokémon structures: party/box decryption, block unshuffling and parsing.

A party Pokémon is 236 bytes: an 8-byte header (PID u32, flags u16, checksum u16), 128 bytes of box data in
four 32-byte blocks (A: species/item/OT/exp/ability, B: moves/PP/IVs, C: nickname, D: OT name/met data)
stored in one of 24 orders chosen by ``((PID & 0x3E000) >> 13) % 24`` and XOR-encrypted with the LCG
``seed = seed * 0x41C64E6D + 0x6073`` (key = seed >> 16 per u16) seeded with the checksum, followed by 100
bytes of battle stats (status, level, HP, stats) encrypted with the same LCG seeded with the PID. The
checksum is the u16 sum of the decrypted 64 box words. (Format as documented by Project Pokémon / PKHeX
``PK4``; verified here on the starter read from RAM, see ``tests/test_gen4_pkm.py``.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

PARTY_SIZE = 236
BOX_SIZE = 136
BLOCK = 32
# Position of the blocks in storage for each shift value: ORDERS[s][i] is the block stored at slot i.
ORDERS = ("ABCD", "ABDC", "ACBD", "ACDB", "ADBC", "ADCB", "BACD", "BADC", "BCAD", "BCDA", "BDAC", "BDCA",
          "CABD", "CADB", "CBAD", "CBDA", "CDAB", "CDBA", "DABC", "DACB", "DBAC", "DBCA", "DCAB", "DCBA")
STATUS_NAMES = ("", "slp", "psn", "brn", "frz", "par", "tox")
LCG_MUL, LCG_ADD = 0x41C64E6D, 0x6073


def _keystream(seed: int, words: int) -> NDArray[np.uint16]:
    out = np.empty(words, dtype=np.uint16)
    for index in range(words):
        seed = (seed * LCG_MUL + LCG_ADD) & 0xFFFFFFFF
        out[index] = seed >> 16
    return out


def _crypt(data: bytes, seed: int) -> bytes:
    words = np.frombuffer(data, dtype="<u2")
    return (words ^ _keystream(seed, len(words))).astype("<u2").tobytes()


def shuffle_value(pid: int) -> int:
    return ((pid & 0x3E000) >> 13) % 24


def checksum(box_plain: bytes) -> int:
    """u16 sum of the 64 decrypted box words (bytes 0x08-0x87 in canonical ABCD order)."""
    return int(np.frombuffer(box_plain, dtype="<u2").sum(dtype=np.uint64)) & 0xFFFF


def unshuffle(blocks: bytes, pid: int) -> bytes:
    order = ORDERS[shuffle_value(pid)]
    return b"".join(blocks[order.index(letter) * BLOCK:(order.index(letter) + 1) * BLOCK] for letter in "ABCD")


def shuffle(blocks: bytes, pid: int) -> bytes:
    order = ORDERS[shuffle_value(pid)]
    return b"".join(blocks["ABCD".index(letter) * BLOCK:("ABCD".index(letter) + 1) * BLOCK] for letter in order)


def decrypt(raw: bytes) -> bytes | None:
    """Encrypted 136/236-byte struct -> canonical plain struct (ABCD order), or None on checksum mismatch."""
    if len(raw) not in (BOX_SIZE, PARTY_SIZE):
        raise ValueError(f"expected 136 or 236 bytes, got {len(raw)}")
    pid = int.from_bytes(raw[0:4], "little")
    check = int.from_bytes(raw[6:8], "little")
    plain_shuffled = _crypt(raw[8:BOX_SIZE], check)
    if checksum(plain_shuffled) != check:
        return None
    plain = raw[:8] + unshuffle(plain_shuffled, pid)
    if len(raw) == PARTY_SIZE:
        plain += _crypt(raw[BOX_SIZE:PARTY_SIZE], pid)
    return plain


def encrypt(plain: bytes) -> bytes:
    """Inverse of :func:`decrypt` (recomputes the checksum)."""
    pid = int.from_bytes(plain[0:4], "little")
    box = plain[8:BOX_SIZE]
    check = checksum(box)
    header = plain[0:6] + check.to_bytes(2, "little")
    out = header + _crypt(shuffle(box, pid), check)
    if len(plain) == PARTY_SIZE:
        out += _crypt(plain[BOX_SIZE:PARTY_SIZE], pid)
    return out


# Gen 4 character table, enough for names typed on the keyboards of the Western games.
_CHARS: dict[int, str] = {0x01DE: " ", 0x01AB: "!", 0x01AC: "?", 0x01AE: ".", 0x01AD: ",", 0x01BE: "-"}
_CHARS.update({0x0121 + i: chr(ord("0") + i) for i in range(10)})
_CHARS.update({0x012B + i: chr(ord("A") + i) for i in range(26)})
_CHARS.update({0x0145 + i: chr(ord("a") + i) for i in range(26)})


def decode_text(raw: bytes) -> str:
    chars = []
    for (code,) in np.frombuffer(raw, dtype="<u2").reshape(-1, 1):
        if code == 0xFFFF:
            break
        chars.append(_CHARS.get(int(code), "?"))
    return "".join(chars)


def status_name(status: int) -> str:
    if status & 0x7:
        return "slp"
    if status & 0x80:
        return "tox"
    for bit, name in ((0x08, "psn"), (0x10, "brn"), (0x20, "frz"), (0x40, "par")):
        if status & bit:
            return name
    return ""


@dataclass(frozen=True)
class Pokemon:
    pid: int
    species: int                    # national dex number (Gen 4 species ids are national numbers)
    held_item: int
    ot_id: int
    ot_secret: int
    exp: int
    friendship: int
    ability: int
    moves: tuple[int, int, int, int]
    pp: tuple[int, int, int, int]
    pp_ups: tuple[int, int, int, int]
    ivs: tuple[int, int, int, int, int, int]   # hp, atk, def, spe, spa, spd
    evs: tuple[int, int, int, int, int, int]
    is_egg: bool
    nickname: str
    status: int = 0
    level: int = 0
    hp: int = 0
    max_hp: int = 0
    stats: dict[str, int] = field(default_factory=dict)   # atk, def, spe, spa, spd

    @property
    def status_name(self) -> str:
        return status_name(self.status)


def parse(plain: bytes) -> Pokemon:
    """Parse a canonical plain struct (output of :func:`decrypt`)."""
    u16 = lambda off: int.from_bytes(plain[off:off + 2], "little")  # noqa: E731
    u32 = lambda off: int.from_bytes(plain[off:off + 4], "little")  # noqa: E731
    iv_word = u32(0x38)
    ivs = tuple((iv_word >> (5 * i)) & 0x1F for i in range(6))
    kwargs: dict[str, object] = {}
    if len(plain) >= PARTY_SIZE:
        kwargs = {"status": u32(0x88), "level": plain[0x8C], "hp": u16(0x8E), "max_hp": u16(0x90),
                  "stats": {"atk": u16(0x92), "def": u16(0x94), "spe": u16(0x96), "spa": u16(0x98),
                            "spd": u16(0x9A)}}
    return Pokemon(pid=u32(0), species=u16(0x08), held_item=u16(0x0A), ot_id=u16(0x0C), ot_secret=u16(0x0E),
                   exp=u32(0x10), friendship=plain[0x14], ability=plain[0x15],
                   moves=(u16(0x28), u16(0x2A), u16(0x2C), u16(0x2E)),
                   pp=(plain[0x30], plain[0x31], plain[0x32], plain[0x33]),
                   pp_ups=(plain[0x34], plain[0x35], plain[0x36], plain[0x37]),
                   ivs=(ivs[0], ivs[1], ivs[2], ivs[3], ivs[4], ivs[5]),
                   evs=(plain[0x18], plain[0x19], plain[0x1A], plain[0x1B], plain[0x1C], plain[0x1D]),
                   is_egg=bool(iv_word >> 30 & 1), nickname=decode_text(plain[0x48:0x5E]),
                   **kwargs)  # type: ignore[arg-type]


def plausible(mon: Pokemon) -> bool:
    if not 1 <= mon.species <= 493:
        return False
    if mon.max_hp and not (1 <= mon.level <= 100 and mon.hp <= mon.max_hp):
        return False
    return all(move <= 467 for move in mon.moves) and mon.moves[0] != 0


def scan(blob: bytes | NDArray[np.uint8], base: int, size: int = PARTY_SIZE, step: int = 4) -> list[tuple[int, Pokemon]]:
    """Find every encrypted Pokémon struct in ``blob`` (checksum + plausibility), vectorised over offsets.

    ``base`` is the address of ``blob[0]``. Returns [(address, Pokemon)] for structs whose decrypted box
    checksum matches and whose fields are plausible.
    """
    data = np.frombuffer(bytes(blob), dtype=np.uint8) if not isinstance(blob, np.ndarray) else blob
    n = (len(data) - size) // step + 1
    if n <= 0:
        return []
    found: list[tuple[int, Pokemon]] = []
    halfwords = data[: len(data) // 2 * 2].view("<u2")
    chunk = 1 << 16
    for first in range(0, n, chunk):
        offsets = np.arange(first, min(n, first + chunk), dtype=np.int64) * step
        pid = data[offsets].astype(np.uint32) | (data[offsets + 1].astype(np.uint32) << 8) | (
            data[offsets + 2].astype(np.uint32) << 16) | (data[offsets + 3].astype(np.uint32) << 24)
        check = halfwords[(offsets + 6) // 2].astype(np.uint32)
        keep = (pid != 0) & (check != 0)
        offsets, check = offsets[keep], check[keep]
        if offsets.size == 0:
            continue
        seed = check.astype(np.uint64)
        total = np.zeros(offsets.size, dtype=np.uint64)
        for word in range(64):
            seed = (seed * LCG_MUL + LCG_ADD) & 0xFFFFFFFF
            plain = halfwords[(offsets + 8) // 2 + word].astype(np.uint64) ^ (seed >> 16)
            total += plain
        hits = offsets[(total & 0xFFFF) == check]
        for offset in hits.tolist():
            plain_struct = decrypt(bytes(data[offset:offset + size]))
            if plain_struct is None:
                continue
            mon = parse(plain_struct)
            if plausible(mon):
                found.append((base + offset, mon))
    return found
