"""Universal player: GB screen parser (dialog box, prompt, menus) on synthetic tile maps."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ares_u.universal.adapter import (CURSOR, FRAME_BL, FRAME_BR, FRAME_H, FRAME_TL, FRAME_TR, FRAME_V,
                                      PROMPT_ARROW, SCREEN_H, SCREEN_W, TILE_SPACE, GBScreenParser, base_charmap)


def blank() -> NDArray[np.uint8]:
    return np.full((SCREEN_H, SCREEN_W), TILE_SPACE, dtype=np.uint8)


def encode(text: str) -> list[int]:
    inverse = {v: k for k, v in base_charmap().items() if len(v) == 1}
    return [inverse[ch] for ch in text]


def put(tiles: NDArray[np.uint8], row: int, col: int, text: str) -> None:
    codes = encode(text)
    tiles[row, col:col + len(codes)] = codes


def frame(tiles: NDArray[np.uint8], top: int, left: int, bottom: int, right: int) -> None:
    tiles[top, left], tiles[top, right] = FRAME_TL, FRAME_TR
    tiles[bottom, left], tiles[bottom, right] = FRAME_BL, FRAME_BR
    tiles[top, left + 1:right] = FRAME_H
    tiles[bottom, left + 1:right] = FRAME_H
    tiles[top + 1:bottom, left] = FRAME_V
    tiles[top + 1:bottom, right] = FRAME_V


def parser() -> GBScreenParser:
    return GBScreenParser(base_charmap())


def test_decode_roundtrip_and_unknown_tiles() -> None:
    p = parser()
    assert p.decode(encode("OAK: Hola!")) == "OAK: Hola!"
    assert p.decode([0x01]) == "·"
    assert p.unknown == {0x01: 1}


def test_dialog_box_lines_and_prompt() -> None:
    tiles = blank()
    frame(tiles, 12, 0, 17, 19)
    put(tiles, 14, 1, "No te vayas!")
    put(tiles, 16, 1, "Ven conmigo")
    tiles[16, 18] = PROMPT_ARROW
    info = parser().screen(tiles)
    assert info.textbox
    assert info.lines[0] == "No te vayas!"
    assert info.lines[1].startswith("Ven conmigo")
    assert info.prompt
    assert info.menu is None


def test_no_textbox_without_full_frame() -> None:
    tiles = blank()
    put(tiles, 14, 1, "HELLO")
    info = parser().screen(tiles)
    assert not info.textbox and info.lines == ()


def test_yes_no_menu_inside_box() -> None:
    tiles = blank()
    frame(tiles, 7, 14, 11, 19)
    put(tiles, 8, 16, "SI")
    put(tiles, 10, 16, "NO")
    tiles[10, 15] = CURSOR
    menu = parser().menu(tiles)
    assert menu is not None
    assert menu.options == ("SI", "NO")
    assert menu.selected == 1
    assert (menu.cursor_row, menu.cursor_col) == (10, 15)


def test_two_by_two_battle_menu_is_split() -> None:
    tiles = blank()
    frame(tiles, 12, 8, 17, 19)
    put(tiles, 14, 10, "LUCHA")
    put(tiles, 14, 16, "PK")
    put(tiles, 16, 10, "OBJ")
    put(tiles, 16, 16, "SAL")
    tiles[14, 9] = CURSOR
    menu = parser().menu(tiles)
    assert menu is not None
    assert menu.options == ("LUCHA", "PK", "OBJ", "SAL")
    assert menu.selected == 0


def test_two_cursors_is_not_a_menu() -> None:
    tiles = blank()
    tiles[3, 3] = CURSOR
    tiles[5, 3] = CURSOR
    assert parser().menu(tiles) is None
