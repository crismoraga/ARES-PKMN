"""RAM-observed title, introduction and naming controls using only GBA keys."""
from __future__ import annotations

import struct
from dataclasses import dataclass

from core.bridge.ffi_mgba import GBABridgeProtocol
from core.game_data import Game, RomSymbol, decode_gba_text, load_symbols


@dataclass(frozen=True)
class NativeUIStatus:
    """Observable callback identity and whether an introduction is active."""
    callback: str
    in_field: bool
    naming: bool


class NativeUIController:
    """Navigate introductions and native character grids without RAM writes.

    The owner calls choose_keys once per frame and advances the hardware itself.
    Cursor and text feedback prevent fixed macros from desynchronizing names.
    """

    def __init__(self, bridge: GBABridgeProtocol, game: Game, player_name: str = "ARES") -> None:
        if not player_name or len(player_name) > 7 or not player_name.isascii() or not player_name.isalpha():
            raise ValueError("player_name must contain1..7 ASCII letters")
        self.bridge = bridge
        self.game = game
        self.player_name = player_name.upper()
        self.symbols = load_symbols(game)
        self.functions = {v.address & ~1: k for k, v in self.symbols.items()
                          if 0x08000000 <= v.address < 0x09000000 and v.size > 0}
        self.frame = 0
        self.last_callback = ""

    def _symbol(self, name: str) -> RomSymbol:
        """Require a symbol from the exact official build."""
        return self.symbols[name]

    def status(self) -> NativeUIStatus:
        """Read callback2, rather than assuming nonzero save pointers imply play."""
        address = self.bridge.read_u32(self._symbol("gMain").address + 4) & ~1
        name = self.functions.get(address, f"callback_{address:08x}")
        return NativeUIStatus(name, name in {"CB2_Overworld", "CB2_OverworldBasic"},
                              name == "CB2_NamingScreen")

    def _naming_key(self) -> int:
        """Find a character in the ROM keyboard and navigate from the RAM cursor."""
        pointer = self.bridge.read_u32(self._symbol("sNamingScreen").address)
        if not 0x02000000 <= pointer <= 0x0203DFFF:
            return 0
        if self.bridge.read_u8(pointer + 0x1E10) != 2:
            return 0
        text = decode_gba_text(self.bridge.read_memory(pointer + 0x1800, 16))
        template = self.bridge.read_u8(pointer + 0x1E2C)
        desired = "BLUE" if self.game == "firered" and template == 4 else self.player_name
        sprite_id = self.bridge.read_u8(pointer + 0x1E23)
        if sprite_id >= 64:
            raise ValueError(f"Invalid naming cursor sprite{sprite_id}")
        cursor = self._symbol("gSprites").address + 68 * sprite_id + 0x2E
        x, y = struct.unpack("<hh", self.bridge.read_memory(cursor, 4))
        if not desired.startswith(text):
            return self.bridge.KEY_B
        if text == desired:
            return self.bridge.KEY_A if x == 8 and y == 2 else self.bridge.KEY_START
        page = self.bridge.read_u8(pointer + 0x1E22)
        if page != 1:
            return self.bridge.KEY_SELECT
        keyboard = self.bridge.read_memory(self._symbol("sKeyboardChars").address + 32, 32)
        char = ord(desired[len(text)]) - ord("A") + 0xBB
        index = keyboard.find(bytes([char]))
        if index < 0:
            raise ValueError(f"Character unavailable on native keyboard:{desired[len(text)]}")
        target_x, target_y = index % 8, index // 8
        if y != target_y:
            return self.bridge.KEY_DOWN if y < target_y else self.bridge.KEY_UP
        if x != target_x:
            return self.bridge.KEY_RIGHT if x < target_x else self.bridge.KEY_LEFT
        return self.bridge.KEY_A

    def choose_keys(self) -> int | None:
        """Return a pulse mask in introductions, or None once overworld takes over."""
        self.frame += 1
        status = self.status()
        self.last_callback = status.callback
        if status.in_field:
            return None
        if self.frame % 12 >= 2:
            return 0
        if status.naming:
            return self._naming_key()
        return self.bridge.KEY_A
