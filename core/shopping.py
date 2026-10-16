"""Buy healing items and Poke Balls like a player does, through the real Mart UI.

A team with an empty bag can only heal at a Pokemon Center and never catch a
teammate. This module decides what is worth buying (``shopping_need`` and
``plan_purchases``) and drives the native shop screens with ordinary keys
(``MartController``), observing the shop's own RAM (item list, list cursor,
quantity) instead of counting blind presses. Nothing here writes RAM.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

from core.bridge.ffi_mgba import GBABridgeProtocol

# Items-pocket healing items usable in and out of battle, with restored HP.
HEALING_ITEMS: dict[str, int] = {
    "ITEM_POTION": 20, "ITEM_SUPER_POTION": 50, "ITEM_HYPER_POTION": 200, "ITEM_MAX_POTION": 999,
    "ITEM_FULL_RESTORE": 999, "ITEM_FRESH_WATER": 50, "ITEM_SODA_POP": 60, "ITEM_LEMONADE": 80,
    "ITEM_MOOMOO_MILK": 100, "ITEM_ENERGY_POWDER": 50, "ITEM_ENERGY_ROOT": 200, "ITEM_BERRY_JUICE": 20,
}
BALLS: tuple[str, ...] = ("ITEM_POKE_BALL", "ITEM_GREAT_BALL", "ITEM_ULTRA_BALL")
_MONEY_OFFSET = {"emerald": 0x490, "firered": 0x290}


def read_money(bridge: GBABridgeProtocol, game: str) -> int:
    memory = bridge.memory_map
    key = bridge.read_u32(bridge.get_save_block2_dynamic_ptr() + memory.encryption_key_offset)
    money = bridge.read_u32(bridge.get_save_block1_dynamic_ptr() + _MONEY_OFFSET[game]) ^ key
    return money if money <= 999_999 else 0


def item_ids(constants: Mapping[str, int], names: Sequence[str]) -> dict[int, str]:
    return {int(constants[name]): name for name in names if name in constants}


def healing_stock(inventory: Mapping[int, int], constants: Mapping[str, int]) -> int:
    """HP the bag can restore with ordinary healing items (Max Potion counts 200)."""
    total = 0
    for item, name in item_ids(constants, tuple(HEALING_ITEMS)).items():
        total += min(200, HEALING_ITEMS[name]) * int(inventory.get(item, 0))
    return total


def ball_stock(inventory: Mapping[int, int], constants: Mapping[str, int]) -> int:
    return sum(int(inventory.get(item, 0)) for item in item_ids(constants, BALLS))


@dataclass(frozen=True)
class ShoppingNeed:
    heal_hp: int
    balls: int
    reason: str

    @property
    def any(self) -> bool:
        return self.heal_hp > 0 or self.balls > 0


def shopping_need(inventory: Mapping[int, int], party: Sequence[Mapping[str, Any]],
                  constants: Mapping[str, int], wants_balls: bool) -> ShoppingNeed:
    """Stock targets: about two full heals of the strongest member, ten balls."""
    healthy = [mon for mon in party if not mon.get("is_egg", False)]
    if not healthy:
        return ShoppingNeed(0, 0, "no team")
    lead = max(healthy, key=lambda mon: (int(mon["level"]), int(mon["maxHP"])))
    heal_target = min(600, 2 * int(lead["maxHP"]) + 40)
    heal_deficit = max(0, heal_target - healing_stock(inventory, constants))
    balls = max(0, 10 - ball_stock(inventory, constants)) if wants_balls and len(healthy) < 6 else 0
    heal = heal_deficit if heal_deficit >= max(40, heal_target // 3) else 0
    balls = balls if balls >= 4 else 0
    parts = []
    if heal:
        parts.append(f"healing items for {heal} HP")
    if balls:
        parts.append(f"{balls} Poke Balls")
    return ShoppingNeed(heal, balls, " and ".join(parts) or "stock is sufficient")


def plan_purchases(available: Sequence[int], money: int, need: ShoppingNeed,
                   price_of: Callable[[int], int], constants: Mapping[str, int],
                   lead_max_hp: int, budget_fraction: float = 0.7) -> list[tuple[int, int]]:
    """Quantities per item, spending at most ``budget_fraction`` of the money."""
    budget = int(money * budget_fraction)
    plan: list[tuple[int, int]] = []
    healing = item_ids(constants, tuple(HEALING_ITEMS))
    balls = item_ids(constants, BALLS)
    if need.heal_hp > 0:
        options: list[tuple[float, int, int]] = []
        for item in available:
            name = healing.get(item)
            if name is None or name in ("ITEM_FULL_RESTORE", "ITEM_MAX_POTION"):
                continue
            price = price_of(item)
            heal = HEALING_ITEMS[name]
            if price <= 0:
                continue
            useful = min(heal, max(20, lead_max_hp))
            options.append((price / useful, item, heal))
        if options:
            _, item, heal = min(options)
            price = price_of(item)
            quantity = min(math.ceil(need.heal_hp / min(heal, max(20, lead_max_hp))), budget // price, 20)
            if quantity > 0:
                plan.append((item, quantity))
                budget -= quantity * price
    if need.balls > 0:
        offered = [item for item in available if item in balls]
        if offered:
            # Great Balls once money is plentiful; ordinary Poke Balls otherwise.
            ranked = sorted(offered, key=lambda item: price_of(item))
            item = ranked[-1] if money >= 6000 and len(ranked) > 1 else ranked[0]
            price = price_of(item)
            quantity = min(need.balls, budget // max(1, price), 20)
            if quantity > 0:
                plan.append((item, quantity))
    return plan


@dataclass
class MartController:
    """Drive the shop screens: BUY, pick each planned item and quantity, leave.

    ``prepare`` supplies the need when a clerk is about to be addressed; the
    item list is read from the opened buy menu, so a Mart that sells less than
    expected simply yields a shorter plan.
    """
    bridge: GBABridgeProtocol
    game: str
    symbols: Mapping[str, int]
    price_of: Callable[[int], int]
    constants: Mapping[str, int]
    pulse_frames: int = 2
    release_frames: int = 5
    need: ShoppingNeed | None = None
    lead_max_hp: int = 60
    plan: list[tuple[int, int]] | None = None
    purchased: list[tuple[int, int]] = field(default_factory=list)
    finished: bool = False
    last_event: str = ""
    _held: int = 0
    _hold_left: int = 0
    _release_left: int = 0
    _inventory_before: dict[int, int] = field(default_factory=dict)
    # Fades between the shop menu and the buy screen show no shop task for a
    # few frames; the session survives such gaps instead of ending early.
    _in_session: bool = False
    _gap: int = 0
    gap_frames: int = 150

    def prepare(self, need: ShoppingNeed, lead_max_hp: int) -> None:
        self.need, self.lead_max_hp = need, lead_max_hp
        self.plan, self.purchased, self.finished = None, [], False
        self.last_event = ""
        try:
            self._inventory_before = dict(self.bridge.get_inventory())
        except (ValueError, RuntimeError):
            self._inventory_before = {}

    def reset(self) -> None:
        self.need = self.plan = None
        self.finished = False
        self._held = self._hold_left = self._release_left = 0
        self._in_session, self._gap = False, 0

    # -- RAM observation -------------------------------------------------
    def _active_tasks(self) -> set[int]:
        tasks = self.bridge.memory_map.tasks
        return {self.bridge.read_u32(tasks + 40 * slot) & ~1 for slot in range(16)
                if self.bridge.read_u8(tasks + 40 * slot + 4)}

    def _task_slot(self, function: str) -> int | None:
        address = self.symbols.get(function)
        if address is None:
            return None
        tasks = self.bridge.memory_map.tasks
        for slot in range(16):
            if (self.bridge.read_u8(tasks + 40 * slot + 4) and
                    self.bridge.read_u32(tasks + 40 * slot) & ~1 == address):
                return slot
        return None

    def screen(self) -> str:
        """shop_menu, buy_list, quantity, buy_message, or '' outside a shop."""
        active = self._active_tasks()
        names = {name: self.symbols.get(name) for name in ("Task_ShopMenu", "Task_BuyMenu",
                                                          "Task_BuyHowManyDialogueHandleInput")}
        if names["Task_BuyHowManyDialogueHandleInput"] in active:
            return "quantity"
        if names["Task_BuyMenu"] in active:
            return "buy_list"
        if names["Task_ShopMenu"] in active:
            return "shop_menu"
        callback = self.bridge.read_u32(self.bridge.memory_map.main + 4) & ~1
        if callback == self.symbols.get("CB2_BuyMenu"):
            return "buy_message"
        return ""

    def _read_words(self, pointer: int, count: int) -> list[int]:
        raw = self.bridge.read_memory(pointer, 2 * count)
        return [int.from_bytes(raw[2 * index:2 * index + 2], "little") for index in range(count)]

    def item_list(self) -> list[int]:
        if self.game == "emerald":
            info = self.symbols["sMartInfo"]
            pointer, count = self.bridge.read_u32(info + 8), self.bridge.read_u16(info + 12)
        else:
            data = self.symbols["sShopData"]
            pointer, count = self.bridge.read_u32(data + 4), self.bridge.read_u16(data + 0x10)
        if not 0 < count <= 64 or not 0x08000000 <= pointer < 0x0A000000 and not 0x02000000 <= pointer < 0x02040000:
            return []
        return self._read_words(pointer, count)

    def list_index(self) -> int:
        if self.game == "emerald":
            data = self.bridge.read_u32(self.symbols["sShopData"])
            return self.bridge.read_u16(data + 0x2008) + self.bridge.read_u16(data + 0x2006)
        data = self.symbols["sShopData"]
        return self.bridge.read_u16(data + 0x0E) + self.bridge.read_u16(data + 0x0C)

    def max_quantity(self) -> int:
        if self.game == "emerald":
            data = self.bridge.read_u32(self.symbols["sShopData"])
            return self.bridge.read_u8(data + 0x200A)
        return self.bridge.read_u16(self.symbols["sShopData"] + 0x14)

    def quantity(self) -> int | None:
        slot = self._task_slot("Task_BuyHowManyDialogueHandleInput")
        if slot is None:
            return None
        value = self.bridge.read_u16(self.bridge.memory_map.tasks + 40 * slot + 8 + 2)
        return value - 0x10000 if value & 0x8000 else value

    # -- key generation ----------------------------------------------------
    def _pulse(self, key: int) -> int:
        self._held, self._hold_left, self._release_left = key, self.pulse_frames - 1, self.release_frames
        return key

    def _pending(self) -> tuple[int, int] | None:
        if self.plan is None:
            return None
        inventory = dict(self.bridge.get_inventory())
        for item, quantity in self.plan:
            gained = int(inventory.get(item, 0)) - int(self._inventory_before.get(item, 0))
            if gained < quantity:
                return item, quantity - max(0, gained)
        return None

    def keys(self) -> int | None:
        """Keys for this frame while a shop screen is open; None outside shops."""
        if self._hold_left:
            self._hold_left -= 1
            return self._held
        if self._release_left:
            self._release_left -= 1
            return 0
        screen = self.screen()
        if not screen:
            if self._in_session and self._gap < self.gap_frames:
                self._gap += 1
                return 0
            self._in_session, self._gap = False, 0
            return None
        self._in_session, self._gap = True, 0
        if screen == "shop_menu":
            if self.need is None or self.finished or (self.plan is not None and self._pending() is None):
                self.finished = True
                return self._pulse(self.bridge.KEY_B)
            # BUY is the default (and non-wrapping) first entry of the menu.
            return self._pulse(self.bridge.KEY_A)
        if screen == "buy_list":
            available = self.item_list()
            if self.plan is None and self.need is not None:
                money = read_money(self.bridge, self.game)
                self.plan = plan_purchases(available, money, self.need, self.price_of, self.constants,
                                           self.lead_max_hp)
                self.last_event = f"plan:{self.plan}"
            pending = self._pending()
            if pending is None or pending[0] not in available:
                if pending is not None and self.plan is not None:
                    self.plan = [entry for entry in self.plan if entry[0] != pending[0]]
                    if self._pending() is not None:
                        return self._pulse(0)
                self.finished = True
                return self._pulse(self.bridge.KEY_B)
            target = available.index(pending[0])
            current = self.list_index()
            if current < target:
                return self._pulse(self.bridge.KEY_DOWN)
            if current > target:
                return self._pulse(self.bridge.KEY_UP)
            return self._pulse(self.bridge.KEY_A)
        if screen == "quantity":
            pending = self._pending()
            value = self.quantity()
            if pending is None or value is None:
                return self._pulse(self.bridge.KEY_B)
            desired = max(1, min(pending[1], self.max_quantity() or 1))
            if value < desired:
                return self._pulse(self.bridge.KEY_RIGHT if desired - value >= 10 else self.bridge.KEY_UP)
            if value > desired:
                return self._pulse(self.bridge.KEY_DOWN)
            return self._pulse(self.bridge.KEY_A)
        # Confirmation text, the default-YES purchase prompt and "Here you are!".
        return self._pulse(self.bridge.KEY_A)

    def bought(self) -> dict[int, int]:
        try:
            inventory = dict(self.bridge.get_inventory())
        except (ValueError, RuntimeError):
            return {}
        return {item: int(inventory.get(item, 0)) - int(self._inventory_before.get(item, 0))
                for item in set(inventory) | set(self._inventory_before)
                if int(inventory.get(item, 0)) > int(self._inventory_before.get(item, 0))}
