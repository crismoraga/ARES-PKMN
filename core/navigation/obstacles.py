"""Source object templates as obstacles, consistent for live and static routing.

Generation-three only spawns object events near the camera. A Cut tree twelve
tiles away is absent from gObjectEvents, so a planner reading only live actors
sees a corridor that does not exist; a static planner that ignores templates
sees the same phantom corridor. Both planners share this model, so replanning
after one step can no longer flip between two contradictory routes.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
import struct
from typing import Literal, Protocol

from core.bridge.ffi_mgba import GBABridgeProtocol
from core.bridge.gba_structs import ObjectEventStruct
from core.navigation.script_gates import GateWorld, RamGateWorld, gate_state, gates_by_map
from core.navigation.source_data import MapKey, load_map_catalog

ObstacleKind = Literal["cut", "rock_smash", "strength", "item", "object"]
XY = tuple[int, int]
Edge = tuple[XY, XY]
# Extra tiles charged for stopping, facing and running an HM field script.
HM_CLEAR_COST = 8.0
# An item ball in the way is picked up (A while facing it): Rocket Hideout's
# spin maze can only be crossed through the tile of its Moon Stone.
ITEM_PICKUP_COST = 6.0
# gSaveBlock1Ptr->objectEventTemplates: 64 runtime copies of 24-byte templates.
# Map scripts move story actors with setobjectxyperm, which edits these copies.
_RUNTIME_TEMPLATES = {"emerald": 0xC70, "firered": 0x8E0}
_TEMPLATE_SIZE, _TEMPLATE_COUNT = 24, 64
_GRAPHICS: tuple[tuple[str, ObstacleKind], ...] = (
    ("OBJ_EVENT_GFX_CUT_TREE", "cut"), ("OBJ_EVENT_GFX_CUTTABLE_TREE", "cut"),
    ("OBJ_EVENT_GFX_ROCK_SMASH_ROCK", "rock_smash"), ("OBJ_EVENT_GFX_BREAKABLE_ROCK", "rock_smash"),
    ("OBJ_EVENT_GFX_PUSHABLE_BOULDER", "strength"), ("OBJ_EVENT_GFX_ITEM_BALL", "item"))


class FlagReader(Protocol):
    def get_event_flag(self, flag_id: int) -> bool:
        """Read one persistent event flag."""


class FlagSnapshot:
    """One RAM read of flags, variables and bag, shared by a whole route search.

    It describes maps the player is *not* standing on: temporary flags and
    VAR_TEMP_* variables restart at zero whenever such a map is entered.
    It also answers gate conditions (``core.navigation.script_gates``).
    """

    def __init__(self, bridge: GBABridgeProtocol) -> None:
        memory = bridge.memory_map
        size = memory.vars_offset - memory.flags_offset
        base = bridge.get_save_block1_dynamic_ptr()
        self.data = bridge.read_memory(base + memory.flags_offset, size)
        try:
            self.vars = bytes(bridge.read_memory(base + memory.vars_offset, 0x200))
        except (ValueError, RuntimeError):
            self.vars = b""
        try:
            inventory = bridge.get_inventory()
            self.inventory: dict[int, int] = {int(item): int(count) for item, count in inventory.items()}
        except (ValueError, RuntimeError, AttributeError):
            self.inventory = {}
        self._bridge = bridge
        self._party_moves: tuple[tuple[int, ...], ...] | None = None
        self.game = str(getattr(memory, "game", ""))

    @property
    def signature(self) -> bytes:
        """Everything a cached obstacle model of another map may depend on."""
        items = b"".join(item.to_bytes(2, "little") for item in sorted(self.inventory) if self.inventory[item] > 0)
        return bytes(self.data) + b"|" + self.vars + b"|" + items

    def get_event_flag(self, flag_id: int) -> bool:
        if not 0 <= flag_id < len(self.data) * 8:
            return False
        return bool(self.data[flag_id >> 3] & (1 << (flag_id & 7)))

    def _constant(self, name: str) -> int | None:
        if self.game not in ("firered", "emerald"):
            return None
        value = load_map_catalog(self.game).constants.get(name)
        return None if value is None else int(value)

    # GateWorld protocol (another map: temporary state restarts at zero).
    def flag(self, name: str) -> bool | None:
        if name.startswith("TRAINER:"):
            trainer, start = self._constant(name.split(":", 1)[1]), self._constant("TRAINER_FLAGS_START")
            flag_id = None if trainer is None or start is None else start + trainer
        else:
            flag_id = self._constant(name)
        if flag_id is None:
            return None
        if flag_id <= int(self._constant("TEMP_FLAGS_END") or 0x1F):
            return False
        return self.get_event_flag(flag_id)

    def var(self, name: str) -> int | None:
        identifier = self._constant(name)
        if identifier is None or not 0x4000 <= identifier < 0x4100:
            return None
        if identifier < 0x4010:
            return 0
        offset = 2 * (identifier - 0x4000)
        if offset + 2 > len(self.vars):
            return None
        return int.from_bytes(self.vars[offset:offset + 2], "little")

    def item_count(self, name: str) -> int | None:
        item = self._constant(name)
        return None if item is None else self.inventory.get(item, 0)

    def party_move_slot(self, name: str) -> int | None:
        move = self._constant(name)
        if move is None:
            return None
        if self._party_moves is None:
            try:
                party = self._bridge.get_decrypted_player_party()
            except (ValueError, RuntimeError, AttributeError):
                return None
            self._party_moves = tuple(tuple(int(value) for value in mon.get("moves", [])) for mon in party)
        return next((index for index, moves in enumerate(self._party_moves) if move in moves), 6)


@dataclass(frozen=True)
class TemplateObject:
    """One source object event, reduced to what collision planning needs."""
    local_id: int
    xy: XY
    kind: ObstacleKind
    mobile: bool
    flag: int | None
    temporary: bool


@dataclass(frozen=True)
class ObstacleModel:
    """Tiles that block regardless of elevation, HM-removable tiles with costs,
    and directed steps into armed pushback gates (guards, sandstorms)."""
    blocked: frozenset[XY]
    removable: tuple[tuple[XY, float], ...]
    gate_edges: frozenset[Edge] = frozenset()

    @property
    def removable_costs(self) -> dict[XY, float]:
        return dict(self.removable)


def gate_entry_edges(tile: XY, pushback: XY) -> frozenset[Edge]:
    """Steps into a gate tile that its script would undo.

    The trigger walks the player along ``pushback``. Entering while moving
    with the push (leaving Saffron northbound) is harmless; entering against
    or across it returns the player to the side they came from.
    """
    edges: set[Edge] = set()
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if pushback[0] * dx + pushback[1] * dy > 0:
            continue
        for distance in (1, 2):  # a ledge jump lands two tiles away
            edges.add(((tile[0] - distance * dx, tile[1] - distance * dy), tile))
    return frozenset(edges)


def _mobile(movement_type: str) -> bool:
    """Wanderers and patrols leave their template tile; in-place animations do not."""
    if "IN_PLACE" in movement_type:
        return False
    return "WANDER" in movement_type or "WALK_" in movement_type or "COPY_PLAYER" in movement_type


@lru_cache(maxsize=4)
def _graphics_kinds(game: str) -> dict[int, ObstacleKind]:
    constants = load_map_catalog(game).constants
    return {constants[name]: kind for name, kind in _GRAPHICS if name in constants}


def live_obstacle_kind(game: str, actor: ObjectEventStruct) -> ObstacleKind | None:
    """Classify a spawned actor as an HM obstacle by its native graphics ID."""
    return _graphics_kinds(game).get(int(actor.graphicsId))


@lru_cache(maxsize=4096)
def template_objects(game: str, key: MapKey) -> tuple[TemplateObject, ...]:
    """Every in-bounds object template of a map, classified once."""
    catalog = load_map_catalog(game)
    raw = catalog.maps.get(key)
    if raw is None:
        return ()
    constants = catalog.constants
    kinds = _graphics_kinds(game)
    width, height = int(raw["width"]), int(raw["height"])
    temp_end = int(constants.get("TEMP_FLAGS_END", 31))
    result: list[TemplateObject] = []
    for index, event in enumerate(raw.get("object_events", [])):
        if str(event.get("type", "object")) == "clone":
            continue  # A clone renders another map's object across a border.
        try:
            x, y = int(event["x"]), int(event["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= x < width and 0 <= y < height):
            continue
        graphics = constants.get(str(event.get("graphics_id", "")), -1)
        script = str(event.get("script", ""))
        kind: ObstacleKind = kinds.get(graphics, "object")
        if kind == "object":
            if "CutTree" in script:
                kind = "cut"
            elif "RockSmash" in script:
                kind = "rock_smash"
            elif "StrengthBoulder" in script:
                kind = "strength"
        flag_name = str(event.get("flag", "0"))
        flag = None if flag_name in ("0", "0x0", "None", "") else constants.get(flag_name)
        result.append(TemplateObject(index + 1, (x, y), kind, _mobile(str(event.get("movement_type", ""))),
                                     flag, flag is not None and flag <= temp_end))
    return tuple(result)


@lru_cache(maxsize=4)
def _movement_names(game: str) -> dict[int, str]:
    constants = load_map_catalog(game).constants
    names: dict[int, str] = {}
    for name, value in constants.items():
        if name.startswith("MOVEMENT_TYPE_"):
            names.setdefault(value, name)
    return names


def runtime_template_objects(bridge: GBABridgeProtocol, game: str, key: MapKey) -> tuple[TemplateObject, ...] | None:
    """Current-map templates as edited by map scripts, or None if RAM is not usable.

    The runtime copy preserves source order, so script names for classification
    come from the source template with the same local ID.
    """
    offset = _RUNTIME_TEMPLATES.get(game)
    catalog = load_map_catalog(game)
    raw = catalog.maps.get(key)
    if offset is None or raw is None:
        return None
    try:
        data = bridge.read_memory(bridge.get_save_block1_dynamic_ptr() + offset, _TEMPLATE_SIZE * _TEMPLATE_COUNT)
    except (ValueError, RuntimeError):
        return None
    width, height = int(raw["width"]), int(raw["height"])
    source = {template.local_id: template for template in template_objects(game, key)}
    kinds, names = _graphics_kinds(game), _movement_names(game)
    temp_end = int(catalog.constants.get("TEMP_FLAGS_END", 31))
    result: list[TemplateObject] = []
    # Loading a map copies only its own templates; later slots keep stale
    # actors of maps visited before (Celadon's NPCs inside the Rocket Hideout).
    count = min(_TEMPLATE_COUNT, len(raw.get("object_events", []) or []))
    for index in range(count):
        start = index * _TEMPLATE_SIZE
        local_id, graphics, kind_id = data[start], data[start + 1], data[start + 2]
        if local_id == 0 or (game == "firered" and kind_id == 1):
            continue  # unused slot, or a FireRed clone of a neighboring map's actor
        x, y = struct.unpack_from("<hh", data, start + 4)
        if not (0 <= x < width and 0 <= y < height):
            continue
        flag = struct.unpack_from("<H", data, start + 20)[0]
        template = source.get(local_id)
        kind: ObstacleKind = kinds.get(graphics, template.kind if template is not None else "object")
        result.append(TemplateObject(local_id, (x, y), kind, _mobile(names.get(data[start + 9], "")),
                                     flag or None, 0 < flag <= temp_end))
    # A transitioning save block can briefly hold another map's templates.
    if source and not result:
        return None
    return tuple(result)


def _flag(bridge: FlagReader | None, flag: int) -> bool:
    if bridge is None:
        return False
    try:
        return bridge.get_event_flag(flag)
    except ValueError:
        return False


def _gate_world(game: str, bridge: GBABridgeProtocol | FlagReader | None,
                current_map: bool) -> GateWorld | None:
    if isinstance(bridge, FlagSnapshot):
        return bridge
    if bridge is not None and all(callable(getattr(bridge, name, None))
                                  for name in ("flag", "var", "item_count", "party_move_slot")):
        return bridge  # type: ignore[return-value]  # any explicit GateWorld (tests, planners)
    if current_map and isinstance(bridge, GBABridgeProtocol):
        try:
            return RamGateWorld(bridge, game, current_map=True)
        except (ValueError, RuntimeError, AttributeError):
            return None
    return None


def closed_gate_edges(game: str, key: MapKey, bridge: GBABridgeProtocol | FlagReader | None,
                      current_map: bool, learned_edges: Iterable[Edge] = (),
                      static_gates: bool = True) -> frozenset[Edge]:
    """Directed entries into gates that are armed and unsatisfied now."""
    edges: set[Edge] = set(learned_edges)
    if not static_gates:
        return frozenset(edges)
    gates = gates_by_map(game).get(key, ())
    if not gates:
        return frozenset(edges)
    world = _gate_world(game, bridge, current_map)
    if world is None:
        return frozenset(edges)
    for gate in gates:
        if gate_state(gate, world) is True:
            edges.update(gate_entry_edges(gate.tile, gate.pushback))
    return frozenset(edges)


def obstacle_model(game: str, key: MapKey, *, bridge: GBABridgeProtocol | FlagReader | None = None,
                   live_objects: Sequence[ObjectEventStruct] | None = None,
                   player_object_id: int = 0, hm_usable: Iterable[str] = (),
                   current_map: bool = False, learned_edges: Iterable[Edge] = (),
                   static_gates: bool = True, story_actors: bool = True) -> ObstacleModel:
    """Merge live actors with templates that are absent because they are off-camera.

    Temporary flags (Cut trees, some scripted actors) reset whenever a map is
    entered, so they hide an object only on the map the player currently stands
    on. Persistent flags (collected items, story actors) are read from RAM when
    a bridge is available. Spawned HM obstacles become removable only when the
    corresponding field move is usable now. Armed pushback gates whose pass
    conditions are unmet (and pushbacks learned at runtime) forbid the steps
    their scripts would undo.
    """
    usable = frozenset(hm_usable)
    loaded: set[int] = set()
    removable: dict[XY, float] = {}
    if live_objects is not None:
        for index, actor in enumerate(live_objects):
            if index == player_object_id or not int(actor.flags) & 1:
                continue
            if (int(actor.mapGroup), int(actor.mapNum)) == key:
                loaded.add(int(actor.localId))
            kind = live_obstacle_kind(game, actor)
            if kind == "item":
                removable[(int(actor.currentCoords_x) - 7, int(actor.currentCoords_y) - 7)] = ITEM_PICKUP_COST
            elif kind is not None and kind in usable:
                removable[(int(actor.currentCoords_x) - 7, int(actor.currentCoords_y) - 7)] = HM_CLEAR_COST
    blocked: set[XY] = set()
    templates = (runtime_template_objects(bridge, game, key)
                 if current_map and isinstance(bridge, GBABridgeProtocol) else None)
    for template in templates if templates is not None else template_objects(game, key):
        if template.local_id in loaded:
            continue
        if template.flag is not None:
            hidden = (current_map and _flag(bridge, template.flag)) if template.temporary else _flag(bridge, template.flag)
            if hidden:
                continue
        if template.kind == "object" and template.mobile:
            continue  # A wandering NPC is a transient live obstruction, not a wall.
        if not story_actors and template.kind == "object" and template.flag is not None and not template.temporary:
            continue  # what-if search: a story actor that some event hides (Aqua's queue)
        if template.kind in ("cut", "rock_smash") and template.kind in usable:
            removable[template.xy] = HM_CLEAR_COST
            continue
        if template.kind == "item":
            removable[template.xy] = ITEM_PICKUP_COST
            continue
        blocked.add(template.xy)
    blocked.difference_update(removable)
    gate_edges = closed_gate_edges(game, key, bridge, current_map, learned_edges, static_gates)
    return ObstacleModel(frozenset(blocked), tuple(sorted(removable.items())), gate_edges)
