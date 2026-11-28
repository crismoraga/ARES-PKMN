"""Game-agnostic battle brain API: what any game adapter hands the brain, and what it gets back.

Every adapter (FireRed/Emerald RAM, Gen1/Gen2 cartridges, NDS games, the Stadium) builds a
``BattleView`` from what the player can see and calls ``BattleBrain.choose(view)``. The required
fields are exactly the team contract; the trailing fields with defaults are optional extras that an
adapter may fill when it knows them (old adapters keep working unchanged).

Conventions
-----------
* Types are lowercase English names (``"fire"``); ``canonical_type`` also accepts the Spanish names
  of the Spanish cartridges and the Gen-3 ROM spelling (``"Fire"``, ``"???"``).
* ``MonView.moves`` keeps the game's move-slot order: ``BrainChoice(kind="move", index=i)`` means
  slot ``i`` of ``view.me.moves``. Moves with ``pp == 0`` stay in the list.
* ``BrainChoice(kind="switch", index=j)`` means ``view.party[j]``; ``kind="run"`` flees a wild
  battle; ``kind="item"`` with ``scores["catch"] == 1`` asks the caller to throw its best ball (the
  brain does not see the bag).
* Gen 1-3 move ids coincide with the Gen-3 ids (Gen 2 and Gen 3 only appended moves), so a Gen-1/2
  adapter should pass its native move id as ``gen3_move_id``.
* For the enemy an honest adapter passes only public information: species, level, HP bar (``hp``
  and ``max_hp`` are read as a fraction), types, status, stat stages, the moves it has revealed and
  ``stats=None``. The brain estimates the rest (base stats, level-up learnset).

Type charts
-----------
``type_chart(generation)`` returns the full attack x defence table of a generation:

* Gen 1 (15 types, no Dark/Steel) with the cartridge's real quirks: Ghost has *no effect* on
  Psychic (the famous RBY bug: 0x instead of 2x), Bug is super effective on Poison (2x), Poison is
  super effective on Bug (2x) and Ice is neutral on Fire (1x).
* Gen 2-5 (17 types): the standard chart; Steel resists Ghost and Dark. A unit test checks every
  cell against the Gen-3 ROM table shipped in ``data/stadium_dex.json``.
* Gen 6+ (18 types): Fairy added and Steel no longer resists Ghost/Dark.

Damage category: Gen 1-3 decide physical/special by the move's *type* (``category_for``); from
Gen 4 on it is a property of each move, so a Gen-4+ adapter must pass ``MoveView.category``.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

# ----------------------------------------------------------------------------- contract dataclasses


@dataclass
class MoveView:
    name: str
    type: str
    power: int
    accuracy: int
    pp: int
    category: str  # "physical" | "special" | "status"
    gen3_move_id: int | None = None
    # optional extras
    max_pp: int | None = None
    priority: int | None = None


@dataclass
class MonView:
    species: str
    national_dex: int
    level: int
    hp: int
    max_hp: int
    types: tuple[str, ...]
    status: str  # "" | "par" | "slp" | "psn" | "brn" | "frz" | "tox"
    moves: list[MoveView]
    stats: dict[str, int] | None = None  # atk, def, spa, spd, spe
    # optional extras
    stages: dict[str, int] | None = None  # atk, def, spa, spd, spe, acc, eva in -6..+6
    ability: str = ""
    item: str = ""

    @property
    def hp_fraction(self) -> float:
        return max(0.0, min(1.0, self.hp / max(1, self.max_hp)))

    @property
    def fainted(self) -> bool:
        return self.hp <= 0


@dataclass
class BattleView:
    game: str
    generation: int
    me: MonView
    enemy: MonView
    party: list[MonView]
    active_index: int
    is_wild: bool
    can_switch: bool = True
    can_run: bool = True
    turn: int = 0
    # optional extras
    enemy_remaining: int | None = None  # enemy members still able to fight, active included
    weather: str = ""
    legal_moves: tuple[int, ...] | None = None  # move slots the game allows now (Disable, Encore...)


@dataclass
class BrainChoice:
    kind: str  # "move" | "switch" | "run" | "item"
    index: int
    reason: str
    scores: dict[str, float] = field(default_factory=dict)


# ----------------------------------------------------------------------------- types

GEN1_TYPES: tuple[str, ...] = ("normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost",
                               "fire", "water", "grass", "electric", "psychic", "ice", "dragon")
GEN2_TYPES: tuple[str, ...] = GEN1_TYPES + ("dark", "steel")
GEN6_TYPES: tuple[str, ...] = GEN2_TYPES + ("fairy",)
UNKNOWN_TYPE = "???"
STATUSES = ("", "par", "slp", "psn", "brn", "frz", "tox")
CATEGORIES = ("physical", "special", "status")
# Gen 1-3: the type decides the damage category.
SPECIAL_TYPES = frozenset({"fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark"})
PHYSICAL_TYPES = frozenset({"normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel"})
# Gen-3 ROM type indices (``core.stadium.dex.TYPE_NAMES``); 9 is the typeless "???".
GEN3_TYPE_INDEX: dict[str, int] = {
    "normal": 0, "fighting": 1, "flying": 2, "poison": 3, "ground": 4, "rock": 5, "bug": 6, "ghost": 7,
    "steel": 8, UNKNOWN_TYPE: 9, "fire": 10, "water": 11, "grass": 12, "electric": 13, "psychic": 14, "ice": 15,
    "dragon": 16, "dark": 17}
GEN3_INDEX_TYPE: dict[int, str] = {index: name for name, index in GEN3_TYPE_INDEX.items()}

TYPE_ES: dict[str, str] = {
    "normal": "Normal", "fighting": "Lucha", "flying": "Volador", "poison": "Veneno", "ground": "Tierra",
    "rock": "Roca", "bug": "Bicho", "ghost": "Fantasma", "steel": "Acero", "fire": "Fuego", "water": "Agua",
    "grass": "Planta", "electric": "Eléctrico", "psychic": "Psíquico", "ice": "Hielo", "dragon": "Dragón",
    "dark": "Siniestro", "fairy": "Hada", UNKNOWN_TYPE: "???"}
STATUS_ES: dict[str, str] = {"par": "parálisis", "slp": "sueño", "psn": "veneno", "brn": "quemadura",
                             "frz": "congelación", "tox": "veneno grave", "": "ninguno"}

_ALIASES: dict[str, str] = {
    # Spanish (cartridges of the reference setup are Spanish)
    "lucha": "fighting", "volador": "flying", "pajaro": "flying", "veneno": "poison", "tierra": "ground",
    "roca": "rock", "bicho": "bug", "fantasma": "ghost", "acero": "steel", "fuego": "fire", "agua": "water",
    "planta": "grass", "electrico": "electric", "psiquico": "psychic", "hielo": "ice", "dragon": "dragon",
    "siniestro": "dark", "hada": "fairy", "normal": "normal",
    # other spellings
    "mystery": UNKNOWN_TYPE, "curse": UNKNOWN_TYPE, "typeless": UNKNOWN_TYPE, "none": UNKNOWN_TYPE,
    "bird": "flying",
}


def _fold(text: str) -> str:
    """Lowercase without accents (``"Eléctrico"`` -> ``"electrico"``)."""
    decomposed = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def canonical_type(name: str) -> str:
    """English lowercase type name; unknown names become ``"???"``."""
    folded = _fold(name)
    if folded in GEN6_TYPES or folded == UNKNOWN_TYPE:
        return folded
    return _ALIASES.get(folded, UNKNOWN_TYPE)


def types_of_generation(generation: int) -> tuple[str, ...]:
    if generation <= 1:
        return GEN1_TYPES
    if generation <= 5:
        return GEN2_TYPES
    return GEN6_TYPES


# Gen 2-5 chart: non-neutral cells only (attacking type -> {defending type: multiplier}).
_GEN2_CHART: dict[str, dict[str, float]] = {
    "normal": {"rock": 0.5, "ghost": 0.0, "steel": 0.5},
    "fighting": {"normal": 2.0, "flying": 0.5, "poison": 0.5, "rock": 2.0, "bug": 0.5, "ghost": 0.0,
                 "steel": 2.0, "psychic": 0.5, "ice": 2.0, "dark": 2.0},
    "flying": {"fighting": 2.0, "rock": 0.5, "bug": 2.0, "steel": 0.5, "grass": 2.0, "electric": 0.5},
    "poison": {"poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0.0, "grass": 2.0},
    "ground": {"flying": 0.0, "poison": 2.0, "rock": 2.0, "bug": 0.5, "steel": 2.0, "fire": 2.0, "grass": 0.5,
               "electric": 2.0},
    "rock": {"fighting": 0.5, "flying": 2.0, "ground": 0.5, "bug": 2.0, "steel": 0.5, "fire": 2.0, "ice": 2.0},
    "bug": {"fighting": 0.5, "flying": 0.5, "poison": 0.5, "ghost": 0.5, "steel": 0.5, "fire": 0.5, "grass": 2.0,
            "psychic": 2.0, "dark": 2.0},
    "ghost": {"normal": 0.0, "ghost": 2.0, "steel": 0.5, "psychic": 2.0, "dark": 0.5},
    "steel": {"rock": 2.0, "steel": 0.5, "fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2.0},
    "fire": {"rock": 0.5, "bug": 2.0, "steel": 2.0, "fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 2.0,
             "dragon": 0.5},
    "water": {"ground": 2.0, "rock": 2.0, "fire": 2.0, "water": 0.5, "grass": 0.5, "dragon": 0.5},
    "grass": {"flying": 0.5, "poison": 0.5, "ground": 2.0, "rock": 2.0, "bug": 0.5, "steel": 0.5, "fire": 0.5,
              "water": 2.0, "grass": 0.5, "dragon": 0.5},
    "electric": {"flying": 2.0, "ground": 0.0, "water": 2.0, "grass": 0.5, "electric": 0.5, "dragon": 0.5},
    "psychic": {"fighting": 2.0, "poison": 2.0, "steel": 0.5, "psychic": 0.5, "dark": 0.0},
    "ice": {"flying": 2.0, "ground": 2.0, "steel": 0.5, "fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 0.5,
            "dragon": 2.0},
    "dragon": {"steel": 0.5, "dragon": 2.0},
    "dark": {"fighting": 0.5, "ghost": 2.0, "steel": 0.5, "psychic": 2.0, "dark": 0.5},
}


@lru_cache(maxsize=8)
def type_chart(generation: int) -> dict[str, dict[str, float]]:
    """Full attack -> defence multiplier table of ``generation`` (every cell present)."""
    names = types_of_generation(generation)
    chart = {attack: {defense: 1.0 for defense in names} for attack in names}
    for attack, row in _GEN2_CHART.items():
        if attack not in chart:
            continue
        for defense, value in row.items():
            if defense in chart[attack]:
                chart[attack][defense] = value
    if generation <= 1:
        chart["bug"]["poison"] = 2.0     # Gen 1: Bug hits Poison super effectively
        chart["poison"]["bug"] = 2.0     # Gen 1: Poison hits Bug super effectively
        chart["ghost"]["psychic"] = 0.0  # RBY bug: Ghost does not affect Psychic
        chart["ice"]["fire"] = 1.0       # Gen 1: Fire does not resist Ice
    elif generation >= 6:
        chart["ghost"]["steel"] = 1.0
        chart["dark"]["steel"] = 1.0
        for defense, value in {"fighting": 2.0, "poison": 0.5, "steel": 0.5, "fire": 0.5, "dragon": 2.0,
                               "dark": 2.0}.items():
            chart["fairy"][defense] = value
        chart["fighting"]["fairy"] = 0.5
        chart["bug"]["fairy"] = 0.5
        chart["dark"]["fairy"] = 0.5
        chart["dragon"]["fairy"] = 0.0
        chart["poison"]["fairy"] = 2.0
        chart["steel"]["fairy"] = 2.0
    return chart


def type_multiplier(generation: int, attack: str, defenders: tuple[str, ...] | list[str]) -> float:
    """Product of the chart cells; typeless or unknown types are neutral."""
    chart = type_chart(generation)
    row = chart.get(canonical_type(attack))
    if row is None:
        return 1.0
    value = 1.0
    for defense in dict.fromkeys(canonical_type(name) for name in defenders):
        value *= row.get(defense, 1.0)
    return value


def category_for(move_type: str, power: int, generation: int, declared: str | None = None) -> str:
    """Damage category: by type in Gen 1-3, by move (``declared``) from Gen 4 on."""
    if generation >= 4 and declared in CATEGORIES:
        return str(declared)
    if power <= 0:
        return "status"
    return "special" if canonical_type(move_type) in SPECIAL_TYPES else "physical"


def normalize_name(name: str) -> str:
    """Key for name lookups: uppercase letters and digits only (``"Sand-Attack"`` -> ``"SANDATTACK"``)."""
    return "".join(char for char in _fold(name).upper() if char.isalnum())


def display_name(name: str) -> str:
    """``"TOTODILE"`` -> ``"Totodile"`` for explanations."""
    text = name.strip()
    return text.title() if text.isupper() or text.islower() else text


__all__ = ["BattleView", "BrainChoice", "CATEGORIES", "GEN1_TYPES", "GEN2_TYPES", "GEN3_INDEX_TYPE",
           "GEN3_TYPE_INDEX", "GEN6_TYPES", "MonView", "MoveView", "PHYSICAL_TYPES", "SPECIAL_TYPES", "STATUSES",
           "STATUS_ES", "TYPE_ES", "UNKNOWN_TYPE", "canonical_type", "category_for", "display_name",
           "normalize_name", "type_chart", "type_multiplier", "types_of_generation"]
