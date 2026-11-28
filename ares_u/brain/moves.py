"""What a move does beyond type and power: the brain's move knowledge.

``describe(move, generation)`` turns a ``MoveView`` into a ``MoveFacts`` record. A Gen-3 move id
(``gen3_move_id``; Gen 1-5 ids 1-354 coincide) resolves the effect from the Gen-3 ROM table of the
Stadium dex (``core.stadium.dex``); otherwise the English name is looked up in the same table, then a
small Spanish table of the status moves that matter for strategy; anything else is a plain attack
(power > 0) or an unknown status move (power 0) built from the view's own type/power/category.

Effect ids are the Gen-3 ``battle_move_effects.h`` constants (pokeemerald).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from core.stadium.dex import Dex, MoveData

from .api import GEN3_INDEX_TYPE, MoveView, canonical_type, category_for, normalize_name

# ----------------------------------------------------------------------------- Gen-3 effect ids
INFLICTS: dict[int, str] = {1: "slp", 66: "psn", 33: "tox", 67: "par", 167: "brn", 49: "cnf", 199: "cnf",
                            187: "yawn", 37: "rest"}
# Damaging moves with a secondary status chance (effect -> status), used only for explanations.
SECONDARY: dict[int, str] = {2: "psn", 4: "brn", 5: "frz", 6: "par", 76: "cnf", 125: "brn", 152: "par",
                             209: "tox"}
_STAT = {"atk": "atk", "def": "def", "spe": "spe", "spa": "spa", "spd": "spd", "acc": "acc", "eva": "eva"}
BOOSTS: dict[int, tuple[tuple[str, int], ...]] = {
    10: (("atk", 1),), 11: (("def", 1),), 12: (("spe", 1),), 13: (("spa", 1),), 14: (("spd", 1),),
    15: (("acc", 1),), 16: (("eva", 1),), 50: (("atk", 2),), 51: (("def", 2),), 52: (("spe", 2),),
    53: (("spa", 2),), 54: (("spd", 2),), 55: (("acc", 2),), 56: (("eva", 2),), 108: (("eva", 1),),
    156: (("def", 1),), 206: (("def", 1), ("spd", 1)), 208: (("atk", 1), ("def", 1)),
    211: (("spa", 1), ("spd", 1)), 212: (("atk", 1), ("spe", 1)), 142: (("atk", 6),),
    109: (("atk", 1), ("def", 1), ("spe", -1))}
DROPS: dict[int, tuple[tuple[str, int], ...]] = {
    18: (("atk", 1),), 19: (("def", 1),), 20: (("spe", 1),), 21: (("spa", 1),), 22: (("spd", 1),),
    23: (("acc", 1),), 24: (("eva", 1),), 58: (("atk", 2),), 59: (("def", 2),), 60: (("spe", 2),),
    61: (("spa", 2),), 62: (("spd", 2),), 63: (("acc", 2),), 64: (("eva", 2),),
    205: (("atk", 1), ("def", 1))}
HEALS: dict[int, float] = {32: 0.5, 157: 0.5, 132: 0.5, 133: 0.5, 134: 0.5, 37: 1.0}
FIXED: dict[int, str] = {41: "40", 130: "20", 87: "level", 88: "level", 40: "half", 189: "endeavor"}
HITS: dict[int, float] = {29: 3.0, 44: 2.0, 77: 2.0, 104: 2.0}
RECHARGE = frozenset({80})
CHARGE = frozenset({39, 75, 145, 151})
SELF_KO = frozenset({7})
OHKO = frozenset({38})
RECOIL: dict[int, float] = {48: 0.25, 198: 0.33, 45: 0.0}
DRAIN: dict[int, float] = {3: 0.5, 8: 0.5}
NEEDS_SLEEPING_TARGET = frozenset({8})  # Dream Eater
FIRST_TURN_ONLY = frozenset({158})       # Fake Out
NOMINAL_VARIABLE_POWER = 60

# Spanish names of the strategy-relevant status moves (Gen-3 ids), for adapters that pass no id.
SPANISH_STATUS_MOVES: dict[str, int] = {
    "ONDATRUENO": 86, "HIPNOSIS": 95, "SOMNIFERO": 79, "PARALIZADOR": 78, "CANTO": 47, "ESPORA": 147,
    "POLVOVENENO": 77, "TOXICO": 92, "FUEGOFATUO": 261, "BOSTEZO": 281, "DESLUMBRAR": 137,
    "BESOAMOROSO": 142, "RAYOCONFUSO": 109, "SUPERSONICO": 48, "DANZAESPADA": 14, "MALICIOSO": 43,
    "GRUNIDO": 45, "LATIGO": 39, "RECUPERACION": 105, "DESCANSO": 156, "MAQUINAZO": 339, "PAZMENTAL": 347,
    "RIZODEFENSA": 111, "FORTALEZA": 106, "AGILIDAD": 97, "MEDITACION": 96, "DANZADRAGON": 349}


@dataclass(frozen=True)
class MoveFacts:
    name: str
    type: str
    power: int          # base power (variable-power moves at a nominal value)
    accuracy: int       # 0 = never misses
    category: str       # physical | special | status
    priority: int
    effect: int         # Gen-3 effect id (0 when unknown)
    gen3_id: int | None
    inflicts: str = ""  # slp | par | psn | tox | brn | cnf | yawn | rest
    boosts: tuple[tuple[str, int], ...] = ()
    drops: tuple[tuple[str, int], ...] = ()
    heal: float = 0.0
    fixed: str = ""
    hits: float = 1.0
    recharge: bool = False
    charge: bool = False
    self_ko: bool = False
    ohko: bool = False
    recoil: float = 0.0
    drain: float = 0.0
    secondary: str = ""
    chance: int = 0

    @property
    def damaging(self) -> bool:
        return self.power > 0 or bool(self.fixed) or self.ohko

    @property
    def kind(self) -> str:
        if self.damaging:
            return "damage"
        if self.inflicts:
            return "status"
        if self.heal:
            return "heal"
        if self.boosts:
            return "boost"
        if self.drops:
            return "drop"
        return "other"


@lru_cache(maxsize=1)
def _dex() -> Dex:
    from ares_u.battle.teams import default_dex

    return default_dex()


@lru_cache(maxsize=1)
def _names() -> dict[str, int]:
    return {normalize_name(move.name): move.move_id for move in _dex().moves if move.move_id}


def gen3_id_for(view: MoveView) -> int | None:
    """The Gen-3 id of ``view``: its declared id, else its English/Spanish name, else ``None``."""
    dex = _dex()
    if view.gen3_move_id is not None and 0 < view.gen3_move_id < len(dex.moves):
        return int(view.gen3_move_id)
    key = normalize_name(view.name)
    found = _names().get(key)
    if found is None:
        found = SPANISH_STATUS_MOVES.get(key)
    return found


def _from_data(data: MoveData, view: MoveView, generation: int) -> MoveFacts:
    effect = int(data.effect)
    power = int(data.power)
    if power == 1 and effect not in FIXED:
        power = NOMINAL_VARIABLE_POWER
    # The view is authoritative on type (Hidden Power, Gen-1 retypes such as Bite/Gust/Karate Chop)
    # and on the Gen-4+ category; the table gives the effect.
    move_type = canonical_type(view.type) if view.type else GEN3_INDEX_TYPE.get(int(data.move_type), "normal")
    if move_type == "???" and view.type == "":
        move_type = GEN3_INDEX_TYPE.get(int(data.move_type), "normal")
    if view.power > 1 and generation != 3:
        power = int(view.power)  # other generations retuned some powers
    accuracy = int(view.accuracy) if generation != 3 and view.accuracy else int(data.accuracy)
    declared = view.category if view.category in ("physical", "special", "status") else None
    category = category_for(move_type, power if effect not in FIXED else max(1, power), generation, declared)
    if power <= 0 and effect not in FIXED and effect not in OHKO:
        category = "status"
    return MoveFacts(
        name=view.name or data.name, type=move_type, power=power if effect not in FIXED else 0,
        accuracy=accuracy, category=category,
        priority=int(view.priority) if view.priority is not None else int(data.priority),
        effect=effect, gen3_id=int(data.move_id), inflicts=INFLICTS.get(effect, ""),
        boosts=BOOSTS.get(effect, ()), drops=DROPS.get(effect, ()), heal=HEALS.get(effect, 0.0),
        fixed=FIXED.get(effect, ""), hits=HITS.get(effect, 1.0), recharge=effect in RECHARGE,
        charge=effect in CHARGE, self_ko=effect in SELF_KO, ohko=effect in OHKO,
        recoil=RECOIL.get(effect, 0.0), drain=DRAIN.get(effect, 0.0), secondary=SECONDARY.get(effect, ""),
        chance=int(data.chance))


def describe(view: MoveView, generation: int) -> MoveFacts:
    """Facts of one move slot (the view's type/power/category win over the Gen-3 table)."""
    move_id = gen3_id_for(view)
    if move_id is not None:
        return _from_data(_dex().moves[move_id], view, generation)
    move_type = canonical_type(view.type)
    power = int(view.power)
    category = category_for(move_type, power, generation, view.category)
    if power <= 0:
        category = "status"
    return MoveFacts(name=view.name, type=move_type, power=max(0, power), accuracy=int(view.accuracy),
                     category=category, priority=int(view.priority or 0), effect=0, gen3_id=None)


__all__ = ["MoveFacts", "NOMINAL_VARIABLE_POWER", "describe", "gen3_id_for"]
