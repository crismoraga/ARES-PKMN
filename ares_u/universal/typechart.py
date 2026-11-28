"""Type names, per-generation internal type ids and the textbook type charts (used as a weak prior).

The universal agent does not trust the textbook chart blindly: ``knowledge.TypeBeliefs`` combines
it with what was observed in battle (effectiveness messages and damage), so generation quirks
(Gen 1: Ghost does not affect Psychic, Bug/Poison are super effective on each other) are learned
from play and carried to the next game.
"""
from __future__ import annotations

TYPES: tuple[str, ...] = ("normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel",
                          "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark")

# Internal type ids of the Game Boy games (pokered/pokecrystal constants).
GB_TYPE_IDS: dict[int, str] = {0x00: "normal", 0x01: "fighting", 0x02: "flying", 0x03: "poison", 0x04: "ground",
                               0x05: "rock", 0x07: "bug", 0x08: "ghost", 0x09: "steel", 0x13: "curse",
                               0x14: "fire", 0x15: "water", 0x16: "grass", 0x17: "electric", 0x18: "psychic",
                               0x19: "ice", 0x1A: "dragon", 0x1B: "dark"}

_SUPER, _HALF, _IMMUNE = 2.0, 0.5, 0.0

# Gen 2+ chart (non-neutral entries only).
_GEN2: dict[tuple[str, str], float] = {}


def _set(table: dict[tuple[str, str], float], attacker: str, value: float, *defenders: str) -> None:
    for defender in defenders:
        table[(attacker, defender)] = value


_set(_GEN2, "normal", _HALF, "rock", "steel")
_set(_GEN2, "normal", _IMMUNE, "ghost")
_set(_GEN2, "fire", _SUPER, "grass", "ice", "bug", "steel")
_set(_GEN2, "fire", _HALF, "fire", "water", "rock", "dragon")
_set(_GEN2, "water", _SUPER, "fire", "ground", "rock")
_set(_GEN2, "water", _HALF, "water", "grass", "dragon")
_set(_GEN2, "electric", _SUPER, "water", "flying")
_set(_GEN2, "electric", _HALF, "electric", "grass", "dragon")
_set(_GEN2, "electric", _IMMUNE, "ground")
_set(_GEN2, "grass", _SUPER, "water", "ground", "rock")
_set(_GEN2, "grass", _HALF, "fire", "grass", "poison", "flying", "bug", "dragon", "steel")
_set(_GEN2, "ice", _SUPER, "grass", "ground", "flying", "dragon")
_set(_GEN2, "ice", _HALF, "fire", "water", "ice", "steel")
_set(_GEN2, "fighting", _SUPER, "normal", "ice", "rock", "dark", "steel")
_set(_GEN2, "fighting", _HALF, "poison", "flying", "psychic", "bug")
_set(_GEN2, "fighting", _IMMUNE, "ghost")
_set(_GEN2, "poison", _SUPER, "grass")
_set(_GEN2, "poison", _HALF, "poison", "ground", "rock", "ghost")
_set(_GEN2, "poison", _IMMUNE, "steel")
_set(_GEN2, "ground", _SUPER, "fire", "electric", "poison", "rock", "steel")
_set(_GEN2, "ground", _HALF, "grass", "bug")
_set(_GEN2, "ground", _IMMUNE, "flying")
_set(_GEN2, "flying", _SUPER, "grass", "fighting", "bug")
_set(_GEN2, "flying", _HALF, "electric", "rock", "steel")
_set(_GEN2, "psychic", _SUPER, "fighting", "poison")
_set(_GEN2, "psychic", _HALF, "psychic", "steel")
_set(_GEN2, "psychic", _IMMUNE, "dark")
_set(_GEN2, "bug", _SUPER, "grass", "psychic", "dark")
_set(_GEN2, "bug", _HALF, "fire", "fighting", "poison", "flying", "ghost", "steel")
_set(_GEN2, "rock", _SUPER, "fire", "ice", "flying", "bug")
_set(_GEN2, "rock", _HALF, "fighting", "ground", "steel")
_set(_GEN2, "ghost", _SUPER, "psychic", "ghost")
_set(_GEN2, "ghost", _HALF, "dark", "steel")
_set(_GEN2, "ghost", _IMMUNE, "normal")
_set(_GEN2, "dragon", _SUPER, "dragon")
_set(_GEN2, "dragon", _HALF, "steel")
_set(_GEN2, "dark", _SUPER, "psychic", "ghost")
_set(_GEN2, "dark", _HALF, "fighting", "dark", "steel")
_set(_GEN2, "steel", _SUPER, "ice", "rock")
_set(_GEN2, "steel", _HALF, "fire", "water", "electric", "steel")

# Gen 1 differences: no steel/dark, Ghost->Psychic immune (the famous bug), Bug<->Poison 2x, Ice->Fire 1x.
_GEN1: dict[tuple[str, str], float] = {key: value for key, value in _GEN2.items()
                                       if "steel" not in key and "dark" not in key}
_GEN1[("ghost", "psychic")] = _IMMUNE
_GEN1[("bug", "poison")] = _SUPER
_GEN1[("poison", "bug")] = _SUPER
_GEN1.pop(("ice", "fire"), None)

CHARTS: dict[int, dict[tuple[str, str], float]] = {1: _GEN1, 2: _GEN2}


def textbook(attacker: str, defender: str, generation: int = 2) -> float:
    """Textbook multiplier of one attacking type against one defending type."""
    chart = CHARTS[1] if generation <= 1 else CHARTS[2]
    return chart.get((attacker, defender), 1.0)


def textbook_multi(attacker: str, defenders: tuple[str, ...], generation: int = 2) -> float:
    result = 1.0
    for defender in dict.fromkeys(defenders):  # a mono-type mon stores its type twice
        result *= textbook(attacker, defender, generation)
    return result


def category_of(multiplier: float) -> str:
    """Bucket of an effectiveness value, as the games announce it."""
    if multiplier == 0.0:
        return "immune"
    if multiplier < 1.0:
        return "weak"
    if multiplier > 1.0:
        return "super"
    return "neutral"
