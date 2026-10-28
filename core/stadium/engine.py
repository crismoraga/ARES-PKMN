"""Deterministic Gen-3 singles battle engine for the ARES stadium.

Damage is never re-derived here: every hit is computed by the project's exact
integer reference (``GbaIntegerSafetyVerifier.compute_context_damage_reference``),
which is parity-tested against the ARM7 implementation. Around it the engine
implements Gen-3 turn structure: switches first, priority then speed (stages,
paralysis, Swift Swim/Chlorophyll), accuracy/evasion stage ratios, 1/16..1/2
critical stages, sleep/freeze/paralysis/confusion/flinch, secondary effects,
stat stages, screens, weather, Spikes, Substitute, Protect, two-turn and
recharge moves, residual damage and the abilities that change battles.

Fidelity limits are explicit: moves whose effect is not modelled still deal
their damage (damaging moves) or fail (status moves), and each such use is
recorded as an ``unsupported`` event so no result silently pretends otherwise.
Bag items are not allowed, exactly as in Gen-3 link battles.

Control loop::

    battle = Battle(dex, team_a, team_b, seed=7)
    while not battle.over:
        battle.submit({side: agents[side](battle, side) for side in battle.pending})
"""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.safety.z3_integer_verifier import DamageContext, GbaIntegerSafetyVerifier

from .dex import TYPE_MYSTERY, Dex, MoveData

# Gen-3 STATUS1 bit layout (the damage reference reads burn from these bits).
SLEEP_MASK = 0x7
POISON = 0x8
BURN = 0x10
FREEZE = 0x20
PARALYSIS = 0x40
TOXIC = 0x80
STATUS_NAMES = {POISON: "psn", BURN: "brn", FREEZE: "frz", PARALYSIS: "par", TOXIC: "tox"}

# statStages indexes: 0 hp (unused), 1 atk, 2 def, 3 spe, 4 spa, 5 spd, 6 acc, 7 eva.
ATK, DEF, SPE, SPA, SPD, ACC, EVA = 1, 2, 3, 4, 5, 6, 7
STAT_NAMES = ("hp", "atk", "def", "spe", "spa", "spd", "acc", "eva")
STAGE_RATIOS = ((10, 40), (10, 35), (10, 30), (10, 25), (10, 20), (10, 15), (10, 10),
                (15, 10), (20, 10), (25, 10), (30, 10), (35, 10), (40, 10))
ACCURACY_RATIOS = ((33, 100), (36, 100), (43, 100), (50, 100), (60, 100), (75, 100), (1, 1),
                   (133, 100), (166, 100), (2, 1), (233, 100), (133, 50), (3, 1))
CRITICAL_CHANCE = (16, 8, 4, 3, 2)
TARGET_USER = 0x10
MOVE_STRUGGLE = 165

# Abilities (Gen-3 ids, verified against the decomp header by the tests).
A = {name: index for index, name in enumerate((
    "NONE", "STENCH", "DRIZZLE", "SPEED_BOOST", "BATTLE_ARMOR", "STURDY", "DAMP", "LIMBER", "SAND_VEIL",
    "STATIC", "VOLT_ABSORB", "WATER_ABSORB", "OBLIVIOUS", "CLOUD_NINE", "COMPOUND_EYES", "INSOMNIA",
    "COLOR_CHANGE", "IMMUNITY", "FLASH_FIRE", "SHIELD_DUST", "OWN_TEMPO", "SUCTION_CUPS", "INTIMIDATE",
    "SHADOW_TAG", "ROUGH_SKIN", "WONDER_GUARD", "LEVITATE", "EFFECT_SPORE", "SYNCHRONIZE", "CLEAR_BODY",
    "NATURAL_CURE", "LIGHTNING_ROD", "SERENE_GRACE", "SWIFT_SWIM", "CHLOROPHYLL", "ILLUMINATE", "TRACE",
    "HUGE_POWER", "POISON_POINT", "INNER_FOCUS", "MAGMA_ARMOR", "WATER_VEIL", "MAGNET_PULL", "SOUNDPROOF",
    "RAIN_DISH", "SAND_STREAM", "PRESSURE", "THICK_FAT", "EARLY_BIRD", "FLAME_BODY", "RUN_AWAY",
    "KEEN_EYE", "HYPER_CUTTER", "PICKUP", "TRUANT", "HUSTLE", "CUTE_CHARM", "PLUS", "MINUS", "FORECAST",
    "STICKY_HOLD", "SHED_SKIN", "GUTS", "MARVEL_SCALE", "LIQUID_OOZE", "OVERGROW", "BLAZE", "TORRENT",
    "SWARM", "ROCK_HEAD", "DROUGHT", "ARENA_TRAP", "VITAL_SPIRIT", "WHITE_SMOKE", "PURE_POWER",
    "SHELL_ARMOR", "CACOPHONY", "AIR_LOCK"))}

# Held items with battle effects (Gen-3 item ids).
ITEM_ORAN, ITEM_LUM, ITEM_SITRUS = 139, 141, 142
ITEM_CHERI, ITEM_CHESTO, ITEM_PECHA, ITEM_RAWST, ITEM_ASPEAR, ITEM_PERSIM = 133, 134, 135, 136, 137, 140
ITEM_BRIGHT_POWDER, ITEM_QUICK_CLAW, ITEM_CHOICE_BAND, ITEM_KINGS_ROCK = 179, 183, 186, 187
ITEM_FOCUS_BAND, ITEM_SCOPE_LENS, ITEM_LEFTOVERS, ITEM_SHELL_BELL = 196, 198, 200, 219
STATUS_BERRIES = {ITEM_CHERI: PARALYSIS, ITEM_CHESTO: SLEEP_MASK, ITEM_PECHA: POISON | TOXIC,
                  ITEM_RAWST: BURN, ITEM_ASPEAR: FREEZE}

SOUND_MOVES = frozenset({45, 46, 47, 48, 103, 173, 195, 215, 253, 304, 319, 320})
# Effects whose damage or status logic is modelled; the rest are flagged.
_STAT_UP = {10: (ATK, 1), 11: (DEF, 1), 12: (SPE, 1), 13: (SPA, 1), 14: (SPD, 1), 15: (ACC, 1), 16: (EVA, 1),
            50: (ATK, 2), 51: (DEF, 2), 52: (SPE, 2), 53: (SPA, 2), 54: (SPD, 2), 55: (ACC, 2), 56: (EVA, 2),
            156: (DEF, 1), 108: (EVA, 1)}
_STAT_DOWN = {18: (ATK, 1), 19: (DEF, 1), 20: (SPE, 1), 21: (SPA, 1), 22: (SPD, 1), 23: (ACC, 1), 24: (EVA, 1),
              58: (ATK, 2), 59: (DEF, 2), 60: (SPE, 2), 61: (SPA, 2), 62: (SPD, 2), 63: (ACC, 2), 64: (EVA, 2)}
_STAT_DOWN_HIT = {68: ATK, 69: DEF, 70: SPE, 71: SPA, 72: SPD, 73: ACC, 74: EVA}
_MULTI_BOOST = {206: ((DEF, 1), (SPD, 1)), 208: ((ATK, 1), (DEF, 1)), 211: ((SPA, 1), (SPD, 1)),
                212: ((ATK, 1), (SPE, 1))}
_TWO_TURN = frozenset({39, 75, 145, 151, 155})
_HEAL_HALF = frozenset({32, 157})
_WEATHER_HEAL = frozenset({132, 133, 134})
_FIXED = frozenset({41, 130, 87, 88, 40, 189})
_MODELLED = (set(_STAT_UP) | set(_STAT_DOWN) | set(_STAT_DOWN_HIT) | set(_MULTI_BOOST) | set(_TWO_TURN) |
             set(_HEAL_HALF) | set(_WEATHER_HEAL) | set(_FIXED) | {
                 0, 1, 2, 3, 4, 5, 6, 7, 8, 17, 25, 27, 28, 29, 31, 33, 35, 36, 37, 38, 42, 43, 44, 45, 46, 47, 48,
                 49, 65, 66, 67, 76, 77, 79, 80, 84, 85, 86, 89, 90, 91, 92, 97, 99, 101, 102, 103, 104, 106,
                 109, 111, 112, 115, 116, 117, 118, 119, 121, 123, 124, 125, 126, 128, 129, 135, 136, 137, 138,
                 139, 140, 142, 143, 144, 146, 147, 149, 150, 152, 158, 164, 166, 167, 168, 169, 170, 171, 175,
                 179, 181, 182, 185, 186, 187, 188, 190, 193, 196, 197, 198, 199, 200, 202, 203, 204, 205, 207,
                 209})

Event = tuple[Any, ...]


@dataclass(frozen=True)
class MonSpec:
    """One team member as it exists in a real save (stats at its level)."""
    species: int
    level: int
    stats: tuple[int, int, int, int, int, int]  # max HP, atk, def, spe, spa, spd
    moves: tuple[int, ...]
    ability: int = 0
    item: int = 0
    friendship: int = 70
    ivs: tuple[int, int, int, int, int, int] = (31, 31, 31, 31, 31, 31)
    name: str = ""
    origin: str = ""

    def __post_init__(self) -> None:
        if not 1 <= self.level <= 100 or not 1 <= len(self.moves) <= 4 or min(self.stats) < 1:
            raise ValueError(f"invalid team member {self.species}@{self.level}")

    def to_json(self) -> dict[str, Any]:
        return {"species": self.species, "level": self.level, "stats": list(self.stats), "moves": list(self.moves),
                "ability": self.ability, "item": self.item, "friendship": self.friendship, "ivs": list(self.ivs),
                "name": self.name, "origin": self.origin}

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> MonSpec:
        stats = tuple(int(value) for value in raw["stats"])
        ivs = tuple(int(value) for value in raw.get("ivs", [31] * 6))
        return cls(int(raw["species"]), int(raw["level"]), (stats[0], stats[1], stats[2], stats[3], stats[4], stats[5]),
                   tuple(int(move) for move in raw["moves"] if int(move)), int(raw.get("ability", 0)),
                   int(raw.get("item", 0)), int(raw.get("friendship", 70)),
                   (ivs[0], ivs[1], ivs[2], ivs[3], ivs[4], ivs[5]), str(raw.get("name", "")), str(raw.get("origin", "")))


class Fighter:
    """Mutable battle state of one team member."""
    __slots__ = ("spec", "types", "hp", "max_hp", "status", "stages", "pp", "move_ids", "toxic_n", "confusion",
                 "flinched", "seeded", "substitute", "protect_streak", "protected", "enduring", "recharge",
                 "charging", "invulnerable", "lock_slot", "lock_turns", "rampage", "trap_turns", "trapped_by",
                 "focus_energy", "yawn", "ingrain", "taken_physical", "taken_special", "hit_by_target",
                 "turns_active", "last_slot", "item", "flash_fire", "taunt", "encore", "disabled_slot",
                 "disabled_turns", "cursed", "truant_idle", "ability", "choice_slot", "fake_out_ready",
                 "focus_punch")

    def __init__(self, spec: MonSpec, types: tuple[int, int], dex: Dex) -> None:
        self.spec = spec
        self.types = types
        self.max_hp = spec.stats[0]
        self.hp = spec.stats[0]
        self.status = 0
        self.move_ids = list(spec.moves)
        self.pp = [dex.moves[move].pp for move in spec.moves]
        self.item = spec.item
        self.ability = spec.ability
        self.reset_volatile()

    def reset_volatile(self) -> None:
        self.stages = [6] * 8
        self.toxic_n = 0
        self.confusion = 0
        self.flinched = False
        self.seeded = False
        self.substitute = 0
        self.protect_streak = 0
        self.protected = False
        self.enduring = False
        self.recharge = False
        self.charging = -1
        self.invulnerable = False
        self.lock_slot = -1
        self.lock_turns = 0
        self.rampage = False
        self.trap_turns = 0
        self.trapped_by = False
        self.focus_energy = False
        self.yawn = 0
        self.ingrain = False
        self.taken_physical = 0
        self.taken_special = 0
        self.hit_by_target = False
        self.turns_active = 0
        self.last_slot = -1
        self.flash_fire = False
        self.taunt = 0
        self.encore = 0
        self.disabled_slot = -1
        self.disabled_turns = 0
        self.cursed = False
        self.truant_idle = False
        self.choice_slot = -1
        self.fake_out_ready = True
        self.focus_punch = False

    @property
    def alive(self) -> bool:
        return self.hp > 0

    def stat(self, index: int) -> int:
        return self.spec.stats[index]

    def clone(self) -> Fighter:
        copy = Fighter.__new__(Fighter)
        for name in Fighter.__slots__:
            value = getattr(self, name)
            setattr(copy, name, list(value) if isinstance(value, list) else value)
        return copy


class Side:
    __slots__ = ("party", "active", "reflect", "light_screen", "spikes", "safeguard", "mist", "wish_turns",
                 "wish_hp", "unsupported")

    def __init__(self, party: list[Fighter]) -> None:
        self.party = party
        self.active = 0
        self.reflect = self.light_screen = self.spikes = self.safeguard = self.mist = 0
        self.wish_turns = self.wish_hp = 0
        self.unsupported = 0

    @property
    def fighter(self) -> Fighter:
        return self.party[self.active]

    def bench(self) -> list[int]:
        return [index for index in range(len(self.party)) if index != self.active]

    def remaining(self) -> int:
        return sum(1 for fighter in self.party if fighter.alive)

    def hp_fraction(self) -> float:
        return sum(fighter.hp for fighter in self.party) / max(1, sum(fighter.max_hp for fighter in self.party))

    def clone(self) -> Side:
        copy = Side.__new__(Side)
        copy.party = [fighter.clone() for fighter in self.party]
        for name in Side.__slots__[1:]:
            setattr(copy, name, getattr(self, name))
        return copy


@dataclass
class BattleSummary:
    winner: int  # 0, 1 or 2 for a draw
    turns: int
    hp_fraction: tuple[float, float]
    remaining: tuple[int, int]
    unsupported: tuple[int, int]
    events: list[Event] = field(default_factory=list)


class Battle:
    """One singles battle; ``pending`` names the sides that must act next."""

    def __init__(self, dex: Dex, team_a: Sequence[MonSpec], team_b: Sequence[MonSpec], seed: int = 0,
                 max_turns: int = 200, record: bool = True) -> None:
        if not team_a or not team_b or len(team_a) > 6 or len(team_b) > 6:
            raise ValueError("each side needs one to six team members")
        self.dex = dex
        self.rng = random.Random(seed)
        self.sides = [Side([Fighter(spec, dex.species[spec.species].types, dex) for spec in team])
                      for team in (team_a, team_b)]
        self.weather = "none"
        self.weather_turns = 0
        self.turn = 0
        self.max_turns = max_turns
        self.winner: int | None = None
        self.record = record
        self.events: list[Event] = []
        self.log: list[Event] = []  # events of the last submit only
        self.pending: tuple[int, ...] = (0, 1)
        self._replacing = False
        for side in (0, 1):
            self._emit(("switch", side, 0, self.sides[side].fighter.spec.species))
        for side in self._speed_order():
            self._on_entry(side)

    # ------------------------------------------------------------------ API
    @property
    def over(self) -> bool:
        return self.winner is not None

    def clone(self, seed: int | None = None) -> Battle:
        """An independent copy (for counterfactual rollouts); history is not copied."""
        copy = Battle.__new__(Battle)
        copy.dex = self.dex
        copy.rng = random.Random(seed)
        if seed is None:
            copy.rng.setstate(self.rng.getstate())  # never perturb the original stream
        copy.sides = [side.clone() for side in self.sides]
        copy.weather, copy.weather_turns, copy.turn = self.weather, self.weather_turns, self.turn
        copy.max_turns, copy.winner, copy.record = self.max_turns, self.winner, False
        copy.events, copy.log = [], []
        copy.pending, copy._replacing = self.pending, self._replacing
        return copy

    def legal_actions(self, side: int) -> list[int]:
        """0-3 move slots (0 is Struggle when no PP is left), 4+k switches to bench k."""
        if side not in self.pending or self.over:
            return []
        own = self.sides[side]
        fighter = own.fighter
        switches = [4 + position for position, index in enumerate(own.bench()[:5]) if own.party[index].alive]
        if self._replacing:
            return switches
        if fighter.recharge or fighter.charging >= 0 or fighter.rampage:
            return [max(0, fighter.lock_slot if fighter.lock_slot >= 0 else fighter.charging)]
        if fighter.encore and fighter.last_slot >= 0 and fighter.pp[fighter.last_slot] > 0:
            moves = [fighter.last_slot]
        else:
            moves = [slot for slot in range(len(fighter.move_ids)) if self._usable(fighter, slot)]
        if not moves:
            moves = [0]  # Struggle
        return moves + ([] if self._trapped(side) else switches)

    def submit(self, actions: Mapping[int, int]) -> list[Event]:
        """Resolve the pending decisions; returns this step's events."""
        if self.over:
            raise RuntimeError("battle already finished")
        for side in self.pending:
            if side not in actions or actions[side] not in self.legal_actions(side):
                raise ValueError(f"side {side} chose an illegal action {actions.get(side)}")
        self.log = []
        if self._replacing:
            for side in sorted(self.pending, key=lambda value: -self._speed(value)):
                self._switch_in(side, self._bench_index(side, actions[side]))
            self._replacing = False
            self._after_faints()
            return self.log
        self._resolve_turn(dict(actions))
        return self.log

    def summary(self) -> BattleSummary:
        return BattleSummary(2 if self.winner is None else self.winner, self.turn,
                             (self.sides[0].hp_fraction(), self.sides[1].hp_fraction()),
                             (self.sides[0].remaining(), self.sides[1].remaining()),
                             (self.sides[0].unsupported, self.sides[1].unsupported), self.events)

    # -------------------------------------------------------------- helpers
    def _emit(self, event: Event) -> None:
        self.log.append(event)
        if self.record:
            self.events.append(event)

    def _usable(self, fighter: Fighter, slot: int) -> bool:
        if slot >= len(fighter.move_ids) or fighter.pp[slot] <= 0:
            return False
        if fighter.disabled_turns and fighter.disabled_slot == slot:
            return False
        if fighter.taunt and not self.dex.moves[fighter.move_ids[slot]].damaging:
            return False
        if fighter.choice_slot >= 0 and fighter.item == ITEM_CHOICE_BAND and slot != fighter.choice_slot:
            return False
        return True

    def _trapped(self, side: int) -> bool:
        fighter = self.sides[side].fighter
        if fighter.trap_turns or fighter.trapped_by or fighter.ingrain:
            return True
        foe = self.sides[1 - side].fighter
        if not foe.alive:
            return False
        if foe.ability == A["SHADOW_TAG"]:
            return True
        if foe.ability == A["ARENA_TRAP"] and 2 not in fighter.types and fighter.ability != A["LEVITATE"]:
            return True
        return foe.ability == A["MAGNET_PULL"] and 8 in fighter.types

    def _bench_index(self, side: int, action: int) -> int:
        return self.sides[side].bench()[action - 4]

    def _weather_active(self) -> str:
        if any(self.sides[side].fighter.alive and self.sides[side].fighter.ability in (A["CLOUD_NINE"], A["AIR_LOCK"])
               for side in (0, 1)):
            return "none"
        return self.weather

    def _speed(self, side: int) -> int:
        fighter = self.sides[side].fighter
        numerator, denominator = STAGE_RATIOS[fighter.stages[SPE]]
        speed = fighter.stat(3) * numerator // denominator
        weather = self._weather_active()
        if (weather == "rain" and fighter.ability == A["SWIFT_SWIM"]) or (
                weather == "sun" and fighter.ability == A["CHLOROPHYLL"]):
            speed *= 2
        if fighter.status & PARALYSIS:
            speed //= 4
        return speed

    def _speed_order(self) -> list[int]:
        first, second = self._speed(0), self._speed(1)
        if first != second:
            return [0, 1] if first > second else [1, 0]
        return [0, 1] if self.rng.random() < 0.5 else [1, 0]

    # --------------------------------------------------------- turn engine
    def _resolve_turn(self, actions: dict[int, int]) -> None:
        self.turn += 1
        for side in (0, 1):
            fighter = self.sides[side].fighter
            fighter.protected = fighter.enduring = fighter.flinched = fighter.hit_by_target = False
            fighter.taken_physical = fighter.taken_special = 0
        switching = [side for side in (0, 1) if actions[side] >= 4]
        for side in sorted(switching, key=lambda value: -self._speed(value)):
            self._emit(("action", side, "switch", actions[side]))
            self._switch_in(side, self._bench_index(side, actions[side]))
            if self.over:
                return
        movers = [side for side in (0, 1) if actions[side] < 4]
        order = self._move_order(movers, actions)
        for position, side in enumerate(order):
            fighter = self.sides[side].fighter
            if not fighter.alive or self.over:
                continue
            slot = actions[side]
            fighter.focus_punch = False
            self._use_move(side, slot, moved_last=position == len(order) - 1 and len(order) == 2)
            if self._check_end():
                return
        if not self.over:
            self._end_of_turn()
        for side in (0, 1):
            fighter = self.sides[side].fighter
            if fighter.alive:
                fighter.turns_active += 1
                fighter.fake_out_ready = False
        self._after_faints()

    def _move_order(self, movers: list[int], actions: dict[int, int]) -> list[int]:
        if len(movers) < 2:
            return movers
        keys: dict[int, tuple[int, int, int, float]] = {}
        for side in movers:
            fighter = self.sides[side].fighter
            slot = actions[side]
            move = self._move_for(fighter, slot)
            claw = int(fighter.item == ITEM_QUICK_CLAW and self.rng.randrange(100) < 20)
            keys[side] = (move.priority, claw, self._speed(side), self.rng.random())
            if move.effect == 170:  # Focus Punch: tightens its focus before anyone moves
                fighter.focus_punch = True
        return sorted(movers, key=lambda side: keys[side], reverse=True)

    def _move_for(self, fighter: Fighter, slot: int) -> MoveData:
        if not any(pp > 0 for pp in fighter.pp) and fighter.charging < 0 and not fighter.rampage:
            return self.dex.moves[MOVE_STRUGGLE]
        return self.dex.moves[fighter.move_ids[slot]]

    def _check_end(self) -> bool:
        alive = [self.sides[side].remaining() for side in (0, 1)]
        if alive[0] and alive[1]:
            return False
        self.winner = 2 if not alive[0] and not alive[1] else (0 if alive[0] else 1)
        self.pending = ()
        self._emit(("end", self.winner, self.turn))
        return True

    def _after_faints(self) -> None:
        if self._check_end():
            return
        needs = tuple(side for side in (0, 1) if not self.sides[side].fighter.alive)
        if needs:
            self._replacing = True
            self.pending = needs
            return
        if self.turn >= self.max_turns:
            # Turn limit: adjudicate by remaining HP fraction, like a Battle Frontier judge.
            first, second = self.sides[0].hp_fraction(), self.sides[1].hp_fraction()
            self.winner = 2 if abs(first - second) < 1e-9 else (0 if first > second else 1)
            self.pending = ()
            self._emit(("end", self.winner, self.turn, "turn_limit"))
            return
        self.pending = (0, 1)

    # ---------------------------------------------------------- switching
    def _switch_in(self, side: int, index: int) -> None:
        own = self.sides[side]
        leaving = own.fighter
        if leaving.alive and leaving.ability == A["NATURAL_CURE"] and leaving.status:
            leaving.status = 0
        foe = self.sides[1 - side].fighter
        if foe.trapped_by:
            foe.trapped_by = False  # Mean Look ends when its user leaves
        leaving.reset_volatile()
        own.active = index
        entering = own.fighter
        entering.reset_volatile()
        if entering.status & TOXIC:
            entering.toxic_n = 1
        self._emit(("switch", side, index, entering.spec.species))
        if own.spikes and 2 not in entering.types and entering.ability != A["LEVITATE"]:
            self._hurt(side, max(1, entering.max_hp // (8, 6, 4)[own.spikes - 1]), "spikes")
        if entering.alive:
            self._on_entry(side)

    def _on_entry(self, side: int) -> None:
        fighter = self.sides[side].fighter
        foe_side = 1 - side
        foe = self.sides[foe_side].fighter
        ability = fighter.ability
        if ability == A["INTIMIDATE"] and foe.alive:
            self._emit(("ability", side, ability))
            self._boost(foe_side, ATK, -1, by_foe=True)
        elif ability == A["DRIZZLE"]:
            self._set_weather("rain", -1, side)
        elif ability == A["DROUGHT"]:
            self._set_weather("sun", -1, side)
        elif ability == A["SAND_STREAM"]:
            self._set_weather("sandstorm", -1, side)
        elif ability == A["TRACE"] and foe.alive and foe.ability not in (A["TRACE"], 0):
            fighter.ability = foe.ability
            self._emit(("ability", side, A["TRACE"]))
            self._emit(("ability", foe_side, foe.ability))

    def _set_weather(self, kind: str, turns: int, side: int | None = None) -> bool:
        if self.weather == kind:
            return False
        self.weather, self.weather_turns = kind, turns
        if side is not None and turns < 0:
            self._emit(("ability", side, self.sides[side].fighter.ability))
        self._emit(("weather", kind))
        return True

    # ------------------------------------------------------------ moving
    def _use_move(self, side: int, slot: int, moved_last: bool) -> None:
        own, foe_side = self.sides[side], 1 - side
        fighter = own.fighter
        if fighter.recharge:
            fighter.recharge = False
            self._emit(("recharge", side))
            return
        if fighter.ability == A["TRUANT"]:
            if fighter.truant_idle:
                fighter.truant_idle = False
                self._emit(("ability", side, A["TRUANT"]))
                return
            fighter.truant_idle = True
        move = self._move_for(fighter, slot)
        struggling = move.move_id == MOVE_STRUGGLE
        if fighter.status & SLEEP_MASK:
            fighter.status -= 2 if fighter.ability == A["EARLY_BIRD"] and fighter.status & SLEEP_MASK >= 2 else 1
            if fighter.status & SLEEP_MASK:
                if move.effect not in (92, 97):  # Snore / Sleep Talk work while asleep
                    self._emit(("asleep", side))
                    self._clear_charge(fighter)
                    return
            else:
                self._emit(("woke", side))
                if move.effect == 92:
                    self._emit(("fail", side, move.move_id))
                    return
        elif move.effect in (92, 97) and not struggling:
            self._consume_pp(side, slot, move)
            self._emit(("fail", side, move.move_id))
            return
        if fighter.status & FREEZE:
            if self.rng.randrange(5) == 0 or move.effect in (125, 200) or move.move_id in (172, 221):
                fighter.status &= ~FREEZE
                self._emit(("thawed", side))
            else:
                self._emit(("frozen", side))
                return
        if fighter.confusion:
            fighter.confusion -= 1
            if not fighter.confusion:
                self._emit(("unconfused", side))
            elif self.rng.random() < 0.5:
                self._confusion_hit(side)
                self._clear_charge(fighter)
                return
        if fighter.status & PARALYSIS and self.rng.randrange(4) == 0:
            self._emit(("paralyzed", side))
            self._clear_charge(fighter)
            return
        if fighter.flinched:
            self._emit(("flinched", side))
            return
        if fighter.focus_punch and fighter.hit_by_target and move.effect == 170:
            self._consume_pp(side, slot, move)
            self._emit(("fail", side, move.move_id))
            return
        if move.effect == 97 and fighter.status & SLEEP_MASK:  # Sleep Talk
            choices = [index for index, move_id in enumerate(fighter.move_ids)
                       if self.dex.moves[move_id].effect not in (97, 39, 75, 145, 151, 155) and move_id]
            self._consume_pp(side, slot, move)
            if not choices:
                self._emit(("fail", side, move.move_id))
                return
            slot = self.rng.choice(choices)
            move = self.dex.moves[fighter.move_ids[slot]]
        elif fighter.charging < 0 and not fighter.rampage and not struggling:
            self._consume_pp(side, slot, move)
        fighter.last_slot = slot
        if fighter.item == ITEM_CHOICE_BAND and fighter.choice_slot < 0 and not struggling:
            fighter.choice_slot = slot
        self._emit(("move", side, slot, move.move_id))
        if fighter.taunt and not move.damaging and not struggling:
            self._emit(("fail", side, move.move_id))
            return
        self._execute(side, slot, move, moved_last, struggling)

    def _consume_pp(self, side: int, slot: int, move: MoveData) -> None:
        fighter = self.sides[side].fighter
        if slot < len(fighter.pp) and fighter.pp[slot] > 0:
            foe = self.sides[1 - side].fighter
            cost = 2 if foe.alive and foe.ability == A["PRESSURE"] and move.target != TARGET_USER else 1
            fighter.pp[slot] = max(0, fighter.pp[slot] - cost)

    def _clear_charge(self, fighter: Fighter) -> None:
        fighter.charging = -1
        fighter.invulnerable = False
        fighter.rampage = False
        fighter.lock_slot = -1

    def _execute(self, side: int, slot: int, move: MoveData, moved_last: bool, struggling: bool) -> None:
        foe_side = 1 - side
        fighter, foe = self.sides[side].fighter, self.sides[foe_side].fighter
        effect = move.effect
        if effect not in _MODELLED and not struggling:
            self.sides[side].unsupported += 1
            self._emit(("unsupported", side, move.move_id))
            if not move.damaging:
                self._emit(("fail", side, move.move_id))
                return
            effect = 0
        # Two-turn moves: charge now, strike next turn (Solar Beam is instant in sun).
        if effect in _TWO_TURN and fighter.charging < 0:
            if not (effect == 151 and self._weather_active() == "sun"):
                fighter.charging = slot
                fighter.lock_slot = slot
                fighter.invulnerable = effect == 155
                if effect == 145:
                    self._boost(side, DEF, 1)
                self._emit(("charging", side, move.move_id))
                return
        if fighter.charging >= 0:
            fighter.charging = -1
            fighter.invulnerable = False
            fighter.lock_slot = -1
        if move.target == TARGET_USER or (not move.damaging and effect in (
                10, 11, 12, 13, 14, 15, 16, 50, 51, 52, 53, 54, 55, 56, 156, 108, 206, 208, 211, 212, 32, 157, 132,
                133, 134, 37, 79, 111, 116, 35, 65, 112, 115, 136, 137, 164, 124, 46, 47, 142, 179, 181, 193, 102)):
            self._self_effect(side, move, effect, moved_last)
            return
        if not foe.alive:
            self._emit(("fail", side, move.move_id))
            return
        if foe.protected and move.flags & 2:
            self._emit(("protected", foe_side))
            if effect in (45,):  # Jump Kick crash
                self._hurt(side, max(1, fighter.max_hp // 2) // 2, "crash")
            return
        if foe.invulnerable and not (move.move_id in (16, 87, 239) or move.move_id in (89, 222)):
            self._emit(("miss", side, move.move_id))
            return
        if move.move_id in SOUND_MOVES and foe.ability == A["SOUNDPROOF"]:
            self._emit(("ability", foe_side, A["SOUNDPROOF"]))
            self._emit(("fail", side, move.move_id))
            return
        if not self._hits(side, move, effect):
            self._emit(("miss", side, move.move_id))
            if effect == 45:
                self._hurt(side, max(1, fighter.max_hp // 2) // 2, "crash")
            if effect == 7:
                self._self_faint(side)
            return
        if move.damaging or effect in _FIXED or effect in (89, 144, 38, 99, 190, 196):
            self._attack(side, slot, move, effect, moved_last, struggling)
        else:
            self._status_move(side, move, effect)

    def _hits(self, side: int, move: MoveData, effect: int) -> bool:
        if move.accuracy == 0 or effect == 17:
            return True
        fighter, foe = self.sides[side].fighter, self.sides[1 - side].fighter
        if effect == 152 and self._weather_active() == "rain":  # Thunder never misses in rain
            return True
        if effect == 38:  # OHKO: level difference + 30
            if fighter.spec.level < foe.spec.level:
                return False
            return self.rng.randrange(100) < fighter.spec.level - foe.spec.level + 30
        stage = max(0, min(12, fighter.stages[ACC] + 6 - foe.stages[EVA]))
        numerator, denominator = ACCURACY_RATIOS[stage]
        accuracy = move.accuracy
        if effect == 152 and self._weather_active() == "sun":
            accuracy = 50
        chance = accuracy * numerator // denominator
        if fighter.ability == A["COMPOUND_EYES"]:
            chance = chance * 130 // 100
        if fighter.ability == A["HUSTLE"] and not move.special:
            chance = chance * 80 // 100
        if foe.ability == A["SAND_VEIL"] and self._weather_active() == "sandstorm":
            chance = chance * 80 // 100
        if foe.item == ITEM_BRIGHT_POWDER:
            chance = chance * 90 // 100
        return self.rng.randrange(100) + 1 <= chance

    # ----------------------------------------------------------- damage
    def _move_type_power(self, side: int, move: MoveData, effect: int) -> tuple[int, int]:
        fighter, foe = self.sides[side].fighter, self.sides[1 - side].fighter
        move_type, power = move.move_type, move.power
        if move.move_id == MOVE_STRUGGLE:
            return 0, 50  # typeless: no modifiers and no STAB (see _damage)
        if effect == 135:  # Hidden Power from IVs (Gen-3 bit formulas)
            ivs = fighter.spec.ivs
            order = (ivs[0], ivs[1], ivs[2], ivs[3], ivs[4], ivs[5])
            type_bits = sum(((value & 1) << index) for index, value in enumerate(order))
            power_bits = sum((((value >> 1) & 1) << index) for index, value in enumerate(order))
            move_type = type_bits * 15 // 63 + 1
            if move_type >= TYPE_MYSTERY:
                move_type += 1
            power = power_bits * 40 // 63 + 30
        elif effect == 203:  # Weather Ball
            weather = self._weather_active()
            if weather != "none":
                power = 100
                move_type = {"rain": 11, "sun": 10, "sandstorm": 5, "hail": 15}[weather]
        elif effect == 121:
            power = max(1, fighter.spec.friendship * 10 // 25)
        elif effect == 123:
            power = max(1, (255 - fighter.spec.friendship) * 10 // 25)
        elif effect == 99:  # Flail / Reversal
            ratio = 48 * fighter.hp // fighter.max_hp
            power = 200 if ratio <= 1 else 150 if ratio <= 4 else 100 if ratio <= 9 else 80 if ratio <= 16 else \
                40 if ratio <= 32 else 20
        elif effect == 190:
            power = max(1, 150 * fighter.hp // fighter.max_hp)
        elif effect == 196:
            power = 60  # Low Kick: weight table not in the stadium dex; median weight class
        elif effect == 126:
            roll = self.rng.randrange(100)
            power = 10 if roll < 5 else 30 if roll < 15 else 50 if roll < 35 else 70 if roll < 65 else \
                90 if roll < 85 else 110 if roll < 95 else 150
        elif effect == 169 and fighter.status:
            power *= 2
        elif effect == 171 and foe.status & PARALYSIS:
            power *= 2
        elif effect == 185 and fighter.hit_by_target:
            power *= 2
        elif effect == 149 and foe.invulnerable:
            power *= 2
        return move_type, power

    def _damage(self, side: int, move: MoveData, move_type: int, power: int, critical: bool,
                fixed: int | None = None, explosion: bool = False) -> tuple[int, float]:
        attacker, defender = self.sides[side].fighter, self.sides[1 - side].fighter
        foe_side = self.sides[1 - side]
        typeless = move_type == TYPE_MYSTERY or move.move_id == MOVE_STRUGGLE
        if typeless:
            modifiers: tuple[tuple[int, int], ...] = ((1, 1),)
        else:
            modifiers = self.dex.modifiers(move_type, defender.types)[:2] or ((1, 1),)
        multiplier = 1.0
        for numerator, denominator in modifiers:
            multiplier *= numerator / denominator
        if fixed is not None and move_type != TYPE_MYSTERY and multiplier == 0:
            return 0, 0.0
        if fixed is not None:
            return fixed, multiplier if multiplier else 1.0
        special = move_type >= 10
        weather = self._weather_active()
        context = DamageContext(
            level=attacker.spec.level, attack=attacker.stat(4 if special else 1), defense=defender.stat(5 if special else 2),
            power=power, move_type=move_type, type_modifiers=modifiers,
            stab=not typeless and move_type in attacker.types,
            attack_stage=attacker.stages[SPA if special else ATK], defense_stage=defender.stages[SPD if special else DEF],
            attacker_ability=attacker.ability, defender_ability=defender.ability,
            attacker_item=attacker.item, defender_item=defender.item,
            attacker_species=attacker.spec.species, defender_species=defender.spec.species,
            attacker_status=attacker.status, defender_status=defender.status,
            attacker_hp=attacker.hp, attacker_max_hp=attacker.max_hp, weather=weather,  # type: ignore[arg-type]
            reflect=foe_side.reflect > 0, light_screen=foe_side.light_screen > 0, explosion=explosion,
            solar_beam=move.effect == 151, flash_fire_active=attacker.flash_fire)
        damage = GbaIntegerSafetyVerifier.compute_context_damage_reference(context, self.rng.randint(85, 100), critical)
        return damage, multiplier

    def _attack(self, side: int, slot: int, move: MoveData, effect: int, moved_last: bool, struggling: bool) -> None:
        foe_side = 1 - side
        attacker, defender = self.sides[side].fighter, self.sides[foe_side].fighter
        move_type, power = self._move_type_power(side, move, effect)
        # Ability immunities that absorb the move.
        absorb = {A["VOLT_ABSORB"]: 13, A["WATER_ABSORB"]: 11}
        if defender.ability in absorb and move_type == absorb[defender.ability] and move.damaging:
            self._emit(("ability", foe_side, defender.ability))
            self._heal(foe_side, defender.max_hp // 4)
            self._emit(("noeffect", foe_side, move_type, side, defender.spec.species))
            return
        if defender.ability == A["FLASH_FIRE"] and move_type == 10 and move.damaging and not defender.status & FREEZE:
            defender.flash_fire = True
            self._emit(("ability", foe_side, defender.ability))
            self._emit(("noeffect", foe_side, move_type, side, defender.spec.species))
            return
        if defender.ability == A["LEVITATE"] and move_type == 4 and move.damaging:
            self._emit(("ability", foe_side, defender.ability))
            self._emit(("noeffect", foe_side, move_type, side, defender.spec.species))
            return
        if effect == 8 and not defender.status & SLEEP_MASK:  # Dream Eater
            self._emit(("fail", side, move.move_id))
            return
        if effect == 38 and defender.ability == A["STURDY"]:
            self._emit(("ability", foe_side, A["STURDY"]))
            self._emit(("fail", side, move.move_id))
            return
        fixed: int | None = None
        if effect == 41:
            fixed = 40
        elif effect == 130:
            fixed = 20
        elif effect == 87:
            fixed = attacker.spec.level
        elif effect == 88:
            fixed = max(1, attacker.spec.level * (self.rng.randrange(11) * 10 + 50) // 100)
        elif effect == 40:
            fixed = max(1, defender.hp // 2)
        elif effect == 189:
            if defender.hp <= attacker.hp:
                self._emit(("fail", side, move.move_id))
                return
            fixed = defender.hp - attacker.hp
        elif effect == 38:
            fixed = defender.hp
        elif effect == 89:
            if not attacker.taken_physical or not moved_last:
                self._emit(("fail", side, move.move_id))
                return
            fixed = attacker.taken_physical * 2
        elif effect == 144:
            if not attacker.taken_special or not moved_last:
                self._emit(("fail", side, move.move_id))
                return
            fixed = attacker.taken_special * 2
        hits = 1
        if effect == 29:
            hits = self.rng.choice((2, 2, 2, 3, 3, 3, 4, 5))
        elif effect in (44, 77):
            hits = 2
        elif effect == 104:
            hits = 3
        total = 0
        category = 1.0
        for hit in range(hits):
            if not defender.alive:
                break
            hit_power = power * (hit + 1) if effect == 104 else power
            stage = (2 if attacker.focus_energy else 0) + int(effect in (43, 200, 209) or move.move_id in (2, 13, 75, 143, 163, 177, 238, 314, 348)) \
                + int(attacker.item == ITEM_SCOPE_LENS)
            critical = (fixed is None and defender.ability not in (A["BATTLE_ARMOR"], A["SHELL_ARMOR"])
                        and self.rng.randrange(CRITICAL_CHANCE[min(4, stage)]) == 0)
            damage, category = self._damage(side, move, move_type, hit_power, critical, fixed, explosion=effect == 7)
            # Fixed-damage and typeless hits print no effectiveness message in Gen 3.
            label = "fixed" if fixed is not None or move.move_id == MOVE_STRUGGLE else self._category_name(category)
            if category == 0:
                self._emit(("noeffect", foe_side, move_type, side, defender.spec.species))
                if effect == 7:
                    self._self_faint(side)
                return
            if defender.ability == A["WONDER_GUARD"] and category <= 1 and fixed is None and move_type != TYPE_MYSTERY:
                self._emit(("ability", foe_side, A["WONDER_GUARD"]))
                self._emit(("noeffect", foe_side, move_type, side, defender.spec.species))
                return
            if effect == 101 or defender.enduring:
                damage = min(damage, defender.hp - 1) if defender.substitute == 0 else damage
            if defender.substitute:
                dealt = min(damage, defender.substitute)
                defender.substitute -= dealt
                self._emit(("hit", foe_side, dealt, label, critical, move_type, side, defender.spec.species, True))
                if not defender.substitute:
                    self._emit(("substitute_broke", foe_side))
                total += dealt
                continue
            if defender.item == ITEM_FOCUS_BAND and damage >= defender.hp and self.rng.randrange(10) == 0:
                damage = defender.hp - 1
                self._emit(("item", foe_side, ITEM_FOCUS_BAND))
            dealt = min(damage, defender.hp)
            defender.hp -= dealt
            total += dealt
            defender.hit_by_target = True
            if move.special:
                defender.taken_special += dealt
            else:
                defender.taken_physical += dealt
            self._emit(("hit", foe_side, dealt, label, critical, move_type, side, defender.spec.species, False))
            if defender.ability == A["COLOR_CHANGE"] and defender.alive and move_type != TYPE_MYSTERY \
                    and move_type not in defender.types and move.damaging:
                defender.types = (move_type, move_type)
                self._emit(("ability", foe_side, A["COLOR_CHANGE"]))
            if move.contact and defender.alive:
                self._contact(side, foe_side)
            if not defender.alive:
                self._emit(("faint", foe_side, self.sides[foe_side].active))
                break
            if not attacker.alive:
                break
        if hits > 1:
            self._emit(("multihit", side, hits))
        if total and attacker.alive:
            self._after_damage(side, move, effect, total, moved_last, struggling, category)
        if effect == 7:
            self._self_faint(side)
        self._berries(foe_side)

    @staticmethod
    def _category_name(multiplier: float) -> str:
        if multiplier == 0:
            return "no_effect"
        if multiplier < 1:
            return "resisted"
        if multiplier > 1:
            return "super"
        return "neutral"

    def _contact(self, side: int, foe_side: int) -> None:
        attacker, defender = self.sides[side].fighter, self.sides[foe_side].fighter
        ability = defender.ability
        if ability == A["ROUGH_SKIN"]:
            self._emit(("ability", foe_side, ability))
            self._hurt(side, max(1, attacker.max_hp // 16), "rough_skin")
            return
        if self.rng.randrange(3):
            return
        kind = {A["STATIC"]: PARALYSIS, A["FLAME_BODY"]: BURN, A["POISON_POINT"]: POISON}.get(ability)
        if ability == A["EFFECT_SPORE"]:
            kind = self.rng.choice((POISON, PARALYSIS, 1))
        if kind is not None and self._inflict(side, kind, foe_side, by_move=False):
            self._emit(("ability", foe_side, ability))

    def _after_damage(self, side: int, move: MoveData, effect: int, total: int, moved_last: bool,
                      struggling: bool, category: float) -> None:
        foe_side = 1 - side
        attacker, defender = self.sides[side].fighter, self.sides[foe_side].fighter
        if struggling:
            self._hurt(side, max(1, total // 4), "recoil")
            return
        if effect in (3, 8):
            amount = max(1, total // 2)
            if defender.ability == A["LIQUID_OOZE"]:
                self._emit(("ability", foe_side, A["LIQUID_OOZE"]))
                self._hurt(side, amount, "liquid_ooze")
            else:
                self._heal(side, amount)
        elif effect in (48, 198) and attacker.ability != A["ROCK_HEAD"]:
            self._hurt(side, max(1, total // (3 if effect == 198 else 4)), "recoil")
        elif effect == 80:
            attacker.recharge = True
        elif effect == 182:
            self._boost(side, ATK, -1)
            self._boost(side, DEF, -1)
        elif effect == 204:
            self._boost(side, SPA, -2)
        elif effect == 186:
            foe = self.sides[foe_side]
            foe.reflect = foe.light_screen = 0
        elif effect == 129:
            attacker.seeded = False
            attacker.trap_turns = 0
            self.sides[side].spikes = 0
        elif effect == 188 and defender.item and defender.alive:
            defender.item = 0
            self._emit(("knocked_off", foe_side))
        elif effect == 171 and defender.status & PARALYSIS:
            defender.status &= ~PARALYSIS
        elif effect == 27:
            if not attacker.rampage:
                attacker.rampage = True
                attacker.lock_slot = attacker.last_slot
                attacker.lock_turns = self.rng.randrange(2, 4)
            attacker.lock_turns -= 1
            if attacker.lock_turns <= 0:
                attacker.rampage = False
                attacker.lock_slot = -1
                self._confuse(side, side)
        elif effect == 42 and defender.alive and not defender.trap_turns:
            defender.trap_turns = self.rng.randrange(2, 6)
            self._emit(("trapped", foe_side))
        if attacker.item == ITEM_SHELL_BELL:
            self._heal(side, max(1, total // 8))
        if not defender.alive:
            return
        chance = move.chance * (2 if attacker.ability == A["SERENE_GRACE"] else 1)
        if defender.ability == A["SHIELD_DUST"] and effect not in (138, 139, 140, 182, 204):
            chance = 0
        if defender.substitute and effect not in (138, 139, 140):
            chance = 0
        roll = self.rng.randrange(100) < chance if chance else effect in (158,)
        if effect == 158 and not attacker.turns_active:
            roll = True
        flinch_kings = attacker.item == ITEM_KINGS_ROCK and move.flags & 32 and self.rng.randrange(100) < 10
        if (roll and effect in (31, 146, 150, 158) or flinch_kings) and not moved_last \
                and defender.ability != A["INNER_FOCUS"]:
            defender.flinched = True
        if not roll:
            return
        secondary = {2: POISON, 4: BURN, 5: FREEZE, 6: PARALYSIS, 125: BURN, 152: PARALYSIS, 200: BURN,
                     202: TOXIC, 209: POISON, 77: POISON, 197: PARALYSIS}
        if effect in secondary:
            self._inflict(foe_side, secondary[effect], side, by_move=True)
        elif effect == 36:
            self._inflict(foe_side, self.rng.choice((PARALYSIS, BURN, FREEZE)), side, by_move=True)
        elif effect in (76,):
            self._confuse(foe_side, side)
        elif effect in _STAT_DOWN_HIT:
            self._boost(foe_side, _STAT_DOWN_HIT[effect], -1, by_foe=True)
        elif effect == 138:
            self._boost(side, DEF, 1)
        elif effect == 139:
            self._boost(side, ATK, 1)
        elif effect == 140:
            for stat in (ATK, DEF, SPE, SPA, SPD):
                self._boost(side, stat, 1)

    # ---------------------------------------------------- status moves
    def _status_move(self, side: int, move: MoveData, effect: int) -> None:
        foe_side = 1 - side
        attacker, defender = self.sides[side].fighter, self.sides[foe_side].fighter
        if defender.substitute and effect not in (28, 86, 90, 175, 106):
            self._emit(("fail", side, move.move_id))
            return
        immune_type = move.move_type in (13,) and effect == 67 and self.dex.multiplier(13, defender.types) == 0
        if immune_type:
            self._emit(("noeffect", foe_side, move.move_type, side, defender.spec.species))
            return
        success = True
        if effect == 1:
            success = self._inflict(foe_side, 1, side, by_move=True)
        elif effect in (66, 67, 33, 167):
            kind = {66: POISON, 67: PARALYSIS, 33: TOXIC, 167: BURN}[effect]
            success = self._inflict(foe_side, kind, side, by_move=True)
        elif effect in (49, 199):
            success = self._confuse(foe_side, side)
        elif effect == 118:
            success = self._boost(foe_side, ATK, 2) | self._confuse(foe_side, side)
        elif effect == 166:
            success = self._boost(foe_side, SPA, 1) | self._confuse(foe_side, side)
        elif effect in _STAT_DOWN:
            stat, amount = _STAT_DOWN[effect]
            success = self._boost(foe_side, stat, -amount, by_foe=True)
        elif effect == 205:
            success = self._boost(foe_side, ATK, -1, by_foe=True) | self._boost(foe_side, DEF, -1, by_foe=True)
        elif effect == 84:
            if 12 in defender.types or defender.seeded:
                success = False
            else:
                defender.seeded = True
                self._emit(("seeded", foe_side))
        elif effect == 28:
            success = self._force_switch(foe_side)
        elif effect == 25:
            for target in (attacker, defender):
                target.stages = [6] * 8
            self._emit(("haze",))
        elif effect == 143:
            attacker.stages = list(defender.stages)
            self._emit(("psych_up", side))
        elif effect == 91:
            average = (attacker.hp + defender.hp) // 2
            attacker.hp, defender.hp = min(attacker.max_hp, average), min(defender.max_hp, average)
            self._emit(("pain_split", side, average))
        elif effect == 106:
            defender.trapped_by = True
            self._emit(("trapped", foe_side))
        elif effect == 109:
            if 7 in attacker.types:
                self._hurt(side, max(1, attacker.max_hp // 2), "curse")
                defender.cursed = True
                self._emit(("cursed", foe_side))
            else:
                self._boost(side, SPE, -1)
                self._boost(side, ATK, 1)
                self._boost(side, DEF, 1)
        elif effect == 168:
            self._boost(foe_side, ATK, -2, by_foe=True)
            self._boost(foe_side, SPA, -2, by_foe=True)
            self._self_faint(side)
        elif effect == 175:
            if defender.taunt:
                success = False
            else:
                defender.taunt = 2
                self._emit(("taunted", foe_side))
        elif effect == 90:
            if defender.encore or defender.last_slot < 0:
                success = False
            else:
                defender.encore = self.rng.randrange(3, 8)
                self._emit(("encored", foe_side))
        elif effect == 86:
            if defender.disabled_turns or defender.last_slot < 0:
                success = False
            else:
                defender.disabled_slot = defender.last_slot
                defender.disabled_turns = self.rng.randrange(2, 6)
                self._emit(("disabled", foe_side, defender.disabled_slot))
        elif effect == 187:
            if defender.status or defender.yawn or self.sides[foe_side].safeguard:
                success = False
            else:
                defender.yawn = 2
                self._emit(("yawn", foe_side))
        elif effect == 85:
            self._emit(("splash", side))
        else:
            success = False
        if not success:
            self._emit(("fail", side, move.move_id))

    def _self_effect(self, side: int, move: MoveData, effect: int, moved_last: bool) -> None:
        own = self.sides[side]
        fighter = own.fighter
        success = True
        if effect in _STAT_UP:
            stat, amount = _STAT_UP[effect]
            success = self._boost(side, stat, amount)
        elif effect in _MULTI_BOOST:
            success = any([self._boost(side, stat, amount) for stat, amount in _MULTI_BOOST[effect]])
        elif effect in _HEAL_HALF:
            success = self._heal(side, fighter.max_hp // 2) > 0
        elif effect in _WEATHER_HEAL:
            weather = self._weather_active()
            share = fighter.max_hp * 2 // 3 if weather == "sun" else (fighter.max_hp // 4 if weather != "none" else fighter.max_hp // 2)
            success = self._heal(side, share) > 0
        elif effect == 37:
            if fighter.hp == fighter.max_hp or fighter.ability in (A["INSOMNIA"], A["VITAL_SPIRIT"]):
                success = False
            else:
                fighter.status = 3
                fighter.toxic_n = 0
                self._heal(side, fighter.max_hp)
                self._emit(("status", side, "slp"))
        elif effect == 79:
            cost = fighter.max_hp // 4
            if fighter.substitute or fighter.hp <= cost:
                success = False
            else:
                fighter.hp -= cost
                fighter.substitute = cost
                self._emit(("substitute", side))
        elif effect in (111, 116):
            # Gen 3: fails when used last in the turn; each consecutive use halves the odds.
            if self.rng.random() < 1 / (2 ** min(fighter.protect_streak, 8)) and not moved_last:
                fighter.protected = effect == 111
                fighter.enduring = effect == 116
                fighter.protect_streak += 1
                self._emit(("protect" if effect == 111 else "endure", side))
            else:
                fighter.protect_streak = 0
                success = False
        elif effect in (35, 65):
            attribute = "light_screen" if effect == 35 else "reflect"
            if getattr(own, attribute):
                success = False
            else:
                setattr(own, attribute, 5)
                self._emit(("screen", side, attribute))
        elif effect == 112:
            foe = self.sides[1 - side]
            if foe.spikes >= 3:
                success = False
            else:
                foe.spikes += 1
                self._emit(("spikes", 1 - side, foe.spikes))
        elif effect in (115, 136, 137, 164):
            success = self._set_weather({115: "sandstorm", 136: "rain", 137: "sun", 164: "hail"}[effect], 5)
        elif effect == 124:
            success = not own.safeguard
            own.safeguard = own.safeguard or 5
        elif effect == 46:
            success = not own.mist
            own.mist = own.mist or 5
        elif effect == 47:
            success = not fighter.focus_energy
            fighter.focus_energy = True
        elif effect == 142:
            if fighter.hp <= fighter.max_hp // 2 or fighter.stages[ATK] >= 12:
                success = False
            else:
                fighter.hp -= fighter.max_hp // 2
                fighter.stages[ATK] = 12
                self._emit(("boost", side, ATK, 12))
        elif effect == 179:
            if own.wish_turns:
                success = False
            else:
                own.wish_turns, own.wish_hp = 2, fighter.max_hp // 2
        elif effect == 181:
            success = not fighter.ingrain
            fighter.ingrain = True
        elif effect == 193:
            if fighter.status & (POISON | BURN | PARALYSIS | TOXIC):
                fighter.status = 0
                self._emit(("cured", side))
            else:
                success = False
        elif effect == 102:
            for member in own.party:
                member.status = 0 if member.alive else member.status
            self._emit(("cured", side))
        else:
            success = False
        if not success:
            self._emit(("fail", side, move.move_id))

    def _force_switch(self, side: int) -> bool:
        own = self.sides[side]
        fighter = own.fighter
        options = [index for index in own.bench() if own.party[index].alive]
        if not options or fighter.ingrain or fighter.ability == A["SUCTION_CUPS"]:
            return False
        self._switch_in(side, self.rng.choice(options))
        return True

    # ----------------------------------------------------- state changes
    def _boost(self, side: int, stat: int, amount: int, by_foe: bool = False) -> bool:
        fighter = self.sides[side].fighter
        if not fighter.alive:
            return False
        if amount < 0 and by_foe:
            if self.sides[side].mist or fighter.ability in (A["CLEAR_BODY"], A["WHITE_SMOKE"]) or \
                    (stat == ATK and fighter.ability == A["HYPER_CUTTER"]) or \
                    (stat == ACC and fighter.ability == A["KEEN_EYE"]):
                if fighter.ability in (A["CLEAR_BODY"], A["WHITE_SMOKE"], A["HYPER_CUTTER"], A["KEEN_EYE"]):
                    self._emit(("ability", side, fighter.ability))
                return False
        new = max(0, min(12, fighter.stages[stat] + amount))
        if new == fighter.stages[stat]:
            return False
        fighter.stages[stat] = new
        self._emit(("boost", side, stat, amount))
        return True

    def _inflict(self, side: int, kind: int, source: int, by_move: bool) -> bool:
        fighter = self.sides[side].fighter
        if not fighter.alive or fighter.status:
            return False
        if by_move and source != side and self.sides[side].safeguard:
            return False
        ability = fighter.ability
        if kind == 1:  # sleep
            if ability in (A["INSOMNIA"], A["VITAL_SPIRIT"]):
                self._emit(("ability", side, ability))
                return False
            fighter.status = self.rng.randrange(2, 6)
            self._emit(("status", side, "slp"))
            return True
        if kind in (POISON, TOXIC) and (3 in fighter.types or 8 in fighter.types or ability == A["IMMUNITY"]):
            return False
        if kind == BURN and (10 in fighter.types or ability == A["WATER_VEIL"]):
            return False
        if kind == FREEZE and (15 in fighter.types or ability == A["MAGMA_ARMOR"] or self._weather_active() == "sun"):
            return False
        if kind == PARALYSIS and ability == A["LIMBER"]:
            return False
        fighter.status = kind
        fighter.toxic_n = 1 if kind == TOXIC else 0
        self._emit(("status", side, STATUS_NAMES[kind]))
        if ability == A["SYNCHRONIZE"] and source != side and kind in (POISON, BURN, PARALYSIS, TOXIC):
            self._emit(("ability", side, ability))
            self._inflict(source, POISON if kind == TOXIC else kind, side, by_move=False)
        self._berries(side)
        return True

    def _confuse(self, side: int, source: int) -> bool:
        fighter = self.sides[side].fighter
        if not fighter.alive or fighter.confusion or fighter.ability == A["OWN_TEMPO"]:
            return False
        if source != side and self.sides[side].safeguard:
            return False
        fighter.confusion = self.rng.randrange(2, 6)
        self._emit(("confused", side))
        return True

    def _confusion_hit(self, side: int) -> None:
        fighter = self.sides[side].fighter
        context = DamageContext(level=fighter.spec.level, attack=fighter.stat(1), defense=fighter.stat(2), power=40,
                                move_type=0, attack_stage=fighter.stages[ATK], defense_stage=fighter.stages[DEF],
                                attacker_status=fighter.status, attacker_hp=fighter.hp, attacker_max_hp=fighter.max_hp)
        damage = GbaIntegerSafetyVerifier.compute_context_damage_reference(context, self.rng.randint(85, 100), False)
        self._emit(("confusion_hit", side))
        self._hurt(side, damage, "confusion")

    def _hurt(self, side: int, amount: int, cause: str) -> None:
        fighter = self.sides[side].fighter
        if not fighter.alive or amount <= 0:
            return
        dealt = min(amount, fighter.hp)
        fighter.hp -= dealt
        self._emit(("residual", side, dealt, cause))
        if not fighter.alive:
            fighter.status = 0
            self._emit(("faint", side, self.sides[side].active))
        else:
            self._berries(side)

    def _heal(self, side: int, amount: int) -> int:
        fighter = self.sides[side].fighter
        healed = min(amount, fighter.max_hp - fighter.hp) if fighter.alive else 0
        if healed > 0:
            fighter.hp += healed
            self._emit(("heal", side, healed))
        return healed

    def _self_faint(self, side: int) -> None:
        fighter = self.sides[side].fighter
        if fighter.alive:
            fighter.hp = 0
            fighter.status = 0
            self._emit(("faint", side, self.sides[side].active))

    def _berries(self, side: int) -> None:
        fighter = self.sides[side].fighter
        if not fighter.alive or not fighter.item:
            return
        item = fighter.item
        if item in (ITEM_ORAN, ITEM_SITRUS) and fighter.hp <= fighter.max_hp // 2:
            fighter.item = 0
            self._emit(("item", side, item))
            self._heal(side, 10 if item == ITEM_ORAN else 30)
        elif item == ITEM_LUM and (fighter.status or fighter.confusion):
            fighter.item = 0
            fighter.status = fighter.confusion = 0
            self._emit(("item", side, item))
        elif item in STATUS_BERRIES and fighter.status & STATUS_BERRIES[item]:
            fighter.item = 0
            fighter.status = 0
            self._emit(("item", side, item))
        elif item == ITEM_PERSIM and fighter.confusion:
            fighter.item = 0
            fighter.confusion = 0
            self._emit(("item", side, item))

    # ------------------------------------------------------ end of turn
    def _end_of_turn(self) -> None:
        weather = self._weather_active()
        if self.weather != "none" and self.weather_turns > 0:
            self.weather_turns -= 1
            if not self.weather_turns:
                self._emit(("weather_end", self.weather))
                self.weather = "none"
        for side in self._speed_order():
            own = self.sides[side]
            fighter = own.fighter
            foe = self.sides[1 - side].fighter
            if not fighter.alive:
                continue
            if weather == "sandstorm" and not {4, 5, 8} & set(fighter.types) and fighter.ability != A["SAND_VEIL"]:
                self._hurt(side, max(1, fighter.max_hp // 16), "sandstorm")
            elif weather == "hail" and 15 not in fighter.types:
                self._hurt(side, max(1, fighter.max_hp // 16), "hail")
            if own.wish_turns:
                own.wish_turns -= 1
                if not own.wish_turns and fighter.alive:
                    self._heal(side, own.wish_hp)
            if fighter.alive and fighter.item == ITEM_LEFTOVERS:
                self._heal(side, max(1, fighter.max_hp // 16))
            if fighter.alive and fighter.ability == A["RAIN_DISH"] and weather == "rain":
                self._heal(side, max(1, fighter.max_hp // 16))
            if fighter.alive and fighter.ingrain:
                self._heal(side, max(1, fighter.max_hp // 16))
            if fighter.alive and fighter.seeded and foe.alive:
                drained = min(fighter.hp, max(1, fighter.max_hp // 8))
                self._hurt(side, drained, "leech_seed")
                self._heal(1 - side, drained)
            if fighter.alive and fighter.status & POISON:
                self._hurt(side, max(1, fighter.max_hp // 8), "poison")
            elif fighter.alive and fighter.status & TOXIC:
                self._hurt(side, max(1, fighter.max_hp * fighter.toxic_n // 16), "toxic")
                fighter.toxic_n = min(15, fighter.toxic_n + 1)
            elif fighter.alive and fighter.status & BURN:
                self._hurt(side, max(1, fighter.max_hp // 8), "burn")
            if fighter.alive and fighter.cursed:
                self._hurt(side, max(1, fighter.max_hp // 4), "curse")
            if fighter.alive and fighter.trap_turns:
                fighter.trap_turns -= 1
                self._hurt(side, max(1, fighter.max_hp // 16), "trap")
            if fighter.alive and fighter.yawn:
                fighter.yawn -= 1
                if not fighter.yawn:
                    self._inflict(side, 1, 1 - side, by_move=False)
            if fighter.alive and fighter.ability == A["SHED_SKIN"] and fighter.status and self.rng.randrange(3) == 0:
                fighter.status = 0
                self._emit(("ability", side, A["SHED_SKIN"]))
            if fighter.alive and fighter.ability == A["SPEED_BOOST"] and fighter.turns_active:
                if self._boost(side, SPE, 1):
                    self._emit(("ability", side, A["SPEED_BOOST"]))
            for counter in ("taunt", "encore", "disabled_turns"):
                value = getattr(fighter, counter)
                if value:
                    setattr(fighter, counter, value - 1)
            if not fighter.protected and not fighter.enduring:
                fighter.protect_streak = 0
            for attribute in ("reflect", "light_screen", "safeguard", "mist"):
                value = getattr(own, attribute)
                if value:
                    setattr(own, attribute, value - 1)
            if fighter.alive:
                self._berries(side)


def team_hp(battle: Battle, side: int) -> float:
    return battle.sides[side].hp_fraction()
