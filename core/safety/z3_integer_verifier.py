"""Gen-3 integer damage, with SMT proofs conditional on complete battle inputs.

The operation order follows pret's ROM-matching ``CalculateBaseDamage`` and
``Cmd_damagecalc``/``Cmd_typecalc``/``ApplyRandomDmgMultiplier``. In particular,
physical damage has a minimum BEFORE the +2 term; special damage does not.
Critical hits ignore negative attacker stages and positive defender stages.
Type modifiers must be supplied in engine type-table order, which need not be
the defender's type-slot order. This model does not certify unobserved moves,
unknown abilities, arbitrary move effects, or multi-turn future damage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import z3

Category = Literal["physical", "special"]
Weather = Literal["none", "rain", "sun", "sandstorm", "hail"]
ProofStatus = Literal["proved", "counterexample", "unknown", "unsupported"]

STAGE_RATIOS: tuple[tuple[int, int], ...] = (
    (2, 8), (2, 7), (2, 6), (2, 5), (2, 4), (2, 3), (2, 2),
    (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2),
)

# Native generation-three IDs, not modern names or the modern category split.
ABILITY_VOLT_ABSORB = 10
ABILITY_WATER_ABSORB = 11
ABILITY_FLASH_FIRE = 18
ABILITY_HUGE_POWER = 37
ABILITY_THICK_FAT = 47
ABILITY_PURE_POWER = 74
ABILITY_HUSTLE = 55
ABILITY_PLUS = 57
ABILITY_MINUS = 58
ABILITY_GUTS = 62
ABILITY_MARVEL_SCALE = 63
ABILITY_OVERGROW = 65
ABILITY_BLAZE = 66
ABILITY_TORRENT = 67
ABILITY_SWARM = 68
ABILITY_LEVITATE = 26
ABILITY_WONDER_GUARD = 25
ABILITY_BATTLE_ARMOR = 4
ABILITY_SHELL_ARMOR = 75
STATUS_BURN = 0x10

# Item ID -> (hold effect, effect parameter, affected move type).
_TYPE_ITEMS: dict[int, tuple[int, int]] = {
    217: (0, 10), 207: (1, 10), 210: (2, 10), 211: (3, 10),
    203: (4, 10), 204: (5, 10), 188: (6, 10), 213: (7, 10),
    199: (8, 10), 215: (10, 10), 209: (11, 10), 205: (12, 10),
    208: (13, 10), 214: (14, 10), 212: (15, 10), 216: (16, 10),
    206: (17, 10),
}


@dataclass(frozen=True, slots=True)
class DamageContext:
    """Known inputs to one incoming damage event.

    ``attack``/``defense`` are the selected raw attack or special-attack and
    defense or special-defense stats. Stage values use the native +6 bias.
    Duplicate defender types must be represented by only one modifier.
    Item effects are resolved from native IDs; an Enigma Berry uses explicit
    type-item boost/effect inputs supplied by its observed hold effect.
    """

    level: int
    attack: int
    defense: int
    power: int
    move_type: int = 0
    type_modifiers: tuple[tuple[int, int], ...] = ((1, 1),)
    stab: bool = False
    attack_stage: int = 6
    defense_stage: int = 6
    category: Category | None = None
    attacker_ability: int = 0
    defender_ability: int = 0
    attacker_item: int = 0
    defender_item: int = 0
    attacker_species: int = 0
    defender_species: int = 0
    attacker_status: int = 0
    defender_status: int = 0
    attacker_hp: int = 100
    attacker_max_hp: int = 100
    attacker_badge_boost: bool = False
    defender_badge_boost: bool = False
    type_item_boost_percent: int = 0
    plus_minus_active: bool = False
    frontier_battle: bool = False
    weather: Weather = "none"
    weather_suppressed: bool = False
    reflect: bool = False
    light_screen: bool = False
    double_battle: bool = False
    defending_side_alive: int = 1
    targets_both: bool = False
    explosion: bool = False
    solar_beam: bool = False
    flash_fire_active: bool = False
    charged_up: bool = False
    helping_hand: bool = False
    mud_sport: bool = False
    water_sport: bool = False
    damage_multiplier: int = 1
    hit_count: int = 1
    fixed_damage: int | None = None
    false_swipe: bool = False
    endure: bool = False
    defender_hp: int | None = None
    substitute_hp: int = 0
    unsupported_effects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 1 <= self.level <= 100:
            raise ValueError("level must be in [1, 100]")
        if not 1 <= self.attack <= 65535 or not 1 <= self.defense <= 65535:
            raise ValueError("raw stats must be positive uint16 values")
        if not 0 <= self.power <= 65535:
            raise ValueError("power must be a uint16 value")
        if not 0 <= self.move_type <= 17:
            raise ValueError("move_type must be a native Gen-3 type ID")
        if not 0 <= self.attack_stage <= 12 or not 0 <= self.defense_stage <= 12:
            raise ValueError("stat stages must be native RAM values in [0, 12]")
        expected = "special" if self.move_type >= 10 else "physical"
        if self.category is not None and self.category != expected:
            raise ValueError("category must follow the Gen-3 type category split")
        if not 1 <= len(self.type_modifiers) <= 2:
            raise ValueError("one or two ordered type modifiers are required")
        for numerator, denominator in self.type_modifiers:
            if denominator <= 0 or numerator < 0 or numerator > denominator * 4:
                raise ValueError("invalid nonnegative type modifier")
        if not 0 <= self.attacker_hp <= self.attacker_max_hp or self.attacker_max_hp < 1:
            raise ValueError("invalid attacker HP")
        if self.defending_side_alive not in (1, 2):
            raise ValueError("defending_side_alive must be 1 or 2")
        if self.weather not in ("none", "rain", "sun", "sandstorm", "hail"):
            raise ValueError("invalid weather")
        if not 0 <= self.type_item_boost_percent <= 100:
            raise ValueError("invalid type item boost")
        if not 1 <= self.damage_multiplier <= 4 or not 1 <= self.hit_count <= 5:
            raise ValueError("invalid scripted multiplier or hit count")
        if self.fixed_damage is not None and not 0 <= self.fixed_damage <= 65535:
            raise ValueError("invalid fixed damage")
        if self.defender_hp is not None and self.defender_hp < 1:
            raise ValueError("defender_hp must be positive when supplied")
        if (self.endure or self.false_swipe) and self.defender_hp is None:
            raise ValueError("False Swipe/Endure requires defender_hp")
        if self.substitute_hp < 0:
            raise ValueError("invalid substitute HP")

    @property
    def selected_category(self) -> Category:
        """Return the category derived from the native type, including reserve9."""
        return "special" if self.move_type >= 10 else "physical"


@dataclass(frozen=True, slots=True)
class DamageResult:
    """A conditional proof and its worst-case damage, never an unconditional guarantee."""

    formally_safe: bool
    worst_case_damage: int
    residual_damage: int
    worst_case_total: int
    proof_status: ProofStatus
    reason: str
    counterexample_rng: int | None = None
    counterexample_critical: bool | None = None


@dataclass(frozen=True, slots=True)
class ResidualContext:
    """Known end-of-turn damage before the next opportunity to act.

    Leech Seed, trapping, weather, poison/burn and Curse are additive. Healing
    is deliberately omitted from a safety bound. Toxic turn is the native
    counter after its increment (1..15). Gen3 has no Magic Guard.
    """

    max_hp: int
    status: int = 0
    toxic_turn: int = 1
    sandstorm: bool = False
    hail: bool = False
    types: tuple[int, ...] = (0,)
    leech_seed: bool = False
    trapped: bool = False
    curse: bool = False
    nightmare: bool = False

    def __post_init__(self) -> None:
        if self.max_hp < 1 or not 1 <= self.toxic_turn <= 15:
            raise ValueError("invalid residual max HP or toxic counter")
        if not 1 <= len(self.types) <= 2 or any(t < 0 or t > 17 for t in self.types):
            raise ValueError("invalid residual types")


def compute_residual_damage(context: ResidualContext) -> int:
    """Conservative sum of known Gen-3 damage sources without future healing."""
    damage = 0
    if context.status & 0x80:
        damage += max(1, context.max_hp // 16) * context.toxic_turn
    elif context.status & (0x08 | STATUS_BURN):
        damage += max(1, context.max_hp // 8)
    if context.sandstorm and not any(t in (4, 5, 8) for t in context.types):
        damage += max(1, context.max_hp // 16)
    if context.hail and 15 not in context.types:
        damage += max(1, context.max_hp // 16)
    if context.leech_seed:
        damage += max(1, context.max_hp // 8)
    if context.trapped:
        damage += max(1, context.max_hp // 16)
    if context.curse:
        damage += max(1, context.max_hp // 4)
    if context.nightmare and context.status & 7:
        damage += max(1, context.max_hp // 4)
    return damage


def _zi(value: int) -> z3.ArithRef:
    return cast("z3.ArithRef", z3.IntVal(value))


def _zx(condition: object, yes: z3.ArithRef, no: z3.ArithRef) -> z3.ArithRef:
    return cast("z3.ArithRef", z3.If(condition, yes, no))


def _nonzero(value: z3.ArithRef) -> z3.ArithRef:
    return _zx(value == 0, _zi(1), value)


class GbaIntegerSafetyVerifier:
    """Exact integer expressions and finite-RNG counterexample queries.

    Solver unknown/timeout and explicitly uncovered effects are never called
    safe. A proof is conditional on the provided context including the rival's
    selected move; callers must enumerate all plausible hidden rival moves.
    ``solver`` remains public for instrumentation and controlled timeout tests.
    """

    def __init__(self, timeout_ms: int = 2000) -> None:
        if timeout_ms < 1:
            raise ValueError("solver timeout must be positive")
        self.solver = z3.Solver()
        self.solver.set(timeout=timeout_ms)
        self._basic_symbols = tuple(z3.Int(f"basic_{name}") for name in (
            "defense", "level", "attack", "power", "num", "den", "stab",
            "critical", "attack_stage", "defense_stage", "rng",
        ))
        d, level, a, power, num, den, stab, crit, astage, dstage, rng = self._basic_symbols
        an = _zx(astage >= 6, astage - 4, _zi(2))
        ad = _zx(astage >= 6, _zi(2), 8 - astage)
        dn = _zx(dstage >= 6, dstage - 4, _zi(2))
        dd = _zx(dstage >= 6, _zi(2), 8 - dstage)
        ae = _zx(z3.And(crit == 2, astage <= 6), a, a * an / ad)
        de = _zx(z3.And(crit == 2, dstage >= 6), d, d * dn / dd)
        base = ae * power * (level * 2 / 5 + 2) / _nonzero(de) / 50
        damage = (_nonzero(base) + 2) * crit
        damage = _zx(stab == 1, damage * 15 / 10, damage)
        damage = _zx(num == 0, _zi(0), _nonzero(damage * num / den))
        damage = _zx(damage == 0, _zi(0), _nonzero(damage * rng / 100))
        self._basic_expression = _zx(power == 0, _zi(0), damage)

    @staticmethod
    def compute_exact_arm7_damage_reference(
        def_stat: int, opp_level: int, opp_atk_stat: int, move_power: int,
        type_mult_num: int, type_mult_den: int, stab: bool = False,
        critical: bool = False, atk_stage: int = 6, def_stage: int = 6,
        rng_roll: int = 100,
    ) -> int:
        """Independent plain-Python physical normal-damage reference."""
        if not 85 <= rng_roll <= 100:
            raise ValueError("damage roll must be in [85, 100]")
        if not 1 <= opp_level <= 100 or def_stat < 1 or opp_atk_stat < 1:
            raise ValueError("invalid damage stats or level")
        if not 0 <= atk_stage <= 12 or not 0 <= def_stage <= 12:
            raise ValueError("invalid native stages")
        if move_power < 0 or type_mult_num < 0 or type_mult_den <= 0:
            raise ValueError("invalid move power or multiplier")
        if move_power == 0 or type_mult_num == 0:
            return 0
        table = ((2, 8), (2, 7), (2, 6), (2, 5), (2, 4), (2, 3), (2, 2),
                 (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2))
        effective_attack = opp_atk_stat
        if not critical or atk_stage > 6:
            effective_attack = effective_attack * table[atk_stage][0] // table[atk_stage][1]
        effective_defense = def_stat
        if not critical or def_stage < 6:
            effective_defense = effective_defense * table[def_stage][0] // table[def_stage][1]
        # Native legal stats do not produce zero denominators. A one-point raw
        # stat test input uses a conservative denominator of1 rather than UB.
        effective_defense = max(1, effective_defense)
        damage = effective_attack * move_power
        damage *= 2 * opp_level // 5 + 2
        damage //= effective_defense
        damage //= 50
        damage = max(1, damage) + 2
        if critical:
            damage *= 2
        if stab:
            damage = damage * 15 // 10
        damage = max(1, damage * type_mult_num // type_mult_den)
        return max(1, damage * rng_roll // 100)

    def compute_exact_arm7_damage_symbolic(
        self, def_stat: int, opp_level: int, opp_atk_stat: int, move_power: int,
        type_mult_num: int, type_mult_den: int, stab: bool = False,
        critical: bool = False, atk_stage: int = 6, def_stage: int = 6,
        rng_roll: int = 100,
    ) -> int:
        """Evaluate the reusable Z3 Int graph; useful for large parity sweeps."""
        DamageContext(opp_level, opp_atk_stat, def_stat, move_power,
                      type_modifiers=((type_mult_num, type_mult_den),),
                      attack_stage=atk_stage, defense_stage=def_stage)
        if not 85 <= rng_roll <= 100:
            raise ValueError("invalid RNG roll")
        values = (def_stat, opp_level, opp_atk_stat, move_power, type_mult_num,
                  type_mult_den, int(stab), 2 if critical else 1, atk_stage,
                  def_stage, rng_roll)
        substitutions = tuple((symbol, _zi(value)) for symbol, value in
                              zip(self._basic_symbols, values, strict=True))
        return int(z3.simplify(z3.substitute(self._basic_expression, *substitutions)).as_long())

    def verify_worst_case_survival(
        self, current_hp: int, def_stat: int, opp_level: int, opp_atk_stat: int,
        move_power: int, type_mult_num: int, type_mult_den: int,
        stab: bool = False, can_crit: bool = True, atk_stage: int = 6,
        def_stage: int = 6,
    ) -> tuple[bool, int]:
        """Locked API: physical normal damage, maximum RNG and optional critical."""
        if not 0 <= current_hp <= 65535:
            raise ValueError("current_hp must be a uint16 HP value")
        worst = self.compute_exact_arm7_damage_symbolic(
            def_stat, opp_level, opp_atk_stat, move_power, type_mult_num,
            type_mult_den, stab, can_crit, atk_stage, def_stage)
        rng, critical = z3.Int("rng_roll"), z3.Int("crit_mult")
        concrete = (def_stat, opp_level, opp_atk_stat, move_power,
                    type_mult_num, type_mult_den, int(stab))
        substitutions = tuple((symbol, _zi(value)) for symbol, value in
                              zip(self._basic_symbols[:7], concrete, strict=True))
        substitutions += ((self._basic_symbols[7], critical),
                          (self._basic_symbols[8], _zi(atk_stage)),
                          (self._basic_symbols[9], _zi(def_stage)),
                          (self._basic_symbols[10], rng))
        expression = z3.substitute(self._basic_expression, *substitutions)
        self.solver.reset()
        self.solver.add(rng >= 85, rng <= 100)
        self.solver.add(z3.Or(critical == 1, critical == 2) if can_crit else critical == 1)
        self.solver.add(expression >= current_hp)
        native_positive_domain = opp_atk_stat * move_power * (2 * opp_level // 5 + 2) * 4 <= 0x7FFFFFFF
        native_positive_domain = native_positive_domain and worst <= 0x7FFFFFFF // 100
        return self.solver.check() == z3.unsat and native_positive_domain, worst

    @staticmethod
    def compute_context_damage_reference(
        context: DamageContext, rng_roll: int = 100, critical: bool = False,
    ) -> int:
        """Independent imperative C-operation-order reference for richer inputs."""
        c = context
        if not 85 <= rng_roll <= 100:
            raise ValueError("invalid RNG roll")
        if c.power == 0 and c.fixed_damage is None:
            return 0
        if any(n == 0 for n, _ in c.type_modifiers):
            return 0
        if c.defender_ability == ABILITY_LEVITATE and c.move_type == 4:
            return 0
        if ((c.defender_ability == ABILITY_VOLT_ABSORB and c.move_type == 13)
                or (c.defender_ability == ABILITY_WATER_ABSORB and c.move_type == 11)
                or (c.defender_ability == ABILITY_FLASH_FIRE and c.move_type == 10)):
            return 0
        numerator, denominator = 1, 1
        for n, d in c.type_modifiers:
            numerator *= n
            denominator *= d
        if c.defender_ability == ABILITY_WONDER_GUARD and numerator <= denominator:
            return 0
        if c.fixed_damage is not None:
            damage = c.fixed_damage
        else:
            physical = c.selected_category == "physical"
            attack, defense, power = c.attack, c.defense, c.power
            if physical and c.attacker_ability in (37, 74):
                attack = (attack * 2) & 0xFFFF
            if c.attacker_badge_boost:
                attack = (attack * 110 // 100) & 0xFFFF
            if c.defender_badge_boost:
                defense = (defense * 110 // 100) & 0xFFFF
            bonus = c.type_item_boost_percent
            if c.attacker_item in _TYPE_ITEMS and _TYPE_ITEMS[c.attacker_item][0] == c.move_type:
                bonus = _TYPE_ITEMS[c.attacker_item][1]
            if bonus:
                attack = (attack * (100 + bonus) // 100) & 0xFFFF
            if physical and c.attacker_item == 186:
                attack = (attack * 150 // 100) & 0xFFFF
            if not physical and not c.frontier_battle:
                if c.attacker_item == 191 and c.attacker_species in (407, 408):
                    attack = (attack * 150 // 100) & 0xFFFF
                if c.defender_item == 191 and c.defender_species in (407, 408):
                    defense = (defense * 150 // 100) & 0xFFFF
            if not physical and ((c.attacker_item == 192 and c.attacker_species == 373)
                                 or (c.attacker_item == 202 and c.attacker_species == 25)):
                attack = (attack * 2) & 0xFFFF
            if not physical and c.defender_item == 193 and c.defender_species == 373:
                defense = (defense * 2) & 0xFFFF
            if physical and c.defender_item == 223 and c.defender_species == 132:
                defense = (defense * 2) & 0xFFFF
            if physical and c.attacker_item == 224 and c.attacker_species in (104, 105):
                attack = (attack * 2) & 0xFFFF
            if not physical and c.defender_ability == 47 and c.move_type in (10, 15):
                attack //= 2
            if physical and c.attacker_ability == 55:
                attack = (attack * 150 // 100) & 0xFFFF
            if not physical and c.attacker_ability in (57, 58) and c.plus_minus_active:
                attack = (attack * 150 // 100) & 0xFFFF
            if physical and c.attacker_ability == 62 and c.attacker_status:
                attack = (attack * 150 // 100) & 0xFFFF
            if physical and c.defender_ability == 63 and c.defender_status:
                defense = (defense * 150 // 100) & 0xFFFF
            if (c.move_type == 13 and c.mud_sport) or (c.move_type == 10 and c.water_sport):
                power //= 2
            starter = {12: 65, 10: 66, 11: 67, 6: 68}
            if c.attacker_ability == starter.get(c.move_type, -1) and c.attacker_hp <= c.attacker_max_hp // 3:
                power = power * 150 // 100
            if physical and c.explosion:
                defense //= 2
            table = ((2, 8), (2, 7), (2, 6), (2, 5), (2, 4), (2, 3), (2, 2),
                     (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2))
            if not critical or c.attack_stage > 6:
                attack = attack * table[c.attack_stage][0] // table[c.attack_stage][1]
            if not critical or c.defense_stage < 6:
                defense = defense * table[c.defense_stage][0] // table[c.defense_stage][1]
            if defense == 0:
                raise ValueError("context would divide by zero in native damage calculation")
            damage = attack * power * (2 * c.level // 5 + 2) // defense // 50
            if physical and c.attacker_status & STATUS_BURN and c.attacker_ability != 62:
                damage //= 2
            screen = c.reflect if physical else c.light_screen
            if screen and not critical:
                damage = 2 * (damage // 3) if c.double_battle and c.defending_side_alive == 2 else damage // 2
            if c.double_battle and c.targets_both and c.defending_side_alive == 2:
                damage //= 2
            if physical:
                damage = max(1, damage)
            else:
                if not c.weather_suppressed:
                    if c.weather == "rain":
                        if c.move_type == 10:
                            damage //= 2
                        elif c.move_type == 11:
                            damage = 15 * damage // 10
                    if c.solar_beam and c.weather in ("rain", "sandstorm", "hail"):
                        damage //= 2
                    if c.weather == "sun":
                        if c.move_type == 10:
                            damage = 15 * damage // 10
                        elif c.move_type == 11:
                            damage //= 2
                if c.flash_fire_active and c.move_type == 10:
                    damage = 15 * damage // 10
            if c.move_type == 9:
                damage = 0
            damage = (damage + 2) * (2 if critical else 1) * c.damage_multiplier
            if c.charged_up and c.move_type == 13:
                damage *= 2
            if c.helping_hand:
                damage = damage * 15 // 10
            if c.stab:
                damage = damage * 15 // 10
            for n, d in c.type_modifiers:
                damage = max(1, damage * n // d)
            damage = max(1, damage * rng_roll // 100)
        if (c.false_swipe or c.endure) and c.substitute_hp == 0:
            assert c.defender_hp is not None
            damage = min(damage, c.defender_hp - 1)
        # Substitute absorbs a hit, not excess damage; each later multi-hit can
        # reach HP after it breaks. This is a conservative upper bound for the
        # remaining hits and does not manufacture a proof from a decoy.
        if c.substitute_hp > 0:
            return damage * max(0, c.hit_count - 1)
        return damage * c.hit_count

    def _context_expression(
        self, c: DamageContext, rng: z3.ArithRef, crit: z3.ArithRef,
    ) -> z3.ArithRef:
        """Build a separately implemented Z3 graph in the native operation order."""
        if c.fixed_damage is not None:
            damage = _zi(c.fixed_damage)
        elif c.power == 0:
            return _zi(0)
        else:
            physical = c.selected_category == "physical"
            attack, defense, power = _zi(c.attack), _zi(c.defense), _zi(c.power)
            if physical and c.attacker_ability in (ABILITY_HUGE_POWER, ABILITY_PURE_POWER):
                attack = attack * 2 % 65536
            if c.attacker_badge_boost:
                attack = attack * 110 / 100 % 65536
            if c.defender_badge_boost:
                defense = defense * 110 / 100 % 65536
            bonus = c.type_item_boost_percent
            item = _TYPE_ITEMS.get(c.attacker_item)
            if item is not None and item[0] == c.move_type:
                bonus = item[1]
            if bonus:
                attack = attack * (100 + bonus) / 100 % 65536
            if physical and c.attacker_item == 186:
                attack = attack * 150 / 100 % 65536
            if not physical and not c.frontier_battle:
                if c.attacker_item == 191 and c.attacker_species in (407, 408):
                    attack = attack * 150 / 100 % 65536
                if c.defender_item == 191 and c.defender_species in (407, 408):
                    defense = defense * 150 / 100 % 65536
            if not physical and ((c.attacker_item == 192 and c.attacker_species == 373)
                                 or (c.attacker_item == 202 and c.attacker_species == 25)):
                attack = attack * 2 % 65536
            if not physical and c.defender_item == 193 and c.defender_species == 373:
                defense = defense * 2 % 65536
            if physical and c.defender_item == 223 and c.defender_species == 132:
                defense = defense * 2 % 65536
            if physical and c.attacker_item == 224 and c.attacker_species in (104, 105):
                attack = attack * 2 % 65536
            if not physical and c.defender_ability == ABILITY_THICK_FAT and c.move_type in (10, 15):
                attack = attack / 2
            if physical and c.attacker_ability == ABILITY_HUSTLE:
                attack = attack * 150 / 100 % 65536
            if not physical and c.attacker_ability in (ABILITY_PLUS, ABILITY_MINUS) and c.plus_minus_active:
                attack = attack * 150 / 100 % 65536
            if physical and c.attacker_ability == ABILITY_GUTS and c.attacker_status:
                attack = attack * 150 / 100 % 65536
            if physical and c.defender_ability == ABILITY_MARVEL_SCALE and c.defender_status:
                defense = defense * 150 / 100 % 65536
            if (c.move_type == 13 and c.mud_sport) or (c.move_type == 10 and c.water_sport):
                power = power / 2
            boosts = ((12, ABILITY_OVERGROW), (10, ABILITY_BLAZE), (11, ABILITY_TORRENT), (6, ABILITY_SWARM))
            if (c.move_type, c.attacker_ability) in boosts and c.attacker_hp <= c.attacker_max_hp // 3:
                power = power * 150 / 100
            if physical and c.explosion:
                defense = defense / 2
            an, ad = STAGE_RATIOS[c.attack_stage]
            dn, dd = STAGE_RATIOS[c.defense_stage]
            attack = _zx(z3.And(crit == 2, c.attack_stage <= 6), attack, attack * an / ad)
            defense = _zx(z3.And(crit == 2, c.defense_stage >= 6), defense, defense * dn / dd)
            denominator = z3.simplify(defense)
            if z3.is_int_value(denominator) and denominator.as_long() == 0:
                raise ValueError("native damage denominator would be zero")
            damage = attack * power * (_zi(2 * c.level) / 5 + 2) / defense / 50
            if physical and c.attacker_status & STATUS_BURN and c.attacker_ability != ABILITY_GUTS:
                damage = damage / 2
            screen = c.reflect if physical else c.light_screen
            if screen:
                reduced = 2 * (damage / 3) if c.double_battle and c.defending_side_alive == 2 else damage / 2
                damage = _zx(crit == 1, reduced, damage)
            if c.double_battle and c.targets_both and c.defending_side_alive == 2:
                damage = damage / 2
            if physical:
                damage = _nonzero(damage)
            else:
                if not c.weather_suppressed:
                    if c.weather == "rain" and c.move_type in (10, 11):
                        damage = damage / 2 if c.move_type == 10 else damage * 15 / 10
                    if c.solar_beam and c.weather in ("rain", "sandstorm", "hail"):
                        damage = damage / 2
                    if c.weather == "sun" and c.move_type in (10, 11):
                        damage = damage * 15 / 10 if c.move_type == 10 else damage / 2
                if c.flash_fire_active and c.move_type == 10:
                    damage = damage * 15 / 10
            if c.move_type == 9:
                damage = _zi(0)
            damage = (damage + 2) * crit * c.damage_multiplier
            if c.charged_up and c.move_type == 13:
                damage = damage * 2
            if c.helping_hand:
                damage = damage * 15 / 10
            if c.stab:
                damage = damage * 15 / 10
            for n, d in c.type_modifiers:
                damage = _zi(0) if n == 0 else _nonzero(damage * n / d)
            damage = _zx(damage == 0, _zi(0), _nonzero(damage * rng / 100))
        if any(n == 0 for n, _ in c.type_modifiers):
            damage = _zi(0)
        if ((c.defender_ability == ABILITY_LEVITATE and c.move_type == 4)
            or (c.defender_ability == ABILITY_VOLT_ABSORB and c.move_type == 13)
            or (c.defender_ability == ABILITY_WATER_ABSORB and c.move_type == 11)
            or (c.defender_ability == ABILITY_FLASH_FIRE and c.move_type == 10)):
            damage = _zi(0)
        numerator = denominator_int = 1
        for n, d in c.type_modifiers:
            numerator *= n
            denominator_int *= d
        if c.defender_ability == ABILITY_WONDER_GUARD and numerator <= denominator_int:
            damage = _zi(0)
        if (c.endure or c.false_swipe) and c.substitute_hp == 0:
            assert c.defender_hp is not None
            cap = _zi(c.defender_hp - 1)
            damage = _zx(damage > cap, cap, damage)
        return damage * (max(0, c.hit_count - 1) if c.substitute_hp > 0 else c.hit_count)

    def compute_context_damage_symbolic(
        self, context: DamageContext, rng_roll: int = 100, critical: bool = False,
    ) -> int:
        """Evaluate the contextual integer graph for deterministic parity tests."""
        if not 85 <= rng_roll <= 100:
            raise ValueError("invalid RNG roll")
        expression = self._context_expression(context, _zi(rng_roll), _zi(2 if critical else 1))
        return int(z3.simplify(expression).as_long())

    def verify_context_survival(
        self, current_hp: int, context: DamageContext, can_crit: bool = True,
        residual_damage: int = 0,
    ) -> DamageResult:
        """Prove that no permitted roll/critical causes HP<=0 before next turn."""
        if not 0 <= current_hp <= 65535 or residual_damage < 0:
            raise ValueError("HP must be uint16 and residual damage nonnegative")
        if context.defender_ability in (ABILITY_BATTLE_ARMOR, ABILITY_SHELL_ARMOR):
            can_crit = False
        worst = self.compute_context_damage_symbolic(context)
        if can_crit:
            worst = max(worst, self.compute_context_damage_symbolic(context, critical=True))
        total = worst + residual_damage
        if (context.false_swipe or context.endure) and context.defender_hp != current_hp:
            return DamageResult(False, worst, residual_damage, total, "unsupported",
                                "damage cap HP does not match the current HP observation")
        # Integer sorts model the positive native arithmetic domain. Reject an
        # overflowing/hacked state rather than silently equating unbounded Z3
        # integers with signed ARM32 wrap.32 is a conservative envelope for all
        # supported raw-stat boosts plus the maximum stage multiplier.
        numerator_bound = context.attack * context.power * (2 * context.level // 5 + 2) * 32 * 3 // 2
        if context.fixed_damage is None and (numerator_bound > 0x7FFFFFFF or worst > 0x7FFFFFFF // 100):
            return DamageResult(False, worst, residual_damage, total, "unsupported",
                                "context exceeds the proved positive ARM32 arithmetic domain")
        if context.unsupported_effects or context.move_type == 9:
            reason = ", ".join(context.unsupported_effects) or "reserved native type is not a legal damage event"
            return DamageResult(False, worst, residual_damage, total, "unsupported", reason)
        self.solver.reset()
        rng, critical, damage = z3.Int("rng_roll"), z3.Int("crit_mult"), z3.Int("damage")
        self.solver.add(rng >= 85, rng <= 100)
        self.solver.add(z3.Or(critical == 1, critical == 2) if can_crit else critical == 1)
        expression = self._context_expression(context, rng, critical)
        self.solver.add(damage == expression, damage + residual_damage >= current_hp)
        status = self.solver.check()
        if status == z3.unsat:
            return DamageResult(True, worst, residual_damage, total, "proved", "no permitted damage outcome reaches current HP")
        if status == z3.sat:
            model = self.solver.model()
            roll = int(model.eval(rng, model_completion=True).as_long())
            crit = int(model.eval(critical, model_completion=True).as_long()) == 2
            return DamageResult(False, worst, residual_damage, total, "counterexample", "a permitted outcome causes fainting", roll, crit)
        return DamageResult(False, worst, residual_damage, total, "unknown", self.solver.reason_unknown())
