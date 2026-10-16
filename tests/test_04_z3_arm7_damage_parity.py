"""100,000 Z3 vectors versus independently compiled ROM-matching C source.

The C oracle compiles CalculateBaseDamage verbatim from the pinned pret checkout,
not an algorithm copied from the Python implementation. It is source parity,
not a claim that a live mGBA battle has been certified. Context fixtures exercise
additional native modifiers and deliberately catch errors in the PDF formula.
"""

from __future__ import annotations

import ctypes
from dataclasses import replace
from pathlib import Path
import random
import re
import shutil
import subprocess
from collections.abc import Callable
from typing import cast

import pytest
import z3

from core.safety import (
    DamageContext, GbaIntegerSafetyVerifier, ResidualContext, compute_residual_damage,
)

ROOT = Path(__file__).resolve().parents[1]
PINNED_EMERALD = "731ad5bfd6e6f265508d0efcca0ba42f9dcf5881"
PINNED_FIRERED = "037335f4c725d7c9aecdac87066f2002b4bd7e14"

_C_PREFIX = r"""
#include <stdint.h>
#include <string.h>
typedef uint8_t u8; typedef uint16_t u16; typedef uint32_t u32;
typedef int32_t s32; typedef uint8_t bool8;
#include "constants/pokemon.h"
#include "constants/species.h"
#include "constants/abilities.h"
#include "constants/moves.h"
#include "constants/items.h"
#include "constants/hold_effects.h"
#include "constants/battle.h"
#include "constants/battle_move_effects.h"
#include "constants/flags.h"
#define ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))
#define IS_TYPE_PHYSICAL(t) ((t) < TYPE_MYSTERY)
#define IS_TYPE_SPECIAL(t) ((t) > TYPE_MYSTERY)
#define DYNAMIC_TYPE_MASK 63
#define MOVE_TARGET_BOTH 8
#define RESOURCE_FLAG_FLASH_FIRE 1
#define BATTLE_ALIVE_DEF_SIDE 2
#define ABILITYEFFECT_FIELD_SPORT 5
#define ABILITYEFFECT_MUD_SPORT 253
#define ABILITYEFFECT_WATER_SPORT 254
#define ABILITY_ON_FIELD2(a) oraclePlusMinus
#define WEATHER_HAS_EFFECT2 (!oracleWeatherSuppressed)
#define APPLY_STAT_MOD(var, mon, stat, statIndex) \
    { (var) = (stat) * gStatStageRatios[(mon)->statStages[(statIndex)]][0]; \
      (var) /= gStatStageRatios[(mon)->statStages[(statIndex)]][1]; }
struct BattlePokemon {
    u16 species, attack, defense, spAttack, spDefense, hp, maxHP, item;
    u8 level, ability, statStages[8]; u32 status1;
};
struct MoveData { u16 power; u8 type, effect, target; };
struct BerryData { u8 holdEffect, holdEffectParam; };
struct ResourceFlags { u8 flags[4]; };
struct Resources { struct ResourceFlags *flags; };
static struct MoveData gBattleMoves[400];
static struct BerryData gEnigmaBerries[4];
static struct ResourceFlags oracleFlags;
static struct Resources oracleResources = { &oracleFlags };
static struct Resources *gBattleResources = &oracleResources;
static u16 gCurrentMove, gBattleMovePower, gBattleWeather;
static u32 gBattleTypeFlags;
static u8 gCritMultiplier;
static int oracleAlive, oracleWeatherSuppressed, oraclePlusMinus;
static int oracleMud, oracleWater, oracleBadgeAttack, oracleBadgeDefense;
static u8 CountAliveMonsInBattle(u8 ignored) { (void)ignored; return oracleAlive; }
static int AbilityBattleEffects(int mode,int battler,int ability,int effect,int arg) {
    (void)mode; (void)battler; (void)ability; (void)arg;
    return effect == ABILITYEFFECT_MUD_SPORT ? oracleMud : oracleWater;
}
static bool8 ShouldGetStatBadgeBoost(u16 flagId, u8 battler) {
    (void)flagId; return battler == 0 ? oracleBadgeAttack : oracleBadgeDefense;
}
static u8 GetItemHoldEffect(u16 item) {
    switch(item) {
    case ITEM_CHOICE_BAND: return HOLD_EFFECT_CHOICE_BAND;
    case ITEM_SOUL_DEW: return HOLD_EFFECT_SOUL_DEW;
    case ITEM_DEEP_SEA_TOOTH: return HOLD_EFFECT_DEEP_SEA_TOOTH;
    case ITEM_DEEP_SEA_SCALE: return HOLD_EFFECT_DEEP_SEA_SCALE;
    case ITEM_LIGHT_BALL: return HOLD_EFFECT_LIGHT_BALL;
    case ITEM_METAL_POWDER: return HOLD_EFFECT_METAL_POWDER;
    case ITEM_THICK_CLUB: return HOLD_EFFECT_THICK_CLUB;
    case ITEM_SILVER_POWDER: return HOLD_EFFECT_BUG_POWER;
    case ITEM_METAL_COAT: return HOLD_EFFECT_STEEL_POWER;
    case ITEM_SOFT_SAND: return HOLD_EFFECT_GROUND_POWER;
    case ITEM_HARD_STONE: return HOLD_EFFECT_ROCK_POWER;
    case ITEM_MIRACLE_SEED: return HOLD_EFFECT_GRASS_POWER;
    case ITEM_BLACK_GLASSES: return HOLD_EFFECT_DARK_POWER;
    case ITEM_BLACK_BELT: return HOLD_EFFECT_FIGHTING_POWER;
    case ITEM_MAGNET: return HOLD_EFFECT_ELECTRIC_POWER;
    case ITEM_MYSTIC_WATER: return HOLD_EFFECT_WATER_POWER;
    case ITEM_SHARP_BEAK: return HOLD_EFFECT_FLYING_POWER;
    case ITEM_POISON_BARB: return HOLD_EFFECT_POISON_POWER;
    case ITEM_NEVER_MELT_ICE: return HOLD_EFFECT_ICE_POWER;
    case ITEM_SPELL_TAG: return HOLD_EFFECT_GHOST_POWER;
    case ITEM_TWISTED_SPOON: return HOLD_EFFECT_PSYCHIC_POWER;
    case ITEM_CHARCOAL: return HOLD_EFFECT_FIRE_POWER;
    case ITEM_DRAGON_FANG: return HOLD_EFFECT_DRAGON_POWER;
    case ITEM_SILK_SCARF: return HOLD_EFFECT_NORMAL_POWER;
    default: return HOLD_EFFECT_NONE;
    }
}
static u8 GetItemHoldEffectParam(u16 item) {
    const u8 effect = GetItemHoldEffect(item);
    return ((effect >= 46 && effect <= 60) || effect == 31 || effect == 42) ? 10 : 0;
}
#define ItemId_GetHoldEffect GetItemHoldEffect
#define ItemId_GetHoldEffectParam GetItemHoldEffectParam
"""

_C_SUFFIX = r"""
#ifdef _WIN32
__declspec(dllexport)
#endif
int ares_oracle(const int *v) {
    struct BattlePokemon attacker = {0}, defender = {0};
    attacker.level=v[0]; attacker.attack=attacker.spAttack=v[1];
    defender.defense=defender.spDefense=v[2];
    attacker.statStages[STAT_ATK]=attacker.statStages[STAT_SPATK]=v[5];
    defender.statStages[STAT_DEF]=defender.statStages[STAT_SPDEF]=v[6];
    gCritMultiplier=v[7] ? 2 : 1;
    attacker.ability=v[14]; defender.ability=v[15];
    attacker.item=v[16]; defender.item=v[17];
    attacker.species=v[18]; defender.species=v[19];
    attacker.status1=v[20]; defender.status1=v[21];
    attacker.hp=v[22]; attacker.maxHP=v[23];
    oracleBadgeAttack=v[24]; oracleBadgeDefense=v[25];
    gBattleWeather=v[26]; oracleWeatherSuppressed=v[27];
    u16 side = (v[28] ? SIDE_STATUS_REFLECT : 0) | (v[29] ? SIDE_STATUS_LIGHTSCREEN : 0);
    gBattleTypeFlags=(v[30] ? BATTLE_TYPE_DOUBLE : 0) | (v[41] ? BATTLE_TYPE_BATTLE_TOWER : 0);
    oracleAlive=v[31]; gCurrentMove = v[34] ? MOVE_SOLAR_BEAM : MOVE_POUND;
    gBattleMoves[gCurrentMove].power=v[3]; gBattleMoves[gCurrentMove].type=v[4];
    gBattleMoves[gCurrentMove].effect=v[33] ? EFFECT_EXPLOSION : EFFECT_HIT;
    gBattleMoves[gCurrentMove].target=v[32] ? MOVE_TARGET_BOTH : 0;
    oracleFlags.flags[0]=v[35] ? RESOURCE_FLAG_FLASH_FIRE : 0;
    oracleMud=v[38]; oracleWater=v[39]; oraclePlusMinus=v[40];
    int damage = CalculateBaseDamage(&attacker, &defender, gCurrentMove,side,0,0,0,1);
    /* Cmd_damagecalc operations, retaining the exact source order. */
    damage = damage * gCritMultiplier * v[42];
    if(v[36] && v[4]==TYPE_ELECTRIC) damage *= 2;
    if(v[37]) damage = damage * 15 / 10;
    /* Cmd_typecalc + ModulateDmgByType, applying each type separately. */
    if(v[9]) damage = damage * 15 / 10;
    for(int i=10; i<=12; i+=2) {
        if(i==12 && v[13]==0) break;
        damage = damage * v[i] / v[i+1];
        if(damage==0 && v[i]!=0) damage=1;
    }
    /* The native DOESNT_AFFECT_FOE flag prevents the HP update even if a
       later table entry makes the working damage register nonzero again. */
    if(v[10]==0 || (v[13]!=0 && v[12]==0)) damage=0;
    if((v[15]==ABILITY_LEVITATE && v[4]==TYPE_GROUND) ||
       (v[15]==ABILITY_VOLT_ABSORB && v[4]==TYPE_ELECTRIC) ||
       (v[15]==ABILITY_WATER_ABSORB && v[4]==TYPE_WATER) ||
       (v[15]==ABILITY_FLASH_FIRE && v[4]==TYPE_FIRE)) damage=0;
    if(v[15]==ABILITY_WONDER_GUARD && v[10]*(v[13] ? v[12] : 1) <= v[11]*(v[13] ? v[13] : 1)) damage=0;
    /* ApplyRandomDmgMultiplier; the RNG is injected, not reimplemented. */
    if(damage != 0) { damage *= v[8]; damage /= 100; if(damage==0) damage=1; }
    return v[3]==0 ? 0 : damage;
}
"""


def _extract_braced(source: str, declaration: str) -> str:
    start = source.index(declaration)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    if source[end:end + 1] == ";":
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def native_c_oracle(tmp_path_factory: pytest.TempPathFactory) -> Callable[[DamageContext, int, bool], int]:
    """Build the untouched native base routine using a narrow C state harness."""
    compiler = shutil.which("gcc")
    if compiler is None:
        raise RuntimeError("GCC is required for independent native-source damage parity")
    work = tmp_path_factory.mktemp("native_damage_oracle")
    native_functions: list[Callable[[ctypes.Array[ctypes.c_int]], int]] = []
    for game, pin in (("pokeemerald", PINNED_EMERALD), ("pokefirered", PINNED_FIRERED)):
        vendor = ROOT / "native" / "vendor" / game
        commit = subprocess.run(["git", "-C", str(vendor), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        assert commit == pin, "Damage oracle must use the recorded pinned source"
        source = (vendor / "src" / "pokemon.c").read_text(encoding="utf-8")
        portions = (
            _extract_braced(source, "const u8 gStatStageRatios["),
            _extract_braced(source, "static const u8 sHoldEffectToType["),
            _extract_braced(source, "s32 CalculateBaseDamage("),
        )
        c_file = work / f"{game}_oracle.c"
        c_file.write_text(_C_PREFIX + "\n" + "\n".join(portions) + "\n" + _C_SUFFIX, encoding="utf-8")
        dll = work / f"{game}_oracle.dll"
        subprocess.run([compiler, "-O2", "-shared", "-std=c99", "-I", str(vendor / "include"),
                        str(c_file), "-o", str(dll)], check=True, capture_output=True, text=True)
        library = ctypes.CDLL(str(dll))
        native = library.ares_oracle
        native.argtypes = [ctypes.POINTER(ctypes.c_int)]
        native.restype = ctypes.c_int
        native_functions.append(cast(Callable[[ctypes.Array[ctypes.c_int]], int], native))

    def call(context: DamageContext, roll: int, critical: bool) -> int:
        c = context
        n1, d1 = c.type_modifiers[0]
        n2, d2 = c.type_modifiers[1] if len(c.type_modifiers) == 2 else (0, 0)
        weather = {"none": 0, "rain": 1, "sun": 32, "sandstorm": 8, "hail": 128}[c.weather]
        values = (
            c.level, c.attack, c.defense, c.power, c.move_type, c.attack_stage,
            c.defense_stage, int(critical), roll, int(c.stab), n1, d1, n2, d2,
            c.attacker_ability, c.defender_ability, c.attacker_item, c.defender_item,
            c.attacker_species, c.defender_species, c.attacker_status, c.defender_status,
            c.attacker_hp, c.attacker_max_hp, int(c.attacker_badge_boost),
            int(c.defender_badge_boost), weather, int(c.weather_suppressed),
            int(c.reflect), int(c.light_screen), int(c.double_battle), c.defending_side_alive,
            int(c.targets_both), int(c.explosion), int(c.solar_beam),
            int(c.flash_fire_active), int(c.charged_up), int(c.helping_hand),
            int(c.mud_sport), int(c.water_sport), int(c.plus_minus_active),
            int(c.frontier_battle), c.damage_multiplier,
        )
        buffer = (ctypes.c_int * len(values))(*values)
        emerald, firered = (function(buffer) for function in native_functions)
        assert emerald == firered, (context, roll, critical, emerald, firered)
        return emerald

    return call


def test_100000_seeded_z3_vectors_against_native_c_and_independent_python(
    native_c_oracle: Callable[[DamageContext, int, bool], int],
) -> None:
    verifier = GbaIntegerSafetyVerifier()
    generator = random.Random(42)
    multipliers = ((0, 1), (1, 4), (1, 2), (1, 1), (2, 1), (4, 1))
    for index in range(100_000):
        level = generator.randint(1, 100)
        attack = generator.randint(5, 999)
        defense = generator.randint(5, 999)
        power = generator.randint(0, 250)
        astage = generator.randint(0, 12)
        dstage = generator.randint(0, 12)
        roll = generator.randint(85, 100)
        critical = bool(generator.getrandbits(1))
        stab = bool(generator.getrandbits(1))
        numerator, denominator = generator.choice(multipliers)
        context = DamageContext(level, attack, defense, power, stab=stab,
                                attack_stage=astage, defense_stage=dstage,
                                type_modifiers=((numerator, denominator),))
        symbolic = verifier.compute_exact_arm7_damage_symbolic(
            defense, level, attack, power, numerator, denominator, stab,
            critical, astage, dstage, roll)
        reference = verifier.compute_exact_arm7_damage_reference(
            defense, level, attack, power, numerator, denominator, stab,
            critical, astage, dstage, roll)
        native = native_c_oracle(context, roll, critical)
        assert symbolic == reference == native, (index, context, roll, critical, symbolic, reference, native)


def test_native_counterexamples_to_pdf_formula(
    native_c_oracle: Callable[[DamageContext, int, bool], int],
) -> None:
    verifier = GbaIntegerSafetyVerifier()
    physical = DamageContext(1, 5, 999, 1)
    special = replace(physical, move_type=10)
    assert native_c_oracle(physical, 100, False) == 3
    assert native_c_oracle(special, 100, False) == 2
    assert verifier.compute_context_damage_symbolic(physical) == 3
    assert verifier.compute_context_damage_symbolic(special) == 2
    # Physical min occurs before critical and STAB, not as a final floor alone.
    assert native_c_oracle(replace(physical, stab=True), 100, True) == 9
    # Each resisted type clamps independently, unlike one aggregated1/4.
    dual = DamageContext(1, 50, 100, 50, move_type=10, type_modifiers=((1, 2), (2, 1)))
    assert native_c_oracle(dual, 100, False) == 2
    assert native_c_oracle(replace(dual, type_modifiers=((2, 1), (1, 2))), 100, False) == 3
    critical = DamageContext(50, 100, 100, 80, attack_stage=0, defense_stage=12)
    assert native_c_oracle(critical, 100, False) == 4
    assert native_c_oracle(critical, 100, True) == 74
    assert verifier.verify_worst_case_survival(74, 100, 50, 100, 80, 1, 1,
                                              atk_stage=0, def_stage=12) == (False, 74)
    assert verifier.verify_worst_case_survival(75, 100, 50, 100, 80, 1, 1,
                                              atk_stage=0, def_stage=12) == (True, 74)


def test_contextual_native_modifiers_and_stage_matrix(
    native_c_oracle: Callable[[DamageContext, int, bool], int],
) -> None:
    verifier = GbaIntegerSafetyVerifier()
    base = DamageContext(53, 123, 89, 95, stab=True)
    fixtures = [
        base, replace(base, attacker_status=0x10),
        replace(base, attacker_status=0x10, attacker_ability=62),
        replace(base, attacker_ability=37), replace(base, attacker_ability=55),
        replace(base, defender_status=0x08, defender_ability=63),
        replace(base, reflect=True, double_battle=True, defending_side_alive=2),
        replace(base, double_battle=True, targets_both=True, defending_side_alive=2),
        replace(base, explosion=True), replace(base, helping_hand=True),
        replace(base, attacker_item=186), replace(base, attacker_item=217),
        replace(base, attacker_item=224, attacker_species=104),
        replace(base, defender_item=223, defender_species=132),
        replace(base, attacker_badge_boost=True, defender_badge_boost=True),
        replace(base, move_type=10, weather="rain"),
        replace(base, move_type=10, weather="sun", defender_ability=47),
        replace(base, move_type=10, attacker_ability=66, attacker_hp=20, water_sport=True),
        replace(base, move_type=11, weather="rain", light_screen=True),
        replace(base, move_type=12, solar_beam=True, weather="sandstorm"),
        replace(base, move_type=10, flash_fire_active=True),
        replace(base, move_type=13, charged_up=True, mud_sport=True),
        replace(base, move_type=13, attacker_item=202, attacker_species=25),
        replace(base, move_type=11, attacker_item=192, attacker_species=373),
        replace(base, move_type=11, defender_item=193, defender_species=373),
        replace(base, move_type=14, attacker_item=191, attacker_species=407),
        replace(base, move_type=14, defender_item=191, defender_species=408),
        replace(base, move_type=13, attacker_ability=57, plus_minus_active=True),
        replace(base, move_type=13, defender_ability=10),
        replace(base, move_type=4, defender_ability=26),
        replace(base, type_modifiers=((1, 2), (1, 2))),
        replace(base, type_modifiers=((0, 1), (2, 1))),
        replace(base, type_modifiers=((2, 1),), defender_ability=25),
        replace(base, type_modifiers=((1, 1),), defender_ability=25),
    ]
    for context in fixtures:
        for roll in (85, 92, 100):
            for critical in (False, True):
                expected = native_c_oracle(context, roll, critical)
                assert verifier.compute_context_damage_reference(context, roll, critical) == expected, context
                assert verifier.compute_context_damage_symbolic(context, roll, critical) == expected, context
    for attack_stage in range(13):
        for defense_stage in range(13):
            context = replace(base, attack_stage=attack_stage, defense_stage=defense_stage)
            for critical in (False, True):
                expected = native_c_oracle(context, 100, critical)
                assert verifier.compute_context_damage_symbolic(context, critical=critical) == expected


def test_contextual_proof_accounts_for_residual_and_unknown_effects() -> None:
    verifier = GbaIntegerSafetyVerifier()
    context = DamageContext(10, 20, 40, 40)
    worst = verifier.compute_context_damage_reference(context, critical=True)
    safe = verifier.verify_context_survival(worst + 1, context)
    assert safe.formally_safe and safe.proof_status == "proved"
    unsafe = verifier.verify_context_survival(worst + 1, context, residual_damage=1)
    assert not unsafe.formally_safe and unsafe.proof_status == "counterexample"
    assert unsafe.counterexample_rng is not None
    unknown = verifier.verify_context_survival(1000, replace(context, unsupported_effects=("unknown enemy move",)))
    assert not unknown.formally_safe and unknown.proof_status == "unsupported"
    # Armor's guaranteed critical protection is respected, Focus Band RNG is not.
    armor = verifier.verify_context_survival(worst, replace(context, defender_ability=4))
    assert armor.formally_safe
    assert compute_residual_damage(ResidualContext(160, status=0x80, toxic_turn=3, sandstorm=True)) == 40
    assert compute_residual_damage(ResidualContext(160, status=0x10, sandstorm=True, types=(8,), curse=True)) == 60
    assert compute_residual_damage(ResidualContext(15, status=0x08, hail=True, leech_seed=True)) == 3


def test_immunity_minimum_fixed_damage_substitute_and_endure() -> None:
    verifier = GbaIntegerSafetyVerifier()
    immune = DamageContext(100, 500, 5, 250, type_modifiers=((0, 1),))
    assert verifier.verify_context_survival(1, immune).formally_safe
    assert verifier.compute_context_damage_symbolic(immune) == 0
    fixed = replace(immune, type_modifiers=((1, 1),), fixed_damage=20)
    assert verifier.compute_context_damage_symbolic(fixed, 85, True) == 20
    assert not verifier.verify_context_survival(20, fixed).formally_safe
    endure = replace(fixed, endure=True, defender_hp=20)
    assert verifier.verify_context_survival(20, endure).formally_safe
    assert not verifier.verify_context_survival(20, endure, residual_damage=1).formally_safe
    assert verifier.verify_context_survival(100, endure).proof_status == "unsupported"
    substitute = replace(fixed, substitute_hp=10)
    assert verifier.verify_context_survival(1, substitute).formally_safe
    assert not verifier.verify_context_survival(20, replace(substitute, hit_count=2)).formally_safe
    assert verifier.compute_context_damage_symbolic(DamageContext(10, 10, 10, 0)) == 0
    assert verifier.verify_worst_case_survival(1, 1, 1, 1, 1, 1, 1, def_stage=0)[0] is False
    overflow = DamageContext(100, 65535, 65535, 65535)
    assert verifier.verify_context_survival(65535, overflow).proof_status == "unsupported"


def test_solver_unknown_is_never_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    verifier = GbaIntegerSafetyVerifier()
    monkeypatch.setattr(verifier.solver, "check", lambda: z3.unknown)
    monkeypatch.setattr(verifier.solver, "reason_unknown", lambda: "forced regression timeout")
    context = DamageContext(10, 20, 40, 40)
    result = verifier.verify_context_survival(1000, context)
    assert not result.formally_safe and result.proof_status == "unknown"
    assert verifier.verify_worst_case_survival(1000, 40, 10, 20, 40, 1, 1)[0] is False


@pytest.mark.parametrize("builder", [
    lambda: DamageContext(0, 20, 40, 40),
    lambda: DamageContext(10, 20, 0, 40),
    lambda: DamageContext(10, 20, 40, 40, attack_stage=13),
    lambda: DamageContext(10, 20, 40, 40, type_modifiers=((1, 0),)),
    lambda: DamageContext(10, 20, 40, 40, false_swipe=True),
])
def test_invalid_context_rejected(builder: Callable[[], DamageContext]) -> None:
    with pytest.raises((ValueError, TypeError)):
        builder()


def test_symbolic_model_uses_only_integer_sorts() -> None:
    verifier = GbaIntegerSafetyVerifier()
    source = (ROOT / "core" / "safety" / "z3_integer_verifier.py").read_text(encoding="utf-8")
    assert re.search(r"\bz3\.Real(?:Val)?\(", source) is None
    assert verifier.verify_worst_case_survival(1000, 100, 50, 100, 80, 1, 1)[0]
    assert all(symbol.sort().kind() == z3.Z3_INT_SORT for symbol in verifier._basic_symbols)


def test_gen3_type_category_split() -> None:
    # Physical types (0..8) and reserve 9
    for move_type in range(10):
        ctx = DamageContext(10, 20, 40, 40, move_type=move_type, category="physical")
        assert ctx.selected_category == "physical"
        with pytest.raises(ValueError, match="category must follow the Gen-3 type category split"):
            DamageContext(10, 20, 40, 40, move_type=move_type, category="special")

    # Special types (10..17, e.g. Grass=12, Fire=10, Water=11)
    for move_type in range(10, 18):
        ctx = DamageContext(10, 20, 40, 40, move_type=move_type, category="special")
        assert ctx.selected_category == "special"
        with pytest.raises(ValueError, match="category must follow the Gen-3 type category split"):
            DamageContext(10, 20, 40, 40, move_type=move_type, category="physical")

    # Status moves with power 0 and special type (like Leech Seed move_type=12)
    leech_seed_ctx = DamageContext(7, 15, 15, 0, move_type=12, category="special")
    assert leech_seed_ctx.selected_category == "special"

