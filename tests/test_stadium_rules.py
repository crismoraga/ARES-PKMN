"""Gen-3 rules of the stadium engine that once behaved wrongly (one regression each)."""
from __future__ import annotations

from core.stadium.dex import load_dex
from core.stadium.engine import MOVE_STRUGGLE, Battle, Event, MonSpec
from core.stadium.teams import gen3_stats

DEX = load_dex()
TACKLE, GROWL, SPLASH, THRASH, SEISMIC_TOSS, EARTHQUAKE, DIG = 33, 45, 150, 37, 69, 89, 91
ROLLOUT, PURSUIT, FAKE_OUT, FOCUS_PUNCH, SWIFT, DEFENSE_CURL = 205, 228, 252, 264, 129, 111


def sid(name: str) -> int:
    return next(data.species_id for data in DEX.species if data.name == name)


def mon(name: str, moves: tuple[int, ...], level: int = 50, ability: int | None = None) -> MonSpec:
    species = sid(name)
    stats = gen3_stats(DEX, species, level, (31,) * 6, (0,) * 6, 0)
    return MonSpec(species, level, stats, moves, DEX.species[species].abilities[0] if ability is None else ability)


def kinds(events: list[Event]) -> list[str]:
    return [str(event[0]) for event in events]


def test_focus_punch_fails_after_being_hit_and_works_untouched() -> None:
    hit = Battle(DEX, [mon("MACHAMP", (FOCUS_PUNCH,))], [mon("PIDGEOT", (SWIFT,))], seed=1)
    log = hit.submit({0: 0, 1: 0})
    assert ("fail", 0, FOCUS_PUNCH) in log
    calm = Battle(DEX, [mon("MACHAMP", (FOCUS_PUNCH,))], [mon("PIDGEOT", (SPLASH,))], seed=1)
    log = calm.submit({0: 0, 1: 0})
    assert any(event[0] == "hit" and event[1] == 1 for event in log)


def test_fake_out_only_works_on_the_first_turn() -> None:
    battle = Battle(DEX, [mon("PERSIAN", (FAKE_OUT,))], [mon("SNORLAX", (SPLASH,))], seed=2)
    first = battle.submit({0: 0, 1: 0})
    assert "hit" in kinds(first) and "flinched" in kinds(first)
    second = battle.submit({0: 0, 1: 0})
    assert ("fail", 0, FAKE_OUT) in second and "hit" not in kinds(second)


def test_struggle_when_the_only_move_cannot_be_selected() -> None:
    battle = Battle(DEX, [mon("PIDGEOT", (GROWL,))], [mon("SNORLAX", (SPLASH,))], seed=3)
    battle.sides[0].fighter.taunt = 3
    assert battle.legal_actions(0) == [0]
    log = battle.submit({0: 0, 1: 0})
    assert any(event[0] == "move" and event[1] == 0 and event[3] == MOVE_STRUGGLE for event in log)
    assert battle.sides[0].fighter.pp == [40]  # Growl's PP were not spent


def test_rollout_locks_doubles_and_ends_after_five_hits() -> None:
    for seed in range(40):  # Rollout is 90% accurate: find a run where all five connect
        battle = Battle(DEX, [mon("GOLEM", (ROLLOUT, TACKLE), level=12)], [mon("SNORLAX", (SPLASH,), level=100)],
                        seed=seed)
        damage: list[int] = []
        for turn in range(5):
            log = battle.submit({0: 0, 1: 0})
            hits = [int(event[2]) for event in log if event[0] == "hit" and event[1] == 1]
            if not hits:
                break
            damage.append(hits[0])
            if turn < 4:
                assert battle.legal_actions(0) == [0]  # locked into Rollout
        if len(damage) == 5:
            assert damage[2] > damage[0] and damage[4] > damage[0] * 6  # 30, 60, 120, 240, 480 power
            assert battle.legal_actions(0) == [0, 1]   # free again
            assert battle.sides[0].fighter.pp[0] == 19  # one PP for the whole sequence
            return
    raise AssertionError("no seed landed five Rollout hits")


def test_a_rampage_ends_when_the_move_cannot_connect() -> None:
    battle = Battle(DEX, [mon("TAUROS", (THRASH, TACKLE))], [mon("GENGAR", (SPLASH,))], seed=4)
    log = battle.submit({0: 0, 1: 0})
    assert "noeffect" in kinds(log)
    assert battle.legal_actions(0) == [0, 1]  # not locked into a move that does nothing


def test_pursuit_strikes_a_switching_target_first_with_double_power() -> None:
    teams = ([mon("HOUNDOOM", (PURSUIT,))], [mon("SNORLAX", (SPLASH,)), mon("PIDGEOT", (SPLASH,))])
    normal: list[int] = []
    chased: list[int] = []
    for seed in range(30):
        stay = Battle(DEX, teams[0], teams[1], seed=seed)
        normal += [int(event[2]) for event in stay.submit({0: 0, 1: 0}) if event[0] == "hit" and not event[4]]
        flee = Battle(DEX, teams[0], teams[1], seed=seed)
        log = flee.submit({0: 0, 1: 4})
        order = kinds(log)
        assert order.index("hit") < order.index("switch")
        caught = next(event for event in log if event[0] == "hit")
        assert caught[7] == sid("SNORLAX")
        assert sum(1 for event in log if event[0] == "move" and event[1] == 0) == 1  # it does not move twice
        if not caught[4]:
            chased.append(int(caught[2]))
    assert min(chased) > max(normal) * 1.5  # doubled power, critical hits excluded


def test_wonder_guard_blocks_fixed_damage_but_not_struggle() -> None:
    shedinja = mon("SHEDINJA", (SPLASH,))
    assert shedinja.stats[0] == 1
    battle = Battle(DEX, [mon("MACHAMP", (SEISMIC_TOSS,))], [shedinja], seed=6)
    log = battle.submit({0: 0, 1: 0})
    assert "noeffect" in kinds(log) and battle.sides[1].fighter.alive
    struggle = Battle(DEX, [mon("MACHAMP", (GROWL,))], [mon("SHEDINJA", (SPLASH,))], seed=6)
    struggle.sides[0].fighter.taunt = 3
    for _ in range(4):  # Struggle may miss nothing (it never misses) but run a few turns anyway
        if struggle.over:
            break
        struggle.submit({side: 0 for side in struggle.pending})
    assert struggle.over and struggle.winner == 0


def test_earthquake_reaches_a_digging_target_for_double_damage() -> None:
    def quake(dig: bool) -> int:
        battle = Battle(DEX, [mon("SNORLAX", (EARTHQUAKE,), level=30)],
                        [mon("SANDSLASH", (DIG, SPLASH), level=100)], seed=7)
        log = battle.submit({0: 0, 1: 0 if dig else 1})
        assert ("charging" in kinds(log)) == dig
        return max(int(event[2]) for event in log if event[0] == "hit" and event[1] == 1)

    assert quake(True) > quake(False) * 1.6
    fly = Battle(DEX, [mon("SNORLAX", (EARTHQUAKE,))], [mon("PIDGEOT", (19,))], seed=7)
    assert "miss" in kinds(fly.submit({0: 0, 1: 0}))  # a flying target is out of reach
