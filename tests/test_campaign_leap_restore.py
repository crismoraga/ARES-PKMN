"""Pre-battle restore must not undo dungeon progress (FireRed Mt. Moon loop)."""
from __future__ import annotations

from typing import Any

import pytest

from core.planner.htn_planner import HTNCampaignPlanner
from core.planner.readiness import FAR_RESTORE_TRIP

BOSS = "MT_MOON_FOSSIL_GUARD"


def _lead(hp: int, max_hp: int = 50) -> dict[str, Any]:
    return {"species": 1, "level": 14, "hp": hp, "maxHP": max_hp, "moves": [33], "pp": [30]}


@pytest.fixture()
def planner(monkeypatch: pytest.MonkeyPatch) -> HTNCampaignPlanner:
    planner = HTNCampaignPlanner()
    monkeypatch.setattr(planner, "move_info", lambda move: (0, 40, 35))
    monkeypatch.setattr(planner, "species_types", lambda species: (0,))
    return planner


def _with_trip(monkeypatch: pytest.MonkeyPatch, planner: HTNCampaignPlanner, trip: float, potions: int = 0) -> None:
    def directive(bridge: object, hp: int, max_hp: int, completed: frozenset[str]) -> dict[str, Any]:
        return {"mode": "HEAL_POKECENTER", "dest": (3, 1, (7, 4)), "trip_cost": trip}
    monkeypatch.setattr(planner, "_healing_directive", directive)
    monkeypatch.setattr(planner, "_bag_healing_hp", lambda bridge: potions)


def test_far_center_with_fit_lead_engages(monkeypatch: pytest.MonkeyPatch, planner: HTNCampaignPlanner) -> None:
    _with_trip(monkeypatch, planner, FAR_RESTORE_TRIP + 70)
    assert planner._pre_battle_restore(object(), BOSS, [_lead(30)], "Restore HP", frozenset()) is None
    assert "Pokemon Center far behind" in planner.last_engage_note and planner._restoring_for == ""


def test_near_center_restores(monkeypatch: pytest.MonkeyPatch, planner: HTNCampaignPlanner) -> None:
    _with_trip(monkeypatch, planner, 20)
    restore = planner._pre_battle_restore(object(), BOSS, [_lead(30)], "Restore HP", frozenset())
    assert restore is not None and restore["pre_battle"] == BOSS and planner._restoring_for == BOSS


def test_far_center_hurt_lead_needs_potions(monkeypatch: pytest.MonkeyPatch, planner: HTNCampaignPlanner) -> None:
    _with_trip(monkeypatch, planner, FAR_RESTORE_TRIP + 70)
    assert planner._pre_battle_restore(object(), BOSS, [_lead(20)], "Restore HP", frozenset()) is not None
    _with_trip(monkeypatch, planner, FAR_RESTORE_TRIP + 70, potions=60)
    assert planner._pre_battle_restore(object(), BOSS, [_lead(20)], "Restore HP", frozenset()) is None
    assert "healing items" in planner.last_engage_note


def test_lost_hurt_once_always_restores(monkeypatch: pytest.MonkeyPatch, planner: HTNCampaignPlanner) -> None:
    planner.preparation.record_defeat(BOSS, 14, prepared=False)
    _with_trip(monkeypatch, planner, FAR_RESTORE_TRIP + 70, potions=200)
    assert planner._pre_battle_restore(object(), BOSS, [_lead(40)], "Restore HP", frozenset()) is not None


def test_cave_floors_make_a_short_trip_far(monkeypatch: pytest.MonkeyPatch, planner: HTNCampaignPlanner) -> None:
    def directive(bridge: object, hp: int, max_hp: int, completed: frozenset[str]) -> dict[str, Any]:
        return {"mode": "HEAL_POKECENTER", "dest": (3, 1, (7, 4)), "trip_cost": 60.0, "trip_cave_maps": 3}
    monkeypatch.setattr(planner, "_healing_directive", directive)
    monkeypatch.setattr(planner, "_bag_healing_hp", lambda bridge: 0)
    assert planner._pre_battle_restore(object(), BOSS, [_lead(34)], "Restore HP", frozenset()) is None
    assert "3 cave floors" in planner.last_engage_note


def test_second_restore_needs_a_real_deficit(planner: HTNCampaignPlanner) -> None:
    policy = planner.preparation
    party = [_lead(34)]  # 68%: below the first-trip bar (85%), above the repeat bar (50%)
    assert policy.engage_reason(BOSS, party, planner.move_info, planner.species_types) is not None
    policy.record_restore(BOSS)
    assert policy.engage_reason(BOSS, party, planner.move_info, planner.species_types) is None
    assert policy.engage_reason(BOSS, [_lead(20)], planner.move_info, planner.species_types) is not None


def test_defeat_escalation_is_capped_by_the_boss_roster(planner: HTNCampaignPlanner) -> None:
    from core.planner.readiness import RETRY_LEVEL_MARGIN
    policy = planner.preparation
    for _ in range(17):
        policy.record_defeat("CERULEAN_RIVAL", 27, prepared=True)
    party = [_lead(50) | {"level": 23}]
    metadata = lambda move: (0, 40)  # noqa: E731
    assert policy.assess("CERULEAN_RIVAL", 20, party, [], metadata).target_level == 30
    capped = policy.assess("CERULEAN_RIVAL", 20, party, [], metadata, roster_level=18)
    assert capped.target_level == 18 + RETRY_LEVEL_MARGIN and capped.ready


def test_training_prefers_wilds_worth_xp(planner: HTNCampaignPlanner) -> None:
    assert planner.catalog is not None
    party = [_lead(50) | {"level": 19}]
    route3, route4 = planner.catalog.by_name["Route3"], planner.catalog.by_name["Route4"]
    assert planner._xp_penalty(route3, party) > planner._xp_penalty(route4, party) + 20


def test_unwalkable_center_trip_is_skipped(monkeypatch: pytest.MonkeyPatch, planner: HTNCampaignPlanner) -> None:
    def directive(bridge: object, hp: int, max_hp: int, completed: frozenset[str]) -> dict[str, Any]:
        return {"mode": "HEAL_POKECENTER", "dest": (8, 4, (7, 2)), "trip_cost": 30.0, "trip_validated": False}
    monkeypatch.setattr(planner, "_healing_directive", directive)
    assert planner._pre_battle_restore(object(), "PETALBURG_PARKER", [_lead(30)], "Restore HP", frozenset()) is None
    assert "No walkable way" in planner.last_engage_note and planner._restoring_for == ""
