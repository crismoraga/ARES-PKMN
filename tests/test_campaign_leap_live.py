"""Live dashboard feed of the FireRed/Emerald campaign worker."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from core.campaign_live import CampaignLivePublisher, describe_event, goal_text, milestone_text


class _Bridge:
    def __init__(self) -> None:
        self.frame_count = 1000

    def framebuffer(self) -> np.ndarray:
        return np.full((160, 240), 0x00FF8040, dtype=np.uint32)

    def get_battle_pokemon(self, slot: int) -> Any:
        if slot == 0:
            return SimpleNamespace(moves=[22, 33, 0, 0])
        return SimpleNamespace(type1=11, type2=11)


class _Controller:
    def __init__(self) -> None:
        self.bridge = _Bridge()
        self.metrics = SimpleNamespace(battles=5, wins=4, losses=1, captures=1, faints=0, recoveries=2,
                                       rewinds=1, heals=3, level_ups=6, evolutions=1, map_transitions=9,
                                       faults=0, detours=0, field_moves=0)
        self.audit: list[dict[str, Any]] = [
            {"kind": "milestone_observed", "frame": 900, "milestone": "GYM_1_BROCK"},
            {"kind": "map_transition", "frame": 950, "source": [3, 1], "destination": [1, 1]},
            {"kind": "stall_detected", "frame": 990, "recovery": {"reason": "repeated_state_cycle"}},
            {"kind": "movement", "frame": 995},
        ]
        self._last_directive = {"mode": "EXECUTE_MILESTONE", "milestone": "MT_MOON_FOSSIL_GUARD"}
        self.last_decision = {"action": "move", "index": 0, "move": "VINE WHIP", "reason": "best damage",
                              "risk": 0.1, "turn": 2}
        self.watchdog = SimpleNamespace(unproductive_frames=lambda frame: 10, strategic_unproductive_frames=20,
                                        visited={(3, 1, 5, 5), (1, 1, 2, 2), (2, 7, 1, 1)})
        maps = {(3, 1): {"name": "Route4"}, (1, 1): {"name": "MtMoon_1F"}}
        self.topology = SimpleNamespace(catalog=SimpleNamespace(maps=maps))
        self.catalog = SimpleNamespace(move=lambda move: SimpleNamespace(move_type=12))

    def _type_multiplier(self, attack_type: int, defender: tuple[int, ...]) -> float:
        return 2.0 if (attack_type, defender) == (12, (11,)) else 1.0

    def telemetry(self) -> dict[str, object]:
        return {"milestones": {"completed": ["OAK_PARCEL", "GYM_1_BROCK"], "total": 50, "next": "MT_MOON_FOSSIL_GUARD"},
                "trainer": {"badge_count": 1, "money": 3000, "dex_owned": 4, "dex_seen": 9, "play_time": "1:02:03"},
                "map_name": "MtMoon_1F",
                "party": [{"name": "IVYSAUR", "species": 2, "level": 17, "hp": 40, "maxHP": 48, "status": "",
                           "types": ["GRASS", "POISON"], "moves": [{"name": "VINE WHIP"}, {"name": "TACKLE"}]}],
                "battle": {"trainer": True, "own": {"name": "IVYSAUR", "level": 17},
                           "enemy": {"name": "GEODUDE", "level": 12, "types": ["ROCK", "GROUND"]}},
                "plan": {"type_chart": {"known_cells": 12, "observations": 40}, "counter_types": ["WATER"]},
                "preparation": {"MT_MOON_FOSSIL_GUARD": {"retry_level": 19, "defeats": 1}}}


def test_status_follows_live_contract(tmp_path: Path) -> None:
    controller = _Controller()
    publisher = CampaignLivePublisher("firered", root=tmp_path)
    assert publisher.maybe_publish(controller, force=True)
    status = json.loads((tmp_path / "campaign-firered" / "status.json").read_text(encoding="utf-8"))
    assert status["kind"] == "campaign" and status["game"] == "firered" and status["generation"] == 3
    assert status["badges"] == 1 and status["milestones"] == ["OAK_PARCEL", "GYM_1_BROCK"]
    assert status["maps_visited"] == 3  # Route4, MtMoon_1F and the unnamed 2.7 from the watchdog tiles
    assert status["party"][0] == {"species": "IVYSAUR", "nickname": "", "level": 17, "hp": 40, "max_hp": 48,
                                  "status": "", "types": ["GRASS", "POISON"], "moves": ["VINE WHIP", "TACKLE"]}
    assert status["goal"] == "Cruzar el Monte Moon y vencer al Supernecio Miguel"
    assert status["battle"]["choice"] == "move" and status["battle"]["effectiveness"] == 2.0
    assert status["stats"]["stalls"] == 1 and status["stats"]["wins"] == 4 and status["stats"]["rewinds"] == 1
    assert status["knowledge"]["lessons"]["MT_MOON_FOSSIL_GUARD"]["retry_level"] == 19
    assert any("Hito completado" in line for line in status["events"])
    assert (tmp_path / "campaign-firered" / "frame.png").stat().st_size > 0
    assert len((tmp_path / "campaign-firered" / "history.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_publisher_is_rate_limited_and_never_raises(tmp_path: Path) -> None:
    controller = _Controller()
    publisher = CampaignLivePublisher("emerald", root=tmp_path, interval=60.0)
    assert publisher.maybe_publish(controller)
    assert not publisher.maybe_publish(controller)  # within the interval
    broken = SimpleNamespace(bridge=None)
    assert not publisher.maybe_publish(broken, force=True)  # failures are swallowed


def test_brain_choice_is_preferred_and_events_are_incremental(tmp_path: Path) -> None:
    controller = _Controller()
    controller.last_battle_choice = SimpleNamespace(kind="switch", index=2, reason="Cambio a un tipo resistente",
                                                    scores=[0.1, 0.9])
    publisher = CampaignLivePublisher("firered", root=tmp_path)
    first = publisher.build_status(controller)
    assert first["battle"]["choice"] == "switch" and first["battle"]["reason"] == "Cambio a un tipo resistente"
    controller.audit.append({"kind": "native_battle_result", "frame": 1200, "outcome": 1, "turns": 3})
    second = publisher.build_status(controller)
    assert second["events"][-1].endswith("Fin del combate: victoria (3 turnos)")
    assert sum("Hito completado" in line for line in second["events"]) == 1  # not re-ingested


def test_spanish_goal_texts() -> None:
    assert goal_text({"mode": "HEAL_POKECENTER", "pre_battle": "GYM_2_MISTY"}).startswith("Curar al equipo")
    assert "nivel 20" in goal_text({"mode": "OPTIMAL_TRAINING_CURRICULUM", "target_level": 20,
                                    "milestone": "CERULEAN_RIVAL", "training_route": "Route_4"})
    assert goal_text({}, "GYM_3_WATTSON") == "Ir al Gimnasio de Malvalona (Erico)"
    assert milestone_text("SOME_NEW_THING") == "Hito: some new thing"
    assert describe_event(None, {"kind": "movement", "frame": 1}) is None
