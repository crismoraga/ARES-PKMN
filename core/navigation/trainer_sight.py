"""Undefeated trainers' lines of sight from source templates, stable off camera.

Generation-three only spawns actors near the camera. Costing a trainer's sight
only while it is spawned makes a path flip each time the trainer enters or
leaves the view: one step up shows Lenny and the plan goes south, the next step
hides him and the plan goes north again. Sight lines here come from the map's
templates (runtime position when a script moved them) and disappear only when
the trainer's defeat flag is set, so the cost no longer depends on the camera.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from core.navigation.obstacles import FlagReader
from core.navigation.script_gates import load_trainers
from core.navigation.source_data import MapKey, load_map_catalog

XY = tuple[int, int]
_FACINGS = {"DOWN": 1, "UP": 2, "LEFT": 3, "RIGHT": 4}


@dataclass(frozen=True)
class TrainerSight:
    local_id: int
    xy: XY
    directions: tuple[int, ...]
    sight: int
    flag: int | None


def _directions(movement_type: str, trainer_type: str) -> tuple[int, ...]:
    if "SEE_ALL" in trainer_type or "BURIED" in trainer_type:
        return (1, 2, 3, 4)
    name = movement_type.replace("MOVEMENT_TYPE_", "")
    if name.startswith("FACE_"):
        found = tuple(sorted({facing for word, facing in _FACINGS.items() if word in name.split("_")}))
        if found:
            return found
    return (1, 2, 3, 4)  # rotating, looking around or walking: any side can see


@lru_cache(maxsize=1024)
def trainer_sights(game: str, key: MapKey) -> tuple[TrainerSight, ...]:
    catalog = load_map_catalog(game)
    raw = catalog.maps.get(key)
    if raw is None:
        return ()
    trainers = load_trainers(game).get(str(raw["name"]), {})
    start = catalog.constants.get("TRAINER_FLAGS_START")
    result: list[TrainerSight] = []
    for index, event in enumerate(raw.get("object_events", []) or []):
        trainer_type = str(event.get("trainer_type", "TRAINER_TYPE_NONE"))
        if trainer_type in ("TRAINER_TYPE_NONE", "0"):
            continue
        try:
            sight = int(str(event.get("trainer_sight_or_berry_tree_id", "0")), 0)
            x, y = int(event["x"]), int(event["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if sight <= 0:
            continue
        constant = trainers.get(str(index + 1))
        trainer_id = catalog.constants.get(constant) if constant else None
        flag = None if trainer_id is None or start is None else int(start) + int(trainer_id)
        result.append(TrainerSight(index + 1, (x, y), _directions(str(event.get("movement_type", "")), trainer_type),
                                   sight, flag))
    return tuple(result)


def undefeated_sights(game: str, key: MapKey, flags: FlagReader | None,
                      positions: dict[int, XY] | None = None) -> list[TrainerSight]:
    """Sight lines of trainers whose defeat flag is not set (unknown flag: assume undefeated)."""
    sights: list[TrainerSight] = []
    for sight in trainer_sights(game, key):
        if sight.flag is not None and flags is not None:
            try:
                if flags.get_event_flag(sight.flag):
                    continue
            except ValueError:
                pass
        xy = positions.get(sight.local_id, sight.xy) if positions else sight.xy
        sights.append(TrainerSight(sight.local_id, xy, sight.directions, sight.sight, sight.flag))
    return sights


def defeated_local_ids(game: str, key: MapKey, flags: FlagReader | None) -> frozenset[int]:
    if flags is None:
        return frozenset()
    defeated: set[int] = set()
    for sight in trainer_sights(game, key):
        if sight.flag is None:
            continue
        try:
            if flags.get_event_flag(sight.flag):
                defeated.add(sight.local_id)
        except ValueError:
            continue
    return frozenset(defeated)
