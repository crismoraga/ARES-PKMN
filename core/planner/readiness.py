"""Campaign preparation uses available offense and bounded training slices.

Defeats are analysed instead of only raising a level target: a team that went
into the fight already hurt or with its best attacks out of PP is restored
before the retry, and only a prepared defeat asks for more levels.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
import math
from typing import Any


@dataclass(frozen=True)
class BattleReadiness:
    ready: bool
    target_level: int
    reason: str
    trained_battles: int
    budget_exhausted: bool


@dataclass
class PreparationRecord:
    frame: int
    wins: int
    retry_level: int = 0
    defeats: int = 0
    unprepared_defeats: int = 0
    restores: int = 0
    forced: int = 0


# A boss is engaged only with the strongest member nearly full and its best
# attacks stocked. Repeated restores without reaching the boss relax this.
ENGAGE_HP_FRACTION = 0.85
ENGAGE_PP_FRACTION = 0.5
LOW_PP_FRACTION = 0.25
MAX_RESTORES_PER_ATTEMPT = 3


def best_attack_pp(mon: Mapping[str, Any], move_info: Callable[[int], tuple[int, int, int]],
                   species_types: Sequence[int] = ()) -> tuple[float, int]:
    """PP fraction of the strongest damaging moves, and how many such moves exist.

    ``move_info`` maps a move to (type, power, max_pp). The strongest two
    attacks by power x STAB are the ones that win fights; Tackle with full PP
    does not make up for an empty Mud Shot.
    """
    ranked: list[tuple[float, float]] = []
    for move, pp in zip(mon.get("moves", []), mon.get("pp", [])):
        move = int(move)
        if move < 1:
            continue
        move_type, power, maximum = move_info(move)
        if power <= 1 or maximum <= 0:
            continue
        value = power * (1.5 if move_type in species_types else 1.0)
        ranked.append((value, min(1.0, int(pp) / maximum)))
    if not ranked:
        return 1.0, 0
    ranked.sort(reverse=True)
    top = ranked[:2]
    return max(fraction for _, fraction in top), len(ranked)


class CampaignPreparationPolicy:
    """Recommended levels are guidance; observed defeats request targeted work.

    A known useful attack with PP enables an earlier attempt. Training gets a
    bounded slice before reassessment instead of pursuing an arbitrary level
    forever. This is an explicit heuristic, never a formal survival proof.
    """

    def __init__(self, battle_budget: int = 8, frame_budget: int = 12000) -> None:
        if min(battle_budget, frame_budget) < 1:
            raise ValueError("Training budgets must be positive")
        self.battle_budget, self.frame_budget = battle_budget, frame_budget
        self.frame = self.wins = 0
        self.records: dict[str, PreparationRecord] = {}

    def observe(self, frame: int, wins: int) -> None:
        if min(frame, wins) < 0:
            raise ValueError("Preparation counters cannot be negative")
        self.frame, self.wins = frame, wins

    def record_defeat(self, objective: str, level: int, prepared: bool = True) -> None:
        """A prepared loss asks for levels; an unprepared one asks for a restore."""
        record = self.records.setdefault(objective, PreparationRecord(self.frame, self.wins))
        record.defeats += 1
        record.restores = record.forced = 0
        if prepared:
            # Repeated prepared losses escalate faster than one level per try.
            prepared_losses = record.defeats - record.unprepared_defeats
            record.retry_level = min(100, max(record.retry_level, level + (2 if prepared_losses <= 1 else 3)))
        else:
            record.unprepared_defeats += 1
            record.retry_level = min(100, max(record.retry_level, level))
        record.frame, record.wins = self.frame, self.wins

    def record_restore(self, objective: str) -> None:
        record = self.records.setdefault(objective, PreparationRecord(self.frame, self.wins))
        record.restores += 1

    def engage_reason(self, objective: str, party: Sequence[Mapping[str, Any]],
                      move_info: Callable[[int], tuple[int, int, int]],
                      species_types: Callable[[int], Sequence[int]]) -> str | None:
        """Why the party should visit a Pokemon Center before engaging a boss."""
        record = self.records.get(objective)
        if record is not None and record.restores >= MAX_RESTORES_PER_ATTEMPT:
            return None  # the way to the boss keeps costing HP; engage as is
        healthy = [mon for mon in party if int(mon["hp"]) > 0 and not mon.get("is_egg", False)]
        if not healthy:
            return None
        lead = max(healthy, key=lambda mon: (int(mon["level"]), int(mon["maxHP"])))
        if int(lead["hp"]) / max(1, int(lead["maxHP"])) < ENGAGE_HP_FRACTION:
            return f"Restore HP before {objective}: strongest member at {lead['hp']}/{lead['maxHP']}"
        fraction, attacks = best_attack_pp(lead, move_info, species_types(int(lead["species"])))
        if attacks and fraction < ENGAGE_PP_FRACTION:
            return f"Restore PP before {objective}: best attacks at {round(100 * fraction)}%"
        fainted = [mon for mon in party if int(mon["hp"]) <= 0 and not mon.get("is_egg", False)]
        if fainted and len(party) > 1:
            return f"Revive fainted team members before {objective}"
        return None

    def assess(self, objective: str, recommended: int, party: Sequence[Mapping[str, Any]],
               required_types: Sequence[int], move_metadata: Callable[[int], tuple[int, int]]) -> BattleReadiness:
        if recommended <= 0:
            return BattleReadiness(True, 0, "Story interaction; no level grinding", 0, False)
        record = self.records.setdefault(objective, PreparationRecord(self.frame, self.wins))
        battles = max(0, self.wins - record.wins)
        exhausted = battles >= self.battle_budget or self.frame - record.frame >= self.frame_budget
        healthy = [mon for mon in party if int(mon["hp"]) > 0 and not mon.get("is_egg", False)]
        if not healthy:
            return BattleReadiness(False, recommended, "A healthy campaign Pokemon is required", battles, exhausted)
        lead = max(healthy, key=lambda mon: (int(mon["level"]), int(mon["maxHP"])))
        offense = advantage = False
        for move, pp in zip(lead.get("moves", []), lead.get("pp", [])):
            if int(move) < 1 or int(pp) < 1:
                continue
            move_type, power = move_metadata(int(move))
            if power > 0:
                offense = True
                advantage |= move_type in required_types and int(pp) >= 3
        level = int(lead["level"])
        target = max(5, math.ceil(recommended * (0.65 if advantage else 0.85)), record.retry_level)
        ready = offense and level >= target
        if not ready and exhausted and advantage and not record.retry_level:
            ready = level >= max(5, math.ceil(recommended * 0.5))
        if not ready and record.forced and offense:
            return BattleReadiness(True, level, "Strategic deadline: attempt with the current preparation",
                                   battles, exhausted)
        if ready:
            reason = ("Useful offensive coverage and PP; attempt story objective" if advantage else
                      "Usable offense and sufficient preparation; attempt story objective")
            if exhausted:
                reason += "; training slice exhausted"
        else:
            reason = ("Observed trainer defeat requires targeted preparation" if record.retry_level else
                      "Prepare useful offense; reassess within a bounded training slice")
        return BattleReadiness(ready, target, reason, battles, exhausted)

    def force_attempt(self, objective: str) -> bool:
        """Strategic deadline during training: stop grinding and try with what we have."""
        record = self.records.setdefault(objective, PreparationRecord(self.frame, self.wins))
        if record.forced:
            return False
        record.forced = 1
        return True

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "frame": self.frame, "wins": self.wins,
                "records": {name: asdict(record) for name, record in self.records.items()}}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 1 or type(state.get("frame")) is not int or type(state.get("wins")) is not int:
            raise ValueError("Invalid preparation checkpoint")
        if min(state["frame"], state["wins"]) < 0:
            raise ValueError("Invalid preparation counters")
        raw = state.get("records")
        if not isinstance(raw, dict) or len(raw) > 512:
            raise ValueError("Invalid preparation objectives")
        records: dict[str, PreparationRecord] = {}
        known = {item.name for item in fields(PreparationRecord)}
        for name, row in raw.items():
            if not isinstance(name, str) or not isinstance(row, dict):
                raise ValueError("Invalid preparation objective")
            values = [row.get(key) for key in ("frame", "wins", "retry_level", "defeats")]
            if any(type(value) is not int or value < 0 for value in values):
                raise ValueError("Invalid preparation record")
            optional = [row.get(key, 0) for key in ("unprepared_defeats", "restores", "forced")]
            if any(type(value) is not int or value < 0 for value in optional):
                raise ValueError("Invalid preparation record")
            records[name] = PreparationRecord(**{key: value for key, value in row.items() if key in known})
        self.frame, self.wins, self.records = state["frame"], state["wins"], records

    def lessons_state(self) -> dict[str, Any]:
        """Timeline-independent part: what each objective has cost so far."""
        return {name: {"retry_level": record.retry_level, "defeats": record.defeats,
                       "unprepared_defeats": record.unprepared_defeats}
                for name, record in self.records.items() if record.defeats}

    def merge_lessons(self, lessons: Mapping[str, Any]) -> None:
        for name, row in list(lessons.items())[:512]:
            if not isinstance(name, str) or not isinstance(row, Mapping):
                continue
            values = {key: row.get(key) for key in ("retry_level", "defeats", "unprepared_defeats")}
            if any(type(value) is not int or not 0 <= value <= 10_000 for value in values.values()):
                continue
            record = self.records.setdefault(name, PreparationRecord(self.frame, self.wins))
            record.retry_level = max(record.retry_level, min(100, int(values["retry_level"] or 0)))
            record.defeats = max(record.defeats, int(values["defeats"] or 0))
            record.unprepared_defeats = max(record.unprepared_defeats, int(values["unprepared_defeats"] or 0))
