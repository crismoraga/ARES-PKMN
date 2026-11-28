"""``BattleView`` -> a ``core.stadium`` battle position, so the E1 Transformer can read any game.

The E1 policy (``ares_u.battle``) observes a Stadium ``Battle`` through ``ares_u.battle.obs.encode``.
``PolicyMapper.build`` reconstructs that position from a view:

* species: national dex -> Stadium (Gen-3 internal) id; a species beyond #386 becomes a synthetic
  species with the view's types and base stats inferred from its stats (own) or 70s (enemy);
* moves: ``gen3_move_id`` (Gen 1-5 ids 1-354 coincide), else the English/Spanish name, else a
  synthetic move with the view's type/power/accuracy;
* types: the view's own types (Gen 1 Magnemite is pure Electric); Fairy is the free ``???`` slot (9);
* type chart: a dex variant per generation family - Gen 1 (Ghost x0 Psychic, Bug x2 Poison,
  Poison x2 Bug, Ice x1 Fire), Gen 2-5 (the Gen-3 ROM table), Gen 6+ (Fairy, Steel no longer
  resists Ghost/Dark) - so the observation's effectiveness features follow the game's chart;
* HP, status, stat stages, PP, weather and turn are copied; the enemy's stats are the public
  estimate (the observation never reads its real stats) and only its revealed moves enter the
  knowledge (honest view);
* the policy sees 3v3 positions (``TEAM_SIZE``): the active plus the two bench members the caller
  ranks first; the enemy team is its active plus hidden placeholders for the members it still has.

Engine actions 0-3 are the move slots and 4+k the k-th shown bench member; ``action_map`` translates
them back to ``("move", slot)`` / ``("switch", party_index)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ares_u.battle.env import MAX_TURNS
from ares_u.battle.obs import Observation, OracleKnowledge, encode
from ares_u.battle.teams import TEAM_SIZE
from core.stadium.dex import Dex, MoveData, SpeciesData
from core.stadium.engine import BURN, FREEZE, PARALYSIS, POISON, TOXIC, Battle, MonSpec

from .api import GEN3_TYPE_INDEX, BattleView, MonView, MoveView, canonical_type
from .moves import gen3_id_for
from .strategy import internal_species

FAIRY_INDEX = 9  # Gen-3 "???" type slot, free in every Gen-3 table: Fairy lives there for Gen 6+
STATUS_BITS = {"psn": POISON, "brn": BURN, "frz": FREEZE, "par": PARALYSIS, "tox": TOXIC}
STAGE_INDEX = {"atk": 1, "def": 2, "spe": 3, "spa": 4, "spd": 5, "acc": 6, "eva": 7}
WEATHER = {"rain": "rain", "lluvia": "rain", "sun": "sun", "sol": "sun", "sandstorm": "sandstorm",
           "arena": "sandstorm", "hail": "hail", "granizo": "hail"}
MOVE_STRUGGLE = 165


def type_index(name: str) -> int:
    canonical = canonical_type(name)
    if canonical == "fairy":
        return FAIRY_INDEX
    return GEN3_TYPE_INDEX.get(canonical, 0 if canonical == "???" else 0)


def chart_family(generation: int) -> int:
    return 1 if generation <= 1 else 3 if generation <= 5 else 6


def _chart_entries(base: tuple[tuple[int, int, int], ...], family: int) -> tuple[tuple[int, int, int], ...]:
    entries = list(base)
    if family == 1:
        entries = [entry for entry in entries
                   if not (entry[0] == 7 and entry[1] == 14) and not (entry[0] == 15 and entry[1] == 10)
                   and not (entry[0] == 6 and entry[1] == 3)]
        entries += [(7, 14, 0), (6, 3, 20), (3, 6, 20)]
    elif family == 6:
        entries = [entry for entry in entries if not (entry[0] in (7, 17) and entry[1] == 8 and entry[2] == 5)]
        fairy = FAIRY_INDEX
        entries += [(fairy, 1, 20), (fairy, 16, 20), (fairy, 17, 20), (fairy, 10, 5), (fairy, 3, 5), (fairy, 8, 5),
                    (1, fairy, 5), (6, fairy, 5), (17, fairy, 5), (16, fairy, 0), (3, fairy, 20), (8, fairy, 20)]
    return tuple(entries)


@dataclass
class _Family:
    dex: Dex
    extra_species: dict[tuple[object, ...], int] = field(default_factory=dict)
    extra_moves: dict[tuple[object, ...], int] = field(default_factory=dict)


@dataclass
class PolicyInput:
    battle: Battle
    observation: Observation
    legal: list[int]
    action_map: dict[int, tuple[str, int]]
    shown_bench: list[int]


class PolicyMapper:
    """Builds Stadium positions from views (one dex variant per chart family, synthetic entries cached)."""

    def __init__(self, dex: Dex | None = None) -> None:
        if dex is None:
            from ares_u.battle.teams import default_dex

            dex = default_dex()
        self.base = dex
        self._families: dict[int, _Family] = {}

    # ------------------------------------------------------------------ dex variants
    def family(self, generation: int) -> _Family:
        key = chart_family(generation)
        if key not in self._families:
            self._families[key] = _Family(self._variant(key, (), ()))
        return self._families[key]

    def _variant(self, family: int, species: tuple[SpeciesData, ...], moves: tuple[MoveData, ...]) -> Dex:
        base = self.base
        dex = Dex(base.game, base.moves + moves, base.species + species, _chart_entries(base.type_entries, family),
                  base.evolutions + ((),) * len(species), base.learnsets + ((),) * len(species), base.source)
        return dex

    def _rebuild(self, generation: int) -> None:
        key = chart_family(generation)
        current = self._families[key]
        species = tuple(sorted((SpeciesData(index, str(spec[0]), spec[1], spec[2], (0, 0))  # type: ignore[arg-type]
                                for spec, index in current.extra_species.items()), key=lambda row: row.species_id))
        moves = tuple(sorted((MoveData(index, str(spec[0]), 0, int(spec[1]), int(spec[2]), int(spec[3]),  # type: ignore[call-overload]
                                       int(spec[4]), 0, 0, int(spec[5]), 0)  # type: ignore[call-overload]
                              for spec, index in current.extra_moves.items()), key=lambda row: row.move_id))
        current.dex = self._variant(key, species, moves)

    def species_id(self, mon: MonView, generation: int, own: bool) -> int:
        internal = internal_species(mon.national_dex) if mon.national_dex else None
        if internal is not None:
            return internal
        family = self.family(generation)
        types = self.types(mon)
        if own and mon.stats:
            level = max(1, mon.level)
            # invert the stat formula at IV 15 (no EVs): base ~ ((stat - 5) * 100 / level - 15) / 2
            def base(value: float, hp: bool = False) -> int:
                offset = level + 10 if hp else 5
                return int(max(5, min(255, ((value - offset) * 100 / level - 15) / 2)))
            stats = (base(mon.max_hp, True), base(mon.stats.get("atk", 70)), base(mon.stats.get("def", 70)),
                     base(mon.stats.get("spe", 70)), base(mon.stats.get("spa", 70)), base(mon.stats.get("spd", 70)))
        else:
            stats = (70, 70, 70, 70, 70, 70)
        key: tuple[object, ...] = (mon.species or "SYNTH", stats, types)
        if key not in family.extra_species:
            family.extra_species[key] = len(self.base.species) + len(family.extra_species)
            self._rebuild(generation)
        return family.extra_species[key]

    def move_id(self, move: MoveView, generation: int) -> int:
        found = gen3_id_for(move)
        if found is not None:
            return found
        family = self.family(generation)
        key = (move.name or "SYNTH", max(0, int(move.power)), type_index(move.type), max(0, int(move.accuracy)),
               max(1, int(move.max_pp or move.pp or 10)), int(move.priority or 0))
        if key not in family.extra_moves:
            family.extra_moves[key] = len(self.base.moves) + len(family.extra_moves)
            self._rebuild(generation)
        return family.extra_moves[key]

    @staticmethod
    def types(mon: MonView) -> tuple[int, int]:
        indices = [type_index(name) for name in mon.types if canonical_type(name) != "???"] or [0]
        return (indices[0], indices[1] if len(indices) > 1 else indices[0])

    # ------------------------------------------------------------------ positions
    def _spec(self, mon: MonView, generation: int, own: bool, dex: Dex) -> tuple[MonSpec, list[int]]:
        species = self.species_id(mon, generation, own)
        moves = [self.move_id(move, generation) for move in mon.moves][:4] or [MOVE_STRUGGLE]
        level = max(1, min(100, int(mon.level)))
        if own:
            stats = mon.stats or {}
            public = _public(dex, species, level)
            values = (max(1, int(mon.max_hp)), int(stats.get("atk", public[1])), int(stats.get("def", public[2])),
                      int(stats.get("spe", public[3])), int(stats.get("spa", public[4])),
                      int(stats.get("spd", public[5])))
        else:
            values = _public(dex, species, level)
        values = tuple(max(1, value) for value in values)  # type: ignore[assignment]
        spec = MonSpec(species, level, (values[0], values[1], values[2], values[3], values[4], values[5]),
                       tuple(moves), name=mon.species)
        return spec, moves

    def build(self, view: BattleView, bench_order: list[int]) -> PolicyInput:
        generation = view.generation
        # make sure every synthetic entry exists before the dex is fixed for this position
        members = [view.active_index] + [index for index in bench_order if index != view.active_index]
        members = members[:TEAM_SIZE]
        mons = [view.me] + [view.party[index] for index in members[1:]]
        for mon in mons:
            self.species_id(mon, generation, True)
            for move in mon.moves:
                self.move_id(move, generation)
        self.species_id(view.enemy, generation, False)
        for move in view.enemy.moves:
            self.move_id(move, generation)
        dex = self.family(generation).dex
        own_specs = [self._spec(mon, generation, True, dex)[0] for mon in mons]
        enemy_spec, enemy_moves = self._spec(view.enemy, generation, False, dex)
        hidden = 0
        if view.enemy_remaining is not None:
            hidden = max(0, min(TEAM_SIZE - 1, int(view.enemy_remaining) - (1 if view.enemy.hp > 0 else 0)))
        enemy_team = [enemy_spec] * (1 + hidden)
        battle = Battle(dex, own_specs, enemy_team, seed=0, max_turns=MAX_TURNS, record=False)
        for fighter, mon in zip(battle.sides[0].party, mons):
            _copy_state(fighter, mon, own=True)
        enemy_fighter = battle.sides[1].party[0]
        _copy_state(enemy_fighter, view.enemy, own=False)
        battle.turn = max(0, int(view.turn))
        weather = WEATHER.get(view.weather.lower(), "") if view.weather else ""
        if weather:
            battle.weather, battle.weather_turns = weather, 5
        forced = view.me.hp <= 0
        if forced:
            battle._replacing = True
            battle.pending = (0,)
        legal = battle.legal_actions(0)
        if not view.can_switch and not forced:
            legal = [action for action in legal if action < 4] or legal
        allowed = getattr(view, "legal_moves", None)
        if allowed is not None and not forced:
            moves_ok = [action for action in legal if action >= 4 or action in allowed]
            legal = moves_ok or legal
        action_map: dict[int, tuple[str, int]] = {slot: ("move", slot) for slot in range(4)}
        for position, member in enumerate(members[1:]):
            action_map[4 + position] = ("switch", member)
        knowledge = OracleKnowledge(dex=dex)
        if view.enemy.moves:  # honest: only what the foe has shown
            knowledge.revealed_moves[enemy_spec.species] = set(enemy_moves)
        revealed = (set(range(len(own_specs))), {0})
        observation = encode(battle, 0, revealed, knowledge, legal)
        return PolicyInput(battle, observation, legal, action_map, members[1:])


def _public(dex: Dex, species: int, level: int) -> tuple[int, int, int, int, int, int]:
    from core.stadium.features import public_stats

    return public_stats(dex, species, level)


def _copy_state(fighter: object, mon: MonView, own: bool) -> None:
    from core.stadium.engine import Fighter

    assert isinstance(fighter, Fighter)
    if own:
        fighter.hp = max(0, min(fighter.max_hp, int(mon.hp)))
    else:
        fighter.hp = 0 if mon.hp <= 0 else max(1, round(mon.hp_fraction * fighter.max_hp))
    status = mon.status
    if status == "slp":
        fighter.status = 2
    else:
        fighter.status = STATUS_BITS.get(status, 0)
    if status == "tox":
        fighter.toxic_n = 1
    for name, stage in (mon.stages or {}).items():
        index = STAGE_INDEX.get(name)
        if index is not None:
            fighter.stages[index] = 6 + max(-6, min(6, int(stage)))
    for slot, move in enumerate(mon.moves[:4]):
        if slot < len(fighter.pp):
            fighter.pp[slot] = max(0, int(move.pp)) if own else fighter.pp[slot]
    types = PolicyMapper.types(mon)
    fighter.types = types


__all__ = ["FAIRY_INDEX", "PolicyInput", "PolicyMapper", "chart_family", "type_index"]
