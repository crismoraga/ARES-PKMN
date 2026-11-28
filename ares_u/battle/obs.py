"""Game-agnostic observation of a battle from one side, without species or move IDs.

Tokens (all fixed size, so observations stack into batches):

* ``field``    (FIELD_DIM): weather, turn, screens/hazards of both sides, remaining members, HP margins.
* ``entities`` (ENTITIES x ENTITY_DIM): own [active, bench...] then foe [active, bench...]. A foe bench
  member that has not appeared yet is a *hidden* token (only its presence and side are visible).
  Features: presence/hidden/own/active flags, type multi-hot, base stats, level, HP fraction, status,
  stat stages and volatile flags. Base stats and the HP bar are public in the games.
* ``actions``  (ACTIONS x ACTION_DIM): engine actions 0-3 are the move slots, 4+k switches to own
  bench member k (= own entity token 1+k). Move tokens carry type, power, accuracy, category, PP,
  effectiveness against the foe active, STAB and the estimated damage fraction; switch tokens carry
  the candidate's HP, its best estimated damage, the threat it faces and its speed edge.
* ``mask``     legal actions of the engine.
* ``species``  internal species ids of the entity tokens (0 when hidden). Only the ``ids`` ablation
  model reads it; the agnostic model never does.

Type effectiveness uses the game's real type chart (the same universal chart in every generation,
which is public game knowledge, not an identity). ``flat`` concatenates everything for the DQN MLP.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from core.stadium.dex import TYPE_COUNT, TYPE_MYSTERY, Dex, MoveData
from core.stadium.engine import (BURN, FREEZE, PARALYSIS, POISON, SLEEP_MASK, SPE, STAGE_RATIOS, TOXIC, Battle,
                                 Fighter)
from core.stadium.features import Perspective
from core.stadium.knowledge import SideKnowledge

from .teams import TEAM_SIZE

ENTITIES = 2 * TEAM_SIZE
ACTIONS = 4 + TEAM_SIZE - 1
ENTITY_DIM = 48
ACTION_DIM = 44
FIELD_DIM = 22
FLAT_DIM = FIELD_DIM + ENTITIES * ENTITY_DIM + ACTIONS * ACTION_DIM + ACTIONS
SPECIES_VOCAB = 412
WEATHERS = ("none", "rain", "sun", "sandstorm", "hail")
_STATUS_EFFECTS = frozenset({1, 66, 67, 33, 167, 49, 187, 118, 199})
_SELF_BOOST = frozenset({10, 11, 12, 13, 14, 16, 50, 51, 52, 53, 54, 56, 108, 142, 156, 206, 208, 211, 212, 109})
_FOE_DROP = frozenset({18, 19, 20, 21, 22, 23, 24, 58, 59, 60, 61, 62, 63, 64, 205})
_HEALS = frozenset({32, 157, 132, 133, 134, 37})
_FIXED_DAMAGE = frozenset({41, 130, 87, 88, 40, 189})
NOMINAL_VARIABLE_POWER = 60

Float = NDArray[np.float32]


@dataclass
class OracleKnowledge(SideKnowledge):
    """Knowledge that knows the real type chart (heuristic baselines play with full chart knowledge)."""
    dex: Dex | None = field(default=None)

    def multiplier(self, attack_type: int, defender_types: tuple[int, int]) -> float | None:
        if self.dex is None:
            return super().multiplier(attack_type, defender_types)
        return self.dex.multiplier(attack_type, defender_types)


class Observation(NamedTuple):
    field: Float          # (FIELD_DIM,)
    entities: Float       # (ENTITIES, ENTITY_DIM)
    actions: Float        # (ACTIONS, ACTION_DIM)
    mask: NDArray[np.bool_]   # (ACTIONS,)
    species: NDArray[np.int64]  # (ENTITIES,)


class Batch(NamedTuple):
    field: Float
    entities: Float
    actions: Float
    mask: NDArray[np.bool_]
    species: NDArray[np.int64]


def stack(observations: list[Observation]) -> Batch:
    return Batch(np.stack([o.field for o in observations]), np.stack([o.entities for o in observations]),
                 np.stack([o.actions for o in observations]), np.stack([o.mask for o in observations]),
                 np.stack([o.species for o in observations]))


def flat(observation: Observation) -> Float:
    """Flat vector of the DQN baseline (same information, no IDs)."""
    vector: Float = np.concatenate([observation.field, observation.entities.ravel(), observation.actions.ravel(),
                                    observation.mask.astype(np.float32)]).astype(np.float32)
    return vector


def flat_batch(batch: Batch) -> Float:
    size = batch.field.shape[0]
    matrix: Float = np.concatenate([batch.field, batch.entities.reshape(size, -1), batch.actions.reshape(size, -1),
                                    batch.mask.astype(np.float32)], axis=1).astype(np.float32)
    return matrix


def _stage(value: float, stage: int) -> float:
    numerator, denominator = STAGE_RATIOS[max(0, min(12, stage))]
    return value * numerator / denominator


def _status_slot(status: int) -> int:
    """0 slp, 1 psn, 2 brn, 3 frz, 4 par, 5 tox, -1 none."""
    if status & SLEEP_MASK:
        return 0
    if status & TOXIC:
        return 5
    if status & POISON:
        return 1
    if status & BURN:
        return 2
    if status & FREEZE:
        return 3
    if status & PARALYSIS:
        return 4
    return -1


def _entity(out: Float, fighter: Fighter, dex: Dex, own: bool, active: bool) -> None:
    out[0] = 1.0
    out[2] = float(own)
    out[3] = float(active)
    out[4 + fighter.types[0]] = 1.0
    out[4 + fighter.types[1]] = 1.0
    base = dex.species[fighter.spec.species].base_stats
    out[22:28] = [value / 255 for value in base]
    out[28] = fighter.spec.level / 100
    out[29] = fighter.hp / max(1, fighter.max_hp)
    out[30] = float(fighter.alive)
    slot = _status_slot(fighter.status)
    if slot >= 0:
        out[31 + slot] = 1.0
    out[37:44] = [(stage - 6) / 6 for stage in fighter.stages[1:8]]
    out[44] = float(fighter.confusion > 0)
    out[45] = float(fighter.substitute > 0)
    out[46] = float(fighter.seeded)
    out[47] = float(fighter.recharge or fighter.charging >= 0 or fighter.rampage or fighter.lock_slot >= 0)


def _hidden(out: Float) -> None:
    out[0] = 1.0
    out[1] = 1.0


def effective_power_move(move: MoveData) -> MoveData:
    """Variable-power moves (power 1 in the tables) are estimated at a nominal power."""
    if move.power == 1 and move.effect not in _FIXED_DAMAGE:
        return replace(move, power=NOMINAL_VARIABLE_POWER)
    return move


_WEATHER_BALL_TYPES = {"rain": 11, "sun": 10, "sandstorm": 5, "hail": 15}


def hidden_power(ivs: Sequence[int]) -> tuple[int, int]:
    """Gen-3 Hidden Power (type, power) from IVs, mirroring ``Battle._move_type_power`` (effect 135)."""
    order = tuple(int(value) for value in ivs[:6])
    type_bits = sum(((value & 1) << index) for index, value in enumerate(order))
    power_bits = sum((((value >> 1) & 1) << index) for index, value in enumerate(order))
    move_type = type_bits * 15 // 63 + 1
    if move_type >= TYPE_MYSTERY:
        move_type += 1
    return move_type, power_bits * 40 // 63 + 30


def resolve_move(battle: Battle, fighter: Fighter, move: MoveData) -> MoveData:
    """The type/power the engine will actually use for type-changing moves (Hidden Power, Weather Ball).

    Deterministic and side-effect free (unlike ``Battle._move_type_power``, which draws RNG for Magnitude).
    """
    if move.effect == 135:
        move_type, power = hidden_power(fighter.spec.ivs)
        return replace(move, move_type=move_type, power=power)
    if move.effect == 203:
        weather = battle._weather_active()
        if weather in _WEATHER_BALL_TYPES:
            return replace(move, move_type=_WEATHER_BALL_TYPES[weather], power=100)
    return move


def _speed(value: float, fighter: Fighter) -> float:
    return _stage(value, fighter.stages[SPE]) * (0.25 if fighter.status & PARALYSIS else 1.0)


def _log_eff(multiplier: float) -> float:
    return -1.5 if multiplier <= 0 else math.log2(multiplier) / 2


def encode(battle: Battle, side: int, revealed: tuple[set[int], set[int]], knowledge: OracleKnowledge,
           legal: list[int] | None = None) -> Observation:
    """Observation of ``battle`` for ``side``; ``revealed[s]`` = party indices of side s seen so far."""
    dex = battle.dex
    own, foe_side = battle.sides[side], battle.sides[1 - side]
    me, foe = own.fighter, foe_side.fighter
    view = Perspective(battle, side, knowledge)
    field_row = np.zeros(FIELD_DIM, dtype=np.float32)
    weather = battle._weather_active()
    field_row[WEATHERS.index(weather) if weather in WEATHERS else 0] = 1.0
    field_row[5] = min(1.0, battle.turn / max(1, battle.max_turns))
    for offset, current in ((6, own), (11, foe_side)):
        field_row[offset:offset + 5] = [float(current.reflect > 0), float(current.light_screen > 0),
                                        current.spikes / 3, float(current.safeguard > 0), float(current.mist > 0)]
    field_row[16] = own.remaining() / len(own.party)
    field_row[17] = foe_side.remaining() / len(foe_side.party)
    field_row[18] = own.hp_fraction()
    # Foe HP margin from public information only: revealed members contribute their HP fraction,
    # unrevealed living members count as full (their max HP is hidden).
    field_row[19] = sum(
        (member.hp / max(1, member.max_hp)) if index in revealed[1 - side] else float(member.alive)
        for index, member in enumerate(foe_side.party)) / max(1, len(foe_side.party))
    field_row[20] = float(battle._replacing)
    foe_public = view.foe_stats
    my_speed, foe_speed = _speed(me.stat(3), me), _speed(foe_public[3], foe)
    field_row[21] = 1.0 if my_speed > foe_speed else 0.5 if my_speed == foe_speed else 0.0

    entities = np.zeros((ENTITIES, ENTITY_DIM), dtype=np.float32)
    species = np.zeros(ENTITIES, dtype=np.int64)
    own_order = [own.active, *own.bench()][:TEAM_SIZE]
    foe_order = [foe_side.active, *foe_side.bench()][:TEAM_SIZE]
    for position, index in enumerate(own_order):
        _entity(entities[position], own.party[index], dex, True, position == 0)
        species[position] = own.party[index].spec.species
    for position, index in enumerate(foe_order):
        row = entities[TEAM_SIZE + position]
        if index in revealed[1 - side]:
            _entity(row, foe_side.party[index], dex, False, position == 0)
            species[TEAM_SIZE + position] = foe_side.party[index].spec.species
        else:
            _hidden(row)

    actions = np.zeros((ACTIONS, ACTION_DIM), dtype=np.float32)
    mask = np.zeros(ACTIONS, dtype=np.bool_)
    for action in (legal if legal is not None else battle.legal_actions(side)):
        if action < ACTIONS:
            mask[action] = True
    foe_hp = max(1e-6, view.foe_hp)
    for slot in range(min(4, len(me.move_ids))):
        out = actions[slot]
        move = resolve_move(battle, me, dex.moves[me.move_ids[slot]])
        estimate = effective_power_move(move)
        out[0], out[2] = 1.0, float(mask[slot])
        out[3 + move.move_type] = 1.0
        out[21] = min(2.0, estimate.power / 150)
        out[22] = float(move.power == 1)
        out[23] = 1.0 if move.accuracy == 0 else move.accuracy / 100
        category = 2 if move.power == 0 else 1 if move.special else 0
        out[24 + category] = 1.0
        out[27] = me.pp[slot] / max(1, move.pp)
        multiplier = dex.multiplier(move.move_type, foe.types)
        out[28] = _log_eff(multiplier) if move.power else 0.0
        out[29] = float(multiplier == 0 and move.power > 0)
        out[30] = float(move.move_type in me.types and move.power > 0)
        low, high, _, _ = view.move_damage(me, estimate, own_attacker=True)
        out[31] = min(2.0, (low + high) / 2 / foe_hp) if foe.alive else 0.0
        out[32] = float(foe.alive and high >= foe_hp and high > 0)
        out[33] = max(-1.0, min(1.0, move.priority / 3))
        out[34] = float(move.effect in _STATUS_EFFECTS and not foe.status)
        out[35] = float(move.effect in _SELF_BOOST)
        out[36] = float(move.effect in _FOE_DROP)
        out[37] = float(move.effect in _HEALS) * (1 - me.hp / max(1, me.max_hp))
    bench = own.bench()
    for position, index in enumerate(bench[:ACTIONS - 4]):
        out = actions[4 + position]
        candidate = own.party[index]
        out[1], out[2] = 1.0, float(mask[4 + position])
        out[38] = candidate.hp / max(1, candidate.max_hp)
        best, best_eff = 0.0, 0.0
        for move_id in candidate.move_ids:
            move = resolve_move(battle, candidate, dex.moves[move_id])
            if move.power:
                low, high, _, _ = view.move_damage(candidate, effective_power_move(move), own_attacker=True)
                best = max(best, (low + high) / 2)
                best_eff = max(best_eff, dex.multiplier(move.move_type, foe.types))
        out[39] = min(2.0, best / foe_hp) if foe.alive else 0.0
        out[40] = min(3.0, view.threat(candidate)) / 3 if foe.alive else 0.0
        out[41] = 1.0 if _speed(candidate.stat(3), candidate) > foe_speed else 0.0
        out[42] = _log_eff(best_eff) if best_eff > 0 else -1.5
        incoming = max((dex.multiplier(move_type, candidate.types) for move_type in set(foe.types)), default=1.0)
        out[43] = _log_eff(incoming) if foe.alive else 0.0
    return Observation(field_row, entities, actions, mask, species)


__all__ = ["ACTIONS", "ACTION_DIM", "Batch", "ENTITIES", "ENTITY_DIM", "FIELD_DIM", "FLAT_DIM", "Observation",
           "OracleKnowledge", "SPECIES_VOCAB", "TYPE_COUNT", "encode", "flat", "flat_batch", "stack"]
