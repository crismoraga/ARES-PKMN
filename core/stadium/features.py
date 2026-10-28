"""Observable decision features: option scoring inputs and deep-policy tensors.

Information contract (identical to the campaign's ``OpponentKnowledge``):
an agent sees its own team completely; of the opponent it sees the species,
level, HP fraction (the HP bar), status, stat stages and whatever moves or
abilities were revealed in battle. Opponent stats are *estimated* from public
base stats (IV 15, no EVs, neutral nature). Type effectiveness comes from the
agent's learnt chart, never from the ROM table.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # lazily imported: NumPy-only workers never load Torch
    import torch

from .dex import Dex, MoveData
from .engine import (ACC, ATK, BURN, DEF, PARALYSIS, SLEEP_MASK, SPA, SPD, SPE, STAGE_RATIOS, TARGET_USER,
                     Battle, Fighter)
from .knowledge import SideKnowledge

OPTION_FEATURES = 25
FEATURE_NAMES = (
    "is_move", "dmg_cur", "dmg_max", "ko_prob", "accuracy", "eff_known", "eff_log2", "is_status",
    "inflicts_status", "self_boost", "foe_drop", "heal_value", "priority", "pp_frac", "is_switch", "own_hp",
    "foe_hp", "faster", "threat", "own_left", "foe_left", "commitment", "switch_offense", "switch_hp", "turn")
_STATUS_EFFECTS = {1: "slp", 66: "psn", 67: "par", 33: "tox", 167: "brn", 49: "cnf", 187: "yawn", 118: "cnf", 199: "cnf"}
_SELF_BOOST = frozenset({10, 11, 12, 13, 14, 16, 50, 51, 52, 53, 54, 56, 108, 142, 156, 206, 208, 211, 212, 109, 47})
_FOE_DROP = frozenset({18, 19, 20, 21, 22, 23, 24, 58, 59, 60, 61, 62, 63, 64, 205})
_HEALS = frozenset({32, 157, 132, 133, 134, 37})
_COMMIT = frozenset({7, 80, 39, 75, 145, 151, 155, 168})
_PRIOR_POWER = 60


def public_stats(dex: Dex, species: int, level: int) -> tuple[int, int, int, int, int, int]:
    base = dex.species[species].base_stats
    values = [(2 * base[index] + 15) * level // 100 + (level + 10 if index == 0 else 5) for index in range(6)]
    return values[0], values[1], values[2], values[3], values[4], values[5]


def _stage(value: int, stage: int) -> float:
    numerator, denominator = STAGE_RATIOS[max(0, min(12, stage))]
    return value * numerator / denominator


def estimate_damage(level: int, attack: float, defense: float, power: int, stab: bool, multiplier: float,
                    burned: bool = False) -> tuple[float, float]:
    """(min, max) damage of the Gen-3 formula, ignoring crits (fast public estimate)."""
    if power <= 0 or multiplier <= 0:
        return 0.0, 0.0
    base = attack * power * (2 * level // 5 + 2) / max(1.0, defense) / 50
    if burned:
        base /= 2
    base = (base + 2) * (1.5 if stab else 1.0) * multiplier
    return base * 0.85, base


class Perspective:
    """Everything one side may know about the current position."""
    __slots__ = ("battle", "side", "knowledge", "dex", "me", "foe", "foe_stats", "foe_hp")

    def __init__(self, battle: Battle, side: int, knowledge: SideKnowledge) -> None:
        self.battle, self.side, self.knowledge, self.dex = battle, side, knowledge, battle.dex
        self.me = battle.sides[side].fighter
        self.foe = battle.sides[1 - side].fighter
        self.foe_stats = public_stats(self.dex, self.foe.spec.species, self.foe.spec.level)
        self.foe_hp = self.foe.hp / max(1, self.foe.max_hp)  # the HP bar is public

    def believed_multiplier(self, move_type: int, defender_types: tuple[int, int]) -> tuple[float, bool]:
        value = self.knowledge.multiplier(move_type, defender_types)
        return (1.0, False) if value is None else (value, True)

    def move_damage(self, attacker: Fighter, move: MoveData, own_attacker: bool) -> tuple[float, float, float, bool]:
        """(min, max) damage as a fraction of the target's max HP, multiplier and whether it is known."""
        if own_attacker:
            defender_types = self.foe.types
            special = move.special
            attack = _stage(attacker.stat(4 if special else 1), attacker.stages[SPA if special else ATK])
            defense = _stage(self.foe_stats[5 if special else 2], self.foe.stages[SPD if special else DEF])
            target_max = self.foe_stats[0]
        else:
            defender_types = self.me.types
            special = move.special
            stats = public_stats(self.dex, attacker.spec.species, attacker.spec.level)
            attack = _stage(stats[4 if special else 1], attacker.stages[SPA if special else ATK])
            defense = _stage(self.me.stat(5 if special else 2), self.me.stages[SPD if special else DEF])
            target_max = self.me.max_hp
        multiplier, known = self.believed_multiplier(move.move_type, defender_types)
        power = move.power if move.effect not in (41, 130, 87, 40) else 0
        low, high = estimate_damage(attacker.spec.level, attack, defense, power, move.move_type in attacker.types,
                                    multiplier, bool(attacker.status & BURN) and not special)
        if move.effect in (41, 130, 87):
            fixed = {41: 40, 130: 20}.get(move.effect, attacker.spec.level)
            low = high = float(fixed) if multiplier > 0 else 0.0
        elif move.effect == 40:
            low = high = (self.foe.hp if own_attacker else self.me.hp) / 2
        if move.effect in (29,):
            low, high = low * 2, high * 3
        elif move.effect in (44, 77):
            low, high = low * 2, high * 2
        return low / max(1, target_max), high / max(1, target_max), multiplier, known

    def threat(self, defender: Fighter) -> float:
        """Largest believed damage the visible foe can deal to ``defender`` (fraction of its current HP)."""
        foe = self.foe
        if not foe.alive or not defender.alive:
            return 0.0
        revealed = self.knowledge.revealed_moves.get(foe.spec.species, set())
        stats = public_stats(self.dex, foe.spec.species, foe.spec.level)
        candidates: list[tuple[int, int]] = [(self.dex.moves[move].move_type, self.dex.moves[move].power)
                                             for move in revealed if self.dex.moves[move].power]
        if not candidates:
            candidates = [(move_type, _PRIOR_POWER) for move_type in set(foe.types)]
        best = 0.0
        for move_type, power in candidates:
            special = move_type >= 10
            multiplier, _ = self.believed_multiplier(move_type, defender.types)
            attack = _stage(stats[4 if special else 1], foe.stages[SPA if special else ATK])
            defense = _stage(defender.stat(5 if special else 2), defender.stages[SPD if special else DEF]) \
                if defender is self.me else defender.stat(5 if special else 2)
            _, high = estimate_damage(foe.spec.level, attack, defense, power, move_type in foe.types, multiplier)
            best = max(best, high * 0.92)
        return min(3.0, best / max(1, defender.hp))

    def speed_edge(self, fighter: Fighter) -> float:
        own = _stage(fighter.stat(3), fighter.stages[SPE]) * (0.25 if fighter.status & PARALYSIS else 1)
        foe = _stage(self.foe_stats[3], self.foe.stages[SPE]) * (0.25 if self.foe.status & PARALYSIS else 1)
        return 1.0 if own > foe else 0.5 if own == foe else 0.0


def option_features(battle: Battle, side: int, knowledge: SideKnowledge,
                    actions: Sequence[int]) -> NDArray[np.float32]:
    view = Perspective(battle, side, knowledge)
    own = battle.sides[side]
    foe_side = battle.sides[1 - side]
    me, foe = view.me, view.foe
    rows = np.zeros((len(actions), OPTION_FEATURES), dtype=np.float32)
    own_left = own.remaining() / len(own.party)
    foe_left = foe_side.remaining() / len(foe_side.party)
    threat_now = view.threat(me) if me.alive else 0.0
    bench = own.bench()
    for row, action in enumerate(actions):
        out = rows[row]
        out[15], out[16] = me.hp / max(1, me.max_hp), view.foe_hp
        out[19], out[20], out[24] = own_left, foe_left, min(1.0, battle.turn / 50)
        if action < 4:
            out[0] = 1.0
            if not any(pp > 0 for pp in me.pp) or action >= len(me.move_ids):
                move = battle.dex.moves[165]
            else:
                move = battle.dex.moves[me.move_ids[action]]
            low, high, multiplier, known = view.move_damage(me, move, own_attacker=True)
            current = max(1e-6, view.foe_hp)
            out[1] = min(3.0, (low + high) / 2 / current)
            out[2] = min(1.5, (low + high) / 2)
            out[3] = 0.0 if high <= 0 else float(np.clip((high - current) / max(1e-6, high - low), 0.0, 1.0)) \
                if high > low else float(high >= current)
            out[4] = 1.0 if move.accuracy == 0 else move.accuracy / 100
            out[5] = float(known)
            out[6] = -1.5 if multiplier == 0 else float(np.log2(multiplier)) / 2
            out[7] = float(not move.power)
            effect = move.effect
            inflicts = effect in _STATUS_EFFECTS and not foe.status and foe.alive
            if inflicts and effect == 67 and view.believed_multiplier(13, foe.types)[0] == 0:
                inflicts = False
            out[8] = float(inflicts and move.target != TARGET_USER)
            out[9] = float(effect in _SELF_BOOST) * max(0.0, 1 - (sum(me.stages[1:6]) - 30) / 6)
            out[10] = float(effect in _FOE_DROP)
            out[11] = float(effect in _HEALS) * (1 - out[15])
            out[12] = float(move.priority > 0)
            slot_pp = me.pp[action] if action < len(me.pp) else 0
            out[13] = slot_pp / max(1, battle.dex.moves[me.move_ids[action]].pp) if action < len(me.move_ids) else 0
            out[17] = view.speed_edge(me)
            out[18] = threat_now
            out[21] = float(effect in _COMMIT)
        else:
            candidate = own.party[bench[action - 4]]
            out[14] = 1.0
            out[17] = view.speed_edge(candidate)
            out[18] = view.threat(candidate)
            best = 0.0
            for move_id in candidate.move_ids:
                move = battle.dex.moves[move_id]
                if move.power:
                    low, high, _, _ = view.move_damage(candidate, move, own_attacker=True)
                    best = max(best, (low + high) / 2)
            out[22] = min(1.5, best / max(1e-6, view.foe_hp)) if foe.alive else 0.0
            out[23] = candidate.hp / max(1, candidate.max_hp)
            out[4] = 1.0
    return rows


# ------------------------------------------------------------ deep policy
def _entity_row(fighter: Fighter, dex: Dex, own: bool, knowledge: SideKnowledge) -> list[float]:
    """The campaign's 48-field layout (core.observations._active_entity) for stadium fighters."""
    row = [0.0] * 48
    row[:6] = [fighter.spec.species / 411, fighter.spec.level / 100, fighter.hp / max(1, fighter.max_hp),
               fighter.types[0] / 17, fighter.types[1] / 17, float(bool(fighter.status))]
    row[6:14] = [(stage - 6) / 6 for stage in fighter.stages]
    if own:
        stats = fighter.spec.stats
        row[14:19] = [stats[1] / 1024, stats[2] / 1024, stats[3] / 1024, stats[4] / 1024, stats[5] / 1024]
        row[19] = fighter.ability / 80
        row[20] = fighter.item / 400
        for index, move_id in enumerate(fighter.move_ids[:4]):
            move = dex.moves[move_id]
            row[21 + index] = move_id / 354
            row[25 + index] = fighter.pp[index] / max(1, move.pp)
            row[29 + index] = move.power / 150
            row[33 + index] = (move.accuracy or 100) / 100
            row[37 + index] = move.move_type / 17
            row[41 + index] = float(move.special and move.power > 0)
    else:
        row[19] = knowledge.revealed_abilities.get(fighter.spec.species, 0) / 80
        for index, move_id in enumerate(sorted(knowledge.revealed_moves.get(fighter.spec.species, set()))[:4]):
            row[21 + index] = move_id / 354
    # Fields 45-47 (unused by the campaign layout) carry learnt matchup facts.
    return row


def _party_row(fighter: Fighter) -> list[float]:
    row = [0.0] * 48
    stats = fighter.spec.stats
    row[:10] = [fighter.spec.species / 411, fighter.spec.level / 100, fighter.hp / max(1, fighter.max_hp),
                stats[1] / 1024, stats[2] / 1024, stats[3] / 1024, stats[4] / 1024, stats[5] / 1024,
                float(bool(fighter.status)), 0.0]
    return row


def net_observation(battle: Battle, side: int, knowledge: SideKnowledge,
                    actions: Sequence[int]) -> tuple[list[list[float]], list[int], list[bool], list[bool]]:
    """Entities, roles, action mask and entity mask in the campaign policy's format."""
    own, foe = battle.sides[side], battle.sides[1 - side]
    rows = [_entity_row(own.fighter, battle.dex, True, knowledge), _entity_row(foe.fighter, battle.dex, False, knowledge)]
    roles = [0, 2]
    present = [own.fighter.alive or True, foe.fighter.alive]
    for index in own.bench()[:5]:
        rows.append(_party_row(own.party[index]))
        roles.append(4)
        present.append(True)
    while len(rows) < 7:
        rows.append([0.0] * 48)
        roles.append(4)
        present.append(False)
    mask = [False] * 9
    for action in actions:
        mask[action] = True
    if not present[1]:
        present[1] = False
    return rows, roles, mask, present


def batch_observations(observations: Sequence[tuple[list[list[float]], list[int], list[bool], list[bool]]]
                       ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    import torch

    entities = torch.tensor([item[0] for item in observations], dtype=torch.float32)
    roles = torch.tensor([item[1] for item in observations], dtype=torch.long)
    masks = torch.tensor([item[2] for item in observations], dtype=torch.bool)
    present = torch.tensor([item[3] for item in observations], dtype=torch.bool)
    targets = torch.zeros(len(observations), 3, dtype=torch.bool)
    targets[:, 0] = True
    return entities, roles, masks, targets, present


__all__ = ["ACC", "OPTION_FEATURES", "FEATURE_NAMES", "Perspective", "batch_observations", "estimate_damage",
           "net_observation", "option_features", "public_stats", "SLEEP_MASK"]
