"""A battle wrapper for learning agents: revealed information, rewards and anti-vice counters.

Reward of a learning side (documented for the report):

* terminal: +1 win, -1 loss, 0 draw; at the engine's turn limit the winner is decided by the
  remaining HP fraction (``core/stadium/engine.py`` ``_after_faints``), so it still scores +-1;
* shaping: potential-based, ``r_t += c * (gamma * Phi(s_{t+1}) - Phi(s_t))`` with
  ``Phi = own team HP fraction - foe team HP fraction`` and ``c = SHAPING`` (0.2). Potential-based
  shaping (Ng et al., 1999) does not change the optimal policy; it only densifies the signal.
  ``Phi`` is 0 at the terminal state, so the shaping of an episode telescopes to ``-c * Phi(s_0)``
  plus discounting terms.

A *decision* of a side is recorded only when it has at least two legal actions; rewards that
arrive between two recorded decisions are accumulated into the earlier one.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from core.stadium.dex import Dex
from core.stadium.engine import Battle, MonSpec

from .obs import Observation, OracleKnowledge, encode, resolve_move

SHAPING = 0.2
MAX_TURNS = 150


def hp_margin(battle: Battle, side: int) -> float:
    return battle.sides[side].hp_fraction() - battle.sides[1 - side].hp_fraction()


@dataclass
class SideCounters:
    """Anti-vice statistics of one side over one battle."""
    voluntary: int = 0
    switches: int = 0
    panic_switches: int = 0
    move_choices: int = 0
    ineffective: int = 0
    last_was_switch: bool = False


@dataclass
class BattleEnv:
    dex: Dex
    team_a: Sequence[MonSpec]
    team_b: Sequence[MonSpec]
    seed: int
    max_turns: int = MAX_TURNS
    battle: Battle = field(init=False)
    revealed: tuple[set[int], set[int]] = field(init=False)
    knowledge: tuple[OracleKnowledge, OracleKnowledge] = field(init=False)
    counters: tuple[SideCounters, SideCounters] = field(init=False)

    def __post_init__(self) -> None:
        self.battle = Battle(self.dex, self.team_a, self.team_b, seed=self.seed, max_turns=self.max_turns,
                             record=False)
        self.revealed = ({self.battle.sides[0].active}, {self.battle.sides[1].active})
        self.knowledge = (OracleKnowledge(dex=self.dex), OracleKnowledge(dex=self.dex))
        self.counters = (SideCounters(), SideCounters())

    @property
    def over(self) -> bool:
        return self.battle.over

    def pending(self) -> tuple[int, ...]:
        return () if self.battle.over else self.battle.pending

    def legal(self, side: int) -> list[int]:
        return self.battle.legal_actions(side)

    def observe(self, side: int, legal: list[int] | None = None) -> Observation:
        return encode(self.battle, side, self.revealed, self.knowledge[side], legal)

    def voluntary(self, side: int) -> bool:
        return not self.battle._replacing and len(self.battle.legal_actions(side)) > 1

    def immune_choice(self, side: int, action: int) -> bool:
        """A damaging move into a type immunity of the foe active (visible in the observation)."""
        if action >= 4:
            return False
        fighter = self.battle.sides[side].fighter
        if action >= len(fighter.move_ids):
            return False
        move = resolve_move(self.battle, fighter, self.dex.moves[fighter.move_ids[action]])
        foe = self.battle.sides[1 - side].fighter
        return move.power > 0 and self.dex.multiplier(move.move_type, foe.types) == 0

    def step(self, actions: Mapping[int, int]) -> None:
        for side, action in actions.items():
            if self.voluntary(side):
                counter = self.counters[side]
                counter.voluntary += 1
                if action >= 4:
                    counter.switches += 1
                    counter.panic_switches += int(counter.last_was_switch)
                    counter.last_was_switch = True
                else:
                    counter.move_choices += 1
                    counter.ineffective += int(self.immune_choice(side, action))
                    counter.last_was_switch = False
        on_field = {side: self.battle.sides[side].fighter.spec.species for side in (0, 1)}
        events = self.battle.submit(actions)
        for event in events:
            if event[0] == "switch":
                self.revealed[int(event[1])].add(int(event[2]))
        for side in (0, 1):
            self.knowledge[side].observe(self.dex, events, side, on_field)

    def outcome(self, side: int) -> float:
        """+1 win, -1 loss, 0 draw (only meaningful once the battle is over)."""
        winner = self.battle.winner
        if winner is None or winner == 2:
            return 0.0
        return 1.0 if winner == side else -1.0
