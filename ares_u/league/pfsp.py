"""Prioritized fictitious self-play (AlphaStar, Vinyals et al. 2019): play the opponents you struggle with.

``PayoffTable`` keeps the learner's (exponentially decayed) results against every league member, with
a weak prior of ``prior_games`` battles at 0.5 so a new member starts as "unknown, probably even".
``pfsp_weights`` maps each member's estimated win probability ``p`` of the learner to
``f(p) = (1 - p) ** power`` (``power = 2`` is AlphaStar's "hard" weighting) plus a small ``floor``, so
the opponents the learner already beats keep a trickle of games (no forgetting) while the hard ones
dominate. ``distribution`` adds the self-play share.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PayoffTable:
    prior_games: float = 4.0
    decay: float = 0.98
    points: dict[str, float] = field(default_factory=dict)
    games: dict[str, float] = field(default_factory=dict)
    total: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, prior: float = 0.5) -> None:
        if name not in self.points:
            self.points[name] = self.prior_games * prior
            self.games[name] = self.prior_games
            self.total[name] = 0

    def remove(self, name: str) -> None:
        for table in (self.points, self.games, self.total):
            table.pop(name, None)

    def update(self, results: Iterable[tuple[str, float]]) -> None:
        """Decay every entry once, then add this batch's outcomes (+1 win, 0 draw, -1 loss)."""
        for name in self.points:
            self.points[name] *= self.decay
            self.games[name] *= self.decay
        for name, outcome in results:
            if name not in self.points:
                self.add(name)
            self.points[name] += (float(outcome) + 1) / 2
            self.games[name] += 1.0
            self.total[name] += 1

    def win_rate(self, name: str) -> float:
        games = self.games.get(name, 0.0)
        return self.points.get(name, 0.0) / games if games > 0 else 0.5

    def rates(self, names: Iterable[str] | None = None) -> dict[str, float]:
        return {name: self.win_rate(name) for name in (names if names is not None else self.points)}

    def state(self) -> dict[str, Any]:
        return {"points": dict(self.points), "games": dict(self.games), "total": dict(self.total)}


def pfsp_weights(win_rates: Mapping[str, float], power: float = 2.0, floor: float = 0.02) -> dict[str, float]:
    """Normalised f(p) = (1 - p)^power + floor over the members."""
    raw = {name: (1.0 - min(1.0, max(0.0, rate))) ** power + floor for name, rate in win_rates.items()}
    total = sum(raw.values())
    if total <= 0:
        return {name: 1.0 / len(raw) for name in raw} if raw else {}
    return {name: value / total for name, value in raw.items()}


def distribution(table: PayoffTable, members: Iterable[str], self_share: float, power: float = 2.0,
                 floor: float = 0.02) -> dict[str, float]:
    """Opponent distribution of one iteration: ``self`` plus PFSP over the members."""
    names = list(members)
    weights = pfsp_weights(table.rates(names), power, floor) if names else {}
    share = self_share if names else 1.0
    result = {"self": share} if share > 0 else {}
    result.update({name: (1.0 - share) * weight for name, weight in weights.items()})
    return result
