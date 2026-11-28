"""End-to-end Stadium smoke test: full battles brain vs heuristics through the evaluation task."""
from __future__ import annotations

from typing import Any

import pytest

from ares_u.brain.brain import DEFAULT_CHECKPOINT
from ares_u.brain.eval_tasks import handle


def play(agent: dict[str, Any], split: str = "gen2", battles: int = 6) -> list[dict[str, Any]]:
    task = {"type": "evaluate", "split": split, "indices": list(range(battles)), "seed": 0, "agent": agent,
            "opponent": {"heuristic": "greedy"}, "concurrency": battles}
    result: list[dict[str, Any]] = handle(task, {})
    return result


def check(records: list[dict[str, Any]], battles: int) -> None:
    assert [row["index"] for row in records] == list(range(battles))
    assert all(row["score"] in (0.0, 0.5, 1.0) and row["turns"] > 0 for row in records)


def test_strategy_plays_full_battles() -> None:
    check(play({"brain": "strategy", "name": "strategy"}, "gen3"), 6)


@pytest.mark.skipif(not DEFAULT_CHECKPOINT.exists(), reason="E1 checkpoint not available")
def test_brain_plays_full_battles_and_is_deterministic() -> None:
    agent = {"brain": "brain", "checkpoint": str(DEFAULT_CHECKPOINT), "weight": 2.0, "name": "brain"}
    first, second = play(agent), play(agent)
    check(first, 6)
    assert [row["score"] for row in first] == [row["score"] for row in second]


def test_campaign_heuristic_port_plays() -> None:
    check(play({"campaign": True}, "gen1_heldout", 4), 4)
