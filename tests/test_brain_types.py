"""Type charts per generation (ares_u.brain.api.type_chart) and name canonicalisation."""
from __future__ import annotations

import json
from pathlib import Path

from ares_u.brain.api import (GEN1_TYPES, GEN2_TYPES, GEN3_INDEX_TYPE, GEN6_TYPES, canonical_type, category_for,
                              type_chart, type_multiplier)

ROOT = Path(__file__).resolve().parents[1]


def test_chart_sizes_per_generation() -> None:
    assert set(type_chart(1)) == set(GEN1_TYPES) and len(GEN1_TYPES) == 15
    assert set(type_chart(2)) == set(GEN2_TYPES) and len(GEN2_TYPES) == 17
    assert set(type_chart(6)) == set(GEN6_TYPES) and len(GEN6_TYPES) == 18
    for generation in (1, 2, 3, 4, 5, 6, 9):
        chart = type_chart(generation)
        assert all(set(row) == set(chart) for row in chart.values())


def test_gen1_quirks() -> None:
    chart = type_chart(1)
    assert "dark" not in chart and "steel" not in chart
    assert chart["ghost"]["psychic"] == 0.0     # RBY bug
    assert chart["bug"]["poison"] == 2.0
    assert chart["poison"]["bug"] == 2.0
    assert chart["ice"]["fire"] == 1.0
    assert chart["ghost"]["normal"] == 0.0 and chart["normal"]["ghost"] == 0.0
    assert chart["electric"]["ground"] == 0.0


def test_gen2_to_5_matches_gen3_rom_table() -> None:
    """Every non-neutral cell of the ROM table (``data/stadium_dex.json``) equals our Gen-3 chart, and
    every cell absent from the ROM table is neutral."""
    rows = json.loads((ROOT / "data" / "stadium_dex.json").read_text(encoding="utf-8"))["types"]
    rom: dict[tuple[str, str], float] = {}
    for attack, defense, factor in rows:
        if 9 in (attack, defense):  # typeless "???"
            continue
        rom[(GEN3_INDEX_TYPE[int(attack)], GEN3_INDEX_TYPE[int(defense)])] = int(factor) / 10.0
    assert len(rom) > 100
    for generation in (2, 3, 4, 5):
        chart = type_chart(generation)
        for attack in GEN2_TYPES:
            for defense in GEN2_TYPES:
                assert chart[attack][defense] == rom.get((attack, defense), 1.0), (generation, attack, defense)


def test_gen6_fairy_and_steel() -> None:
    chart = type_chart(6)
    assert chart["dragon"]["fairy"] == 0.0
    assert chart["fairy"]["dragon"] == 2.0 and chart["fairy"]["fire"] == 0.5
    assert chart["steel"]["fairy"] == 2.0 and chart["poison"]["fairy"] == 2.0
    assert chart["ghost"]["steel"] == 1.0 and chart["dark"]["steel"] == 1.0
    assert type_chart(5)["ghost"]["steel"] == 0.5


def test_dual_types_spanish_names_and_categories() -> None:
    assert type_multiplier(3, "Eléctrico", ("Roca", "Tierra")) == 0.0
    assert type_multiplier(3, "water", ("rock", "ground")) == 4.0
    assert type_multiplier(3, "Fire", ("???",)) == 1.0
    assert type_multiplier(1, "ghost", ("psychic",)) == 0.0 and type_multiplier(2, "ghost", ("psychic",)) == 2.0
    assert canonical_type("Psíquico") == "psychic" and canonical_type("nonsense") == "???"
    assert category_for("dark", 80, 3) == "special" and category_for("dark", 80, 4, "physical") == "physical"
    assert category_for("fire", 0, 3) == "status"
