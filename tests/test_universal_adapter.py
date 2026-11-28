"""Universal player: Gen 1 (Rojo) adapter reads on the saved dev state, map transitions settle correctly."""
from __future__ import annotations

from pathlib import Path

import pytest

from ares_u.gal.backends import MULTI_DLL
from ares_u.gal.games import game

ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "artifacts" / "universal" / "dev" / "r1.state"

pytestmark = pytest.mark.skipif(not MULTI_DLL.exists() or not game("rojo").rom.exists() or not R1.exists(),
                                reason="multi-core shim, Rojo ROM or dev state missing")


@pytest.fixture(scope="module")
def adapter():  # type: ignore[no-untyped-def]
    from ares_u.universal.gen1 import Gen1Adapter

    ad = Gen1Adapter(game("rojo"))
    yield ad
    ad.close()


def load_r1(ad) -> None:  # type: ignore[no-untyped-def]
    ad.load_state(R1.read_bytes())
    ad.wait(2)


def test_reads_on_r1_state(adapter) -> None:  # type: ignore[no-untyped-def]
    load_r1(adapter)
    key, x, y = adapter.position()
    assert (key, x, y) == ((0, 38), 5, 1)          # Red's room (2F)
    snap = adapter.snapshot()
    assert snap.mode == "overworld"
    assert snap.party == ()                        # before Oak: no Pokemon yet
    assert snap.badges == 0
    assert snap.money == 3000
    assert adapter.map_name((0, 0)) == "Pueblo Paleta"
    # ROM tables found in the Spanish ROM
    assert adapter.move_names[1] == "DESTRUCTOR"
    assert adapter.species_names[0x99] == "BULBASAUR"


def test_room_grid_and_stairs_warp(adapter) -> None:  # type: ignore[no-untyped-def]
    from ares_u.universal.model import WARP_CELL

    load_r1(adapter)
    info = adapter.map_info()
    assert (info.width, info.height) == (8, 8)
    assert [(w.x, w.y) for w in info.warps] == [(7, 1)]
    assert info.cell(7, 1) == WARP_CELL
    assert info.cell(5, 1) in (1, 4)               # the player stands on floor


def test_warp_waits_for_the_new_map_data(adapter) -> None:  # type: ignore[no-untyped-def]
    """The map id flips ~30 frames before coords/header/blocks load; walk() must return a settled map."""
    load_r1(adapter)
    assert adapter.walk("right") == "moved"
    outcome = adapter.walk("right")
    assert outcome == "warped"
    key, x, y = adapter.position()
    assert key == (0, 37)                          # house 1F
    info = adapter.map_info()
    assert info.key == (0, 37)
    assert info.cell(x, y) in (1, 4, 9)            # standing on a real cell of the *new* map
    assert any(w.dest_map == (0, 0) for w in info.warps)   # front door leads to Pallet
    # signature is stable once settled: a second ensure_settled does not wait
    waits = adapter.settle_waits
    assert adapter.ensure_settled() is False
    assert adapter.settle_waits == waits
