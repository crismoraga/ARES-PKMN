"""Random 3v3 level-50 teams drawn from per-generation species pools.

Species indexing: the dex uses the *internal* Gen-3 species order, not national
Pokedex numbers. Internal ids 1-251 coincide with national numbers (Gen 1 and
Gen 2), internal ids 252-276 are 25 unused placeholder slots (named ``?``) and
the 135 Hoenn species occupy internal ids 277-411 in Hoenn order (e.g. TREECKO
is internal 277 / national 252; CHIMECHO is internal 411 / national 358). So
the national range #252-386 is exactly the *set* of internal ids 277-411.

Each member gets neutral IVs (31), no EVs, a random nature and ability, and up
to four legal level-up moves known by level 50 whose effect the engine models
(``core.stadium.engine._MODELLED``). Teams are deterministic given their seed.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from core.stadium.dex import Dex, MoveData, load_dex
from core.stadium.engine import _MODELLED, MOVE_STRUGGLE, MonSpec
from core.stadium.teams import gen3_stats

Split = Literal["gen1_train", "gen1_heldout", "gen2", "gen3"]
SPLITS: tuple[Split, ...] = ("gen1_train", "gen1_heldout", "gen2", "gen3")
# ``gen1_train`` evaluation teams come from the training distribution (Gen1 species sets that are
# not held out), so ``gen1_train - split`` measures the generalisation gap.
EVAL_SPLITS: tuple[Split, ...] = ("gen1_train", "gen1_heldout", "gen2", "gen3")
LEVEL = 50
TEAM_SIZE = 3
GEN3_INTERNAL_FIRST = 277
GEN3_INTERNAL_LAST = 411
PLACEHOLDER_FIRST, PLACEHOLDER_LAST = 252, 276
# Excluded on purpose: SHEDINJA (1 HP + Wonder Guard makes random battles degenerate).
EXCLUDED_NAMES = frozenset({"SHEDINJA"})
HELDOUT_SIGNATURES = 4096  # held-out Gen1 species sets that training never samples


def generation_of(internal_id: int) -> int:
    """Generation (1, 2 or 3) of an internal species id; 0 for the empty/placeholder slots."""
    if 1 <= internal_id <= 151:
        return 1
    if 152 <= internal_id <= 251:
        return 2
    if GEN3_INTERNAL_FIRST <= internal_id <= GEN3_INTERNAL_LAST:
        return 3
    return 0


def usable_move(move: MoveData) -> bool:
    return move.move_id not in (0, MOVE_STRUGGLE) and move.pp > 0 and move.effect in _MODELLED


def legal_moves(dex: Dex, species: int, level: int = LEVEL) -> tuple[int, ...]:
    """Level-up moves known by ``level`` that the engine models."""
    return tuple(move for move in dex.learnable(species, level) if usable_move(dex.moves[move]))


def eligible(dex: Dex, species: int) -> bool:
    data = dex.species[species]
    if data.name in EXCLUDED_NAMES or data.name == "?" or generation_of(species) == 0:
        return False
    return any(dex.moves[move].power > 0 for move in legal_moves(dex, species))


_POOL_CACHE: dict[int, dict[int, tuple[int, ...]]] = {}
_SIGNATURE_CACHE: dict[int, frozenset[frozenset[int]]] = {}


def species_pool(dex: Dex, generation: int) -> tuple[int, ...]:
    """Eligible internal species ids of one generation (cached per dex object)."""
    pools = _POOL_CACHE.get(id(dex))
    if pools is None:
        found: dict[int, list[int]] = {1: [], 2: [], 3: []}
        for species in range(len(dex.species)):
            generation_id = generation_of(species)
            if generation_id and eligible(dex, species):
                found[generation_id].append(species)
        pools = {key: tuple(members) for key, members in found.items()}
        _POOL_CACHE[id(dex)] = pools
    return pools[generation]


def split_generation(split: Split) -> int:
    return {"gen1_train": 1, "gen1_heldout": 1, "gen2": 2, "gen3": 3}[split]


def choose_moves(dex: Dex, species: int, rng: random.Random, level: int = LEVEL) -> tuple[int, ...]:
    """Up to four moves: damaging moves first (power-weighted, type-diverse), at most one status move."""
    pool = list(legal_moves(dex, species, level))
    damaging = [move for move in pool if dex.moves[move].power > 0]
    status = [move for move in pool if dex.moves[move].power == 0]
    chosen: list[int] = []
    types_seen: set[int] = set()
    want_damaging = 3 if status else 4
    candidates = list(damaging)
    while candidates and len(chosen) < want_damaging:
        weights = []
        for move in candidates:
            data = dex.moves[move]
            power = 60 if data.power == 1 else data.power
            weight = max(10, power) * (0.35 if data.move_type in types_seen else 1.0)
            weights.append(float(weight))
        pick = rng.choices(candidates, weights=weights, k=1)[0]
        candidates.remove(pick)
        chosen.append(pick)
        types_seen.add(dex.moves[pick].move_type)
    if status and len(chosen) < 4:
        chosen.append(rng.choice(status))
        status.remove(chosen[-1])
    leftovers = candidates + status
    rng.shuffle(leftovers)
    while leftovers and len(chosen) < 4:
        chosen.append(leftovers.pop())
    return tuple(chosen)


def build_member(dex: Dex, species: int, rng: random.Random, level: int = LEVEL) -> MonSpec:
    data = dex.species[species]
    nature = rng.randrange(25)
    abilities = [ability for ability in data.abilities if ability] or [0]
    ability = rng.choice(abilities)
    stats = gen3_stats(dex, species, level, (31,) * 6, (0,) * 6, nature)
    return MonSpec(species, level, stats, choose_moves(dex, species, rng, level), ability, 0, 70, (31,) * 6,
                   data.name, "ares_u")


def random_team(dex: Dex, pool: Sequence[int], rng: random.Random, size: int = TEAM_SIZE,
                forbidden: frozenset[frozenset[int]] = frozenset()) -> list[MonSpec]:
    for _ in range(1000):
        members = rng.sample(list(pool), size)
        if frozenset(members) not in forbidden:
            return [build_member(dex, species, rng) for species in members]
    raise RuntimeError("could not draw a team outside the forbidden species sets")


def _team_rng(split: Split, index: int, side: int, namespace: str) -> random.Random:
    return random.Random(f"ares_u/{namespace}/{split}/{index}/{side}")


def heldout_signatures(dex: Dex, count: int = HELDOUT_SIGNATURES) -> frozenset[frozenset[int]]:
    """Species sets of the first ``count`` held-out Gen1 teams (both sides); training avoids them."""
    cached = _SIGNATURE_CACHE.get(id(dex))
    if cached is not None and count == HELDOUT_SIGNATURES:
        return cached
    pool = list(species_pool(dex, 1))
    signatures: set[frozenset[int]] = set()
    for index in range(count):
        for side in (0, 1):
            rng = _team_rng("gen1_heldout", index, side, "eval")
            signatures.add(frozenset(rng.sample(pool, TEAM_SIZE)))
    result = frozenset(signatures)
    if count == HELDOUT_SIGNATURES:
        _SIGNATURE_CACHE[id(dex)] = result
    return result


def eval_team(dex: Dex, split: Split, index: int, side: int) -> list[MonSpec]:
    """Deterministic evaluation team ``index`` of ``split`` for ``side``.

    ``gen1_train`` teams follow the training sampler's distribution (never a held-out species set);
    their RNG namespace differs from the training samplers', so they are fresh draws of it.
    """
    pool = species_pool(dex, split_generation(split))
    forbidden = heldout_signatures(dex) if split == "gen1_train" else frozenset()
    return random_team(dex, pool, _team_rng(split, index, side, "eval"), forbidden=forbidden)


@dataclass
class TrainTeamSampler:
    """Training teams: Gen1 species, never one of the held-out species sets."""
    dex: Dex
    rng: random.Random

    def __post_init__(self) -> None:
        self.pool = species_pool(self.dex, 1)
        self.forbidden = heldout_signatures(self.dex)

    def team(self) -> list[MonSpec]:
        return random_team(self.dex, self.pool, self.rng, forbidden=self.forbidden)


def default_dex() -> Dex:
    """The ROM-free snapshot (identical on every machine, so experiments are reproducible)."""
    return load_dex("emerald", prefer_snapshot=True)
