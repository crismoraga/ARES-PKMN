"""Teams harvested from the real FireRed/Emerald campaign checkpoints.

Every checkpoint's ``emulator.state`` is loaded read-only into a native mGBA
handle and its encrypted party is decoded (species, level, stats, IVs/EVs,
moves, held item, ability bit). Each Pokemon is identified by its personality
value, so the roster of a side is "every Pokemon its campaign agent ever
raised", at the most advanced snapshot seen. No emulator frame is stepped and
no RAM is written: the campaign saves are only read.

Stadium rules can normalise every member to one level (default 50), recomputing
the exact Gen-3 stats from base stats, IVs, EVs and nature, so a FireRed team
from Mt. Moon can still be compared with an Emerald team from Mauville.
"""
from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.game_data import ROM_NAMES, ROOT

from .dex import Dex, Game
from .engine import MonSpec

CAMPAIGN_ROOT = ROOT / "artifacts" / "campaigns" / "native"
BANK_PATH = ROOT / "artifacts" / "stadium" / "team_bank.json"


@dataclass(frozen=True)
class RosterMon:
    """One raised Pokemon: what stats/moves it had and how it can be rebuilt at any level."""
    pid: int
    species: int
    level: int
    stats: tuple[int, int, int, int, int, int]
    moves: tuple[int, ...]
    ivs: tuple[int, int, int, int, int, int]
    evs: tuple[int, int, int, int, int, int]
    ability: int
    item: int
    friendship: int
    name: str
    origin: str

    def to_json(self) -> dict[str, Any]:
        return {"pid": self.pid, "species": self.species, "level": self.level, "stats": list(self.stats),
                "moves": list(self.moves), "ivs": list(self.ivs), "evs": list(self.evs), "ability": self.ability,
                "item": self.item, "friendship": self.friendship, "name": self.name, "origin": self.origin}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> RosterMon:
        def six(values: Sequence[Any]) -> tuple[int, int, int, int, int, int]:
            items = [int(value) for value in values]
            return items[0], items[1], items[2], items[3], items[4], items[5]
        return cls(int(raw["pid"]), int(raw["species"]), int(raw["level"]), six(raw["stats"]),
                   tuple(int(move) for move in raw["moves"]), six(raw["ivs"]), six(raw["evs"]), int(raw["ability"]),
                   int(raw["item"]), int(raw["friendship"]), str(raw["name"]), str(raw["origin"]))

    def species_at(self, dex: Dex, level: int | None) -> int:
        """Evolution projection: what this Pokemon becomes when levelled to the stadium level."""
        target = self.level if level is None else level
        if target <= self.level:
            return self.species
        return dex.evolve(self.species, target, self.pid, self.friendship, self.stats)

    def spec(self, dex: Dex, level: int | None = None, moves: Sequence[int] | None = None) -> MonSpec:
        """A battle-ready member; ``level`` evolves it if due and recomputes exact Gen-3 stats."""
        target = self.level if level is None else level
        species = self.species_at(dex, level)
        if target == self.level and species == self.species:
            stats = self.stats
        else:
            stats = gen3_stats(dex, species, target, self.ivs, self.evs, self.pid % 25)
        chosen = tuple(int(move) for move in (moves if moves is not None else self.moves) if int(move))[:4]
        ability = self.ability
        if species != self.species:
            old, new = dex.species[self.species].abilities, dex.species[species].abilities
            slot = 1 if old[1] and self.ability == old[1] and new[1] else 0
            ability = new[slot] or new[0]
        return MonSpec(species, target, stats, chosen or self.moves[:4], ability, self.item,
                       self.friendship, self.ivs, dex.species[species].name, self.origin)

    def move_pool(self, dex: Dex, level: int | None = None) -> tuple[int, ...]:
        """Known moves (TMs/HMs included) plus level-up moves of its line up to the stadium level."""
        target = max(self.level, level or self.level)
        pool = list(self.moves)
        for species in dict.fromkeys((self.species, self.species_at(dex, target))):
            for move in dex.learnable(species, target):
                if move not in pool and move and dex.moves[move].pp:
                    pool.append(move)
        return tuple(pool)

    def best_moves(self, dex: Dex, level: int | None = None) -> tuple[int, ...]:
        """A sensible default set at the stadium level: strong STAB plus type coverage."""
        species = self.species_at(dex, level)
        types = dex.species[species].types
        pool = self.move_pool(dex, level)

        def value(move: int) -> float:
            data = dex.moves[move]
            if not data.power:
                return 25.0 if move in self.moves else 5.0
            return data.power * (1.5 if data.move_type in types else 1.0) * (data.accuracy or 100) / 100

        ranked = sorted(pool, key=value, reverse=True)
        chosen: list[int] = []
        seen_types: set[int] = set()
        for move in ranked:  # prefer type coverage among the top attacks
            data = dex.moves[move]
            if data.power and data.move_type in seen_types and len(ranked) > 4:
                continue
            chosen.append(move)
            if data.power:
                seen_types.add(data.move_type)
            if len(chosen) == 4:
                break
        for move in ranked:
            if len(chosen) == 4:
                break
            if move not in chosen:
                chosen.append(move)
        return tuple(chosen)


def gen3_stats(dex: Dex, species: int, level: int, ivs: Sequence[int], evs: Sequence[int],
               nature: int) -> tuple[int, int, int, int, int, int]:
    """The native CalculateMonStats formula (HP, then nature-adjusted stats)."""
    base = dex.species[species].base_stats
    values: list[int] = []
    for index in range(6):
        core = (2 * base[index] + ivs[index] + evs[index] // 4) * level // 100
        if index == 0:
            values.append(1 if species == 292 else core + level + 10)  # Shedinja always has 1 HP
            continue
        stat = core + 5
        if nature // 5 != nature % 5:
            if index - 1 == nature // 5:
                stat = stat * 110 // 100
            elif index - 1 == nature % 5:
                stat = stat * 90 // 100
        values.append(stat)
    return values[0], values[1], values[2], values[3], values[4], values[5]


@dataclass
class TeamBank:
    """Rosters per game plus the exact party of each checkpoint (historical teams)."""
    rosters: dict[str, list[RosterMon]] = field(default_factory=dict)
    parties: dict[str, list[list[int]]] = field(default_factory=dict)  # pids per snapshot
    sources: dict[str, int] = field(default_factory=dict)

    def roster(self, game: str) -> list[RosterMon]:
        return self.rosters.get(game, [])

    def to_json(self) -> dict[str, Any]:
        return {"version": 1, "rosters": {game: [mon.to_json() for mon in mons] for game, mons in self.rosters.items()},
                "parties": self.parties, "sources": self.sources}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> TeamBank:
        if raw.get("version") != 1:
            raise ValueError("unsupported team bank version")
        return cls({game: [RosterMon.from_json(row) for row in rows] for game, rows in raw["rosters"].items()},
                   {game: [list(map(int, party)) for party in parties] for game, parties in raw["parties"].items()},
                   {game: int(count) for game, count in raw.get("sources", {}).items()})

    def save(self, path: Path = BANK_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.to_json()), encoding="utf-8")
        temporary.replace(path)
        return path


def _record_from_party(mon: dict[str, Any], dex: Dex, origin: str) -> RosterMon | None:
    if mon.get("is_egg") or not mon.get("species") or not 0 < int(mon["species"]) < len(dex.species):
        return None
    moves = tuple(int(move) for move in mon["moves"] if int(move))
    if not moves:
        return None
    species = dex.species[int(mon["species"])]
    ability_slot = int(mon.get("ability_num", 0)) & 1
    ability = species.abilities[ability_slot] or species.abilities[0]
    ivs = tuple(int(value) for value in mon.get("ivs", [15] * 6))
    evs = tuple(int(value) for value in mon.get("evs", [0] * 6))
    stats = (int(mon["maxHP"]), int(mon["attack"]), int(mon["defense"]), int(mon["speed"]),
             int(mon["spAttack"]), int(mon["spDefense"]))
    if min(stats) < 1:
        return None
    return RosterMon(int(mon["pid"]), int(mon["species"]), int(mon["level"]), stats, moves,
                     (ivs[0], ivs[1], ivs[2], ivs[3], ivs[4], ivs[5]), (evs[0], evs[1], evs[2], evs[3], evs[4], evs[5]),
                     ability, int(mon.get("held_item", 0)), int(mon.get("friendship", 70)), species.name, origin)


def harvest(game: Game, dex: Dex, root: Path = CAMPAIGN_ROOT, limit: int | None = None) -> tuple[list[RosterMon], list[list[int]], int]:
    """Read every checkpoint party of ``game`` (read-only native loads)."""
    from core.bridge.ffi_mgba import NativeLibMGBABridge  # native only when harvesting

    directory = root / game
    checkpoints = sorted((path for path in directory.glob("frame-*") if (path / "emulator.state").is_file()),
                         key=lambda path: path.name)
    if limit is not None:
        checkpoints = checkpoints[-limit:]
    best: dict[int, RosterMon] = {}
    parties: list[list[int]] = []
    seen_parties: set[tuple[int, ...]] = set()
    if not checkpoints:
        return [], [], 0
    bridge = NativeLibMGBABridge(ROOT / "native" / "build" / "ares_mgba.dll", ROOT.parent / ROM_NAMES[game],
                                 game == "emerald")
    loaded = 0
    try:
        for checkpoint in checkpoints:
            try:
                bridge.load_state((checkpoint / "emulator.state").read_bytes())
                party = bridge.get_decrypted_player_party()
            except (ValueError, RuntimeError, OSError):
                continue
            loaded += 1
            frame = checkpoint.name.split("-")[1] if "-" in checkpoint.name else checkpoint.name
            pids: list[int] = []
            for mon in party:
                record = _record_from_party(mon, dex, f"{game}@{int(frame)}")
                if record is None:
                    continue
                pids.append(record.pid)
                current = best.get(record.pid)
                if current is None or (record.level, len(record.moves)) >= (current.level, len(current.moves)):
                    best[record.pid] = record
            key = tuple(sorted(pids))
            if pids and key not in seen_parties:
                seen_parties.add(key)
                parties.append(pids)
    finally:
        bridge.close()
    roster = sorted(best.values(), key=lambda mon: (-mon.level, mon.species, mon.pid))
    return roster, parties, loaded


def load_bank(dex: Dex, refresh: bool = False, path: Path = BANK_PATH,
              games: Iterable[Game] = ("firered", "emerald")) -> TeamBank:
    """Cached bank; ``refresh`` re-reads the checkpoints (new campaign progress)."""
    if path.is_file() and not refresh:
        try:
            return TeamBank.from_json(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, KeyError, OSError):
            pass
    bank = TeamBank()
    for game in games:
        roster, parties, loaded = harvest(game, dex)
        if roster:
            bank.rosters[game], bank.parties[game], bank.sources[game] = roster, parties, loaded
    if not bank.rosters:
        bank = fallback_bank(dex)
    bank.save(path)
    return bank


def fallback_bank(dex: Dex) -> TeamBank:
    """Starter-line rosters for machines without campaign checkpoints (tests, CI)."""
    lines = {"firered": (3, 18, 26, 34, 65, 130, 143, 59), "emerald": (260, 254, 257, 282, 334, 306, 310, 350)}
    bank = TeamBank()
    for game, species_ids in lines.items():
        roster: list[RosterMon] = []
        for index, species in enumerate(species_ids):
            level = 45
            moves = tuple(move for move in dex.learnable(species, level) if dex.moves[move].pp)[-4:]
            ivs = (20, 20, 20, 20, 20, 20)
            stats = gen3_stats(dex, species, level, ivs, (0,) * 6, index * 3 % 25)
            data = dex.species[species]
            roster.append(RosterMon(index * 3 + 1000 * (game == "emerald"), species, level, stats, moves, ivs,
                                    (0, 0, 0, 0, 0, 0), data.abilities[0], 0, 70, data.name, f"{game}@fallback"))
        bank.rosters[game] = roster
        bank.parties[game] = [[mon.pid for mon in roster[:6]]]
        bank.sources[game] = 0
    return bank


# ------------------------------------------------------------------ team genes
@dataclass(frozen=True)
class TeamGenes:
    """Which roster members (in lead order) and which four moves each one uses."""
    members: tuple[int, ...]           # indexes into the side's roster
    moves: tuple[tuple[int, ...], ...]  # chosen moves per member

    def to_json(self) -> dict[str, Any]:
        return {"members": list(self.members), "moves": [list(moves) for moves in self.moves]}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> TeamGenes:
        return cls(tuple(int(index) for index in raw["members"]),
                   tuple(tuple(int(move) for move in moves) for moves in raw["moves"]))

    def build(self, roster: Sequence[RosterMon], dex: Dex, level: int | None) -> list[MonSpec]:
        return [roster[index].spec(dex, level, moves) for index, moves in zip(self.members, self.moves)]


def initial_genes(roster: Sequence[RosterMon], parties: Sequence[Sequence[int]], rng: random.Random,
                  size: int = 6, dex: Dex | None = None, level: int | None = None) -> TeamGenes:
    """A real campaign party when available (the agent's own choice), else a random draw."""
    by_pid = {mon.pid: index for index, mon in enumerate(roster)}
    candidates = [[by_pid[pid] for pid in party if pid in by_pid] for party in parties]
    candidates = [party for party in candidates if party]
    if candidates and rng.random() < 0.7:
        members = list(rng.choice(candidates))[:size]
    else:
        members = rng.sample(range(len(roster)), min(size, len(roster)))
    def moves_of(index: int) -> tuple[int, ...]:
        if dex is not None and rng.random() < 0.8:
            return roster[index].best_moves(dex, level)
        return roster[index].moves[:4]

    return TeamGenes(tuple(members), tuple(moves_of(index) for index in members))


def mutate_genes(genes: TeamGenes, roster: Sequence[RosterMon], dex: Dex, rng: random.Random,
                 rate: float = 0.35, size: int = 6, level: int | None = None) -> TeamGenes:
    """Legal edits only: lead order, roster swaps and moves from each member's legal pool."""
    members = list(genes.members)
    moves = [list(chosen) for chosen in genes.moves]
    if rng.random() < rate and len(members) > 1:  # new lead / order
        a, b = rng.sample(range(len(members)), 2)
        members[a], members[b] = members[b], members[a]
        moves[a], moves[b] = moves[b], moves[a]
    if rng.random() < rate * 0.6 and len(roster) > len(members):  # bring another raised Pokemon
        unused = [index for index in range(len(roster)) if index not in members]
        slot = rng.randrange(len(members))
        replacement = rng.choice(unused)
        members[slot] = replacement
        moves[slot] = list(roster[replacement].moves[:4])
    if rng.random() < rate * 0.4 and len(members) < size and len(roster) > len(members):
        unused = [index for index in range(len(roster)) if index not in members]
        replacement = rng.choice(unused)
        members.append(replacement)
        moves.append(list(roster[replacement].moves[:4]))
    if rng.random() < rate:  # swap one move for another legal one
        slot = rng.randrange(len(members))
        pool = [move for move in roster[members[slot]].move_pool(dex, level) if move not in moves[slot]]
        if pool:
            if len(moves[slot]) < 4:
                moves[slot].append(rng.choice(pool))
            else:
                moves[slot][rng.randrange(len(moves[slot]))] = rng.choice(pool)
    return TeamGenes(tuple(members), tuple(tuple(chosen) for chosen in moves))
