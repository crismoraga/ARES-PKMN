"""Stable neural encodings with an explicit boundary around hidden enemy data."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, cast

import torch
from torch import Tensor

from core.battle.menu_cursor_controller import native_party_slot_order, native_player_input_callbacks
from core.bridge.ffi_mgba import GBABridgeProtocol
from core.bridge.gba_structs import BattlePokemonStruct
from core.game_data import Game, RomCatalog, RomSymbol, load_symbols


@dataclass
class OpponentKnowledge:
    """Only battle disclosures admitted to policy features and opponent beliefs."""
    revealed_moves: dict[int, list[int]] = field(default_factory=dict)
    revealed_abilities: dict[int, int] = field(default_factory=dict)
    revealed_items: dict[int, int] = field(default_factory=dict)

    def reveal_move(self, species: int, move: int) -> None:
        """Record an observed executed move, never the unseen move array in RAM."""
        if species < 1 or not 1 <= move <= 354:
            raise ValueError("Invalid disclosed species/move")
        moves = self.revealed_moves.setdefault(species, [])
        if move not in moves:
            moves.append(move)


@dataclass(frozen=True)
class BattleFeatures:
    """Fixed batch-first model inputs for one visible battle decision."""
    entities: Tensor
    roles: Tensor
    action_mask: Tensor
    target_mask: Tensor
    padding_mask: Tensor
    acting_battler: int = 0
    active_party_index: int = 0
    switch_party_indices: tuple[int, ...] = ()
    target_battlers: tuple[int, int, int] = (1, 3, 2)
    doubles: bool = False
    forced_switch: bool = False


def _party_features(mon: dict[str, Any]) -> list[float]:
    """Encode ten fully observable own-party state fields."""
    maximum = max(1, int(mon["maxHP"]))
    return [int(mon["species"]) / 411, int(mon["level"]) / 100, int(mon["hp"]) / maximum,
            int(mon["attack"]) / 1024, int(mon["defense"]) / 1024,
            int(mon["speed"]) / 1024, int(mon["spAttack"]) / 1024,
            int(mon["spDefense"]) / 1024, float(bool(mon["status"])),
            int(mon["experience"]) / 2_000_000]


def encode_world_state(bridge: GBABridgeProtocol) -> Tensor:
    """Encode a documented96-vector without enemy-party or type-table access.

    Indices0..5 are coordinates/map/state;6..65 six party records of10 fields;
    66..81 cardinal neighborhood collision/elevation;82..95 inventory/battle.
    """
    features = [0.0] * 96
    position = bridge.get_player_coords_and_map()
    features[:6] = [position["x"] / 256, position["y"] / 256, position["z"] / 15,
                    position["map_group"] / 64, position["map_num"] / 256,
                    position["tile_transition_state"] / 3]
    party = bridge.get_decrypted_player_party()
    for index, mon in enumerate(party[:6]):
        features[6 + 10 * index:16 + 10 * index] = _party_features(mon)
    _, collisions, elevations = bridge.extract_live_map_matrices()
    x, y = position["x"], position["y"]
    offsets = ((0, -1), (1, 0), (0, 1), (-1, 0), (0, -2), (2, 0), (0, 2), (-2, 0))
    for index, (dx, dy) in enumerate(offsets):
        nx, ny = x + dx, y + dy
        if 0 <= ny < collisions.shape[0] and 0 <= nx < collisions.shape[1]:
            features[66 + 2 * index] = int(collisions[ny, nx]) / 3
            features[67 + 2 * index] = int(elevations[ny, nx]) / 15
        else:
            features[66 + 2 * index] = 1.0
    inventory = bridge.get_inventory()
    for index, item in enumerate((4, 3, 2, 1, 13, 14, 15, 16, 17, 18, 19, 20)):
        features[82 + index] = min(999, inventory.get(item, 0)) / 999
    in_battle, trainer = bridge.is_in_battle()
    features[94:] = [float(in_battle), float(trainer)]
    return torch.tensor(features, dtype=torch.float32)


def _active_entity(mon: BattlePokemonStruct, own: bool, knowledge: OpponentKnowledge,
                   catalog: RomCatalog | None = None) -> list[float]:
    """Encode48 fields; hidden enemy stats, item, ability and moves are excluded."""
    row = [0.0] * 48
    row[:6] = [mon.species / 411, mon.level / 100, mon.hp / max(1, mon.maxHP),
               mon.type1 / 17, mon.type2 / 17, float(bool(mon.status1))]
    stages = [int(s) for s in mon.statStages]
    row[6:14] = [(s - 6) / 6 for s in stages]
    if own:
        row[14:19] = [mon.attack / 1024, mon.defense / 1024, mon.speed / 1024,
                      mon.spAttack / 1024, mon.spDefense / 1024]
        row[19] = mon.ability / 80
        row[20] = mon.item / 400
        row[21:25] = [int(move) / 354 for move in mon.moves]
        row[25:29] = [int(pp) / 64 for pp in mon.pp]
        if catalog is not None:
            for index, move_id in enumerate(mon.moves):
                if move_id:
                    move = catalog.move(int(move_id))
                    row[25 + index] = int(mon.pp[index]) / max(1, move.pp)
                    row[29 + index] = move.power / 150
                    row[33 + index] = (move.accuracy or 100) / 100
                    row[37 + index] = move.move_type / 17
                    row[41 + index] = float(move.category == "special")
    else:
        row[19] = knowledge.revealed_abilities.get(mon.species, 0) / 80
        row[20] = knowledge.revealed_items.get(mon.species, 0) / 400
        for index, revealed_move in enumerate(knowledge.revealed_moves.get(mon.species, [])[:4]):
            row[21 + index] = revealed_move / 354
    return row


@lru_cache(maxsize=2)
def _battle_symbols(game: Game) -> dict[str, RomSymbol]:
    """Pinned native metadata addresses; ROM/backend validation happens in the bridge."""
    return load_symbols(game)


def _native_acting_battler(bridge: GBABridgeProtocol, symbols: dict[str, RomSymbol]) -> int:
    callbacks = native_player_input_callbacks(bridge.memory_map.game)
    base = symbols["gBattlerControllerFuncs"].address
    for mode in ("action", "move", "target", "replacement"):
        for battler in (0, 2):
            if bridge.read_u32(base + 4 * battler) & ~1 in callbacks[mode]:
                return battler
    return 0


def player_party_in_battle_order(bridge: GBABridgeProtocol) -> list[dict[str, Any]]:
    """Normalize the temporary native Party-screen permutation for policy use."""
    party = bridge.get_decrypted_player_party()
    order = native_party_slot_order(bridge)
    if order is None:
        return party
    normalized: list[dict[str, Any] | None] = [None] * len(party)
    for visual_slot, mon in enumerate(party):
        stable_slot = order[visual_slot]
        if stable_slot >= len(party):
            raise ValueError("Native party menu order does not match the party span")
        normalized[stable_slot] = mon
    if any(mon is None for mon in normalized):
        raise ValueError("Native party menu order is incomplete")
    return cast(list[dict[str, Any]], normalized)


def encode_battle(bridge: GBABridgeProtocol, knowledge: OpponentKnowledge,
                  active_party_index: int = 0, catalog: RomCatalog | None = None,
                  acting_battler: int | None = None) -> BattleFeatures:
    """Encode one acting mon, its partner, visible rivals and the actual own bench.

    Nine actions remain four moves then up to five switches. switch_party_indices
    maps switch slots back to native party indexes; double-active mons are never
    offered as switch choices. target_battlers maps left/right rival and ally to
    physical battler indexes. Missing entities are explicitly padded.
    """
    if active_party_index < 0 or active_party_index > 5 or acting_battler not in (None, 0, 2):
        raise ValueError("invalid own active party/battler index")
    party = player_party_in_battle_order(bridge)  # fail on transient/corrupt party snapshots
    native = bridge.menu_state().mode == "battle"
    doubles = bool(bridge.read_u32(bridge.memory_map.battle_flags) & 1)
    slots = [0, 1, 2, 3] if doubles else [0, 1]
    positions = {slot: slot for slot in slots}
    party_indexes = {0: active_party_index}
    absent = 0
    actor = acting_battler if acting_battler is not None else 0
    if native or doubles:
        game: Game = "emerald" if bridge.memory_map.game == "emerald" else "firered"
        symbols = _battle_symbols(game)
        count = bridge.read_u8(symbols["gBattlersCount"].address)
        if native and count not in (2, 4):
            raise ValueError("native battlers are not yet initialized")
        if count in (2, 4):
            slots = list(range(count))
            native_positions = bridge.read_memory(symbols["gBattlerPositions"].address, count)
            if len(set(native_positions)) != count or any(position > 3 for position in native_positions):
                raise ValueError("invalid native battler position layout")
            positions = {slot: int(native_positions[slot]) for slot in slots}
            absent = bridge.read_u8(symbols["gAbsentBattlerFlags"].address)
            party_indexes = {slot: bridge.read_u16(symbols["gBattlerPartyIndexes"].address + 2 * slot)
                             for slot in slots if positions[slot] in (0, 2)}
        if acting_battler is None and native:
            actor = _native_acting_battler(bridge, symbols)
    if actor not in slots or positions.get(actor) not in (0, 2):
        raise ValueError("acting battler is absent or belongs to the opponent")
    my_party_index = party_indexes.get(actor, active_party_index)
    if my_party_index >= len(party):
        raise ValueError("own battler has an invalid native party index")
    by_position = {position: slot for slot, position in positions.items()}
    my_position = positions[actor]
    partner = by_position.get(my_position ^ 2)
    opponent_left, opponent_right = by_position.get(1), by_position.get(3)
    ordered_slots: list[tuple[int | None, int, bool]] = [(actor, 0, True)]
    if doubles:
        ordered_slots.append((partner, 1, True))
    ordered_slots.append((opponent_left, 2, False))
    if doubles:
        ordered_slots.append((opponent_right, 3, False))
    mons = {slot: bridge.get_battle_pokemon(slot) for slot in slots}
    rows: list[list[float]] = []
    roles: list[int] = []
    padding: list[bool] = []
    for slot, role, own in ordered_slots:
        present = slot is not None and not absent & (1 << slot) and int(mons[slot].species) > 0
        rows.append(_active_entity(mons[slot], own, knowledge, catalog) if present and slot is not None else [0.0] * 48)
        roles.append(role)
        padding.append(not present)
    me = mons[actor]
    forced_switch = int(me.hp) == 0 or bool(absent & (1 << actor))
    moves = [bool(me.moves[index] and me.pp[index]) and not forced_switch for index in range(4)]
    active_party = {index for slot, index in party_indexes.items()
                    if slot in slots and not absent & (1 << slot) and int(mons[slot].hp) > 0}
    # Always exclude the acting party slot, including its fainted form.
    active_party.add(my_party_index)
    bench_indexes = tuple(index for index in range(len(party)) if index not in active_party)[:5]
    for index in bench_indexes:
        row = [0.0] * 48
        row[:10] = _party_features(party[index])
        rows.append(row)
        roles.append(4)
        padding.append(False)
    moves += [int(party[index]["hp"]) > 0 for index in bench_indexes]
    moves += [False] * (9 - len(moves))
    targets = (opponent_left, opponent_right, partner)
    target_mask = [slot is not None and not absent & (1 << slot) and int(mons[slot].hp) > 0
                   for slot in targets]
    target_battlers = (-1 if opponent_left is None else opponent_left,
                      -1 if opponent_right is None else opponent_right,
                      -1 if partner is None else partner)
    return BattleFeatures(torch.tensor([rows], dtype=torch.float32),
                          torch.tensor([roles], dtype=torch.long),
                          torch.tensor([moves], dtype=torch.bool),
                          torch.tensor([target_mask], dtype=torch.bool),
                          torch.tensor([padding], dtype=torch.bool), actor, my_party_index,
                          bench_indexes, target_battlers,
                          doubles, forced_switch)
