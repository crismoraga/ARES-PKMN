"""Collect state transitions from real synthetic hardware frames, not random labels."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch import Tensor

from core.bridge.synthetic_gba_silicon import SyntheticGBASiliconBridge, SyntheticMove
from core.game_data import Game, RomCatalog
from core.observations import encode_world_state


def collect_world_transitions(count: int = 1024, seed: int = 42, game: Game = "firered") -> tuple[Tensor, Tensor, Tensor]:
    """Generate diverse legitimate RAM snapshots by holding movement keys to completion."""
    if count < 64:
        raise ValueError("At least64 transitions required for variance regularization")
    rng = random.Random(seed)
    catalog = RomCatalog(game)
    current_states: list[Tensor] = []
    next_states: list[Tensor] = []
    actions: list[int] = []
    bridge = SyntheticGBASiliconBridge(game == "emerald", seed)
    try:
        for episode in range((count + 31) // 32):
            shape = (12, 14)
            collision = np.zeros(shape, dtype=np.uint16)
            collision[0, :] = collision[-1, :] = 1
            collision[:, 0] = collision[:, -1] = 1
            elevation = np.full(shape, rng.choice((0, 3, 4, 6)), dtype=np.uint16)
            metatiles = np.full(shape, episode % 64, dtype=np.uint16)
            group, number = episode % 20, episode % 40
            bridge.configure_map(group, number, metatiles, collision, elevation)
            species = rng.choice((1, 4, 7, 25, 74, 129, 277, 283))
            mon = catalog.make_party_data(species, rng.randint(5, 65), rng.randrange(1 << 20))
            bridge.configure_party([mon])
            for move_id in mon["moves"]:
                if move_id:
                    move = catalog.move(int(move_id))
                    bridge.configure_move(SyntheticMove(move.move_id, move.power, move.move_type,
                                                        move.category == "special", move.accuracy or 100, move.pp))
            bridge.transition_map(group, number, 6, 6)
            for _ in range(32):
                if len(actions) == count:
                    break
                action = rng.randrange(4)
                key = (bridge.KEY_UP, bridge.KEY_RIGHT, bridge.KEY_DOWN, bridge.KEY_LEFT)[action]
                current_states.append(encode_world_state(bridge))
                bridge.set_hardware_keys(key)
                for _frame in range(16):
                    bridge.step_frame()
                bridge.set_hardware_keys(0)
                bridge.step_frame()
                next_states.append(encode_world_state(bridge))
                actions.append(action)
    finally:
        bridge.close()
    return torch.stack(current_states), torch.tensor(actions, dtype=torch.long), torch.stack(next_states)
