"""Numerical differential, learning and real-rollout checks for neural modules."""
from __future__ import annotations

import copy
import itertools
import math
import struct
from typing import Callable, cast

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim.adam import Adam

from core.battle.mamba_entity_net import AresBattlePolicyNetwork, SelectiveSSMBlock
from core.battle.menu_cursor_controller import BattleMenuCursorController
from core.battle.psro_league_trainer import BatchedIntegerBattleArena, PSROLeagueTrainer
from core.bridge.ffi_mgba import MenuState
from core.bridge.gba_structs import BattlePokemonStruct
from core.bridge.synthetic_gba_silicon import SyntheticGBASiliconBridge
from core.game_data import Game, load_symbols
from core.observations import OpponentKnowledge, encode_battle
from core.planner.gflownet_team import TeamCandidate, TeamSynthesisGFlowNet
from core.world_model.hjepa import LatentHJEPAWorldModel


@pytest.fixture(autouse=True)
def deterministic_cpu() -> None:
    torch.manual_seed(42)
    torch.set_num_threads(4)


@pytest.mark.parametrize("length", [1, 3, 31, 32, 33, 65])
def test_ssd_chunk_recurrence_and_streaming_match(length: int) -> None:
    model = SelectiveSSMBlock(32, 8, head_dim=16, groups=2).double()
    inputs = torch.randn(2, length, 32, dtype=torch.float64)
    outputs, cache = model.forward_sequence(inputs, chunk_size=32)
    recurrent = model.initial_state(2, dtype=torch.float64)
    reference: list[Tensor] = []
    for step in range(length):
        result, recurrent = model.step_recurrent(inputs[:, step], recurrent)
        reference.append(result)
    torch.testing.assert_close(outputs, torch.stack(reference, 1), rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(cache.ssm_state, recurrent.ssm_state, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(cache.conv_state, recurrent.conv_state)
    if length > 1:
        cut = length // 2
        first, streaming = model.forward_sequence(inputs[:, :cut], chunk_size=7)
        second, streaming = model.forward_sequence(inputs[:, cut:], streaming, chunk_size=11)
        torch.testing.assert_close(outputs, torch.cat((first, second), 1), rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(cache.ssm_state, streaming.ssm_state, rtol=1e-9, atol=1e-10)
    assert model.A_log.shape == (model.num_heads,)  # scalar A per head, not diagonal state-vector A
    assert cache.conv_state.shape[-1] == 3


def test_ssd_gradient_parity_and_future_causality() -> None:
    chunked = SelectiveSSMBlock(16, 4, head_dim=8).double()
    recurrent = copy.deepcopy(chunked)
    x1 = torch.randn(2, 9, 16, dtype=torch.float64, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1, _ = chunked.forward_sequence(x1, chunk_size=4)
    state = recurrent.initial_state(2, dtype=torch.float64)
    turns: list[Tensor] = []
    for step in range(x2.shape[1]):
        output, state = recurrent.step_recurrent(x2[:, step], state)
        turns.append(output)
    y2 = torch.stack(turns, 1)
    cast(Callable[[], None], y1.square().mean().backward)()
    cast(Callable[[], None], y2.square().mean().backward)()
    assert x1.grad is not None and x2.grad is not None
    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-7, atol=1e-9)
    for left, right in zip(chunked.parameters(), recurrent.parameters(), strict=True):
        assert left.grad is not None and right.grad is not None
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-7, atol=1e-9)
        assert torch.isfinite(left.grad).all()
    future_changed = x1.detach().clone()
    future_changed[:, 5:] += 100
    changed, _ = chunked.forward_sequence(future_changed)
    torch.testing.assert_close(y1[:, :5], changed[:, :5], rtol=1e-9, atol=1e-10)


def test_entity_missing_tokens_masks_and_full_state() -> None:
    policy = AresBattlePolicyNetwork(entity_dim=48, d_model=32, num_layers=3)
    entities = torch.randn(2, 5, 48)
    roles = torch.tensor([[0, 2, 4, 3, 6], [0, 2, 4, 3, 6]])
    present = torch.tensor([[True, True, False, False, True]]).expand(2, -1)
    actions = torch.tensor([[True, False, True, False, False, False, False, False, False]]).expand(2, -1)
    targets = torch.tensor([[True, False, False]]).expand(2, -1)
    initial = policy.initial_state(2)
    logits, target_logits, value, cache = policy.forward_turn(entities, roles, actions, initial, targets, present)
    dirty = entities.clone()
    dirty[:, 2:4] = float("nan")
    altered = policy.forward_turn(dirty, roles, actions, initial, targets, present)
    torch.testing.assert_close(logits, altered[0])
    torch.testing.assert_close(value, altered[2])
    assert torch.isneginf(logits[~actions]).all()
    assert torch.isneginf(target_logits[~targets]).all()
    assert torch.isfinite(value).all()
    assert cache.ssm_state.shape[0] == 2 and cache.conv_state.shape[-1] == 3
    reset = cache.reset(torch.tensor([True, False]))
    assert reset.ssm_state[0].count_nonzero() == 0
    torch.testing.assert_close(reset.ssm_state[1], cache.ssm_state[1])
    with pytest.raises(ValueError, match="legal choice"):
        policy.forward_turn(entities, roles, torch.zeros_like(actions), initial)


def test_hjepa_ema_vicreg_multihorizon_learning_and_planning() -> None:
    model = LatentHJEPAWorldModel(state_dim=8, action_dim=4, latent_dim=16, ema_decay=0.99)
    before = [p.clone() for p in model.target_encoder.parameters()]
    with torch.no_grad():
        for p in model.online_encoder.parameters():
            p.add_(0.1)
    model.update_target_ema()
    for old, target in zip(before, model.target_encoder.parameters(), strict=True):
        torch.testing.assert_close(target, old + 0.001)
        assert not target.requires_grad
    states = torch.randn(32, 5, 8)
    actions = torch.randint(0, 4, (32, 4))
    for step in range(4):
        states[:, step + 1] = states[:, step] * 0.8 + F.one_hot(actions[:, step], 8).float() * 0.3
    optimizer = Adam(model.parameters(), lr=2e-3)
    initial = float(model.forward_sequence_loss(states, actions)[0].detach())
    for _ in range(35):
        loss, metrics = model.forward_sequence_loss(states, actions)
        optimizer.zero_grad(set_to_none=True)
        cast(Callable[[], None], loss.backward)()
        optimizer.step()
    final = float(model.forward_sequence_loss(states, actions)[0].detach())
    assert final < initial * 0.85
    assert set(metrics) == {"horizon_1", "horizon_2", "horizon_4"}
    assert all(p.grad is None for p in model.target_encoder.parameters())
    candidates = torch.randint(0, 4, (2, 3, 4))
    selected = model.plan_actions(states[:2, 0], candidates, states[:2, -1])
    assert selected.shape == (2,) and torch.all((selected >= 0) & (selected < 3))
    singleton, singleton_metrics = model.forward_loss(states[:1, 0], actions[:1, 0], states[:1, 1])
    assert torch.isfinite(singleton) and all(math.isfinite(v) for v in singleton_metrics.values())


def _all_team_path_ratios(network: TeamSynthesisGFlowNet) -> tuple[Tensor, Tensor, Tensor]:
    terminals: list[Tensor] = []
    ratios: list[Tensor] = []
    rewards: list[float] = []
    for ordering in itertools.permutations(range(3), 2):
        state = torch.zeros(3)
        ratio = network.log_Z * 0
        for step, chosen in enumerate(ordering):
            logits = cast(Tensor, network.forward_policy_net(state)).masked_fill(state.bool(), -torch.inf)
            ratio = ratio + F.log_softmax(logits, -1)[chosen] + math.log(step + 1)
            state = state.clone()
            state[chosen] = 1
        terminals.append(state)
        ratios.append(ratio)
        rewards.append(math.log(9 if set(ordering) == {0, 1} else 1))
    return torch.stack(terminals), torch.stack(ratios), torch.tensor(rewards)


def test_gflownet_balance_learns_reward_proportions_and_legal_moves() -> None:
    network = TeamSynthesisGFlowNet(3, hidden_dim=32, num_available_moves=16)
    optimizer = Adam([{"params": [network.log_Z], "lr": 0.08},
                      {"params": list(network.forward_policy_net.parameters()), "lr": 0.006}])
    states, ratios, rewards = _all_team_path_ratios(network)
    initial = float(network.compute_trajectory_balance_loss(states, ratios, rewards).detach())
    for _ in range(140):
        states, ratios, rewards = _all_team_path_ratios(network)
        loss = network.compute_trajectory_balance_loss(states, ratios, rewards)
        optimizer.zero_grad(set_to_none=True)
        cast(Callable[[], None], loss.backward)()
        optimizer.step()
    states, ratios, rewards = _all_team_path_ratios(network)
    final = float(network.compute_trajectory_balance_loss(states, ratios, rewards).detach())
    assert final < initial * 0.02
    # Six ordered paths correspond to three unordered terminal teams; P_B=1/2.
    path_probs = (ratios - math.log(2)).exp()
    preferred = path_probs[0] + path_probs[2]
    assert abs(float(preferred.detach()) - 9 / 11) < 0.03
    catalog = (TeamCandidate(1, (12, 3), (1, 2, 3, 4, 5)),
               TeamCandidate(4, (10, 10), (6, 7)), TeamCandidate(0, (0, 0), ()))
    trajectory = network.sample_team_and_moves(catalog, 2, available_mask=torch.tensor([True, True, False]))
    assert trajectory.species == (1, 4)
    assert len(trajectory.moves[0]) == 4 and set(trajectory.moves[0]) <= {1, 2, 3, 4, 5}
    assert trajectory.moves[1] == (6, 7)
    assert network.coverage_reward(catalog[:2]) > 0
    with pytest.raises(ValueError, match="insufficient"):
        network.sample_team_trajectory(2, available_mask=torch.tensor([True, False, False]))


def test_recurrent_ppo_actual_rollouts_frozen_league_and_nash() -> None:
    policy = AresBattlePolicyNetwork(d_model=32, num_layers=1)
    trainer = PSROLeagueTrainer(policy)
    arena = BatchedIntegerBattleArena(2, seed=42)
    observed = arena.reset()
    assert observed.entities[:, 1, 14:33].count_nonzero() == 0  # hidden enemy data boundary
    rollout = trainer.collect_rollout(arena, 16)
    assert rollout.done.any() and rollout.results.abs().sum() > 0
    assert not rollout.old_log_probs.requires_grad
    assert torch.isfinite(rollout.advantages).all()
    previous = [p.detach().clone() for p in policy.parameters()]
    metrics = trainer.ppo_update(rollout, epochs=2, sequence_length=8)
    assert metrics["gradient_norm"] > 0 and metrics["updates"] == 4
    assert all(math.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(old, new) for old, new in zip(previous, policy.parameters(), strict=True))
    trainer.freeze_current_policy()
    frozen = [p.clone() for p in trainer.frozen_policies[0].parameters()]
    with torch.no_grad():
        next(policy.parameters()).add_(0.05)
    for old, new in zip(frozen, trainer.frozen_policies[0].parameters(), strict=True):
        torch.testing.assert_close(old, new)
        assert not new.requires_grad
    trainer.freeze_current_policy()
    payoffs = trainer.evaluate_league(episodes_per_match=4)
    torch.testing.assert_close(payoffs, -payoffs.T)
    assert payoffs.shape == (2, 2) and torch.all(payoffs.abs() <= 1)
    assert float(trainer.nash_distribution.sum()) == pytest.approx(1)
    rps = torch.tensor([[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]])
    mixture = trainer.fictitious_play(rps, 3000)
    torch.testing.assert_close(mixture, torch.full((3,), 1 / 3), rtol=0, atol=0.025)
    dominated = trainer.fictitious_play(torch.tensor([[0., 1.], [-1., 0.]]), 1000)
    assert float(dominated[0]) > 0.99


class _NativeMenuMemory(SyntheticGBASiliconBridge):
    """RAM fixture with pinned native callback bindings, not campaign simulation."""
    test_learning_opcode: int | None = None

    def read_memory(self, address: int, size: int) -> bytes:
        if address == 0x08001000 and size == 1 and self.test_learning_opcode is not None:
            return bytes([self.test_learning_opcode])
        return super().read_memory(address, size)

    def menu_state(self) -> MenuState:
        action, move = self.get_battle_cursors()
        return MenuState(True, action, move, 0, False, "battle")


_NATIVE_MENU_BINDINGS: tuple[dict[str, int], ...] = (
    {"callbacks": 0x03004FE0, "action": 0x0802E438, "move": 0x0802EA10, "target": 0x0802E674,
     "party": 0x0203B0A0, "party_task": 0x0811FB28, "party_context": 0x08122C5C,
     "bag": 0x0203ACFC, "bag_task": 0x08108F0C, "bag_context": 0x08109BE4,
     "menu": 0x0203ADE4, "positions": 0x02023BD6, "target_cursor": 0x03004FF4,
     "text": 0x02020034},
    {"callbacks": 0x03005D60, "action": 0x08057588, "move": 0x08057BFC, "target": 0x08057824,
     "party": 0x0203CEC8, "party_task": 0x081B1370, "party_context": 0x081B3730,
     "bag": 0x0203CE58, "bag_task": 0x081ABD28, "bag_context": 0x081ACC04,
     "menu": 0x0203CD90, "positions": 0x02024076, "target_cursor": 0x03005D74,
     "text": 0x020201B0},
)


def _native_menu_fixture(emerald: bool) -> tuple[_NativeMenuMemory, BattleMenuCursorController, dict[str, int]]:
    bridge = _NativeMenuMemory(is_emerald=emerald)
    bridge.write_memory(bridge.memory_map.main + 0x439, b"\x02")
    binding = _NATIVE_MENU_BINDINGS[int(emerald)]
    bridge.write_memory(binding["callbacks"], bytes(16))
    bridge.write_memory(bridge.memory_map.tasks, bytes(16 * 40))
    bridge.write_memory(binding["positions"], bytes((0, 1, 2, 3)))
    bridge.write_memory(binding["text"], bytes(32 * 36))
    return bridge, BattleMenuCursorController(bridge, pulse_frames=1, release_frames=1), binding


def _consume_release(bridge: _NativeMenuMemory, controller: BattleMenuCursorController) -> None:
    bridge.step_frame()
    assert controller.tick() == 0
    bridge.step_frame()


@pytest.mark.parametrize("emerald", [False, True])
def test_native_menu_binding_double_battler_cursors_and_text(emerald: bool) -> None:
    bridge, controller, binding = _native_menu_fixture(emerald)
    bridge.write_memory(binding["callbacks"] + 8, struct.pack("<I", binding["action"] | 1))
    bridge.write_memory(bridge.memory_map.action_cursor + 2, b"\x03")
    observed = controller.observe_menu()
    assert observed.mode == "action" and observed.battler == 2 and observed.action_cursor == 3
    controller.request_move(1, target=2)
    frame = bridge.frame_count
    assert controller.tick() == bridge.KEY_LEFT
    assert bridge.frame_count == frame  # only the caller owns frame progression
    _consume_release(bridge, controller)
    bridge.write_memory(bridge.memory_map.action_cursor + 2, b"\x02")
    assert controller.tick() == bridge.KEY_UP
    _consume_release(bridge, controller)
    bridge.write_memory(bridge.memory_map.action_cursor + 2, b"\x00")
    assert controller.tick() == bridge.KEY_A
    _consume_release(bridge, controller)
    bridge.write_memory(binding["callbacks"] + 8, struct.pack("<I", binding["move"] | 1))
    bridge.write_memory(bridge.memory_map.move_cursor + 2, b"\x03")
    bridge.write_memory(bridge.memory_map.battle_flags, struct.pack("<I", 1))
    assert controller.tick() == bridge.KEY_UP
    _consume_release(bridge, controller)
    bridge.write_memory(bridge.memory_map.move_cursor + 2, b"\x01")
    assert controller.tick() == bridge.KEY_A
    _consume_release(bridge, controller)
    bridge.write_memory(binding["callbacks"] + 8, struct.pack("<I", binding["target"] | 1))
    bridge.write_memory(binding["target_cursor"], b"\x01")
    assert controller.tick() == bridge.KEY_RIGHT
    _consume_release(bridge, controller)
    bridge.write_memory(binding["target_cursor"], b"\x00")  # input battler2's ally is battler0
    assert controller.tick() == bridge.KEY_A
    _consume_release(bridge, controller)
    bridge.write_memory(binding["callbacks"], bytes(16))
    bridge.write_memory(binding["text"] + 0x1B, b"\x01\x02")
    assert controller.menu_mode() == "text" and controller.tick() == bridge.KEY_A
    _consume_release(bridge, controller)
    bridge.write_memory(binding["text"] + 0x1C, b"\x05")  # sound wait is not an input request
    assert controller.menu_mode() == "animation" and controller.tick() == 0
    bridge.close()


@pytest.mark.parametrize("emerald", [False, True])
def test_native_bag_pocket_scroll_item_use_and_forced_replacement(emerald: bool) -> None:
    bridge, controller, binding = _native_menu_fixture(emerald)
    # Actual encrypted pocket slot order with two balls, deliberately unsorted IDs.
    pocket = 1 if emerald else 2
    descriptor = bridge.memory_map.bag_pockets + pocket * 8
    pointer = 0x0201F800
    key = bridge.read_u32(bridge.get_save_block2_dynamic_ptr() + bridge.memory_map.encryption_key_offset) & 0xFFFF
    bridge.write_memory(bridge.memory_map.bag_pockets, bytes(5 * 8))
    bridge.write_memory(descriptor, struct.pack("<IB3x", pointer, 2))
    bridge.write_memory(pointer, struct.pack("<HHHH", 3, 5 ^ key, 4, 8 ^ key))
    bridge.write_memory(bridge.memory_map.tasks, struct.pack("<IB3x", binding["bag_task"] | 1, 1))
    bridge.write_memory(binding["bag"], bytes(28 if emerald else 20))
    controller.request_capture(4)
    assert controller.menu_mode() == "bag" and controller.tick() == bridge.KEY_RIGHT
    _consume_release(bridge, controller)
    if emerald:
        bridge.write_memory(binding["bag"] + 5, bytes([pocket]))
    else:
        bridge.write_memory(binding["bag"] + 6, struct.pack("<H", pocket))
    assert controller.tick() == bridge.KEY_DOWN
    _consume_release(bridge, controller)
    # Persistent scroll1+row0 means item4 is already selected; do not reset cursor.
    scroll_offset = 18 if emerald else 8
    bridge.write_memory(binding["bag"] + scroll_offset + 2 * pocket, struct.pack("<H", 1))
    assert controller.tick() == bridge.KEY_A
    _consume_release(bridge, controller)
    bridge.write_memory(bridge.memory_map.tasks, struct.pack("<IB3x", binding["bag_context"] | 1, 1))
    bridge.write_memory(binding["menu"] + 2, b"\x01")
    assert controller.tick() == bridge.KEY_UP
    _consume_release(bridge, controller)
    bridge.write_memory(binding["menu"] + 2, b"\x00")
    assert controller.tick() == bridge.KEY_A
    _consume_release(bridge, controller)
    assert controller.pending is not None  # a submitted A edge is not confirmed consumption
    bridge.write_memory(pointer + 6, struct.pack("<H", 7 ^ key))
    assert controller.tick() == 0
    assert not controller.busy
    # A forced replacement starts directly in the native party task, bypassing Fight.
    party = bridge.get_decrypted_player_party()
    party[0]["hp"] = 0
    replacement = dict(party[0])
    replacement.update({"hp": 30, "species": 16, "pid": 43})
    bridge.configure_party([party[0], replacement])
    bridge.write_memory(bridge.memory_map.tasks, struct.pack("<IB3x", binding["party_task"] | 1, 1))
    bridge.write_memory(binding["party"] + 9, b"\x00")
    controller.request_switch(1)
    assert controller.tick() == bridge.KEY_DOWN
    _consume_release(bridge, controller)
    bridge.write_memory(binding["party"] + 9, b"\x01")
    assert controller.tick() == bridge.KEY_A
    _consume_release(bridge, controller)
    assert controller.pending is not None  # native selection still needs Shift/Send Out confirmation
    bridge.write_memory(bridge.memory_map.tasks, struct.pack("<IB3x", binding["party_context"] | 1, 1))
    bridge.write_memory(binding["menu"] + 2, b"\x00")
    assert controller.tick() == bridge.KEY_A
    _consume_release(bridge, controller)
    assert controller.pending is not None  # wait until the native battle screen returns
    bridge.write_memory(bridge.memory_map.tasks, bytes(16 * 40))
    bridge.write_memory(binding["callbacks"], struct.pack("<I", binding["action"] | 1))
    assert controller.tick() == 0
    assert not controller.busy
    with pytest.raises(ValueError, match="healthy"):
        controller.request_switch(0)
    bridge.write_memory(bridge.memory_map.battle_flags, struct.pack("<I", 8))
    with pytest.raises(ValueError, match="trainer"):
        controller.request_capture(4)
    bridge.close()


@pytest.mark.parametrize("callback", [0x0802E438, 0x080E763C])
def test_firered_normal_and_oak_first_battle_input_callbacks(callback: int) -> None:
    bridge, controller, binding = _native_menu_fixture(False)
    bridge.write_memory(binding["callbacks"], struct.pack("<I", callback | 1))
    assert controller.menu_mode() == "action"
    controller.request_move(0)
    assert controller.tick() == bridge.KEY_A
    _consume_release(bridge, controller)
    bridge.write_memory(binding["callbacks"], struct.pack("<I", 0x080E7989))
    assert controller.menu_mode() == "move"
    bridge.close()


@pytest.mark.parametrize("emerald", [False, True])
def test_battle_controller_checkpoint_exact_key_continuation_and_atomic_validation(emerald: bool) -> None:
    bridge, _, binding = _native_menu_fixture(emerald)
    original = BattleMenuCursorController(bridge, pulse_frames=4, release_frames=3)
    bridge.write_memory(binding["callbacks"] + 8, struct.pack("<I", binding["action"] | 1))
    bridge.write_memory(bridge.memory_map.action_cursor + 2, b"\x03")
    original.request_move(2, target=1)
    assert original.tick() == bridge.KEY_LEFT
    checkpoint = original.state_dict()
    assert checkpoint["hold_left"] == 3 and checkpoint["release_left"] == 3
    emulator = bridge.save_state()
    restored_bridge = _NativeMenuMemory(is_emerald=emerald)
    restored_bridge.load_state(emulator)
    restored = BattleMenuCursorController(restored_bridge)  # saved timing overrides defaults
    restored.load_state_dict(checkpoint)
    assert restored.state_dict() == original.state_dict()
    # Exact held/released waveform survives a restart; no repeated directional edge.
    expected = [bridge.KEY_LEFT] * 3 + [0] * 3
    for key in expected:
        assert original.tick() == restored.tick() == key
        bridge.step_frame()
        restored_bridge.step_frame()
    assert restored.state_dict() == original.state_dict()
    before = restored.state_dict()
    corrupt = dict(checkpoint)
    corrupt["input_battler"] = 1
    with pytest.raises(ValueError, match="player side"):
        restored.load_state_dict(corrupt)
    assert restored.state_dict() == before
    corrupt = dict(checkpoint)
    corrupt["hold_left"] = 4
    with pytest.raises(ValueError, match="hold_left"):
        restored.load_state_dict(corrupt)
    assert restored.state_dict() == before
    # Intent restore must not decode transient native plaintext party memory.
    restored_bridge.write_memory(restored_bridge.memory_map.party + 0x20, bytes(48))
    restored.load_state_dict(checkpoint)
    assert restored.pending == original.pending
    bridge.close()
    restored_bridge.close()


def _battle_mon(species: int, hp: int = 50) -> BattlePokemonStruct:
    mon = BattlePokemonStruct()
    mon.species, mon.hp, mon.maxHP, mon.level = species, hp, 100, 25
    mon.attack, mon.defense, mon.speed, mon.spAttack, mon.spDefense = 90, 80, 70, 100, 85
    for index, (move, pp) in enumerate(zip((33, 55, 52, 10), (2, 0, 4, 0), strict=True)):
        mon.moves[index], mon.pp[index] = move, pp
    for index in range(8):
        mon.statStages[index] = 6
    return mon


@pytest.mark.parametrize("emerald", [False, True])
def test_double_observations_acting_partner_legal_switches_and_hidden_enemy_boundary(emerald: bool) -> None:
    bridge, controller, binding = _native_menu_fixture(emerald)
    party = bridge.get_decrypted_player_party()
    roster: list[dict[str, object]] = []
    for index, species in enumerate((1, 4, 7, 16, 19, 25)):
        entry = dict(party[0])
        entry.update({"species": species, "pid": 50 + index, "hp": 0 if index == 4 else 30})
        roster.append(entry)
    bridge.configure_party(roster)
    game: Game = "emerald" if emerald else "firered"
    symbols = load_symbols(game)
    bridge.write_memory(bridge.memory_map.battle_flags, struct.pack("<I", 1))
    bridge.write_memory(symbols["gBattlersCount"].address, b"\x04")
    bridge.write_memory(symbols["gBattlerPartyIndexes"].address, struct.pack("<HHHH", 1, 0, 3, 1))
    bridge.write_memory(symbols["gAbsentBattlerFlags"].address, b"\x08")  # right rival absent
    bridge.write_memory(binding["callbacks"] + 8, struct.pack("<I", binding["action"] | 1))
    for slot, species in enumerate((4, 74, 16, 95)):
        bridge.write_memory(bridge.memory_map.battle_mons + slot * 88, bytes(_battle_mon(species)))
    knowledge = OpponentKnowledge()
    knowledge.reveal_move(74, 33)
    observed = encode_battle(bridge, knowledge)
    assert observed.acting_battler == 2 and observed.active_party_index == 3 and observed.doubles
    assert observed.roles[0, :4].tolist() == [0, 1, 2, 3]
    assert observed.switch_party_indices == (0, 2, 4, 5)
    assert observed.target_battlers == (1, 3, 0)
    assert observed.action_mask.tolist() == [[True, False, True, False, True, True, False, True, False]]
    assert observed.target_mask.tolist() == [[True, False, True]]
    assert observed.padding_mask[0, :4].tolist() == [False, False, False, True]
    assert observed.entities[0, 0, 0] == pytest.approx(16 / 411)
    assert observed.entities[0, 1, 0] == pytest.approx(4 / 411)
    enemy = bridge.get_battle_pokemon(1)
    enemy.attack, enemy.spAttack, enemy.item, enemy.ability = 65535, 65535, 999, 75
    for index in range(4):
        enemy.moves[index], enemy.pp[index] = 300 + index, 63
    bridge.write_memory(bridge.memory_map.battle_mons + 88, bytes(enemy))
    concealed_changed = encode_battle(bridge, knowledge)
    torch.testing.assert_close(observed.entities, concealed_changed.entities)
    assert observed.entities[0, 2, 14:19].count_nonzero() == 0
    assert observed.entities[0, 2, 21] == pytest.approx(33 / 354)
    # The fainted acting slot gets no move choices and cannot switch to the live ally.
    fainted = bridge.get_battle_pokemon(2)
    fainted.hp = 0
    bridge.write_memory(bridge.memory_map.battle_mons + 176, bytes(fainted))
    bridge.write_memory(symbols["gAbsentBattlerFlags"].address, b"\x0C")
    waiting = 0x08059828 if emerald else 0x08030684
    bridge.write_memory(binding["callbacks"] + 8, struct.pack("<I", waiting | 1))
    bridge.write_memory(bridge.memory_map.tasks, struct.pack("<IB3x", binding["party_task"] | 1, 1))
    assert controller.observe_menu().mode == "party" and controller.observe_menu().battler == 2
    forced = encode_battle(bridge, knowledge)
    assert forced.acting_battler == 2
    assert forced.forced_switch and not forced.action_mask[0, :4].any()
    assert 1 not in forced.switch_party_indices and 3 not in forced.switch_party_indices
    assert forced.padding_mask[0, 0]
    bridge.close()


@pytest.mark.parametrize("emerald", [False, True])
def test_native_battle_levelup_learning_reads_summary_cursor_and_preserves_hms(emerald: bool) -> None:
    bridge, controller, _ = _native_menu_fixture(emerald)
    symbols = load_symbols("emerald" if emerald else "firered")
    battle_struct = 0x0201F400
    summary = 0x02030000
    bridge.write_memory(symbols["gBattleStruct"].address, struct.pack("<I", battle_struct))
    bridge.write_memory(battle_struct + 0x10, b"\x00")
    bridge.write_memory(symbols["gMoveToLearn"].address, struct.pack("<H", 44))
    bridge.write_memory(symbols["gBattlescriptCurrInstr"].address, struct.pack("<I", 0x08001000))
    bridge.write_memory(symbols["gBattleScripting"].address + 0x1F, b"\x01")
    bridge.write_memory(symbols["gBattleCommunication"].address + 1, b"\x01")
    bridge.test_learning_opcode = 0x5A
    party = bridge.get_decrypted_player_party()
    party[0]["moves"], party[0]["pp"] = [15, 33, 22, 45], [30, 35, 10, 40]
    bridge.configure_party(party)
    assert controller.menu_mode() == "learn_yesno"
    assert controller.tick() == bridge.KEY_UP  # navigate to Yes using battle communication cursor
    _consume_release(bridge, controller)
    bridge.write_memory(symbols["gBattleCommunication"].address + 1, b"\x00")
    assert controller.tick() == bridge.KEY_A
    _consume_release(bridge, controller)
    bridge.write_memory(symbols["sMonSummaryScreen"].address, struct.pack("<I", summary))
    task_name = "Task_HandleReplaceMoveInput" if emerald else "Task_InputHandler_SelectOrForgetMove"
    bridge.write_memory(bridge.memory_map.tasks, struct.pack("<IB3x", symbols[task_name].address | 1, 1))
    cursor_address = summary + 0x40C6 if emerald else symbols["sMoveSelectionCursorPos"].address
    bridge.write_memory(cursor_address, b"\x01")
    if not emerald:
        bridge.write_memory(summary + 0x3288, b"\x01")
        assert controller.tick() == 0  # FR fade/setup state ignores move-selection keys
        bridge.write_memory(summary + 0x3288, b"\x02")
    assert controller.menu_mode() == "learn_forget"
    assert controller.tick() == bridge.KEY_DOWN  # slot3 Growl, never slot0 Cut(HM)
    _consume_release(bridge, controller)
    bridge.write_memory(cursor_address, b"\x03")
    assert controller.tick() == bridge.KEY_A
    _consume_release(bridge, controller)
    checkpoint = controller.state_dict()
    assert checkpoint["learning_slot"] == 3 and checkpoint["learning_move"] == 44
    # Every old slot is an HM: cancel replacement, then confirm stop learning.
    party[0]["moves"] = [15, 19, 57, 70]
    bridge.configure_party(party)
    assert controller.tick() == bridge.KEY_B
    _consume_release(bridge, controller)
    bridge.write_memory(bridge.memory_map.tasks, bytes(16 * 40))
    bridge.test_learning_opcode = 0x5B
    bridge.write_memory(symbols["gBattleCommunication"].address + 1, b"\x01")
    assert controller.menu_mode() == "learn_stop" and controller.tick() == bridge.KEY_UP
    _consume_release(bridge, controller)
    bridge.write_memory(symbols["gBattleCommunication"].address + 1, b"\x00")
    assert controller.tick() == bridge.KEY_A
    assert controller.state_dict()["learning_slot"] is None
    bridge.close()


@pytest.mark.parametrize("emerald", [False, True])
def test_native_level_up_stats_two_pages_need_new_key_edges_only_at_ready_states(emerald: bool) -> None:
    bridge, controller, _ = _native_menu_fixture(emerald)
    symbols = load_symbols("emerald" if emerald else "firered")
    bridge.test_learning_opcode = 0x6C
    bridge.write_memory(symbols["gBattlescriptCurrInstr"].address, struct.pack("<I", 0x08001000))
    state_address = symbols["gBattleScripting"].address + 0x1E
    bridge.write_memory(state_address, b"\x05")
    assert controller.menu_mode() == "animation" and controller.tick() == 0
    bridge.write_memory(state_address, b"\x06")
    assert controller.menu_mode() == "level_stats" and controller.tick() == bridge.KEY_A
    _consume_release(bridge, controller)
    bridge.write_memory(state_address, b"\x07")
    assert controller.menu_mode() == "animation" and controller.tick() == 0
    bridge.write_memory(state_address, b"\x08")
    assert controller.menu_mode() == "level_stats" and controller.tick() == bridge.KEY_A
    _consume_release(bridge, controller)
    bridge.write_memory(state_address, b"\x09")
    assert controller.tick() == 0
    bridge.close()
