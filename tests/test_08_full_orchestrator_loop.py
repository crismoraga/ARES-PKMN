"""Full physical-frame integration, recovery and truthful certification bounds."""
from __future__ import annotations

from pathlib import Path
import struct
from typing import Any, cast

import pytest
import torch

from core.bridge.synthetic_gba_silicon import SyntheticGBASiliconBridge
from core.field_controls import FieldMenuController
from core.game_data import Game, ROOT, ROM_NAMES, RomCatalog
from core.observations import player_party_in_battle_order
from scripts.run_autonomous_verification_demo import build_laboratory, run_demo


@pytest.mark.parametrize("emerald", [False, True])
def test_full_physical_campaign_lab_exercises_all_layers(emerald: bool) -> None:
    torch.set_num_threads(2)
    result = run_demo(emerald, map_count=10, max_frames=20000)
    assert result["complete"] and result["native_campaign_certified"] is False
    metrics = cast(dict[str, Any], result["metrics"])
    assert metrics["captures"] >= 1 and metrics["wins"] >= 2
    assert metrics["causal_observations"] > 0 and metrics["belief_updates"] > 0
    assert metrics["neural_decisions"] > 0 and metrics["neural_plans"] > 0
    assert metrics["gflownet_samples"] > 0 and metrics["proved_decisions"] > 0
    assert metrics["faints"] == 0


def test_single_frame_owner_and_resume_preserve_physical_state(tmp_path: Path) -> None:
    torch.set_num_threads(2)
    first = build_laboratory(map_count=10, checkpoint_dir=tmp_path)
    second = build_laboratory(map_count=10, checkpoint_dir=tmp_path)
    try:
        for _ in range(270):
            previous = first.bridge.frame_count
            first.step_vblank_cycle()
            assert first.bridge.frame_count == previous + 1
        checkpoint = first.save_checkpoint("test")
        second.load_checkpoint(checkpoint)
        assert second.bridge.get_player_coords_and_map() == first.bridge.get_player_coords_and_map()
        assert second.bridge.get_decrypted_player_party() == first.bridge.get_decrypted_player_party()
        assert second.causal.state_dict() == first.causal.state_dict()
        assert torch.equal(second.cache.ssm_state, first.cache.ssm_state)
        assert torch.equal(second.cache.conv_state, first.cache.conv_state)
        assert second.bridge.frame_count == first.bridge.frame_count
    finally:
        first.close()
        second.close()


def test_checkpoint_integrity_rejects_corruption(tmp_path: Path) -> None:
    controller = build_laboratory(map_count=4, checkpoint_dir=tmp_path)
    try:
        checkpoint = controller.save_checkpoint("test")
        path = checkpoint / "emulator.state"
        data = path.read_bytes()
        path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        with pytest.raises(ValueError, match="integrity"):
            controller.load_checkpoint(checkpoint)
    finally:
        controller.close()


def test_synthetic_never_certifies_native_clearance() -> None:
    controller = build_laboratory(map_count=4)
    try:
        evidence = controller.native_completion_evidence()
        assert evidence["complete"] is False
        assert controller.status()["native_campaign_certified"] is False
    finally:
        controller.close()


def test_training_intent_cache_survives_tile_motion_but_invalidates_on_roster_change(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Training cannot run a whole-world route search on each tile/VBlank."""
    controller = build_laboratory(map_count=4)
    bridge = cast(SyntheticGBASiliconBridge, controller.bridge)
    calls = 0

    def training(*args: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"mode": "OPTIMAL_TRAINING_CURRICULUM", "dest": (0, 0, (5, 2))}

    monkeypatch.setattr(controller.planner, "get_active_directive", training)
    try:
        frame = bridge.frame_count
        controller._execute_overworld_cycle()
        assert calls == 1
        controller.movement.clear()
        bridge.transition_map(0, 0, 2, 2, relocate=False)
        controller._execute_overworld_cycle()
        assert calls == 1
        assert controller.movement.pending
        controller.movement.clear()
        party = bridge.get_decrypted_player_party()
        party[0]["level"] += 1
        bridge.configure_party(party)
        controller._execute_overworld_cycle()
        assert calls == 2
        assert bridge.frame_count == frame  # Every planner still owns zero frames.
    finally:
        controller.close()


@pytest.mark.parametrize("phase", ["movement", "battle_hold", "battle_release"])
def test_resume_mid_input_has_identical_subsequent_physical_frames(tmp_path: Path, phase: str) -> None:
    """A savestate in a held pulse must not replay or discard a hardware edge."""
    torch.set_num_threads(2)
    first = build_laboratory(map_count=4, checkpoint_dir=tmp_path / "first")
    second = build_laboratory(map_count=4, checkpoint_dir=tmp_path / "second")
    try:
        found = False
        for _ in range(300):
            first.step_vblank_cycle()
            menu = first.menu.state_dict()
            if phase == "movement":
                found = first.movement.pending and first.bridge.read_u8(first.bridge.memory_map.avatar + 3) == 1
            elif phase == "battle_hold":
                found = first.bridge.is_in_battle()[0] and cast(int, menu["hold_left"]) > 0
            else:
                found = first.bridge.is_in_battle()[0] and cast(int, menu["hold_left"]) == 0 and cast(int, menu["release_left"]) > 0
            if found:
                break
        assert found, f"No physical{phase} state was reached"
        checkpoint = first.save_checkpoint("mid_input")
        second.load_checkpoint(checkpoint)
        assert second.movement.state_dict() == first.movement.state_dict()
        assert second.menu.state_dict() == first.menu.state_dict()
        assert second.bridge.save_state() == first.bridge.save_state()
        for _ in range(80):
            # Both independent branches receive the same saved policy RNG stream.
            policy_rng = torch.get_rng_state()
            first.step_vblank_cycle()
            torch.set_rng_state(policy_rng)
            second.step_vblank_cycle()
            assert first.bridge.save_state() == second.bridge.save_state()
            assert first.movement.state_dict() == second.movement.state_dict()
            assert first.menu.state_dict() == second.menu.state_dict()
            assert torch.equal(first.cache.ssm_state, second.cache.ssm_state)
            assert torch.equal(first.cache.conv_state, second.cache.conv_state)
    finally:
        first.close()
        second.close()


def _field_fixture(game: Game) -> tuple[SyntheticGBASiliconBridge, FieldMenuController]:
    """Native ROM metadata with source-shaped RAM; this is not native gameplay."""
    if not (ROOT.parent / ROM_NAMES[game]).is_file():
        pytest.skip("Field ABI tests require the supplied matching game metadata")
    bridge = SyntheticGBASiliconBridge(game == "emerald")
    field = FieldMenuController(bridge, game, RomCatalog(game))
    party = bridge.get_decrypted_player_party()
    party[0]["species"] = 1
    party[0]["moves"] = [33, 45, 22, 73]
    party[0]["pp"] = [35, 40, 10, 10]
    bridge.configure_party(party)
    bridge.configure_inventory({339: 1, 341: 1, 342: 1, 364: 1})
    bridge.write_memory(bridge.memory_map.main + 4, struct.pack("<I", field.symbols["CB2_Overworld"].address | 1))
    bridge.write_memory(bridge.memory_map.tasks, bytes(16 * 40))
    return bridge, field


def _set_native_task(field: FieldMenuController, task: str) -> None:
    bridge = field.bridge
    bridge.write_memory(bridge.memory_map.tasks, bytes(16 * 40))
    bridge.write_memory(bridge.memory_map.tasks, struct.pack("<I", field.symbols[task].address | 1) + b"\x01")


@pytest.mark.native
@pytest.mark.parametrize("game", ["firered", "emerald"])
def test_party_observations_restore_stable_battle_indexes_only_during_party_ui(
        game: Game, monkeypatch: pytest.MonkeyPatch) -> None:
    bridge, field = _field_fixture(game)
    try:
        first = bridge.get_decrypted_player_party()[0]
        second = dict(first, pid=1234, species=283)
        bridge.configure_party([first, second])
        monkeypatch.setattr(bridge, "is_in_battle", lambda: (True, False))
        _set_native_task(field, "Task_HandleChooseMonInput")
        bridge.write_memory(field.symbols["gBattlePartyCurrentOrder"].address, b"\x10\x23\x45")
        normalized = player_party_in_battle_order(bridge)
        assert [mon["pid"] for mon in normalized] == [second["pid"], first["pid"]]
        bridge.write_memory(bridge.memory_map.tasks, bytes(16 * 40))
        assert [mon["pid"] for mon in player_party_in_battle_order(bridge)] == [first["pid"], second["pid"]]
    finally:
        bridge.close()


@pytest.mark.native
@pytest.mark.parametrize("game", ["firered", "emerald"])
def test_field_teaching_uses_native_compatibility_and_protects_hms(game: Game) -> None:
    bridge, field = _field_fixture(game)
    try:
        # Native64-bit learnsets: Bulbasaur cannot Surf; Mudkip can.
        with pytest.raises(ValueError, match="cannot learn"):
            field.request_teach_move(341, 0)
        party = bridge.get_decrypted_player_party()
        party[0]["species"] = 16  # Pidgey cannot learn Cut either.
        second = dict(party[0])
        second["species"] = 283
        second["pid"] = 213
        second["moves"] = [33, 45, 55, 0]
        second["pp"] = [35, 40, 25, 0]
        bridge.configure_party([party[0], second])
        assert field.request_required_move(57)
        assert field.pending is not None and field.pending.party_index == 1
        assert field.pending.forget_slot == 3  # A free slot takes priority.
        field.pending = None
        second["moves"] = [57, 33, 45, 55]
        second["pp"] = [15, 35, 40, 25]
        bridge.configure_party([party[0], second])
        with pytest.raises(ValueError, match="HM cannot be forgotten"):
            field.request_teach_move(342, 1, 0)
        second["is_egg"] = True
        bridge.configure_party([second])
        with pytest.raises(ValueError, match="Egg cannot learn"):
            field.request_teach_move(341, 0)
    finally:
        bridge.close()


@pytest.mark.native
@pytest.mark.parametrize("game", ["firered", "emerald"])
def test_exact_native_forget_task_drives_persistent_cursor_without_frames(game: Game) -> None:
    bridge, field = _field_fixture(game)
    try:
        field.request_teach_move(339, 0, 2)
        name = "Task_InputHandler_SelectOrForgetMove" if game == "firered" else "Task_HandleReplaceMoveInput"
        _set_native_task(field, name)
        pointer = 0x02020000
        bridge.write_memory(field.symbols["sMonSummaryScreen"].address, struct.pack("<I", pointer))
        cursor_address = field.symbols["sMoveSelectionCursorPos"].address if game == "firered" else pointer + 0x40C6
        bridge.write_memory(cursor_address, b"\x03")
        if game == "firered":
            bridge.write_memory(pointer + 0x3288, b"\x01")
            assert field.poll_key_mask() == 0  # Fade/cry initialization does not accept keys.
            bridge.write_memory(pointer + 0x3288, b"\x02")
        frame = bridge.frame_count
        assert field.poll_key_mask() == bridge.KEY_UP
        assert bridge.frame_count == frame
        bridge.write_memory(cursor_address, b"\x02")
        assert field.poll_key_mask() == bridge.KEY_A
        bridge.write_memory(cursor_address, b"\x00")
        assert field.poll_key_mask() == bridge.KEY_DOWN
        if game == "emerald":
            _set_native_task(field, "Task_HandleInputCantForgetHMsMoves")
            assert field.poll_key_mask() == bridge.KEY_DOWN
        bridge.write_memory(field.symbols["sMonSummaryScreen"].address, struct.pack("<I", 0x0203F000))
        with pytest.raises(ValueError, match="outside EWRAM"):
            field.poll_key_mask()
    finally:
        bridge.close()


@pytest.mark.native
def test_firered_tm_case_scroll_and_context_are_observed() -> None:
    bridge, field = _field_fixture("firered")
    try:
        # A real native pocket descriptor: TM01, HM01, HM03 in pocket3.
        key = bridge.read_u32(bridge.get_save_block2_dynamic_ptr() + bridge.memory_map.encryption_key_offset) & 65535
        bridge.write_memory(bridge.memory_map.bag_pockets, bytes(5 * 8))
        pointer = 0x02021000
        items = (289, 339, 341)
        bridge.write_memory(pointer, b"".join(struct.pack("<HH", item, 1 ^ key) for item in items))
        bridge.write_memory(bridge.memory_map.bag_pockets + 3 * 8, struct.pack("<IB3x", pointer, len(items)))
        field.request_teach_move(339, 0, 1)
        _set_native_task(field, "Task_HandleListInput")
        case = field.symbols["sTMCaseStaticResources"].address
        bridge.write_memory(case + 8, struct.pack("<HH", 0, 2))
        assert field.poll_key_mask() == bridge.KEY_UP  # Desired HM01 is row1, current absolute row2.
        bridge.write_memory(case + 8, struct.pack("<HH", 0, 1))
        assert field.poll_key_mask() == bridge.KEY_A
        _set_native_task(field, "Task_ContextMenu_HandleInput")
        bridge.write_memory(field.symbols["sMenu"].address + 2, b"\x01")
        assert field.poll_key_mask() == bridge.KEY_UP
        bridge.write_memory(field.symbols["sMenu"].address + 2, b"\x00")
        assert field.poll_key_mask() == bridge.KEY_A
    finally:
        bridge.close()


@pytest.mark.native
@pytest.mark.parametrize("game", ["firered", "emerald"])
def test_teaching_canceled_summary_retries_instead_of_accepting_stop(game: Game) -> None:
    bridge, field = _field_fixture(game)
    try:
        field.request_teach_move(339, 0, 1)
        _set_native_task(field, "Task_HandleStopLearningMoveYesNoInput")
        cursor = field.symbols["sMenu"].address + 2
        bridge.write_memory(cursor, b"\x00")
        assert field.poll_key_mask() == bridge.KEY_DOWN  # No, keep learning.
        bridge.write_memory(cursor, b"\x01")
        assert field.poll_key_mask() == bridge.KEY_A
        _set_native_task(field, "Task_HandleReplaceMoveYesNoInput")
        assert field.poll_key_mask() == bridge.KEY_UP  # Yes, replace a move.
        # Success comes exclusively from the decrypted physical party, not clicking Yes.
        assert field.last_result == "pending"
        party = bridge.get_decrypted_player_party()
        party[0]["moves"][1] = 15
        bridge.configure_party(party)
        field.poll_key_mask()
        assert field.last_result == "learned" and field.pending is None
    finally:
        bridge.close()


@pytest.mark.native
@pytest.mark.parametrize("game", ["firered", "emerald"])
def test_field_checkpoint_restores_intent_atomically(game: Game) -> None:
    bridge, field = _field_fixture(game)
    try:
        field.request_teach_move(339, 0, 1)
        saved = field.state_dict()
        restored = FieldMenuController(bridge, game, field.catalog)
        restored.load_state_dict(saved)
        assert restored.state_dict() == saved
        malformed = dict(saved)
        malformed["menu_choice"] = "1"
        with pytest.raises(ValueError, match="integer"):
            restored.load_state_dict(malformed)
        assert restored.state_dict() == saved
        malformed = dict(saved)
        malformed["closing"] = "false"
        with pytest.raises(ValueError, match="status"):
            restored.load_state_dict(malformed)
        assert restored.state_dict() == saved
    finally:
        bridge.close()


def test_native_checksum_retry_policy_neutral_frames_do_not_use_invalid_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a native transient read over the real physical-frame laboratory."""
    controller = build_laboratory(map_count=4)
    try:
        original = controller.bridge.get_decrypted_player_party
        failures = 2

        def temporarily_invalid() -> list[dict[str, Any]]:
            nonlocal failures
            if failures:
                failures -= 1
                raise ValueError("Party contains an invalid checksum")
            return original()

        controller.native = True  # Exercise the retry policy, without claiming native execution.
        monkeypatch.setattr(controller.bridge, "get_decrypted_player_party", temporarily_invalid)
        for _ in range(2):
            frame = controller.bridge.frame_count
            controller.step_vblank_cycle()
            assert controller.bridge.frame_count == frame + 1
            assert controller.metrics.neural_plans == controller.metrics.neural_decisions == 0
            assert not controller.movement.pending
        controller.step_vblank_cycle()
        assert controller._snapshot_failures == 0
        assert controller.metrics.neural_plans == 1
        assert sum(event["kind"] == "memory_snapshot_retry" for event in controller.audit) == 1
    finally:
        controller.close()


def test_persistent_native_checksum_error_stops_after_bounded_neutral_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = build_laboratory(map_count=4)
    try:
        def invalid() -> list[dict[str, Any]]:
            raise ValueError("Party contains an invalid checksum")

        controller.native = True
        monkeypatch.setattr(controller.bridge, "get_decrypted_player_party", invalid)
        controller.run_frames(120)
        frame = controller.bridge.frame_count
        with pytest.raises(ValueError, match="remained invalid"):
            controller.step_vblank_cycle()
        assert controller.bridge.frame_count == frame
        assert controller.metrics.neural_plans == controller.metrics.neural_decisions == 0
    finally:
        controller.close()
