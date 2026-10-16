"""Never-stuck guarantees: scene ownership, fault containment and recovery escalation."""
from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any, cast

import pytest
import torch

from core.battle.menu_cursor_controller import BattleMenuCursorController
from core.bridge.ffi_mgba import MenuState
from core.bridge.synthetic_gba_silicon import SyntheticGBASiliconBridge
from core.field_controls import FieldMenuController
from core.game_data import Game, ROOT, ROM_NAMES, RomCatalog
from core.orchestrator import AresAutonomousOrchestrator
from core.progress_watchdog import CampaignStalled, ProgressObservation, ProgressWatchdog, RecoveryRequest
from scripts.run_autonomous_verification_demo import build_laboratory


class _NativeShapedBridge(SyntheticGBASiliconBridge):
    """Native screen discovery (tasks, callbacks) over bounded synthetic RAM."""

    def menu_state(self) -> MenuState:
        battle, _ = self.is_in_battle()
        action, move = self.get_battle_cursors()
        return MenuState(battle, action, move, 0, False, "battle" if battle else "overworld")


def _evolution_fixture(game: Game) -> tuple[_NativeShapedBridge, FieldMenuController, BattleMenuCursorController]:
    if not (ROOT.parent / ROM_NAMES[game]).is_file():
        pytest.skip("Native evolution ABI checks require the matching supplied ROM")
    bridge = _NativeShapedBridge(game == "emerald")
    field = FieldMenuController(bridge, game, RomCatalog(game))
    menu = BattleMenuCursorController(bridge, pulse_frames=1, release_frames=1)
    bridge.write_memory(bridge.memory_map.tasks, bytes(16 * 40))
    bridge.write_memory(field.symbols["sTextPrinters"].address, bytes(36 * 32))
    bridge.write_memory(menu._native_symbols["gBattlerControllerFuncs"], bytes(16))
    # gMain.inBattle stays set during post-battle evolution (ReturnFromBattleToOverworld clears it).
    bridge.write_memory(bridge.memory_map.main + 0x439, b"\x02")
    bridge.write_memory(bridge.memory_map.main + 4, struct.pack("<I", field.symbols["CB2_EvolutionSceneUpdate"].address | 1))
    party = bridge.get_decrypted_player_party()
    party[0].update(species=284, moves=[33, 117, 189, 55], pp=[35, 10, 10, 25])
    bridge.configure_party(party)
    return bridge, field, menu


def _evolution_task(bridge: SyntheticGBASiliconBridge, field: FieldMenuController, state: int,
                    learn_state: int = 0, yes_state: int = 0) -> None:
    bridge.write_memory(bridge.memory_map.tasks, bytes(16 * 40))
    bridge.write_memory(bridge.memory_map.tasks, struct.pack("<IB3x", field.symbols["Task_EvolutionScene"].address | 1, 1))
    bridge.write_memory(bridge.memory_map.tasks + 8, struct.pack("<H", state))
    bridge.write_memory(bridge.memory_map.tasks + 20, struct.pack("<HH", learn_state, yes_state))
    bridge.write_memory(bridge.memory_map.tasks + 28, struct.pack("<H", 0))


def _pulse_frame(bridge: SyntheticGBASiliconBridge) -> None:
    while bridge.frame_count % 8 >= 2:
        bridge.step_frame()


@pytest.mark.native
@pytest.mark.parametrize("game", ["firered", "emerald"])
def test_post_battle_evolution_yes_no_is_owned_by_the_scene_controller(game: Game) -> None:
    """The Emerald stall: 'Delete a move to make room for MUD SHOT?' while inBattle is set."""
    bridge, field, menu = _evolution_fixture(game)
    try:
        _evolution_task(bridge, field, state=22, learn_state=4, yes_state=5)
        bridge.write_memory(field.symbols["gMoveToLearn"].address, struct.pack("<H", 341))
        bridge.write_memory(field.symbols["gBattleCommunication"].address + 1, b"\x01")
        # Battle input must stay neutral instead of treating the prompt as an animation.
        assert menu.menu_mode() == "evolution"
        assert menu.tick() == 0 and menu.recovery_key() is None
        _pulse_frame(bridge)
        assert field.scene_keys() == bridge.KEY_UP  # Yes: Mud Shot is worth a slot
        bridge.write_memory(field.symbols["gBattleCommunication"].address + 1, b"\x00")
        _pulse_frame(bridge)
        assert field.scene_keys() == bridge.KEY_A
        assert field.last_learning_choice is not None and field.last_learning_choice.forget_slot == 1
        assert field.evolution_active()
    finally:
        bridge.close()


@pytest.mark.native
@pytest.mark.parametrize("game", ["firered", "emerald"])
def test_evolution_summary_screen_selects_the_slot_and_never_presses_b(game: Game) -> None:
    bridge, field, _ = _evolution_fixture(game)
    try:
        _evolution_task(bridge, field, state=22, learn_state=6, yes_state=5)
        bridge.write_memory(field.symbols["gMoveToLearn"].address, struct.pack("<H", 341))
        # The summary screen task runs while Task_EvolutionScene stays alive.
        name = "Task_InputHandler_SelectOrForgetMove" if game == "firered" else "Task_HandleReplaceMoveInput"
        bridge.write_memory(bridge.memory_map.tasks + 40, struct.pack("<IB3x", field.symbols[name].address | 1, 1))
        pointer = 0x02020000
        bridge.write_memory(field.symbols["sMonSummaryScreen"].address, struct.pack("<I", pointer))
        cursor = field.symbols["sMoveSelectionCursorPos"].address if game == "firered" else pointer + 0x40C6
        if game == "firered":
            bridge.write_memory(pointer + 0x3288, b"\x02")
        bridge.write_memory(cursor, b"\x03")
        seen: set[int] = set()
        for _ in range(16):
            keys = field.scene_keys()
            assert keys is not None
            seen.add(keys)
            bridge.step_frame()
        assert bridge.KEY_UP in seen and not seen & {bridge.KEY_B}
        bridge.write_memory(cursor, b"\x01")
        _pulse_frame(bridge)
        assert field.scene_keys() == bridge.KEY_A  # Forget Bide (slot 1)
    finally:
        bridge.close()


@pytest.mark.native
@pytest.mark.parametrize("game", ["firered", "emerald"])
def test_stale_evolution_session_cannot_hijack_later_battle_learning(game: Game) -> None:
    bridge, field, _ = _evolution_fixture(game)
    try:
        _evolution_task(bridge, field, state=13)
        assert field.scene_keys() is not None and field.state_dict()["evolution_party_index"] == 0
        bridge.write_memory(bridge.memory_map.tasks, bytes(16 * 40))
        bridge.write_memory(bridge.memory_map.main + 4, struct.pack("<I", field.symbols["BattleMainCB2"].address | 1))
        assert field.scene_keys() is None
        assert field.state_dict()["evolution_party_index"] is None
    finally:
        bridge.close()


@pytest.mark.native
@pytest.mark.parametrize("game", ["firered", "emerald"])
def test_orchestrator_routes_post_battle_evolution_before_combat(game: Game, monkeypatch: pytest.MonkeyPatch) -> None:
    bridge, field, _ = _evolution_fixture(game)
    controller = AresAutonomousOrchestrator(bridge=bridge, online_learning=False)
    try:
        controller.native, controller.field = True, field
        controller.menu = BattleMenuCursorController(bridge)
        _evolution_task(bridge, field, state=22, learn_state=4, yes_state=5)
        bridge.write_memory(field.symbols["gMoveToLearn"].address, struct.pack("<H", 341))

        def forbidden(*args: object) -> None:
            raise AssertionError("battle policy must not own an evolution scene")

        monkeypatch.setattr(controller, "_execute_combat_cycle", forbidden)
        monkeypatch.setattr(controller, "_process_disclosures", forbidden)
        _pulse_frame(bridge)
        controller._select_frame_input()
        assert controller.scene == "evolution"
        assert bridge.hardware_keys in (bridge.KEY_UP, bridge.KEY_A)
        assert any(event["kind"] == "scene_started" for event in controller.audit)
    finally:
        controller.close()


@pytest.mark.native
@pytest.mark.parametrize("game", ["firered", "emerald"])
def test_options_are_set_through_the_native_option_screen(game: Game) -> None:
    bridge, field, _ = _evolution_fixture(game)
    try:
        bridge.write_memory(bridge.memory_map.main + 0x439, b"\x00")
        bridge.write_memory(bridge.memory_map.main + 4, bytes(4))  # the option screen's own callback
        bridge.write_memory(bridge.memory_map.tasks, bytes(16 * 40))
        assert field.options_values() == (0, 0, 0) and field.options_due()
        field.request_options()
        assert field.busy
        if game == "emerald":
            task = field.symbols["Task_OptionMenuProcessInput"].address
            bridge.write_memory(bridge.memory_map.tasks, struct.pack("<IB3x", task | 1, 1))
            data = bridge.memory_map.tasks + 8

            def option_state(row: int, values: tuple[int, int, int]) -> None:
                bridge.write_memory(data, struct.pack("<4H", row, *values))
        else:
            task = field.symbols["Task_OptionMenu"].address
            bridge.write_memory(bridge.memory_map.tasks, struct.pack("<IB3x", task | 1, 1))
            pointer = 0x02021000
            bridge.write_memory(field.symbols["sOptionMenuPtr"].address, struct.pack("<I", pointer))

            def option_state(row: int, values: tuple[int, int, int]) -> None:
                bridge.write_memory(pointer, struct.pack("<3H", *values))
                bridge.write_memory(pointer + 0x0E, struct.pack("<HB", row, 2))

        def keys_for(row: int, values: tuple[int, int, int]) -> int | None:
            option_state(row, values)
            _pulse_frame(bridge)
            return field.poll_key_mask()

        assert keys_for(0, (0, 0, 0)) == bridge.KEY_RIGHT  # text speed toward FAST
        assert keys_for(0, (2, 0, 0)) == bridge.KEY_DOWN  # next row
        assert keys_for(6, (2, 0, 0)) == bridge.KEY_UP
        assert keys_for(1, (2, 0, 0)) == bridge.KEY_RIGHT  # battle scene OFF
        assert keys_for(2, (2, 1, 0)) == bridge.KEY_RIGHT  # battle style SET
        assert keys_for(2, (2, 1, 1)) == bridge.KEY_B  # B saves and exits
        # Success is read back from SaveBlock2, never assumed from key presses.
        save2 = bridge.get_save_block2_dynamic_ptr()
        bridge.write_memory(save2 + 0x14, struct.pack("<H", 2 | 1 << 9 | 1 << 10))
        _pulse_frame(bridge)
        field.poll_key_mask()
        assert not field.options_pending and field.last_result == "options_set" and not field.options_due()
        restored = FieldMenuController(bridge, game, field.catalog)
        restored.load_state_dict(field.state_dict())
        assert restored.state_dict() == field.state_dict()
    finally:
        bridge.close()


@pytest.mark.native
def test_looping_multichoice_is_answered_differently_until_it_stops_repeating() -> None:
    """Emerald's Match Call tutorial re-asks until option 3 (POKéNAV) is chosen."""
    bridge, field, _ = _evolution_fixture("emerald")
    try:
        bridge.write_memory(bridge.memory_map.main + 0x439, b"\x00")
        bridge.write_memory(bridge.memory_map.main + 4, struct.pack("<I", field.symbols["CB2_Overworld"].address | 1))
        label = field.symbols["RustboroCity_EventScript_MatchCallTutorial"].address
        bridge.write_memory(bridge.memory_map.script + 8, struct.pack("<I", label + 7))
        menu = field.symbols["sMenu"].address
        task = field.symbols["Task_HandleMultichoiceInput"].address

        def ask(cursor: int) -> int | None:
            bridge.write_memory(bridge.memory_map.tasks, struct.pack("<IB3x", task | 1, 1))
            bridge.write_memory(menu + 2, bytes((cursor, 0, 7)))  # cursor, min, max: eight options
            _pulse_frame(bridge)
            return field.poll_key_mask()

        def answered() -> None:
            bridge.write_memory(bridge.memory_map.tasks, bytes(16 * 40))
            for _ in range(8):
                bridge.step_frame()
            field.poll_key_mask()  # the question disappeared; the next one is fresh

        assert ask(0) == bridge.KEY_A  # first time: the directive's choice (0)
        answered()
        assert ask(0) == bridge.KEY_DOWN and field.last_result.endswith(":1")
        answered()
        ask(0)
        answered()
        assert ask(2) == bridge.KEY_DOWN and ask(3) == bridge.KEY_A  # fourth appearance: option 3
    finally:
        bridge.close()


@pytest.mark.native
def test_pokenav_tutorial_is_completed_from_its_own_ram_state() -> None:
    bridge, field, _ = _evolution_fixture("emerald")
    try:
        bridge.write_memory(bridge.memory_map.main + 0x439, b"\x00")
        bridge.write_memory(bridge.memory_map.main + 4, struct.pack("<I", field.symbols["CB2_Pokenav"].address | 1))
        bridge.write_memory(bridge.memory_map.tasks, bytes(16 * 40))
        resources, menu, match_call = 0x02021000, 0x02021100, 0x02021200
        bridge.write_memory(field.symbols["gPokenavResources"].address, struct.pack("<I", resources))

        def state(mode: int, menu_item: int | None = None, handler: str | None = None, option: int = 0) -> int | None:
            bridge.write_memory(resources, bytes(16 + 4 * 21))
            bridge.write_memory(resources + 8, struct.pack("<H", mode))
            if menu_item is not None:
                bridge.write_memory(resources + 16 + 4, struct.pack("<I", menu))
                bridge.write_memory(menu, struct.pack("<HhH", 2, menu_item, menu_item))
            if handler is not None:
                bridge.write_memory(resources + 16 + 20, struct.pack("<I", match_call))
                bridge.write_memory(match_call, struct.pack("<H", option))
                bridge.write_memory(match_call + 24, struct.pack("<I", field.symbols[handler].address | 1))
            _pulse_frame(bridge)
            return field.poll_key_mask()

        assert state(1, menu_item=0) == bridge.KEY_DOWN  # toward MATCH CALL
        assert state(1, menu_item=2) == bridge.KEY_A
        assert state(1, handler="CB2_HandleMatchCallInput") == bridge.KEY_A  # pick a contact
        assert state(1, handler="CB2_HandleMatchCallOptionsInput", option=1) == bridge.KEY_UP
        assert state(1, handler="CB2_HandleMatchCallOptionsInput", option=0) == bridge.KEY_A  # CALL
        assert state(2, handler="CB2_HandleCallExitInput") == bridge.KEY_A
        assert state(2, handler="CB2_HandleMatchCallInput") == bridge.KEY_B  # leave the list
        assert state(2, menu_item=2) == bridge.KEY_DOWN and state(2, menu_item=4) == bridge.KEY_A  # SWITCH OFF
        assert state(0, menu_item=0) == bridge.KEY_B  # ordinary PokéNav: just close it
    finally:
        bridge.close()


def test_recurring_identical_stall_rewinds_deeper_and_varies_inputs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.set_num_threads(2)
    controller = build_laboratory(map_count=4, checkpoint_dir=tmp_path)
    try:
        # This test isolates rewind depth within one story state. Completing
        # laboratory milestones between snapshots would correctly forbid
        # rewinding to the earlier story state.
        monkeypatch.setattr(controller, "_select_frame_input", lambda: controller.bridge.set_hardware_keys(0))
        for _ in range(3):
            controller.save_checkpoint("periodic")
            controller.run_frames(30)
        progress_frame = controller.bridge.frame_count
        frames = sorted(frame for frame in (json.loads((path / "manifest.json").read_text(encoding="utf-8"))["frame"]
                                            for path in tmp_path.glob("frame-*")) if frame <= progress_frame)
        assert len(frames) >= 3
        # The identical stall (same objective, screen and place) recurs after each rewind.
        coords = controller.bridge.get_player_coords_and_map()
        targets = []
        for _ in range(3):
            controller._recover(_request(5, progress_frame), "script", False, coords, "objective")
            targets.append(controller.bridge.frame_count)
        # Each repetition of the same stall goes one checkpoint further back.
        assert targets == [frames[-1], frames[-2], frames[-3]]
        key = controller._stall_key("objective", "script", coords)
        probes = []
        for recurrence in (1, 2):
            controller._stall_memory[key] = recurrence
            controller._recover(_request(2, progress_frame), "script", False, coords, "objective")
            probes.append(list(controller._probe_keys))
        assert probes[0] != probes[1]
    finally:
        controller.close()


def test_forced_replacement_after_a_faint_chooses_a_switch_without_faulting() -> None:
    """A fainted active battler made the policy assess a 0-HP defender and stall."""
    from core.bridge.synthetic_gba_silicon import SyntheticEncounter
    controller = build_laboratory(map_count=4)
    try:
        bridge = cast(SyntheticGBASiliconBridge, controller.bridge)
        lead = bridge.get_decrypted_player_party()[0]
        reserve = dict(lead, pid=lead["pid"] + 1, species=16, moves=[33, 0, 0, 0], pp=[35, 0, 0, 0])
        bridge.configure_party([lead, reserve])
        opponent = {"species": 19, "pid": 7, "otid": 1, "level": 3, "hp": 12, "maxHP": 12, "attack": 8,
                    "defense": 8, "speed": 8, "spAttack": 8, "spDefense": 8, "moves": [33, 0, 0, 0], "pp": [35, 0, 0, 0],
                    "types": [0, 0], "status": 0}
        bridge.start_battle(SyntheticEncounter(opponent, catch_rate=255))
        fainted = bridge.get_battle_pokemon(0)
        fainted.hp = 0
        bridge.write_memory(bridge.memory_map.battle_mons, bytes(fainted))
        decision = controller.choose_battle_decision()
        assert decision.action == "switch" and decision.index == 1
        assert controller.metrics.faults == 0
    finally:
        controller.close()


def test_policy_fault_is_contained_and_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.set_num_threads(2)
    controller = build_laboratory(map_count=4)
    controller.fault_tolerant = True
    try:
        def broken() -> None:
            raise ValueError("category must follow the Gen-3 type category split")

        monkeypatch.setattr(controller, "_execute_overworld_cycle", broken)
        for _ in range(3):
            frame = controller.bridge.frame_count
            controller.step_vblank_cycle()
            assert controller.bridge.frame_count == frame + 1
        assert controller.metrics.faults == 3
        faults = [event for event in controller.audit if event["kind"] == "agent_fault"]
        assert len(faults) == 1  # repeats are throttled, not lost: the counter keeps them
        assert "Gen-3" in str(controller.last_fault["error"])
        controller.fault_tolerant = False
        with pytest.raises(ValueError, match="Gen-3"):
            controller.step_vblank_cycle()
    finally:
        controller.close()


def test_probe_keys_take_precedence_with_release_edges() -> None:
    controller = build_laboratory(map_count=4)
    try:
        controller._queue_probe([controller.bridge.KEY_A], hold=2, gap=2)
        emitted = []
        for _ in range(4):
            controller._select_frame_input()
            emitted.append(controller.bridge.hardware_keys)
            controller.bridge.step_frame()
        assert emitted == [controller.bridge.KEY_A, controller.bridge.KEY_A, 0, 0]
        assert controller.metrics.probes == 1
    finally:
        controller.close()


def _request(attempt: int, progress_frame: int, rewinds: int = 0) -> RecoveryRequest:
    return RecoveryRequest(attempt, "field", "stationary", 600, 600, rewinds, progress_frame)


def test_exhausted_ladder_rewinds_to_a_pre_stall_checkpoint(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.set_num_threads(2)
    controller = build_laboratory(map_count=4, checkpoint_dir=tmp_path)
    try:
        monkeypatch.setattr(controller, "_select_frame_input", lambda: controller.bridge.set_hardware_keys(0))
        early = controller.save_checkpoint("periodic")
        controller.run_frames(120)
        progress_frame = controller.bridge.frame_count
        controller.run_frames(60)
        controller.metrics.recoveries, controller.metrics.faults = 7, 3
        frames = [json.loads((path / "manifest.json").read_text(encoding="utf-8"))["frame"]
                  for path in tmp_path.glob("frame-*")]
        expected = max(frame for frame in frames if frame <= progress_frame)
        assert expected >= int(early.name.split("-")[1])
        coords = controller.bridge.get_player_coords_and_map()
        controller._recover(_request(5, progress_frame), "overworld", False, coords, "objective")
        # The newest bundle saved no later than the last gain, never a post-stall one.
        assert controller.bridge.frame_count == expected <= progress_frame
        assert controller.watchdog.rewinds == 1 and controller.metrics.rewinds == 1
        assert controller.metrics.recoveries >= 7 and controller.metrics.faults >= 3  # diagnostics survive
        assert any(event["kind"] == "stall_rewind" for event in controller.audit)
        with pytest.raises(CampaignStalled, match="rewinds"):
            controller._recover(_request(5, progress_frame, rewinds=controller.max_rewinds),
                                "overworld", False, coords, "objective")
    finally:
        controller.close()


def test_without_any_checkpoint_the_ladder_ends_with_campaign_stalled() -> None:
    controller = build_laboratory(map_count=4)
    try:
        coords = controller.bridge.get_player_coords_and_map()
        with pytest.raises(CampaignStalled):
            controller._recover(_request(5, 0), "overworld", False, coords, "objective")
    finally:
        controller.close()


def test_watchdog_rewind_counter_resets_only_on_real_gains() -> None:
    watch = ProgressWatchdog(stationary_limits={"field": 60, "menu": 60, "battle": 60, "bootstrap": 60},
                             cycle_limit=600, cooldown=30)

    def seen(frame: int, xp: int = 0, xy: tuple[int, int] = (1, 1)) -> RecoveryRequest | None:
        return watch.observe(ProgressObservation(frame, "field", ("overworld", *xy), (xp,), (0, 0, *xy), "goal"))

    seen(0)
    watch.rewinds = 2
    seen(500, xy=(2, 1))
    watch.rewinds = 2
    # A rewind moves frames backwards: deadlines restart but visited tiles persist.
    seen(100, xy=(1, 1))
    assert watch.rewinds == 2 and (0, 0, 2, 1) in watch.visited
    seen(130, xy=(2, 1))
    assert watch.rewinds == 2  # an abandoned timeline's tile is not exploration
    seen(160, xp=5)
    assert watch.rewinds == 0 and watch.attempts == 0
    restored = ProgressWatchdog()
    restored.load_state_dict(watch.state_dict())
    assert restored.state_dict() == watch.state_dict()


def test_checkpoint_retention_keeps_recent_durable_and_pinned(tmp_path: Path) -> None:
    controller = build_laboratory(map_count=4, checkpoint_dir=tmp_path, )
    controller.retain_checkpoints = 3
    try:
        milestone = controller.save_checkpoint("milestone")
        pinned = controller.save_checkpoint("periodic")
        (pinned / "PINNED").write_text("regression fixture", encoding="utf-8")
        for _ in range(6):
            controller.step_vblank_cycle()
            controller.save_checkpoint("periodic")
        remaining = sorted(path.name for path in tmp_path.glob("frame-*"))
        assert milestone.name in remaining and pinned.name in remaining
        assert len(remaining) == 5  # three newest + durable milestone + pinned fixture
        latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))["directory"]
        assert latest in remaining
    finally:
        controller.close()


def test_single_frame_map_flicker_is_not_a_transition(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = build_laboratory(map_count=4)
    try:
        base = controller.bridge.get_player_coords_and_map()
        sequence = [base, dict(base, map_group=0, map_num=3), base, base]
        sequence += [dict(base, map_num=1)] * 3
        frames = iter(sequence)
        monkeypatch.setattr(controller.bridge, "get_player_coords_and_map", lambda: next(frames))
        controller._last_position = None
        for _ in range(len(sequence)):
            controller._track_map_transition()
        transitions = [event for event in controller.audit if event["kind"] == "map_transition"]
        assert len(transitions) == 1 and transitions[0]["destination"] == [0, 1]
    finally:
        controller.close()


def test_telemetry_reports_team_milestones_and_health_without_raising() -> None:
    controller = build_laboratory(map_count=4)
    try:
        controller.run_frames(30)
        data = controller.telemetry()
        party = cast(list[dict[str, Any]], data["party"])
        assert party and party[0]["level"] == 50 and party[0]["moves"]
        milestones = cast(dict[str, Any], data["milestones"])
        assert milestones["total"] == len(controller.planner.campaign_dag)
        status = controller.status()
        assert cast(dict[str, Any], status["progress_watchdog"])["max_rewinds"] == controller.max_rewinds
        json.dumps(data, default=str)
    finally:
        controller.close()
