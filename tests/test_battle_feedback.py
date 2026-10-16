"""A visible immunity must change the next action, including zero HP deltas."""
from __future__ import annotations

from pathlib import Path
import struct
from typing import Any, cast

import pytest
import torch

from core.bridge.synthetic_gba_silicon import SyntheticEncounter, SyntheticGBASiliconBridge, SyntheticMove
from core.game_data import load_symbols
from core.orchestrator import AresAutonomousOrchestrator
from core.field_controls import FieldMenuController
from scripts.run_autonomous_verification_demo import build_laboratory


def _wingull_battle() -> AresAutonomousOrchestrator:
    torch.set_num_threads(2)
    bridge = SyntheticGBASiliconBridge(True)
    bridge.configure_move(SyntheticMove(341, 55, 4, max_pp=15))
    own = bridge.get_decrypted_player_party()[0]
    own.update(species=283, level=18, hp=58, maxHP=58, attack=37, defense=34,
               spAttack=29, spDefense=31, speed=25, moves=[341, 55, 33, 0], pp=[15, 25, 35, 0])
    bridge.configure_party([own])
    rival = dict(own, species=309, level=13, hp=35, maxHP=35, types=[11, 2],
                 pid=42, attack=20, defense=18, spDefense=18, speed=30, moves=[33, 0, 0, 0], pp=[35, 0, 0, 0])
    bridge.start_battle(SyntheticEncounter(rival, trainer=True))
    return AresAutonomousOrchestrator(bridge=bridge, online_learning=False)


def test_visible_no_effect_removes_ground_attacks_without_a_type_oracle() -> None:
    controller = _wingull_battle()
    try:
        own, rival = controller.bridge.get_battle_pokemon(0), controller.bridge.get_battle_pokemon(1)
        controller._observe_no_effect(rival, 341)
        estimates = controller._outgoing_damage_estimates(own, rival)
        assert estimates[0].expected_damage == 0 and estimates[1].expected_damage > 0
        decision = controller.choose_battle_decision()
        assert decision.action == "move" and decision.index != 0
        assert controller._fallback_battle_decision(own, "fixture").index != 0
        # No type/ability cause was identified; no false causal proof is added.
        assert controller.causal.intervention_count.sum() == 0
    finally:
        controller.close()


@pytest.mark.parametrize("flags,current_move,attacker,should_learn", [
    (8, 341, 0, True), (1, 341, 0, False), (0x29, 341, 0, False),
    (8, 33, 0, False), (8, 341, 1, False),
])
def test_native_immunity_disclosure_needs_a_current_own_intervention(
        flags: int, current_move: int, attacker: int, should_learn: bool) -> None:
    controller = _wingull_battle()
    try:
        bridge = controller.bridge
        controller._symbols = load_symbols("emerald")
        for name, payload in {
            "gBattlersCount": b"\x02", "gBattlerPositions": b"\x00\x01",
            "gBattlerAttacker": bytes([attacker]), "gBattlerTarget": b"\x01",
            "gMoveResultFlags": bytes([flags]), "gCurrentMove": struct.pack("<H", current_move),
        }.items():
            bridge.write_memory(controller._symbols[name].address, payload)
        controller._native_interventions[0] = (341, bridge.frame_count)
        enemy_hp = int(bridge.get_battle_pokemon(1).hp)
        controller._process_native_battle_disclosures()
        assert bool(controller._observed_no_effect) == should_learn
        assert int(bridge.get_battle_pokemon(1).hp) == enemy_hp
    finally:
        controller.close()


def test_resume_keeps_the_observed_immunity_of_an_unfinished_battle(tmp_path: Path) -> None:
    controller = _wingull_battle()
    second = _wingull_battle()
    try:
        controller.checkpoint_dir = tmp_path
        controller._observe_no_effect(controller.bridge.get_battle_pokemon(1), 341)
        checkpoint = controller.save_checkpoint("immunity_regression")
        second.load_checkpoint(checkpoint)
        assert second._observed_no_effect == controller._observed_no_effect
        assert second.choose_battle_decision().index != 0
    finally:
        controller.close()
        second.close()


@pytest.mark.parametrize("mode,trainer,forced,attempts,wanted,escapes", [
    ("EXECUTE_MILESTONE", False, False, 0, False, True),
    ("EXECUTE_MILESTONE", True, False, 0, False, False),
    ("OPTIMAL_TRAINING_CURRICULUM", False, False, 0, False, False),
    ("EXECUTE_MILESTONE", False, True, 0, False, False),
    ("EXECUTE_MILESTONE", False, False, 2, False, False),
    ("EXECUTE_MILESTONE", False, False, 0, True, False),
])
def test_optional_wild_battles_do_not_replace_story_progress(
        mode: str, trainer: bool, forced: bool, attempts: int, wanted: bool, escapes: bool) -> None:
    controller = _wingull_battle()
    try:
        controller.native = True
        controller._last_directive = {"mode": mode}
        controller._battle_escape_attempts = attempts
        own, enemy = controller.bridge.get_battle_pokemon(0), controller.bridge.get_battle_pokemon(1)
        if wanted:
            controller.desired_species.add(int(enemy.species))
        decision = controller._story_escape_decision(own, enemy, trainer, forced, 0.5)
        assert (decision is not None) == escapes
        if decision is not None:
            assert decision.action == "run" and not decision.proof.formally_safe
            controller._audit_decision(decision, own, enemy)
            assert controller._battle_escape_attempts == 1
    finally:
        controller.close()


def test_two_tile_interaction_requires_a_real_counter() -> None:
    controller = _wingull_battle()
    try:
        bridge = cast(SyntheticGBASiliconBridge, controller.bridge)
        directive: dict[str, Any] = {"action": "interact", "dest": (0, 0, (2, 2)), "npc_local_id": 1}
        assert (2, 4) not in controller._goal_tiles(directive, (2, 2))
        tiles, collisions, elevations = bridge.extract_live_map_matrices()
        tiles = tiles.copy()
        tiles[3, 2] = int(tiles.max()) + 1
        behaviors = bridge.extract_behavior_map().copy()
        behaviors[3, 2] = 0x80
        bridge.configure_map(0, 0, tiles, collisions, elevations, behaviors)
        bridge.transition_map(0, 0, 1, 1, relocate=False)
        assert (2, 4) in controller._goal_tiles(directive, (2, 2))
        assert (4, 2) not in controller._goal_tiles(directive, (2, 2))
    finally:
        controller.close()


def test_finished_npc_cannot_reopen_dialogue_before_the_planner_observes_progress(
        monkeypatch: pytest.MonkeyPatch) -> None:
    controller = build_laboratory(map_count=4)

    class Field:
        busy = False
        menu_choice = 0

        def interaction_key(self, local_id: int) -> int | None:
            raise AssertionError("A completed objective must release NPC interception")

    class PlannerReached(Exception):
        pass

    def next_stage() -> None:
        raise PlannerReached

    try:
        controller.field = cast(FieldMenuController, Field())
        task = controller.planner.campaign_dag[0]
        controller.bridge.set_event_flag(task.completion_flag_id, True)
        coords = controller.bridge.get_player_coords_and_map()
        controller._last_directive = {"mode": "EXECUTE_MILESTONE", "milestone": task.milestone_id,
            "dest": (coords["map_group"], coords["map_num"], (3, 2)), "npc_local_id": 1}
        monkeypatch.setattr(controller.movement, "tick", next_stage)
        with pytest.raises(PlannerReached):
            controller._execute_overworld_cycle()
    finally:
        controller.close()


def test_species_stat_prior_is_rebuilt_for_the_visible_opponent_level() -> None:
    controller = _wingull_battle()
    try:
        enemy = controller.bridge.get_battle_pokemon(1)
        weak = controller._ensure_belief(enemy)
        enemy.level = 30
        stronger = controller._ensure_belief(enemy)
        assert stronger.expected_stat("defense") > weak.expected_stat("defense")
        assert stronger.expected_stat("attack") > weak.expected_stat("attack")
        # Hidden raw stats do not change the species/level prior.
        enemy.attack, enemy.defense, enemy.speed = 65535, 65535, 65535
        assert controller._ensure_belief(enemy) is stronger
        assert controller._belief_levels[int(enemy.species)] == 30
    finally:
        controller.close()


def test_visible_bulk_up_changes_physical_estimate_and_torrent_changes_water_damage() -> None:
    controller = _wingull_battle()
    try:
        own, enemy = controller.bridge.get_battle_pokemon(0), controller.bridge.get_battle_pokemon(1)
        before = controller._outgoing_damage_estimates(own, enemy)
        enemy.statStages[2] = 12
        after = controller._outgoing_damage_estimates(own, enemy)
        assert after[0].damage_on_hit < before[0].damage_on_hit  # physical Mud Shot vs boosted defense
        assert after[1].damage_on_hit == before[1].damage_on_hit  # special Water Gun
        own.ability, own.hp = 67, 10  # own, known Torrent at low HP
        torrent = controller._outgoing_damage_estimates(own, enemy)
        assert torrent[1].damage_on_hit > after[1].damage_on_hit
        assert torrent[0].damage_on_hit == after[0].damage_on_hit
    finally:
        controller.close()


def test_visible_accuracy_and_speed_changes_affect_the_plan() -> None:
    controller = _wingull_battle()
    try:
        own, enemy = controller.bridge.get_battle_pokemon(0), controller.bridge.get_battle_pokemon(1)
        before = controller._outgoing_damage_estimates(own, enemy)
        own.statStages[6] = 0
        after = controller._outgoing_damage_estimates(own, enemy)
        assert after[0].accuracy < before[0].accuracy
        baseline = controller._effective_speed(30, enemy)
        enemy.statStages[3] = 4
        assert controller._effective_speed(30, enemy) < baseline
    finally:
        controller.close()
