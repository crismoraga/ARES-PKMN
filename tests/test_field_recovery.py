"""Field scene and HM activation regression checks over bounded native-shaped RAM."""
from __future__ import annotations

import struct

import pytest

from core.bridge.gba_structs import ObjectEventStruct
from core.bridge.synthetic_gba_silicon import SyntheticGBASiliconBridge
from core.field_controls import FieldMenuController
from core.game_data import Game, ROOT, ROM_NAMES, RomCatalog
from core.navigation.source_data import load_map_catalog


def field_fixture(game: Game) -> tuple[SyntheticGBASiliconBridge, FieldMenuController]:
    if not (ROOT.parent / ROM_NAMES[game]).is_file():
        pytest.skip("Native field metadata requires the matching supplied ROM")
    bridge = SyntheticGBASiliconBridge(game == "emerald")
    field = FieldMenuController(bridge, game, RomCatalog(game))
    bridge.write_memory(bridge.memory_map.tasks, bytes(16 * 40))
    bridge.write_memory(field.symbols["sTextPrinters"].address, bytes(36 * 32))
    bridge.write_memory(bridge.memory_map.main + 4, struct.pack("<I", field.symbols["CB2_Overworld"].address | 1))
    bridge.write_memory(bridge.memory_map.map_header + 0x17, b"\x01")
    return bridge, field


def install_task(field: FieldMenuController, name: str) -> None:
    field.bridge.write_memory(field.bridge.memory_map.tasks, bytes(40 * 16))
    field.bridge.write_memory(field.bridge.memory_map.tasks, struct.pack("<IB3x", field.symbols[name].address | 1, 1))


@pytest.mark.native
@pytest.mark.parametrize("game", ["firered", "emerald"])
def test_live_npc_interception_never_substitutes_another_actor(game: Game) -> None:
    bridge, field = field_fixture(game)
    try:
        coords = bridge.get_player_coords_and_map()
        x, y = coords["x"], coords["y"]
        player = bridge.get_object_events()[0]
        player.facingDirection = 2  # Peeko is north; the requested NPC is east.
        bridge.write_memory(bridge.memory_map.objects, bytes(player))
        briney = ObjectEventStruct()
        briney.flags, briney.localId = 1, 1
        briney.mapGroup, briney.mapNum = coords["map_group"], coords["map_num"]
        briney.currentCoords_x, briney.currentCoords_y = x + 8, y + 7
        briney.elevations = coords["z"]
        peeko = ObjectEventStruct.from_buffer_copy(bytes(briney))
        peeko.localId = 2
        peeko.currentCoords_x, peeko.currentCoords_y = x + 7, y + 6
        bridge.write_memory(bridge.memory_map.objects + 36, bytes(briney))
        bridge.write_memory(bridge.memory_map.objects + 72, bytes(peeko))
        before = bridge.save_state()
        assert field.interaction_key(1) == bridge.KEY_RIGHT
        assert bridge.save_state() == before
        player.facingDirection = 4
        bridge.write_memory(bridge.memory_map.objects, bytes(player))
        assert field.interaction_key(1) == bridge.KEY_A
        # A different map or elevation may not be interacted with through an
        # equally numbered stale object slot.
        briney.mapGroup = (coords["map_group"] + 1) % 256
        bridge.write_memory(bridge.memory_map.objects + 36, bytes(briney))
        assert field.interaction_key(1) is None
        briney.mapGroup = coords["map_group"]
        briney.elevations = 2 if coords["z"] != 2 else 3
        bridge.write_memory(bridge.memory_map.objects + 36, bytes(briney))
        assert field.interaction_key(1) is None
        # Missing Briney never redirects the requested conversation to Peeko.
        briney.flags = 0
        bridge.write_memory(bridge.memory_map.objects + 36, bytes(briney))
        assert field.interaction_key(1) is None
    finally:
        bridge.close()


@pytest.mark.native
def test_patrol_interception_waits_off_the_circuit_and_has_a_persisted_deadline() -> None:
    bridge, field = field_fixture("emerald")
    try:
        constants = load_map_catalog("emerald").constants
        bridge.transition_map(0, 0, 2, 2, relocate=False)
        target = ObjectEventStruct()
        target.flags, target.localId = 3, 1
        target.elevations = bridge.get_player_coords_and_map()["z"]
        target.movementType = constants["MOVEMENT_TYPE_WALK_SEQUENCE_DOWN_LEFT_UP_RIGHT"]
        target.rangeX_Y = 0x33
        target.initialCoords_x, target.initialCoords_y = 5 + 7, 2 + 7
        target.currentCoords_x, target.currentCoords_y = 5 + 7, 2 + 7
        target.previousCoords_x, target.previousCoords_y = 4 + 7, 2 + 7
        bridge.write_memory(bridge.memory_map.objects + 36, bytes(target))
        before = bridge.save_state()
        # Patrol covers x2..5,y2..5. Wait on x1 instead of blocking its circuit.
        assert field.interaction_key(1) == bridge.KEY_LEFT
        assert bridge.save_state() == before
        bridge.transition_map(0, 0, 1, 2, relocate=False)
        bridge.write_memory(bridge.memory_map.objects + 36, bytes(target))
        assert field.interaction_key(1) == 0
        assert field.state_dict()["interaction_wait"] is not None
        # Walking away for the far half of the circuit must not restart chasing.
        target.currentCoords_y, target.previousCoords_y = 5 + 7, 4 + 7
        bridge.write_memory(bridge.memory_map.objects + 36, bytes(target))
        assert field.interaction_key(1) == 0
        restored = FieldMenuController(bridge, "emerald", field.catalog)
        restored.load_state_dict(field.state_dict())
        assert restored.state_dict() == field.state_dict()
        for _ in range(240):
            bridge.step_frame()
        assert restored.interaction_key(1) is None
        assert restored.state_dict()["interaction_retry_after"] == bridge.frame_count + 120
    finally:
        bridge.close()


@pytest.mark.native
@pytest.mark.parametrize("game", ["firered", "emerald"])
@pytest.mark.parametrize("distance", [1, 2])
def test_early_patrol_interception_leaves_stationary_nurses_to_directive_replanning(game: Game, distance: int) -> None:
    bridge, field = field_fixture(game)
    try:
        constants = load_map_catalog(game).constants
        if distance == 2:
            _publish_tile_behavior(bridge, 3, 2, constants.get("MB_COUNTER", 0x80))
        bridge.transition_map(0, 0, 2, 2, relocate=False)
        player = bridge.get_object_events()[0]
        player.facingDirection = 4
        bridge.write_memory(bridge.memory_map.objects, bytes(player))
        nurse = ObjectEventStruct()
        nurse.flags, nurse.localId = 1, 1
        nurse.elevations = bridge.get_player_coords_and_map()["z"]
        nurse.movementType = constants["MOVEMENT_TYPE_FACE_DOWN"]
        nurse.currentCoords_x, nurse.currentCoords_y = 2 + distance + 7, 2 + 7
        nurse.previousCoords_x, nurse.previousCoords_y = nurse.currentCoords_x, nurse.currentCoords_y
        bridge.write_memory(bridge.memory_map.objects + 36, bytes(nurse))
        # Public interaction remains available after planning, including counters.
        assert field.interaction_key(1) == bridge.KEY_A
        before = bridge.save_state()
        assert field.patrol_interaction_key(1) is None
        assert bridge.save_state() == before
        # Briney's real WALK_SEQUENCE movement still receives the early intercept.
        nurse.movementType = constants["MOVEMENT_TYPE_WALK_SEQUENCE_DOWN_LEFT_UP_RIGHT"]
        bridge.write_memory(bridge.memory_map.objects + 36, bytes(nurse))
        assert field.patrol_interaction_key(1) == bridge.KEY_A
        nurse.mapGroup = 1
        bridge.write_memory(bridge.memory_map.objects + 36, bytes(nurse))
        assert field.patrol_interaction_key(1) is None
    finally:
        bridge.close()


def teach_all_for_fixture(bridge: SyntheticGBASiliconBridge) -> None:
    first = bridge.get_decrypted_player_party()[0]
    first["moves"], first["pp"] = [15, 57, 70, 249], [30, 15, 15, 15]
    second = dict(first, pid=43, moves=[291, 127, 33, 0], pp=[10, 15, 35, 0])
    bridge.configure_party([first, second])


@pytest.mark.native
@pytest.mark.parametrize("game", ["firered", "emerald"])
def test_evolution_and_unknown_scenes_never_receive_blanket_cancel(game: Game) -> None:
    bridge, field = field_fixture(game)
    try:
        bridge.write_memory(bridge.memory_map.main + 4, struct.pack("<I", field.symbols["CB2_EvolutionSceneUpdate"].address | 1))
        install_task(field, "Task_EvolutionScene")
        bridge.write_memory(bridge.memory_map.tasks + 8, struct.pack("<H", 8))  # cancelable sprite cycle
        field.cancel("old_stalled_menu")
        before = bridge.save_state()
        assert field.poll_key_mask() == 0
        assert bridge.save_state() == before
        # If the native text printer explicitly asks for an edge, only A is sent.
        bridge.write_memory(field.symbols["sTextPrinters"].address + 0x1B, b"\x01\x01")
        assert field.poll_key_mask() == bridge.KEY_A
        bridge.write_memory(bridge.memory_map.tasks, bytes(40 * 16))
        bridge.write_memory(bridge.memory_map.main + 4, bytes(4))
        assert field.poll_key_mask() == 0  # unknown cutscene stays neutral, including a stale close intent
        bridge.write_memory(bridge.memory_map.main + 4, struct.pack("<I", field.symbols["CB2_BagMenuRun"].address | 1))
        assert field.poll_key_mask() == bridge.KEY_B  # an identified dismissible menu can close
    finally:
        bridge.close()


@pytest.mark.native
@pytest.mark.parametrize("game", ["firered", "emerald"])
def test_evolution_learning_preserves_hms_and_works_with_battle_flag(game: Game) -> None:
    bridge, field = field_fixture(game)
    try:
        teach_all_for_fixture(bridge)
        install_task(field, "Task_EvolutionScene")
        bridge.write_memory(bridge.memory_map.tasks + 8, struct.pack("<H", 22))
        bridge.write_memory(bridge.memory_map.tasks + 20, struct.pack("<HH", 4, 5))
        bridge.write_memory(bridge.memory_map.tasks + 28, struct.pack("<H", 0))
        bridge.write_memory(bridge.memory_map.main + 0x439, b"\x02")
        bridge.write_memory(field.symbols["gBattleCommunication"].address + 1, b"\x00")
        assert field.poll_key_mask() == bridge.KEY_DOWN  # No: all four retained moves are HMs
        bridge.write_memory(field.symbols["gBattleCommunication"].address + 1, b"\x01")
        assert field.poll_key_mask() == bridge.KEY_A
        bridge.write_memory(bridge.memory_map.tasks + 22, struct.pack("<H", 11))
        assert field.poll_key_mask() == bridge.KEY_UP  # Yes to stop learning this move
        assert field.state_dict()["evolution_party_index"] == 0
        restored = FieldMenuController(bridge, game, field.catalog)
        restored.load_state_dict(field.state_dict())
        assert restored.state_dict() == field.state_dict()
    finally:
        bridge.close()


@pytest.mark.native
@pytest.mark.parametrize("game", ["firered", "emerald"])
def test_hm_capabilities_require_native_badge_and_own_non_egg_carrier(game: Game) -> None:
    bridge, field = field_fixture(game)
    try:
        teach_all_for_fixture(bridge)
        constants = load_map_catalog(game).constants
        # HM possession alone proves neither the badge nor a learned move.
        bridge.configure_inventory({339: 1, 341: 1, 342: 1, 344: 1, 345: 1, 346: 1})
        assert not field.capabilities()["cut"].usable
        gate = "FLAG_BADGE01_GET" if game == "emerald" else "FLAG_BADGE02_GET"
        bridge.set_event_flag(constants[gate], True)
        before = bridge.save_state()
        assert field.capabilities()["cut"].usable
        assert not field.capabilities()["surf"].usable
        assert bridge.save_state() == before
        # A learned field move remains usable without retaining its HM item.
        bridge.configure_inventory({})
        assert field.capabilities()["cut"].usable
        assert not field.capabilities()["cut"].hm_in_inventory
        party = bridge.get_decrypted_player_party()
        party[0]["is_egg"] = True
        bridge.configure_party(party)
        assert not field.capabilities()["cut"].usable
        assert field.capabilities()["dive"].supported == (game == "emerald")
    finally:
        bridge.close()


@pytest.mark.native
@pytest.mark.parametrize("game", ["firered", "emerald"])
def test_surf_and_obstacle_activation_face_then_pulse_without_advancing(game: Game) -> None:
    bridge, field = field_fixture(game)
    try:
        teach_all_for_fixture(bridge)
        constants = load_map_catalog(game).constants
        for badge in range(1, 9):
            bridge.set_event_flag(constants[f"FLAG_BADGE{badge:02d}_GET"], True)
        player = bridge.get_object_events()[0]
        player.facingDirection = 2
        bridge.write_memory(bridge.memory_map.objects, bytes(player))
        before = bridge.save_state()
        assert field.choose_transition_key((2, 2), (3, 2), constants["MB_POND_WATER"]) == bridge.KEY_RIGHT
        assert bridge.save_state() == before
        player.facingDirection = 4
        bridge.write_memory(bridge.memory_map.objects, bytes(player))
        assert field.choose_transition_key((2, 2), (3, 2), constants["MB_POND_WATER"]) == bridge.KEY_A
        bridge.write_memory(bridge.memory_map.avatar, b"\x08")
        assert field.choose_transition_key((2, 2), (3, 2), constants["MB_POND_WATER"]) is None
        boulder = ObjectEventStruct()
        boulder.graphicsId = constants["OBJ_EVENT_GFX_PUSHABLE_BOULDER"]
        assert field.obstacle_move(boulder) == "strength"
        assert field.choose_transition_key((2, 2), (3, 2), 0, boulder) == bridge.KEY_A
        bridge.set_event_flag(constants["FLAG_SYS_USE_STRENGTH"], True)
        assert field.choose_transition_key((2, 2), (3, 2), 0, boulder) is None  # ordinary push may proceed
        rock = ObjectEventStruct()
        graphic = "OBJ_EVENT_GFX_BREAKABLE_ROCK" if game == "emerald" else "OBJ_EVENT_GFX_ROCK_SMASH_ROCK"
        rock.graphicsId = constants[graphic]
        assert field.obstacle_move(rock) == "rock_smash"
    finally:
        bridge.close()


def _publish_tile_behavior(bridge: SyntheticGBASiliconBridge, x: int, y: int, behavior: int) -> None:
    """Give one tile its own metatile and behavior, then load it into map RAM.

    The field controller reads the live metatile attribute table, exactly as on
    hardware, so fixture data must be published through a map load.
    """
    scenario = bridge._maps[(0, 0)]
    scenario.metatiles[y, x] = 2 + behavior % 500
    scenario.behaviors[y, x] = behavior
    bridge.transition_map(0, 0, x, y, relocate=False)


@pytest.mark.native
def test_emerald_dive_uses_native_a_down_b_up_and_respects_no_surface_tiles() -> None:
    bridge, field = field_fixture("emerald")
    try:
        teach_all_for_fixture(bridge)
        constants = load_map_catalog("emerald").constants
        bridge.set_event_flag(constants["FLAG_BADGE07_GET"], True)
        bridge.write_memory(bridge.memory_map.avatar, b"\x08")
        _publish_tile_behavior(bridge, 2, 2, constants["MB_DEEP_WATER"])
        assert field.choose_transition_key((2, 2), (2, 2), 0, intent="dive") == bridge.KEY_A
        bridge.write_memory(bridge.memory_map.map_header + 0x17, bytes((constants["MAP_TYPE_UNDERWATER"],)))
        assert field.choose_transition_key((2, 2), (2, 2), 0, intent="dive") == bridge.KEY_B
        _publish_tile_behavior(bridge, 2, 2, constants["MB_NO_SURFACING"])
        assert field.choose_transition_key((2, 2), (2, 2), 0, intent="dive") is None
    finally:
        bridge.close()
