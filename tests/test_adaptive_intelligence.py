"""Gates, refused passages, preparation lessons, shopping and never-stopping recovery.

These cover the behaviours added after the FireRed Saffron-gate stall (the guard
kept pushing the agent back while it alternated between both gatehouses) and the
Emerald Brawly loop (fighting with Mud Shot out of PP, then rewinding away the
levels it had trained).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.navigation.obstacles import gate_entry_edges, obstacle_model
from core.navigation.pushback import PushbackLearner
from core.navigation.script_gates import (
    analyze_gates, gate_state, gates_by_map, load_gates, load_trainers, unlocking_flags,
)
from core.navigation.source_data import load_map_catalog, source_root
from core.navigation.trainer_sight import _directions, trainer_sights, undefeated_sights
from core.planner.readiness import CampaignPreparationPolicy, best_attack_pp
from core.progress_watchdog import ProgressObservation, ProgressWatchdog
from core.shopping import ShoppingNeed, healing_stock, plan_purchases, shopping_need


class _World:
    """Minimal observable world for gate conditions."""

    def __init__(self, flags: dict[str, bool] | None = None, variables: dict[str, int] | None = None,
                 items: dict[str, int] | None = None) -> None:
        self.flags, self.variables, self.items = flags or {}, variables or {}, items or {}

    def flag(self, name: str) -> bool | None:
        return self.flags.get(name, False)

    def var(self, name: str) -> int | None:
        return self.variables.get(name, 0)

    def item_count(self, name: str) -> int | None:
        return self.items.get(name, 0)

    def party_move_slot(self, name: str) -> int | None:
        return 6

    # FlagReader for obstacle_model's templates
    def get_event_flag(self, flag_id: int) -> bool:
        return False


def _gate(game: str, map_name: str, tile: tuple[int, int]) -> Any:
    catalog = load_map_catalog(game)
    key = catalog.by_name[map_name]
    return next(gate for gate in gates_by_map(game)[key] if gate.tile == tile)


def test_saffron_guards_require_tea_and_push_back() -> None:
    gate = _gate("firered", "Route5_SouthEntrance", (4, 5))
    assert gate.pushback == (0, -1)
    assert unlocking_flags(gate) == ("FLAG_GOT_TEA",)
    assert "FLAG_GOT_TEA" in gate.requirement
    closed = _World()
    assert gate_state(gate, closed) is True
    assert gate_state(gate, _World(flags={"FLAG_GOT_TEA": True})) is False
    # Once a guard drank the tea the shared scene variable disarms every gate.
    assert gate_state(gate, _World(variables={"VAR_MAP_SCENE_ROUTE5_ROUTE6_ROUTE7_ROUTE8_GATES": 1})) is False
    for name in ("Route6_NorthEntrance", "Route7_EastEntrance", "Route8_WestEntrance"):
        key = load_map_catalog("firered").by_name[name]
        assert all(unlocking_flags(item) == ("FLAG_GOT_TEA",) for item in gates_by_map("firered")[key])


def test_emerald_sandstorm_requires_go_goggles() -> None:
    gate = next(item for item in load_gates("emerald") if item.script.startswith("Route111_EventScript_ViciousSandstorm"))
    assert gate_state(gate, _World()) is True
    assert gate_state(gate, _World(items={"ITEM_GO_GOGGLES": 1})) is False


def test_gate_blocks_only_steps_its_script_would_undo() -> None:
    edges = gate_entry_edges((4, 5), (0, -1))
    assert ((4, 4), (4, 5)) in edges            # walking south into the guard: refused
    assert ((4, 6), (4, 5)) not in edges        # leaving Saffron northbound: carried along
    assert ((3, 5), (4, 5)) in edges            # sideways along the trigger row: refused
    # Petalburg's gym scene walks newcomers east: entering from Route 102 is fine.
    gym = _gate("emerald", "PetalburgCity", (8, 10))
    assert gym.pushback[0] > 0
    assert ((7, 10), (8, 10)) not in gate_entry_edges(gym.tile, gym.pushback)
    assert ((9, 10), (8, 10)) in gate_entry_edges(gym.tile, gym.pushback)


def test_obstacle_model_closes_gate_until_the_requirement_is_met() -> None:
    key = load_map_catalog("firered").by_name["Route5_SouthEntrance"]
    closed = obstacle_model("firered", key, bridge=_World())
    assert ((4, 4), (4, 5)) in closed.gate_edges
    learned = {((1, 1), (1, 2))}
    relaxed = obstacle_model("firered", key, bridge=_World(), static_gates=False,
                             learned_edges=learned)
    assert relaxed.gate_edges == frozenset(learned)


def test_exported_gate_knowledge_matches_pinned_sources() -> None:
    if not (source_root("firered") / "data" / "maps").is_dir():
        pytest.skip("pinned decomp sources are not vendored")
    exported = {(gate.map_name, gate.tile, gate.script) for gate in load_gates("firered")}
    analyzed = {(gate.map_name, gate.tile, gate.script) for gate in analyze_gates("firered")}
    assert exported == analyzed
    assert load_trainers("firered")["RockTunnel_1F"]["6"] == "TRAINER_HIKER_LENNY"


def test_pushback_learning_needs_a_script_moving_the_avatar_back() -> None:
    key = load_map_catalog("firered").by_name["Route6_NorthEntrance"]
    learner = PushbackLearner("firered")
    learner.note_step(key, (4, 4), (4, 5), 100)
    assert learner.observe(key, (4, 5), True, 101, "w") is None      # guard talks
    assert learner.observe(key, (4, 4), True, 140, "w") is None      # walk_up while scripted
    barrier = learner.observe(key, (4, 4), False, 160, "w")
    assert barrier is not None and "FLAG_GOT_TEA" in barrier.requirement
    edges = learner.active_edges("w")[key]
    assert ((4, 4), (4, 5)) in edges and ((3, 4), (3, 5)) in edges  # sibling trigger tiles too
    assert learner.active_edges("new milestone") == {}              # a changed world earns a retry


def test_walking_back_by_itself_is_not_a_refusal() -> None:
    key = load_map_catalog("firered").by_name["RockTunnel_1F"]
    learner = PushbackLearner("firered")
    for frame in range(0, 60, 6):
        learner.note_step(key, (5, 11), (5, 10), frame)
        assert learner.observe(key, (5, 10), False, frame + 1, "w") is None
        learner.note_step(key, (5, 10), (5, 11), frame + 3)
        assert learner.observe(key, (5, 11), False, frame + 4, "w") is None
    assert learner.barriers == {}


def test_trainer_sight_comes_from_templates_until_defeated() -> None:
    key = load_map_catalog("firered").by_name["RockTunnel_1F"]
    lenny = next(sight for sight in trainer_sights("firered", key) if sight.local_id == 6)
    assert lenny.directions == (1,) and lenny.sight == 4 and lenny.flag is not None

    class Flags:
        def get_event_flag(self, flag_id: int) -> bool:
            return flag_id == lenny.flag

    assert all(sight.local_id != 6 for sight in undefeated_sights("firered", key, Flags()))
    assert _directions("MOVEMENT_TYPE_LOOK_AROUND", "TRAINER_TYPE_NORMAL") == (1, 2, 3, 4)
    assert _directions("MOVEMENT_TYPE_FACE_DOWN_AND_UP", "TRAINER_TYPE_NORMAL") == (1, 2)


def _info(move: int) -> tuple[int, int, int]:
    table = {33: (0, 35, 35), 341: (4, 55, 15), 189: (4, 20, 10), 55: (11, 40, 25)}
    return table[move]


def _marshtomp(pp: list[int], hp: int = 77) -> dict[str, Any]:
    return {"species": 284, "level": 24, "hp": hp, "maxHP": 77, "moves": [33, 341, 189, 55], "pp": pp}


def test_best_attack_pp_ignores_full_tackle_when_mud_shot_is_empty() -> None:
    fraction, attacks = best_attack_pp(_marshtomp([33, 0, 10, 7]), _info, (11, 4))
    assert attacks == 4 and fraction < 0.3


def test_boss_is_engaged_only_restored_and_defeats_are_analysed() -> None:
    policy = CampaignPreparationPolicy()
    types = lambda species: (11, 4)  # noqa: E731
    assert "PP" in str(policy.engage_reason("GYM_2_BRAWLY", [_marshtomp([33, 0, 10, 7])], _info, types))
    assert "HP" in str(policy.engage_reason("GYM_2_BRAWLY", [_marshtomp([35, 15, 10, 25], hp=50)], _info, types))
    assert policy.engage_reason("GYM_2_BRAWLY", [_marshtomp([35, 15, 10, 25])], _info, types) is None
    # An unprepared loss does not inflate the level target; a prepared one does.
    policy.record_defeat("GYM_2_BRAWLY", 22, prepared=False)
    assert policy.records["GYM_2_BRAWLY"].retry_level == 22
    policy.record_defeat("GYM_2_BRAWLY", 22, prepared=True)
    assert policy.records["GYM_2_BRAWLY"].retry_level == 24
    # Restores that keep failing on the way do not trap the agent in a loop.
    for _ in range(3):
        policy.record_restore("GYM_2_BRAWLY")
    assert policy.engage_reason("GYM_2_BRAWLY", [_marshtomp([33, 0, 10, 7])], _info, types) is None
    # A strategic deadline stops endless grinding with a usable attack.
    party = [_marshtomp([35, 15, 10, 25])]
    assert not policy.assess("GYM_2_BRAWLY", 30, party, [2, 14], lambda move: _info(move)[:2]).ready
    assert policy.force_attempt("GYM_2_BRAWLY")
    assert policy.assess("GYM_2_BRAWLY", 30, party, [2, 14], lambda move: _info(move)[:2]).ready
    lessons = policy.lessons_state()
    fresh = CampaignPreparationPolicy()
    fresh.merge_lessons(json.loads(json.dumps(lessons)))
    assert fresh.records["GYM_2_BRAWLY"].defeats == 2


def _observe(watchdog: ProgressWatchdog, frame: int, score: int, story: int = 0) -> Any:
    return watchdog.observe(ProgressObservation(frame, "field", (frame,), (frame,), (0, 0, frame % 50, 0),
                                                "GYM", strategic_gains=(story,), preparation=score))


def test_training_levels_earn_bounded_strategic_credit() -> None:
    watchdog = ProgressWatchdog(strategic_limit=1000, preparation_credit=400, preparation_budget=1200,
                                cycle_limit=10**9)
    _observe(watchdog, 0, 100)
    _observe(watchdog, 900, 100)
    assert watchdog.strategic_unproductive_frames == 900
    _observe(watchdog, 901, 101)  # one level
    assert watchdog.strategic_unproductive_frames == 501
    _observe(watchdog, 902, 110)  # budget caps the credit
    assert watchdog.strategic_unproductive_frames == 0
    # Re-earning a level lost to a rewind is not new preparation.
    _observe(watchdog, 903, 105)
    _observe(watchdog, 950, 110)
    assert watchdog.strategic_unproductive_frames == 48
    assert not watchdog.grant_strategic_grace(500)  # the budget is spent
    state = watchdog.state_dict()
    restored = ProgressWatchdog(strategic_limit=1000)
    restored.load_state_dict(state)
    assert restored.state_dict()["strategic_preparation_best"] == 110


def test_shopping_plan_buys_useful_items_within_budget() -> None:
    constants = load_map_catalog("emerald").constants
    potion, super_potion, ball = (constants[name] for name in ("ITEM_POTION", "ITEM_SUPER_POTION", "ITEM_POKE_BALL"))
    party = [_marshtomp([35, 15, 10, 25])]
    need = shopping_need({}, party, constants, wants_balls=True)
    assert need.heal_hp >= 150 and need.balls == 10
    prices = {potion: 300, super_potion: 700, ball: 200}
    plan = dict(plan_purchases([ball, potion, super_potion], 1685, need, prices.__getitem__, constants, 77))
    spent = sum(prices[item] * count for item, count in plan.items())
    assert spent <= int(1685 * 0.7) and (plan.get(super_potion) or plan.get(potion))
    stocked = {super_potion: 4, ball: 10}
    assert not shopping_need(stocked, party, constants, wants_balls=True).any
    assert healing_stock(stocked, constants) == 200
    assert plan_purchases([], 5000, ShoppingNeed(100, 5, ""), prices.__getitem__, constants, 77) == []


def test_stagnant_budget_pauses_instead_of_ending_the_campaign(tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_campaign as campaign
    from tests.test_runtime_recovery import _simulate_supervision, _simulated_grind_outcome

    _simulate_supervision(monkeypatch)
    attempts: list[str | None] = []

    def grind(args: campaign.WorkerArgs, generation: str, sender: Any, checkpoint: str | None) -> None:
        attempts.append(checkpoint)
        _simulated_grind_outcome(args, generation, sender, len(attempts) - 1, [])

    task: campaign.WorkerArgs = ("emerald", "synthetic", False, None, str(tmp_path), 42, 7)
    report = campaign.supervise_campaign(task, child_target=grind, stagnant_restart_limit=2,
                                         stagnation_cooldown=30.0)
    assert len(attempts) == 8 and "recovery_blocked" not in report
    events = [json.loads(line) for line in
              (tmp_path / "synthetic" / "emerald" / "supervisor-events.jsonl").read_text().splitlines()]
    assert sum(event["kind"] == "recovery_cooldown" for event in events) >= 2


def test_type_chart_is_learned_from_battle_messages_and_solved_across_dual_types() -> None:
    from core.strategy_knowledge import NEUTRAL, NO_EFFECT, RESISTED, SUPER, TypeChartMemory, result_category

    water, fire, ground, rock, grass, flying, electric = 11, 10, 4, 5, 12, 2, 13
    chart = TypeChartMemory()
    assert chart.multiplier(water, (fire,)) is None              # nothing known before seeing it
    assert chart.observe(water, (fire,), SUPER, species=4)
    assert chart.multiplier(water, (fire,)) == 2.0
    # Dual-type evidence: Water vs Rock/Ground is "super effective" and Water vs
    # Ground alone is known super, so the solver may not pin Rock yet...
    chart.observe(water, (ground,), SUPER)
    chart.observe(water, (rock, ground), SUPER)
    assert chart.multiplier(water, (rock, ground)) == 2.0 or chart.multiplier(water, (rock, ground)) == 4.0
    # Grass is x2 on Water; "super" on Water/Ground leaves Ground in {1, 2}: unknown.
    chart.observe(grass, (water,), SUPER)
    chart.observe(grass, (water, ground), SUPER)
    assert chart.factor(grass, ground) is None
    # Electric is x2 on Water; "doesn't affect" on Water/Ground can only mean Ground x0.
    chart.observe(electric, (water,), SUPER)
    chart.observe(electric, (water, ground), NO_EFFECT, species=259)
    assert chart.factor(electric, ground) == 0.0
    assert result_category(0x02) == SUPER and result_category(0x04) == RESISTED
    assert result_category(0x00) == NEUTRAL and result_category(0x08) == NO_EFFECT and result_category(0x01) is None
    # An immunity a Levitate species can explain is not a type fact until confirmed twice.
    assert not chart.observe(ground, (electric,), NO_EFFECT, species=81, species_abilities=(26, 26))
    assert chart.observe(ground, (electric,), NO_EFFECT, species=100, species_abilities=(26, 26))
    assert chart.multiplier(ground, (electric,)) == 0.0
    restored = TypeChartMemory()
    restored.merge_state(json.loads(json.dumps(chart.state())))
    assert restored.multiplier(water, (fire,)) == 2.0 and restored.known_cells() == chart.known_cells()


def test_counters_come_from_the_remembered_roster_and_the_learned_chart() -> None:
    from core.strategy_knowledge import NEUTRAL, SUPER, OpponentMemory, TypeChartMemory, counter_species

    fighting, flying, psychic, normal = 1, 2, 14, 0
    chart = TypeChartMemory()
    chart.observe(flying, (fighting,), SUPER)
    chart.observe(psychic, (fighting,), SUPER)
    chart.observe(normal, (fighting,), NEUTRAL)
    memory = OpponentMemory()
    for species, level in ((66, 17), (307, 18), (296, 19)):  # Machop, Meditite, Makuhita
        memory.observe("GYM_2_BRAWLY", species, level, move_type=fighting)
    types = {66: (fighting, fighting), 307: (fighting, psychic), 296: (fighting, fighting)}
    counters = memory.counters("GYM_2_BRAWLY", types, chart)
    assert counters[0] in (flying, psychic) and normal not in counters
    assert memory.threats("GYM_2_BRAWLY") == [fighting]
    assert counter_species([304, 280, 309], [flying], {304: (0, 2), 280: (10, 10), 309: (13, 13)}) == [304]


def test_firered_spin_tiles_slide_until_a_stop_tile() -> None:
    import numpy as np
    from core.navigation.constrained_astar import GbaConstrainedAStar

    # Row 0: start . R(spin right) . . . S(stop) . goal-wall ; row 1 open floor.
    collision = np.zeros((3, 9), dtype=np.uint16)
    elevation = np.full((3, 9), 3, dtype=np.uint16)
    behavior = np.zeros((3, 9), dtype=np.uint16)
    behavior[0, 2] = 0x54  # spin right
    behavior[0, 6] = 0x58  # stop spinning
    collision[0, 8] = 1
    planner = GbaConstrainedAStar(collision, elevation, behavior, "firered")
    path = planner.find_optimal_path((1, 0, 3), (6, 0))
    assert path is not None and path[0] == (2, 0) and path[-1] == (6, 0)
    # The slide skips (3..5, 0): those tiles cannot be stopped on from the spin.
    costs = planner.reachable_costs((1, 0, 3))
    assert costs[(6, 0)] <= 6
    # Emerald has no spin tiles: the same behavior is ordinary floor there.
    emerald = GbaConstrainedAStar(collision, elevation, behavior, "emerald")
    assert emerald.find_optimal_path((1, 0, 3), (4, 0)) == [(2, 0), (3, 0), (4, 0)]


def test_elevator_panels_become_floor_edges_with_their_requirements() -> None:
    from core.navigation.script_gates import load_elevators
    from core.navigation.warp_topology import HierarchicalWarpGraph

    panels = load_elevators("firered")["RocketHideout_Elevator"]
    floors = {floor["map"]: floor["choice"] for floor in panels[0]["floors"]}
    assert floors == {"MAP_ROCKET_HIDEOUT_B1F": 0, "MAP_ROCKET_HIDEOUT_B2F": 1, "MAP_ROCKET_HIDEOUT_B4F": 2}
    assert panels[0]["required_flags"] == ["FLAG_CAN_USE_ROCKET_HIDEOUT_LIFT"]
    catalog = load_map_catalog("firered")
    graph = HierarchicalWarpGraph.from_source("firered")
    elevator = catalog.by_name["RocketHideout_Elevator"]
    b4f = catalog.by_name["RocketHideout_B4F"]
    edge = next(edge for edge in graph.adjacency[elevator] if edge.destination == b4f)
    assert edge.panel_xy == (0, 2) and edge.menu_choice == 2 and edge.destination_xy == (20, 23)
    assert edge.required_flags == (catalog.constants["FLAG_CAN_USE_ROCKET_HIDEOUT_LIFT"],)
    # Emerald's attendant is an NPC whose rows use the two-argument warp form.
    lilycove = load_elevators("emerald")["LilycoveCity_DepartmentStoreElevator"][0]
    assert lilycove["npc_local_id"] == 1 and len(lilycove["floors"]) == 5


def test_on_load_barriers_follow_the_trainer_that_opens_them() -> None:
    from core.navigation.script_gates import map_overlay

    class Trainers(_World):
        def __init__(self, defeated: bool) -> None:
            super().__init__()
            self.defeated = defeated

        def flag(self, name: str) -> bool | None:
            return self.defeated if name == "TRAINER:TRAINER_TEAM_ROCKET_GRUNT_12" else False

    walled = dict(((x, y), value) for x, y, value in map_overlay("firered", "RocketHideout_B1F", Trainers(False)))
    assert walled[(20, 19)] == 1 and walled[(21, 20)] == 1
    assert map_overlay("firered", "RocketHideout_B1F", Trainers(True)) == ()
    assert map_overlay("firered", "RocketHideout_B1F", None) == ()


def test_item_balls_are_picked_up_instead_of_blocking_a_spin_maze() -> None:
    catalog = load_map_catalog("firered")
    key = catalog.by_name["RocketHideout_B2F"]
    model = obstacle_model("firered", key, bridge=_World())
    assert (2, 5) in model.removable_costs and (2, 5) not in model.blocked  # the Moon Stone
    import numpy as np
    from core.navigation.constrained_astar import GbaConstrainedAStar

    collision = np.zeros((1, 6), dtype=np.uint16)
    elevation = np.full((1, 6), 3, dtype=np.uint16)
    behavior = np.zeros((1, 6), dtype=np.uint16)
    behavior[0, 1] = 0x54  # spin right
    planner = GbaConstrainedAStar(collision, elevation, behavior, "firered")
    planner.add_static_obstacles((), {(4, 0): 6.0})
    # The slide stops before the item; it is collected from (3, 0) and walked over.
    assert planner.reachable_costs((0, 0, 3))[(3, 0)] == 3.0
    assert planner.find_optimal_path((0, 0, 3), (5, 0)) == [(1, 0), (3, 0), (4, 0), (5, 0)]


def test_floor_switches_are_the_triggers_that_redraw_the_room() -> None:
    from core.navigation.script_gates import load_floor_switches

    switches = {tuple(tile) for tile in load_floor_switches("emerald")["MauvilleCity_Gym"]}
    assert switches == {(0, 15), (4, 12), (3, 9), (8, 9)}
    assert "MtPyre_Summit" not in load_floor_switches("emerald")  # story triggers do not redraw
