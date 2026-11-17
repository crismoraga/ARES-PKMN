"""ARES-ADVERSARIAL stadium: engine fidelity, information contract, GAME and learning.

The engine must agree with the ROM and the project's exact damage reference;
agents must never see hidden opponent data; GAME must implement Algorithms
1-5 of Anne et al. (GECCO'26); the learning channels must actually change the
policies they target; and the league must checkpoint and resume.
"""
from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from core.game_data import RomCatalog
from core.navigation.source_data import source_root
from core.safety.z3_integer_verifier import DamageContext, GbaIntegerSafetyVerifier
from core.stadium.dex import TYPE_MYSTERY, load_dex
from core.stadium.engine import A, MOVE_STRUGGLE, PARALYSIS, Battle, MonSpec
from core.stadium.evaluation import Evaluator, SideSpec, Solution
from core.stadium.features import OPTION_FEATURES, net_observation, option_features
from core.stadium.game_qd import (GameConfig, GenerationReport, GrowingArchive, SpaceOps, Variant, budget_matched,
                                  pareto_select, ranking_vectors, run_game, select_tasks)
from core.stadium.knowledge import SideKnowledge, exchange
from core.stadium.deep import DeepLearner, StadiumEnvironment
from core.stadium.learning import (apply_lessons, counterfactual_lessons,
                                   train_mlp)
from core.stadium.match import BEHAVIOR_DIM, fitness_from, play
from core.stadium.metrics import elo_ratings, holm_bonferroni, min_counter_set, side_report
from core.stadium.policies import GenomePolicy, GreedyDamagePolicy, MLPGenome, RandomPolicy, parameter_count
from core.stadium.teams import RosterMon, fallback_bank, gen3_stats, initial_genes, mutate_genes

DEX = load_dex()


def sid(name: str) -> int:
    """Internal Gen-3 species id (Hoenn species do not use national numbers)."""
    return next(data.species_id for data in DEX.species if data.name == name)


def member(species: int, level: int, moves: tuple[int, ...], ability: int | None = None) -> MonSpec:
    data = DEX.species[species]
    stats = gen3_stats(DEX, species, level, (31,) * 6, (0,) * 6, 0)
    return MonSpec(species, level, stats, moves, data.abilities[0] if ability is None else ability)


VENUSAUR = member(3, 50, (75, 22, 73, 77))      # Razor Leaf, Vine Whip, Leech Seed, Poison Powder
SWAMPERT = member(sid("SWAMPERT"), 50, (89, 57, 33, 45))  # Earthquake, Surf, Tackle, Growl
PIKACHU = member(25, 50, (85, 86, 98, 21))      # Thunderbolt, Thunder Wave, Quick Attack, Slam
GENGAR = member(94, 50, (247, 33, 85, 101))     # Shadow Ball, Tackle, Thunderbolt, Night Shade


def run_battle(team_a: list[MonSpec], team_b: list[MonSpec], seed: int,
               chooser: Callable[[Battle, int], int] | None = None) -> Battle:
    battle = Battle(DEX, team_a, team_b, seed=seed)
    rng = random.Random(seed)
    while not battle.over:
        battle.submit({side: (chooser(battle, side) if chooser else rng.choice(battle.legal_actions(side)))
                       for side in battle.pending})
    return battle


# ------------------------------------------------------------------ dex
def test_dex_type_table_matches_the_rom_oracle_and_keeps_native_order() -> None:
    catalog = RomCatalog(DEX.game)
    rng = random.Random(1)
    for _ in range(400):
        attack = rng.choice([value for value in range(18) if value != TYPE_MYSTERY])
        defenders = (rng.randrange(18), rng.randrange(18))
        numerator, denominator = catalog.type_multiplier_oracle(attack, defenders)
        assert DEX.multiplier(attack, defenders) == pytest.approx(numerator / denominator)
    # Ground into Flying/Fire: the immunity appears in the native modifier list (applied in order).
    assert (0, 10) in DEX.modifiers(4, (2, 10))
    assert DEX.multiplier(0, (7, 7)) == 0.0


def test_evolution_projection_follows_the_rom_evolution_table() -> None:
    assert DEX.evolve(1, 50) == 3                     # Bulbasaur -> Venusaur
    assert DEX.evolve(1, 20) == 2                     # Ivysaur at 20
    assert DEX.evolve(10, 50) == 12                   # Caterpie -> Butterfree
    wurmple = sid("WURMPLE")
    silcoon = DEX.evolve(wurmple, 50, personality=0)  # Wurmple by personality
    cascoon = DEX.evolve(wurmple, 50, personality=9 << 16)
    assert {silcoon, cascoon} == {sid("BEAUTIFLY"), sid("DUSTOX")}
    assert DEX.evolve(41, 50, friendship=70) != DEX.evolve(41, 50, friendship=255)  # Golbat -> Crobat


def test_ability_ids_match_the_decomp_header() -> None:
    header = source_root("emerald") / "include" / "constants" / "abilities.h"
    if not header.is_file():
        pytest.skip("decomp sources are not vendored")
    defined = {name: int(value) for name, value in
               re.findall(r"#define ABILITY_(\w+)\s+(\d+)", header.read_text(encoding="utf-8"))}
    for name, value in A.items():
        assert defined[name] == value, name


# ------------------------------------------------------------------ engine
def test_battles_are_deterministic_and_clones_never_perturb_the_original() -> None:
    first = run_battle([VENUSAUR, PIKACHU], [SWAMPERT, GENGAR], seed=11)
    second = run_battle([VENUSAUR, PIKACHU], [SWAMPERT, GENGAR], seed=11)
    assert first.events == second.events and first.winner == second.winner

    rng = random.Random(3)
    plain = Battle(DEX, [VENUSAUR], [SWAMPERT], seed=5)
    probed = Battle(DEX, [VENUSAUR], [SWAMPERT], seed=5)
    while not plain.over:
        actions = {side: rng.choice(plain.legal_actions(side)) for side in plain.pending}
        clone = probed.clone()
        clone.submit({side: clone.legal_actions(side)[0] for side in clone.pending})
        plain.submit(actions)
        probed.submit(actions)
        assert plain.log == probed.log


def test_every_hit_is_within_the_exact_integer_damage_reference() -> None:
    battle = Battle(DEX, [VENUSAUR], [SWAMPERT], seed=2)
    battle.submit({0: 0, 1: 2})  # Razor Leaf vs Tackle
    hits = [event for event in battle.log if event[0] == "hit" and event[6] == 0]
    assert hits and hits[0][3] == "super"
    attacker, defender = VENUSAUR, SWAMPERT
    context = DamageContext(level=50, attack=attacker.stats[4], defense=defender.stats[5], power=55, move_type=12,
                            type_modifiers=DEX.modifiers(12, DEX.species[sid("SWAMPERT")].types), stab=True,
                            attacker_ability=attacker.ability, defender_ability=defender.ability,
                            attacker_species=3, defender_species=sid("SWAMPERT"), attacker_hp=attacker.stats[0],
                            attacker_max_hp=attacker.stats[0])
    low = GbaIntegerSafetyVerifier.compute_context_damage_reference(context, 85, False)
    high = GbaIntegerSafetyVerifier.compute_context_damage_reference(context, 100, True)
    assert low <= hits[0][2] <= max(high, defender.stats[0])


def test_type_and_ability_immunities_and_gen3_status_rules() -> None:
    # Earthquake cannot touch Gengar (Levitate) and Thunder Wave fails on a Ground type.
    battle = Battle(DEX, [SWAMPERT], [GENGAR], seed=1)
    battle.submit({0: 0, 1: 1})
    assert any(event[0] == "noeffect" for event in battle.log)
    assert ("ability", 1, A["LEVITATE"]) in battle.log
    battle = Battle(DEX, [PIKACHU], [SWAMPERT], seed=1)
    battle.submit({0: 1, 1: 3})
    assert any(event[0] == "noeffect" for event in battle.log)
    assert not battle.sides[1].fighter.status & PARALYSIS


def test_struggle_switch_legality_and_turn_limit() -> None:
    battle = Battle(DEX, [VENUSAUR, PIKACHU], [SWAMPERT], seed=3, max_turns=3)
    assert battle.legal_actions(0) == [0, 1, 2, 3, 4]
    battle.sides[0].fighter.pp = [0, 0, 0, 0]
    assert battle.legal_actions(0)[0] == 0  # Struggle
    battle.submit({0: 0, 1: 2})
    assert any(event[0] == "move" and event[3] == MOVE_STRUGGLE for event in battle.log)
    while not battle.over:
        battle.submit({side: battle.legal_actions(side)[0] for side in battle.pending})
    assert battle.winner in (0, 1, 2) and battle.turn <= 3


def test_intimidate_and_entry_hazards() -> None:
    gyarados = member(130, 50, (56, 82, 44, 37))
    battle = Battle(DEX, [VENUSAUR], [gyarados], seed=1)
    assert battle.sides[0].fighter.stages[1] == 5  # -1 Attack on entry
    spikes = member(91, 50, (191, 33, 57, 59))     # Cloyster with Spikes
    battle = Battle(DEX, [spikes], [PIKACHU, SWAMPERT], seed=1)
    battle.submit({0: 0, 1: 2})
    assert battle.sides[1].spikes == 1
    before = battle.sides[1].party[1].hp
    battle.submit({0: 1, 1: 4})
    assert battle.sides[1].party[1].hp < before  # Spikes on switch-in


# ------------------------------------------------------------------ teams
def test_gen3_stats_reproduce_real_campaign_parties() -> None:
    bank_path = Path("artifacts/stadium/team_bank.json")
    if not bank_path.is_file():
        pytest.skip("no harvested campaign bank on this machine")
    raw = json.loads(bank_path.read_text(encoding="utf-8"))
    rows = [RosterMon.from_json(row) for rows in raw["rosters"].values() for row in rows]
    assert rows
    exact = 0
    for mon in rows:
        # Gen 3 recalculates stats only on level-up: EVs earned since then are not visible yet.
        with_evs = gen3_stats(DEX, mon.species, mon.level, mon.ivs, mon.evs, mon.pid % 25)
        without = gen3_stats(DEX, mon.species, mon.level, mon.ivs, (0,) * 6, mon.pid % 25)
        assert all(low <= value <= high for low, value, high in zip(without, mon.stats, with_evs)), mon.name
        exact += with_evs == mon.stats
    assert exact >= 0.8 * len(rows)


def test_team_mutation_stays_legal() -> None:
    bank = fallback_bank(DEX)
    roster = bank.roster("firered")
    rng = random.Random(4)
    genes = initial_genes(roster, bank.parties["firered"], rng, dex=DEX, level=50)
    for _ in range(60):
        genes = mutate_genes(genes, roster, DEX, rng, rate=0.9, level=50)
        assert len(set(genes.members)) == len(genes.members) <= 6
        for index, moves in zip(genes.members, genes.moves):
            assert 1 <= len(moves) <= 4 and set(moves) <= set(roster[index].move_pool(DEX, 50))
    team = genes.build(roster, DEX, 50)
    assert all(spec.level == 50 for spec in team)


# -------------------------------------------------------------- knowledge
def test_type_chart_is_learnt_from_messages_and_exchanged() -> None:
    first, second = SideKnowledge(), SideKnowledge()
    battle = run_battle([VENUSAUR, PIKACHU], [SWAMPERT, GENGAR], seed=8)
    on_field = {0: 3, 1: sid("SWAMPERT")}
    assert first.observe(DEX, battle.events, 0, on_field) > 0
    assert first.known_cells() > 0 and second.known_cells() == 0
    gained = exchange(first, second)
    assert gained == (0, first.known_cells()) and second.known_cells() == first.known_cells()
    assert exchange(first, second) == (0, 0)  # idempotent


def test_observations_never_leak_hidden_opponent_data() -> None:
    knowledge = SideKnowledge()
    honest = Battle(DEX, [VENUSAUR], [SWAMPERT], seed=1)
    secret = member(sid("SWAMPERT"), 50, (63, 94, 105, 156))       # same species/level, different hidden moves and stats
    bumped = secret.stats
    secret = MonSpec(secret.species, secret.level, (bumped[0] + 7, bumped[1] + 7, bumped[2] + 7, bumped[3] + 7,
                                                    bumped[4] + 7, bumped[5] + 7), secret.moves, secret.ability)
    twisted = Battle(DEX, [VENUSAUR], [secret], seed=1)
    twisted.sides[1].fighter.hp = honest.sides[1].fighter.hp * secret.stats[0] // SWAMPERT.stats[0]
    actions = honest.legal_actions(0)
    first, second = option_features(honest, 0, knowledge, actions), option_features(twisted, 0, knowledge, actions)
    assert np.allclose(first, second, atol=1e-6)
    assert net_observation(honest, 0, knowledge, actions)[0][1] == net_observation(twisted, 0, knowledge, actions)[0][1]


# --------------------------------------------------------------- policies
def test_prior_genome_embeds_the_expert_heuristic_and_mutation_is_the_papers() -> None:
    rng = random.Random(0)
    prior = MLPGenome.prior(rng, noise=0.0)
    features = np.random.default_rng(1).normal(size=(9, OPTION_FEATURES)).astype(np.float32)
    weights = np.zeros(OPTION_FEATURES)
    for name, value in __import__("core.stadium.policies", fromlist=["_PRIOR"])._PRIOR.items():
        weights[__import__("core.stadium.features", fromlist=["FEATURE_NAMES"]).FEATURE_NAMES.index(name)] = value
    assert np.argsort(prior.scores(features)).tolist() == np.argsort(features @ weights).tolist()
    child = prior.mutate(rng)
    changed = float(np.mean(child.weights != prior.weights))
    assert 0.2 < changed < 0.4 and child.weights.shape == (parameter_count(),)
    assert MLPGenome.from_json(child.to_json()).weights.tolist() == pytest.approx(child.weights.tolist(), abs=1e-5)


def test_expert_prior_beats_random_play_and_fitness_is_zero_sum() -> None:
    rng = random.Random(2)
    knowledge = (SideKnowledge(), SideKnowledge())
    mirror = [VENUSAUR, PIKACHU, GENGAR]
    teams = (mirror, mirror)
    wins = 0
    for seed in range(30):
        result = play(DEX, teams, (GenomePolicy(MLPGenome.prior(rng)), RandomPolicy()), knowledge, seed)
        wins += result.winner == 0
        assert 0.0 <= result.fitness <= 1.0
        assert all(behavior.shape == (BEHAVIOR_DIM,) and behavior.min() >= 0 and behavior.max() <= 1
                   for behavior in result.behavior)
    assert wins >= 21
    assert fitness_from(0, (0.5, 0.0)) + fitness_from(1, (0.0, 0.5)) == pytest.approx(1.0)


# -------------------------------------------------------------------- GAME
def _solution(uid: int) -> Solution:
    bank = fallback_bank(DEX)
    genes = initial_genes(bank.roster("firered"), bank.parties["firered"], random.Random(uid))
    return Solution(uid, "red", genes, MLPGenome.random(random.Random(uid)))


def test_growing_archive_follows_algorithm_3() -> None:
    archive = GrowingArchive(3)
    for index, point in enumerate(([0, 0], [1, 1], [0.05, 0])):
        assert archive.update(_solution(index), 0.5, np.array(point, dtype=np.float32))
    assert len(archive) == 3
    # A new, far behaviour replaces one centroid of the closest pair ([0,0] and [0.05,0]).
    archive.update(_solution(10), 0.1, np.array([5, 5], dtype=np.float32))
    assert any(np.allclose(centroid, [5, 5]) for centroid in archive.centroids)
    assert sum(np.allclose(centroid, [0, 0]) or np.allclose(centroid, [0.05, 0]) for centroid in archive.centroids) == 1
    # A close behaviour only replaces the cell elite when it is fitter.
    assert not archive.update(_solution(11), 0.2, np.array([1, 1.01], dtype=np.float32))
    assert archive.update(_solution(12), 0.9, np.array([1, 1.01], dtype=np.float32))


def test_ranking_vectors_and_pareto_selection() -> None:
    fitness = np.array([[0.9, 0.1, 0.5], [0.2, 0.8, 0.5], [0.1, 0.1, 0.1]])
    ranks = ranking_vectors(fitness)
    assert ranks.min() == -1 and ranks.max() == 1 and ranks[0, 0] == 1 and ranks[1, 1] == 1
    assert sorted(pareto_select(fitness, 2)) == [0, 1]
    extra = budget_matched(GameConfig(variant="random", n_task=4, n_cell=5, budget=100))
    assert extra.budget == 100 + 4 * 4 * 4


def test_game_alternates_sides_and_task_selection_variants_work() -> None:
    bank = fallback_bank(DEX)
    sides = {"red": SideSpec("red", "firered", bank.roster("firered"), bank.parties["firered"], 40, 3),
             "blue": SideSpec("blue", "emerald", bank.roster("emerald"), bank.parties["emerald"], 40, 3)}
    reports: list[GenerationReport] = []
    with Evaluator(sides, workers=0, repeats=1, seed=1) as evaluator:
        config = GameConfig(n_gen=2, n_task=3, n_cell=3, budget=24, n_init=6, batch=12, variant="ranking")
        last = run_game(sides, evaluator, config, seed=3, on_generation=reports.append)
        assert [report.side for report in reports] == ["red", "blue"]
        assert len(last["red"]) == 3 and len(last["blue"]) == 3
        assert reports[0].selection.ranking is not None
        counter = iter(range(10_000, 20_000))
        ops = SpaceOps(sides["red"], DEX, random.Random(0), lambda: next(counter))
        variants: tuple[Variant, ...] = ("pareto", "behavior", "random")
        for variant in variants:
            selection = select_tasks(reports[0].archives, last["blue"], evaluator,
                                     GameConfig(n_task=3, variant=variant), random.Random(0))
            assert 1 <= len(selection.tasks) <= 3 and selection.bootstrap
        assert ops.random_solution().side == "red"
        assert evaluator.knowledge["red"].known_cells() > 0  # the population learnt the chart by playing


# ----------------------------------------------------------------- metrics
def test_six_metrics_on_a_hand_computed_tournament() -> None:
    fitness = np.array([[0.9, 0.6, 0.4], [0.3, 0.7, 0.8], [0.2, 0.1, 0.6]])
    report = side_report(fitness, {"a": [0, 1], "b": [2]}, 2)
    a, b = report["a"], report["b"]
    assert a.win_rate == pytest.approx(4 / 6) and b.win_rate == pytest.approx(1 / 3)
    assert a.robustness == pytest.approx(0.4) and a.expertise == pytest.approx(0.7)
    assert a.aqd == 2.0  # row 0 loses only to col 2, row 1 only to col 0
    assert min_counter_set([{1}, {1, 2}], 3) == 1.0
    assert min_counter_set([set()], 3) == float("inf")
    red, blue = elo_ratings(fitness)
    assert red[0] > red[2]
    significance = holm_bonferroni({"x": [0.9] * 8, "y": [0.1] * 8, "z": [0.88, 0.9, 0.91, 0.89, 0.9, 0.92, 0.87, 0.9]})
    assert significance["x"] and not significance["y"]


# ----------------------------------------------------------------- learning
def test_counterfactual_lessons_and_mlp_training_change_the_learner() -> None:
    rng = random.Random(5)
    genomes = (MLPGenome.prior(rng), MLPGenome.random(rng, scale=0.6))
    knowledge = (SideKnowledge(), SideKnowledge())
    teams = ([VENUSAUR, PIKACHU], [SWAMPERT, GENGAR])
    decisions = []
    for seed in range(4):
        decisions += play(DEX, teams, (GenomePolicy(genomes[0]), GenomePolicy(genomes[1])), knowledge, seed,
                          snapshot_every=1).decisions
    lessons = counterfactual_lessons(decisions, genomes, knowledge, rng, rollouts=4, max_points=12, margin=0.0)
    assert lessons and all(lesson.target in lesson.actions and lesson.target != lesson.rejected for lesson in lessons)
    learner = lessons[0].learner
    refined, accuracy = apply_lessons(genomes[learner], lessons, learner)
    assert accuracy >= 0.5 and not np.allclose(refined.weights, genomes[learner].weights)
    features = np.eye(OPTION_FEATURES, dtype=np.float32)[:4]
    trained, accuracy = train_mlp(MLPGenome.random(rng), [(features, 2, 1.0)] * 8, epochs=120, anchor=0.0)
    assert accuracy == 1.0 and int(np.argmax(trained.scores(features))) == 2


def test_stadium_environment_drives_the_campaign_policy_with_ppo() -> None:
    knowledge = SideKnowledge()
    environment = StadiumEnvironment(DEX, [[VENUSAUR, PIKACHU]], [([SWAMPERT, GENGAR], MLPGenome.prior(random.Random(1)))],
                                     knowledge, num_envs=2, seed=1)
    observation = environment.reset()
    assert observation.entities.shape == (2, 7, 48) and observation.action_mask.shape == (2, 9)
    learner = DeepLearner(seed=0)
    before = [parameter.detach().clone() for parameter in learner.network.parameters()]
    metrics = learner.ppo(environment, updates=1, steps=8)
    assert metrics["updates"] >= 1 and np.isfinite(metrics["value_loss"])
    assert any(not np.allclose(a.numpy(), b.detach().numpy()) for a, b in zip(before, learner.network.parameters()))


# ------------------------------------------------------------------- league
def test_league_runs_publishes_and_resumes(tmp_path: Path) -> None:
    from core.stadium.league import League, LeagueConfig

    config = LeagueConfig(generations=2, n_task=3, n_cell=3, budget=24, batch=12, n_init=6, repeats=1, workers=0,
                          duel_battles=1, lesson_points=4, rollouts=2, deep=False, anchor_battles=2, seed=2)
    run_dir = tmp_path / "league"
    League(config, run_dir=run_dir, log=lambda _: None).run()
    live = json.loads((run_dir / "live.json").read_text(encoding="utf-8"))
    assert live["generation"] == 2 and live["status"] == "stopped"
    assert live["sides"]["red"]["champion"] and live["sides"]["blue"]["champion"] and live["duel"]["battles"]
    assert live["metrics"]["red"]["win_rate"] is not None
    assert len((run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))["run"] == "league"
    resumed = League(LeagueConfig(**{**config.__dict__, "generations": 3}), run_dir=run_dir, resume=True,
                     log=lambda _: None)
    assert resumed.generation == 2 and resumed.state["red"].tasks
    resumed.run()
    assert json.loads((run_dir / "live.json").read_text(encoding="utf-8"))["generation"] == 3


def test_variant_report_handles_never_beaten_sets_and_significance(tmp_path: Path) -> None:
    from scripts.run_stadium import write_report

    rng = np.random.default_rng(0)
    finals: dict[str, list[tuple[str, int, Solution]]] = {"red": [], "blue": []}
    for replication in range(5):
        for variant in ("strong", "weak"):
            for index in range(3):
                uid = replication * 100 + index + (50 if variant == "weak" else 0)
                finals["red"].append((variant, replication, _solution(uid)))
                finals["blue"].append((variant, replication, _solution(uid + 10_000)))
    strong_rows = np.array([item[0] == "strong" for item in finals["red"]])
    matrix = np.where(strong_rows[:, None], 0.9, 0.2) + rng.normal(0, 0.02, (30, 30))
    report = write_report(tmp_path, finals, ["strong", "weak"], 5, np.clip(matrix, 0, 1), 3, 0, "# t", "intro", {})
    red = report["sides"]["red"]
    assert red["strong"]["win_rate"]["median"] > 0.95 and red["weak"]["win_rate"]["median"] < 0.05
    assert report["significance"]["red"]["win_rate"] == {"strong": True, "weak": False}
    assert red["strong"]["aqd"]["median"] >= red["strong"]["aqd"]["infinite"]  # never beaten -> best value
    assert "∞" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_lamarckian_variation_uses_the_rivals_lessons() -> None:
    from core.stadium.learning import Lesson

    bank = fallback_bank(DEX)
    spec = SideSpec("red", "firered", bank.roster("firered"), bank.parties["firered"], 40, 3)
    counter = iter(range(1, 1000))
    ops = SpaceOps(spec, DEX, random.Random(1), lambda: next(counter))
    parent = ops.random_solution()
    calls: list[int] = []

    def refine(genome: MLPGenome) -> MLPGenome:
        calls.append(1)
        return genome
    ops.refine, ops.refine_rate = refine, 1.0
    child = ops.vary(parent, None, GameConfig())
    assert child.origin == "lamarck" and calls
    lesson = Lesson(0, np.ones((3, OPTION_FEATURES), dtype=np.float32), (0, 1, 4), 4, 0, 0.2, 3, "imitar_rival")
    again = Lesson.from_state(json.loads(json.dumps(lesson.to_state())))
    assert again.actions == lesson.actions and again.target == 4 and np.allclose(again.features, lesson.features)


def test_evaluator_rebuilds_a_broken_worker_pool() -> None:
    bank = fallback_bank(DEX)
    sides = {"red": SideSpec("red", "firered", bank.roster("firered"), bank.parties["firered"], 40, 2),
             "blue": SideSpec("blue", "emerald", bank.roster("emerald"), bank.parties["emerald"], 40, 2)}
    counter = iter(range(1, 1000))
    red = SpaceOps(sides["red"], DEX, random.Random(1), lambda: next(counter)).random_solution()
    blue = SpaceOps(sides["blue"], DEX, random.Random(2), lambda: next(counter)).random_solution()
    with Evaluator(sides, workers=2, repeats=1, seed=0) as evaluator:
        assert len(evaluator.evaluate([(red, blue)])) == 1
        assert evaluator._pool is not None
        for process in list(evaluator._pool._processes.values()):  # simulate a killed worker
            process.kill()
            process.join()
        assert len(evaluator.evaluate([(red, blue), (red, blue)])) == 2
        assert evaluator.restarts >= 1
