"""Tests of ARES-Battle (E1): species pools, ID-free observations, networks, PPO, evaluation maths."""
from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from ares_u.battle.bc import bc_loss
from ares_u.battle.dqn import Replay, dqn_loss
from ares_u.battle.env import BattleEnv
from ares_u.battle.evaluate import evaluation_complete, summarise, wilson, write_json
from ares_u.battle.model import MASK_VALUE, EntityTransformer, ModelConfig, QNetwork, batch_tensors
from ares_u.battle.obs import (ACTION_DIM, ACTIONS, ENTITIES, ENTITY_DIM, FIELD_DIM, FLAT_DIM, encode, flat, stack)
from ares_u.battle.ppo import PPOConfig, explained_variance, make_optimizer, minibatches, ppo_update
from ares_u.battle.report import aggregate, bootstrap_ci, holm
from ares_u.battle.rollout import WorkerPool, gae
from ares_u.battle.tasks import demos, evaluate
from ares_u.battle.teams import (EVAL_SPLITS, GEN3_INTERNAL_FIRST, GEN3_INTERNAL_LAST, TEAM_SIZE, TrainTeamSampler,
                                 default_dex, eval_team, generation_of, heldout_signatures, legal_moves, species_pool)
from ares_u.battle.training import linear, split_indices
from core.stadium.dex import Dex
from core.stadium.engine import _MODELLED, MonSpec
from scripts.ares_u_battle import job_state

DEX = default_dex()


# ------------------------------------------------------------------ species pools
def test_generation_mapping_uses_internal_gen3_order() -> None:
    names = {index: DEX.species[index].name for index in range(len(DEX.species))}
    assert generation_of(1) == 1 and names[1] == "BULBASAUR"
    assert generation_of(151) == 1 and names[151] == "MEW"
    assert generation_of(152) == 2 and names[152] == "CHIKORITA"
    assert generation_of(251) == 2 and names[251] == "CELEBI"
    # Internal 252-276 are unused placeholders; Hoenn starts at internal 277 (national #252).
    assert all(generation_of(index) == 0 and names[index] == "?" for index in range(252, 277))
    assert names[GEN3_INTERNAL_FIRST] == "TREECKO" and generation_of(GEN3_INTERNAL_FIRST) == 3
    assert GEN3_INTERNAL_LAST - GEN3_INTERNAL_FIRST + 1 == 386 - 251
    gen3_names = {names[index] for index in range(GEN3_INTERNAL_FIRST, GEN3_INTERNAL_LAST + 1)}
    assert {"TREECKO", "BLAZIKEN", "RAYQUAZA", "JIRACHI", "DEOXYS", "CHIMECHO"} <= gen3_names
    assert not gen3_names & {names[index] for index in range(1, 252)}
    assert generation_of(0) == 0 and generation_of(412) == 0


def test_species_pools_are_disjoint_and_eligible() -> None:
    pools = {generation: species_pool(DEX, generation) for generation in (1, 2, 3)}
    assert len(pools[1]) > 140 and len(pools[2]) > 90 and len(pools[3]) > 120
    assert all(1 <= s <= 151 for s in pools[1])
    assert all(152 <= s <= 251 for s in pools[2])
    assert all(GEN3_INTERNAL_FIRST <= s <= GEN3_INTERNAL_LAST for s in pools[3])
    assert "SHEDINJA" not in {DEX.species[s].name for s in pools[3]}
    for pool in pools.values():
        for species in pool:
            moves = legal_moves(DEX, species)
            assert any(DEX.moves[move].power > 0 for move in moves)
            assert all(DEX.moves[move].effect in _MODELLED for move in moves)


def test_teams_are_deterministic_legal_and_heldout_is_never_trained_on() -> None:
    first, second = eval_team(DEX, "gen3", 7, 1), eval_team(DEX, "gen3", 7, 1)
    assert first == second and len(first) == TEAM_SIZE
    assert len({member.species for member in first}) == TEAM_SIZE
    for member in first:
        assert generation_of(member.species) == 3 and member.level == 50
        assert 1 <= len(member.moves) <= 4 and len(set(member.moves)) == len(member.moves)
        assert set(member.moves) <= set(legal_moves(DEX, member.species))
        assert any(DEX.moves[move].power > 0 for move in member.moves)
    forbidden = heldout_signatures(DEX)
    assert frozenset(m.species for m in eval_team(DEX, "gen1_heldout", 3, 0)) in forbidden
    sampler = TrainTeamSampler(DEX, random.Random(5))
    for _ in range(300):
        team = sampler.team()
        assert all(generation_of(member.species) == 1 for member in team)
        assert frozenset(member.species for member in team) not in forbidden
    assert TrainTeamSampler(DEX, random.Random(9)).team() == TrainTeamSampler(DEX, random.Random(9)).team()


# ------------------------------------------------------------------ observations
def _relabelled_dex(species_from: int, species_to: int, move_from: int, move_to: int) -> Dex:
    """A dex where ``species_to``/``move_to`` carry exactly the data of ``species_from``/``move_from``."""
    species = list(DEX.species)
    species[species_to] = replace(DEX.species[species_from], species_id=species_to,
                                  name=DEX.species[species_to].name)
    moves = list(DEX.moves)
    moves[move_to] = replace(DEX.moves[move_from], move_id=move_to, name=DEX.moves[move_to].name)
    return Dex(DEX.game, tuple(moves), tuple(species), DEX.type_entries, DEX.evolutions, DEX.learnsets, "test")


def test_observation_contains_no_species_or_move_ids() -> None:
    team_a, team_b = eval_team(DEX, "gen1_heldout", 0, 0), eval_team(DEX, "gen1_heldout", 0, 1)
    source = team_a[0]
    target_species = 400 if source.species != 400 else 401
    source_move = source.moves[0]
    target_move = 300 if source_move != 300 else 301
    dex = _relabelled_dex(source.species, target_species, source_move, target_move)
    renamed = replace(source, species=target_species,
                      moves=tuple(target_move if move == source_move else move for move in source.moves))
    original = BattleEnv(dex, team_a, team_b, seed=3).observe(0)
    relabelled = BattleEnv(dex, [renamed, *team_a[1:]], team_b, seed=3).observe(0)
    assert np.array_equal(original.field, relabelled.field)
    assert np.array_equal(original.entities, relabelled.entities)
    assert np.array_equal(original.actions, relabelled.actions)
    assert np.array_equal(original.mask, relabelled.mask)
    assert original.species[0] != relabelled.species[0]  # only the ablation channel sees the ID


def test_observation_shapes_masks_and_hidden_foes() -> None:
    env = BattleEnv(DEX, eval_team(DEX, "gen2", 1, 0), eval_team(DEX, "gen2", 1, 1), seed=11)
    obs = env.observe(0)
    assert obs.field.shape == (FIELD_DIM,) and obs.entities.shape == (ENTITIES, ENTITY_DIM)
    assert obs.actions.shape == (ACTIONS, ACTION_DIM) and obs.mask.shape == (ACTIONS,)
    assert flat(obs).shape == (FLAT_DIM,)
    legal = env.legal(0)
    assert [index for index in range(ACTIONS) if obs.mask[index]] == legal
    # Foe bench members have not been revealed: hidden tokens with no species information.
    for position in (TEAM_SIZE + 1, TEAM_SIZE + 2):
        assert obs.entities[position, 1] == 1.0 and obs.entities[position, 4:].sum() == 0.0
        assert obs.species[position] == 0
    assert obs.entities[TEAM_SIZE, 1] == 0.0 and obs.entities[0, 2] == 1.0 and obs.entities[0, 3] == 1.0
    rng = random.Random(0)
    replacement_seen = False
    while not env.over:
        actions = {side: rng.choice(env.legal(side)) for side in env.pending()}
        env.step(actions)
        for side in env.pending():
            observed = env.observe(side)
            assert [i for i in range(ACTIONS) if observed.mask[i]] == env.legal(side)
            if env.battle._replacing:
                replacement_seen = True
                assert not observed.mask[:4].any() and observed.field[20] == 1.0
    assert replacement_seen


def test_switch_tokens_follow_bench_order() -> None:
    env = BattleEnv(DEX, eval_team(DEX, "gen3", 2, 0), eval_team(DEX, "gen3", 2, 1), seed=1)
    obs = env.observe(0)
    bench = env.battle.sides[0].bench()
    for position, index in enumerate(bench):
        fighter = env.battle.sides[0].party[index]
        assert obs.actions[4 + position, 1] == 1.0
        assert obs.actions[4 + position, 38] == pytest.approx(fighter.hp / fighter.max_hp)
        assert obs.entities[1 + position, 29] == pytest.approx(fighter.hp / fighter.max_hp)


# ------------------------------------------------------------------ networks
def _batch(count: int = 5) -> tuple[torch.Tensor, ...]:
    observations = []
    for index in range(count):
        env = BattleEnv(DEX, eval_team(DEX, "gen1_heldout", index, 0), eval_team(DEX, "gen1_heldout", index, 1),
                        seed=index)
        observations.append(env.observe(0))
    return batch_tensors(stack(observations))


def test_transformer_shapes_and_masking() -> None:
    torch.manual_seed(0)
    model = EntityTransformer(ModelConfig(d_model=32, layers=2, heads=4, ff=64))
    field_t, entities, actions, mask, species = _batch()
    logits, values = model(field_t, entities, actions, mask, species)
    assert logits.shape == (5, ACTIONS) and values.shape == (5,)
    assert torch.all(logits[~mask] == MASK_VALUE)
    assert torch.all(logits[mask] > MASK_VALUE / 2)
    probabilities = torch.softmax(logits, dim=-1)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(5))
    assert torch.all(probabilities[~mask] == 0)
    # The agnostic model ignores species ids; the ablation model uses them.
    shuffled = torch.randint(1, 400, species.shape)
    same, _ = model(field_t, entities, actions, mask, shuffled)
    assert torch.allclose(same, logits)
    ids_model = EntityTransformer(ModelConfig(d_model=32, layers=1, heads=2, ff=64, species_ids=True))
    with_ids, _ = ids_model(field_t, entities, actions, mask, species)
    changed, _ = ids_model(field_t, entities, actions, mask, shuffled)
    assert not torch.allclose(with_ids[mask], changed[mask])


def test_q_network_shape() -> None:
    q = QNetwork(hidden=32)(torch.zeros(3, FLAT_DIM))
    assert q.shape == (3, ACTIONS)


def test_ppo_update_on_cpu_improves_a_toy_objective() -> None:
    torch.manual_seed(0)
    model = EntityTransformer(ModelConfig(d_model=32, layers=1, heads=2, ff=64))
    field_t, entities, actions, mask, species = _batch(8)
    size = 64
    repeat = size // 8
    data = {"field": field_t.numpy().repeat(repeat, 0), "entities": entities.numpy().repeat(repeat, 0),
            "actions": actions.numpy().repeat(repeat, 0), "mask": mask.numpy().repeat(repeat, 0),
            "species": species.numpy().repeat(repeat, 0)}
    with torch.no_grad():
        logits, values = model(*(torch.as_tensor(data[key]) for key in ("field", "entities", "actions", "mask",
                                                                           "species")))
    legal = data["mask"]
    rng = np.random.default_rng(0)
    chosen = np.array([rng.choice(np.flatnonzero(row)) for row in legal])
    target = int(np.flatnonzero(legal[0])[0])  # the toy task: action ``target`` is good when legal
    advantage = np.where(chosen == target, 1.0, -0.2).astype(np.float32)
    log_probs = torch.log_softmax(logits, dim=-1).numpy()
    data.update({"act": chosen.astype(np.int64), "logp": log_probs[np.arange(size), chosen].astype(np.float32),
                 "value": values.numpy().astype(np.float32), "adv": advantage,
                 "ret": (values.numpy() + advantage).astype(np.float32)})

    def toy_loss() -> float:
        with torch.no_grad():
            new_logits, new_values = model(*(torch.as_tensor(data[k]) for k in ("field", "entities", "actions",
                                                                               "mask", "species")))
            probabilities = torch.softmax(new_logits, dim=-1)[:, target]
            value_error = ((new_values - torch.as_tensor(data["ret"])) ** 2).mean()
        return float(-probabilities.mean() + value_error)

    before = toy_loss()
    optimizer = make_optimizer(model, 3e-3)
    config = PPOConfig(epochs=4, minibatch=16, ent_coef=0.0, lr=3e-3, lr_warmup=1, target_kl=0.0)
    stats = ppo_update(model, optimizer, data, config, torch.device("cpu"), np.random.default_rng(1))
    assert toy_loss() < before
    assert set(stats) == {"policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "kl_first", "kl_update",
                          "updates_frac", "epochs_run", "explained_var"}
    assert all(np.isfinite(value) for value in stats.values())
    assert stats["updates_frac"] == 1.0 and stats["epochs_run"] == 4.0
    assert abs(stats["kl_first"]) < 1e-5  # the behaviour policy is the start-of-update policy here
    # Target-KL early stopping: a huge step size breaks the update off before the planned minibatches.
    aggressive = PPOConfig(epochs=4, minibatch=16, ent_coef=0.0, lr=0.5, lr_warmup=1, target_kl=1e-4)
    stopped = ppo_update(model, make_optimizer(model, 0.5), data, aggressive, torch.device("cpu"),
                         np.random.default_rng(2))
    assert stopped["updates_frac"] < 1.0 and stopped["kl_update"] > 1e-4


def test_minibatches_schedules_and_explained_variance() -> None:
    blocks = minibatches(10, 4, np.random.default_rng(0))
    assert len(blocks) == 2 and all(len(block) == 4 for block in blocks)
    assert len(set(np.concatenate(blocks).tolist())) == 8  # disjoint; the remainder is dropped
    assert [len(block) for block in minibatches(3, 8, np.random.default_rng(0))] == [3]
    assert linear(1.0, 0.1, 0.0) == 1.0 and linear(1.0, 0.1, 1.0) == pytest.approx(0.1)
    assert linear(1.0, 0.1, 0.5) == pytest.approx(0.55) and linear(1.0, 0.1, 3.0) == pytest.approx(0.1)
    returns = np.array([1.0, -1.0, 0.5, 0.0])
    assert explained_variance(returns, returns) == pytest.approx(1.0)
    assert explained_variance(np.zeros(4), returns) == pytest.approx(0.0)
    config = PPOConfig(total_steps=1_000, step_offset=200)
    assert config.progress(200) == 0.0 and config.progress(600) == pytest.approx(0.5)


def test_dqn_replay_and_loss() -> None:
    replay = Replay(8)
    count = 12
    data = {"state": np.ones((count, FLAT_DIM), dtype=np.float16), "next": np.zeros((count, FLAT_DIM), np.float16),
            "next_mask": np.ones((count, ACTIONS), dtype=np.bool_), "act": np.arange(count) % ACTIONS,
            "reward": np.arange(count, dtype=np.float32), "done": np.arange(count) % 2 == 0}
    replay.add(data)
    assert replay.size == 8 and replay.cursor == count % 8
    assert set(replay.reward.tolist()) == set(range(4, 12))
    online, target = QNetwork(hidden=16), QNetwork(hidden=16)
    loss = dqn_loss(online, target, replay.sample(np.random.default_rng(0), 4), 0.99, torch.device("cpu"))
    assert loss.shape == () and torch.isfinite(loss)


# ------------------------------------------------------------------ returns and evaluation maths
def test_gae_matches_manual_computation() -> None:
    rewards, values = [0.1, -0.2, 1.0], [0.5, 0.4, 0.3]
    gamma, lam = 0.9, 0.8
    deltas = [0.1 + 0.9 * 0.4 - 0.5, -0.2 + 0.9 * 0.3 - 0.4, 1.0 - 0.3]
    expected = [deltas[0] + 0.72 * (deltas[1] + 0.72 * deltas[2]), deltas[1] + 0.72 * deltas[2], deltas[2]]
    advantages, returns = gae(rewards, values, gamma, lam)
    assert np.allclose(advantages, expected, atol=1e-6)
    assert np.allclose(returns, np.asarray(expected) + np.asarray(values), atol=1e-6)


def test_wilson_interval_known_values() -> None:
    point, low, high = wilson(50, 100)
    assert point == 0.5 and low == pytest.approx(0.4038, abs=1e-4) and high == pytest.approx(0.5962, abs=1e-4)
    point, low, high = wilson(0, 10)
    assert point == 0.0 and low == 0.0 and high == pytest.approx(0.2775, abs=1e-4)
    point, low, high = wilson(10, 10)
    assert low == pytest.approx(0.7225, abs=1e-4) and high == pytest.approx(1.0)
    assert wilson(0, 0) == (0.0, 0.0, 1.0)


def test_summarise_counts_draws_and_vices() -> None:
    records = [{"outcome": 1.0, "turns": 10, "voluntary": 4, "switches": 2, "panic_switches": 1,
                "move_choices": 2, "ineffective": 1},
               {"outcome": 0.0, "turns": 20, "voluntary": 4, "switches": 0, "panic_switches": 0,
                "move_choices": 4, "ineffective": 0},
               {"outcome": -1.0, "turns": 30, "voluntary": 2, "switches": 0, "panic_switches": 0,
                "move_choices": 2, "ineffective": 0}]
    summary = summarise(records)
    assert summary["score"] == pytest.approx(0.5) and summary["draws"] == 1
    assert summary["mean_turns"] == 20 and summary["switch_rate"] == pytest.approx(0.2)
    assert summary["panic_switch_rate"] == pytest.approx(0.1) and summary["ineffective_rate"] == pytest.approx(0.125)


def test_holm_bootstrap_and_chunking() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    low, high = bootstrap_ci([0.5, 0.6, 0.7])
    assert 0.5 <= low <= 0.6 <= high <= 0.7
    chunks = split_indices(11, 3)
    assert sorted(index for chunk in chunks for index in chunk) == list(range(11))
    assert all(chunk[0] % 2 == 0 for chunk in chunks)


def test_evaluation_is_deterministic_and_alternates_sides() -> None:
    task = {"split": "gen2", "indices": list(range(6)), "agent": {"heuristic": "greedy"},
            "opponent": {"heuristic": "random"}}
    first = evaluate(task, {"worker_id": 0})
    second = evaluate(task, {"worker_id": 1})
    assert first == second and [record["index"] for record in first] == list(range(6))
    mirror = evaluate({**task, "opponent": {"heuristic": "greedy"}, "indices": [0, 1]}, {"worker_id": 0})
    assert len(mirror) == 2


def test_teams_build_valid_monspecs() -> None:
    member = eval_team(DEX, "gen1_heldout", 0, 0)[0]
    assert isinstance(member, MonSpec) and member.stats[0] > 0


# ------------------------------------------------------------------ imitation warm-start
def test_demonstrations_are_legal_expert_decisions_on_gen1_training_teams() -> None:
    task = {"seed": 3, "chunk": 0, "steps": 60, "experts": {"prior": 0.5, "greedy": 0.5},
            "opponents": {"greedy": 0.5, "random": 0.5}, "gamma": 0.99, "concurrency": 4}
    data = demos(task, {"worker_id": 0})
    size = len(data["act"])
    assert size >= 60
    for key in ("field", "entities", "actions", "mask", "species", "ret", "source"):
        assert data[key].shape[0] == size
    assert data["mask"][np.arange(size), data["act"]].all()  # every expert action is legal
    assert set(data["source"].tolist()) <= {0, 1}
    assert np.all(np.abs(data["ret"]) <= 1.5) and np.isfinite(data["ret"]).all()
    gen1 = set(species_pool(DEX, 1))
    assert set(data["species"][data["species"] > 0].tolist()) <= gen1  # Gen1 only, never Gen2/Gen3
    assert np.array_equal(data["act"], demos(task, {"worker_id": 0})["act"])  # deterministic given the seed
    assert {item["expert"] for item in data["stats"]} <= {"prior", "greedy"}


def test_bc_loss_respects_the_mask_and_learns_the_expert() -> None:
    torch.manual_seed(0)
    data = demos({"seed": 1, "chunk": 0, "steps": 64, "experts": {"greedy": 1.0}, "opponents": {"random": 1.0},
                  "concurrency": 8}, {"worker_id": 0})
    batch = {key: torch.as_tensor(data[key])
             for key in ("field", "entities", "actions", "mask", "species", "act", "ret")}
    model = EntityTransformer(ModelConfig(d_model=32, layers=1, heads=2, ff=64))
    total, cross_entropy, value_loss, prediction = bc_loss(model, batch, 0.1, 0.5)
    assert torch.isfinite(total) and torch.isfinite(cross_entropy) and torch.isfinite(value_loss)
    assert prediction.shape == batch["act"].shape
    assert bool(batch["mask"][torch.arange(len(prediction)), prediction].all())
    optimizer = make_optimizer(model, 3e-3)
    for _ in range(40):
        loss = bc_loss(model, batch, 0.1, 0.5)[0]
        optimizer.zero_grad()
        torch.autograd.backward(loss)
        optimizer.step()
    after_total, _, _, after_prediction = bc_loss(model, batch, 0.1, 0.5)
    assert float(after_total) < float(total)
    assert float((after_prediction == batch["act"]).float().mean()) > 0.8


# ------------------------------------------------------------------ evaluation splits, resumability, report
def test_gen1_train_split_follows_the_training_distribution() -> None:
    assert EVAL_SPLITS == ("gen1_train", "gen1_heldout", "gen2", "gen3")
    forbidden = heldout_signatures(DEX)
    for index in range(40):
        team = eval_team(DEX, "gen1_train", index, index % 2)
        assert all(generation_of(member.species) == 1 for member in team)
        assert frozenset(member.species for member in team) not in forbidden
    assert eval_team(DEX, "gen1_train", 5, 0) == eval_team(DEX, "gen1_train", 5, 0)
    assert eval_team(DEX, "gen1_train", 5, 0) != eval_team(DEX, "gen1_heldout", 5, 0)


def _fake_eval(battles: int, splits: tuple[str, ...] = EVAL_SPLITS, score: float = 0.5) -> dict[str, Any]:
    wins = int(battles * score)
    cell = {"n": battles, "score": score, "ci_low": 0.4, "ci_high": 0.6, "wins": wins, "losses": battles - wins,
            "draws": 0, "mean_turns": 9.0, "switch_rate": 0.0, "panic_switch_rate": 0.0, "ineffective_rate": 0.0}
    return {"results": {split: {opponent: dict(cell) for opponent in ("random", "greedy", "prior")}
                        for split in splits}}


def test_resumable_protocol_job_states(tmp_path: Path) -> None:
    out = tmp_path / "ppo" / "seed0"
    assert job_state(tmp_path, "ppo", 0, 500) == "todo"
    out.mkdir(parents=True)
    (out / "final.pt").write_bytes(b"checkpoint")
    assert job_state(tmp_path, "ppo", 0, 500) == "trained"
    write_json(out / "eval.json", _fake_eval(200))
    assert job_state(tmp_path, "ppo", 0, 500) == "trained"  # too few battles for this protocol
    write_json(out / "eval.json", _fake_eval(500, splits=("gen1_heldout", "gen2", "gen3")))
    assert not evaluation_complete(out / "eval.json", 500)  # an evaluation without gen1_train is incomplete
    write_json(out / "eval.json", _fake_eval(500))
    assert job_state(tmp_path, "ppo", 0, 500) == "done" and not (out / "eval.json.tmp").exists()
    (out / "eval.json").write_text("{truncated", encoding="utf-8")
    assert job_state(tmp_path, "ppo", 0, 500) == "trained"


def test_report_planned_contrasts_and_generalisation_gap() -> None:
    def seeds(scores: list[float], drop: float) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for index, score in enumerate(scores):
            evaluation = _fake_eval(500, score=score)
            for split in ("gen2", "gen3"):
                for cell in evaluation["results"][split].values():
                    cell["score"] = score - drop
            result[f"seed{index}"] = {"eval": evaluation, "config": {"device_used": "cpu"}, "metrics": [],
                                      "budget": {"steps": 100, "demo_steps": 0, "rl_steps": 100, "wall": 1.0,
                                                 "bc_shared": False}, "learn_s": 0.5, "rollout_s": 0.5}
        return result

    data = {"methods": {"ppo": seeds([0.50, 0.51, 0.52, 0.53, 0.54], 0.05),
                        "bc_ppo": seeds([0.60, 0.61, 0.62, 0.63, 0.64], 0.0)}, "baselines": None}
    summary = aggregate(data)
    planned = [t for t in summary["tests"] if t["split"] == "gen1_train" and t["opponent"] == "greedy"]
    assert len(planned) == 1 and planned[0]["a"] == "bc_ppo" and planned[0]["b"] == "ppo"
    assert planned[0]["p"] == pytest.approx(2 / 252) and planned[0]["A"] == 1.0
    assert planned[0]["p_holm"] >= planned[0]["p"]
    gap = summary["gaps"]["ppo"]["greedy"]
    assert gap["gen3"]["mean"] == pytest.approx(0.05) and gap["gen1_heldout"]["mean"] == pytest.approx(0.0)
    assert summary["gaps"]["bc_ppo"]["greedy"]["gen2"]["mean"] == pytest.approx(0.0)
    parity = [row for row in summary["parity"] if row["method"] == "bc_ppo" and row["opponent"] == "greedy"]
    assert len(parity) == 4 and all(row["p_holm"] < 0.05 for row in parity)


def test_worker_pool_pipelined_submit_and_collect() -> None:
    with WorkerPool(2) as pool:
        pool.submit([{"type": "ping"}, {"type": "ping"}])
        assert pool.pending
        with pytest.raises(RuntimeError):
            pool.submit([{"type": "ping"}, {"type": "ping"}])
        with pytest.raises(RuntimeError):
            pool.map([{"type": "ping"}])
        assert pool.collect() == [0, 1] and not pool.pending
        with pytest.raises(RuntimeError):
            pool.collect()
        assert pool.broadcast([{"type": "ping"}, {"type": "ping"}]) == [0, 1]
        pool.submit([{"type": "ping"}, {"type": "ping"}])  # close() drains an abandoned submission


def test_review_fixes_hidden_power_and_public_foe_margin() -> None:
    """Regression for the design review: Hidden Power resolves like the engine; foe HP margin is public."""
    from core.stadium.engine import Battle as _Battle

    from ares_u.battle.obs import OracleKnowledge as _Knowledge
    from ares_u.battle.obs import encode as _encode
    from ares_u.battle.obs import hidden_power, resolve_move
    from ares_u.battle.teams import eval_team as _eval_team

    from core.stadium.dex import load_dex as _load_dex

    dex = _load_dex(prefer_snapshot=True)
    team_a, team_b = _eval_team(dex, "gen3", 0, 0), _eval_team(dex, "gen3", 0, 1)
    battle = _Battle(dex, team_a, team_b, seed=1, record=False)
    me = battle.sides[0].fighter
    hp_moves = [move for move in dex.moves if move.effect == 135]
    if hp_moves:
        move = hp_moves[0]
        expected = battle._move_type_power(0, move, 135)
        assert hidden_power(me.spec.ivs) == expected
        assert resolve_move(battle, me, move).move_type == expected[0]
    # Damage only the hidden bench of the foe: the public margin must not change.
    obs_before = _encode(battle, 0, ({0}, {battle.sides[1].active}), _Knowledge(dex=dex))
    bench = battle.sides[1].bench()
    if bench:
        battle.sides[1].party[bench[0]].hp = 1
        obs_after = _encode(battle, 0, ({0}, {battle.sides[1].active}), _Knowledge(dex=dex))
        assert obs_before.field[19] == obs_after.field[19]
