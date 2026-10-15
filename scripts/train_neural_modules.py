"""Train all requested neural modules on RAM transitions and actual HP battles."""
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn
from torch.autograd import backward
from torch.nn import functional as F
from torch.optim.adam import Adam

from core.battle.mamba_entity_net import AresBattlePolicyNetwork
from core.battle.psro_league_trainer import PSROLeagueTrainer
from core.devices import ModelProbe, choose_training_device
from core.game_data import ROOT, RomCatalog
from core.planner.gflownet_team import TeamCandidate, TeamSynthesisGFlowNet
from core.training_data import collect_world_transitions
from core.world_model.hjepa import LatentHJEPAWorldModel


def _catalog() -> tuple[tuple[TeamCandidate, ...], Tensor]:
    """Enumerate legal ROM-internal species/moves, masking unused Unown slots."""
    rom = RomCatalog("firered")
    candidates: list[TeamCandidate] = []
    allowed = torch.zeros(411, dtype=torch.bool)
    for species_id in range(1, 412):
        valid = species_id <= 251 or species_id >= 277
        species = rom.species(species_id)
        legal = tuple(rom.level_up_moves(species_id)) if valid else ()
        candidates.append(TeamCandidate(species_id, species.types, legal))
        allowed[species_id - 1] = bool(legal)
    return tuple(candidates), allowed


def _tb_batch(model: TeamSynthesisGFlowNet, catalog: tuple[TeamCandidate, ...],
              available: Tensor, count: int = 8) -> Tensor:
    """Train species AND move construction with the complete trajectory ratio."""
    states: list[Tensor] = []
    ratios: list[Tensor] = []
    rewards: list[float] = []
    for _ in range(count):
        trajectory = model.sample_team_and_moves(catalog, available_mask=available)
        team = tuple(catalog[species - 1] for species in trajectory.species)
        states.append(trajectory.state)
        ratios.append(trajectory.log_pf_minus_pb)
        rewards.append(math.log(model.coverage_reward(team)))
    return model.compute_trajectory_balance_loss(torch.stack(states), torch.stack(ratios),
                                                torch.tensor(rewards, device=model.log_Z.device))


def train(steps: int = 128, seed: int = 42, requested_device: str = "auto") -> dict[str, object]:
    """Measure before/after losses, real RL outcomes, and persist reusable weights."""
    if steps < 8:
        raise ValueError("At least8 updates required")
    torch.set_num_threads(8)
    torch.manual_seed(seed)
    states, actions, successors = collect_world_transitions(1024, seed)
    world = LatentHJEPAWorldModel()
    flow = TeamSynthesisGFlowNet(411)
    policy = AresBattlePolicyNetwork()
    catalog, allowed = _catalog()
    probe_entities = torch.randn(4, 7, 48)
    probe_roles = torch.tensor([[0, 1, 2, 3, 4, 4, 6]] * 4)

    def world_probe(module: nn.Module, device: torch.device) -> Tensor:
        active = cast(LatentHJEPAWorldModel, module)
        return active.forward_loss(states[:64].to(device), actions[:64].to(device), successors[:64].to(device))[0]

    def flow_probe(module: nn.Module, device: torch.device) -> Tensor:
        active = cast(TeamSynthesisGFlowNet, module)
        state = torch.zeros(411, device=device)
        ratio = active.log_Z * 0
        for step in range(6):
            logits = cast(Tensor, active.forward_policy_net(state))
            log_probabilities = F.log_softmax(logits.masked_fill(state.bool(), -torch.inf), dim=-1)
            ratio = ratio + log_probabilities[step] + math.log(step + 1)
            state = state.clone()
            state[step] = 1
        move_input = torch.cat((state, torch.zeros(355, device=device)))
        move_logits = cast(Tensor, active.move_policy_net(move_input))
        return (active.log_Z + ratio).square() + move_logits.square().mean()

    def policy_probe(module: nn.Module, device: torch.device) -> Tensor:
        active = cast(AresBattlePolicyNetwork, module)
        logits, targets, value, _ = active.forward_turn(probe_entities.to(device), probe_roles.to(device),
            torch.ones(4, 9, dtype=torch.bool, device=device), active.initial_state(4),
            torch.ones(4, 3, dtype=torch.bool, device=device))
        return logits.square().mean() + targets.square().mean() + value.square().mean()

    probes: list[tuple[nn.Module, ModelProbe]] = [(world, world_probe), (flow, flow_probe), (policy, policy_probe)]
    selection = choose_training_device(probes, requested_device)
    print(f"Backend: {selection.backend}; {selection.reason}", flush=True)
    world.to(selection.device)
    flow.to(selection.device)
    train_states, train_actions, train_next = (tensor[:896].to(selection.device) for tensor in (states, actions, successors))
    valid_states, valid_actions, valid_next = (tensor[896:].to(selection.device) for tensor in (states, actions, successors))
    allowed = allowed.to(selection.device)
    with torch.no_grad():
        initial_world = float(world.forward_loss(valid_states, valid_actions, valid_next)[0].cpu())
        torch.manual_seed(seed + 1)
        initial_flow = float(_tb_batch(flow, catalog, allowed, 16).cpu())
    world_optimizer = Adam(world.parameters(), lr=2e-3)
    flow_optimizer = Adam([{"params": [p for name, p in flow.named_parameters() if name != "log_Z"], "lr": 2e-3},
                          {"params": [flow.log_Z], "lr": 0.4}])
    history: list[dict[str, float]] = []
    for step in range(steps):
        indices = torch.randint(0, 896, (64,)).to(selection.device)
        world_optimizer.zero_grad()
        world_loss, _ = world.forward_loss(train_states[indices], train_actions[indices], train_next[indices])
        backward(world_loss)
        torch.nn.utils.clip_grad_norm_(world.parameters(), 10)
        world_optimizer.step()
        world.update_target_ema()
        flow_optimizer.zero_grad()
        flow_loss = _tb_batch(flow, catalog, allowed)
        backward(flow_loss)
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 100)
        flow_optimizer.step()
        if step % 16 == 0 or step == steps - 1:
            entry = {"update": float(step), "vicreg": float(world_loss.detach().cpu()), "trajectory_balance": float(flow_loss.detach().cpu())}
            history.append(entry)
            print(json.dumps(entry), flush=True)
    with torch.no_grad():
        final_world = float(world.forward_loss(valid_states, valid_actions, valid_next)[0].cpu())
        torch.manual_seed(seed + 1)
        final_flow = float(_tb_batch(flow, catalog, allowed, 16).cpu())
    # Recurrent PPO uses the validated CPU integer arena, independently of optional
    # DirectML selection for the portable world/team networks.
    trainer = PSROLeagueTrainer(policy=policy.cpu(), seed=seed)
    initial_policy = trainer.evaluate(episodes=64)
    initial_snapshot = copy.deepcopy(trainer.policy).eval()
    initial_snapshot.requires_grad_(False)
    best_weights = copy.deepcopy(trainer.policy.state_dict())
    best_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    final_policy = initial_policy
    reinforcement_history: list[dict[str, float]] = []
    # Establish a competent baseline before training against a weaker frozen
    # self-policy. Validation selects measured checkpoints; no label imitation.
    for cycle in range(32):
        reinforcement_history.extend(trainer.train_best_response(updates=8, rollout_steps=32, num_envs=2))
        measured = trainer.evaluate(episodes=64)
        print(json.dumps({"ppo_cycle": cycle, "win_rate": measured["win_rate"],
                          "value_loss": reinforcement_history[-1]["value_loss"]}), flush=True)
        if measured["win_rate"] > final_policy["win_rate"]:
            final_policy = measured
            best_weights = copy.deepcopy(trainer.policy.state_dict())
            best_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
        if cycle >= 1 and final_policy["win_rate"] >= min(1.0, initial_policy["win_rate"] + 0.05):
            break
    trainer.policy.load_state_dict(best_weights)
    trainer.optimizer.load_state_dict(best_optimizer)
    trainer.frozen_policies.append(initial_snapshot)
    trainer.payoff_matrix = torch.zeros(1, 1)
    trainer.nash_distribution = torch.ones(1)
    trainer.freeze_current_policy()
    payoff_matrix = trainer.evaluate_league(episodes_per_match=16)
    league_history = trainer.train_best_response(updates=8, rollout_steps=32, num_envs=2)
    reinforcement_history.extend(league_history)
    league_validation = trainer.evaluate(episodes=64)
    if league_validation["win_rate"] >= final_policy["win_rate"]:
        final_policy = league_validation
        best_weights = copy.deepcopy(trainer.policy.state_dict())
        best_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    else:
        trainer.policy.load_state_dict(best_weights)
        trainer.optimizer.load_state_dict(best_optimizer)
    result: dict[str, object] = {
        "backend": selection.backend, "backend_reason": selection.reason,
        "cpu_probe_seconds": selection.cpu_seconds, "directml_probe_seconds": selection.directml_seconds,
        "vicreg": {"initial": initial_world, "final": final_world},
        "trajectory_balance": {"initial": initial_flow, "final": final_flow},
        "ppo": {"initial": initial_policy, "final": final_policy, "updates": reinforcement_history},
        "psro": {"payoffs": payoff_matrix.tolist(), "nash_distribution": trainer.nash_distribution.tolist()},
        "world_updates": history, "seed": seed,
    }
    directory = ROOT / "artifacts" / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = {"world_model": world.cpu().state_dict(), "gflownet": flow.cpu().state_dict(),
        "battle_policy": trainer.policy.state_dict(), "world_optimizer": world_optimizer.state_dict(),
        "flow_optimizer": flow_optimizer.state_dict(), "ppo_optimizer": trainer.optimizer.state_dict(),
        "metadata": {"state_dim": 96, "action_dim": 16, "latent_dim": 128, "entity_dim": 48,
                     "gflownet_species": 411, "seed": seed}, "metrics": result}
    temporary = directory / "neural.pt.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(directory / "neural.pt")
    (directory / "training_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not math.isfinite(final_world) or not math.isfinite(final_flow) or final_world >= initial_world or final_flow >= initial_flow:
        raise RuntimeError(f"Neural loss gate failed: VICReg{initial_world}->{final_world}; TB{initial_flow}->{final_flow}")
    if final_policy["win_rate"] < initial_policy["win_rate"]:
        raise RuntimeError(f"RL evaluation regressed:{initial_policy}->{final_policy}")
    print(json.dumps({"vicreg": result["vicreg"], "trajectory_balance": result["trajectory_balance"],
                      "initial_win_rate": initial_policy["win_rate"], "final_win_rate": final_policy["win_rate"],
                      "checkpoint": str(directory / "neural.pt")}), flush=True)
    return result


def main() -> None:
    """Run reproducible training gates with an explicit hardware preference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "directml"), default="auto")
    args = parser.parse_args()
    train(args.steps, args.seed, args.device)


if __name__ == "__main__":
    main()
