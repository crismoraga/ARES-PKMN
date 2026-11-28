"""Rivals of the league: specs (picklable dictionaries shipped to workers) and the controllers they become.

Spec forms (all carry ``name``):

* ``{"archetype": "stall"}``          - a hand-written style of ``ares_u.league.archetypes``;
* ``{"lookahead": 8}``                 - the full-information Monte-Carlo planner (``ares_u.battle.headroom``);
* ``{"heuristic": "greedy"}``          - an E1 heuristic baseline (random / greedy / prior);
* ``{"method": "ppo", "weights": W, "greedy": True}`` - a neural agent (``dqn`` = Q-network).

The neural personalities are the final E1 checkpoints (``artifacts/ares_u/battle/final``): PPO seeds,
behaviour cloning, BC -> PPO, PPO with species IDs and the DQN baseline. Exploiters and snapshots of
the main agent are neural specs created during league training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ares_u.battle.agents import Controller, HeuristicController, make_neural
from ares_u.battle.headroom import LookaheadController

from .archetypes import ARCHETYPES, DESCRIPTIONS, ArchetypeController

ROOT = Path(__file__).resolve().parents[2]
E1 = ROOT / "artifacts" / "ares_u" / "battle" / "final"
LOOKAHEAD_ROLLOUTS = 8
NEURAL: dict[str, Path] = {
    **{f"e1_ppo_s{seed}": E1 / "ppo" / f"seed{seed}" / "final.pt" for seed in range(5)},
    "e1_bc_s0": E1 / "bc" / "seed0" / "final.pt",
    "e1_bcppo_s0": E1 / "bc_ppo" / "seed0" / "final.pt",
    "e1_ids_s0": E1 / "ppo_ids" / "seed0" / "final.pt",
    "e1_dqn_s0": E1 / "dqn" / "seed0" / "final.pt",
}
NEURAL_DESCRIPTIONS = {
    "e1_ppo": "E1 PPO self-play entity Transformer (ID-free)",
    "e1_bc": "E1 behaviour cloning of the ARES prior + Greedy",
    "e1_bcppo": "E1 BC warm start then PPO",
    "e1_ids": "E1 PPO with species-ID embeddings (ablation)",
    "e1_dqn": "E1 DQN-MLP baseline",
}


def describe(name: str) -> str:
    if name in DESCRIPTIONS:
        return DESCRIPTIONS[name]
    for prefix, text in NEURAL_DESCRIPTIONS.items():
        if name.startswith(prefix):
            return text
    if name.startswith("exploiter"):
        return "main exploiter: PPO trained only to beat a frozen main agent"
    if name.startswith("main@"):
        return "frozen past snapshot of the league main agent"
    return name


def rival_spec(name: str, weights_cache: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Spec of a catalogue rival (archetypes, ``lookahead`` and the E1 neural personalities)."""
    if name in ARCHETYPES:
        return {"archetype": name, "name": name}
    if name == "lookahead":
        return {"lookahead": LOOKAHEAD_ROLLOUTS, "name": name}
    if name in NEURAL:
        if weights_cache is not None and name in weights_cache:
            return weights_cache[name]
        from ares_u.battle.ppo import load_weights

        method, weights, _ = load_weights(NEURAL[name])
        spec = {"method": method, "weights": weights, "greedy": True, "name": name}
        if weights_cache is not None:
            weights_cache[name] = spec
        return spec
    raise ValueError(f"unknown rival {name}")


def neural_spec(name: str, weights: Any, method: str = "ppo", greedy: bool = True) -> dict[str, Any]:
    return {"method": method, "weights": weights, "greedy": greedy, "name": name}


def build_controller(spec: dict[str, Any]) -> Controller:
    """Worker-side controller of a spec (no caching; see ``ares_u.league.tasks.controller``)."""
    if "archetype" in spec:
        return ArchetypeController(str(spec["archetype"]))
    if "lookahead" in spec:
        return LookaheadController(int(spec["lookahead"]))
    if "heuristic" in spec:
        return HeuristicController(str(spec["heuristic"]))
    return make_neural(str(spec["method"]), spec["weights"], bool(spec.get("greedy", True)),
                       str(spec.get("name", "neural")))


def strip_weights(spec: dict[str, Any]) -> dict[str, Any]:
    """The spec without its weights (for logs and reports)."""
    return {key: value for key, value in spec.items() if key != "weights"}
