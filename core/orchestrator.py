"""Single-VBlank owner integrating observed state, planning and physical input.

Native progress is always read from the cartridge.  This controller never writes
party, bag, HP, coordinates or campaign flags.  Synthetic laboratories expose
their scenario definitions separately; their results are not native certificates.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import bisect
import json
import math
from pathlib import Path
import random
import time
from typing import Any, cast

import torch
from torch import Tensor
from torch.optim.adam import Adam

from core.battle.mamba_entity_net import AresBattlePolicyNetwork, SSMCache
from core.battle.menu_cursor_controller import BattleMenuCursorController, CommandKind
from core.bridge.ffi_mgba import GBABridgeProtocol, NativeLibMGBABridge
from core.bridge.gba_structs import BattlePokemonStruct
from core.bridge.synthetic_gba_silicon import SyntheticGBASiliconBridge
from core.causal.capture_and_training import (
    BallInventory, CaptureContext, EpistemicCaptureOptimizer,
    MetacognitiveTrainingCurriculum, TrainingRoute,
)
from core.causal.epistemic_scm import (
    BeliefStateEstimator, DamageObservation, EpistemicCausalController,
    HiddenOpponentSet, PLAYABLE_TYPES,
)
from core.game_data import Game, MoveInfo, ROOT, RomCatalog, load_symbols
from core.field_controls import FieldMenuController
from core.native_bootstrap import NativeUIController
from core.navigation.constrained_astar import GbaConstrainedAStar
from core.navigation.movement_fsm import MovementMode, StateFeedbackMovementController
from core.navigation.warp_topology import HierarchicalWarpGraph, WarpTransition
from core.observations import OpponentKnowledge, encode_battle, encode_world_state
from core.planner.gflownet_team import TeamCandidate, TeamSynthesisGFlowNet
from core.planner.htn_planner import HTNCampaignPlanner
from core.safety.z3_integer_verifier import (
    DamageContext, DamageResult, GbaIntegerSafetyVerifier, ResidualContext,
    compute_residual_damage,
)
from core.world_model.hjepa import LatentHJEPAWorldModel

BALL_ITEMS = {"MASTER_BALL": 1, "ULTRA_BALL": 2, "GREAT_BALL": 3,
              "POKE_BALL": 4, "SAFARI_BALL": 5, "NET_BALL": 6,
              "DIVE_BALL": 7, "NEST_BALL": 8, "REPEAT_BALL": 9,
              "TIMER_BALL": 10, "LUXURY_BALL": 11, "PREMIER_BALL": 12}
_ACTION_ID = {16: 1, 32: 2, 64: 3, 128: 4}


@dataclass
class RuntimeMetrics:
    """Counters distinguish proofs, estimates, physical outcomes and model uses."""
    frames: int = 0
    moves: int = 0
    blocked_steps: int = 0
    map_transitions: int = 0
    battles: int = 0
    wins: int = 0
    losses: int = 0
    captures: int = 0
    faints: int = 0
    proved_decisions: int = 0
    unproven_decisions: int = 0
    causal_observations: int = 0
    belief_updates: int = 0
    neural_decisions: int = 0
    neural_plans: int = 0
    gflownet_samples: int = 0
    world_model_updates: int = 0
    checkpoints: int = 0
    completed_milestones: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BattleDecision:
    """Input command and the explicitly conditional safety assessment behind it."""
    action: str
    index: int
    item: int
    estimated_risk: float
    proof: DamageResult
    reason: str
    target: int = 0


@dataclass(frozen=True)
class SyntheticThreatEnvelope:
    """Declared laboratory bounds, distinct from hidden runtime opponent reads."""
    maximum_level: int
    maximum_attack: int
    maximum_special_attack: int
    maximum_move_power: int

    def __post_init__(self) -> None:
        if not 1 <= self.maximum_level <= 100 or min(self.maximum_attack, self.maximum_special_attack) < 1 or self.maximum_move_power < 0:
            raise ValueError("Invalid laboratory damage envelope")


class AresAutonomousOrchestrator:
    """Observe first, choose keys, then advance exactly one hardware frame.

    The PDF's (library, ROM, is_emerald) construction remains supported.  Tests
    and independent laboratory scenarios may instead supply a bridge directly.
    The persistent nine-action model uses four moves and up to five bench slots;
    capture is selected by its inventory-aware arithmetic controller.
    """

    def __init__(self, libmgba_so: str | Path | GBABridgeProtocol | None = None,
                 rom_file: str | Path | None = None, is_emerald: bool = False, *,
                 bridge: GBABridgeProtocol | None = None,
                 planner: HTNCampaignPlanner | None = None,
                 topology: HierarchicalWarpGraph | None = None,
                 catalog: RomCatalog | None = None,
                 checkpoint_dir: str | Path | None = None,
                 neural_checkpoint: str | Path | None = None,
                 threat_envelopes: Mapping[int, SyntheticThreatEnvelope] | None = None,
                 seed: int = 42, online_learning: bool = True) -> None:
        if bridge is not None and libmgba_so is not None:
            raise ValueError("Provide one bridge or one library/ROM pair")
        if bridge is None:
            if isinstance(libmgba_so, (str, Path)):
                if rom_file is None:
                    raise ValueError("Native construction requires a ROM path")
                bridge = NativeLibMGBABridge(libmgba_so, rom_file, is_emerald)
            elif libmgba_so is not None:
                bridge = libmgba_so
            else:
                raise ValueError("A bridge or library/ROM pair is required")
        self.bridge = bridge
        self.game = cast(Game, bridge.memory_map.game)
        self.native = isinstance(bridge, NativeLibMGBABridge)
        if self.native and threat_envelopes:
            raise ValueError("Laboratory threat envelopes cannot certify native opponents")
        self.threat_envelopes = dict(threat_envelopes or {})
        self.catalog = catalog or (RomCatalog(self.game, Path(rom_file) if rom_file else None)
                                   if self.native else None)
        self.planner = planner or HTNCampaignPlanner(is_emerald=self.game == "emerald")
        self.topology = topology or (HierarchicalWarpGraph.from_source(self.game)
                                    if self.native else HierarchicalWarpGraph())
        self.movement = StateFeedbackMovementController(bridge)
        self.menu = BattleMenuCursorController(bridge)
        self.causal = EpistemicCausalController()
        self.capturer = EpistemicCaptureOptimizer()
        self.curriculum = MetacognitiveTrainingCurriculum()
        self.verifier = GbaIntegerSafetyVerifier(timeout_ms=100)
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.world_model = LatentHJEPAWorldModel().eval()
        self.gflownet = TeamSynthesisGFlowNet(411).eval()
        self.battle_policy = AresBattlePolicyNetwork().eval()
        self.world_optimizer = Adam(self.world_model.parameters(), lr=1e-4)
        self.cache: SSMCache = self.battle_policy.initial_state(1)
        self.knowledge = OpponentKnowledge()
        self.beliefs: dict[int, BeliefStateEstimator] = {}
        self.owned_species: set[int] = set()
        self.observed_species: set[int] = set()
        self.desired_species: set[int] = set()
        self.desired_moves: dict[int, tuple[int, ...]] = {}
        self.metrics = RuntimeMetrics()
        self.audit: list[dict[str, object]] = []
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._last_checkpoint_time = time.monotonic()
        self._last_position: tuple[int, int, int, int] | None = None
        self._last_party_hp: tuple[int, ...] = ()
        self._in_battle = False
        self._synthetic_event_index = 0
        self._last_enemy_move = 0
        self._battle_decision: BattleDecision | None = None
        self._last_battle_signature: tuple[int, ...] | None = None
        self._last_native_callback = ""
        self._active_transition: WarpTransition | None = None
        self._last_directive: dict[str, Any] = {}
        self._last_milestone = ""
        self._last_world_state: Tensor | None = None
        self._last_action_id = 0
        self._transitions: list[tuple[Tensor, int, Tensor]] = []
        self.online_learning = online_learning
        self._ui = NativeUIController(bridge, self.game) if self.native else None
        self.field = FieldMenuController(bridge, self.game, self.catalog) if self.native else None
        self._bootstrapped = not self.native
        self._symbols = load_symbols(self.game) if self.native else {}
        self._team_signature: tuple[int, ...] = ()
        self._last_capture_turn = -1
        self._battle_turns = 0
        self._last_progress_frame = bridge.frame_count
        self._blocked_at: dict[tuple[int, int, int, int, int, int], int] = {}
        self._snapshot_failures = 0
        self._directive_signature: tuple[object, ...] = ()
        self._directive_cache: dict[str, Any] = {}
        self._native_enemy_hp: dict[int, float] = {}
        self._native_own_pp: dict[int, tuple[int, ...]] = {}
        self._native_interventions: dict[int, tuple[int, int]] = {}
        self._battle_scripts = sorted((symbol.address, name) for name, symbol in self._symbols.items()
                                      if name.startswith("BattleScript_") and 0x08000000 <= symbol.address < 0x09000000)
        checkpoint = Path(neural_checkpoint) if neural_checkpoint else ROOT / "artifacts/checkpoints/neural.pt"
        if checkpoint.is_file():
            self.load_neural_checkpoint(checkpoint)

    def _record(self, kind: str, **values: object) -> None:
        event: dict[str, object] = {"kind": kind, "frame": self.bridge.frame_count,
                                    "game": self.game, "backend": "native" if self.native else "synthetic"}
        event.update(values)
        self.audit.append(event)
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            with (self.checkpoint_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")

    def load_neural_checkpoint(self, path: str | Path) -> None:
        """Load locally created, dimension-checked training weights before inference."""
        data = cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))
        metadata = data.get("metadata", {})
        if metadata.get("gflownet_species", 411) != 411:
            raise ValueError("GFlowNet checkpoint must contain411 internal species")
        self.world_model.load_state_dict(data["world_model"])
        self.gflownet.load_state_dict(data["gflownet"])
        self.battle_policy.load_state_dict(data["battle_policy"])
        self.cache = self.battle_policy.initial_state(1)
        self._record("neural_weights_loaded", path=str(path))

    def _move(self, move_id: int) -> MoveInfo:
        if self.catalog is not None:
            return self.catalog.move(move_id)
        if isinstance(self.bridge, SyntheticGBASiliconBridge) and move_id in self.bridge.moves:
            move = self.bridge.moves[move_id]
            return MoveInfo(move.move_id, f"synthetic_{move.move_id}", 0, move.power,
                            move.move_type, move.accuracy, move.max_pp, 0, 0, 0, 0)
        raise ValueError(f"No observed move metadata for{move_id}")

    def _species_types(self, species: int) -> tuple[int, int]:
        if self.catalog is not None:
            return self.catalog.species(species).types
        return {1: (12, 3), 16: (0, 2), 19: (0, 0), 74: (5, 4),
                280: (10, 10), 283: (11, 11)}.get(species, (0, 0))

    def _legal_moves(self, species: int, level: int = 100) -> tuple[int, ...]:
        if self.catalog is not None:
            return tuple(self.catalog.level_up_moves(species, level)) or (33,)
        if isinstance(self.bridge, SyntheticGBASiliconBridge):
            return tuple(self.bridge.moves)
        raise ValueError("Move legality requires game metadata")

    def _refresh_roster(self, party: list[dict[str, Any]]) -> None:
        self.owned_species = {int(mon["species"]) for mon in party if int(mon["species"]) > 0}
        signature = tuple(sorted(self.owned_species | self.observed_species))
        if not signature or signature == self._team_signature:
            return
        available = torch.zeros(411, dtype=torch.bool)
        for species in signature:
            if 1 <= species <= 411:
                available[species - 1] = True
        with torch.no_grad():
            state, _ = self.gflownet.sample_team_trajectory(min(6, int(available.sum())), available_mask=available)
            selected = {int(index) + 1 for index in state.nonzero().flatten().tolist()}
            candidates = tuple(TeamCandidate(species, self._species_types(species), self._legal_moves(species),
                                  tuple(float(x) for x in self.causal.scm_matrix[self._species_types(species)[0]]))
                               for species in sorted(selected))
            reward = self.gflownet.coverage_reward(candidates)
            # Learned legal move heads choose intended future learnsets; no party edit occurs.
            self.desired_moves.clear()
            for candidate in candidates:
                own = torch.zeros(411)
                own[candidate.species_id - 1] = 1
                chosen: list[int] = []
                move_state = torch.zeros(355)
                for _ in range(min(4, len(candidate.legal_moves))):
                    logits = cast(Tensor, self.gflownet.move_policy_net(torch.cat((own, move_state))))
                    allowed = torch.zeros(355, dtype=torch.bool)
                    allowed[list(candidate.legal_moves)] = True
                    logits = logits.masked_fill(~allowed | move_state.bool(), -torch.inf)
                    index = int(torch.multinomial(logits.softmax(-1), 1))
                    move_state[index] = 1
                    chosen.append(index)
                self.desired_moves[candidate.species_id] = tuple(chosen)
        self.desired_species, self._team_signature = selected, signature
        self.metrics.gflownet_samples += 1
        self._record("team_plan", species=sorted(selected), moves=self.desired_moves, coverage_reward=reward)

    def _ensure_belief(self, enemy: BattlePokemonStruct) -> BeliefStateEstimator:
        species = int(enemy.species)
        if species not in self.beliefs:
            level = max(1, int(enemy.level))
            legal = self._legal_moves(species, level)
            bases = self.catalog.species(species).base_stats if self.catalog else (45, 45, 45, 45, 45, 45)
            hypotheses: list[HiddenOpponentSet] = []
            for quality, iv in (("low", 0), ("median", 15), ("high", 31)):
                stats = [max(1, (2 * base + iv) * level // 100 + 5) for base in bases[1:]]
                hypotheses.append(HiddenOpponentSet(quality, species, legal,
                    attack=stats[0], defense=stats[1], speed=stats[2],
                    special_attack=stats[3], special_defense=stats[4]))
            self.beliefs[species] = BeliefStateEstimator(hypotheses)
        return self.beliefs[species]

    def _estimated_enemy_hp(self, enemy: BattlePokemonStruct) -> tuple[int, int]:
        """Use visible HP fraction and a species/level prior for hidden numeric HP."""
        if not self.native:
            return int(enemy.hp), max(1, int(enemy.maxHP))
        level = max(1, int(enemy.level))
        base = self.catalog.species(int(enemy.species)).base_stats[0] if self.catalog else 50
        maximum = 1 if int(enemy.species) == 303 else (2 * base + 15) * level // 100 + level + 10
        return max(0, math.ceil(int(enemy.hp) / max(1, int(enemy.maxHP)) * maximum)), maximum

    def _incoming_assessment(self, defender: BattlePokemonStruct,
                             enemy: BattlePokemonStruct) -> tuple[DamageResult, float]:
        """Use no hidden enemy moves, stats, ability or item in a decision.

        Synthetic mechanics are fully declared and bounded.  Native movesets and
        exceptional effects are uncertain, so an arithmetic envelope remains an
        estimate with an unsupported proof status, even when ordinary damage fits.
        """
        belief = self._ensure_belief(enemy)
        upper_stat = max(1, (((2 * 255 + 31 + 63) * max(1, int(enemy.level)) // 100) + 5) * 110 // 100)
        largest_power = (max((move.power for move in self.bridge.moves.values()), default=250)
                         if isinstance(self.bridge, SyntheticGBASiliconBridge) else 250)
        envelope = self.threat_envelopes.get(int(enemy.species))
        if envelope is not None:
            upper_stat = max(envelope.maximum_attack, envelope.maximum_special_attack)
            largest_power = envelope.maximum_move_power
        context = DamageContext(level=envelope.maximum_level if envelope else max(1, int(enemy.level)), attack=upper_stat,
            defense=max(1, min(int(defender.defense), int(defender.spDefense))),
            power=largest_power, move_type=0, type_modifiers=((2, 1), (2, 1)), stab=True,
            defender_ability=int(defender.ability), defender_hp=int(defender.hp),
            unsupported_effects=(("unobserved native moves and exceptional effects",) if self.native else
                                 (() if envelope else ("no declared synthetic threat envelope",))))
        residual = compute_residual_damage(ResidualContext(max(1, int(defender.maxHP)),
                                                          status=int(defender.status1)))
        proof = self.verifier.verify_context_survival(int(defender.hp), context,
                                                     residual_damage=residual)
        revealed = self.knowledge.revealed_moves.get(int(enemy.species), [])
        powers = [self._move(move).power for move in revealed]
        estimated_power = max(powers, default=50)
        estimated_attack = max(belief.expected_stat("attack"), belief.expected_stat("special_attack"))
        estimate = DamageContext(max(1, int(enemy.level)), max(1, int(estimated_attack)),
                                 context.defense, estimated_power, stab=True)
        expected = self.verifier.compute_context_damage_reference(estimate, 93) + residual
        return proof, min(1.0, expected / max(1, int(defender.hp)))

    def choose_battle_decision(self) -> BattleDecision:
        """Combine recurrent logits, causal EFE and a conditional survival filter."""
        party = self.bridge.get_decrypted_player_party()
        features = encode_battle(self.bridge, self.knowledge, catalog=self.catalog)
        me = self.bridge.get_battle_pokemon(features.acting_battler)
        target_index = next((index for index in (0, 1) if bool(features.target_mask[0, index])), 0)
        enemy_slot = features.target_battlers[target_index]
        enemy = self.bridge.get_battle_pokemon(enemy_slot if enemy_slot >= 0 else 1)
        self.observed_species.add(int(enemy.species))
        self._refresh_roster(party)
        belief = self._ensure_belief(enemy)
        proof, retaliation = self._incoming_assessment(me, enemy)
        if not features.action_mask.any():
            # Exhausted move PP forces Struggle; physical A confirms it in the engine.
            features.action_mask[0, 0] = True
        with torch.no_grad():
            logits, targets, _, cache = self.battle_policy.forward_turn(features.entities, features.roles,
                features.action_mask, self.cache, features.target_mask, ~features.padding_mask)
            self.cache = cache.detach()
        self.metrics.neural_decisions += 1
        hp_ratio = int(me.hp) / max(1, int(me.maxHP))
        opponents = tuple(dict.fromkeys((int(enemy.type1), int(enemy.type2))))
        enemy_hp, enemy_max_hp = self._estimated_enemy_hp(enemy)
        options: list[tuple[float, float, BattleDecision]] = []
        for slot in range(4):
            if not me.moves[slot] or not me.pp[slot]:
                continue
            move = self._move(int(me.moves[slot]))
            efe, _, _ = self.causal.evaluate_free_energy(move.move_type, list(opponents), move.power, hp_ratio)
            attack = me.spAttack if move.category == "special" else me.attack
            defense = belief.expected_stat("special_defense" if move.category == "special" else "defense")
            ctx = DamageContext(max(1, int(me.level)), max(1, int(attack)), max(1, int(defense)),
                                move.power, move_type=move.move_type,
                                stab=move.move_type in (int(me.type1), int(me.type2)),
                                category="special" if move.category == "special" else "physical")
            neutral = self.verifier.compute_context_damage_reference(ctx, 93)
            distribution = self.causal.efficacy_distribution(move.move_type, opponents)
            expected_damage = neutral * sum(mult * prob for mult, prob in distribution.items())
            knockout_probability = sum(prob for mult, prob in distribution.items()
                                      if neutral * mult >= enemy_hp) * (move.accuracy or 100) / 100
            risk = retaliation * (1 - knockout_probability)
            # Protection is applied before scores; estimated fallback prioritizes faint risk.
            score = efe - expected_damage / max(1, enemy_max_hp) * 4 - float(logits[0, slot]) * 0.05
            options.append((risk, score - float(targets[0, target_index]) * 0.02,
                BattleDecision("move", slot, 0, risk, proof,
                    "recurrent policy, causal EFE and expected KO/retaliation", target_index)))
        for bench_action, party_index in enumerate(features.switch_party_indices, start=4):
            if not bool(features.action_mask[0, bench_action]):
                continue
            mon = party[party_index]
            replacement = BattlePokemonStruct()
            replacement.species, replacement.level = int(mon["species"]), int(mon["level"])
            replacement.hp, replacement.maxHP = int(mon["hp"]), int(mon["maxHP"])
            replacement.attack, replacement.defense = int(mon["attack"]), int(mon["defense"])
            replacement.spAttack, replacement.spDefense = int(mon["spAttack"]), int(mon["spDefense"])
            replacement.type1, replacement.type2 = self._species_types(int(mon["species"]))
            replacement.status1 = int(mon["status"])
            replacement.item = int(mon["held_item"])
            if self.catalog is not None:
                replacement.ability = self.catalog.species(int(mon["species"])).abilities[int(mon["ability_num"])]
            switch_proof, switch_risk = self._incoming_assessment(replacement, enemy)
            # A switch consumes this turn; prefer attacking when retaliation is
            # comparable. Dead/exhausted actives can still select a healthy bench.
            score = 1.0 - float(logits[0, bench_action]) * 0.05
            options.append((switch_risk, score, BattleDecision("switch", party_index, 0,
                switch_risk, switch_proof, "healthy legal bench and estimated switch-turn retaliation")))
        _, trainer = self.bridge.is_in_battle()
        if not trainer and int(enemy.species) in self.desired_species and int(enemy.species) not in self.owned_species:
            inventory = BallInventory({name: self.bridge.get_inventory().get(item, 0)
                                       for name, item in BALL_ITEMS.items()})
            context = CaptureContext(int(enemy.species), enemy_max_hp, enemy_hp,
                opponents, level=max(1, int(enemy.level)), status=int(enemy.status1),
                turn_count=self._battle_turns,
                catch_rate=self.catalog.species(int(enemy.species)).catch_rate if self.catalog else None,
                failure_faint_probability=retaliation)
            coverage = set(PLAYABLE_TYPES) - {t for species in self.owned_species for t in self._species_types(species)}
            capture = self.capturer.plan_capture(context, inventory, self.owned_species, coverage)
            if capture.ball_choice is not None and capture.action.startswith("THROW_"):
                ball = capture.ball_choice
                decision = BattleDecision("ball", 0, BALL_ITEMS[ball.ball], capture.estimated_faint_probability,
                                          proof, capture.reason)
                # Safe new coverage is a strategic objective, before battle damage ranking.
                if proof.formally_safe or capture.estimated_faint_probability < min((x[0] for x in options), default=1):
                    return self._audit_decision(decision)
        if not options:
            decision = BattleDecision("move", 0, 0, retaliation, proof, "engine-forced Struggle after PP exhaustion")
        else:
            safe = [option for option in options if option[2].proof.formally_safe]
            decision = min(safe, key=lambda option: option[1])[2] if safe else min(options, key=lambda option: (option[0], option[1]))[2]
        return self._audit_decision(decision)

    def _audit_decision(self, decision: BattleDecision) -> BattleDecision:
        if decision.proof.formally_safe:
            self.metrics.proved_decisions += 1
        else:
            self.metrics.unproven_decisions += 1
        self._record("battle_decision", decision=asdict(decision),
                     selection="proved" if decision.proof.formally_safe else "minimum_estimated_risk")
        self._battle_turns += 1
        return decision

    def _process_disclosures(self) -> None:
        """Consume engine-visible outcomes; no hidden moveset array enters beliefs."""
        if isinstance(self.bridge, SyntheticGBASiliconBridge):
            events = self.bridge.events[self._synthetic_event_index:]
            self._synthetic_event_index = len(self.bridge.events)
            for event in events:
                kind = event["kind"]
                if kind == "faint":
                    self.metrics.faints += 1
                    self._record("faint", party_index=event["party_index"])
                elif kind == "battle_end":
                    result = event["result"]
                    self.metrics.wins += result == "won"
                    self.metrics.losses += result == "lost"
                    self.metrics.captures += result == "caught"
                    self.curriculum.record_outcome("current_map", result == "lost")
                    self._record("battle_result", result=result)
                elif kind == "damage":
                    attacker = int(event["attacker"])
                    me, enemy = self.bridge.get_battle_pokemon(0), self.bridge.get_battle_pokemon(1)
                    move = self._move(int(event["move"]))
                    if attacker == 1:
                        self.knowledge.reveal_move(int(enemy.species), move.move_id)
                        self._ensure_belief(enemy).observe_move(move.move_id)
                        self.metrics.belief_updates += 1
                    elif event["text"] != "miss":
                        belief = self._ensure_belief(enemy)
                        defense = belief.expected_stat("special_defense" if move.category == "special" else "defense")
                        context = DamageContext(max(1, int(me.level)),
                            max(1, int(me.spAttack if move.category == "special" else me.attack)),
                            max(1, int(defense)), move.power, move_type=move.move_type,
                            stab=move.move_type in (int(me.type1), int(me.type2)),
                            category="special" if move.category == "special" else "physical")
                        neutral = self.verifier.compute_context_damage_reference(context, 93)
                        self.causal.observe_intervention(DamageObservation(move.move_type,
                            tuple(dict.fromkeys((int(enemy.type1), int(enemy.type2)))), float(event["damage"]),
                            neutral, engine_text_flag=str(event["text"]), is_status_move=move.power == 0,
                            defender_hp_before=int(enemy.hp) + int(event["damage"]), relative_noise=0.5))
                        self.metrics.causal_observations += 1
            return
        if self.bridge.is_in_battle()[0]:
            self._process_native_battle_disclosures()
        party = self.bridge.get_decrypted_player_party()
        hp = tuple(int(mon["hp"]) for mon in party)
        for index, (before, after) in enumerate(zip(self._last_party_hp, hp)):
            if before > 0 and after == 0:
                self.metrics.faints += 1
                self._record("faint", party_index=index)
        self._last_party_hp = hp

    def _process_native_battle_disclosures(self) -> None:
        """Learn from executed moves and direct HP-bar changes; exclude residuals."""
        count = self.bridge.read_u8(self._symbols["gBattlersCount"].address)
        if count not in (2, 4):
            return
        positions = self.bridge.read_memory(self._symbols["gBattlerPositions"].address, count)
        attacker = self.bridge.read_u8(self._symbols["gBattlerAttacker"].address)
        target = self.bridge.read_u8(self._symbols["gBattlerTarget"].address)
        for slot, position in enumerate(positions):
            mon = self.bridge.get_battle_pokemon(slot)
            if not mon.species or not mon.level or not mon.maxHP:
                continue
            if position & 1:
                move_id = self.bridge.read_u16(self._symbols["gLastMoves"].address + 2 * slot)
                if 0 < move_id <= 354 and move_id not in self.knowledge.revealed_moves.get(int(mon.species), []):
                    self.knowledge.reveal_move(int(mon.species), move_id)
                    self._ensure_belief(mon).observe_move(move_id)
                    self.metrics.belief_updates += 1
                    self._record("opponent_move_revealed", species=int(mon.species), move=move_id)
                fraction = int(mon.hp) / int(mon.maxHP)
                previous = self._native_enemy_hp.get(slot, fraction)
                intervention = self._native_interventions.get(attacker)
                if target == slot and intervention is not None and fraction < previous and not positions[attacker] & 1:
                    move_id, started = intervention
                    pointer = self.bridge.read_u32(self._symbols["gBattlescriptCurrInstr"].address)
                    index = bisect.bisect_right(self._battle_scripts, (pointer, "\uffff")) - 1
                    script = self._battle_scripts[index][1] if index >= 0 else ""
                    residual = any(word in script for word in ("Poison", "Burn", "Sandstorm", "Hail", "Leech", "Wrap", "Curse", "Recoil"))
                    result_flags = self.bridge.read_u8(self._symbols["gMoveResultFlags"].address)
                    move = self._move(move_id)
                    if not residual and not result_flags & 0x29 and move.power and self.bridge.frame_count - started < 1800:
                        own = self.bridge.get_battle_pokemon(attacker)
                        belief = self._ensure_belief(mon)
                        context = DamageContext(max(1, int(own.level)),
                            max(1, int(own.spAttack if move.category == "special" else own.attack)),
                            max(1, int(belief.expected_stat("special_defense" if move.category == "special" else "defense"))),
                            move.power, move_type=move.move_type, category="special" if move.category == "special" else "physical",
                            stab=move.move_type in (int(own.type1), int(own.type2)))
                        neutral = self.verifier.compute_context_damage_reference(context, 93)
                        _, maximum = self._estimated_enemy_hp(mon)
                        text = "super_effective" if result_flags & 2 else ("not_very_effective" if result_flags & 4 else "neutral")
                        self.causal.observe_intervention(DamageObservation(move.move_type,
                            tuple(dict.fromkeys((int(mon.type1), int(mon.type2)))),
                            (previous - fraction) * maximum, neutral, engine_text_flag=text,
                            defender_hp_before=previous * maximum, relative_noise=0.5))
                        self.metrics.causal_observations += 1
                        self._record("direct_damage_observed", move=move_id, target=slot,
                                     hp_fraction_lost=previous - fraction, numeric_hp="species_level_estimate",
                                     script=script, effectiveness=text)
                        self._native_interventions.pop(attacker, None)
                self._native_enemy_hp[slot] = fraction
            else:
                pp = tuple(int(value) for value in mon.pp)
                old = self._native_own_pp.get(slot, pp)
                used = next((index for index in range(4) if pp[index] < old[index]), None)
                if used is not None and mon.moves[used]:
                    self._native_interventions[slot] = (int(mon.moves[used]), self.bridge.frame_count)
                self._native_own_pp[slot] = pp

    def _world_observation(self, action_id: int = 0) -> Tensor:
        state = encode_world_state(self.bridge)
        if self._last_world_state is not None:
            self._transitions.append((self._last_world_state, self._last_action_id, state))
            self._transitions = self._transitions[-256:]
        self._last_world_state, self._last_action_id = state, action_id
        if self.online_learning and len(self._transitions) >= 64 and len(self._transitions) % 32 == 0:
            rows = self._transitions[-64:]
            initial = torch.stack([row[0] for row in rows])
            actions = torch.tensor([row[1] for row in rows], dtype=torch.long)
            next_states = torch.stack([row[2] for row in rows])
            self.world_optimizer.zero_grad(set_to_none=True)
            loss, _ = self.world_model.forward_loss(initial, actions, next_states)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite observed-transition JEPA loss")
            cast(Callable[[], None], loss.backward)()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
            self.world_optimizer.step()
            self.world_model.update_target_ema()
            self.metrics.world_model_updates += 1
        return state

    def _plan_movement(self, path: list[tuple[int, int]], target: tuple[int, int]) -> MovementMode:
        """JEPA chooses between legal walk/run plans along the proven tile route."""
        coords = self.bridge.get_player_coords_and_map()
        origin = (coords["x"], coords["y"])
        directions: list[int] = []
        for destination in path[:8]:
            dx, dy = destination[0] - origin[0], destination[1] - origin[1]
            key = 16 if dx > 0 else (32 if dx < 0 else (128 if dy > 0 else 64))
            directions.append(_ACTION_ID[key])
            origin = destination
        current = self._world_observation()
        goal = current.clone()
        goal[0], goal[1] = target[0] / 256, target[1] / 256
        candidates = torch.tensor([[directions, [direction + 4 for direction in directions]]], dtype=torch.long)
        with torch.no_grad():
            selection = int(self.world_model.plan_actions(current.unsqueeze(0), candidates, goal.unsqueeze(0)))
        self._last_action_id = directions[0] + (4 if selection else 0)
        self.metrics.neural_plans += 1
        return "run" if selection else "walk"

    def _execute_overworld_cycle(self) -> None:
        event = self.movement.tick()
        if event is not None:
            self.metrics.moves += event.status == "arrived"
            self.metrics.blocked_steps += event.status == "blocked"
            self._record("movement", movement=asdict(event))
            if event.status == "blocked":
                for edge in self.movement.blocked_edges:
                    self._blocked_at.setdefault(edge, self.bridge.frame_count)
        if self.movement.pending:
            return
        self.bridge.set_hardware_keys(0)
        # Script locks and moving actors are temporary obstructions. Static
        # walls already come from live tiles; a stale input timeout is not a wall.
        for edge in tuple(self.movement.blocked_edges):
            if self.bridge.frame_count - self._blocked_at.get(edge, 0) >= 600:
                self.movement.blocked_edges.discard(edge)
                self._blocked_at.pop(edge, None)
        if self.field is not None:
            field_keys = self.field.poll_key_mask()
            if field_keys is not None:
                self.bridge.set_hardware_keys(field_keys)
                return
        if not self.bridge.is_hardware_ready():
            self.bridge.set_hardware_keys(self.bridge.KEY_A if self.bridge.frame_count % 12 == 0 else 0)
            return
        party = self.bridge.get_decrypted_player_party()
        self._refresh_roster(party)
        completed = self.planner.completed_milestones(self.bridge, party)
        current_coords = self.bridge.get_player_coords_and_map()
        signature: tuple[object, ...] = (
            tuple(current_coords.values()), tuple(sorted(completed)), self.metrics.losses,
            tuple((mon["pid"], mon["hp"], mon["maxHP"], mon["level"], mon["status"],
                   tuple(mon["moves"])) for mon in party))
        if signature != self._directive_signature:
            self._directive_cache = self.planner.get_active_directive(self.bridge, party)
            self._directive_signature = signature
        directive = dict(self._directive_cache)
        self._last_directive = directive
        if self.field is not None:
            self.field.menu_choice = int(directive.get("menu_choice", 0))
            if directive.get("mode") == "TEACH_FIELD_MOVE":
                move_ids = cast(list[int], directive["moves"])
                if move_ids and self.field.request_required_move(move_ids[0]):
                    self.bridge.set_hardware_keys(self.field.poll_key_mask() or 0)
                elif self.bridge.frame_count % 600 == 0:
                    self._record("campaign_blocked", reason="required field move has no compatible stocked HM", moves=move_ids)
                return
        if directive.get("mode") == "OPTIMAL_TRAINING_CURRICULUM":
            candidates_raw = cast(list[dict[str, Any]], directive.get("training_candidates", []))
            if candidates_raw:
                routes = [TrainingRoute(str(entry["route"]), float(entry["expected_experience"]),
                    float(entry["duration_seconds"]), float(entry["faint_probability"]),
                    travel_seconds=float(entry.get("travel_seconds", 0))) for entry in candidates_raw]
                ranked = self.curriculum.select_route(routes)
                entry = next(row for row in candidates_raw if row["route"] == ranked.route.name)
                directive["dest"] = (int(entry["map_group"]), int(entry["map_num"]), tuple(entry["coords"]))
                directive["training_route"] = ranked.route.name
                directive["training_score"] = ranked.risk_adjusted_score
        newly_completed = completed - set(self.metrics.completed_milestones)
        for task in self.planner.campaign_dag:
            if task.milestone_id in newly_completed:
                self.metrics.completed_milestones.append(task.milestone_id)
                self._record("milestone_observed", milestone=task.milestone_id)
        if newly_completed and self.checkpoint_dir is not None:
            self.save_checkpoint("milestone")
        milestone = str(directive.get("milestone", ""))
        if milestone != self._last_milestone:
            self._last_milestone = milestone
            self._last_progress_frame = self.bridge.frame_count
        if directive.get("mode") == "POSTGAME_COMPLETE":
            return
        coords = self.bridge.get_player_coords_and_map()
        current_map = (coords["map_group"], coords["map_num"])
        dest = directive.get("dest")
        if dest is None:
            self._record("campaign_blocked", reason=directive.get("reason", "directive has no destination"), directive=directive)
            return
        destination_map = (int(dest[0]), int(dest[1]))
        target = (int(dest[2][0]), int(dest[2][1]))
        transition: WarpTransition | None = None
        if current_map != destination_map:
            transition = self.topology.next_transition(current_map, destination_map,
                (coords["x"], coords["y"]), self.planner.completed_milestones(self.bridge, party))
            if transition is None:
                if self.bridge.frame_count % 600 == 0:
                    self._record("campaign_blocked", reason="no source-defined map route", directive=directive)
                return
            target = transition.source_xy
        self._active_transition = transition
        if self.field is not None and transition is not None and transition.kind == "script":
            self.field.menu_choice = transition.menu_choice
        position = (coords["x"], coords["y"])
        _, collisions, elevations = self.bridge.extract_live_map_matrices()
        astar = GbaConstrainedAStar(collisions, elevations, self.bridge.extract_behavior_map(), self.game,
                                  allow_surf=bool(directive.get("allow_surf", False)))
        objects = self.bridge.get_object_events()
        player_id = self.bridge.read_u8(self.bridge.memory_map.avatar + 5)
        astar.register_object_events(objects, player_id)
        astar.blocked_edges = {((x, y), (tx, ty)) for group, number, x, y, tx, ty in self.movement.blocked_edges
                               if (group, number) == current_map}
        path: list[tuple[int, int]] | None = None
        if self.native and transition is None and directive.get("mode") == "OPTIMAL_TRAINING_CURRICULUM":
            behaviors = self.bridge.extract_behavior_map()
            grass = [(x, y) for y in range(behaviors.shape[0]) for x in range(behaviors.shape[1])
                     if int(behaviors[y, x]) in (2, 3) and (x, y) != position]
            for point in sorted(grass, key=lambda xy: abs(xy[0] - position[0]) + abs(xy[1] - position[1])):
                candidate = astar.find_optimal_path((coords["x"], coords["y"], coords["z"]), point)
                if candidate:
                    path, target = candidate, point
                    break
        facing: Any = directive.get("face_direction")
        door_approach = False
        if transition is not None and transition.kind == "warp" and self.native:
            behaviors = self.bridge.extract_behavior_map()
            if int(behaviors[target[1], target[0]]) == 0x69:
                target = (target[0], target[1] + 1)
                door_approach = True
        if self.native and ((transition is None and (int(directive.get("npc_local_id", 0)) > 0 or
                                                    directive.get("action") == "interact")) or
                            (transition is not None and transition.kind == "script")):
            local_id = transition.npc_local_id if transition is not None else int(directive.get("npc_local_id", 0))
            npc = next((actor for index, actor in enumerate(objects)
                        if index != player_id and int(actor.flags) & 1 and
                        int(actor.localId) == local_id), None)
            if npc is not None:
                target = (int(npc.currentCoords_x) - 7, int(npc.currentCoords_y) - 7)
            candidates: list[tuple[list[tuple[int, int]], tuple[int, int], int]] = []
            behaviors = self.bridge.extract_behavior_map()
            for dx, dy, direction in ((0, 1, 2), (0, -1, 1), (1, 0, 3), (-1, 0, 4)):
                adjacent = (target[0] + dx, target[1] + dy)
                if (0 <= adjacent[1] < behaviors.shape[0] and 0 <= adjacent[0] < behaviors.shape[1] and
                        int(behaviors[adjacent[1], adjacent[0]]) == 0x80):
                    adjacent = (target[0] + 2 * dx, target[1] + 2 * dy)
                candidate = astar.find_optimal_path((coords["x"], coords["y"], coords["z"]), adjacent)
                if candidate is not None:
                    candidates.append((candidate, adjacent, direction))
            if candidates:
                path, target, facing = min(candidates, key=lambda entry: len(entry[0]))
        if position == target:
            if door_approach:
                self.bridge.set_hardware_keys(self.bridge.KEY_UP)
                return
            if transition is not None and transition.kind == "warp" and self.native:
                behavior = int(self.bridge.extract_behavior_map()[position[1], position[0]])
                warp_keys = {0x62: 16, 0x63: 32, 0x64: 64, 0x65: 128,
                             0x6C: 16, 0x6D: 32, 0x6E: 16, 0x6F: 32}
                if behavior in warp_keys:
                    self.bridge.set_hardware_keys(warp_keys[behavior])
                    return
            if transition is not None and transition.kind == "connection":
                direction = {"right": 16, "left": 32, "up": 64, "down": 128}[transition.direction]
                self.movement.request_step(direction)
                return
            if isinstance(facing, int) and facing in (1, 2, 3, 4):
                actor = objects[player_id]
                if int(actor.facingDirection) & 15 != facing:
                    self.bridge.set_hardware_keys({1: 128, 2: 64, 3: 32, 4: 16}[facing])
                    return
            self.bridge.set_hardware_keys(self.bridge.KEY_A if self.bridge.frame_count % 12 == 0 else 0)
            return
        if path is None:
            path = astar.find_optimal_path((coords["x"], coords["y"], coords["z"]), target)
        if not path:
            if self.bridge.frame_count % 600 == 0:
                self._record("campaign_blocked", reason="live collision path unavailable", directive=directive,
                             position=coords, target=list(target))
            return
        mode = self._plan_movement(path, target)
        self.movement.move_toward(path[0], mode)

    def _execute_combat_cycle(self, is_trainer_battle: bool) -> None:
        """Keep one decision through the cursor FSM; never advance nested frames."""
        del is_trainer_battle
        observed_menu = self.menu.observe_menu()
        mode = observed_menu.mode
        me = self.bridge.get_battle_pokemon(observed_menu.battler)
        enemy = self.bridge.get_battle_pokemon(1)
        if self.native and mode == "party" and self._battle_decision is None and not self.menu.busy:
            party = self.bridge.get_decrypted_player_party()
            replacement = next((index for index, mon in enumerate(party) if int(mon["hp"]) > 0), None)
            if replacement is not None:
                self.menu.request_switch(replacement)
                self.bridge.set_hardware_keys(self.menu.tick())
                return
        if self.native and self._battle_decision is None and mode not in {"action", "move"}:
            self.bridge.set_hardware_keys(self.menu.tick())
            return
        if not me.species or not enemy.species or not me.level or not enemy.level:
            self.bridge.set_hardware_keys(self.menu.tick())
            return
        signature = (int(me.hp), int(enemy.hp), *[int(pp) for pp in me.pp])
        if self._battle_decision is None:
            self._battle_decision = self.choose_battle_decision()
            self._last_battle_signature = signature
            decision = self._battle_decision
            self.menu.request_command(cast(CommandKind, decision.action), index=decision.index,
                                      item=decision.item or 4, target=decision.target)
        self.bridge.set_hardware_keys(self.menu.tick())
        if signature != self._last_battle_signature:
            self._battle_decision = None
            self.menu.cancel()
            self._last_battle_signature = signature

    def _select_frame_input(self) -> None:
        """Observe one stable snapshot and select keys without advancing frames."""
        if self._ui is not None and not self._bootstrapped:
            keys = self._ui.choose_keys()
            if self._ui.last_callback != self._last_native_callback:
                self._record("native_callback", callback=self._ui.last_callback)
                self._last_native_callback = self._ui.last_callback
            if keys is not None:
                self.bridge.set_hardware_keys(keys)
                return
            self._bootstrapped = True
        # Always dereference both relocatable pointers afresh before any RAM policy.
        self.bridge.get_save_block1_dynamic_ptr()
        self.bridge.get_save_block2_dynamic_ptr()
        self._process_disclosures()
        in_battle, trainer = self.bridge.is_in_battle()
        if in_battle and not self._in_battle:
            self.metrics.battles += 1
            self.cache = self.battle_policy.initial_state(1)
            self._battle_decision = None
            self._last_enemy_move = 0
            self._battle_turns = 0
            self._native_enemy_hp.clear()
            self._native_own_pp.clear()
            self._native_interventions.clear()
            self.movement.clear()
            self._record("battle_entered", trainer=trainer)
        if not in_battle and self._in_battle:
            self._battle_decision = None
            self.menu.cancel()
            self.cache = self.battle_policy.initial_state(1)
            if self.native:
                outcome = self.bridge.read_u8(self._symbols["gBattleOutcome"].address)
                self.metrics.wins += outcome == 1
                self.metrics.losses += outcome in (2, 9)
                self.metrics.captures += outcome == 7
                self._record("native_battle_result", outcome=outcome)
        self._in_battle = in_battle
        if in_battle:
            self._execute_combat_cycle(trainer)
        else:
            self._execute_overworld_cycle()
    def step_vblank_cycle(self) -> None:
        """Own one VBlank; retry transient decryption without accepting bad data.

        ARM code can straddle VBlank while GetMonData temporarily decrypts a
        party record. This snapshot cannot become a policy observation.
        Persistent checksum failure remains an error after120 neutral frames.
        """
        try:
            self._select_frame_input()
        except ValueError as error:
            if not self.native or str(error) != "Party contains an invalid checksum":
                raise
            self._snapshot_failures += 1
            if self._snapshot_failures > 120:
                raise ValueError("Native party checksum remained invalid for120 frames") from error
            self.bridge.set_hardware_keys(0)
            if self._snapshot_failures == 1:
                self._record("memory_snapshot_retry", reason="party checksum invalid at VBlank")
        else:
            self._snapshot_failures = 0
        self.bridge.step_frame()
        self.metrics.frames = self.bridge.frame_count
        if self._bootstrapped:
            coords = self.bridge.get_player_coords_and_map()
            position = (coords["map_group"], coords["map_num"], coords["x"], coords["y"])
            if self._last_position is not None and position[:2] != self._last_position[:2]:
                self.metrics.map_transitions += 1
                self._record("map_transition", source=list(self._last_position[:2]), destination=list(position[:2]))
            self._last_position = position
        if self.checkpoint_dir is not None and time.monotonic() - self._last_checkpoint_time >= 600:
            self.save_checkpoint("periodic")

    tick = step_vblank_cycle

    def run_frames(self, frames: int) -> dict[str, object]:
        if frames < 0:
            raise ValueError("Frame count cannot be negative")
        for _ in range(frames):
            self.step_vblank_cycle()
        return self.status()

    def run_until(self, predicate: Callable[[AresAutonomousOrchestrator], bool],
                  max_frames: int | None = None) -> dict[str, object]:
        """Continue until the observed condition, without an implicit training limit."""
        started = self.bridge.frame_count
        while not predicate(self):
            if max_frames is not None and self.bridge.frame_count - started >= max_frames:
                break
            self.step_vblank_cycle()
        return self.status()

    def status(self) -> dict[str, object]:
        """Report factual ROM/synthetic progress and unresolved controller directives."""
        result: dict[str, object] = {"game": self.game, "backend": "native" if self.native else "synthetic",
            "frames": self.bridge.frame_count, "metrics": asdict(self.metrics),
            "directive": self._last_directive, "native_campaign_certified": False}
        try:
            result["position"] = self.bridge.get_player_coords_and_map()
            result["party_species"] = sorted(self.owned_species)
            if self.native and self._bootstrapped:
                evidence = self.native_completion_evidence()
                result["native_completion_evidence"] = evidence
                result["native_campaign_certified"] = evidence["complete"]
        except ValueError:
            result["position"] = None
        return result

    def native_completion_evidence(self) -> dict[str, object]:
        """Require all eight badges, game-clear and encrypted HOF-entry count."""
        if not self.native:
            return {"complete": False, "reason": "Synthetic execution cannot certify a native campaign"}
        badge_base, game_clear, stat_offset = ((0x867, 0x864, 0x159C) if self.game == "emerald"
                                              else (0x820, 0x82C, 0x1200))
        badges = [self.bridge.get_event_flag(badge_base + index) for index in range(8)]
        clear = self.bridge.get_event_flag(game_clear)
        key = self.bridge.read_u32(self.bridge.get_save_block2_dynamic_ptr() + self.bridge.memory_map.encryption_key_offset)
        hall_entries = self.bridge.read_u32(self.bridge.get_save_block1_dynamic_ptr() + stat_offset + 10 * 4) ^ key
        return {"badges": badges, "game_clear": clear, "hall_of_fame_entries": hall_entries,
                "rom_sha1": self.bridge.memory_map.rom_sha1,
                "complete": all(badges) and clear and 0 < hall_entries < 100000}

    def save_checkpoint(self, reason: str = "manual") -> Path:
        """Publish immutable emulator/model/belief bundles with SHA256 integrity."""
        if self.checkpoint_dir is None:
            raise ValueError("A checkpoint directory is required")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        directory = self.checkpoint_dir / f"frame-{self.bridge.frame_count:012d}-{time.time_ns()}"
        directory.mkdir()
        emulator = self.bridge.save_state()
        (directory / "emulator.state").write_bytes(emulator)
        if isinstance(self.bridge, NativeLibMGBABridge):
            (directory / "battery.sav").write_bytes(self.bridge.save_battery())
        beliefs = {species: {"hypotheses": [asdict(h) for h in belief.hypotheses],
                              "probabilities": belief.probabilities, "evidence_count": belief.evidence_count}
                   for species, belief in self.beliefs.items()}
        state: dict[str, Any] = {
            "world_model": self.world_model.state_dict(), "gflownet": self.gflownet.state_dict(),
            "battle_policy": self.battle_policy.state_dict(), "world_optimizer": self.world_optimizer.state_dict(),
            "cache_ssm": self.cache.ssm_state, "cache_conv": self.cache.conv_state,
            "causal": self.causal.state_dict(), "beliefs": beliefs,
            "knowledge": asdict(self.knowledge), "metrics": asdict(self.metrics),
            "owned_species": self.owned_species, "observed_species": self.observed_species,
            "desired_species": self.desired_species, "desired_moves": self.desired_moves,
            "team_signature": self._team_signature, "in_battle": self._in_battle,
            "synthetic_event_index": self._synthetic_event_index,
            "last_enemy_move": self._last_enemy_move, "battle_turns": self._battle_turns,
            "last_milestone": self._last_milestone, "last_directive": self._last_directive,
            "blocked_edges": self.movement.blocked_edges, "rng": self.rng.getstate(),
            "torch_rng": torch.get_rng_state(), "last_world_state": self._last_world_state,
            "last_action_id": self._last_action_id, "transitions": self._transitions,
            "game": self.game, "native": self.native, "frame": self.bridge.frame_count,
            "bootstrapped": self._bootstrapped,
            "threat_envelopes": {species: asdict(envelope) for species, envelope in self.threat_envelopes.items()},
            "last_party_hp": self._last_party_hp,
            "field_controller": self.field.state_dict() if self.field is not None else None,
            "blocked_at": self._blocked_at,
            "movement_controller": self.movement.state_dict(),
            "battle_controller": self.menu.state_dict(),
            "battle_decision": self._battle_decision,
            "last_battle_signature": self._last_battle_signature,
            "last_position": self._last_position,
            "snapshot_failures": self._snapshot_failures,
            "native_enemy_hp": self._native_enemy_hp,
            "native_own_pp": self._native_own_pp,
            "native_interventions": self._native_interventions,
            "hardware_keys": self.bridge.hardware_keys,
        }
        torch.save(state, directory / "controller.pt")
        self.movement.save_obstacles(directory / "obstacles.json")
        manifest = {"game": self.game, "native": self.native, "frame": self.bridge.frame_count,
                    "reason": reason, "emulator_sha256": hashlib.sha256(emulator).hexdigest(),
                    "controller_sha256": hashlib.sha256((directory / "controller.pt").read_bytes()).hexdigest(),
                    "status": self.status()}
        if (directory / "battery.sav").is_file():
            manifest["battery_sha256"] = hashlib.sha256((directory / "battery.sav").read_bytes()).hexdigest()
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary = self.checkpoint_dir / "latest.tmp"
        temporary.write_text(json.dumps({"directory": directory.name}), encoding="utf-8")
        temporary.replace(self.checkpoint_dir / "latest.json")
        self.metrics.checkpoints += 1
        self._last_checkpoint_time = time.monotonic()
        self._record("checkpoint", path=str(directory), reason=reason)
        return directory

    def load_checkpoint(self, directory: str | Path | None = None) -> None:
        """Validate a complete checkpoint, then restore hardware and cognitive state."""
        if directory is None:
            if self.checkpoint_dir is None:
                raise ValueError("A checkpoint directory is required")
            pointer = json.loads((self.checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
            directory = self.checkpoint_dir / str(pointer["directory"])
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["game"] != self.game or bool(manifest["native"]) != self.native:
            raise ValueError("Checkpoint game/backend mismatch")
        emulator = (directory / "emulator.state").read_bytes()
        controller = (directory / "controller.pt").read_bytes()
        if hashlib.sha256(emulator).hexdigest() != manifest["emulator_sha256"] or hashlib.sha256(controller).hexdigest() != manifest["controller_sha256"]:
            raise ValueError("Checkpoint integrity check failed")
        if "battery_sha256" in manifest and hashlib.sha256((directory / "battery.sav").read_bytes()).hexdigest() != manifest["battery_sha256"]:
            raise ValueError("Checkpoint battery integrity check failed")
        state = cast(dict[str, Any], torch.load(directory / "controller.pt", map_location="cpu", weights_only=False))
        self.load_neural_checkpoint(directory / "controller.pt")
        self.world_optimizer.load_state_dict(state["world_optimizer"])
        self.causal.load_state_dict(state["causal"])
        self.beliefs.clear()
        for raw_species, entry in state["beliefs"].items():
            hypotheses = [HiddenOpponentSet(**row) for row in entry["hypotheses"]]
            belief = BeliefStateEstimator(hypotheses, [entry["probabilities"][h.identifier] for h in hypotheses])
            belief.evidence_count = entry["evidence_count"]
            self.beliefs[int(raw_species)] = belief
        self.bridge.load_state(emulator)
        if isinstance(self.bridge, NativeLibMGBABridge) and (directory / "battery.sav").is_file():
            self.bridge.load_battery((directory / "battery.sav").read_bytes())
        self.cache = SSMCache(state["cache_ssm"], state["cache_conv"])
        self.knowledge = OpponentKnowledge(**state["knowledge"])
        self.metrics = RuntimeMetrics(**state["metrics"])
        for attribute in ("owned_species", "observed_species", "desired_species", "desired_moves"):
            setattr(self, attribute, state[attribute])
        self._team_signature = state["team_signature"]
        self._in_battle = state["in_battle"]
        self._bootstrapped = state.get("bootstrapped", not self.native)
        self.threat_envelopes = {int(species): SyntheticThreatEnvelope(**values)
                                 for species, values in state.get("threat_envelopes", {}).items()}
        self._last_party_hp = state.get("last_party_hp", ())
        if self.field is not None and state.get("field_controller") is not None:
            self.field.load_state_dict(state["field_controller"])
        self._blocked_at = state.get("blocked_at", {})
        self._synthetic_event_index = state["synthetic_event_index"]
        self._last_enemy_move = state["last_enemy_move"]
        self._battle_turns = state["battle_turns"]
        self._last_milestone = state["last_milestone"]
        self._last_directive = state["last_directive"]
        self.movement.blocked_edges = state["blocked_edges"]
        self.rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"])
        self._last_world_state = state["last_world_state"]
        self._last_action_id = state["last_action_id"]
        self._transitions = state["transitions"]
        self._battle_decision = state.get("battle_decision")
        self._last_battle_signature = state.get("last_battle_signature")
        self._last_position = state.get("last_position")
        self._snapshot_failures = state.get("snapshot_failures", 0)
        self._native_enemy_hp = state.get("native_enemy_hp", {})
        self._native_own_pp = state.get("native_own_pp", {})
        self._native_interventions = state.get("native_interventions", {})
        if "movement_controller" in state:
            self.movement.load_state_dict(state["movement_controller"])
        else:
            self.movement.clear()
        if "battle_controller" in state:
            self.menu.load_state_dict(state["battle_controller"])
        else:
            self.menu.cancel()
        if "hardware_keys" in state:
            self.bridge.set_hardware_keys(int(state["hardware_keys"]))
        self._directive_signature = ()
        self._last_checkpoint_time = time.monotonic()
        self._record("checkpoint_restored", path=str(directory))

    def close(self) -> None:
        """Release held keys and the native handle exactly once."""
        self.bridge.set_hardware_keys(0)
        self.bridge.close()
