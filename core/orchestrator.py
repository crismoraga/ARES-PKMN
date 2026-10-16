"""Single-VBlank owner integrating observed state, planning and physical input.

Native progress is always read from the cartridge.  This controller never writes
party, bag, HP, coordinates or campaign flags.  Synthetic laboratories expose
their scenario definitions separately; their results are not native certificates.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import bisect
import json
import math
from pathlib import Path
import random
import shutil
import time
import traceback
from typing import Any, cast

import torch
from torch import Tensor
from torch.optim.adam import Adam

from core.battle.mamba_entity_net import AresBattlePolicyNetwork, SSMCache
from core.battle.menu_cursor_controller import BattleMenuCursorController, CommandKind
from core.battle.risk_horizon import MoveDamageEstimate, estimate_attack_plan, RISK_MODEL
from core.bridge.ffi_mgba import GBABridgeProtocol, NativeLibMGBABridge
from core.bridge.gba_structs import BattlePokemonStruct, ObjectEventStruct
from core.bridge.synthetic_gba_silicon import SyntheticGBASiliconBridge
from core.causal.capture_and_training import (
    BallInventory, CaptureContext, EpistemicCaptureOptimizer,
    MetacognitiveTrainingCurriculum, TrainingRoute,
)
from core.causal.epistemic_scm import (
    BeliefStateEstimator, DamageObservation, EpistemicCausalController,
    HiddenOpponentSet, PLAYABLE_TYPES,
)
from core.game_data import TYPE_NAMES, Game, MoveInfo, ROOT, RomCatalog, load_symbols
from core.field_controls import FieldMenuController
from core.native_bootstrap import NativeUIController
from core.navigation.constrained_astar import GbaConstrainedAStar
from core.navigation.movement_fsm import MovementMode, StateFeedbackMovementController
from core.navigation.obstacles import (
    FlagSnapshot, live_obstacle_kind, obstacle_model, runtime_template_objects, template_objects,
)
from core.navigation.puzzle_execution import FeedbackPuzzleExecutor
from core.navigation.pushback import LearnedBarrier, PushbackLearner
from core.navigation.script_gates import (
    RamGateWorld, completion_requirements, gate_at, gate_state, gates_by_map, load_floor_switches, unlocking_flags,
    unmet_requirements,
)
from core.navigation.trainer_sight import undefeated_sights
from core.navigation.warp_topology import HierarchicalWarpGraph, WarpTransition
from core.observations import OpponentKnowledge, encode_battle, encode_world_state, player_party_in_battle_order
from core.planner.gflownet_team import TeamCandidate, TeamSynthesisGFlowNet
from core.planner.htn_planner import HTNCampaignPlanner
from core.planner.readiness import ENGAGE_HP_FRACTION, ENGAGE_PP_FRACTION, best_attack_pp
from core.progress_watchdog import (
    CampaignStalled, ProgressContext, ProgressObservation, ProgressWatchdog, RecoveryRequest,
)
from core.shopping import HEALING_ITEMS, MartController, ShoppingNeed, ball_stock, healing_stock, item_ids
from core.strategy_knowledge import (
    NEUTRAL, NO_EFFECT, RESISTED, SUPER, UNTYPED_MOVES, OpponentMemory, SharedKnowledge, counter_species,
    result_category,
)
from core.safety.z3_integer_verifier import (
    DamageContext, DamageResult, GbaIntegerSafetyVerifier, ResidualContext,
    STAGE_RATIOS, compute_residual_damage,
)
from core.telemetry import describe_battle, describe_mon, move_name, species_name, trainer_card
from core.world_model.hjepa import LatentHJEPAWorldModel

BALL_ITEMS = {"MASTER_BALL": 1, "ULTRA_BALL": 2, "GREAT_BALL": 3,
              "POKE_BALL": 4, "SAFARI_BALL": 5, "NET_BALL": 6,
              "DIVE_BALL": 7, "NEST_BALL": 8, "REPEAT_BALL": 9,
              "TIMER_BALL": 10, "LUXURY_BALL": 11, "PREMIER_BALL": 12}
_ACTION_ID = {16: 1, 32: 2, 64: 3, 128: 4}
# Generation-three EFFECT_EXPLOSION: the user faints, so it is never a damage plan.
_SELF_KO_EFFECT = 7
# Checkpoint reasons kept by retention even when they are old.
_DURABLE_CHECKPOINTS = frozenset({"milestone", "worker_start", "complete"})
_ROUTE_COMMITMENT_FRAMES = 1800
_TABU_FRAMES = 2400


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
    recoveries: int = 0
    faults: int = 0
    rewinds: int = 0
    probes: int = 0
    detours: int = 0
    heals: int = 0
    evolutions: int = 0
    level_ups: int = 0
    field_moves: int = 0
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
    risk_model: str = "conditional_one_turn_retaliation_estimate"
    risk_proxy: float | None = None
    expected_turns_to_ko: int | None = None


@dataclass(frozen=True)
class RouteCommitment:
    """A chosen exit kept while reachable, so one step cannot flip the route."""
    source_map: tuple[int, int]
    destination_map: tuple[int, int]
    edge: WarpTransition
    until: int


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


# Opponent rosters recorded before defeats were attributed to the objective's
# own fight are not trusted (version 2).
_OPPONENTS_VERSION = 2


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
                 seed: int = 42, online_learning: bool = True,
                 fault_tolerant: bool | None = None, max_rewinds: int = 6,
                 retain_checkpoints: int = 24) -> None:
        if max_rewinds < 0 or retain_checkpoints < 0:
            raise ValueError("Rewind and retention budgets cannot be negative")
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
        self._belief_levels: dict[int, int] = {}
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
        self._battle_escape_attempts = 0
        self._battle_heals = 0
        self._last_progress_frame = bridge.frame_count
        self._blocked_at: dict[tuple[int, int, int, int, int, int], int] = {}
        self._snapshot_failures = 0
        self.watchdog = ProgressWatchdog()
        self._directive_signature: tuple[object, ...] = ()
        self._directive_cache: dict[str, Any] = {}
        self._native_enemy_hp: dict[int, float] = {}
        self._native_own_pp: dict[int, tuple[int, ...]] = {}
        self._native_interventions: dict[int, tuple[int, int]] = {}
        # Public "doesn't affect" evidence is scoped to this battle. It must
        # constrain actions even when the causal type posterior is uncertain;
        # an ability immunity must not be mislabelled as a learned type rule.
        self._observed_no_effect: set[tuple[int, int, int, int]] = set()
        self._battle_scripts = sorted((symbol.address, name) for name, symbol in self._symbols.items()
                                      if name.startswith("BattleScript_") and 0x08000000 <= symbol.address < 0x09000000)
        # Robustness: faults never end a native campaign; the watchdog escalates.
        self.seed = seed
        self.fault_tolerant = self.native if fault_tolerant is None else fault_tolerant
        self.max_rewinds = max_rewinds
        self.retain_checkpoints = retain_checkpoints
        self.scene = "bootstrap" if self.native else "overworld"
        self.last_decision: dict[str, object] | None = None
        self.recent_decisions: deque[dict[str, object]] = deque(maxlen=24)
        self.last_fault: dict[str, object] = {}
        self.last_recovery: dict[str, object] = {}
        self._fault_frames: deque[int] = deque(maxlen=128)
        self._probe_keys: list[int] = []
        self._route_commitment: RouteCommitment | None = None
        self._tabu_transitions: dict[tuple[int, int, int, int, int, int], int] = {}
        self._detour: tuple[tuple[int, int], tuple[int, int], int] | None = None
        self._reentry: tuple[tuple[int, int], WarpTransition, int] | None = None
        self._options_reported = ""
        self._stall_memory: dict[str, int] = {}
        self._rewind_memory: dict[str, int] = {}
        # Physical puzzles (Strength boulders, ice, same-map portals) engage only
        # when ordinary paths and section re-entry cannot reach the target.
        self._puzzle = FeedbackPuzzleExecutor(bridge, self.planner) if self.native else None
        self._puzzle_engaged: tuple[tuple[int, int], str] | None = None
        self._recent_tiles: deque[tuple[int, int, int, int]] = deque(maxlen=16)
        self._throttle: dict[tuple[str, str], tuple[int, int]] = {}
        self._pending_map: tuple[tuple[int, int], int] | None = None
        self._party_snapshot: dict[int, tuple[int, int, tuple[int, ...]]] = {}
        self._flags_seen = bytearray()
        self._party_was_hurt = False
        # Passages a script refused (guards, sandstorms) and the world in which
        # that happened: avoided until a milestone, badge or key item changes.
        self.pushback = PushbackLearner(self.game)
        self._world = ""
        self._gate_relaxed: str = ""
        self._pending_unlock: dict[str, Any] | None = None
        self._provider_cache: dict[str, list[dict[str, Any]]] = {}
        self._no_route_memo: dict[tuple[object, ...], tuple[int, dict[str, Any] | None]] = {}
        # Talking to an objective's NPC that does not complete it: count, then
        # read its script to learn what it is waiting for.
        self._objective_talks: dict[str, int] = {}
        self._talking_to = ""
        self._requirement_memo: dict[str, tuple[int, dict[str, Any] | None]] = {}
        self._switch_tried: set[tuple[tuple[int, int], str, tuple[int, int]]] = set()
        # The switch being walked to, kept until the layout changes or we stand on it.
        self._switch_goal: tuple[tuple[int, int], str, tuple[int, int]] | None = None
        self._switch_tiles: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
        # (map, layout) whose reachable switches were all tried: puzzles may engage.
        self._switch_exhausted: set[tuple[tuple[int, int], str]] = set()
        # Mart purchases (healing items, balls) through the native shop screens.
        self.mart: MartController | None = None
        if self.native and self.catalog is not None and self.topology.catalog is not None:
            self.mart = MartController(bridge, self.game, {name: symbol.address for name, symbol in self._symbols.items()},
                                       self._item_price, self.topology.catalog.constants)
        self._shopping = False
        # Learned Pokemon knowledge: the type chart (shared by both games) and
        # what each objective's trainer brought, for counter-preparation.
        self.shared = SharedKnowledge(self.checkpoint_dir.parent / "shared_knowledge.json"
                                      if self.checkpoint_dir is not None else None)
        self.shared.load()
        self.opponents = OpponentMemory()
        self._battle_roster: dict[int, int] = {}
        self._battle_move_types: dict[int, set[int]] = {}
        self._last_trainer_roster: tuple[dict[int, int], dict[int, set[int]]] = ({}, {})
        self._own_personalities: frozenset[int] = frozenset()
        self._own_hp_seen: dict[int, int] = {}
        self._enemy_hit_key: tuple[int, int, int] | None = None
        self._chart_replayed = False
        # Party state when the latest trainer battle began, for defeat analysis.
        self._battle_start: dict[str, Any] = {}
        self._last_level_frame = 0
        # Durable lessons survive rewinds and supervisor restarts (lessons.json).
        self._lessons_dirty = False
        self._lessons_saved_frame = -1
        self.lessons_path = self.checkpoint_dir / "lessons.json" if self.checkpoint_dir is not None else None
        checkpoint = Path(neural_checkpoint) if neural_checkpoint else ROOT / "artifacts/checkpoints/neural.pt"
        if checkpoint.is_file():
            self.load_neural_checkpoint(checkpoint)
        self._load_lessons()

    def _record(self, kind: str, **values: object) -> None:
        event: dict[str, object] = {"kind": kind, "frame": self.bridge.frame_count, "t": round(time.time(), 3),
                                    "game": self.game, "backend": "native" if self.native else "synthetic"}
        event.update(values)
        self.audit.append(event)
        if len(self.audit) > 20000:
            del self.audit[:5000]
        if self.checkpoint_dir is not None:
            try:
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                with (self.checkpoint_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, sort_keys=True, default=str) + "\n")
            except OSError:
                pass  # Telemetry must never stop the frame owner (viewer locks, full disk).

    def _record_throttled(self, kind: str, key: str, interval: int, **values: object) -> None:
        """Record a repetitive event at most once per interval, counting the rest."""
        frame = self.bridge.frame_count
        previous = self._throttle.get((kind, key))
        if previous is not None and 0 <= frame - previous[0] < interval:
            self._throttle[(kind, key)] = (previous[0], previous[1] + 1)
            return
        if len(self._throttle) > 512:
            self._throttle.clear()
        self._throttle[(kind, key)] = (frame, 0)
        if previous is not None and previous[1]:
            values["suppressed_repeats"] = previous[1]
        self._record(kind, **values)

    @staticmethod
    def _compact_directive(directive: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in directive.items() if key != "training_candidates"}

    # ------------------------------------------------------------------
    # Durable lessons: what failed before must shape the next decision.

    def _lessons_state(self) -> dict[str, Any]:
        return {"version": 1, "game": self.game, "saved_at": round(time.time(), 3),
                "barriers": self.pushback.state_dict(),
                "no_effect": [list(row) for row in sorted(self._observed_no_effect)],
                "preparation": self.planner.preparation.lessons_state(),
                "opponents": self.opponents.state(), "opponents_version": _OPPONENTS_VERSION}

    def _merge_lessons(self, data: Mapping[str, Any]) -> None:
        """Union lessons from another timeline; never forget a learned failure."""
        if data.get("version") != 1 or data.get("game") != self.game:
            return
        barriers = data.get("barriers")
        if isinstance(barriers, dict) and isinstance(barriers.get("barriers"), list):
            try:
                self.pushback.merge(LearnedBarrier.from_json(row) for row in barriers["barriers"][:256]
                                    if isinstance(row, dict))
            except (ValueError, KeyError, TypeError):
                pass
        for row in data.get("no_effect", [])[:2048] if isinstance(data.get("no_effect"), list) else []:
            if isinstance(row, (list, tuple)) and len(row) == 4 and all(type(value) is int for value in row):
                self._observed_no_effect.add((row[0], row[1], row[2], row[3]))
        preparation = data.get("preparation")
        if isinstance(preparation, dict):
            self.planner.preparation.merge_lessons(preparation)
        opponents = data.get("opponents")
        if isinstance(opponents, dict) and data.get("opponents_version") == _OPPONENTS_VERSION:
            # Older rosters mixed in wild whiteouts and trainers met on the way.
            self.opponents.merge_state(opponents)
        self._world = ""  # recompute the active barrier set on the next plan

    def _load_lessons(self) -> None:
        if self.lessons_path is None or not self.lessons_path.is_file():
            return
        try:
            raw = json.loads(self.lessons_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(raw, dict):
            self._merge_lessons(raw)

    def _save_lessons(self, force: bool = False) -> None:
        if self.lessons_path is None or not (self._lessons_dirty or force):
            return
        frame = self.bridge.frame_count
        if not force and 0 <= frame - self._lessons_saved_frame < 600:
            return
        try:
            self.lessons_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.lessons_path.with_name(f"lessons.{time.time_ns()}.tmp")
            temporary.write_text(json.dumps(self._lessons_state(), sort_keys=True), encoding="utf-8")
            temporary.replace(self.lessons_path)
            self._lessons_dirty = False
            self._lessons_saved_frame = frame
        except OSError:
            pass  # lessons are best effort; a locked file only delays them

    def _item_price(self, item: int) -> int:
        if self.catalog is None:
            return 0
        try:
            return self.catalog.item_price(item)
        except ValueError:
            return 0

    def _mart_cycle(self) -> bool:
        """Own the keys while a shop screen is open; True when this frame is handled."""
        if self.mart is None:
            return False
        keys = self.mart.keys()
        if keys is not None:
            if not self._shopping:
                self._shopping = True
                self._record("shop_opened", need=None if self.mart.need is None else
                             [self.mart.need.heal_hp, self.mart.need.balls])
            self.movement.clear()
            self.bridge.set_hardware_keys(keys)
            return True
        if self._shopping:
            self._shopping = False
            bought = self.mart.bought()
            self._record("shop_closed", bought={str(item): count for item, count in bought.items()},
                         plan=[list(entry) for entry in (self.mart.plan or [])])
            # Whatever happened at the counter, do not walk straight back to it.
            self.planner.shop_cooldown_until = self.bridge.frame_count + (36000 if bought else 18000)
            self.mart.reset()
            self._directive_signature = ()
        return False

    def _next_floor_switch(self, current_map: tuple[int, int], coords: Mapping[str, int],
                           astar: GbaConstrainedAStar) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
        """A reachable step-on switch not yet tried in the current barrier layout.

        Step triggers whose scripts change a variable or flag (and are not
        pushback gates) usually reshape the room. Each (layout, switch) pair is
        tried once, so the exploration ends instead of cycling.
        """
        catalog = self.topology.catalog
        if catalog is None:
            return None
        position = (coords["x"], coords["y"])
        layout = self._layout_hash()
        start = (coords["x"], coords["y"], coords["z"])

        def path_to(tile: tuple[int, int]) -> list[tuple[int, int]] | None:
            # Only the chosen switch may be entered; the others stay blocked.
            was_blocked = tile in astar.static_blocked
            astar.static_blocked.discard(tile)
            try:
                return astar.find_optimal_path(start, tile)
            finally:
                if was_blocked:
                    astar.static_blocked.add(tile)

        goal = self._switch_goal
        if goal is not None:
            if goal[:2] == (current_map, layout) and goal[2] != position:
                path = path_to(goal[2])
                if path:
                    return path, goal[2]
            self._switch_goal = None
        candidates: list[tuple[int, list[tuple[int, int]], tuple[int, int]]] = []
        for tile in self._floor_switches(current_map):
            if tile == position or (current_map, layout, tile) in self._switch_tried:
                continue
            path = path_to(tile)
            if path:
                candidates.append((len(path), path, tile))
        if not candidates:
            if self._floor_switches(current_map):
                self._switch_exhausted.add((current_map, layout))
            return None
        _, path, tile = min(candidates, key=lambda row: (row[0], row[2]))
        if len(self._switch_tried) > 512:
            self._switch_tried.clear()
        self._switch_tried.add((current_map, layout, tile))
        self._switch_goal = (current_map, layout, tile)
        self._record("floor_switch", map=list(current_map), tile=list(tile), layout=layout)
        return path, tile

    def _layout_hash(self) -> str:
        _, collision, _ = self.bridge.extract_live_map_matrices()
        return hashlib.blake2b(collision.tobytes(), digest_size=8).hexdigest()

    def _switches_pending(self, current_map: tuple[int, int]) -> bool:
        """Floor switches of this room still to be explored in its current layout."""
        return bool(self._floor_switches(current_map)) and (current_map, self._layout_hash()) not in self._switch_exhausted

    def _floor_switches(self, current_map: tuple[int, int]) -> tuple[tuple[int, int], ...]:
        """Step triggers whose scripts redraw the room (and are not pushback gates)."""
        cached = self._switch_tiles.get(current_map)
        if cached is not None:
            return cached
        catalog = self.topology.catalog
        raw = catalog.maps.get(current_map) if catalog is not None else None
        if catalog is None or raw is None:
            return ()
        rows = load_floor_switches(self.game).get(str(raw["name"]), [])
        self._switch_tiles[current_map] = tuple((int(row[0]), int(row[1])) for row in rows
                                                if gate_at(self.game, current_map, (int(row[0]), int(row[1]))) is None)
        return self._switch_tiles[current_map]

    def _elevator_step(self, transition: WarpTransition) -> WarpTransition:
        """An elevator exit leads wherever its panel last pointed: choose the floor first.

        SaveBlock1.dynamicWarp (+0x14 group, +0x15 number) is set on entry to
        the floor we came from and by the panel's menu, so it is checked live.
        """
        base = self.bridge.get_save_block1_dynamic_ptr()
        pointed = self.bridge.read_u8(base + 0x14), self.bridge.read_u8(base + 0x15)
        if pointed == transition.destination or transition.panel_xy is None:
            return transition
        self._record_throttled("elevator_floor", f"{transition.source}->{transition.destination}", 1800,
                               destination=list(transition.destination), choice=transition.menu_choice,
                               pointed=list(pointed), panel=list(transition.panel_xy))
        return WarpTransition(transition.source, transition.source, transition.panel_xy, transition.panel_xy,
                              kind="script", script=transition.panel_script, npc_local_id=transition.panel_local_id,
                              menu_choice=transition.menu_choice, activation="interact")

    def rearm_recovery(self) -> None:
        """Called by a freshly started worker after restoring its checkpoint."""
        self.watchdog.rearm()
        self._rewind_memory.clear()
        self._record("recovery_rearmed", strategic_unproductive_frames=self.watchdog.strategic_unproductive_frames)

    def _observe_objective_talk(self) -> None:
        """Count conversations with the objective's NPC that ended without completing it."""
        directive = self._last_directive
        script_mode = self.bridge.read_u8(self.bridge.memory_map.script + 1)
        milestone = str(directive.get("milestone", ""))
        if script_mode != 0:
            if (not self._talking_to and directive.get("mode") == "EXECUTE_MILESTONE" and
                    directive.get("action") == "interact" and milestone):
                dest: Any = directive.get("dest")
                coords = self.bridge.get_player_coords_and_map()
                try:
                    near = ((coords["map_group"], coords["map_num"]) == (int(dest[0]), int(dest[1])) and
                            abs(coords["x"] - int(dest[2][0])) + abs(coords["y"] - int(dest[2][1])) <= 3)
                except (TypeError, ValueError, IndexError):
                    near = False
                if near:
                    self._talking_to = milestone
            return
        if not self._talking_to:
            return
        talked, self._talking_to = self._talking_to, ""
        task = next((task for task in self.planner.campaign_dag if task.milestone_id == talked), None)
        try:
            done = task is not None and task.complete(self.bridge, self.bridge.get_decrypted_player_party())
        except ValueError:
            done = False
        if not done:
            self._objective_talks[talked] = self._objective_talks.get(talked, 0) + 1

    def _objective_requirement_directive(self, directive: Mapping[str, Any],
                                         completed: frozenset[str]) -> dict[str, Any] | None:
        """After repeated fruitless talks, pursue what the NPC's own script checks for."""
        milestone = str(directive.get("milestone", ""))
        if (directive.get("mode") != "EXECUTE_MILESTONE" or directive.get("action") != "interact" or
                self._objective_talks.get(milestone, 0) < 3):
            return None
        memo = self._requirement_memo.get(milestone)
        progress = self._flags_progress()
        if memo is not None and memo[0] == progress:
            return memo[1]
        task = next((task for task in self.planner.campaign_dag if task.milestone_id == milestone), None)
        result: dict[str, Any] | None = None
        if task is not None and self.topology.catalog is not None:
            variable, minimum = "", 1
            for condition in task.completion_conditions:
                if condition.kind == "var_at_least":
                    variable = next((name for name, value in self.topology.catalog.constants.items()
                                     if value == condition.identifier and name.startswith("VAR_")), "")
                    minimum = condition.value
            flag = task.flag_name if task.flag_name.startswith("FLAG_") else ""
            branches = completion_requirements(self.game, task.script, flag, variable, minimum)
            needs = unmet_requirements(branches, RamGateWorld(self.bridge, self.game, current_map=True)) if branches else []
            coords = self.bridge.get_player_coords_and_map()
            current_map = coords["map_group"], coords["map_num"]
            for kind, name in needs:
                for target in self._providers(kind, name):
                    key = (int(target["map_group"]), int(target["map_num"]))
                    goal = self._goal_tiles({"npc_local_id": target["npc_local_id"], "action": "interact",
                                             "dest": (*key, (target["x"], target["y"]))},
                                            (int(target["x"]), int(target["y"])))
                    if key != current_map and self._plan_route(current_map, key, coords, completed,
                                                               self._hm_usable(), goal) is None:
                        continue
                    result = {"mode": "UNLOCK_GATE", "dest": (*key, (int(target["x"]), int(target["y"]))),
                              "action": "interact", "script": target["script"], "npc_local_id": target["npc_local_id"],
                              "map_name": target["map_name"], "milestone": milestone,
                              "reason": f"{milestone} requires {name}", "requirement": name}
                    break
                if result is not None:
                    break
            self._record("objective_requirement", milestone=milestone, talks=self._objective_talks.get(milestone, 0),
                         needs=[list(need) for need in needs],
                         target=None if result is None else result["map_name"])
        self._requirement_memo[milestone] = (progress, result)
        return result

    def _key_items(self) -> tuple[int, ...]:
        """Key Items pocket contents (Tea, Go-Goggles, tickets): world state for gates."""
        pocket = 4 if self.game == "emerald" else 1
        descriptor = self.bridge.memory_map.bag_pockets + pocket * 8
        pointer, capacity = self.bridge.read_u32(descriptor), self.bridge.read_u8(descriptor + 4)
        if not pointer or not 0 < capacity <= 64:
            return ()
        raw = self.bridge.read_memory(pointer, capacity * 4)
        return tuple(sorted({int.from_bytes(raw[4 * slot:4 * slot + 2], "little") for slot in range(capacity)} - {0}))

    def _world_signature(self, completed: frozenset[str]) -> str:
        """Milestones and key items: a refused passage is retried when these change."""
        try:
            items = self._key_items()
        except (ValueError, RuntimeError):
            items = ()
        digest = hashlib.blake2b(repr((tuple(sorted(completed)), items)).encode("utf-8"), digest_size=8)
        return digest.hexdigest()

    def _refresh_world(self, completed: frozenset[str]) -> None:
        world = self._world_signature(completed)
        if world != self._world or self.bridge.frame_count % 600 == 0:
            self._world = world
            self.topology.set_learned_gate_edges(self.pushback.active_edges(world, self.bridge.frame_count))

    def _observe_pushback(self, coords: Mapping[str, int]) -> None:
        """Detect a script walking the avatar back after a voluntary step."""
        if not self.native or self.topology.catalog is None:
            return
        script_mode = self.bridge.read_u8(self.bridge.memory_map.script + 1)
        barrier = self.pushback.observe((coords["map_group"], coords["map_num"]), (coords["x"], coords["y"]),
                                        script_mode != 0, self.bridge.frame_count, self._world)
        if barrier is None:
            return
        self.topology.set_learned_gate_edges(self.pushback.active_edges(self._world, self.bridge.frame_count))
        self._lessons_dirty = True
        # The refused exit must not stay committed; the planner picks another way.
        self._route_commitment = None
        self._active_transition = None
        self._reentry = None
        self._directive_signature = ()
        self.movement.clear()
        self._record("passage_refused", map=barrier.map_name, tile=list(barrier.tile),
                     direction=list(barrier.direction), script=barrier.script,
                     requirement=barrier.requirement or "desconocido", observations=barrier.observations)
        self._save_lessons(force=True)

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
        objective = next((task.milestone_id for task in self.planner.campaign_dag
                          if task.milestone_id not in self.metrics.completed_milestones), "")
        self.desired_species |= set(self.planner.counter_targets.get(objective, ()))
        self.metrics.gflownet_samples += 1
        self._record("team_plan", species=sorted(selected), moves=self.desired_moves, coverage_reward=reward)

    def _ensure_belief(self, enemy: BattlePokemonStruct) -> BeliefStateEstimator:
        species = int(enemy.species)
        level = max(1, int(enemy.level))
        if species not in self.beliefs or self._belief_levels.get(species) != level:
            legal = self._legal_moves(species, level)
            bases = self.catalog.species(species).base_stats if self.catalog else (45, 45, 45, 45, 45, 45)
            hypotheses: list[HiddenOpponentSet] = []
            for quality, iv in (("low", 0), ("median", 15), ("high", 31)):
                stats = [max(1, (2 * base + iv) * level // 100 + 5) for base in bases[1:]]
                hypotheses.append(HiddenOpponentSet(quality, species, legal,
                    attack=stats[0], defense=stats[1], speed=stats[2],
                    special_attack=stats[3], special_defense=stats[4]))
            self.beliefs[species] = BeliefStateEstimator(hypotheses)
            self._belief_levels[species] = level
        return self.beliefs[species]

    @staticmethod
    def _effective_speed(raw: float, battler: BattlePokemonStruct) -> float:
        numerator, denominator = STAGE_RATIOS[max(0, min(12, int(battler.statStages[3])))]
        return raw * numerator / denominator / (4 if int(battler.status1) & 0x40 else 1)

    def _outgoing_context(self, own: BattlePokemonStruct, enemy: BattlePokemonStruct,
                          move: MoveInfo, belief: BeliefStateEstimator) -> DamageContext:
        """Known own mechanics and visible boosts, with rival raw stats still a prior."""
        return DamageContext(max(1, int(own.level)),
            max(1, int(own.spAttack if move.is_special else own.attack)),
            max(1, int(belief.expected_stat("special_defense" if move.is_special else "defense"))),
            move.power, move_type=move.move_type, category="special" if move.is_special else "physical",
            stab=move.move_type in (int(own.type1), int(own.type2)),
            attack_stage=int(own.statStages[4 if move.is_special else 1]),
            defense_stage=int(enemy.statStages[5 if move.is_special else 2]),
            attacker_ability=int(own.ability), attacker_item=int(own.item),
            attacker_species=int(own.species), attacker_status=int(own.status1),
            attacker_hp=max(1, int(own.hp)), attacker_max_hp=max(1, int(own.maxHP)),
            defender_species=int(enemy.species), defender_status=int(enemy.status1),
            defender_ability=self.knowledge.revealed_abilities.get(int(enemy.species), 0),
            defender_item=self.knowledge.revealed_items.get(int(enemy.species), 0))

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
        # A revealed attack's threat is its power times what the chart says it
        # does to this defender: Karate Chop is no worry for a Ghost.
        defender_types = tuple(dict.fromkeys((int(defender.type1), int(defender.type2))))
        powers = [self._move(move).power * self._known_multiplier(self._move(move).move_type, defender_types, 1.0)
                  for move in revealed]
        if not powers:
            # Nothing revealed yet: its own types are its likely attacks (STAB),
            # weighed by what the learned chart says they do to this defender.
            powers = [50 * self._known_multiplier(attack_type, defender_types, 1.0)
                      for attack_type in dict.fromkeys((int(enemy.type1), int(enemy.type2))) if attack_type != 9]
        estimated_power = int(max(powers, default=50))
        estimated_attack = max(belief.expected_stat("attack"), belief.expected_stat("special_attack"))
        estimate = DamageContext(max(1, int(enemy.level)), max(1, int(estimated_attack)),
                                 context.defense, estimated_power, stab=True,
                                 attack_stage=max(int(enemy.statStages[1]), int(enemy.statStages[4])),
                                 defense_stage=min(int(defender.statStages[2]), int(defender.statStages[5])),
                                 attacker_status=int(enemy.status1))
        expected = self.verifier.compute_context_damage_reference(estimate, 93) + residual
        return proof, expected / max(1, int(defender.hp))

    def _outgoing_damage_estimates(self, own: BattlePokemonStruct,
                                   enemy: BattlePokemonStruct) -> tuple[MoveDamageEstimate, ...]:
        """Own moves and observed-type/hidden-stat beliefs, without rival stats."""
        belief = self._ensure_belief(enemy)
        opponents = tuple(dict.fromkeys((int(enemy.type1), int(enemy.type2))))
        estimates: list[MoveDamageEstimate] = []
        for slot in range(4):
            if not own.moves[slot] or not own.pp[slot]:
                continue
            move = self._move(int(own.moves[slot]))
            expected = 0.0
            if (move.power > 0 and move.effect != _SELF_KO_EFFECT and
                    self._no_effect_key(enemy, move.move_type) not in self._observed_no_effect):
                context = self._outgoing_context(own, enemy, move, belief)
                neutral = self.verifier.compute_context_damage_reference(context, 93)
                expected = neutral * self._type_multiplier(move.move_type, opponents)
            accuracy_difference = int(own.statStages[6]) - int(enemy.statStages[7])
            accuracy = min(1.0, (move.accuracy or 100) / 100 *
                           (3 + max(0, accuracy_difference)) / (3 + max(0, -accuracy_difference)))
            estimates.append(MoveDamageEstimate(slot, expected, accuracy, int(own.pp[slot])))
        return tuple(estimates)

    def _known_multiplier(self, attack_type: int, defender_types: tuple[int, ...], default: float) -> float:
        value = self.shared.chart.multiplier(attack_type, defender_types)
        return default if value is None else value

    def _type_multiplier(self, attack_type: int, defender_types: tuple[int, ...]) -> float:
        """Learned exact multiplier when known; otherwise the causal posterior mean."""
        value = self.shared.chart.multiplier(attack_type, defender_types)
        if value is not None:
            return value
        return sum(mult * prob for mult, prob in self.causal.efficacy_distribution(attack_type, defender_types).items())

    def _learn_effectiveness(self, attack_type: int, defender_types: tuple[int, ...], category: str,
                             species: int, origin: str) -> None:
        """One visible battle message becomes durable type-chart knowledge."""
        abilities: tuple[int, ...] = ()
        known = False
        if self.catalog is not None and 0 < species < self.catalog.species_count:
            try:
                abilities, known = tuple(self.catalog.species(species).abilities), True
            except ValueError:
                abilities = ()
        # Without the species' possible abilities an immunity is not attributed
        # to types until a second species confirms it.
        if self.shared.chart.observe(attack_type, defender_types, category, species, abilities, known):
            self.shared.dirty = True
            flag = {NO_EFFECT: "no_effect", RESISTED: "not_very_effective", SUPER: "super_effective",
                    NEUTRAL: "neutral"}[category]
            try:
                self.causal.observe_effectiveness(attack_type, defender_types, flag)
            except (ValueError, ArithmeticError):
                pass
            self._record("type_fact_learned", attack_type=attack_type, defender_types=list(defender_types),
                         category=category, origin=origin, known_cells=self.shared.chart.known_cells())

    def _replay_chart(self) -> None:
        """Make the causal posterior agree with remembered facts after a load."""
        for key, categories in self.shared.chart.facts.items():
            if len(categories) != 1:
                continue
            flag = {NO_EFFECT: "no_effect", RESISTED: "not_very_effective", SUPER: "super_effective",
                    NEUTRAL: "neutral"}[next(iter(categories))]
            try:
                self.causal.observe_effectiveness(key[0], key[1:], flag)
            except (ValueError, ArithmeticError):
                continue
        self._chart_replayed = True

    @staticmethod
    def _no_effect_key(enemy: BattlePokemonStruct, attack_type: int) -> tuple[int, int, int, int]:
        first, second = sorted((int(enemy.type1), int(enemy.type2)))
        return int(enemy.species), first, second, attack_type

    def _observe_no_effect(self, enemy: BattlePokemonStruct, move_id: int) -> None:
        """Avoid repeating a visibly immune attack, without consulting a type oracle."""
        move = self._move(move_id)
        if not move.power:
            return
        key = self._no_effect_key(enemy, move.move_type)
        if move.move_id not in UNTYPED_MOVES and move.power > 1:
            self._learn_effectiveness(move.move_type, tuple(dict.fromkeys((int(enemy.type1), int(enemy.type2)))),
                                      NO_EFFECT, int(enemy.species), "own_move")
        if key not in self._observed_no_effect:
            self._observed_no_effect.add(key)
            self._lessons_dirty = True
            self._record("move_no_effect_observed", move=move_id, species=int(enemy.species),
                         attack_type=move.move_type, scope="campaign", cause="type_or_ability_unknown")

    def choose_battle_decision(self) -> BattleDecision:
        """Combine recurrent logits, causal EFE and a conditional survival filter."""
        party = player_party_in_battle_order(self.bridge)
        features = encode_battle(self.bridge, self.knowledge, catalog=self.catalog)
        me = self.bridge.get_battle_pokemon(features.acting_battler)
        target_index = next((index for index in (0, 1) if bool(features.target_mask[0, index])), 0)
        enemy_slot = features.target_battlers[target_index]
        enemy = self.bridge.get_battle_pokemon(enemy_slot if enemy_slot >= 0 else 1)
        self.observed_species.add(int(enemy.species))
        self._refresh_roster(party)
        belief = self._ensure_belief(enemy)
        if int(me.hp) > 0:
            proof, retaliation = self._incoming_assessment(me, enemy)
        else:
            # A fainted active battler cannot be hit again; only the replacement's
            # exposure matters, and each switch option below assesses it.
            proof = DamageResult(False, 0, 0, 0, "unsupported", "active battler fainted; replacement required")
            retaliation = 1.0
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
        outgoing = self._outgoing_damage_estimates(me, enemy)
        acts_first = self._effective_speed(int(me.speed), me) > self._effective_speed(belief.expected_stat("speed"), enemy)
        productive_attack = False
        options: list[tuple[float, float, BattleDecision]] = []
        for slot in range(4):
            if (features.forced_switch or not bool(features.action_mask[0, slot]) or
                    not me.moves[slot] or not me.pp[slot]):
                continue
            move = self._move(int(me.moves[slot]))
            if (self._no_effect_key(enemy, move.move_type) in self._observed_no_effect and
                    any(estimate.expected_damage > 0 for estimate in outgoing)):
                continue
            efe, _, _ = self.causal.evaluate_free_energy(move.move_type, list(opponents), move.power, hp_ratio)
            selected = next(estimate for estimate in outgoing if estimate.index == slot)
            damaging = selected.expected_damage > 0
            horizon = estimate_attack_plan(max(1, int(me.hp)), enemy_hp, retaliation * max(1, int(me.hp)),
                outgoing, first_move_index=slot if damaging else None, acts_first=acts_first,
                initial_delay_turns=0 if damaging else 1)
            productive_attack |= damaging and horizon.can_finish
            risk = -math.expm1(-horizon.risk_proxy)
            expected_damage = selected.expected_damage
            # Compare both attacks and switches until an estimated KO. The
            # bounded indicator below is explicitly not a calibrated probability.
            score = efe - expected_damage / max(1, enemy_max_hp) * 4 - float(logits[0, slot]) * 0.05
            options.append((horizon.risk_proxy, score - float(targets[0, target_index]) * 0.02,
                BattleDecision("move", slot, 0, risk, proof,
                    "recurrent policy and expected damage burden until KO", target_index,
                    RISK_MODEL, horizon.risk_proxy, horizon.turns_to_ko)))
        for bench_action, party_index in enumerate(features.switch_party_indices, start=4):
            if not bool(features.action_mask[0, bench_action]):
                continue
            mon = party[party_index]
            replacement = BattlePokemonStruct()
            replacement.species, replacement.level = int(mon["species"]), int(mon["level"])
            replacement.hp, replacement.maxHP = int(mon["hp"]), int(mon["maxHP"])
            replacement.attack, replacement.defense = int(mon["attack"]), int(mon["defense"])
            replacement.spAttack, replacement.spDefense = int(mon["spAttack"]), int(mon["spDefense"])
            replacement.speed = int(mon["speed"])
            for stage in range(8):
                replacement.statStages[stage] = 6
            for slot in range(4):
                replacement.moves[slot], replacement.pp[slot] = int(mon["moves"][slot]), int(mon["pp"][slot])
            replacement.type1, replacement.type2 = self._species_types(int(mon["species"]))
            replacement.status1 = int(mon["status"])
            replacement.item = int(mon["held_item"])
            if self.catalog is not None:
                replacement.ability = self.catalog.species(int(mon["species"])).abilities[int(mon["ability_num"])]
            switch_proof, switch_risk = self._incoming_assessment(replacement, enemy)
            horizon = estimate_attack_plan(int(replacement.hp), enemy_hp,
                switch_risk * int(replacement.hp), self._outgoing_damage_estimates(replacement, enemy),
                switching=True, acts_first=self._effective_speed(int(replacement.speed), replacement) >
                self._effective_speed(belief.expected_stat("speed"), enemy))
            if not horizon.can_finish and productive_attack and not features.forced_switch:
                continue
            risk = -math.expm1(-horizon.risk_proxy)
            score = 1.0 - float(logits[0, bench_action]) * 0.05
            options.append((horizon.risk_proxy, score, BattleDecision("switch", party_index, 0,
                risk, switch_proof, "switch free-hit plus replacement damage burden until KO", 0,
                RISK_MODEL, horizon.risk_proxy, horizon.turns_to_ko)))
        heal = self._healing_item_option(me, enemy_hp, retaliation, outgoing, acts_first,
                                         bool(features.forced_switch), proof)
        if heal is not None:
            options.append(heal)
        _, trainer = self.bridge.is_in_battle()
        if not features.forced_switch and not trainer and int(enemy.species) in self.desired_species and int(enemy.species) not in self.owned_species:
            inventory = BallInventory({name: self.bridge.get_inventory().get(item, 0)
                                       for name, item in BALL_ITEMS.items()})
            context = CaptureContext(int(enemy.species), enemy_max_hp, enemy_hp,
                opponents, level=max(1, int(enemy.level)), status=int(enemy.status1),
                turn_count=self._battle_turns,
                catch_rate=self.catalog.species(int(enemy.species)).catch_rate if self.catalog else None,
                failure_faint_probability=min(1.0, retaliation))
            coverage = set(PLAYABLE_TYPES) - {t for species in self.owned_species for t in self._species_types(species)}
            capture = self.capturer.plan_capture(context, inventory, self.owned_species, coverage)
            if capture.ball_choice is not None and capture.action.startswith("THROW_"):
                ball = capture.ball_choice
                decision = BattleDecision("ball", 0, BALL_ITEMS[ball.ball], capture.estimated_faint_probability,
                                          proof, capture.reason)
                # Safe new coverage is a strategic objective, before battle damage ranking.
                if proof.formally_safe or capture.estimated_faint_probability < min((x[2].estimated_risk for x in options), default=1):
                    return self._audit_decision(decision, me, enemy, len(options))
        escape = self._story_escape_decision(me, enemy, trainer, features.forced_switch, retaliation)
        if (escape is not None and not any(option[2].proof.formally_safe for option in options) and
                escape.estimated_risk <= min((option[2].estimated_risk for option in options), default=1.0)):
            return self._audit_decision(escape, me, enemy, len(options) + 1)
        if not options:
            decision = BattleDecision("move", 0, 0, min(1.0, retaliation), proof, "engine-forced Struggle after PP exhaustion")
        else:
            safe = [option for option in options if option[2].proof.formally_safe]
            decision = min(safe, key=lambda option: option[1])[2] if safe else min(options, key=lambda option: (option[0], option[1]))[2]
        return self._audit_decision(decision, me, enemy, len(options))

    def _healing_item_option(self, me: BattlePokemonStruct, enemy_hp: int, retaliation: float,
                             outgoing: tuple[MoveDamageEstimate, ...], acts_first: bool, forced_switch: bool,
                             proof: DamageResult) -> tuple[float, float, BattleDecision] | None:
        """Drink a Potion when attacking now would likely lose and the heal outpaces the damage.

        The same damage-burden model ranks it against attacks and switches:
        one turn spent healing, then the best attacks from the healed HP.
        """
        if not self.native or forced_switch or self.topology.catalog is None or int(me.hp) <= 0:
            return None
        hp, max_hp = int(me.hp), max(1, int(me.maxHP))
        if hp / max_hp > 0.5 or self._battle_heals >= 3:
            return None
        try:
            inventory = self.bridge.get_inventory()
        except (ValueError, RuntimeError):
            return None
        stocked = sorted((HEALING_ITEMS[name], item) for item, name in
                         item_ids(self.topology.catalog.constants, tuple(HEALING_ITEMS)).items()
                         if inventory.get(item, 0) > 0)
        if not stocked:
            return None
        deficit = max_hp - hp
        amount, item = next((row for row in stocked if row[0] >= 0.6 * deficit), stocked[-1])
        incoming = max(0.0, retaliation) * max(1, hp)
        healed = min(max_hp, hp + amount)
        if healed - hp <= incoming * 1.1:
            return None  # healing cannot outpace the opponent's damage: attacking is better
        horizon = estimate_attack_plan(healed, enemy_hp, incoming, outgoing, acts_first=acts_first,
                                       initial_delay_turns=1)
        risk = -math.expm1(-horizon.risk_proxy)
        return (horizon.risk_proxy, 0.5, BattleDecision("ball", 0, item, risk, proof,
                f"use healing item (+{min(amount, deficit)} HP) before the decisive exchange", 0,
                RISK_MODEL, horizon.risk_proxy, horizon.turns_to_ko))

    def _story_escape_decision(self, own: BattlePokemonStruct, enemy: BattlePokemonStruct,
                               trainer: bool, forced_switch: bool, retaliation: float) -> BattleDecision | None:
        """Spend travel time on the story instead of farming optional wild fights.

        Escape uses own condition and a public species/level speed prior. It is
        an estimate, never a proof, and only two failed attempts are allowed.
        Potential trapping abilities come from the species catalog, not hidden
        runtime ability reads. Training, trainers and wanted captures retain
        their ordinary battle plans.
        """
        if (not self.native or trainer or forced_switch or int(own.hp) <= 0 or
                self._battle_escape_attempts >= 2 or self._last_directive.get("mode") not in
                ("EXECUTE_MILESTONE", "TEACH_FIELD_MOVE", "HEAL_POKECENTER")):
            return None
        if (int(enemy.species) in self.desired_species and int(enemy.species) not in self.owned_species and
                any(count > 0 for item, count in self.bridge.get_inventory().items() if item in BALL_ITEMS.values())):
            return None
        if int(own.status2) & ((7 << 13) | (1 << 26)):
            return None
        if self.catalog is not None and set(self.catalog.species(int(enemy.species)).abilities) & {23, 42, 71}:
            return None
        # The native speed formula is compared to a prior with no asserted
        # knowledge of the rival's actual IVs, nature, EVs or effective speed.
        speed_prior = max(1.0, self._effective_speed(self._ensure_belief(enemy).expected_stat("speed"), enemy))
        speed = self._effective_speed(int(own.speed), own)
        probability = (1.0 if speed >= speed_prior else
                       min(1.0, (speed * 128 / speed_prior + 30 * (self._battle_escape_attempts + 1)) / 256))
        if probability < 0.8 or (int(own.level) < int(enemy.level) + 5 and self._last_directive.get("mode") != "HEAL_POKECENTER"):
            return None
        risk = min(1.0, max(0.0, (1 - probability) * retaliation))
        proof = DamageResult(False, 0, 0, 0, "unsupported", "escape chance estimated from visible data and speed prior")
        return BattleDecision("run", 0, 0, risk, proof,
            "Skip optional wild battle to pursue the current story or healing objective", risk_model="escape_failure_retaliation_estimate")

    def _decision_summary(self, decision: BattleDecision, me: BattlePokemonStruct | None,
                          enemy: BattlePokemonStruct | None, alternatives: int) -> dict[str, object]:
        """Human-readable record of what was chosen, against what, and why."""
        summary: dict[str, object] = {
            "frame": self.bridge.frame_count, "turn": self._battle_turns + 1, "action": decision.action,
            "index": decision.index, "risk": round(float(decision.estimated_risk), 4),
            "risk_proxy": None if decision.risk_proxy is None else round(float(decision.risk_proxy), 4),
            "turns_to_ko": decision.expected_turns_to_ko, "proof": decision.proof.proof_status,
            "formally_safe": decision.proof.formally_safe, "reason": decision.reason,
            "alternatives": alternatives, "trainer": bool(self.bridge.is_in_battle()[1])}
        if decision.action == "move" and me is not None and 0 <= decision.index < 4:
            summary["move"] = move_name(self.catalog, int(me.moves[decision.index])) if me.moves[decision.index] else "STRUGGLE"
        elif decision.action == "ball":
            summary["move"] = next((name for name, item in BALL_ITEMS.items() if item == decision.item),
                                   f"ITEM #{decision.item}")
        elif decision.action == "switch":
            summary["move"] = f"party {decision.index + 1}"
        elif decision.action == "run":
            summary["move"] = "RUN"
        if me is not None and me.species:
            summary["own"] = {"name": species_name(self.catalog, int(me.species)), "level": int(me.level),
                              "hp": int(me.hp), "maxHP": int(me.maxHP)}
        if enemy is not None and enemy.species:
            summary["enemy"] = {"name": species_name(self.catalog, int(enemy.species)), "level": int(enemy.level),
                                "hp_fraction": round(int(enemy.hp) / max(1, int(enemy.maxHP)), 4)}
        return summary

    def _audit_decision(self, decision: BattleDecision, me: BattlePokemonStruct | None = None,
                        enemy: BattlePokemonStruct | None = None, alternatives: int = 0) -> BattleDecision:
        if decision.proof.formally_safe:
            self.metrics.proved_decisions += 1
        else:
            self.metrics.unproven_decisions += 1
        summary = self._decision_summary(decision, me, enemy, alternatives)
        self.last_decision = summary
        self.recent_decisions.append(summary)
        self._record("battle_decision", decision=asdict(decision), summary=summary,
                     selection="proved" if decision.proof.formally_safe else "minimum_estimated_risk")
        self._battle_turns += 1
        if decision.action == "run":
            self._battle_escape_attempts += 1
        if decision.action == "ball" and decision.item > 12:
            self._battle_heals += 1
        return decision

    def _fallback_battle_decision(self, me: BattlePokemonStruct, reason: str) -> BattleDecision:
        """Strongest legal attack by ROM power when the full policy cannot decide."""
        best_slot, best_value = 0, -1.0
        for slot in range(4):
            move_id, pp = int(me.moves[slot]), int(me.pp[slot])
            if not move_id or not pp:
                continue
            try:
                info = self._move(move_id)
                enemy = self.bridge.get_battle_pokemon(1)
                if self._no_effect_key(enemy, info.move_type) in self._observed_no_effect:
                    value = 0.0
                else:
                    value = 0.5 if info.effect == _SELF_KO_EFFECT else info.power * (info.accuracy or 100) / 100 + 1
            except ValueError:
                value = 1.0
            if value > best_value:
                best_slot, best_value = slot, value
        proof = DamageResult(False, 0, 0, 0, "unsupported", "fallback policy after a decision fault")
        return BattleDecision("move", best_slot, 0, 1.0, proof, f"fallback strongest legal attack: {reason}")

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
                        if str(event["text"]).lower() in ("immune", "no_effect", "doesnt_affect"):
                            self._observe_no_effect(enemy, move.move_id)
                        belief = self._ensure_belief(enemy)
                        is_special = move.move_type >= 10
                        defense = belief.expected_stat("special_defense" if is_special else "defense")
                        context = DamageContext(max(1, int(me.level)),
                            max(1, int(me.spAttack if is_special else me.attack)),
                            max(1, int(defense)), move.power, move_type=move.move_type,
                            stab=move.move_type in (int(me.type1), int(me.type2)),
                            category="special" if is_special else "physical")
                        neutral = self.verifier.compute_context_damage_reference(context, 93)
                        self.causal.observe_intervention(DamageObservation(move.move_type,
                            tuple(dict.fromkeys((int(enemy.type1), int(enemy.type2)))), float(event["damage"]),
                            neutral, engine_text_flag=str(event["text"]), is_status_move=move.power == 0,
                            defender_hp_before=int(enemy.hp) + int(event["damage"]), relative_noise=0.5))
                        self.metrics.causal_observations += 1
            return
        in_battle = self.bridge.is_in_battle()[0]
        if in_battle:
            self._process_native_battle_disclosures()
        party = player_party_in_battle_order(self.bridge)
        hp = tuple(int(mon["hp"]) for mon in party)
        for index, (before, after) in enumerate(zip(self._last_party_hp, hp)):
            if before > 0 and after == 0:
                self.metrics.faints += 1
                self._record("faint", party_index=index)
        self._last_party_hp = hp
        self._track_party_changes(party, in_battle)

    def _track_party_changes(self, party: list[dict[str, Any]], in_battle: bool) -> None:
        """Timeline events for evolutions, level-ups, learned moves and full heals."""
        current = {int(mon["pid"]): (int(mon["species"]), int(mon["level"]), tuple(int(move) for move in mon["moves"]))
                   for mon in party if not mon.get("is_egg", False)}
        previous = self._party_snapshot
        if previous:
            for pid, (species, level, moves) in current.items():
                old = previous.get(pid)
                if old is None:
                    self._record("party_joined", species=species, name=species_name(self.catalog, species), level=level)
                    continue
                if species != old[0]:
                    self.metrics.evolutions += 1
                    self._record("evolution", from_species=old[0], to_species=species,
                                 from_name=species_name(self.catalog, old[0]), to_name=species_name(self.catalog, species))
                if level > old[1]:
                    self.metrics.level_ups += level - old[1]
                    self._last_level_frame = self.bridge.frame_count
                    self._record("level_up", species=species, name=species_name(self.catalog, species),
                                 level=level, previous_level=old[1])
                if moves != old[2]:
                    learned = [move for move in moves if move and move not in old[2]]
                    forgotten = [move for move in old[2] if move and move not in moves]
                    if learned or forgotten:
                        self._record("moves_changed", species=species, name=species_name(self.catalog, species),
                                     learned=[move_name(self.catalog, move) for move in learned],
                                     forgotten=[move_name(self.catalog, move) for move in forgotten])
        self._party_snapshot = current
        if party and not in_battle:
            healthy = all(int(mon["hp"]) == int(mon["maxHP"]) and not int(mon.get("status", 0)) for mon in party)
            if healthy and self._party_was_hurt:
                self.metrics.heals += 1
                pre_battle = self._last_directive.get("pre_battle")
                if pre_battle:
                    self.planner.preparation.record_restore(str(pre_battle))
                self._record("party_healed", members=len(party), pre_battle=pre_battle or None)
            self._party_was_hurt = not healthy

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
                if self.bridge.is_in_battle()[1] and int(mon.personality) not in self._own_personalities:
                    self._battle_roster[int(mon.species)] = max(int(mon.level), self._battle_roster.get(int(mon.species), 0))
                move_id = self.bridge.read_u16(self._symbols["gLastMoves"].address + 2 * slot)
                if 0 < move_id <= 354 and self.bridge.is_in_battle()[1]:
                    self._battle_move_types.setdefault(int(mon.species), set()).add(self._move(move_id).move_type)
                if 0 < move_id <= 354 and move_id not in self.knowledge.revealed_moves.get(int(mon.species), []):
                    self.knowledge.reveal_move(int(mon.species), move_id)
                    self._ensure_belief(mon).observe_move(move_id)
                    self.metrics.belief_updates += 1
                    self._record("opponent_move_revealed", species=int(mon.species), move=move_id)
                fraction = int(mon.hp) / int(mon.maxHP)
                previous = self._native_enemy_hp.get(slot, fraction)
                intervention = self._native_interventions.get(attacker)
                # Immunity produces no HP delta, so the direct-damage branch
                # below can never see it. Decode the result of the own move
                # while that intervention is active; misses/failed moves and
                # stale flags from an opponent's turn are not evidence.
                if target == slot and intervention is not None and not positions[attacker] & 1:
                    attempted_move, started_frame = intervention
                    result_flags = self.bridge.read_u8(self._symbols["gMoveResultFlags"].address)
                    current_move = self.bridge.read_u16(self._symbols["gCurrentMove"].address)
                    if (result_flags & 8 and not result_flags & 0x21 and current_move == attempted_move and
                            0 <= self.bridge.frame_count - started_frame < 1800):
                        self._observe_no_effect(mon, attempted_move)
                        self._native_interventions.pop(attacker, None)
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
                        context = self._outgoing_context(own, mon, move, belief)
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
                        # Damage is counter-evidence to a remembered immunity.
                        self._observed_no_effect.discard(self._no_effect_key(mon, move.move_type))
                        category = result_category(result_flags)
                        if category is not None and move.power > 1 and move.move_id not in UNTYPED_MOVES:
                            self._learn_effectiveness(move.move_type, tuple(dict.fromkeys((int(mon.type1), int(mon.type2)))),
                                                      category, int(mon.species), "own_move")
                        self._native_interventions.pop(attacker, None)
                self._native_enemy_hp[slot] = fraction
            else:
                self._observe_enemy_hit(slot, mon, positions, attacker, target)
                pp = tuple(int(value) for value in mon.pp)
                old = self._native_own_pp.get(slot, pp)
                used = next((index for index in range(4) if pp[index] < old[index]), None)
                if used is not None and mon.moves[used]:
                    self._native_interventions[slot] = (int(mon.moves[used]), self.bridge.frame_count)
                self._native_own_pp[slot] = pp

    def _observe_enemy_hit(self, slot: int, mon: BattlePokemonStruct, positions: bytes,
                           attacker: int, target: int) -> None:
        """The opponent's hits on our Pokemon teach the chart for our own types."""
        hp = int(mon.hp)
        previous = self._own_hp_seen.get(slot, hp)
        self._own_hp_seen[slot] = hp
        if target != slot or not attacker < len(positions) or not positions[attacker] & 1:
            return
        current_move = self.bridge.read_u16(self._symbols["gCurrentMove"].address)
        if not 0 < current_move <= 354 or current_move in UNTYPED_MOVES:
            return
        move = self._move(current_move)
        if move.power <= 1:
            return  # fixed damage (Seismic Toss, Dragon Rage) shows no effectiveness
        flags = self.bridge.read_u8(self._symbols["gMoveResultFlags"].address)
        category = result_category(flags)
        if category is None or (category != NO_EFFECT and hp >= previous):
            return
        key = (attacker, current_move, self.bridge.frame_count // 240)
        if key == self._enemy_hit_key:
            return
        self._enemy_hit_key = key
        self._learn_effectiveness(move.move_type, tuple(dict.fromkeys((int(mon.type1), int(mon.type2)))),
                                  category, int(mon.species), "opponent_move")

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
        if self.native and self._can_run():
            # Running reaches the same tile in half the frames and changes no
            # encounter or trainer rule; the world model's walk preference was
            # only ever costing time once the Running Shoes are obtained.
            selection = 1
        self._last_action_id = directions[0] + (4 if selection else 0)
        self.metrics.neural_plans += 1
        return "run" if selection else "walk"

    def _can_run(self) -> bool:
        catalog = self.topology.catalog
        flag = catalog.constants.get("FLAG_SYS_B_DASH") if catalog is not None else None
        if flag is None:
            return False
        try:
            return self.bridge.get_event_flag(int(flag))
        except ValueError:
            return False

    def _hm_usable(self) -> frozenset[str]:
        return self.field.hm_usable() if self.field is not None else frozenset()

    def _note_arrival(self, map_key: tuple[int, int], xy: tuple[int, int]) -> None:
        """Detect two-tile back-and-forth that is not intentional grass training."""
        self._recent_tiles.append((*map_key, *xy))
        if len(self._recent_tiles) < 12 or self._last_directive.get("mode") == "OPTIMAL_TRAINING_CURRICULUM":
            return
        tiles = set(list(self._recent_tiles)[-12:])
        if len(tiles) > 2:
            return
        self._recent_tiles.clear()
        self._record_throttled("oscillation_detected", str(map_key), 1200, tiles=sorted(list(tile) for tile in tiles),
                               directive=self._compact_directive(self._last_directive))
        commitment = self._route_commitment
        if commitment is not None:
            self._route_commitment = None
            self._tabu(commitment.edge, "oscillation while following a committed exit",
                       goal=commitment.destination_map)
        elif self.native:
            self._start_detour("oscillation without a committed exit")

    def _directive_goal(self) -> tuple[int, int] | None:
        commitment = self._route_commitment
        if commitment is not None:
            return commitment.destination_map
        dest = self._last_directive.get("dest")
        try:
            return (int(dest[0]), int(dest[1])) if dest is not None else None
        except (TypeError, ValueError, IndexError):
            return None

    def _tabu(self, edge: WarpTransition, reason: str, frames: int = _TABU_FRAMES,
              goal: tuple[int, int] | None = None) -> bool:
        """Temporarily forbid an exit so the planner must try an alternative.

        An exit that is the only known route to the goal is never forbidden:
        that would turn a recovery into a guaranteed stall.
        """
        edge_id = HierarchicalWarpGraph._edge_id(edge)
        if edge_id in self._tabu_transitions:
            return True
        self.topology.block_transition_id(edge_id)
        goal = goal if goal is not None else self._directive_goal()
        if goal is not None and goal != edge.source:
            completed = frozenset(self.metrics.completed_milestones)
            # Validate the alternative with walking costs, gates and learned
            # refusals: a coarse map graph alone once banned the only real exit.
            coords = self.bridge.get_player_coords_and_map()
            here = (coords["map_group"], coords["map_num"]) == edge.source
            alternative = self.topology.find_route(
                edge.source, goal, completed, (coords["x"], coords["y"]) if here else None,
                self.bridge if self.native else None, validate_local=self.native, hm_usable=self._hm_usable())
            if alternative is None:
                self.topology.unblock_transition(edge_id)
                self._record_throttled("transition_tabu_skipped", str(edge_id), 1800, exit=list(edge.source_xy),
                                       reason=f"{reason}; it is the only known route")
                return False
        self._tabu_transitions[edge_id] = self.bridge.frame_count + frames
        self._record("transition_tabu", source=list(edge.source), destination=list(edge.destination),
                     exit=list(edge.source_xy), exit_kind=edge.kind, frames=frames, reason=reason)
        return True

    def _expire_tabu(self) -> None:
        if not self._tabu_transitions:
            return
        frame = self.bridge.frame_count
        for edge_id, until in list(self._tabu_transitions.items()):
            if frame >= until or until - frame > 4 * _TABU_FRAMES:
                del self._tabu_transitions[edge_id]
                self.topology.unblock_transition(edge_id)

    def _goal_tiles(self, directive: Mapping[str, Any], target: tuple[int, int]) -> frozenset[tuple[int, int]]:
        """Tiles from which the directive can be executed (beside an NPC, across a counter)."""
        if int(directive.get("npc_local_id", 0)) > 0 or directive.get("action") == "interact":
            goals = {(target[0] + dx, target[1] + dy) for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))}
            destination = directive.get("dest")
            coords = self.bridge.get_player_coords_and_map()
            current_map = coords["map_group"], coords["map_num"]
            key = tuple(destination[:2]) if isinstance(destination, (list, tuple)) else current_map
            behaviors = None
            if key == current_map:
                behaviors = self.bridge.extract_behavior_map()
            elif self.topology.catalog is not None and len(key) == 2:
                behaviors = self.topology.catalog.behavior_grid((int(key[0]), int(key[1])))
            if behaviors is not None:
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    x, y = target[0] + dx, target[1] + dy
                    if (0 <= y < behaviors.shape[0] and 0 <= x < behaviors.shape[1] and
                            int(behaviors[y, x]) == 0x80):
                        goals.add((target[0] + 2 * dx, target[1] + 2 * dy))
            return frozenset(goals)
        return frozenset({target})

    def _choose_transition(self, current_map: tuple[int, int], destination_map: tuple[int, int],
                           coords: Mapping[str, int], completed: frozenset[str],
                           hm_usable: frozenset[str], allow_surf: bool,
                           goal_tiles: frozenset[tuple[int, int]] = frozenset()) -> WarpTransition | None:
        """Best exit, but keep a committed exit while it stays reachable.

        The search targets the tiles where the directive can run, so a cave
        section reached only through another floor is entered correctly.
        Replanning after every step can flip between nearly equal routes and,
        across a whole region, costs seconds; a commitment holds the chosen
        exit for a bounded time and is checked before any new search.
        """
        frame = self.bridge.frame_count
        leg = f"{current_map}->{destination_map}"
        commitment = self._route_commitment
        same_leg = (commitment is not None and
                    (commitment.source_map, commitment.destination_map) == (current_map, destination_map))
        if (commitment is not None and same_leg and frame < commitment.until and
                HierarchicalWarpGraph._edge_id(commitment.edge) not in self.topology.blocked_transitions and
                self.topology.edge_reachable(commitment.edge, self.bridge, allow_surf, hm_usable)):
            return commitment.edge
        memo_key = (current_map, destination_map, goal_tiles, self._world, self._flags_progress(),
                    tuple(sorted(completed)), len(self._tabu_transitions))
        memo = self._no_route_memo.get(memo_key)
        if memo is not None and memo[0] > frame:
            # An exhaustive failed search costs seconds; never repeat it every
            # step. Any new persistent flag (an event done) invalidates it.
            self._pending_unlock = memo[1]
            self._route_commitment = None
            return None
        route = self._plan_route(current_map, destination_map, coords, completed, hm_usable, goal_tiles)
        if route is None and any(not barrier.explained for barrier in self.pushback.barriers.values()):
            # An unexplained refusal may itself be the mistake that walls the
            # agent in: forget those before concluding that no way exists.
            forgotten = self.pushback.forget_unexplained()
            self.topology.set_learned_gate_edges(self.pushback.active_edges(self._world, frame))
            self._lessons_dirty = True
            self._record("barrier_forgotten", count=forgotten, reason="no route while unexplained refusals were active")
            route = self._plan_route(current_map, destination_map, coords, completed, hm_usable, goal_tiles)
        if route is None and self._tabu_transitions:
            # A temporary exit ban must never be the reason no route exists.
            for edge_id in list(self._tabu_transitions):
                self.topology.unblock_transition(edge_id)
            self._tabu_transitions.clear()
            self._record_throttled("transition_tabu_lifted", leg, 1800, reason="no route while exits were banned")
            route = self._plan_route(current_map, destination_map, coords, completed, hm_usable, goal_tiles)
        if route is None and self.native and self.topology.static_gates:
            # Every way is closed. A guard that wants an item or a flag, or a
            # queue of story actors that some event dismisses, is a sub-goal:
            # find the event and do it first. Only gates (never standing NPCs)
            # are then tried anyway, so a wrong analysis is corrected by a real
            # refusal that is learned.
            self.topology.set_static_gates(False)
            self.topology.set_story_actors(False)
            try:
                what_if = self._plan_route(current_map, destination_map, coords, completed, hm_usable, goal_tiles)
            finally:
                self.topology.set_static_gates(True)
                self.topology.set_story_actors(True)
            if what_if:
                self._pending_unlock = self._gate_unlock_directive(what_if, completed, coords)
                if self._pending_unlock is None:
                    self.topology.set_static_gates(False)
                    try:
                        relaxed = self._plan_route(current_map, destination_map, coords, completed, hm_usable,
                                                   goal_tiles)
                    finally:
                        self.topology.set_static_gates(True)
                    if relaxed:
                        if self._gate_relaxed != leg:
                            self._record("gate_relaxed", source=list(current_map), destination=list(destination_map),
                                         reason="no gated route; attempting the closed passage to verify it")
                        self._gate_relaxed = leg
                        route = relaxed
        elif route is not None and self._gate_relaxed == leg:
            self._gate_relaxed = ""
        best: WarpTransition | None = route[0] if route else None
        if best is None:
            if len(self._no_route_memo) > 256:
                self._no_route_memo.clear()
            self._no_route_memo[memo_key] = (frame + 900, self._pending_unlock)
            self._route_commitment = None
            return None
        self._route_commitment = RouteCommitment(current_map, destination_map, best, frame + _ROUTE_COMMITMENT_FRAMES)
        if commitment is None or not same_leg or commitment.edge != best:
            self._record_throttled("route_committed", f"{current_map}->{destination_map}@{best.source_xy}", 900,
                                   source=list(current_map), destination=list(destination_map),
                                   exit=list(best.source_xy), via=list(best.destination), exit_kind=best.kind)
        return best

    def _plan_route(self, current_map: tuple[int, int], destination_map: tuple[int, int],
                    coords: Mapping[str, int], completed: frozenset[str], hm_usable: frozenset[str],
                    goal_tiles: frozenset[tuple[int, int]]) -> list[WarpTransition] | None:
        """Full route (first exit first); tile-aware when the goal tiles are known."""
        route: list[WarpTransition] | None = None
        if goal_tiles and self.native and self.topology.catalog is not None:
            route = self.topology.find_route_to_tile(current_map, destination_map, goal_tiles, completed,
                                                     (coords["x"], coords["y"]), self.bridge, hm_usable)
        if not route:
            route = self.topology.find_route(current_map, destination_map, completed, (coords["x"], coords["y"]),
                                             self.bridge, hm_usable=hm_usable) or route
        return route if route else None

    def _story_blockers(self, route: list[WarpTransition], coords: Mapping[str, int]) -> list[tuple[str, str, str]]:
        """Visible flagged story actors standing on a what-if route's walking path.

        The flag that hides such an actor (Team Aqua's museum queue hides once
        Dock at the shipyard is visited) is the requirement to clear the way.
        """
        catalog = self.topology.catalog
        if catalog is None:
            return []
        world = FlagSnapshot(self.bridge)
        names: dict[int, list[str]] = {}
        for name, value in catalog.constants.items():
            if name.startswith("FLAG_HIDE_") or name.startswith("FLAG_"):
                names.setdefault(int(value), []).append(name)
        needs: list[tuple[str, str, str]] = []
        position = (coords["x"], coords["y"])
        for index, edge in enumerate(route):
            key = edge.source
            entry = position if index == 0 else route[index - 1].destination_xy
            raw = catalog.maps.get(key)
            if raw is None:
                continue
            packed = catalog.grid(key)
            if not (0 <= entry[1] < packed.shape[0] and 0 <= entry[0] < packed.shape[1]):
                continue
            planner = GbaConstrainedAStar(cast(Any, (packed >> 10) & 3), cast(Any, packed >> 12),
                                          catalog.behavior_grid(key), self.game)
            model = obstacle_model(self.game, key, bridge=world, story_actors=False, static_gates=False)
            planner.add_static_obstacles(model.blocked, model.removable_costs)
            origin = (entry[0], entry[1], int(packed[entry[1], entry[0]]) >> 12)
            path: list[tuple[int, int]] | None = None
            for tile in self.topology._approach_tiles(edge):
                path = planner.find_optimal_path(origin, tile)
                if path:
                    break
            if not path:
                continue
            on_path = set(path)
            for template in template_objects(self.game, key):
                if (template.flag is None or template.temporary or template.kind != "object" or template.mobile or
                        template.xy not in on_path or world.get_event_flag(template.flag)):
                    continue
                for name in names.get(template.flag, []):
                    needs.append(("flag", name, str(raw["name"])))
        return list(dict.fromkeys(needs))

    def _gate_requirements(self, route: list[WarpTransition],
                           coords: Mapping[str, int] | None = None) -> list[tuple[str, str, str]]:
        """(kind, name, map) for flags/items that open closed gates or move story actors."""
        world = FlagSnapshot(self.bridge)
        maps = [edge.source for edge in route] + [route[-1].destination]
        needs: list[tuple[str, str, str]] = []
        if coords is not None:
            needs.extend(self._story_blockers(route, coords))
        for key in dict.fromkeys(maps):
            for gate in gates_by_map(self.game).get(key, ()):
                if gate_state(gate, world) is not True:
                    continue
                for flag in unlocking_flags(gate):
                    needs.append(("flag", flag, gate.map_name))
                for conjunction in gate.pass_conditions:
                    items = [literal for literal in conjunction if literal[0] == "cmp" and literal[1][0] == "result"
                             and literal[1][1] == "item"]
                    if len(conjunction) == 1 and items:
                        left, right, accepted = items[0][1], items[0][2], items[0][3]
                        if (1 in accepted) == (right[0] == "const" and int(right[1]) == 1):
                            needs.append(("item", str(left[2]), gate.map_name))
        return list(dict.fromkeys(needs))

    def _gate_unlock_directive(self, route: list[WarpTransition], completed: frozenset[str],
                               coords: Mapping[str, int] | None = None) -> dict[str, Any] | None:
        """Means-ends step: the event that sets a gate's missing flag or gives its item."""
        catalog = self.topology.catalog
        if catalog is None:
            return None
        coords = coords if coords is not None else self.bridge.get_player_coords_and_map()
        current_map = coords["map_group"], coords["map_num"]
        for kind, name, gate_map in self._gate_requirements(route, coords):
            providers = self._providers(kind, name)
            for target in providers:
                key = (int(target["map_group"]), int(target["map_num"]))
                goal = self._goal_tiles({"npc_local_id": target["npc_local_id"], "action": "interact",
                                         "dest": (*key, (target["x"], target["y"]))}, (int(target["x"]), int(target["y"])))
                path = (self._plan_route(current_map, key, coords, completed, self._hm_usable(), goal)
                        if key != current_map else [])
                if path is None:
                    continue
                return {"mode": "UNLOCK_GATE", "dest": (*key, (int(target["x"]), int(target["y"]))),
                        "action": "interact", "script": target["script"], "npc_local_id": target["npc_local_id"],
                        "map_name": target["map_name"], "milestone": self._last_directive.get("milestone", ""),
                        "reason": f"{gate_map} requires {name}", "requirement": name}
        return None

    def _providers(self, kind: str, name: str) -> list[dict[str, Any]]:
        """Interactable source events whose scripts set a flag or give an item."""
        cache_key = f"{kind}:{name}"
        cached = self._provider_cache.get(cache_key)
        if cached is not None:
            return cached
        catalog = self.topology.catalog
        found: list[dict[str, Any]] = []
        if catalog is not None:
            for raw in catalog.maps.values():
                for target in catalog.targets(str(raw["name"])):
                    if target["action"] != "interact":
                        continue
                    effect = catalog.effects(str(target["script"]))
                    if (kind == "flag" and name in effect.set_flags) or (kind == "item" and name in effect.items):
                        found.append(target)
        self._provider_cache[cache_key] = found[:8]
        return self._provider_cache[cache_key]

    def _start_detour(self, reason: str) -> bool:
        """Walk to a random reachable tile to break a deterministic loop."""
        if not self.native or self.topology.catalog is None:
            return False
        coords = self.bridge.get_player_coords_and_map()
        current_map = (coords["map_group"], coords["map_num"])
        walking = self.topology.live_walking_costs(self.bridge, hm_usable=self._hm_usable())
        if not walking:
            return False
        position = (coords["x"], coords["y"])
        exits = {edge.source_xy for edge in self.topology.adjacency.get(current_map, [])}
        pool = sorted(xy for xy, cost in walking.items() if 6 <= cost <= 40 and xy not in exits and xy != position)
        if not pool:
            pool = sorted(xy for xy, cost in walking.items() if cost >= 1 and xy not in exits and xy != position)
        if not pool:
            return False
        target = self.rng.choice(pool)
        self._detour = (current_map, target, self.bridge.frame_count + 900)
        self.metrics.detours += 1
        self._record("exploration_detour", target=list(target), map=list(current_map), reason=reason)
        return True

    def _field_move_keys(self, origin: tuple[int, int], step: tuple[int, int],
                         objects: list[ObjectEventStruct], player_id: int) -> int | None:
        """Cut, Rock Smash, Strength, Surf and Waterfall through native field scripts.

        None means an ordinary step is fine; otherwise the field controller faces
        the target and pulses A, and the script's own Yes/No is answered by it.
        """
        assert self.field is not None
        if abs(step[0] - origin[0]) + abs(step[1] - origin[1]) != 1:
            return None
        actor = next((obj for index, obj in enumerate(objects) if index != player_id and int(obj.flags) & 1 and
                      (int(obj.currentCoords_x) - 7, int(obj.currentCoords_y) - 7) == step), None)
        if actor is not None and live_obstacle_kind(self.game, actor) == "item" and self.bridge.is_hardware_ready():
            # An item ball on the path: face it and pick it up, then walk on.
            dx, dy = step[0] - origin[0], step[1] - origin[1]
            direction = 4 if dx > 0 else (3 if dx < 0 else (1 if dy > 0 else 2))
            if int(objects[player_id].facingDirection) & 15 != direction:
                return {1: self.bridge.KEY_DOWN, 2: self.bridge.KEY_UP,
                        3: self.bridge.KEY_LEFT, 4: self.bridge.KEY_RIGHT}[direction]
            self._record_throttled("item_pickup", f"{step}", 600, target=list(step))
            return self.bridge.KEY_A if self.bridge.frame_count % 8 < 2 else 0
        behaviors = self.bridge.extract_behavior_map()
        if not 0 <= step[1] < behaviors.shape[0] or not 0 <= step[0] < behaviors.shape[1]:
            return None
        keys = self.field.choose_transition_key(origin, step, int(behaviors[step[1], step[0]]), actor)
        if keys:
            if keys == self.bridge.KEY_A:
                self.metrics.field_moves += 1
            self._record_throttled("field_move", f"{step}", 600, target=list(step),
                                   actor_graphics=None if actor is None else int(actor.graphicsId))
        return keys

    def _execute_overworld_cycle(self) -> None:
        if self.native:
            # Every overworld frame, including script frames: a guard walking the
            # avatar back after a step is a refused passage, not a random wobble.
            self._observe_pushback(self.bridge.get_player_coords_and_map())
            self._observe_objective_talk()
            if self._mart_cycle():
                return
        # An interrupted battle/field operation may leave a native menu open.
        # Never feed path directions into its cursors or mistake those frames
        # for movement. HM teaching retains ownership of its requested UI.
        if self.native and (self.field is None or not self.field.busy):
            recovery_key = self.menu.recovery_key()
            if recovery_key is not None:
                self.movement.clear()
                self.bridge.set_hardware_keys(recovery_key)
                return
        if self.field is not None and not self.field.busy:
            # A walking NPC can pass beside the player between tile plans.
            # Intercept that real interaction window before continuing a stale
            # movement command; chasing its latest coordinate never guarantees
            # that the old frame%12 confirmation window will coincide with it.
            coords_now = self.bridge.get_player_coords_and_map()
            map_now = coords_now["map_group"], coords_now["map_num"]
            local_id = 0
            active = self._active_transition
            if active is not None and active.kind == "script" and active.source == map_now:
                local_id = active.npc_local_id
                self.field.menu_choice = active.menu_choice
            else:
                destination = self._last_directive.get("dest")
                if isinstance(destination, (tuple, list)) and tuple(destination[:2]) == map_now:
                    local_id = int(self._last_directive.get("npc_local_id", 0))
                    self.field.menu_choice = int(self._last_directive.get("menu_choice", 0))
            milestone = self._last_directive.get("milestone")
            task = next((task for task in self.planner.campaign_dag if task.milestone_id == milestone), None)
            if local_id and task is not None and task.complete(self.bridge, self.bridge.get_decrypted_player_party()):
                # Finishing an NPC's script must give the planner its next turn.
                # Otherwise this fast interception can reopen the same dialogue
                # before the normal milestone observation below ever executes.
                local_id = 0
                self._directive_signature = ()
            intercept = self.field.patrol_interaction_key(local_id) if local_id else None
            if intercept is not None:
                self.movement.clear()
                self.bridge.set_hardware_keys(intercept)
                return
        event = self.movement.tick()
        if event is not None:
            self.metrics.moves += event.status == "arrived"
            self.metrics.blocked_steps += event.status == "blocked"
            self._record("movement", movement=asdict(event))
            if event.status == "blocked":
                for edge in self.movement.blocked_edges:
                    self._blocked_at.setdefault(edge, self.bridge.frame_count)
            elif event.status == "arrived":
                self._note_arrival(event.destination_map, event.destination)
            if (event.status in ("arrived", "interrupted") and event.origin_map == event.destination_map and
                    event.origin != event.destination):
                self.pushback.note_step(event.origin_map, event.origin, event.destination, self.bridge.frame_count)
        if self.movement.pending:
            return
        self.bridge.set_hardware_keys(0)
        # Script locks and moving actors are temporary obstructions. Static
        # walls already come from live tiles; a stale input timeout is not a wall.
        for edge in tuple(self.movement.blocked_edges):
            if self.bridge.frame_count - self._blocked_at.get(edge, 0) >= 600:
                self.movement.blocked_edges.discard(edge)
                self._blocked_at.pop(edge, None)
        self._expire_tabu()
        if self.field is not None:
            field_keys = self.field.poll_key_mask()
            if field_keys is not None:
                self.bridge.set_hardware_keys(field_keys)
                return
        if not self.bridge.is_hardware_ready():
            self.bridge.set_hardware_keys(self.bridge.KEY_A if self.bridge.frame_count % 12 == 0 else 0)
            return
        if self.field is not None:
            if self.field.last_result in ("options_set", "options_timeout") and self._options_reported != self.field.last_result:
                self._options_reported = self.field.last_result
                self._record("options_result", result=self.field.last_result, values=list(self.field.options_values()))
            if self.field.options_due():
                # Fast text, no battle animations and SET style roughly halve
                # the frames per battle; the change goes through the Option menu.
                self.field.request_options()
                self._options_reported = ""
                self._record("options_setup", target={"text_speed": "fast", "battle_scene": "off", "battle_style": "set"})
                return
        if self.native and self.topology.catalog is not None:
            settled = self.bridge.get_player_coords_and_map()
            raw_map = self.topology.catalog.maps.get((settled["map_group"], settled["map_num"]))
            if raw_map is not None and not (0 <= settled["x"] < int(raw_map["width"]) and
                                            0 <= settled["y"] < int(raw_map["height"])):
                # Mid-connection frame: the new map's coordinates are not settled.
                # Planning now would cache a directive built from a phantom tile
                # for the whole map visit (no route to the nurse -> no heal).
                self.bridge.set_hardware_keys(0)
                return
        party = self.bridge.get_decrypted_player_party()
        self._refresh_roster(party)
        self.planner.preparation.observe(self.bridge.frame_count, self.metrics.wins)
        completed = self.planner.completed_milestones(self.bridge, party)
        if self.native:
            self._refresh_world(completed)
            self._save_lessons()
        current_coords = self.bridge.get_player_coords_and_map()
        # Story, healing and training intents depend on the map and roster;
        # their tile path is replanned separately below. Rebuilding every
        # source-map route whenever tileTransitionState changes can consume
        # seconds per VBlank. Puzzle directives alone need tile-level intent.
        intent_position: tuple[int, ...] = (current_coords["map_group"], current_coords["map_num"])
        if self._directive_cache.get("puzzle"):
            intent_position += (current_coords["x"], current_coords["y"], current_coords["z"],
                                self.bridge.frame_count // 120)
        signature: tuple[object, ...] = (
            intent_position, tuple(sorted(completed)), self.metrics.losses,
            tuple((mon["pid"], mon["hp"], mon["maxHP"], mon["level"], mon["status"],
                   tuple(mon["moves"]), tuple(mon["pp"])) for mon in party),
            tuple(sorted(self.bridge.get_inventory().items())))
        if signature != self._directive_signature:
            self._directive_cache = self.planner.get_active_directive(self.bridge, party)
            self._directive_signature = signature
        directive = dict(self._directive_cache)
        if self.native:
            prerequisite = self._objective_requirement_directive(directive, completed)
            if prerequisite is not None:
                directive = dict(prerequisite)
        self._last_directive = directive
        if self.mart is not None:
            if directive.get("mode") == "SHOP" and self.mart.need is None:
                need_raw = directive.get("shop_need", [0, 0])
                lead = max(party, key=lambda mon: (int(mon["level"]), int(mon["maxHP"])), default=None)
                self.mart.prepare(ShoppingNeed(int(need_raw[0]), int(need_raw[1]), str(directive.get("reason", ""))),
                                  int(lead["maxHP"]) if lead is not None else 60)
            elif directive.get("mode") != "SHOP" and self.mart.need is not None and not self._shopping:
                self.mart.reset()
            self.planner.wants_balls = len([mon for mon in party if not mon.get("is_egg", False)]) < 6
        if self.field is not None:
            self.field.menu_choice = int(directive.get("menu_choice", 0))
            if directive.get("mode") == "TEACH_FIELD_MOVE":
                move_ids = cast(list[int], directive["moves"])
                if move_ids and self.field.request_required_move(move_ids[0]):
                    self.bridge.set_hardware_keys(self.field.poll_key_mask() or 0)
                else:
                    self._record_throttled("campaign_blocked", "required field move has no compatible stocked HM", 1800,
                                           reason="required field move has no compatible stocked HM", moves=move_ids)
                return
        if directive.get("mode") == "OPTIMAL_TRAINING_CURRICULUM" and not directive.get("fixed_route"):
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
        if newly_completed:
            # Story progress: earlier identical-stall history no longer applies.
            self._stall_memory.clear()
            self._rewind_memory.clear()
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
        position = (coords["x"], coords["y"])
        if self._detour is not None:
            detour_map, detour_target, until = self._detour
            if detour_map != current_map or self.bridge.frame_count >= until or position == detour_target:
                self._detour = None
            else:
                directive = {"mode": "EXPLORE_DETOUR", "dest": (*detour_map, detour_target), "action": "walk",
                             "milestone": directive.get("milestone", ""), "reason": "break a repeated state"}
                self._last_directive = directive
        dest = directive.get("dest")
        if dest is None:
            reason = str(directive.get("reason", "directive has no destination"))
            self._record_throttled("campaign_blocked", reason, 1800, reason=reason,
                                   directive=self._compact_directive(directive))
            return
        destination_map = (int(dest[0]), int(dest[1]))
        target = (int(dest[2][0]), int(dest[2][1]))
        hm_usable = self._hm_usable()
        surfing = bool(self.bridge.read_u8(self.bridge.memory_map.avatar) & 8)
        allow_surf = bool(directive.get("allow_surf", False)) or surfing or "surf" in hm_usable
        if self._puzzle_engaged is not None and self._puzzle is not None:
            engaged_map, engaged_milestone = self._puzzle_engaged
            if (engaged_map != current_map or engaged_milestone != str(directive.get("milestone", "")) or
                    self._switches_pending(current_map)):
                # Floor switches are explored systematically first (below).
                self._puzzle_engaged = None
                self._puzzle.clear()
            else:
                result = self._puzzle.tick(target, allow_surf=allow_surf, allow_interaction_fallback=True)
                if result.status != "waiting" or result.reason != self._puzzle.last_reason:
                    self._record_throttled("puzzle_step", f"{result.status}:{result.reason}", 300,
                                           status=result.status, reason=result.reason)
                if result.status == "keys" and result.keys is not None:
                    self.bridge.set_hardware_keys(result.keys)
                    return
                if result.status == "waiting":
                    self.bridge.set_hardware_keys(result.keys or 0)
                    return
                if result.status == "directive" and result.directive is not None:
                    directive = dict(result.directive, milestone=directive.get("milestone", ""))
                    self._last_directive = directive
                    dest = directive["dest"]
                    destination_map = (int(dest[0]), int(dest[1]))
                    target = (int(dest[2][0]), int(dest[2][1]))
                elif result.status in ("complete", "unavailable", "blocked"):
                    self._puzzle_engaged = None
                    self._puzzle.clear()
        goal_tiles = self._goal_tiles(directive, target)
        transition: WarpTransition | None = None
        if current_map != destination_map:
            self._reentry = None
            self._pending_unlock = None
            transition = self._choose_transition(current_map, destination_map, coords, completed, hm_usable,
                                                 allow_surf, goal_tiles)
            unlock = self._pending_unlock
            if transition is None and unlock is not None:
                # A closed gate stands on every route: first obtain what it asks for.
                self._pending_unlock = None
                self._record_throttled("gate_requirement", str(unlock["requirement"]), 3600,
                                       requirement=unlock["requirement"], reason=unlock["reason"],
                                       target=unlock["map_name"], milestone=unlock.get("milestone", ""))
                directive = unlock
                self._last_directive = directive
                unlock_dest = directive["dest"]
                destination_map = (int(unlock_dest[0]), int(unlock_dest[1]))
                target = (int(unlock_dest[2][0]), int(unlock_dest[2][1]))
                goal_tiles = self._goal_tiles(directive, target)
                if current_map != destination_map:
                    transition = self._choose_transition(current_map, destination_map, coords, completed,
                                                         hm_usable, allow_surf, goal_tiles)
                    self._pending_unlock = None
                    if transition is None:
                        self._record_throttled("campaign_blocked", "gate requirement unreachable", 1800,
                                               reason="gate requirement unreachable",
                                               directive=self._compact_directive(directive))
                        return
                    target = transition.source_xy
            elif transition is None:
                self._record_throttled("campaign_blocked", "no source-defined map route", 1800,
                                       reason="no source-defined map route", directive=self._compact_directive(directive))
                return
            else:
                target = transition.source_xy
        elif self._reentry is not None:
            reentry_map, reentry_edge, until = self._reentry
            if reentry_map == current_map and self.bridge.frame_count < until:
                transition, target = reentry_edge, reentry_edge.source_xy
            else:
                self._reentry = None
        if self.native and transition is not None and transition.panel_xy is not None:
            transition = self._elevator_step(transition)
            target = transition.source_xy
        self._active_transition = transition
        if self.field is not None and transition is not None and transition.kind == "script":
            self.field.menu_choice = transition.menu_choice
        _, collisions, elevations = self.bridge.extract_live_map_matrices()
        astar = GbaConstrainedAStar(collisions, elevations, self.bridge.extract_behavior_map(), self.game,
                                  allow_surf=allow_surf)
        astar.terminal_tiles = {edge.source_xy for edge in self.topology.adjacency.get(current_map, [])
                                if edge.kind == "warp"}
        objects = self.bridge.get_object_events()
        player_id = self.bridge.read_u8(self.bridge.memory_map.avatar + 5)
        if self.native:
            # Occupancy from live actors; sight costs from templates of trainers
            # not yet defeated, so the plan cannot flip as they (de)spawn.
            astar.register_object_events(objects, player_id, penalty=0.0)
            runtime_positions = {template.local_id: template.xy for template in
                                 runtime_template_objects(self.bridge, self.game, current_map) or ()}
            for sight in undefeated_sights(self.game, current_map, self.bridge, runtime_positions):
                for direction in sight.directions:
                    astar.register_trainer_cone(sight.xy[0], sight.xy[1], direction, sight.sight)
        else:
            astar.register_object_events(objects, player_id)
        gate_edges: frozenset[tuple[tuple[int, int], tuple[int, int]]] = frozenset()
        learned_here = self.topology.learned_gate_edges.get(current_map, frozenset())
        if self.native:
            # Off-camera trees, rocks and story actors still block, exactly as
            # the route search assumes; usable HM obstacles become clearable.
            # Armed guard/sandstorm gates and learned refusals forbid the steps
            # their scripts would undo, again identically to the route search.
            relaxed = self._gate_relaxed == f"{current_map}->{destination_map}"
            obstacles = obstacle_model(self.game, current_map, bridge=self.bridge, live_objects=objects,
                                       player_object_id=player_id, hm_usable=hm_usable, current_map=True,
                                       learned_edges=learned_here,
                                       static_gates=self.topology.static_gates and not relaxed)
            astar.add_static_obstacles(obstacles.blocked, obstacles.removable_costs)
            gate_edges = obstacles.gate_edges
        astar.blocked_edges = {((x, y), (tx, ty)) for group, number, x, y, tx, ty in self.movement.blocked_edges
                               if (group, number) == current_map}
        if self.native:
            # Floor switches reshape the room when stepped on: never cross one
            # by accident (two switches on a path flip the barriers back and
            # forth forever); stepping on one is a deliberate exploration act.
            astar.add_static_obstacles(set(self._floor_switches(current_map)) - {position})
        ungated_edges = set(astar.blocked_edges) | set(learned_here)
        astar.blocked_edges.update(gate_edges)
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
            elif transition is None:
                # Off camera: map scripts may have moved the actor (setobjectxyperm).
                runtime = runtime_template_objects(self.bridge, self.game, current_map) or ()
                moved = next((template for template in runtime if template.local_id == local_id), None)
                if moved is not None:
                    target = moved.xy
            candidates: list[tuple[list[tuple[int, int]], tuple[int, int], int]] = []
            behaviors = self.bridge.extract_behavior_map()
            for dx, dy, direction in ((0, 1, 2), (0, -1, 1), (1, 0, 3), (-1, 0, 4)):
                adjacent = (target[0] + dx, target[1] + dy)
                if (0 <= adjacent[1] < behaviors.shape[0] and 0 <= adjacent[0] < behaviors.shape[1] and
                        int(behaviors[adjacent[1], adjacent[0]]) == 0x80):
                    adjacent = (target[0] + 2 * dx, target[1] + 2 * dy)
                if adjacent == position:
                    # Already adjacent: no path needed, only facing and A.
                    candidates.append(([], adjacent, direction))
                    continue
                candidate = astar.find_optimal_path((coords["x"], coords["y"], coords["z"]), adjacent)
                if candidate is not None:
                    candidates.append((candidate, adjacent, direction))
            if candidates:
                # Ties break on tile order, so an equal-length choice cannot flip.
                path, target, facing = min(candidates, key=lambda entry: (len(entry[0]), entry[1]))
        if position == target:
            if self._detour is not None:
                self._detour = None
                return
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
                # Step-triggered warps (ladders, cave holes) fire only when a step
                # lands on them; after arriving on one, step off and back on.
                for dx, dy in ((0, 1), (1, 0), (-1, 0), (0, -1)):
                    beside = (position[0] + dx, position[1] + dy)
                    if beside in astar.terminal_tiles:
                        continue
                    if astar.find_optimal_path((coords["x"], coords["y"], coords["z"]), beside) == [beside]:
                        self.movement.move_toward(beside, "walk")
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
        if not path and self.native and gate_edges - learned_here:
            # Only source-derived gates separate us from the target. Verify by
            # trying; a real refusal is then learned and never repeated.
            gated_edges = set(astar.blocked_edges)
            astar.blocked_edges = set(ungated_edges)
            path = astar.find_optimal_path((coords["x"], coords["y"], coords["z"]), target)
            if path:
                self._record_throttled("gate_relaxed", f"{current_map}:{target}", 3600, source=list(current_map),
                                       destination=list(current_map), target=list(target),
                                       reason="no gated path on this map; verifying the closed passage")
            else:
                astar.blocked_edges = gated_edges
        if not path and self.native and learned_here and self.pushback.forget_unexplained(current_map):
            # Unexplained refusals around the avatar left no step at all: they
            # were the mistake. Forget them and replan from fresh state.
            self.topology.set_learned_gate_edges(self.pushback.active_edges(self._world, self.bridge.frame_count))
            self._lessons_dirty = True
            self._record("barrier_forgotten", map=list(current_map), reason="refusals left no walkable step")
            return
        if not path and transition is None and self.native and self.topology.catalog is not None:
            # The target may lie in another section of this map (caves): leave
            # through the exit whose onward route re-enters next to the target.
            reentry_key = ("reentry", current_map, target, self._world, self._flags_progress(), tuple(sorted(completed)))
            route = None
            reentry_memo = self._no_route_memo.get(reentry_key)
            if reentry_memo is None or reentry_memo[0] <= self.bridge.frame_count:
                route = self.topology.find_route_to_tile(current_map, current_map, self._goal_tiles(directive, target),
                                                         completed, position, self.bridge, hm_usable)
                if not route:
                    self._no_route_memo[reentry_key] = (self.bridge.frame_count + 600, None)
            if route:
                self._reentry = (current_map, route[0], self.bridge.frame_count + 3600)
                self._record_throttled("section_reentry", f"{current_map}:{target}", 1800, target=list(target),
                                       exit=list(route[0].source_xy), via=list(route[0].destination), maps=len(route))
                return
            switch = self._next_floor_switch(current_map, coords, astar)
            if switch is not None:
                # Floor switches (Mauville's barriers): try one we have not yet
                # tried in this barrier layout, then re-plan from the new map.
                path, target = switch
            elif self._puzzle is not None and directive.get("puzzle") and self._puzzle_engaged is None:
                self._puzzle.clear()
                self._puzzle_engaged = (current_map, str(directive.get("milestone", "")))
                self._record("puzzle_engaged", puzzle=str(directive.get("puzzle")), target=list(target))
                return
        if not path:
            commitment = self._route_commitment
            if transition is not None and commitment is not None and commitment.edge == transition:
                # The live map disagrees with the commitment: drop it and try another exit.
                self._route_commitment = None
                self._tabu(transition, "committed exit has no live collision path", goal=destination_map)
            self._record_throttled("campaign_blocked", "live collision path unavailable", 1800,
                                   reason="live collision path unavailable", directive=self._compact_directive(directive),
                                   position=coords, target=list(target))
            return
        if self.field is not None and self.native:
            field_keys = self._field_move_keys(position, path[0], objects, player_id)
            if field_keys is not None:
                self.bridge.set_hardware_keys(field_keys)
                return
        mode = self._plan_movement(path, target)
        self.movement.move_toward(path[0], mode)

    def _request_battle_command(self, decision: BattleDecision, me: BattlePokemonStruct) -> None:
        """Submit a decision; an engine-rejected request degrades to the strongest attack."""
        try:
            self.menu.request_command(cast(CommandKind, decision.action), index=decision.index,
                                      item=decision.item or 4, target=decision.target)
        except (ValueError, RuntimeError) as error:
            if not self.fault_tolerant or str(error) == "Party contains an invalid checksum":
                raise
            self._note_fault(error, "battle_command")
            self.menu.cancel()
            fallback = self._fallback_battle_decision(me, f"{decision.action} rejected")
            self._battle_decision = fallback
            self.menu.request_command("move", index=fallback.index)

    def _decide_battle(self, me: BattlePokemonStruct, enemy: BattlePokemonStruct) -> BattleDecision:
        try:
            return self.choose_battle_decision()
        except (ValueError, ArithmeticError, RuntimeError, KeyError, IndexError) as error:
            if not self.fault_tolerant or str(error) == "Party contains an invalid checksum":
                raise
            self._note_fault(error, "battle_decision")
            return self._audit_decision(self._fallback_battle_decision(me, type(error).__name__), me, enemy)

    def _forced_replacement(self) -> int | None:
        """First healthy, non-active party member, used only when the policy faults.

        Active battlers come from gBattlersCount/gBattlerPositions: in singles
        gBattlerPartyIndexes[2] is stale and must not exclude a valid reserve.
        """
        party = player_party_in_battle_order(self.bridge)
        active: set[int] = set()
        if all(name in self._symbols for name in ("gBattlersCount", "gBattlerPositions", "gBattlerPartyIndexes")):
            count = self.bridge.read_u8(self._symbols["gBattlersCount"].address)
            if count in (2, 4):
                positions = self.bridge.read_memory(self._symbols["gBattlerPositions"].address, count)
                for slot, position in enumerate(positions):
                    index = self.bridge.read_u16(self._symbols["gBattlerPartyIndexes"].address + 2 * slot)
                    if position in (0, 2) and index < len(party) and int(party[index]["hp"]) > 0:
                        active.add(index)  # a living partner in doubles; a fainted lead needs no exclusion
        return next((index for index, mon in enumerate(party)
                     if index not in active and int(mon["hp"]) > 0 and not mon.get("is_egg", False)), None)

    def _execute_combat_cycle(self, is_trainer_battle: bool) -> None:
        """Keep one decision through the cursor FSM; never advance nested frames."""
        del is_trainer_battle
        observed_menu = self.menu.observe_menu()
        mode = observed_menu.mode
        me = self.bridge.get_battle_pokemon(observed_menu.battler)
        enemy = self.bridge.get_battle_pokemon(1)
        if self.native and mode == "party" and self._battle_decision is None and not self.menu.busy:
            # A voluntarily opened Party is not a forced replacement. Picking
            # the first healthy slot can select the active mon indefinitely.
            if not me.hp:
                decision = self._decide_battle(me, enemy)
                if decision.action != "switch":
                    replacement = self._forced_replacement() if self.fault_tolerant else None
                    if replacement is None:
                        self.bridge.set_hardware_keys(self.menu.tick())
                        return
                    decision = BattleDecision("switch", replacement, 0, 1.0, decision.proof,
                                              "forced replacement: first healthy reserve")
                self._battle_decision = decision
                try:
                    self.menu.request_switch(decision.index)
                except (ValueError, RuntimeError) as error:
                    if not self.fault_tolerant or str(error) == "Party contains an invalid checksum":
                        raise
                    self._note_fault(error, "forced_replacement")
                    self._battle_decision = None
                    self.menu.cancel()
                    self.bridge.set_hardware_keys(self.bridge.KEY_A if self.bridge.frame_count % 8 < 2 else 0)
                    return
                self.bridge.set_hardware_keys(self.menu.tick())
                return
        if self.native and mode not in {"action", "move"}:
            if self.menu.pending is None:
                self._battle_decision = None
            self.bridge.set_hardware_keys(self.menu.tick())
            return
        if not me.species or not enemy.species or not me.level or not enemy.level:
            self.bridge.set_hardware_keys(self.menu.tick())
            return
        signature = (int(me.hp), int(enemy.hp), *[int(pp) for pp in me.pp])
        if mode == "action" and not self.menu.busy and self.menu.pending is None:
            if self._last_battle_signature == signature and self._battle_decision is not None:
                self._battle_decision = None
        if self._battle_decision is None:
            self._battle_decision = self._decide_battle(me, enemy)
            self._last_battle_signature = signature
            self._request_battle_command(self._battle_decision, me)
        self.bridge.set_hardware_keys(self.menu.tick())
        if signature != self._last_battle_signature:
            self._battle_decision = None
            self.menu.cancel()
            self._last_battle_signature = signature

    def _select_frame_input(self) -> None:
        """Observe one stable snapshot and select keys without advancing frames."""
        if self._probe_keys:
            self.scene = "recovery_probe"
            self.bridge.set_hardware_keys(self._probe_keys.pop(0))
            return
        if self._ui is not None and not self._bootstrapped:
            self.scene = "bootstrap"
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
        if self.field is not None:
            # Evolution (including post-battle evolution while gMain.inBattle is
            # still set) owns input before any battle or field policy.
            scene_keys = self.field.scene_keys()
            if scene_keys is not None:
                if self.scene != "evolution":
                    self._record("scene_started", scene="evolution")
                    self.movement.clear()
                    self.menu.cancel()
                    self._battle_decision = None
                self.scene = "evolution"
                self.bridge.set_hardware_keys(scene_keys)
                return
        self._process_disclosures()
        in_battle, trainer = self.bridge.is_in_battle()
        if in_battle and not self._in_battle:
            self.metrics.battles += 1
            self.cache = self.battle_policy.initial_state(1)
            self._battle_decision = None
            self._last_enemy_move = 0
            self._battle_turns = 0
            self._battle_escape_attempts = 0
            self._battle_heals = 0
            self._native_enemy_hp.clear()
            self._native_own_pp.clear()
            self._native_interventions.clear()
            # Observed immunities persist across battles (and in lessons.json):
            # Mud Shot into the same Taillow again would repeat a known mistake.
            self.movement.clear()
            self._detour = None
            self._battle_start = self._battle_readiness_snapshot(trainer)
            self._battle_roster, self._battle_move_types = {}, {}
            self._last_trainer_roster = ({}, {})
            self._own_hp_seen.clear()
            self._record("battle_entered", trainer=trainer,
                         prepared=self._battle_start.get("prepared"),
                         lead_hp=self._battle_start.get("lead_hp_fraction"),
                         attack_pp=self._battle_start.get("attack_pp_fraction"))
        if not in_battle and self._in_battle:
            self._battle_decision = None
            self.menu.cancel()
            self.cache = self.battle_policy.initial_state(1)
            if self.native:
                outcome = self.bridge.read_u8(self._symbols["gBattleOutcome"].address)
                self.metrics.wins += outcome == 1
                self.metrics.losses += outcome in (2, 9)
                self.metrics.captures += outcome == 7
                self._record("native_battle_result", outcome=outcome, turns=self._battle_turns)
                if self._battle_roster:
                    self._last_trainer_roster = (dict(self._battle_roster),
                                                 {species: set(types) for species, types in self._battle_move_types.items()})
                self.shared.save()
                if outcome in (2, 9):
                    self._record_defeat()
        self._in_battle = in_battle
        if in_battle:
            self.scene = "battle"
            self._execute_combat_cycle(trainer)
        else:
            self.scene = "overworld"
            self._execute_overworld_cycle()

    def _battle_readiness_snapshot(self, trainer: bool) -> dict[str, Any]:
        """Condition of the team as a fight begins: did we enter it prepared?"""
        snapshot: dict[str, Any] = {"trainer": trainer, "frame": self.bridge.frame_count, "prepared": True}
        if not self.native:
            return snapshot
        coords = self.bridge.get_player_coords_and_map()
        snapshot["map"] = [coords["map_group"], coords["map_num"]]
        try:
            party = self.bridge.get_decrypted_player_party()
            self._own_personalities = frozenset(int(mon.get("pid", -1)) for mon in party)
            healthy = [mon for mon in party if int(mon["hp"]) > 0 and not mon.get("is_egg", False)]
            if not healthy:
                return snapshot
            lead = max(healthy, key=lambda mon: (int(mon["level"]), int(mon["maxHP"])))
            fraction = int(lead["hp"]) / max(1, int(lead["maxHP"]))
            pp_fraction, _ = best_attack_pp(lead, self.planner.move_info,
                                            self.planner.species_types(int(lead["species"])))
            snapshot.update(lead_hp_fraction=round(fraction, 3), attack_pp_fraction=round(pp_fraction, 3),
                            prepared=fraction >= ENGAGE_HP_FRACTION and pp_fraction >= ENGAGE_PP_FRACTION,
                            team=len(healthy))
        except (ValueError, RuntimeError, KeyError, OSError):
            pass
        return snapshot

    def _update_counter_plan(self, objective: str) -> None:
        """Counter types against the remembered roster, and catchable species that carry them."""
        types = {species: self._species_types(species) for species in self.opponents.rosters.get(objective, {})}
        counters = self.opponents.counters(objective, types, self.shared.chart)
        if not counters:
            return
        self.planner.learned_counters[objective] = counters
        candidates = self.planner.catchable_species()
        targets = counter_species(candidates, counters, {species: self._species_types(species) for species in candidates})
        self.planner.counter_targets[objective] = targets[:12]
        self.desired_species |= set(targets[:12])
        self._directive_signature = ()
        self._record("counter_plan", objective=objective, counter_types=counters,
                     threats=self.opponents.threats(objective), catch_targets=targets[:12],
                     roster={str(species): level for species, level in self.opponents.rosters.get(objective, {}).items()})

    def _record_defeat(self) -> None:
        """A whiteout is analysed: unprepared starts ask for a restore, prepared ones for levels."""
        objective = next((task.milestone_id for task in self.planner.campaign_dag
                          if task.milestone_id not in self.metrics.completed_milestones), "")
        try:
            party = self.bridge.get_decrypted_player_party()
        except ValueError:
            party = []
        level = max((int(mon["level"]) for mon in party), default=5)
        if objective:
            task = next((task for task in self.planner.campaign_dag if task.milestone_id == objective), None)
            roster, move_types = self._last_trainer_roster
            # Only the objective's own fight describes the objective: a wild
            # whiteout or a trainer met on the way has a different roster.
            own_fight = (task is not None and bool(self._battle_start.get("trainer")) and
                         self._battle_start.get("map") == list(task.map_key))
            for species, seen_level in (roster.items() if own_fight else ()):
                self.opponents.observe(objective, species, seen_level)
                for move_type in move_types.get(species, ()):
                    self.opponents.observe(objective, species, seen_level, move_type)
            self._update_counter_plan(objective)
            prepared = bool(self._battle_start.get("prepared", True))
            self.planner.preparation.record_defeat(objective, level, prepared=prepared)
            self._directive_signature = ()
            self._lessons_dirty = True
            self._record("objective_defeat", objective=objective, lead_level=level, prepared=prepared,
                         lead_hp=self._battle_start.get("lead_hp_fraction"),
                         attack_pp=self._battle_start.get("attack_pp_fraction"),
                         lesson=("restore HP/PP before the retry" if not prepared else
                                 "prepared loss: raise level target and reconsider the team"))
            self._save_lessons(force=True)

    def _ui_fingerprint(self) -> str:
        """Dialog and menu progress that a coarse position signature cannot see.

        Only state that changes when the game accepts input or prints text is
        hashed: callback, active task identities and states, text cursors and
        menu cursors. Animation timers are excluded so a wait still looks stuck.
        """
        digest = hashlib.blake2b(digest_size=8)
        digest.update(self.bridge.read_memory(self.bridge.memory_map.main + 4, 4))
        tasks = self.bridge.memory_map.tasks
        for slot in range(16):
            header = self.bridge.read_memory(tasks + 40 * slot, 10)
            if header[4]:
                digest.update(header[:4] + header[8:10])
        printers = self._symbols.get("sTextPrinters")
        if printers is not None:
            raw = self.bridge.read_memory(printers.address, 36 * 32)
            for window in range(32):
                start = 36 * window
                digest.update(raw[start:start + 4] + raw[start + 0x1B:start + 0x1D])
        communication = self._symbols.get("gBattleCommunication")
        if communication is not None:
            digest.update(self.bridge.read_memory(communication.address, 8))
        menu = self.menu._native_symbols.get("sMenu")
        if menu is not None:
            digest.update(self.bridge.read_memory(menu, 12))
        return digest.hexdigest()

    def _flags_progress(self) -> int:
        """Monotonic count of persistent flags ever observed set.

        Temporary flags (0..31) reset on map entry and are excluded. A flag that
        toggles back and forth counts once, so a two-map ping-pong cannot fake
        story progress, while any new event, item or trainer flag does count.
        """
        memory = self.bridge.memory_map
        size = memory.vars_offset - memory.flags_offset
        data = self.bridge.read_memory(self.bridge.get_save_block1_dynamic_ptr() + memory.flags_offset, size)
        if len(self._flags_seen) != size:
            self._flags_seen = bytearray(size)
        merged = int.from_bytes(self._flags_seen, "little") | (int.from_bytes(data, "little") >> 32 << 32)
        self._flags_seen = bytearray(merged.to_bytes(size, "little"))
        return bin(merged).count("1")

    def _observe_progress(self) -> None:
        """Diagnose gameplay stalls; recovery still uses ordinary key input."""
        coords = self.bridge.get_player_coords_and_map()
        party = player_party_in_battle_order(self.bridge)
        mode = self.menu.menu_mode()
        battle = self.bridge.is_in_battle()[0]
        context: ProgressContext = "battle" if battle else "field"
        if mode not in ("overworld", "script", "action", "move", "animation"):
            context = "menu"
        phase: tuple[object, ...] = (mode, coords["tile_transition_state"])
        if mode != "overworld":
            phase += (self._ui_fingerprint(),)
        if battle:
            own, enemy = self.bridge.get_battle_pokemon(0), self.bridge.get_battle_pokemon(1)
            phase += (int(own.hp), int(enemy.hp), tuple(int(x) for x in own.pp))
        position = (coords["map_group"], coords["map_num"], coords["x"], coords["y"])
        gains: tuple[object, ...] = (
            tuple(self.metrics.completed_milestones), self.metrics.wins, self.metrics.captures,
            tuple((int(mon["pid"]), int(mon["species"]), int(mon["level"]), int(mon.get("experience", 0)),
                   tuple(mon["moves"])) for mon in party), tuple(sorted(self.bridge.get_inventory().items())),
            self._flags_progress(), self.metrics.heals)
        objective = next((task.milestone_id for task in self.planner.campaign_dag
                          if task.milestone_id not in self.metrics.completed_milestones), "POSTGAME_COMPLETE")
        preparation = sum(int(mon["level"]) + 5 for mon in party if not mon.get("is_egg", False))
        request = self.watchdog.observe(ProgressObservation(
            self.bridge.frame_count, context, (*position, *phase), gains, position, objective,
            strategic_gains=(tuple(self.metrics.completed_milestones), self._flags_progress()),
            preparation=preparation))
        if request is None:
            return
        self.metrics.recoveries += 1
        key = self._stall_key(objective, mode, coords)
        if request.attempt == 1:
            self._stall_memory[key] = self._stall_memory.get(key, 0) + 1
            if len(self._stall_memory) > 256:
                self._stall_memory.clear()
        self._record("stall_detected", recovery=asdict(request), position=coords, mode=mode,
                     recurrence=self._stall_memory.get(key, 1), directive=self._compact_directive(self._last_directive))
        self._recover(request, mode, battle, coords, objective)

    @staticmethod
    def _stall_key(objective: str, mode: str, coords: Mapping[str, int]) -> str:
        """Same objective, screen kind and map area: the "same" stall across rewinds."""
        return (f"{objective}|{mode}|{coords['map_group']}:{coords['map_num']}|"
                f"{coords['x'] // 4}:{coords['y'] // 4}")

    def _queue_probe(self, sequence: list[int], hold: int = 3, gap: int = 9) -> None:
        """Ordinary presses for unrecognized screens, each with a release edge."""
        frames: list[int] = []
        for key in sequence:
            frames.extend([key] * hold + [0] * gap)
        self._probe_keys = frames
        self.metrics.probes += 1

    def _recover(self, request: RecoveryRequest, mode: str, battle: bool,
                 coords: Mapping[str, int], objective: str) -> None:
        """Escalate: replan, probe inputs, detour/tabu, then rewind to a pre-stall checkpoint.

        A stall only ends the worker after every rewind budget is spent; the
        supervisor then restarts from an older verified checkpoint.
        """
        attempt = request.attempt
        position = (coords["map_group"], coords["map_num"], coords["x"], coords["y"])
        key = self._stall_key(objective, mode, coords)
        recurrence = max(0, self._stall_memory.get(key, 1) - 1)
        if request.reason == "objective_no_progress":
            # A strategic deadline asks for a different plan, and a new plan
            # needs time to play out (heal, travel, fight) before escalating.
            strategy = self._strategy_shift(request, objective)
            if strategy and self.watchdog.grant_strategic_grace(self.watchdog.strategic_limit // 2):
                self.movement.clear()
                self._directive_signature = ()
                self._route_commitment = None
                self._detour = None
                self.last_recovery = {"frame": self.bridge.frame_count, "attempt": attempt, "reason": request.reason,
                                      "mode": mode, "actions": [strategy], "rewinds": request.rewinds,
                                      "recurrence": recurrence}
                self._record("stall_recovery", action=f"strategy_shift:{strategy}", attempt=attempt, mode=mode,
                             rewinds=request.rewinds, reason=request.reason, recurrence=recurrence)
                return
        if attempt > 4:
            repeated = self._rewind_memory.get(key, 0)
            if (request.rewinds < self.max_rewinds and repeated < 3 * self.max_rewinds and
                    self._rewind(request, depth=request.rewinds + repeated)):
                self._rewind_memory[key] = repeated + 1
                return
            raise CampaignStalled(f"No gameplay progress after {attempt - 1} recoveries and "
                                  f"{request.rewinds} rewinds ({repeated} for this stall): {mode}; {objective}; {position}")
        failed_edge = self._route_commitment.edge if self._route_commitment is not None else self._active_transition
        failed_goal = self._directive_goal()
        self.movement.clear()
        self.menu.cancel()
        self._battle_decision = None
        self._last_battle_signature = None
        self._directive_signature = ()
        self._active_transition = None
        self.movement.blocked_edges.clear()
        self._blocked_at.clear()
        if attempt >= 2:
            self._route_commitment = None
        actions = ["release_and_replan"]
        if self.field is not None and self.field.busy and attempt >= 2:
            self.field.cancel("stalled")
            actions.append("cancel_stalled_field_operation")
        evolution = mode == "evolution"
        ready_field = not battle and mode == "overworld" and self.bridge.is_hardware_ready()
        # Dismissing with B is tried once; if that screen persists it is not optional.
        keys = self.menu.recovery_key(mode) if attempt == 1 else None
        if keys is not None:
            self.bridge.set_hardware_keys(keys)
            actions.append("dismiss_unexpected_menu")
        elif not ready_field:
            # An unrecognized screen: try ordinary inputs with growing variety.
            # B is never sent to an evolution, because it cancels evolving.
            a, b = self.bridge.KEY_A, self.bridge.KEY_B
            up, down, start = self.bridge.KEY_UP, self.bridge.KEY_DOWN, self.bridge.KEY_START
            left, right, select = self.bridge.KEY_LEFT, self.bridge.KEY_RIGHT, self.bridge.KEY_SELECT
            if battle and mode in ("party", "party_context", "party_error"):
                # A forced replacement ignores B: choose another row and confirm it.
                pool = [[down, a, a], [down, down, a, a], [up, a, a, b, down, a, a], [down, down, down, a, a],
                        [up, up, a, a], [down] * 4 + [a, a]]
            else:
                # A recurring stall walks through a wider pool of ordinary inputs
                # (other menu rows, cancels, start/select) instead of repeating one.
                pool = [[a] * 6, [b, b, a, a, b, a], [down, a, up, a, b, start, b, b], [down, down, a, a],
                        [down, down, down, a, a], [up, a, a], [right, a, left, a], [b, b, b, b],
                        [a, b, a, b, a, b], [start, b, b], [down] * 4 + [a, a], [select, a, b],
                        [up, up, a, a], [left, a, right, a], [down] * 5 + [a, a], [a] * 12]
                if evolution:
                    pool = [sequence for sequence in pool if b not in sequence and start not in sequence and select not in sequence]
            if attempt in (2, 3, 4):
                sequence = pool[(recurrence * 3 + attempt - 2) % len(pool)]
                self._queue_probe(sequence)
                actions.append(f"input_probe_{attempt}_{(recurrence * 3 + attempt - 2) % len(pool)}")
        elif attempt == 2:
            # Explore only collision-checked ordinary adjacent steps. No RAM
            # teleport or progress flag writes are used to escape an obstruction.
            _, collisions, elevations = self.bridge.extract_live_map_matrices()
            pathfinder = GbaConstrainedAStar(collisions, elevations, self.bridge.extract_behavior_map(), self.game)
            pathfinder.terminal_tiles = {edge.source_xy for edge in self.topology.adjacency.get(position[:2], [])
                                        if edge.kind == "warp"}
            actors = self.bridge.get_object_events()
            pathfinder.register_object_events(actors, self.bridge.read_u8(self.bridge.memory_map.avatar + 5))
            neighbors = ((1, 0), (0, 1), (-1, 0), (0, -1))
            for offset in range(4):
                dx, dy = neighbors[(offset + attempt) % 4]
                target = (coords["x"] + dx, coords["y"] + dy)
                if target in pathfinder.terminal_tiles:
                    continue
                path = pathfinder.find_optimal_path((coords["x"], coords["y"], coords["z"]), target)
                if path and len(path) == 1:
                    self.movement.move_toward(path[0], "walk")
                    actions.append("try_legal_alternate_step")
                    break
        else:
            if failed_edge is not None and self._tabu(failed_edge, f"stall attempt {attempt} while using this exit",
                                                      goal=failed_goal):
                actions.append("tabu_exit")
            if self._start_detour(f"stall attempt {attempt}"):
                actions.append("exploration_detour")
        self.last_recovery = {"frame": self.bridge.frame_count, "attempt": attempt, "reason": request.reason,
                              "mode": mode, "actions": actions, "rewinds": request.rewinds, "recurrence": recurrence}
        self._record("stall_recovery", action="+".join(actions), attempt=attempt, mode=mode,
                     rewinds=request.rewinds, reason=request.reason, recurrence=recurrence)
        if attempt in (1, 4) and self.checkpoint_dir is not None:
            self.save_checkpoint("stall_recovery")

    def _strategy_shift(self, request: RecoveryRequest, objective: str) -> str:
        """A strategic deadline changes the plan instead of replaying the same one.

        Grinding that no longer earns credit ends: the objective is attempted
        with the current team. Otherwise stale exits are forgotten so the
        route search reconsiders every way, and learned refusals stay.
        """
        mode = str(self._last_directive.get("mode", ""))
        if mode == "OPTIMAL_TRAINING_CURRICULUM" and objective:
            if self.planner.preparation.force_attempt(objective):
                self._directive_signature = ()
                self._lessons_dirty = True
                self._record("strategy_shift", objective=objective, change="attempt_with_current_team",
                             reason="training stopped earning strategic credit")
                return "attempt_objective_now"
        if self._tabu_transitions:
            for edge_id in list(self._tabu_transitions):
                self.topology.unblock_transition(edge_id)
            self._tabu_transitions.clear()
            self._record("strategy_shift", objective=objective, change="forget_temporary_exit_bans",
                         reason="no story progress; reconsider every route")
            return "forget_tabu"
        return ""

    def _rewind_candidates(self, progress_frame: int, minimum_frame: int = 0,
                           minimum_level_ups: int = 0) -> list[Path]:
        """Intact checkpoints of this game saved no later than the last observed gain."""
        if self.checkpoint_dir is None:
            return []
        rows: list[tuple[int, str, Path]] = []
        achieved = set(self.metrics.completed_milestones)
        for path in self.checkpoint_dir.glob("frame-*"):
            try:
                manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(manifest, dict) or manifest.get("game") != self.game or bool(manifest.get("native")) != self.native:
                continue
            if manifest.get("reason") in ("error", "stall_recovery"):
                continue
            status = manifest.get("status", {})
            metrics = status.get("metrics", {}) if isinstance(status, dict) else {}
            saved_milestones = metrics.get("completed_milestones", []) if isinstance(metrics, dict) else []
            if (not isinstance(saved_milestones, list) or any(not isinstance(value, str) for value in saved_milestones) or
                    not achieved.issubset(set(saved_milestones))):
                # A local recovery must retain already achieved story progress.
                # Rewinding a level-53 nurse loop to pre-fossil Mt. Moon only
                # replaces the loop with lost campaign work.
                continue
            level_ups = metrics.get("level_ups", 0) if isinstance(metrics, dict) else 0
            if minimum_level_ups and (not isinstance(level_ups, int) or level_ups < minimum_level_ups):
                continue  # restoring it would discard levels trained since
            frame = manifest.get("frame")
            if isinstance(frame, int) and minimum_frame <= frame <= progress_frame:
                rows.append((frame, path.name, path))
        rows.sort()
        return [path for _, _, path in rows]

    def _rewind(self, request: RecoveryRequest, depth: int | None = None) -> bool:
        """Restore a pre-stall checkpoint, keeping what the failed timeline taught.

        Visited tiles, monotonic flag progress, tabu exits and diagnostics carry
        over, and the policy RNG is reseeded, so the replay can take a different
        path instead of repeating the same deterministic loop.
        """
        if request.reason == "objective_no_progress":
            # A strategic stall is not a physical one: never throw away the
            # levels trained since then by restoring a pre-training save.
            candidates = self._rewind_candidates(self.bridge.frame_count, minimum_frame=self._last_level_frame,
                                                 minimum_level_ups=self.metrics.level_ups)
        else:
            candidates = self._rewind_candidates(request.progress_frame)
        if not candidates:
            return False
        steps = request.rewinds if depth is None else depth
        target = candidates[-(1 + min(len(candidates) - 1, steps))]
        frame_before = self.bridge.frame_count
        watchdog_before = self.watchdog.state_dict()
        stall_memory, rewind_memory = dict(self._stall_memory), dict(self._rewind_memory)
        rewinds = request.rewinds + 1
        visited = set(self.watchdog.visited)
        flags_seen = bytes(self._flags_seen)
        diagnostics = {name: int(getattr(self.metrics, name))
                       for name in ("recoveries", "faults", "rewinds", "probes", "detours", "checkpoints")}
        remaining = {edge_id: until - frame_before for edge_id, until in self._tabu_transitions.items() if until > frame_before}
        failed_edge = self._route_commitment.edge if self._route_commitment is not None else self._active_transition
        failed_goal = self._directive_goal()
        try:
            self.load_checkpoint(target)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            self._record("stall_rewind_failed", target=str(target), error=f"{type(error).__name__}: {error}")
            return False
        for name, value in diagnostics.items():
            setattr(self.metrics, name, max(int(getattr(self.metrics, name)), value))
        self._stall_memory, self._rewind_memory = stall_memory, rewind_memory
        self.metrics.rewinds += 1
        self.watchdog.rewinds, self.watchdog.attempts = rewinds, 0
        self.watchdog.restore_strategic_progress(watchdog_before, reset_attempts=True)
        for location in visited:
            if len(self.watchdog.visited) >= 16384:
                break
            self.watchdog.visited.add(location)
        if len(flags_seen) == len(self._flags_seen):
            self._flags_seen = bytearray(a | b for a, b in zip(self._flags_seen, flags_seen))
        frame = self.bridge.frame_count
        for edge_id, frames in remaining.items():
            self._tabu_transitions[edge_id] = frame + frames
            self.topology.block_transition_id(edge_id)
        if failed_edge is not None:
            self._tabu(failed_edge, "exit in use when the abandoned timeline stalled", goal=failed_goal)
        mix = (self.seed + 7919 * rewinds + frame_before) % (2**31 - 1)
        self.rng.seed(mix)
        torch.manual_seed(mix)
        self._record("stall_rewind", target=str(target), from_frame=frame_before, to_frame=frame,
                     rewinds=rewinds, reason=request.reason)
        self.last_recovery = {"frame": frame, "attempt": request.attempt, "reason": request.reason,
                              "mode": "rewind", "actions": ["rewind_checkpoint"], "rewinds": rewinds,
                              "target": target.name}
        return True

    def _observe_bootstrap_progress(self) -> None:
        """Observe title/naming stalls without reading uninitialized party RAM."""
        if self._ui is None:
            return
        ui = self._ui.status()
        callback = self.bridge.read_u32(self.bridge.memory_map.main + 4)
        naming: tuple[object, ...] = ()
        if ui.naming:
            pointer = self.bridge.read_u32(self._symbols["sNamingScreen"].address)
            if 0x02000000 <= pointer <= 0x0203DFFF:
                naming = (self.bridge.read_memory(pointer + 0x1800, 16),
                          self.bridge.read_memory(pointer + 0x1E10, 1),
                          self.bridge.read_memory(pointer + 0x1E22, 2))
                sprite = self.bridge.read_u8(pointer + 0x1E23)
                if sprite < 64:
                    naming += (self.bridge.read_memory(self._symbols["gSprites"].address + sprite * 68 + 0x2E, 4),)
        request = self.watchdog.observe(ProgressObservation(self.bridge.frame_count, "bootstrap",
            (ui.callback, *naming), (), (0, 0, callback, 0), "BOOTSTRAP"))
        if request is None:
            return
        self.metrics.recoveries += 1
        self._record("stall_detected", recovery=asdict(request), callback=ui.callback)
        if request.attempt > 6:
            raise CampaignStalled(f"Introduction stalled after6 recovery attempts: {ui.callback}")
        # A neutral edge then a fresh accepted key is sufficient for title/text
        # waits. Naming retains its observed-character navigation controller.
        if not ui.naming:
            keys = [self.bridge.KEY_START if "Title" in ui.callback else self.bridge.KEY_A] * 3
            if request.attempt >= 4:
                keys += [self.bridge.KEY_B, self.bridge.KEY_A, self.bridge.KEY_START]
            self._queue_probe(keys)
        self._record("stall_recovery", action="retry_introduction_input", attempt=request.attempt,
                     callback=ui.callback)

    @staticmethod
    def _fatal(error: BaseException) -> bool:
        """Errors that recovery cannot fix: a closed emulator or exhausted memory."""
        return isinstance(error, MemoryError) or (isinstance(error, RuntimeError) and str(error) == "Bridge is closed")

    def _note_fault(self, error: BaseException, where: str) -> None:
        frame = self.bridge.frame_count
        self.metrics.faults += 1
        self._fault_frames.append(frame)
        message = f"{type(error).__name__}: {error}"[:400]
        frames = traceback.extract_tb(error.__traceback__)[-3:]
        origin = " <- ".join(f"{Path(entry.filename).name}:{entry.lineno}:{entry.name}" for entry in reversed(frames))
        self.last_fault = {"frame": frame, "where": where, "error": message, "origin": origin, "t": round(time.time(), 3)}
        self._record_throttled("agent_fault", f"{where}:{message[:160]}", 900, where=where, error=message, origin=origin)

    def _handle_fault(self, error: Exception) -> None:
        """Contain a policy error to this frame and fall back to a safe input.

        Releasing intent forces a fresh plan; A only advances text and is the one
        input that is safe on unknown screens. Persistent faults stall progress,
        so the watchdog ladder (probes, detours, rewinds) still applies.
        """
        self._note_fault(error, self.scene)
        self.movement.clear()
        self.menu.cancel()
        self._battle_decision = None
        self._last_battle_signature = None
        self._directive_signature = ()
        self._route_commitment = None
        try:
            busy = self.bridge.is_in_battle()[0] or not self.bridge.is_hardware_ready()
        except (ValueError, RuntimeError):
            busy = True
        self.bridge.set_hardware_keys(self.bridge.KEY_A if busy and self.bridge.frame_count % 8 < 2 else 0)

    def _observe_safely(self) -> None:
        try:
            if self._bootstrapped:
                self._observe_progress()
            else:
                self._observe_bootstrap_progress()
        except CampaignStalled:
            raise
        except Exception as error:
            if isinstance(error, ValueError) and str(error) == "Party contains an invalid checksum":
                return
            if not self.fault_tolerant or self._fatal(error):
                raise
            self._note_fault(error, "progress_watchdog")

    def _track_map_transition(self) -> None:
        """Count a map change only after it is stable for three frames.

        Saves relocate during warps and battle transitions, so a single frame can
        read a transient location such as (0, 0) and flip straight back.
        """
        try:
            coords = self.bridge.get_player_coords_and_map()
        except ValueError:
            return
        position = (coords["map_group"], coords["map_num"], coords["x"], coords["y"])
        if self._last_position is None or position[:2] == self._last_position[:2]:
            self._pending_map = None
            self._last_position = position
            return
        pending = self._pending_map
        count = pending[1] + 1 if pending is not None and pending[0] == position[:2] else 1
        if count < 3:
            self._pending_map = (position[:2], count)
            return
        self._pending_map = None
        self.metrics.map_transitions += 1
        self._record("map_transition", source=list(self._last_position[:2]), destination=list(position[:2]))
        self._last_position = position

    def step_vblank_cycle(self) -> None:
        """Own one VBlank; retry transient decryption without accepting bad data.

        ARM code can straddle VBlank while GetMonData temporarily decrypts a
        party record. This snapshot cannot become a policy observation.
        Persistent checksum failure remains an error after120 neutral frames.
        Any other policy error is contained to its frame when fault tolerant.
        """
        snapshot_failed = False
        try:
            self._select_frame_input()
        except CampaignStalled:
            raise
        except ValueError as error:
            if self.native and str(error) == "Party contains an invalid checksum":
                snapshot_failed = True
                self._snapshot_failures += 1
                if self._snapshot_failures > 120:
                    raise ValueError("Native party checksum remained invalid for120 frames") from error
                self.bridge.set_hardware_keys(0)
                if self._snapshot_failures == 1:
                    self._record("memory_snapshot_retry", reason="party checksum invalid at VBlank")
            elif self.fault_tolerant:
                self._handle_fault(error)
            else:
                raise
        except Exception as error:
            if not self.fault_tolerant or self._fatal(error):
                raise
            self._handle_fault(error)
        else:
            self._snapshot_failures = 0
        if not snapshot_failed and self.native and self.bridge.frame_count % 30 == 0:
            self._observe_safely()
        self.bridge.step_frame()
        self.metrics.frames = self.bridge.frame_count
        if self._bootstrapped:
            self._track_map_transition()
        if self.checkpoint_dir is not None and time.monotonic() - self._last_checkpoint_time >= 600:
            try:
                self.save_checkpoint("periodic")
            except OSError as error:
                # A full disk or locked file must not end the campaign; retry later.
                self._last_checkpoint_time = time.monotonic()
                self._note_fault(error, "periodic_checkpoint")

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
        frame = self.bridge.frame_count
        result: dict[str, object] = {"game": self.game, "backend": "native" if self.native else "synthetic",
            "frames": frame, "metrics": asdict(self.metrics),
            "directive": self._compact_directive(self._last_directive), "native_campaign_certified": False,
            "scene": self.scene,
            "progress_watchdog": {"attempts": self.watchdog.attempts, "rewinds": self.watchdog.rewinds,
                "max_rewinds": self.max_rewinds,
                "stationary_frames": self.watchdog.stationary_frames(frame),
                "unproductive_frames": self.watchdog.unproductive_frames(frame),
                "strategic_unproductive_frames": self.watchdog.strategic_unproductive_frames,
                "strategic_attempts": self.watchdog.strategic_attempts,
                "strategic_limit": self.watchdog.strategic_limit,
                "last_recovery": self.last_recovery},
            "faults": {"count": self.metrics.faults, "last": self.last_fault}}
        try:
            result["position"] = self.bridge.get_player_coords_and_map()
            result["party_species"] = sorted(self.owned_species)
            if self._bootstrapped:
                try:
                    result["party_levels"] = [int(mon["level"]) for mon in self.bridge.get_decrypted_player_party()]
                except (ValueError, RuntimeError):
                    pass
            if self.native and self._bootstrapped:
                evidence = self.native_completion_evidence()
                result["native_completion_evidence"] = evidence
                result["native_campaign_certified"] = evidence["complete"]
        except ValueError:
            result["position"] = None
        return result

    def telemetry(self) -> dict[str, object]:
        """Rich, display-only state for live dashboards. It never raises."""
        objective = next((task.milestone_id for task in self.planner.campaign_dag
                          if task.milestone_id not in self.metrics.completed_milestones), "POSTGAME_COMPLETE")
        commitment = self._route_commitment
        data: dict[str, object] = {
            "scene": self.scene, "last_decision": self.last_decision,
            "recent_decisions": list(self.recent_decisions)[-12:],
            "last_fault": self.last_fault, "last_recovery": self.last_recovery,
            "milestones": {"completed": list(self.metrics.completed_milestones),
                           "total": len(self.planner.campaign_dag), "next": objective,
                           "order": [task.milestone_id for task in self.planner.campaign_dag]},
            "route_commitment": None if commitment is None else {
                "exit": list(commitment.edge.source_xy), "via": list(commitment.edge.destination),
                "kind": commitment.edge.kind, "until": commitment.until},
            "detour": None if self._detour is None else {"target": list(self._detour[1]), "until": self._detour[2]},
            "tabu_exits": len(self._tabu_transitions),
            "probe_frames_left": len(self._probe_keys),
            "learned_barriers": self.pushback.describe(self._world),
            "gate_relaxed": self._gate_relaxed,
            "preparation": self.planner.preparation.lessons_state()}
        try:
            data["plan"] = self._plan_summary(objective)
        except (ValueError, RuntimeError, KeyError, IndexError, TypeError):
            data["plan"] = {"objective": objective}
        if not self._bootstrapped:
            return data
        try:
            data["party"] = [describe_mon(mon, self.catalog) for mon in self.bridge.get_decrypted_player_party()]
        except (ValueError, RuntimeError, KeyError):
            data["party_unavailable"] = True
        try:
            data["menu"] = self.menu.menu_mode()
        except (ValueError, RuntimeError, KeyError):
            data["menu"] = "unknown"
        if self.native:
            try:
                data["trainer"] = trainer_card(self.bridge, self.game)
            except (ValueError, RuntimeError, KeyError):
                pass
            try:
                data["battle"] = describe_battle(self.bridge, self.catalog)
            except (ValueError, RuntimeError, KeyError):
                pass
            catalog = self.topology.catalog
            try:
                coords = self.bridge.get_player_coords_and_map()
                if catalog is not None:
                    raw = catalog.maps.get((coords["map_group"], coords["map_num"]))
                    data["map_name"] = str(raw["name"]) if raw is not None else ""
            except (ValueError, RuntimeError):
                pass
        return data

    def _plan_summary(self, objective: str) -> dict[str, object]:
        """What the agent is trying to do, why, and what it knows about it."""
        directive = self._last_directive
        def type_names(values: list[int]) -> list[str]:
            return [TYPE_NAMES[value] for value in values if 0 <= value < len(TYPE_NAMES)]
        roster = [{"name": species_name(self.catalog, species), "level": level,
                   "types": type_names(list(dict.fromkeys(self._species_types(species))))}
                  for species, level in sorted(self.opponents.rosters.get(objective, {}).items(), key=lambda row: row[1])]
        summary: dict[str, object] = {
            "objective": objective, "mode": directive.get("mode", ""),
            "reason": directive.get("reason") or directive.get("preparation_reason") or directive.get("training_reason") or "",
            "requirement": directive.get("requirement", ""),
            "training_route": directive.get("training_route", ""),
            "hunt": [species_name(self.catalog, species) for species in directive.get("hunt_species", []) or []],
            "boss_roster": roster,
            "counter_types": type_names(self.planner.learned_counters.get(objective, [])),
            "threat_types": type_names(self.opponents.threats(objective)),
            "catch_targets": [species_name(self.catalog, species) for species in self.planner.counter_targets.get(objective, [])][:8],
            "type_chart": {"known_cells": self.shared.chart.known_cells(), "observations": self.shared.chart.observations,
                           "types": [TYPE_NAMES[value] for value in range(18) if value != 9],
                           "table": self.shared.chart.table()},
        }
        if self.native and self.topology.catalog is not None and self._bootstrapped:
            try:
                inventory = self.bridge.get_inventory()
                constants = self.topology.catalog.constants
                summary["stock"] = {"healing_hp": healing_stock(inventory, constants),
                                    "balls": ball_stock(inventory, constants)}
            except (ValueError, RuntimeError):
                pass
        return summary

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
            "belief_levels": self._belief_levels,
            "knowledge": asdict(self.knowledge), "metrics": asdict(self.metrics),
            "owned_species": self.owned_species, "observed_species": self.observed_species,
            "desired_species": self.desired_species, "desired_moves": self.desired_moves,
            "team_signature": self._team_signature, "in_battle": self._in_battle,
            "synthetic_event_index": self._synthetic_event_index,
            "last_enemy_move": self._last_enemy_move, "battle_turns": self._battle_turns,
            "battle_escape_attempts": self._battle_escape_attempts,
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
            "observed_no_effect": sorted(self._observed_no_effect),
            "hardware_keys": self.bridge.hardware_keys,
            "progress_watchdog": self.watchdog.state_dict(),
            "preparation": self.planner.preparation.state_dict(),
            "flags_seen": bytes(self._flags_seen),
            "tabu_transitions": [[*edge_id, max(0, until - self.bridge.frame_count)]
                                 for edge_id, until in self._tabu_transitions.items()],
            "puzzle_executor": self._puzzle.state_dict() if self._puzzle is not None else None,
            "stall_memory": dict(self._stall_memory), "rewind_memory": dict(self._rewind_memory),
            "puzzle_engaged": list(self._puzzle_engaged[0]) + [self._puzzle_engaged[1]] if self._puzzle_engaged else None,
            "pushback": self.pushback.state_dict(),
            "battle_start": dict(self._battle_start),
            "last_level_frame": self._last_level_frame,
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
        self._prune_checkpoints()
        return directory

    def _prune_checkpoints(self) -> None:
        """Bound disk use: keep recent bundles, durable milestones and pinned ones.

        The bundle named by latest.json is never removed, and nothing is pruned
        when that pointer cannot be read.
        """
        if self.checkpoint_dir is None or self.retain_checkpoints <= 0:
            return
        try:
            pointer = json.loads((self.checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
            latest = str(pointer["directory"])
        except (OSError, ValueError, KeyError, TypeError):
            return
        entries: list[tuple[int, Path]] = []
        for path in self.checkpoint_dir.glob("frame-*"):
            try:
                entries.append((int(path.name.rsplit("-", 1)[1]), path))
            except (IndexError, ValueError):
                continue
        entries.sort(reverse=True)
        durable = 0
        for rank, (_, path) in enumerate(entries):
            if rank < self.retain_checkpoints or path.name == latest or (path / "PINNED").exists():
                continue
            try:
                reason = json.loads((path / "manifest.json").read_text(encoding="utf-8")).get("reason")
            except (OSError, ValueError, AttributeError):
                reason = None
            if reason in _DURABLE_CHECKPOINTS and durable < 64:
                durable += 1
                continue
            shutil.rmtree(path, ignore_errors=True)

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
        self._belief_levels = {int(species): int(level) for species, level in state.get("belief_levels", {}).items()}
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
        self._battle_escape_attempts = state.get("battle_escape_attempts", 0)
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
        self._observed_no_effect = {tuple(row) for row in state.get("observed_no_effect", [])}
        if "progress_watchdog" in state:
            self.watchdog.load_state_dict(state["progress_watchdog"])
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
        if isinstance(state.get("preparation"), dict):
            try:
                self.planner.preparation.load_state_dict(state["preparation"])
            except ValueError:
                pass  # An invalid optional record only resets training budgets.
        flags_seen = state.get("flags_seen", b"")
        self._flags_seen = bytearray(flags_seen) if isinstance(flags_seen, (bytes, bytearray)) else bytearray()
        for edge_id in self._tabu_transitions:
            self.topology.unblock_transition(edge_id)
        self._tabu_transitions = {}
        for row in state.get("tabu_transitions", []):
            if isinstance(row, (list, tuple)) and len(row) == 7 and all(isinstance(value, int) for value in row):
                restored_id = cast(tuple[int, int, int, int, int, int], tuple(row[:6]))
                self._tabu_transitions[restored_id] = self.bridge.frame_count + int(row[6])
                self.topology.block_transition_id(restored_id)
        for name in ("stall_memory", "rewind_memory"):
            saved_memory = state.get(name, {})
            if isinstance(saved_memory, dict):
                setattr(self, f"_{name}", {str(key): int(value) for key, value in saved_memory.items()
                                           if isinstance(value, int)})
        self._puzzle_engaged = None
        if self._puzzle is not None:
            self._puzzle.clear()
            engaged, saved = state.get("puzzle_engaged"), state.get("puzzle_executor")
            if isinstance(engaged, list) and len(engaged) == 3 and isinstance(saved, dict):
                try:
                    self._puzzle.load_state_dict(saved)
                    self._puzzle_engaged = ((int(engaged[0]), int(engaged[1])), str(engaged[2]))
                except (ValueError, KeyError, TypeError):
                    self._puzzle.clear()  # a stale plan is replanned from fresh RAM
        saved_pushback = state.get("pushback")
        if isinstance(saved_pushback, dict):
            try:
                self.pushback.load_state_dict(saved_pushback)
            except (ValueError, KeyError, TypeError):
                self.pushback = PushbackLearner(self.game)
        self.pushback.reset()
        battle_start = state.get("battle_start")
        self._battle_start = dict(battle_start) if isinstance(battle_start, dict) else {}
        level_frame = state.get("last_level_frame", 0)
        self._last_level_frame = level_frame if isinstance(level_frame, int) else 0
        # Lessons from every timeline (including the abandoned one) still apply.
        self._load_lessons()
        self.shared.load()
        self._replay_chart()
        self._world = ""
        self._gate_relaxed = ""
        self._pending_unlock = None
        # Transient intent belongs to the abandoned timeline, never to the restored one.
        self._probe_keys = []
        self._route_commitment = None
        self._detour = None
        self._reentry = None
        self._party_snapshot = {}
        self._pending_map = None
        self._recent_tiles.clear()
        self._directive_signature = ()
        self._last_checkpoint_time = time.monotonic()
        self._record("checkpoint_restored", path=str(directory))

    def close(self) -> None:
        """Release held keys and the native handle exactly once."""
        self.bridge.set_hardware_keys(0)
        self.bridge.close()
