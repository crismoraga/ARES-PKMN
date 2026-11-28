"""``BattleBrain``: the E1 Transformer policy plus the explicit strategic layer, for every game.

``choose(view)`` runs both on the same ``BattleView``:

* the policy: the view is rebuilt as a Stadium position (``stadium_map``) and encoded with the E1
  observation; the checkpoint is the best E1 PPO seed on the validation split (``ppo/seed4``), or
  ``ARES_BRAIN_CKPT``, or the league champion when ``artifacts/ares_u/league/report.json`` says it
  beats E1;
* the strategy (``strategy.Strategist``): the analytic duel model scores every option (moves,
  switches, run, catch) with a win probability and a Spanish explanation, and vetoes options that
  cannot work (a move into an immunity, status on a statused or immune foe).

Modes (``mode``): ``"brain"`` (default) picks ``argmax(log pi(a) + w * V(a))`` over the options the
strategy does not veto (w = ``strategy_weight``; options the policy cannot see get ``UNSEEN_LOGP``);
a strategic run/catch decision (wild battles) is taken as is. ``"policy"`` is the raw policy argmax
(ablation) and ``"strategy"`` the strategy alone. Every choice carries the scores of all options.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ares_u.battle.model import build_policy, load_numpy, policy_numpy
from ares_u.battle.obs import stack

from .api import BattleView, BrainChoice, display_name
from .stadium_map import PolicyInput, PolicyMapper
from .strategy import Analysis, Option, Strategist

ROOT = Path(__file__).resolve().parents[2]
E1_DIR = ROOT / "artifacts" / "ares_u" / "battle" / "final" / "ppo"
DEFAULT_SEED = 4  # best mean score vs Greedy+prior on the validation split gen1_train (per_battle.json)
DEFAULT_CHECKPOINT = E1_DIR / f"seed{DEFAULT_SEED}" / "final.pt"
LEAGUE_DIR = ROOT / "artifacts" / "ares_u" / "league"
MODES = ("brain", "policy", "strategy")
STRATEGY_WEIGHT = 4.0
UNSEEN_LOGP = math.log(0.05)


def resolve_checkpoint() -> tuple[Path, str]:
    """(checkpoint, why): ``ARES_BRAIN_CKPT`` > league champion that beats E1 > best E1 PPO seed."""
    override = os.environ.get("ARES_BRAIN_CKPT", "").strip()
    if override:
        return Path(override), "ARES_BRAIN_CKPT"
    report = LEAGUE_DIR / "report.json"
    champion = LEAGUE_DIR / "champion.pt"
    if report.exists() and champion.exists():
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        if _champion_beats_e1(data):
            return champion, "league champion (report.json: beats E1)"
    return DEFAULT_CHECKPOINT, f"E1 PPO seed{DEFAULT_SEED} (best on validation)"


def _champion_beats_e1(data: object) -> bool:
    """Tolerant reader of the league verdict (any explicit boolean that says the champion beats E1)."""
    if not isinstance(data, dict):
        return False
    for key in ("champion_beats_e1", "beats_e1"):
        if isinstance(data.get(key), bool):
            return bool(data[key])
    criterion = data.get("criterion")
    if isinstance(criterion, dict):
        for key in ("champion_beats_e1", "beats_e1", "passed", "met"):
            if isinstance(criterion.get(key), bool):
                return bool(criterion[key])
    return False


class PolicyNet:
    """E1 Transformer on CPU, forward only."""

    def __init__(self, checkpoint: Path) -> None:
        from ares_u.battle.ppo import load_weights

        method, weights, meta = load_weights(checkpoint)
        self.method = method
        self.model = build_policy(method)
        load_numpy(self.model, weights)
        self.model.eval()
        self.checkpoint = checkpoint
        self.meta = meta

    def probs(self, inputs: Sequence[PolicyInput]) -> NDArray[np.float64]:
        probs, _ = policy_numpy(self.model, stack([item.observation for item in inputs]))
        return probs


@dataclass
class _Plan:
    view: BattleView
    analysis: Analysis
    policy_input: PolicyInput | None
    error: str = ""


class BattleBrain:
    """One explainable battle brain for every game (see the module docstring)."""

    def __init__(self, mode: str = "brain", checkpoint: Path | str | None = None,
                 strategy_weight: float = STRATEGY_WEIGHT, load_policy: bool = True) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        self.mode = mode
        self.strategy_weight = float(strategy_weight)
        self.strategist = Strategist()
        self.mapper = PolicyMapper(self.strategist.dex)
        self.policy: PolicyNet | None = None
        self.checkpoint_reason = ""
        if mode != "strategy" and load_policy:
            path, why = (Path(checkpoint), "explicit") if checkpoint is not None else resolve_checkpoint()
            self.policy = PolicyNet(path)
            self.checkpoint_reason = why
        self.last_analysis: Analysis | None = None
        self.last_probs: dict[str, float] = {}

    # ------------------------------------------------------------------ public API
    def choose(self, view: BattleView) -> BrainChoice:
        return self.choose_many([view])[0]

    def choose_many(self, views: Sequence[BattleView]) -> list[BrainChoice]:
        plans = [self._plan(view) for view in views]
        need = [plan for plan in plans if plan.policy_input is not None]
        probs: list[NDArray[np.float64] | None] = [None] * len(plans)
        if need and self.policy is not None:
            table = self.policy.probs([plan.policy_input for plan in need if plan.policy_input is not None])
            row = 0
            for index, plan in enumerate(plans):
                if plan.policy_input is not None:
                    probs[index] = table[row]
                    row += 1
        choices = [self._decide(plan, probs[index]) for index, plan in enumerate(plans)]
        if plans:
            self.last_analysis = plans[-1].analysis
        return choices

    def analyze(self, view: BattleView) -> Analysis:
        return self.strategist.analyze(view)

    # ------------------------------------------------------------------ internals
    def _plan(self, view: BattleView) -> _Plan:
        analysis = self.strategist.analyze(view)
        policy_input = None
        error = ""
        if self.policy is not None and self.mode in ("brain", "policy"):
            try:
                policy_input = self.mapper.build(view, self._bench_order(view, analysis))
            except (ValueError, IndexError, KeyError) as exc:  # unusual view: strategy only
                error = f"{type(exc).__name__}: {exc}"
        return _Plan(view, analysis, policy_input, error)

    @staticmethod
    def _bench_order(view: BattleView, analysis: Analysis) -> list[int]:
        """Bench members shown to the policy: alive ones by strategic duel value, then fainted ones."""
        alive = [index for index, mon in enumerate(view.party) if index != view.active_index and mon.hp > 0]
        fainted = [index for index, mon in enumerate(view.party) if index != view.active_index and mon.hp <= 0]

        def value(index: int) -> float:
            option = analysis.option("switch", index)
            return option.details.get("win", option.value) if option is not None else 0.0

        return sorted(alive, key=lambda index: -value(index)) + fainted

    def _option_logp(self, plan: _Plan, probs: NDArray[np.float64] | None) -> dict[tuple[str, int], float]:
        if probs is None or plan.policy_input is None:
            return {}
        out: dict[tuple[str, int], float] = {}
        for action in plan.policy_input.legal:
            target = plan.policy_input.action_map.get(action)
            if target is not None and action < len(probs):
                out[target] = math.log(max(1e-9, float(probs[action])))
        return out

    def _decide(self, plan: _Plan, probs: NDArray[np.float64] | None) -> BrainChoice:
        analysis = plan.analysis
        logp = self._option_logp(plan, probs)
        self.last_probs = {f"{kind}{index}": round(math.exp(value), 4) for (kind, index), value in logp.items()}
        options = [option for option in analysis.options if option.legal]
        if not options:
            return BrainChoice("move", 0, "Sin opciones legales: Forcejeo", {"forcejeo": 1.0})
        if self.mode == "policy" and logp:
            (kind, index), _ = max(logp.items(), key=lambda item: item[1])
            option = analysis.option(kind, index)
            reason = f"Red E1: {option.label if option else kind} (p={math.exp(logp[(kind, index)]):.0%})"
            return BrainChoice(kind, index, reason, self._scores(analysis, logp, {}))
        usable = [option for option in options if option.usable] or options
        strategic = max(usable, key=lambda item: (item.value, item.damage, -item.index))
        if self.mode == "strategy" or not logp or strategic.kind in ("run", "item"):
            return self._choice(strategic, analysis, logp, {}, policy_agrees=None)
        combined: dict[tuple[str, int], float] = {}
        for option in usable:
            prior = logp.get((option.kind, option.index), UNSEEN_LOGP)
            combined[(option.kind, option.index)] = prior + self.strategy_weight * option.value
        (kind, index), _ = max(combined.items(), key=lambda item: item[1])
        chosen = analysis.option(kind, index)
        assert chosen is not None
        top_policy = max(logp.items(), key=lambda item: item[1])[0] if logp else None
        return self._choice(chosen, analysis, logp, combined, policy_agrees=(top_policy == (kind, index)))

    def _choice(self, option: Option, analysis: Analysis, logp: dict[tuple[str, int], float],
                combined: dict[tuple[str, int], float], policy_agrees: bool | None) -> BrainChoice:
        reason = option.reason or option.label
        policy_p = math.exp(logp[(option.kind, option.index)]) if (option.kind, option.index) in logp else None
        if policy_agrees is True and policy_p is not None:
            reason += f" · red E1 de acuerdo ({policy_p:.0%})"
        elif policy_agrees is False and logp:
            top = max(logp.items(), key=lambda item: item[1])[0]
            preferred = analysis.option(*top)
            if preferred is not None:
                reason += f" · la red prefería {preferred.label} ({math.exp(logp[top]):.0%})"
        scores = self._scores(analysis, logp, combined)
        if option.kind == "item":
            scores["catch"] = 1.0
            scores["fallback_move"] = option.details.get("fallback_move", 0.0)
        if option.kind == "run":
            fallback = max((item for item in analysis.options if item.kind == "move" and item.usable),
                           key=lambda item: item.value, default=None)
            scores["fallback_move"] = float(fallback.index) if fallback is not None else 0.0
        return BrainChoice(option.kind, option.index, reason, scores)

    @staticmethod
    def _scores(analysis: Analysis, logp: dict[tuple[str, int], float],
                combined: dict[tuple[str, int], float]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for option in analysis.options:
            if not option.legal:
                continue
            key = _score_key(option)
            scores[key] = round(option.value, 4)
            if (option.kind, option.index) in logp:
                scores[f"red:{key}"] = round(math.exp(logp[(option.kind, option.index)]), 4)
        return scores

    # ------------------------------------------------------------------ live status helper
    def describe(self, view: BattleView, choice: BrainChoice) -> dict[str, object]:
        """The live-status ``battle`` block: me, enemy, choice, reason, effectiveness, scores."""
        analysis = self.last_analysis if self.last_analysis is not None and self.last_analysis.view is view \
            else self.strategist.analyze(view)
        option = analysis.option(choice.kind, choice.index)
        effectiveness = None
        if option is not None and choice.kind == "move":
            effectiveness = option.multiplier
        label = option.label if option is not None else choice.kind
        return {"me": f"{display_name(view.me.species)} Nv.{view.me.level} {view.me.hp}/{view.me.max_hp}",
                "enemy": f"{display_name(view.enemy.species)} Nv.{view.enemy.level} {view.enemy.hp_fraction:.0%}",
                "choice": {"move": label, "switch": f"cambio a {label}", "run": "huir", "item": "captura"}
                .get(choice.kind, label),
                "reason": choice.reason, "effectiveness": effectiveness,
                "scores": {key: value for key, value in choice.scores.items() if not key.startswith("red:")}}


def _score_key(option: Option) -> str:
    if option.kind == "move":
        return option.label
    if option.kind == "switch":
        return f"cambio:{option.label}"
    return {"run": "huir", "item": "captura"}.get(option.kind, option.kind)


__all__ = ["BattleBrain", "DEFAULT_CHECKPOINT", "MODES", "PolicyNet", "STRATEGY_WEIGHT", "UNSEEN_LOGP",
           "resolve_checkpoint"]
