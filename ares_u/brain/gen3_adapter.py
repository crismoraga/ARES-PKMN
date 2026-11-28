"""Gen-3 campaign adapter: mGBA bridge (FireRed/Emerald RAM) -> ``BattleView`` -> ``BattleDecision``.

``core.orchestrator.AresOrchestrator.choose_battle_decision`` calls ``learned_battle_decision`` when
``ARES_BATTLE_BRAIN=learned`` (the default, see ``brain_mode``). The function returns ``None`` to hand
the turn to the analytic heuristic in the situations the brain does not own: doubles, captures the
campaign wants, story escapes (the heuristic knows the directives and the bag) and a brain choice
of ``run``/``item``. Any exception is caught by the caller, which logs ``battle_brain_fallback`` and
runs the heuristic, so a bad read can never stall a battle.

Honesty: the enemy is described with public information only (species, level, HP fraction, types,
status, stat stages and the moves it has used, from ``OpponentKnowledge``); ``stats=None`` lets the
brain estimate the hidden stats from base stats and level.

Pure helpers (``status_of``, ``stages_of``, ``mon_from_battler``, ``mon_from_party``, ``build_view``,
``decision_from_choice``) need no emulator and are unit-tested with a fake catalog.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .api import GEN3_INDEX_TYPE, BattleView, BrainChoice, MonView, MoveView, category_for
from .strategy import national_number

ENV = "ARES_BATTLE_BRAIN"
DEFAULT_MODE = "learned"  # stronger per artifacts/brain/report.md (paired Stadium evaluation)
# Gen-3 statStages order: hp, atk, def, spe, spa, spd, acc, eva (6 = neutral)
_STAGE_KEYS = (None, "atk", "def", "spe", "spa", "spd", "acc", "eva")
_ESCAPE_MODES = ("EXECUTE_MILESTONE", "TEACH_FIELD_MOVE", "HEAL_POKECENTER")


class Catalog(Protocol):
    def species(self, species_id: int) -> Any: ...

    def move(self, move_id: int) -> Any: ...


def brain_mode() -> str:
    """``learned`` (battle brain) or ``heuristic`` (analytic campaign heuristic) from ``ARES_BATTLE_BRAIN``."""
    value = os.environ.get(ENV, "").strip().lower()
    return value if value in ("learned", "heuristic") else DEFAULT_MODE


def status_of(status1: int) -> str:
    """Gen-3 ``status1`` bit field -> brain status code."""
    value = int(status1)
    if value & 0x7:
        return "slp"
    if value & 0x80:
        return "tox"
    if value & 0x8:
        return "psn"
    if value & 0x10:
        return "brn"
    if value & 0x20:
        return "frz"
    if value & 0x40:
        return "par"
    return ""


def stages_of(raw: Sequence[int]) -> dict[str, int]:
    return {key: int(raw[index]) - 6 for index, key in enumerate(_STAGE_KEYS) if key is not None and index < len(raw)}


def _types(type1: int, type2: int) -> tuple[str, ...]:
    names = [GEN3_INDEX_TYPE.get(int(value), "???") for value in (type1, type2)]
    return tuple(dict.fromkeys(names))


def _move_view(catalog: Catalog, move_id: int, pp: int | None) -> MoveView:
    move = catalog.move(int(move_id))
    move_type = GEN3_INDEX_TYPE.get(int(move.move_type), "???")
    power = int(move.power)
    return MoveView(name=str(move.name), type=move_type, power=power, accuracy=int(move.accuracy),
                    pp=int(move.pp if pp is None else pp), category=category_for(move_type, power, 3),
                    gen3_move_id=int(move_id), max_pp=int(move.pp), priority=int(getattr(move, "priority", 0)))


def _species_name(catalog: Catalog, species: int) -> str:
    try:
        name = str(catalog.species(int(species)).name).strip()
    except Exception:  # noqa: BLE001 - a name is cosmetic
        name = ""
    return name or f"#{national_number(int(species))}"


def mon_from_battler(mon: Any, catalog: Catalog, own: bool, revealed: Sequence[int] = ()) -> MonView:
    """``BattlePokemonStruct`` -> ``MonView`` (own: full knowledge; enemy: public information only)."""
    species = int(mon.species)
    if own:
        moves = [_move_view(catalog, int(mon.moves[slot]), int(mon.pp[slot])) for slot in range(4)
                 if int(mon.moves[slot])]
        stats: dict[str, int] | None = {"atk": int(mon.attack), "def": int(mon.defense), "spa": int(mon.spAttack),
                                        "spd": int(mon.spDefense), "spe": int(mon.speed)}
    else:
        moves = [_move_view(catalog, int(move), None) for move in revealed if int(move)]
        stats = None
    return MonView(species=_species_name(catalog, species), national_dex=national_number(species),
                   level=int(mon.level), hp=int(mon.hp), max_hp=max(1, int(mon.maxHP)),
                   types=_types(int(mon.type1), int(mon.type2)), status=status_of(int(mon.status1)),
                   moves=moves, stats=stats, stages=stages_of(list(mon.statStages)))


def mon_from_party(mon: dict[str, Any], catalog: Catalog) -> MonView:
    """Decrypted party dict (``get_decrypted_player_party``) -> ``MonView``."""
    species = int(mon["species"])
    info = catalog.species(species)
    type1, type2 = (int(value) for value in info.types)
    moves = [_move_view(catalog, int(move), int(pp)) for move, pp in zip(mon["moves"], mon["pp"]) if int(move)]
    return MonView(species=_species_name(catalog, species), national_dex=national_number(species),
                   level=int(mon["level"]), hp=int(mon["hp"]), max_hp=max(1, int(mon["maxHP"])),
                   types=_types(type1, type2), status=status_of(int(mon.get("status", 0))), moves=moves,
                   stats={"atk": int(mon["attack"]), "def": int(mon["defense"]), "spa": int(mon["spAttack"]),
                          "spd": int(mon["spDefense"]), "spe": int(mon["speed"])})


@dataclass
class Turn:
    """What the bridge says about this decision point (from ``encode_battle``)."""
    active_index: int
    move_legal: tuple[bool, bool, bool, bool]
    switchable: tuple[int, ...]  # party indices the game lets us send in
    forced_switch: bool
    target_index: int = 0
    doubles: bool = False


def turn_of(features: Any, party: Sequence[dict[str, Any]]) -> Turn:
    mask = features.action_mask
    legal = tuple(bool(mask[0, slot]) for slot in range(4))
    switchable = tuple(int(index) for offset, index in enumerate(features.switch_party_indices)
                       if bool(mask[0, 4 + offset]) and int(party[int(index)]["hp"]) > 0)
    target = next((index for index in (0, 1) if bool(features.target_mask[0, index])), 0)
    return Turn(int(features.active_party_index), (legal[0], legal[1], legal[2], legal[3]), switchable,
                bool(features.forced_switch), target, bool(features.doubles))


def build_view(game: str, me: Any, enemy: Any, party: Sequence[dict[str, Any]], turn: Turn, trainer: bool,
               catalog: Catalog, revealed: Sequence[int] = (), turn_number: int = 0) -> BattleView:
    """Assemble the ``BattleView``; the active slot uses the in-battle struct (stages, current types)."""
    views = [mon_from_party(mon, catalog) for mon in party]
    me_view = mon_from_battler(me, catalog, True)
    if 0 <= turn.active_index < len(views):
        views[turn.active_index] = me_view
    legal = tuple(slot for slot in range(len(me_view.moves)) if turn.move_legal[slot])
    return BattleView(game=game, generation=3, me=me_view, enemy=mon_from_battler(enemy, catalog, False, revealed),
                      party=views, active_index=turn.active_index, is_wild=not trainer,
                      can_switch=bool(turn.switchable), can_run=False, turn=turn_number,
                      legal_moves=legal if not turn.forced_switch else ())


def decision_from_choice(choice: BrainChoice, view: BattleView, turn: Turn, proof: Any,
                         win: float | None = None) -> Any | None:
    """``BrainChoice`` -> ``core.orchestrator.BattleDecision``; ``None`` = let the heuristic decide.

    ``win`` is the strategist's win-probability estimate of the chosen option; the decision's
    ``estimated_risk`` is its complement (an estimate, never a proof).
    """
    from core.orchestrator import BattleDecision

    risk = 1.0 - max(0.0, min(1.0, float(win))) if win is not None else 0.5
    model = "battle_brain_win_probability_complement"
    reason = f"brain: {choice.reason}"[:240]
    if choice.kind == "move":
        if turn.forced_switch:
            return None
        legal = [slot for slot in range(4) if turn.move_legal[slot]]
        if legal and choice.index not in legal:
            return None
        index = choice.index if legal else 0  # no legal move: the engine plays Struggle
        return BattleDecision("move", index, 0, risk, proof, reason, turn.target_index, model, risk)
    if choice.kind == "switch":
        if choice.index not in turn.switchable or choice.index >= len(view.party):
            return None
        return BattleDecision("switch", choice.index, 0, risk, proof, reason, 0, model, risk)
    return None  # run / item: the heuristic owns escapes and the bag


def _brain(controller: Any) -> Any:
    brain = getattr(controller, "_ares_battle_brain", None)
    if brain is None:
        from .brain import BattleBrain

        brain = BattleBrain("brain")
        controller._ares_battle_brain = brain
    return brain


def learned_battle_decision(controller: Any, party: Sequence[dict[str, Any]], features: Any, me: Any,
                            enemy: Any) -> Any | None:
    """Brain decision for this turn of a campaign battle, or ``None`` to defer to the heuristic."""
    from core.safety.z3_integer_verifier import DamageResult

    controller.last_battle_choice = None
    controller.last_brain_deferral = ""
    turn = turn_of(features, party)
    if turn.doubles or controller.catalog is None:
        controller.last_brain_deferral = "doubles" if turn.doubles else "no_catalog"
        return None
    _, trainer = controller.bridge.is_in_battle()
    species = int(enemy.species)
    if not trainer and not turn.forced_switch:
        wanted = species in getattr(controller, "desired_species", set()) and \
            species not in getattr(controller, "owned_species", set())
        directive = getattr(controller, "_last_directive", {}) or {}
        escape = (bool(getattr(controller, "native", False)) and int(me.hp) > 0 and
                  int(getattr(controller, "_battle_escape_attempts", 0)) < 2 and directive.get("mode") in _ESCAPE_MODES)
        if wanted or escape:
            controller.last_brain_deferral = "wanted_capture" if wanted else "story_escape"
            return None
    knowledge = getattr(controller, "knowledge", None)
    revealed = list(getattr(knowledge, "revealed_moves", {}).get(species, [])) if knowledge is not None else []
    view = build_view(str(getattr(controller, "game", "emerald")), me, enemy, party, turn, bool(trainer),
                      controller.catalog, revealed, int(getattr(controller, "_battle_turns", 0)))
    brain = _brain(controller)
    choice = brain.choose(view)
    proof = DamageResult(False, 0, 0, 0, "unsupported", "battle brain: estimated win probability, no proof")
    analysis = brain.last_analysis
    option = analysis.option(choice.kind, choice.index) if analysis is not None else None
    decision = decision_from_choice(choice, view, turn, proof, option.value if option is not None else None)
    if decision is not None:
        live = dict(brain.describe(view, choice))
        live.update({"kind": choice.kind, "index": choice.index, "source": "learned"})
        controller.last_battle_choice = live
    else:
        controller.last_brain_deferral = f"choice_{choice.kind}" + ("_forced" if turn.forced_switch else "")
    return decision


__all__ = ["DEFAULT_MODE", "ENV", "Turn", "brain_mode", "build_view", "decision_from_choice",
           "learned_battle_decision", "mon_from_battler", "mon_from_party", "stages_of", "status_of", "turn_of"]
