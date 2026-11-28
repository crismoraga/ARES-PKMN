"""Live dashboard feed for the FireRed/Emerald campaign worker (``core.live_status`` contract).

The campaign worker owns the emulator; this module only *reads* the orchestrator
(telemetry, metrics, audit events, framebuffer) about once per wall-second and
publishes ``artifacts/live/campaign-<game>/``. Every failure is swallowed: the live
view must never become a gameplay fault.

Text meant for viewers (goal, events, battle reason) is plain Spanish.
"""
from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.live_status import LIVE_ROOT, publish

_OUTCOMES = {1: "victoria", 2: "derrota", 3: "empate", 4: "huida", 5: "huida (teletransporte)",
             6: "el rival huyó", 7: "captura", 8: "sin Safari Balls", 9: "abandono", 10: "el rival se teletransportó"}

# Plain-Spanish objective per campaign milestone (Spanish localisation names).
_MILESTONES_ES: dict[str, str] = {
    # FireRed
    "OAK_ROUTE1_TRIGGER": "Salir de Pueblo Paleta hacia la Ruta 1 (aparece Oak)",
    "STARTER_BULBASAUR": "Elegir a Bulbasaur en el laboratorio de Oak",
    "RIVAL_LAB": "Combatir contra el rival en el laboratorio",
    "OAK_PARCEL_PICKUP": "Recoger el paquete de Oak en la Tienda de Ciudad Verde",
    "OAK_PARCEL": "Entregar el paquete al Profesor Oak",
    "GYM_1_BROCK": "Ir al Gimnasio de Plateada (Brock)",
    "MT_MOON_FOSSIL_GUARD": "Cruzar el Monte Moon y vencer al Supernecio Miguel",
    "MOUNT_MOON": "Tomar un fósil en el Monte Moon",
    "CERULEAN_RIVAL": "Vencer al rival en Ciudad Celeste",
    "BILL_RESCUE": "Ayudar a Bill en su cabaña (Ruta 25)",
    "SS_TICKET": "Recibir el Ticket del S.S. Anne de Bill",
    "GYM_2_MISTY": "Ir al Gimnasio de Celeste (Misty)",
    "SS_ANNE_CUT": "Conseguir la MO Corte del capitán del S.S. Anne",
    "SURGE_SWITCHES": "Encontrar los interruptores del Gimnasio de Carmín",
    "GYM_3_SURGE": "Vencer al Teniente Surge (Gimnasio de Carmín)",
    "CELADON_TEA": "Conseguir el Té en Ciudad Azulona",
    "ROCKET_POSTER_GUARD": "Vencer al Rocket del Casino de Azulona",
    "ROCKET_POSTER": "Abrir la Guarida Rocket (póster del Casino)",
    "ROCKET_KEY_GUARD": "Vencer al Rocket que guarda la Llave Ascensor",
    "ROCKET_LIFT_KEY": "Conseguir la Llave Ascensor",
    "ROCKET_BOSS_GUARD_LEFT": "Vencer al guardia Rocket (izquierda)",
    "ROCKET_BOSS_GUARD_RIGHT": "Vencer al guardia Rocket (derecha)",
    "ROCKET_GIOVANNI": "Vencer a Giovanni en la Guarida Rocket",
    "SILPH_SCOPE": "Recoger el Scope Silph",
    "GYM_4_ERIKA": "Ir al Gimnasio de Azulona (Erika)",
    "FUJI_RESCUE": "Rescatar al Señor Fuji en la Torre Pokémon",
    "POKE_FLUTE": "Recibir la Poké Flauta",
    "SAFARI_GOLD_TEETH": "Encontrar los Dientes de Oro en la Zona Safari",
    "HM_SURF": "Conseguir la MO Surf",
    "HM_STRENGTH": "Conseguir la MO Fuerza",
    "GYM_5_KOGA": "Ir al Gimnasio de Fucsia (Koga)",
    "SILPH_CARD_KEY": "Conseguir la Llave Magnética en Silph S.A.",
    "SILPH_GIOVANNI": "Vencer a Giovanni en Silph S.A.",
    "SILPH_CO": "Liberar Silph S.A.",
    "LAPRAS_GIFT": "Recibir a Lapras en Silph S.A.",
    "GYM_6_SABRINA": "Ir al Gimnasio de Azafrán (Sabrina)",
    "MANSION_SECRET_KEY": "Conseguir la Llave Secreta en la Mansión Pokémon",
    "GYM_7_BLAINE": "Ir al Gimnasio de Isla Canela (Blaine)",
    "GYM_8_GIOVANNI": "Ir al Gimnasio de Ciudad Verde (Giovanni)",
    "ELITE_LORELEI": "Alto Mando: vencer a Lorelei",
    "ELITE_BRUNO": "Alto Mando: vencer a Bruno",
    "ELITE_AGATHA": "Alto Mando: vencer a Agatha",
    "ELITE_LANCE": "Alto Mando: vencer a Lance",
    # Emerald
    "SET_CLOCK": "Ajustar el reloj de casa",
    "WATCH_TV_REPORT": "Ver el reportaje de la tele con mamá",
    "MEET_RIVAL": "Conocer a la vecina rival",
    "BIRCH_RESCUE_TRIGGER": "Ir a la Ruta 101 a socorrer al Profesor Abedul",
    "STARTER_MUDKIP": "Elegir a Mudkip y salvar al Profesor Abedul",
    "BIRCH_LAB_STARTER": "Volver al laboratorio del Profesor Abedul",
    "RIVAL_ROUTE103": "Combatir con la rival en la Ruta 103",
    "POKEDEX": "Recibir la Pokédex",
    "WALLY_TUTORIAL": "Ayudar a Blasco a capturar su primer Pokémon",
    "PETALBURG_WOODS_AQUA": "Bosque Petalia: vencer al Team Aqua",
    "GYM_1_ROXANNE": "Ir al Gimnasio de Ciudad Férrica (Petra)",
    "DEVON_GOODS": "Recuperar los bienes Devon en el Túnel Fervegal",
    "DEVON_RETURN": "Devolver los bienes Devon",
    "DEVON_POKENAV": "Recibir el PokéNav del Señor Stone",
    "HM_CUT": "Conseguir la MO Corte",
    "STEVEN_LETTER": "Entregar la carta a Máximo en la Cueva Granito",
    "GYM_2_BRAWLY": "Ir al Gimnasio de Pueblo Azuliza (Marcial)",
    "STERN_GOODS": "Entregar los bienes Devon al Capitán Aguado (Portual)",
    "RIVAL_ROUTE110": "Combatir con la rival en la Ruta 110",
    "WALLY_MAUVILLE": "Vencer a Blasco en Ciudad Malvalona",
    "GYM_3_WATTSON": "Ir al Gimnasio de Malvalona (Erico)",
    "HM_ROCK_SMASH": "Conseguir la MO Golpe Roca",
    "METEOR_FALLS_TEAMS": "Ir a las Cataratas Meteoro (Team Magma y Aqua)",
    "MT_CHIMNEY": "Vencer a Magno en la cima del Monte Cenizo",
    "GYM_4_FLANNERY": "Ir al Gimnasio de Pueblo Lavacalda (Candela)",
    "PETALBURG_RANDALL": "Gimnasio de Petalia: vencer a Randall",
    "PETALBURG_PARKER": "Gimnasio de Petalia: vencer a Parker",
    "PETALBURG_JODY": "Gimnasio de Petalia: vencer a Jody",
    "GYM_5_NORMAN": "Ir al Gimnasio de Petalia (Norman)",
    "WEATHER_INSTITUTE": "Liberar el Instituto Meteorológico",
    "HM_FLY": "Conseguir la MO Vuelo",
    "DEVON_SCOPE": "Recibir el Detector Devon de Máximo",
    "GYM_6_WINONA": "Ir al Gimnasio de Arborada (Alana)",
    "MT_PYRE_EMBLEM": "Monte Pírico: recuperar el orbe",
    "MAGMA_HIDEOUT": "Infiltrarse en la Guarida Magma",
    "SLATEPORT_SUBMARINE": "Detener al Team Aqua en el puerto de Portual",
    "AQUA_HIDEOUT": "Infiltrarse en la Guarida Aqua",
    "GYM_7_TATE_LIZA": "Ir al Gimnasio de Algaria (Vito y Leti)",
    "SPACE_CENTER": "Defender el Centro Espacial de Algaria",
    "HM_DIVE": "Conseguir la MO Buceo",
    "SEAFLOOR_CAVERN": "Llegar al fondo de la Cueva Abisal",
    "WALLACE_ORIGIN": "Hablar con Plubio en la Cueva Ancestral",
    "RAYQUAZA_AWAKEN": "Despertar a Rayquaza en el Pilar Celeste",
    "SOOTOPOLIS_MAXIE": "Arrecípolis: encontrar a Magno",
    "SOOTOPOLIS_TRUCE": "Arrecípolis: tregua entre Aquiles y Magno",
    "HM_WATERFALL": "Conseguir la MO Cascada",
    "GYM_8_JUAN": "Ir al Gimnasio de Arrecípolis (Galano)",
    "VICTORY_ROAD_WALLY": "Calle Victoria: vencer a Blasco",
    "ELITE_SIDNEY": "Alto Mando: vencer a Sixto",
    "ELITE_PHOEBE": "Alto Mando: vencer a Fátima",
    "ELITE_GLACIA": "Alto Mando: vencer a Nívea",
    "ELITE_DRAKE": "Alto Mando: vencer a Dracón",
    "CHAMPION": "Vencer al Campeón",
    "ELITE_FOUR": "Entrar en el Salón de la Fama",
    "POSTGAME_COMPLETE": "Campaña completada",
}


def milestone_text(milestone: str) -> str:
    """Spanish description of a campaign milestone (generic fallback for unknown ids)."""
    if milestone in _MILESTONES_ES:
        return _MILESTONES_ES[milestone]
    return "Hito: " + milestone.replace("_", " ").lower() if milestone else "Explorar"


def goal_text(directive: Mapping[str, Any], next_milestone: str = "") -> str:
    """The orchestrator's current directive in plain Spanish."""
    mode = str(directive.get("mode", "") or "")
    milestone = str(directive.get("milestone") or directive.get("pre_battle") or next_milestone or "")
    target = milestone_text(milestone) if milestone else ""
    if mode == "HEAL_POKECENTER":
        before = directive.get("pre_battle")
        text = "Curar al equipo en el Centro Pokémon"
        return text + (f" antes de: {milestone_text(str(before))}" if before else "")
    if mode == "OPTIMAL_TRAINING_CURRICULUM":
        level = directive.get("target_level")
        route = str(directive.get("training_route") or "").replace("_", " ")
        text = "Entrenar al equipo" + (f" hasta nivel {level}" if level else "")
        text += f" en {route}" if route else ""
        return text + (f" para: {target}" if target else "")
    if mode == "SHOP":
        return "Comprar objetos (Pociones / Poké Balls) en la Tienda"
    if mode == "TEACH_FIELD_MOVE":
        return "Enseñar un movimiento de campo (MO) a un miembro del equipo" + (f" para: {target}" if target else "")
    if mode == "EXECUTE_PUZZLE":
        puzzle = str(directive.get("puzzle") or "").replace("_", " ")
        return f"Resolver el puzle {puzzle}".strip() + (f" ({target})" if target else "")
    if mode == "UNLOCK_GATE":
        return "Desbloquear el paso cerrado" + (f" para: {target}" if target else "")
    if mode == "EXPLORE_DETOUR":
        return "Explorar un desvío para salir de un atasco" + (f" ({target})" if target else "")
    if mode == "POSTGAME_COMPLETE":
        return "Campaña completada"
    return target or "Explorar"


def _map_name(controller: Any, key: tuple[int, int]) -> str:
    try:
        raw = controller.topology.catalog.maps.get(key)
        if raw is not None:
            return str(raw["name"])
    except (AttributeError, KeyError, TypeError):
        pass
    return f"{key[0]}.{key[1]}"


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _pretty(name: str) -> str:
    return name.replace("_", " ")


def describe_event(controller: Any, event: Mapping[str, Any]) -> str | None:
    """One audit event as a short Spanish line, or None when it is not worth showing."""
    kind = event.get("kind")
    frame = event.get("frame", 0)
    prefix = f"[{frame}] "
    if kind == "milestone_observed":
        milestone = str(event.get("milestone", ""))
        return prefix + f"Hito completado: {milestone_text(milestone)}"
    if kind == "battle_entered":
        return prefix + ("Combate contra un entrenador" if event.get("trainer") else "Combate contra un Pokémon salvaje")
    if kind == "native_battle_result":
        outcome = _OUTCOMES.get(int(event.get("outcome", 0) or 0), f"resultado {event.get('outcome')}")
        return prefix + f"Fin del combate: {outcome} ({event.get('turns', '?')} turnos)"
    if kind == "level_up":
        name = event.get("name") or event.get("species", "")
        return prefix + f"Subida de nivel: {name} -> nivel {event.get('level', '?')}"
    if kind == "evolution":
        return prefix + f"Evolución: {event.get('from_name', '?')} -> {event.get('to_name', '?')}"
    if kind == "party_healed":
        return prefix + "Equipo curado en el Centro Pokémon"
    if kind == "party_joined":
        return prefix + f"Nuevo miembro del equipo: {event.get('name', event.get('species', '?'))}"
    if kind == "stall_detected":
        recovery = event.get("recovery")
        reason = recovery.get("reason", "") if isinstance(recovery, Mapping) else ""
        return prefix + f"Atasco detectado ({reason})"
    if kind == "stall_recovery":
        return prefix + f"Recuperación de atasco: {event.get('action', '')}"
    if kind == "stall_rewind":
        return prefix + f"Rebobinado al frame {event.get('to_frame', '?')} ({event.get('reason', '')})"
    if kind == "stall_strategy":
        return prefix + f"Cambio de estrategia ante el atasco: {event.get('strategy', event.get('action', ''))}"
    if kind == "strategy_shift":
        return prefix + f"Cambio de estrategia: {event.get('change', '')} ({event.get('objective', '')})"
    if kind == "objective_defeat":
        return prefix + (f"Derrota contra {milestone_text(str(event.get('objective', '')))} con el líder a nivel "
                         f"{event.get('lead_level', '?')}; se preparará mejor")
    if kind == "readiness_plan":
        return prefix + f"Plan de preparación: {event.get('summary', event.get('reason', ''))}"
    if kind == "map_transition":
        try:
            source = _pretty(_map_name(controller, (int(event["source"][0]), int(event["source"][1]))))
            destination = _pretty(_map_name(controller, (int(event["destination"][0]), int(event["destination"][1]))))
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        return prefix + f"Mapa: {source} -> {destination}"
    if kind == "shop_opened":
        return prefix + "Entrando a comprar en la Tienda"
    if kind == "evolution_cancelled":
        return prefix + "Evolución cancelada"
    return None


class CampaignLivePublisher:
    """Rate-limited bridge between the orchestrator and ``core.live_status.publish``."""

    def __init__(self, game: str, *, run_id: str | None = None, interval: float = 1.0,
                 root: Path | None = None, agent: str | None = None) -> None:
        self.game = game
        self.run_id = run_id or f"campaign-{game}"
        self.interval = interval
        self.root = root
        self.agent = agent or "ARES: planificador HTN + navegación por warps + cerebro de combate"
        self._last = 0.0
        self._fps_anchor: tuple[float, int] | None = None
        self._fps = 0.0
        self._audit_len = 0
        self._events: deque[str] = deque(maxlen=20)
        self._stalls = self._session_battles = 0
        self._maps: set[str] = set()
        self._visited_size = -1
        self._load_previous()

    def _directory(self) -> Path:
        return (self.root or LIVE_ROOT) / self.run_id

    def _load_previous(self) -> None:
        """Keep the cumulative map list and recent events across worker restarts."""
        try:
            previous = json.loads((self._directory() / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        names = previous.get("map_names")
        if isinstance(names, list):
            self._maps.update(str(name) for name in names[:2048])
        events = previous.get("events")
        if isinstance(events, list):
            self._events.extend(str(line) for line in events[-20:])

    def maybe_publish(self, controller: Any, *, force: bool = False, running: bool = True,
                      error: str = "") -> bool:
        now = time.monotonic()
        if not force and now - self._last < self.interval:
            return False
        self._last = now
        try:
            status = self.build_status(controller, now, running=running, error=error)
            publish(self.run_id, status, frame=self._frame(controller), root=self.root)
        except Exception:  # Monitoring must never become a gameplay fault.
            return False
        return True

    @staticmethod
    def _frame(controller: Any) -> Any:
        try:
            return controller.bridge.framebuffer()
        except Exception:
            return None

    def _ingest_events(self, controller: Any) -> None:
        audit = getattr(controller, "audit", None)
        if not isinstance(audit, list):
            return
        if len(audit) < self._audit_len:  # the orchestrator trims its in-memory audit
            self._audit_len = max(0, self._audit_len - 5000)
            if self._audit_len > len(audit):
                self._audit_len = max(0, len(audit) - 50)
        fresh = audit[self._audit_len:][-400:]
        self._audit_len = len(audit)
        for event in fresh:
            if not isinstance(event, Mapping):
                continue
            kind = event.get("kind")
            if kind == "stall_detected":
                self._stalls += 1
            elif kind == "battle_entered":
                self._session_battles += 1
            elif kind == "map_transition":
                try:
                    for key in ("source", "destination"):
                        value = event[key]
                        self._maps.add(_map_name(controller, (int(value[0]), int(value[1]))))
                except (KeyError, IndexError, TypeError, ValueError):
                    pass
            line = describe_event(controller, event)
            if line is not None:
                self._events.append(line)

    def _ingest_visited(self, controller: Any) -> None:
        """Maps of the restored timeline: the watchdog keeps every visited (group, num, x, y) tile."""
        visited = getattr(getattr(controller, "watchdog", None), "visited", None)
        if not isinstance(visited, set) or len(visited) == self._visited_size:
            return
        self._visited_size = len(visited)
        keys = {(int(location[0]), int(location[1])) for location in list(visited)
                if isinstance(location, tuple) and len(location) >= 2}
        self._maps.update(_map_name(controller, key) for key in keys)

    def build_status(self, controller: Any, now: float | None = None, *, running: bool = True,
                     error: str = "") -> dict[str, Any]:
        now = time.monotonic() if now is None else now
        frames = int(controller.bridge.frame_count)
        if self._fps_anchor is not None and now > self._fps_anchor[0] and frames >= self._fps_anchor[1]:
            elapsed = now - self._fps_anchor[0]
            if elapsed >= 0.5:
                self._fps = (frames - self._fps_anchor[1]) / elapsed
                self._fps_anchor = (now, frames)
        else:
            self._fps_anchor = (now, frames)
        self._ingest_events(controller)
        telemetry: dict[str, Any] = {}
        try:
            telemetry = dict(controller.telemetry())
        except Exception:
            telemetry = {}
        milestones = _mapping(telemetry.get("milestones"))
        completed = [str(item) for item in milestones.get("completed", [])]
        next_milestone = str(milestones.get("next", ""))
        trainer = _mapping(telemetry.get("trainer"))
        map_name = str(telemetry.get("map_name") or "")
        if map_name:
            self._maps.add(map_name)
        self._ingest_visited(controller)
        metrics = getattr(controller, "metrics", None)
        directive = getattr(controller, "_last_directive", {}) or {}
        status: dict[str, Any] = {
            "kind": "campaign", "game": self.game, "generation": 3, "agent": self.agent,
            "frames": frames, "fps": round(self._fps, 1), "running": running,
            "badges": int(trainer.get("badge_count", 0) or 0),
            "milestones": completed,
            "milestones_total": int(milestones.get("total", 0) or 0),
            "next_milestone": next_milestone,
            "maps_visited": len(self._maps),
            "map_names": sorted(self._maps)[:2048],
            "map": _pretty(map_name),
            "party": self._party(telemetry),
            "goal": goal_text(directive, next_milestone),
            "directive": {key: directive.get(key) for key in ("mode", "milestone", "reason", "preparation_reason",
                                                               "target_level", "training_route") if key in directive},
            "battle": self._battle(controller, telemetry),
            "stats": self._stats(controller, metrics),
            "knowledge": self._knowledge(controller, telemetry),
            "events": list(self._events),
        }
        if trainer:
            status["trainer"] = {key: trainer.get(key) for key in ("money", "dex_owned", "dex_seen", "play_time")}
        if error:
            status["error"] = error
        return status

    @staticmethod
    def _party(telemetry: Mapping[str, Any]) -> list[dict[str, Any]]:
        party = telemetry.get("party")
        if not isinstance(party, list):
            return []
        result: list[dict[str, Any]] = []
        for mon in party:
            if not isinstance(mon, Mapping):
                continue
            moves = [str(move.get("name", "")) for move in mon.get("moves", []) if isinstance(move, Mapping)]
            result.append({"species": mon.get("name") or mon.get("species"), "nickname": mon.get("nickname", ""),
                           "level": mon.get("level"), "hp": mon.get("hp"), "max_hp": mon.get("maxHP"),
                           "status": mon.get("status", ""), "types": list(mon.get("types", []) or []),
                           "moves": moves})
        return result

    @staticmethod
    def _battle(controller: Any, telemetry: Mapping[str, Any]) -> dict[str, Any] | None:
        battle = telemetry.get("battle")
        if not isinstance(battle, Mapping):
            return None
        own = battle.get("own") if isinstance(battle.get("own"), Mapping) else None
        enemy = battle.get("enemy") if isinstance(battle.get("enemy"), Mapping) else None
        result: dict[str, Any] = {"trainer": bool(battle.get("trainer")), "me": own, "enemy": enemy}
        decision = getattr(controller, "last_decision", None)
        brain = getattr(controller, "last_battle_choice", None)
        if brain is not None:
            result["choice"] = getattr(brain, "kind", None) or (brain.get("kind") if isinstance(brain, Mapping) else None)
            result["index"] = getattr(brain, "index", None) if not isinstance(brain, Mapping) else brain.get("index")
            result["reason"] = getattr(brain, "reason", None) if not isinstance(brain, Mapping) else brain.get("reason")
            scores = getattr(brain, "scores", None) if not isinstance(brain, Mapping) else brain.get("scores")
            if scores is not None:
                result["scores"] = scores
        if isinstance(decision, Mapping):
            result.setdefault("choice", decision.get("action"))
            result["move"] = decision.get("move")
            result.setdefault("reason", decision.get("reason"))
            result["risk"] = decision.get("risk")
            result["decision_turn"] = decision.get("turn")
            result["effectiveness"] = CampaignLivePublisher._effectiveness(controller, decision)
        return result

    @staticmethod
    def _effectiveness(controller: Any, decision: Mapping[str, Any]) -> float | None:
        """Type multiplier of the chosen move against the visible enemy (learned chart first)."""
        if decision.get("action") != "move":
            return None
        try:
            index = int(decision.get("index", -1))
            me, enemy = controller.bridge.get_battle_pokemon(0), controller.bridge.get_battle_pokemon(1)
            if not 0 <= index < 4 or not me.moves[index]:
                return None
            move_type = int(controller.catalog.move(int(me.moves[index])).move_type)
            defender = tuple(dict.fromkeys((int(enemy.type1), int(enemy.type2))))
            return round(float(controller._type_multiplier(move_type, defender)), 3)
        except Exception:
            return None

    def _stats(self, controller: Any, metrics: Any) -> dict[str, Any]:
        stats: dict[str, Any] = {"stalls": self._stalls, "session_battles": self._session_battles}
        for key in ("battles", "wins", "losses", "captures", "faints", "recoveries", "rewinds", "heals",
                    "level_ups", "evolutions", "map_transitions", "faults", "detours", "field_moves"):
            value = getattr(metrics, key, None)
            if value is not None:
                stats[key] = value
        watchdog: Any = getattr(controller, "watchdog", None)
        try:
            stats["unproductive_frames"] = int(watchdog.unproductive_frames(int(controller.bridge.frame_count)))
            stats["story_unproductive_frames"] = int(watchdog.strategic_unproductive_frames)
        except Exception:
            pass
        return stats

    @staticmethod
    def _knowledge(controller: Any, telemetry: Mapping[str, Any]) -> dict[str, Any]:
        knowledge: dict[str, Any] = {}
        plan = telemetry.get("plan")
        if isinstance(plan, Mapping):
            chart = plan.get("type_chart")
            if isinstance(chart, Mapping):
                knowledge["type_chart_cells"] = chart.get("known_cells")
                knowledge["type_observations"] = chart.get("observations")
            for key in ("counter_types", "threat_types", "boss_roster", "catch_targets", "training_route"):
                if plan.get(key):
                    knowledge[key] = plan.get(key)
        preparation = telemetry.get("preparation")
        if isinstance(preparation, Mapping) and preparation:
            knowledge["lessons"] = dict(list(preparation.items())[:24])
        barriers = telemetry.get("learned_barriers")
        if isinstance(barriers, (list, dict)):
            knowledge["learned_barriers"] = len(barriers)
        return knowledge
