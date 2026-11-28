"""``UniversalAgent``: one game-agnostic player loop for any ``GameAdapter``.

Every tick reads a ``Snapshot`` from the adapter and dispatches on the mode:

* ``battle``    -> battle menus (brain choice translated into menu inputs, verified on screen/RAM),
                   effectiveness messages are turned into type-chart evidence;
* ``menu``      -> yes/no questions answered from the knowledge base (and learned from outcomes),
                   unknown menus closed;
* ``text``      -> dialog advanced when the prompt arrow waits for a button;
* ``naming``    -> the naming keyboard is left with a learned key sequence;
* ``overworld`` -> frontier exploration over an online map graph (unused warps/edges, NPCs not
                   talked to, items, signs, unvisited cells), returning to the Pokemon Center when
                   weak, training in grass after a lost trainer battle, and an anti-stuck escalation
                   (tabu, interact around, random burst, reload of the last good savestate).

All the game knowledge the agent uses is either read through the adapter (RAM, map grid, text) or
stored in the ``KnowledgeBase`` that persists across games.
"""
from __future__ import annotations

import importlib
import json
import random
import time
import unicodedata
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .adapter import GameAdapter
from .knowledge import KnowledgeBase, words
from .model import DELTAS, DIRECTIONS, Direction, MapInfo, MapKey, MenuInfo, MonInfo, MoveInfo, Npc, Snapshot
from .world import PASSABLE, Cell, Target, World, bfs, edge_cells, first_step, interaction_cells

STRONG = {"event": 1.0, "new_map": 1.0, "party": 3.0, "item": 2.0, "badge": 10.0, "win": 0.5, "level": 0.3,
          "heal": 0.2}
DEFAULT_STARTER_PREFERENCE = ["water", "grass", "fire"]


def fold(text: str) -> str:
    """Lowercase without accents (matching of Spanish text)."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


class UniversalAgent:
    def __init__(self, adapter: GameAdapter, kb: KnowledgeBase, *, run_id: str, run_dir: Path,
                 sequence: int = 1, publish: bool = True, seed: int = 0, use_brain: bool = True,
                 log: Callable[[str], None] | None = None) -> None:
        self.ad = adapter
        self.kb = kb
        self.run_id = run_id
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.sequence = sequence
        self.publish_enabled = publish
        self.rng = random.Random(seed)
        self.log_fn = log
        self.world = World()
        self.tick = 0
        self.stats: dict[str, int] = {k: 0 for k in (
            "actions", "battles", "wins", "losses", "ran", "caught", "stalls", "recoveries", "reloads", "heals",
            "talks", "naming_screens", "naming_presses", "menu_probes", "menu_exits", "yesno", "wrong_answers",
            "repeated_questions", "blackouts", "type_obs", "walk_blocked", "random_steps", "warps", "maps")}
        self.events: deque[str] = deque(maxlen=40)
        self.milestones: dict[str, dict[str, Any]] = {}
        self.flags: set[str] = set()
        self.prev: Snapshot | None = None
        self.progress = 0.0
        self.progress_events = 0
        self.progress_tick = 0
        self.weak_tick = 0
        self.target: Target | None = None
        self.target_since = 0
        self.target_dist0 = 0
        self.tabu: dict[tuple[Any, ...], int] = {}
        self.bad_goals: dict[tuple[Any, ...], int] = {}
        self.walk_fail: dict[tuple[MapKey, Cell], int] = {}
        self.push_tries: dict[tuple[Any, ...], int] = {}
        self.lastgood: bytes | None = None
        self.lastgood_tick = 0
        self.save_pending = False
        self.goal = "Comenzar"
        self.last_walk: tuple[MapKey, Cell, Direction, int] | None = None
        self.last_talk: dict[str, Any] | None = None
        self.recent_lines: deque[str] = deque(maxlen=12)
        self.screen_sig = b""
        self.static_ticks = 0
        self.question: dict[str, Any] | None = None
        self.questions_seen: dict[str, int] = {}
        self.learn_ctx: dict[str, Any] | None = None
        self.declined_starters = 0
        self.naming_tries = 0
        self.naming_seq: list[str] | None = None
        self.battle: dict[str, Any] | None = None
        self.turn_choice: Any = None
        self.pending_probe: dict[str, Any] | None = None
        self.pending_attack: dict[str, Any] | None = None
        self.attack_ctx: dict[str, Any] | None = None
        self.seen_battle_lines: deque[str] = deque(maxlen=8)
        self.train_level = 0
        self.heal_spots: list[dict[str, Any]] = []
        self.return_mode: dict[str, Any] | None = None
        self.completed: tuple[str, int] | None = None   # (target kind, tick) awaiting its return
        self.stuck_level = 0
        self.stuck_open: dict[str, Any] | None = None
        self.stuck_log: list[dict[str, Any]] = []
        self.positions: deque[tuple[MapKey, int, int]] = deque(maxlen=240)
        self.use_brain = use_brain
        self.brain: Any = None
        self.brain_name = "type-chart fallback"
        self.brain_loaded = False
        self.start_wall = time.time()
        self.last_publish = 0.0
        self.last_kb_save = time.time()
        self.fps = 0.0
        self._fps_mark = (time.time(), 0)
        self.inherited = self._inherited_summary()
        self.timeline: list[dict[str, Any]] = []
        self.settle_until = 0
        self.target_ticks = 0

    # ================================================================== infrastructure
    def log(self, text: str) -> None:
        stamp = f"[{self.ad.frame_count:>9}] {text}"
        self.events.append(stamp)
        if self.log_fn is not None:
            self.log_fn(stamp)

    def _inherited_summary(self) -> dict[str, Any]:
        kb = self.kb
        return {"games": sorted(kb.games), "type_entries": len(kb.types), "menu_labels": len(kb.menus),
                "dialog_words": len(kb.dialog), "heal_events": kb.healing.get("events", 0),
                "skills": sorted(kb.skills), "target_kinds": sorted(kb.targets)}

    def is_tabu(self, key: tuple[Any, ...]) -> bool:
        until = self.tabu.get(key)
        return until is not None and until > self.tick

    def set_tabu(self, key: tuple[Any, ...], ticks: int) -> None:
        self.tabu[key] = self.tick + ticks

    # ================================================================== main loop
    def run(self, max_frames: int, max_seconds: float = 1e12, checkpoint_every: float = 120.0) -> dict[str, Any]:
        start_frame = self.ad.frame_count
        last_ckpt = time.time()
        while self.ad.frame_count - start_frame < max_frames and time.time() - self.start_wall < max_seconds:
            self.step()
            now = time.time()
            if now - self.last_publish >= 1.0:
                self.publish()
            if now - last_ckpt >= checkpoint_every:
                self.checkpoint()
                last_ckpt = now
        self.checkpoint()
        self.publish()
        return self.report()

    def step(self) -> None:
        frame0 = self.ad.frame_count
        self._step()
        if self.ad.frame_count == frame0:
            self.ad.wait(1)  # every tick consumes at least one frame (no busy loops)

    def _step(self) -> None:
        snap = self.ad.snapshot()
        self.observe(snap)
        if snap.mode == "battle":
            self.battle_tick(snap)
        elif snap.mode == "naming":
            self.naming_tick(snap)
        elif snap.mode == "menu":
            self.menu_tick(snap)
        elif snap.mode == "text":
            self.text_tick(snap)
        elif snap.joy_locked:
            self.goal = "Escena del juego (esperar)"
            self.ad.wait(8)
        else:
            self.overworld_tick(snap)
        self.prev = snap
        self.tick += 1
        self.world.tick = self.tick
        self.stats["actions"] += 1

    # ================================================================== observation / bookkeeping
    def observe(self, snap: Snapshot) -> None:
        for line in snap.screen.lines:
            last = self.recent_lines[-1] if self.recent_lines else ""
            if line == last or line in list(self.recent_lines)[-3:]:
                continue
            if last and (line.startswith(last) or last.startswith(line)):
                self.recent_lines[-1] = line if len(line) > len(last) else last  # text still being typed
            else:
                self.recent_lines.append(line)
        sig = "|".join(snap.screen.rows).encode("utf-8") if snap.screen.rows else b""
        self.static_ticks = self.static_ticks + 1 if sig == self.screen_sig else 0
        self.screen_sig = sig
        prev = self.prev
        if prev is None:
            memory = self.world.mem(snap.map)
            memory.visits += 1
            self.stats["maps"] = len(self.world.visited_maps)
            self.check_milestones(snap)
            return
        if snap.map != prev.map:
            self.on_map_change(prev, snap)
        if snap.event_flags > prev.event_flags:
            self.gain("event", min(5, snap.event_flags - prev.event_flags), snap,
                      f"Evento de historia (+{snap.event_flags - prev.event_flags} flags)")
        if len(snap.party) > len(prev.party):
            mon = snap.party[-1]
            self.gain("party", 1, snap, f"Nuevo Pokemon: {mon.name} nv{mon.level}")
            if self.battle is not None and self.battle.get("kind") == "wild":
                self.battle["caught"] = True
        prev_items = {item for item, _ in prev.bag}
        new_items = [item for item, _ in snap.bag if item not in prev_items]
        if new_items:
            self.gain("item", len(new_items), snap, f"Objeto nuevo: {', '.join(self.ad.item_name(i) for i in new_items)}")
            picked = self.completed is not None and self.completed[0] == "item" and self.tick - self.completed[1] < 30
            if not picked and snap.mode != "battle":
                self.start_return_mode(snap, new_items)
        if snap.badges > prev.badges:
            self.gain("badge", snap.badges - prev.badges, snap, f"MEDALLA obtenida (total {snap.badges})")
        for old, new in zip(prev.party, snap.party):
            if new.species_id == old.species_id and new.level > old.level:
                self.gain("level", 1, snap, f"{new.name} sube a nv{new.level}")
        if (snap.battle is None and prev.battle is None and snap.party and len(snap.party) == len(prev.party)
                and snap.party_hp_fraction >= 0.999 and prev.party_hp_fraction < 0.98
                and all(m.hp == m.max_hp for m in snap.party)):
            self.on_heal(prev, snap)
        if snap.battle is not None and self.battle is None:
            self.on_battle_start(snap)
        if snap.battle is None and self.battle is not None and snap.mode != "battle":
            self.on_battle_end(prev, snap)
        self.evaluate_question(snap)
        if self.completed is not None and self.tick - self.completed[1] > 60:
            self.kb.learn_target(self.completed[0], 0.0)
            self.completed = None
        self.check_milestones(snap)
        self.positions.append(snap.cell)

    def gain(self, kind: str, amount: float, snap: Snapshot, message: str) -> None:
        weight = STRONG.get(kind, 0.5) * amount
        self.progress += weight
        if kind != "level":
            self.progress_events += 1
        self.progress_tick = self.tick
        self.weak_tick = self.tick
        if kind in ("event", "item", "party", "badge"):
            memory = self.world.mem(snap.map)
            memory.story += weight
            if self.last_talk is not None and self.tick - self.last_talk["tick"] < 40:
                talk_mem = self.world.mem(self.last_talk["map"])
                npc = self.last_talk.get("npc")
                if npc is not None:
                    talk_mem.talk_gain[npc] = talk_mem.talk_gain.get(npc, 0.0) + weight
            if self.return_mode is not None and self.tick - self.return_mode["since"] > 5 \
                    and kind in ("event", "badge") and memory.key in self.return_mode["targets"]:
                self.finish_return_mode(True, snap)
        if self.completed is not None and self.tick - self.completed[1] <= 60:
            self.kb.learn_target(self.completed[0], weight)
            self.completed = None
        if self.stuck_open is not None and kind != "level":
            self.stuck_open["resolved_by"] = self.stuck_level
            self.stuck_open["resolved_tick"] = self.tick
            self.stuck_open["resolved_frame"] = self.ad.frame_count
            self.stuck_log.append(self.stuck_open)
            self.stats["recoveries"] += 1
            self.log(f"Atasco resuelto (nivel {self.stuck_level}) por {kind}")
            self.stuck_open = None
            self.stuck_level = 0
        if self.question is not None:
            self.question["progress"] = True
        self.save_pending = True
        self.kb.record_story(self.ad.key, kind, self.ad.frame_count) if kind in ("party", "badge", "item") else None
        self.log(message)

    def on_map_change(self, prev: Snapshot, snap: Snapshot) -> None:
        memory = self.world.mem(snap.map)
        first = memory.visits == 0
        memory.visits += 1
        walk = self.last_walk
        if walk is not None and walk[3] == self.tick - 1 and walk[0] == prev.map:
            old_info = self.world.maps[prev.map]
            (px, py), direction = walk[1], walk[2]
            dx, dy = DELTAS[direction]
            nx, ny = px + dx, py + dy
            if not (0 <= nx < old_info.width and 0 <= ny < old_info.height) and direction in old_info.connections:
                self.world.record_connection(prev.map, direction, snap.map, (snap.x, snap.y), (px, py))
            else:
                cell = (nx, ny) if (nx, ny) in old_info.warps else (px, py)
                if cell in old_info.warps:
                    self.world.record_warp(prev.map, cell, snap.map, (snap.x, snap.y))
                    self.stats["warps"] += 1
        self.stats["maps"] = len(self.world.visited_maps)
        self.target = None
        self.settle_until = self.tick + 3
        self.ad.invalidate()
        name = self.ad.map_name(snap.map)
        if first:
            self.gain("new_map", 1, snap, f"Nuevo mapa: {name}")
        else:
            self.log(f"Entra a {name}")

    def on_heal(self, prev: Snapshot, snap: Snapshot) -> None:
        via = "other"
        spot = None
        if prev.able_count == 0 or self.flags.intersection({"blackout_pending"}):
            via = "blackout"
            self.flags.discard("blackout_pending")
        elif self.last_talk is not None and self.tick - self.last_talk["tick"] < 80:
            via = "talk"
            spot = dict(self.last_talk)
        self.stats["heals"] += 1
        lines = list(self.recent_lines)[-6:]
        self.kb.learn_heal(self.ad.key, self.ad.heal_signature(), lines, via)
        if spot is not None:
            key = (spot["map"], spot["cell"])
            if all((s["map"], s["cell"]) != key for s in self.heal_spots):
                self.heal_spots.append(spot)
                self.log(f"Lugar de curacion aprendido: {self.ad.map_name(spot['map'])} {spot['cell']}")
        self.flags.add("healed_once")
        self.gain("heal", 1, snap, f"Equipo curado ({via}) en {self.ad.map_name(snap.map)}")

    def check_milestones(self, snap: Snapshot) -> None:
        visited = set(self.world.visited_maps)
        for name in self.ad.milestones(snap, visited, self.flags):
            if name not in self.milestones:
                self.milestones[name] = {"frame": self.ad.frame_count, "tick": self.tick,
                                         "wall_s": round(time.time() - self.start_wall, 1),
                                         "maps_visited": len(visited), "battles": self.stats["battles"],
                                         "wins": self.stats["wins"], "losses": self.stats["losses"],
                                         "actions": self.stats["actions"], "stalls": self.stats["stalls"]}
                self.log(f"HITO: {name}")
                self.save_state_file(f"m_{name}")
                self.timeline.append({"milestone": name, **self.milestones[name]})

    # ================================================================== story-return pattern
    def start_return_mode(self, snap: Snapshot, items: list[int]) -> None:
        story_maps = sorted(((m.story, k) for k, m in self.world.maps.items() if m.story > 0 and k != snap.map),
                            reverse=True)
        if not story_maps:
            return
        stats = self.kb.skills.setdefault("story_return", {"tries": 0, "success": 0})
        self.return_mode = {"since": self.tick, "items": items, "map": snap.map,
                            "targets": [k for _, k in story_maps[:2]]}
        names = ", ".join(self.ad.map_name(k) for _, k in story_maps[:2])
        belief = (stats["success"] + 0.0) / (stats["tries"] + 1.0)
        self.log(f"Patron 'volver al lugar de la historia' activado ({names}; creencia {belief:.2f})")

    def finish_return_mode(self, success: bool, snap: Snapshot) -> None:
        if self.return_mode is None:
            return
        stats = self.kb.skills.setdefault("story_return", {"tries": 0, "success": 0})
        stats["tries"] += 1
        if success:
            stats["success"] += 1
        self.log(f"Patron 'volver' {'confirmado' if success else 'fallido'} ({stats['success']}/{stats['tries']})")
        self.return_mode = None

    def return_boost(self, key: MapKey) -> float:
        if self.return_mode is None:
            return 1.0
        if self.tick - self.return_mode["since"] > 6000:
            self.finish_return_mode(False, self.prev) if self.prev is not None else None
            return 1.0
        if key not in self.return_mode["targets"]:
            return 1.0
        stats = self.kb.skills.get("story_return", {"tries": 0, "success": 0})
        belief = stats["success"] / (stats["tries"] + 1.0)
        return 1.0 + 6.0 * belief

    # ================================================================== text / menus / naming
    def text_tick(self, snap: Snapshot) -> None:
        self.goal = "Leer dialogo"
        if snap.screen.prompt or (not snap.screen.textbox and self.static_ticks > 3):
            self.ad.press("A", 3, 8)
        elif self.static_ticks > 15:
            self.ad.press("B", 3, 8)
            if self.static_ticks > 60:
                self.ad.press("A", 3, 8)
        else:
            self.ad.wait(6)

    def question_text(self) -> str:
        return " ".join(list(self.recent_lines)[-3:])

    def menu_tick(self, snap: Snapshot) -> None:
        menu = snap.screen.menu
        assert menu is not None
        if len(menu.options) == 2:
            self.answer_question(snap, menu)
            return
        if self.learn_ctx is not None and self.learn_ctx.get("stage") == "forget":
            self.forget_move(snap, menu)
            return
        # Unknown overworld menu (shop, PC, START...): leave it.
        self.goal = "Cerrar menu"
        self.stats["menu_exits"] += 1
        self.ad.press("B", 3, 10)

    def decide_question(self, question: str, snap: Snapshot) -> tuple[str, str]:
        text = fold(question)
        type_words = self.ad.type_words()
        if not snap.party:
            for word, type_name in type_words.items():
                if fold(word) in text:
                    preference = self.kb.skills.get("starter_preference", {}).get("order",
                                                                                 DEFAULT_STARTER_PREFERENCE)
                    rank = preference.index(type_name) if type_name in preference else len(preference)
                    if rank <= self.declined_starters:
                        return "yes", f"Elegir inicial tipo {type_name} (preferencia {preference})"
                    self.declined_starters += 1
                    return "no", f"Rechazar inicial tipo {type_name}; se prefiere {preference[0]}"
        move = self.ad.find_move_in_text(question)
        if move is not None and snap.party:
            if self.learn_ctx is not None and self.learn_ctx.get("move") == move.name \
                    and self.learn_ctx.get("stage") == "declined":
                self.learn_ctx = None
                return "yes", f"Confirmar no aprender {move.name}"
            mon = self.mon_named_in(question, snap) or snap.party[0]
            if len(mon.moves) >= 4:
                worst = min(range(len(mon.moves)), key=lambda i: self.move_value(mon.moves[i], mon))
                if self.move_value(move, mon) > self.move_value(mon.moves[worst], mon) * 1.1:
                    self.learn_ctx = {"move": move.name, "stage": "forget", "slot": worst, "mon": mon.name}
                    return "yes", f"Aprender {move.name} olvidando {mon.moves[worst].name}"
                self.learn_ctx = {"move": move.name, "stage": "declined"}
                return "no", f"No aprender {move.name} (peor que los actuales)"
        heal_words = self.kb.heal_words(2)
        if self.target is not None and self.target.kind == "heal":
            return "yes", "Curar equipo (objetivo de curacion)"
        if heal_words and sum(1 for w in words(question) if w in heal_words) >= 2:
            return "yes", "Pregunta con palabras de curacion (aprendido)"
        answer, strength = self.kb.answer(question)
        if strength >= 0.9:
            return answer, f"Respuesta aprendida ({answer}, evidencia {strength:.1f})"
        return "yes", "Por defecto: aceptar (avanza la historia)"

    def answer_question(self, snap: Snapshot, menu: MenuInfo) -> None:
        question = self.question_text()
        signature = fold(question)[-60:]
        self.stats["yesno"] += 1
        seen = self.questions_seen.get(signature)
        if seen is not None and self.tick - seen < 400 and self.question is None:
            self.stats["repeated_questions"] += 1
        self.questions_seen[signature] = self.tick
        answer, reason = self.decide_question(question, snap)
        self.goal = f"Responder '{answer}': {reason}"
        self.log(f"Pregunta: '{question[-70:]}' -> {answer} ({reason})")
        if answer == "yes":
            if menu.selected not in (0, -1):
                self.ad.press("up", 2, 6)
            self.ad.press("A", 3, 10)
        else:
            self.ad.press("B", 3, 10)
        for _ in range(12):  # wait until the menu is gone so one question gets one answer
            current = self.ad.screen().menu
            if current is None or current.options != menu.options:
                break
            self.ad.wait(4)
        self.question = {"text": question, "sig": signature, "answer": answer, "tick": self.tick,
                         "progress": False, "naming": False}

    def evaluate_question(self, snap: Snapshot) -> None:
        q = self.question
        if q is None:
            return
        age = self.tick - q["tick"]
        if snap.mode == "naming" and not q["naming"]:
            q["naming"] = True
            other = "no" if q["answer"] == "yes" else "yes"
            self.kb.learn_answer(q["text"], other, 2.0)
            self.stats["wrong_answers"] += 1
            self.log("La respuesta abrio un teclado de nombre: aprendido responder lo contrario")
        if age > 3 and snap.mode == "menu" and snap.screen.menu is not None and len(snap.screen.menu.options) == 2:
            if fold(self.question_text())[-60:] == q["sig"] and age < 300:
                other = "no" if q["answer"] == "yes" else "yes"
                if not q.get("repeat_learned"):
                    self.kb.learn_answer(q["text"], other, 1.0)
                    q["repeat_learned"] = True
                    self.stats["wrong_answers"] += 1
                    self.log("La misma pregunta se repite: aprendido responder lo contrario")
        if age > 120:
            if q["progress"] and not q["naming"]:
                self.kb.learn_answer(q["text"], q["answer"], 1.0)
            self.question = None

    def naming_tick(self, snap: Snapshot) -> None:
        self.goal = "Salir del teclado de nombre"
        if self.naming_tries == 0:
            self.stats["naming_screens"] += 1
        known = self.kb.skills.get("naming_exit", {}).get("keys")
        candidates: list[list[str]] = [["START"], ["START", "A"], ["B", "B", "B", "START", "A"], ["A"]]
        sequence = list(known) if known and self.naming_tries < 2 else candidates[self.naming_tries % len(candidates)]
        self.naming_seq = sequence
        for key in sequence:
            self.ad.press(key, 3, 12)
            self.stats["naming_presses"] += 1
        self.naming_tries += 1
        self.ad.wait(20)
        if not self.ad.screen().naming:
            entry = self.kb.skills.setdefault("naming_exit", {"keys": sequence, "n": 0})
            entry["keys"], entry["n"] = sequence, entry.get("n", 0) + 1
            self.log(f"Teclado de nombre cerrado con {sequence} tras {self.naming_tries} intentos")
            self.naming_tries = 0

    def forget_move(self, snap: Snapshot, menu: MenuInfo) -> None:
        assert self.learn_ctx is not None
        slot = int(self.learn_ctx.get("slot", 0))
        self.goal = f"Olvidar movimiento {slot} para aprender {self.learn_ctx.get('move')}"
        for _ in range(6):
            index = self.ad.menu_index()
            if index == slot:
                break
            self.ad.press("down" if index < slot else "up", 2, 6)
        self.ad.press("A", 3, 10)
        self.log(f"Aprende {self.learn_ctx.get('move')} (olvida ranura {slot})")
        self.learn_ctx = None

    def mon_named_in(self, text: str, snap: Snapshot) -> MonInfo | None:
        folded = fold(text)
        for mon in snap.party:
            if mon.name and fold(mon.name) in folded:
                return mon
        return None

    def move_value(self, move: MoveInfo, mon: MonInfo) -> float:
        if move.power <= 1:
            return 12.0
        stab = 1.5 if move.type in mon.types else 1.0
        return move.power * max(move.accuracy, 30) / 100.0 * stab

    # ================================================================== battle
    def on_battle_start(self, snap: Snapshot) -> None:
        assert snap.battle is not None
        self.stats["battles"] += 1
        enemy = snap.battle.enemy
        self.battle = {"kind": snap.battle.kind, "start_tick": self.tick, "start_frame": self.ad.frame_count,
                       "enemy_levels": {enemy.level} if enemy else set(), "turns": 0, "choice": None,
                       "reason": "", "enemy_fainted": False, "caught": False, "ran": False, "eff": None,
                       "enemies": [enemy.name] if enemy else []}
        self.turn_choice = None
        self.pending_attack = None
        self.attack_ctx = None
        who = f"{enemy.name} nv{enemy.level}" if enemy else "?"
        self.log(f"Combate {'salvaje' if snap.battle.kind == 'wild' else 'contra entrenador'}: {who}")

    def on_battle_end(self, prev: Snapshot, snap: Snapshot) -> None:
        battle = self.battle
        self.battle = None
        if battle is None:
            return
        if self.pending_probe is not None and self.pending_probe.get("expected") in (None, "run"):
            self.resolve_probe("run")
        self.pending_probe = None
        self.finish_attack(prev)
        if snap.able_count == 0 or (prev.party and prev.able_count == 0):
            self.stats["losses"] += 1
            self.stats["blackouts"] += 1
            self.flags.add("blackout_pending")
            levels = battle["enemy_levels"] or {5}
            if battle["kind"] == "trainer":
                self.train_level = max(self.train_level, max(levels) + 2)
            else:
                self.train_level = max(self.train_level, max(levels) + 1)
            self.log(f"Derrota contra {battle['enemies']}: entrenar hasta nv{self.train_level}")
        elif battle.get("caught"):
            self.stats["caught"] += 1
            self.log("Pokemon capturado")
        elif battle.get("ran") and not battle.get("enemy_fainted"):
            self.stats["ran"] += 1
            self.log("Huida del combate")
        else:
            self.stats["wins"] += 1
            if "first_battle_won" not in self.flags:
                self.flags.add("first_battle_won")
            if battle["kind"] == "trainer":
                self.flags.add("trainer_won")
            self.gain("win", 1, snap, f"Combate ganado ({battle['kind']}) contra {battle['enemies']}")
        self.turn_choice = None

    def battle_tick(self, snap: Snapshot) -> None:
        battle = snap.battle
        if self.battle is None and battle is not None:
            self.on_battle_start(snap)
        if battle is not None and battle.enemy is not None and self.battle is not None:
            enemy = battle.enemy
            if not self.battle["enemies"] or self.battle["enemies"][-1] != enemy.name:
                self.battle["enemies"].append(enemy.name)
            self.battle["enemy_levels"].add(enemy.level)
            if enemy.hp == 0:
                self.battle["enemy_fainted"] = True
        screen = snap.screen
        menu = screen.menu
        self.scan_battle_text(snap)
        if menu is None:
            if self.pending_probe is not None:
                self.pending_probe["left"] = True
            self.goal = "Combate: texto"
            if screen.prompt:
                self.ad.press("A", 3, 8)
            elif self.static_ticks > 20:
                self.ad.press("B", 3, 8)
            else:
                self.ad.wait(4)
            return
        kind = self.classify_battle_menu(snap, menu)
        probe = self.pending_probe
        if probe is not None and menu.options == probe["options"]:
            if probe["left"] and kind == "main":
                # Text and back to the same menu: the option tried to flee ("can't escape").
                self.resolve_probe("run")
            elif self.tick - probe["tick"] < 30:
                self.ad.wait(4)  # the selection is still being processed
                return
            else:
                self.pending_probe = None
        if self.pending_probe is not None and kind in ("moves", "party", "other"):
            self.resolve_probe({"moves": "fight", "party": "pokemon", "other": "bag"}[kind])
        if kind == "yesno":
            self.battle_question(snap, menu)
        elif kind == "main":
            self.battle_main(snap, menu)
        elif kind == "moves":
            self.battle_moves(snap, menu)
        elif kind == "party":
            self.battle_party(snap, menu)
        elif kind == "submenu":
            self.goal = "Combate: confirmar cambio"
            for _ in range(4):
                if self.ad.screen().menu is None or (self.ad.screen().menu or menu).selected == 0:
                    break
                self.ad.press("up", 2, 6)
            self.ad.press("A", 3, 10)
        else:
            self.goal = "Combate: cerrar menu desconocido"
            self.ad.press("B", 3, 10)

    def resolve_probe(self, meaning: str) -> None:
        probe = self.pending_probe
        self.pending_probe = None
        if probe is None:
            return
        self.kb.learn_menu(probe["label"], meaning, self.ad.key)
        if probe.get("expected") != meaning:
            if probe.get("expected") is not None:
                self.stats["menu_relearned"] = self.stats.get("menu_relearned", 0) + 1
            self.log(f"Menu aprendido: '{probe['label']}' = {meaning}")

    def classify_battle_menu(self, snap: Snapshot, menu: MenuInfo) -> str:
        battle = snap.battle
        options = [fold(o) for o in menu.options]
        if battle is not None and battle.me is not None:
            move_names = {fold(m.name) for m in battle.me.moves}
            if options and sum(1 for o in options if o in move_names) >= max(1, min(2, len(move_names))):
                return "moves"
        party_names = [fold(m.name) for m in snap.party]
        rows = [fold(r) for r in snap.screen.rows]
        if party_names and menu.cursor_col <= 1:
            hits = sum(1 for name in party_names if any(name in row for row in rows[:12]))
            if hits >= min(2, len(party_names)):
                return "party"
        if len(menu.options) == 2:
            return "yesno"
        meanings = [self.kb.menu_meaning(o) for o in menu.options]
        if "fight" in meanings or (len(menu.options) == 4 and menu.cursor_row >= 12):
            return "main"
        if len(menu.options) == 3 and self.turn_choice is not None and getattr(self.turn_choice, "kind", "") == "switch":
            return "submenu"
        return "other"

    def battle_question(self, snap: Snapshot, menu: MenuInfo) -> None:
        question = self.question_text()
        move = self.ad.find_move_in_text(question)
        if move is not None:
            self.answer_question(snap, menu)
            return
        # "Use next POKeMON?" -> yes when someone can still fight; Gen 2 "switch?" prompts -> no.
        if snap.able_count > 0 and snap.battle is not None and snap.battle.me is not None and snap.battle.me.hp == 0:
            self.goal = "Combate: sacar otro Pokemon"
            self.ad.press("A", 3, 10)
        else:
            self.goal = "Combate: no cambiar"
            self.ad.press("B", 3, 10)

    def battle_view(self, snap: Snapshot) -> Any:
        from ares_u.brain.api import BattleView, MonView, MoveView, category_for

        generation = self.ad.generation

        def move_view(move: MoveInfo) -> Any:
            return MoveView(name=move.name, type=move.type, power=move.power, accuracy=move.accuracy, pp=move.pp,
                            category=category_for(move.type, move.power, generation), gen3_move_id=move.move_id,
                            max_pp=move.max_pp)

        def mon_view(mon: MonInfo, honest: bool) -> Any:
            return MonView(species=mon.name, national_dex=mon.national_dex, level=mon.level, hp=mon.hp,
                           max_hp=mon.max_hp, types=mon.types, status=mon.status,
                           moves=[] if honest else [move_view(m) for m in mon.moves],
                           stats=None if honest else dict(mon.stats))

        battle = snap.battle
        assert battle is not None and battle.me is not None and battle.enemy is not None
        party = [mon_view(m, False) for m in snap.party]
        active = battle.active_index if 0 <= battle.active_index < len(snap.party) else 0
        me = mon_view(battle.me, False)
        if active < len(party):
            party[active] = me
        return BattleView(game=self.ad.key, generation=generation, me=me, enemy=mon_view(battle.enemy, True),
                          party=party, active_index=active, is_wild=battle.kind == "wild",
                          can_switch=snap.able_count > 1, can_run=battle.kind == "wild", turn=self.battle["turns"]
                          if self.battle else 0)

    def load_brain(self) -> None:
        self.brain_loaded = True
        if not self.use_brain:
            return
        for module_name in ("ares_u.brain", "ares_u.brain.brain", "ares_u.brain.policy", "ares_u.brain.core"):
            try:
                module = importlib.import_module(module_name)
            except Exception:  # noqa: BLE001 - the brain is optional
                continue
            cls = getattr(module, "BattleBrain", None)
            if cls is not None:
                try:
                    self.brain = cls()
                    self.brain_name = f"ares_u.brain ({module_name}.BattleBrain)"
                    self.log(f"Cerebro de combate cargado: {module_name}")
                    return
                except Exception as exc:  # noqa: BLE001
                    self.log(f"Cerebro no disponible ({exc.__class__.__name__}); uso politica propia")
                    return

    def choose(self, snap: Snapshot) -> Any:
        from ares_u.brain.api import BrainChoice

        if not self.brain_loaded:
            self.load_brain()
        if self.brain is not None:
            try:
                choice = self.brain.choose(self.battle_view(snap))
                if choice.kind != "item" or self.can_use_items():
                    return choice
                self.stats["brain_item_declined"] = self.stats.get("brain_item_declined", 0) + 1
            except Exception as exc:  # noqa: BLE001 - fall back to the own policy
                self.stats["brain_errors"] = self.stats.get("brain_errors", 0) + 1
                if self.stats["brain_errors"] <= 5:
                    self.log(f"Cerebro fallo ({exc.__class__.__name__}: {str(exc)[:60]}); politica propia")
        battle = snap.battle
        assert battle is not None and battle.me is not None and battle.enemy is not None
        me, enemy = battle.me, battle.enemy
        scores: dict[str, float] = {}
        best, best_value, best_eff = 0, -1.0, 1.0
        for index, move in enumerate(me.moves):
            if move.pp <= 0:
                continue
            eff, _ = self.kb.effectiveness(move.type, enemy.types, self.ad.generation)
            stab = 1.5 if move.type in me.types else 1.0
            if move.power > 1:
                value = move.power * max(move.accuracy, 30) / 100.0 * stab * eff
            else:
                value = 6.0 if (self.battle or {}).get("turns", 0) <= 1 and enemy.hp_fraction > 0.7 else 1.0
            scores[move.name] = round(value, 2)
            if value > best_value:
                best, best_value, best_eff = index, value, eff
        if best_value < 0:
            return BrainChoice(kind="move", index=0, reason="Sin PP: forcejeo", scores=scores)
        if battle.kind == "wild" and me.hp_fraction < 0.2 and snap.able_count <= 1 and self.need_heal(snap):
            return BrainChoice(kind="run", index=0, reason="Salvaje y PS muy bajos: huir para curar", scores=scores)
        move = me.moves[best]
        eff_word = {0.0: "sin efecto", 0.5: "poco eficaz", 0.25: "muy poco eficaz", 2.0: "super eficaz",
                    4.0: "x4"}.get(round(best_eff * 4) / 4, "neutro")
        return BrainChoice(kind="move", index=best, reason=f"{move.name} ({move.type}, poder {move.power}, "
                           f"{eff_word} contra {'/'.join(enemy.types)})", scores=scores)

    def can_use_items(self) -> bool:
        return False

    def battle_main(self, snap: Snapshot, menu: MenuInfo) -> None:
        if self.battle is not None:
            self.battle["turns"] += 1
        self.finish_attack(snap)
        choice = self.choose(snap)
        self.turn_choice = choice
        if self.battle is not None:
            self.battle["choice"] = f"{choice.kind}:{choice.index}"
            self.battle["reason"] = choice.reason
            self.battle["scores"] = dict(getattr(choice, "scores", {}) or {})
        meaning = {"move": "fight", "switch": "pokemon", "run": "run", "item": "bag"}.get(choice.kind, "fight")
        labels = {self.kb.menu_meaning(o): i for i, o in enumerate(menu.options) if self.kb.menu_meaning(o)}
        if meaning == "run" and self.battle is not None:
            self.battle["ran"] = True
        if meaning in labels:
            index = labels[meaning]
        else:
            unknown = [i for i, o in enumerate(menu.options) if self.kb.menu_meaning(o) is None]
            if not unknown:
                index = 0
            else:
                # Probe: the first unknown option (the game usually puts FIGHT first).
                index = unknown[0]
                self.stats["menu_probes"] += 1
                self.log(f"Explorar opcion de menu desconocida '{menu.options[index]}'")
        self.goal = f"Combate: {choice.reason}"
        self.navigate_to(menu, index)
        self.pending_probe = {"label": menu.options[index], "tick": self.tick, "options": menu.options,
                              "left": False, "expected": self.kb.menu_meaning(menu.options[index])}
        self.ad.press("A", 3, 10)

    def navigate_to(self, menu: MenuInfo, index: int) -> None:
        """Move the cursor onto option ``index`` using the label positions on screen."""
        current = menu
        for _ in range(8):
            if current.selected == index or not current.positions:
                return
            row, col = current.positions[index]
            if current.cursor_row < row:
                self.ad.press("down", 2, 6)
            elif current.cursor_row > row:
                self.ad.press("up", 2, 6)
            elif current.cursor_col < col - 1:
                self.ad.press("right", 2, 6)
            else:
                self.ad.press("left", 2, 6)
            refreshed = self.ad.screen().menu
            if refreshed is None or len(refreshed.options) != len(menu.options):
                return
            current = refreshed

    def battle_moves(self, snap: Snapshot, menu: MenuInfo) -> None:
        battle = snap.battle
        assert battle is not None and battle.me is not None
        choice = self.turn_choice
        if choice is None or choice.kind != "move":
            choice = self.choose(snap)
            self.turn_choice = choice
        index = choice.index if 0 <= choice.index < len(battle.me.moves) else 0
        move = battle.me.moves[index]
        options = [fold(o) for o in menu.options]
        target = options.index(fold(move.name)) if fold(move.name) in options else index
        for _ in range(6):
            current = self.ad.screen().menu
            if current is None or current.selected == target:
                break
            self.ad.press("down" if current.selected < target else "up", 2, 6)
        self.goal = f"Combate: usar {move.name}"
        self.ad.press("A", 3, 10)
        enemy = battle.enemy
        self.pending_attack = {"move": move.name, "type": move.type, "power": move.power,
                               "enemy_types": enemy.types if enemy else (), "enemy_hp": enemy.hp if enemy else 0,
                               "enemy_species": enemy.species_id if enemy else -1, "effect": None,
                               "tick": self.tick}
        self.attack_ctx = None
        self.turn_choice = None

    def battle_party(self, snap: Snapshot, menu: MenuInfo) -> None:
        battle = snap.battle
        choice = self.turn_choice
        forced = battle is None or battle.me is None or battle.me.hp == 0
        if choice is not None and choice.kind == "switch" and 0 <= choice.index < len(snap.party) and not forced:
            target = choice.index
        else:
            if not forced and choice is not None and choice.kind != "switch":
                # Came here by a probe: go back to the main menu.
                self.ad.press("B", 3, 10)
                return
            able = [i for i, m in enumerate(snap.party) if m.hp > 0]
            if not able:
                self.ad.press("B", 3, 10)
                return
            enemy = battle.enemy if battle is not None else None

            def score(i: int) -> float:
                mon = snap.party[i]
                best = 1.0
                if enemy is not None:
                    for move in mon.moves:
                        if move.power > 1 and move.pp > 0:
                            eff, _ = self.kb.effectiveness(move.type, enemy.types, self.ad.generation)
                            best = max(best, move.power * eff * (1.5 if move.type in mon.types else 1.0))
                return best * (0.3 + mon.hp_fraction) * mon.level

            target = max(able, key=score)
        self.goal = f"Combate: elegir {snap.party[target].name}"
        for _ in range(8):
            index = self.ad.menu_index()
            if index == target:
                break
            self.ad.press("down" if index < target else "up", 2, 6)
        self.ad.press("A", 3, 12)
        self.turn_choice = type("Switch", (), {"kind": "switch", "index": target})()

    def finish_attack(self, snap: Snapshot) -> None:
        attack = self.pending_attack
        if attack is None:
            return
        self.pending_attack = None
        if attack["effect"] is not None or attack["power"] <= 1:
            return
        battle = snap.battle
        enemy = battle.enemy if battle is not None else None
        damaged = enemy is not None and enemy.species_id == attack["enemy_species"] and enemy.hp < attack["enemy_hp"]
        fainted = enemy is None or enemy.hp == 0
        if damaged or (fainted and attack["enemy_hp"] > 0):
            self.kb.observe_effectiveness(attack["type"], tuple(attack["enemy_types"]), 1.0, self.ad.generation, 0.5)
            self.stats["type_obs"] += 1

    def scan_battle_text(self, snap: Snapshot) -> None:
        for line in snap.screen.lines:
            if line in self.seen_battle_lines:
                continue
            self.seen_battle_lines.append(line)
            text = fold(line)
            attack = self.pending_attack
            if attack is not None and fold(attack["move"]) in text:
                self.attack_ctx = {"who": "me", "type": attack["type"], "defenders": tuple(attack["enemy_types"])}
                continue
            move = self.ad.find_move_in_text(line)
            if move is not None and snap.battle is not None and snap.battle.me is not None:
                self.attack_ctx = {"who": "enemy", "type": move.type, "defenders": snap.battle.me.types}
                continue
            for phrase, multiplier in self.ad.effectiveness_phrases():
                if phrase in text and self.attack_ctx is not None:
                    ctx = self.attack_ctx
                    self.kb.observe_effectiveness(ctx["type"], tuple(ctx["defenders"]), multiplier,
                                                  self.ad.generation)
                    self.stats["type_obs"] += 1
                    if ctx["who"] == "me" and attack is not None:
                        attack["effect"] = multiplier
                    if self.battle is not None:
                        self.battle["eff"] = f"{ctx['type']}>{'/'.join(ctx['defenders'])} x{multiplier}"
                    self.log(f"Eficacia observada: {ctx['type']} contra {'/'.join(ctx['defenders'])} = x{multiplier}")
                    self.attack_ctx = None
                    break

    # ================================================================== overworld
    def need_heal(self, snap: Snapshot) -> bool:
        if not snap.party:
            return False
        lead = snap.party[0]
        return snap.party_hp_fraction < 0.4 or snap.able_count == 0 or (lead.hp_fraction < 0.25 and snap.able_count <= 1)

    def training(self, snap: Snapshot) -> bool:
        if not snap.party or self.train_level <= 0:
            return False
        best = max(m.level for m in snap.party)
        if best >= self.train_level:
            self.log(f"Entrenamiento completado (nv{best} >= {self.train_level})")
            self.train_level = 0
            return False
        return True

    def overworld_tick(self, snap: Snapshot) -> None:
        if self.save_pending and snap.mode == "overworld" and snap.battle is None:
            self.lastgood = self.ad.save_state()
            self.lastgood_tick = self.tick
            self.save_pending = False
        if self.tick < self.settle_until:
            self.ad.invalidate()
            self.ad.wait(8)  # the new map's header/warps are still being loaded
            return
        info = self.ad.map_info()
        npcs = self.ad.npcs()
        memory = self.world.mem(info.key)
        cell = (snap.x, snap.y)
        if cell not in memory.visited:
            self.weak_tick = self.tick
        memory = self.world.observe(info, npcs, snap.x, snap.y)
        if self.anti_stuck(snap, info):
            return
        npc_cells = {(n.x, n.y) for n in npcs}
        blocked = (self.world.blocked_cells(info.key) | npc_cells) - {cell}
        targets = self.local_targets(snap, info, npcs)
        goals: set[Cell] = set()
        for target in targets:
            goals.update(target.goals)
        dist, parent = bfs(info, cell, blocked, goals)
        best: Target | None = None
        best_goal: Cell | None = None
        best_score = 0.0
        for target in targets:
            if self.is_tabu(target.key):
                continue
            reach = [(dist[c], c) for c in target.goals if c in dist and self.bad_goals.get((target.key, c), 0) < 2]
            if not reach:
                continue
            d, goal = min(reach)
            score = target.value / (1.0 + d / 30.0)
            if self.target is not None and target.key == self.target.key:
                score *= 1.4
            if score > best_score:
                best, best_goal, best_score = target, goal, score
        remote = self.remote_target(snap, info, dist)
        if remote is not None and remote[0] > best_score:
            best_score, best, best_goal = remote
        if best is None or best_goal is None:
            self.goal = "Explorar al azar (sin objetivos alcanzables)"
            self.random_step(snap, info)
            return
        if self.target is None or self.target.key != best.key:
            self.target = best
            self.target_since = self.tick
            self.target_ticks = 0
            self.target_dist0 = dist.get(best_goal, 0)
        self.target_ticks += 1
        self.goal = self.describe_goal(best, info)
        if self.target_ticks > 3 * self.target_dist0 + 40:
            self.log(f"Objetivo abandonado por tiempo: {best.describe()}")
            self.set_tabu(best.key, 1500)
            self.target = None
            return
        if cell == best_goal:
            self.execute(best, best_goal, snap, info)
            return
        direction = first_step(parent, cell, best_goal)
        if direction is None:
            self.execute(best, best_goal, snap, info)
            return
        self.do_walk(direction, snap, info, best)

    def describe_goal(self, target: Target, info: MapInfo) -> str:
        kind = target.kind
        if kind == "heal":
            return f"Curar al equipo en {info.name}"
        if kind == "route":
            return target.label
        if kind == "new_map":
            return f"Explorar: salida nueva en {info.name}"
        if kind == "npc":
            return f"Hablar con un personaje en {info.name}"
        if kind == "retalk":
            return f"Volver a hablar con un personaje en {info.name}"
        if kind == "item":
            return f"Recoger objeto en {info.name}"
        if kind == "sign":
            return f"Leer cartel en {info.name}"
        if kind == "grass":
            return f"Entrenar en la hierba hasta nv{self.train_level}"
        if kind == "coverage":
            return f"Recorrer {info.name}"
        return f"{kind} en {info.name}"

    def local_targets(self, snap: Snapshot, info: MapInfo, npcs: list[Npc]) -> list[Target]:
        kb = self.kb
        memory = self.world.mem(info.key)
        out: list[Target] = []
        heal_mode = self.need_heal(snap) and bool(self.heal_spots)
        train = self.training(snap) and not heal_mode
        boost = self.return_boost(info.key)
        for (wx, wy), dest in memory.warps.items():
            used = (wx, wy) in memory.warps_used
            if used:
                continue
            unvisited = dest is None or self.world.mem(dest).visits == 0
            kind = "new_map" if unvisited else "exit"
            value = kb.target_value("new_map") if unvisited else 0.08
            out.append(Target(key=("warp", info.key, (wx, wy)), kind=kind, value=value * (0.3 if heal_mode else 1),
                              goals={(wx, wy): None}))
        for direction, dest in memory.connections.items():
            if direction in memory.connections_used:
                continue
            unvisited = self.world.mem(dest).visits == 0
            cells = self.world.crossing_goals(info, direction)  # type: ignore[arg-type]
            if not cells:
                continue
            value = kb.target_value("new_map") if unvisited else 0.08
            out.append(Target(key=("conn", info.key, direction), kind="new_map" if unvisited else "exit",
                              value=value * (0.3 if heal_mode else 1),
                              goals={c: direction for c in cells}))  # type: ignore[misc]
        for npc in npcs:
            stands = interaction_cells(info, npc.x, npc.y)
            if not stands:
                continue
            last = memory.talked.get(npc.index)
            if npc.is_item:
                if last is not None:
                    continue
                kind, value = "item", kb.target_value("item")
            elif last is None:
                kind, value = "npc", kb.target_value("npc")
            elif self.progress_events > last:
                importance = 1.0 + min(3.0, memory.talk_gain.get(npc.index, 0.0)) + min(2.0, 0.2 * memory.story)
                kind, value = "retalk", kb.target_value("retalk") * importance * boost
            else:
                continue
            for spot in self.heal_spots:
                if heal_mode and spot["map"] == info.key and spot.get("npc") == npc.index:
                    kind, value = "heal", 5.0
            out.append(Target(key=(kind if kind != "heal" else "npc", info.key, npc.index), kind=kind,
                              value=value * (0.2 if heal_mode and kind != "heal" else 1.0),
                              goals=dict(stands), interact=True, npc=npc.index))
        if heal_mode:
            for spot in self.heal_spots:
                if spot["map"] == info.key and spot.get("npc") is None:
                    out.append(Target(key=("heal", info.key, spot["cell"]), kind="heal", value=5.0,
                                      goals={tuple(spot["cell"]): spot["dir"]}, interact=True))  # type: ignore[dict-item]
        for sx, sy in memory.signs:
            if (sx, sy) in memory.signs_read:
                continue
            stands = interaction_cells(info, sx, sy, counters=False)
            if stands:
                out.append(Target(key=("sign", info.key, (sx, sy)), kind="sign", value=kb.target_value("sign"),
                                  goals=dict(stands), interact=True))
        if train and memory.grass:
            out.append(Target(key=("grass", info.key), kind="grass", value=2.0,
                              goals={c: None for c in memory.grass}))
        unvisited_cells = [(int(x), int(y)) for y, x in zip(*np.isin(info.grid, list(PASSABLE)).nonzero())
                           if (int(x), int(y)) not in memory.visited]
        if unvisited_cells:
            if len(unvisited_cells) > 600:
                unvisited_cells = self.rng.sample(unvisited_cells, 600)
            out.append(Target(key=("coverage", info.key), kind="coverage",
                              value=kb.target_value("coverage") * (0.3 if heal_mode else 1.0),
                              goals={c: None for c in unvisited_cells}))
        return out

    def map_pending_value(self, key: MapKey, heal_mode: bool, train: bool) -> float:
        memory = self.world.maps.get(key)
        kb = self.kb
        if memory is None or memory.visits == 0:
            return kb.target_value("new_map") * (0.3 if heal_mode else 1.0)
        if heal_mode:
            return 5.0 if any(s["map"] == key for s in self.heal_spots) else 0.0
        best = 0.0
        for index, (_, _, _, is_item) in memory.npcs.items():
            if self.is_tabu(("npc", key, index)) or self.is_tabu(("item", key, index)):
                continue
            last = memory.talked.get(index)
            if last is None:
                best = max(best, kb.target_value("item" if is_item else "npc"))
            elif self.progress_events > last and not is_item and not self.is_tabu(("retalk", key, index)):
                importance = 1.0 + min(3.0, memory.talk_gain.get(index, 0.0)) + min(2.0, 0.2 * memory.story)
                best = max(best, kb.target_value("retalk") * importance * self.return_boost(key))
        for cell, dest in memory.warps.items():
            if cell not in memory.warps_used and dest is None and not self.is_tabu(("warp", key, cell)):
                best = max(best, kb.target_value("new_map"))
        if train and memory.grass:
            best = max(best, 2.0)
        if memory.visited and memory.walkable and len(memory.visited) < 0.5 * memory.walkable \
                and not self.is_tabu(("coverage", key)):
            best = max(best, kb.target_value("coverage"))
        return best

    def remote_target(self, snap: Snapshot, info: MapInfo, dist: dict[Cell, int]
                      ) -> tuple[float, Target, Cell] | None:
        routes = self.world.route(info.key)
        heal_mode = self.need_heal(snap) and bool(self.heal_spots)
        train = self.train_level > 0
        best: tuple[float, Target, Cell] | None = None
        for key, (cost, via) in routes.items():
            if key == info.key or via is None or self.is_tabu(("route", key)):
                continue
            if self.is_tabu(("warp", info.key, via[1]) if via[0] == "warp" else ("conn", info.key, via[1])):
                continue
            value = self.map_pending_value(key, heal_mode, train)
            if value <= 0:
                continue
            kind, data = via
            if kind == "warp":
                goals: dict[Cell, Direction | None] = {data: None}
            else:
                goals = {c: data for c in self.world.crossing_goals(info, data)}
            reach = [(dist[c], c) for c in goals if c in dist]
            if not reach:
                continue
            d, goal = min(reach)
            score = 0.6 * value / (1.0 + (d + cost) / 30.0)  # finish the local frontier first
            if best is None or score > best[0]:
                target = Target(key=("route", key), kind="route", value=value, goals=goals,
                                label=f"Viajar a {self.ad.map_name(key)}"
                                + (" para curar" if heal_mode else ""),
                                via=("warp", info.key, data) if kind == "warp" else ("conn", info.key, data))
                best = (score, target, goal)
        return best

    def execute(self, target: Target, goal: Cell, snap: Snapshot, info: MapInfo) -> None:
        direction = target.goals.get(goal)
        memory = self.world.mem(info.key)
        if target.interact and direction is not None:
            self.ad.face_and_press(direction)
            self.ad.wait(4)
            opened = self.ad.dialog_open() or self.ad.in_battle()
            self.last_talk = {"map": info.key, "npc": target.npc, "cell": goal, "dir": direction, "tick": self.tick}
            if target.kind == "sign":
                memory.signs_read.add(target.key[2])
            elif target.npc is not None:
                if opened:
                    memory.talked[target.npc] = self.progress_events
                else:
                    self.bad_goals[(target.key, goal)] = self.bad_goals.get((target.key, goal), 0) + 1
                    if self.bad_goals[(target.key, goal)] >= 2:
                        memory.talked[target.npc] = self.progress_events
            if opened:
                self.stats["talks"] += 1
            self.completed = (target.kind, self.tick)
            self.target = None
            return
        if target.kind in ("coverage", "grass"):
            if target.kind == "grass":
                options = [d for d in DIRECTIONS
                           if (goal[0] + DELTAS[d][0], goal[1] + DELTAS[d][1]) in memory.grass]
                if options:
                    self.do_walk(self.rng.choice(options), snap, info, target)
                    return
            self.target = None
            return
        tkey = target.via or target.key   # failures belong to the transition, not to the destination
        tries = self.push_tries.get(tkey, 0)
        self.push_tries[tkey] = tries + 1
        limit = 14 if direction is not None else 6
        if tries >= limit:
            self.log(f"No se pudo usar {target.describe()}: tabu")
            self.set_tabu(tkey, 3000)
            self.set_tabu(target.key, 1500)
            self.push_tries.pop(tkey, None)
            self.target = None
            return
        if direction is not None:
            outcome = self.do_walk(direction, snap, info, target)
            if outcome in ("blocked", "turned", "timeout"):
                self.bad_goals[(target.key, goal)] = self.bad_goals.get((target.key, goal), 0) + 2
            return
        # On a warp cell that did not fire: push toward the nearest map edge first.
        x, y = goal
        order = sorted(DIRECTIONS, key=lambda d: {"up": y, "down": info.height - 1 - y, "left": x,
                                                  "right": info.width - 1 - x}[d])
        self.do_walk(order[tries % 4], snap, info, target)

    def do_walk(self, direction: Direction, snap: Snapshot, info: MapInfo, target: Target | None = None) -> str:
        self.last_walk = (info.key, (snap.x, snap.y), direction, self.tick)
        outcome = self.ad.walk(direction)
        if outcome in ("blocked", "turned", "timeout"):
            self.stats["walk_blocked"] += 1
            dx, dy = DELTAS[direction]
            dest = (snap.x + dx, snap.y + dy)
            key = (info.key, dest)
            self.walk_fail[key] = self.walk_fail.get(key, 0) + 1
            if self.walk_fail[key] >= 3 and info.inside(*dest):
                self.world.mem(info.key).blocked[dest] = self.tick + 600
                self.walk_fail[key] = 0
        return outcome

    def random_step(self, snap: Snapshot, info: MapInfo) -> None:
        self.stats["random_steps"] += 1
        direction = self.rng.choice(DIRECTIONS)
        self.do_walk(direction, snap, info)

    # ================================================================== anti-stuck
    def anti_stuck(self, snap: Snapshot, info: MapInfo) -> bool:
        stall = self.tick - max(self.progress_tick, self.weak_tick)
        story_stall = self.tick - self.progress_tick
        level = 0
        if stall > 2500 or story_stall > 12000:
            level = 3
        elif stall > 900:
            level = 2
        elif stall > 300:
            level = 1
        if level <= self.stuck_level:
            return False
        if self.stuck_open is None:
            self.stats["stalls"] += 1
            self.stuck_open = {"start_tick": self.tick, "start_frame": self.ad.frame_count,
                               "map": self.ad.map_name(snap.map), "cell": [snap.x, snap.y]}
        self.stuck_level = level
        if level == 1:
            self.log("Atasco nivel 1: objetivo actual a la lista tabu")
            if self.target is not None:
                self.set_tabu(self.target.key, 2000)
            self.target = None
            return False
        if level == 2:
            self.log("Atasco nivel 2: interactuar alrededor y rafaga aleatoria")
            for direction in DIRECTIONS:
                self.ad.face_and_press(direction)
                self.ad.wait(10)
                if self.ad.dialog_open() or self.ad.in_battle():
                    return True
            for _ in range(24):
                self.ad.walk(self.rng.choice(DIRECTIONS))
                if self.ad.dialog_open() or self.ad.in_battle():
                    break
            self.stats["random_steps"] += 24
            for key in list(self.world.mem(info.key).talked):
                self.world.mem(info.key).talked[key] = -1  # allow re-talking everyone here
            return True
        self.log("Atasco nivel 3: recargar el ultimo estado bueno")
        if self.target is not None:
            self.set_tabu(self.target.key, 6000)
        if self.lastgood is not None:
            self.ad.load_state(self.lastgood)
            self.stats["reloads"] += 1
        self.rng.seed(self.tick)
        for memory in self.world.maps.values():
            for key in list(memory.talked):
                memory.talked[key] = -1
            memory.blocked.clear()
        self.bad_goals.clear()
        self.push_tries.clear()
        self.progress_tick = self.weak_tick = self.tick
        self.stuck_level = 0
        self.target = None
        self.prev = None
        return True

    # ================================================================== persistence / reporting
    def save_state_file(self, name: str) -> None:
        try:
            (self.run_dir / f"{name}.state").write_bytes(self.ad.save_state())
        except OSError:
            pass

    def checkpoint(self) -> None:
        self.kb.games.setdefault(self.ad.key, {})
        self.kb.games[self.ad.key].update({"runs": self.kb.games[self.ad.key].get("runs_base", 0) + 1,
                                           "milestones": sorted(self.milestones), "frames": self.ad.frame_count,
                                           "maps": len(self.world.visited_maps), "sequence": self.sequence})
        self.kb.save()
        self.save_state_file("latest")
        (self.run_dir / "report.json").write_text(json.dumps(self.report(), indent=1, ensure_ascii=False,
                                                             default=str), encoding="utf-8")

    def report(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "game": self.ad.key, "generation": self.ad.generation,
                "sequence": self.sequence, "frames": self.ad.frame_count, "ticks": self.tick,
                "wall_s": round(time.time() - self.start_wall, 1), "milestones": self.milestones,
                "stats": self.stats, "maps_visited": len(self.world.visited_maps),
                "maps": {self.ad.map_name(k): m.to_json() for k, m in self.world.maps.items() if m.visits},
                "stuck_episodes": self.stuck_log[-50:], "open_stuck": self.stuck_open,
                "heal_spots": [{"map": self.ad.map_name(s["map"]), "cell": s["cell"]} for s in self.heal_spots],
                "inherited": self.inherited, "knowledge": self.kb.summary(), "brain": self.brain_name,
                "train_level": self.train_level, "timeline": self.timeline}

    def status(self) -> dict[str, Any]:
        snap = self.prev
        now = time.time()
        mark_t, mark_f = self._fps_mark
        if now - mark_t >= 2.0:
            self.fps = (self.ad.frame_count - mark_f) / (now - mark_t)
            self._fps_mark = (now, self.ad.frame_count)
        party = []
        battle: dict[str, Any] | None = None
        if snap is not None:
            party = [{"species": m.name, "level": m.level, "hp": m.hp, "max_hp": m.max_hp, "types": list(m.types),
                      "status": m.status, "moves": [{"name": mv.name, "type": mv.type, "power": mv.power,
                                                     "pp": mv.pp} for mv in m.moves]} for m in snap.party]
            if snap.battle is not None and self.battle is not None:
                me, enemy = snap.battle.me, snap.battle.enemy
                battle = {"kind": snap.battle.kind,
                          "me": {"species": me.name, "level": me.level, "hp": me.hp, "max_hp": me.max_hp,
                                 "types": list(me.types)} if me else None,
                          "enemy": {"species": enemy.name, "level": enemy.level, "hp": enemy.hp,
                                    "max_hp": enemy.max_hp, "types": list(enemy.types)} if enemy else None,
                          "choice": self.battle.get("choice"), "reason": self.battle.get("reason"),
                          "effectiveness": self.battle.get("eff"), "scores": self.battle.get("scores", {})}
        return {"kind": "universal", "game": self.ad.key, "generation": self.ad.generation,
                "agent": f"ARES-U universal (sin campana; combate: {self.brain_name})", "sequence": self.sequence,
                "frames": self.ad.frame_count, "fps": round(self.fps, 1), "badges": snap.badges if snap else 0,
                "milestones": [name for name, _ in sorted(self.milestones.items(), key=lambda kv: kv[1]["frame"])],
                "milestone_frames": {k: v["frame"] for k, v in self.milestones.items()},
                "maps_visited": len(self.world.visited_maps),
                "map": self.ad.map_name(snap.map) if snap else "", "position": [snap.x, snap.y] if snap else [],
                "money": snap.money if snap else 0, "party": party, "goal": self.goal, "battle": battle,
                "stats": {**self.stats, "progress": round(self.progress, 1), "stuck_level": self.stuck_level,
                          "train_level": self.train_level},
                "knowledge": {"inherited": self.inherited, "items": self.kb.summary(),
                              "type_chart_learned": len(self.kb.types)},
                "events": list(self.events)[-20:]}

    def publish(self) -> None:
        self.last_publish = time.time()
        if not self.publish_enabled:
            return
        try:
            from core.live_status import publish

            publish(self.run_id, self.status(), frame=self.ad.framebuffer())
        except Exception as exc:  # noqa: BLE001 - live view must never stop the run
            if self.tick % 500 == 0:
                print(f"publish failed: {exc}")
