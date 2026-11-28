"""Persistent, game-agnostic knowledge of the universal player (``artifacts/universal/knowledge.json``).

Everything here is learned from play and is meant to transfer to the next game:

* ``types``: type-effectiveness evidence measured in battle (damage ratio and the game's
  effectiveness message), per attacking type and defending type set, tagged with the generation.
* ``menus``: meaning of menu labels discovered by trying them ("LUCHA" -> fight, "HUIR" -> run).
* ``dialog``: words of yes/no questions whose answer was learned from its consequence (a "yes" that
  opened a naming keyboard teaches "no"; a "yes" that healed the party teaches "yes").
* ``targets``: average story progress produced by each kind of exploration target (talk to a new
  NPC, enter an unvisited map, re-talk to an NPC who triggered events before, read a sign, ...).
  The explorer weights its candidates with these returns, so a second game starts with the
  priorities learned in the first one.
* ``healing``: how healing happened (talking to an NPC indoors after a yes/no question) and the
  words of that dialog.
* ``story``: the order of progress events observed (got first Pokemon, first battle, got key item,
  delivered it, badge), used to describe the expected structure of a new game.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .typechart import category_of, textbook_multi

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "artifacts" / "universal" / "knowledge.json"
CATEGORY_VALUE = {"immune": 0.0, "weak": 0.5, "neutral": 1.0, "super": 2.0, "super2": 4.0, "weak2": 0.25}
WORD_RE = re.compile(r"[A-ZÁÉÍÓÚÑÜa-záéíóúñü]{3,}")
TARGET_KINDS = ("new_map", "npc", "item", "retalk", "retalk_important", "sign", "exit", "coverage")
# Weak priors (pseudo-observations) of the explorer before any game was played.
TARGET_PRIOR = {"new_map": 1.0, "npc": 1.0, "item": 1.0, "retalk": 0.3, "retalk_important": 0.6, "sign": 0.3,
                "exit": 0.5, "coverage": 0.2}


def words(text: str) -> list[str]:
    """Upper-cased content words (>= 3 letters) of a dialog line; language-agnostic tokenizer."""
    return [w.upper() for w in WORD_RE.findall(text)]


@dataclass
class Running:
    """Running mean with a count (JSON-friendly)."""
    n: float = 0.0
    mean: float = 0.0

    def add(self, value: float, weight: float = 1.0) -> None:
        self.n += weight
        self.mean += weight * (value - self.mean) / self.n

    def to_json(self) -> list[float]:
        return [round(self.n, 3), round(self.mean, 4)]

    @classmethod
    def from_json(cls, raw: Iterable[float]) -> Running:
        n, mean = list(raw)[:2]
        return cls(float(n), float(mean))


@dataclass
class KnowledgeBase:
    path: Path = DEFAULT_PATH
    types: dict[str, dict[str, Any]] = field(default_factory=dict)      # "fire>grass" -> {gen: [n, mean]}
    menus: dict[str, dict[str, Any]] = field(default_factory=dict)      # label -> {meaning, n, games}
    dialog: dict[str, dict[str, float]] = field(default_factory=dict)   # word -> {"yes": score, "no": score}
    targets: dict[str, Running] = field(default_factory=dict)
    healing: dict[str, Any] = field(default_factory=dict)
    story: list[dict[str, Any]] = field(default_factory=list)
    games: dict[str, dict[str, Any]] = field(default_factory=dict)       # per-game summary of runs
    skills: dict[str, Any] = field(default_factory=dict)                 # e.g. how to leave a naming screen
    version: int = 1

    # ------------------------------------------------------------------ persistence
    @classmethod
    def load(cls, path: Path | None = None) -> KnowledgeBase:
        target = path or DEFAULT_PATH
        kb = cls(path=target)
        if not target.exists():
            return kb
        raw = json.loads(target.read_text(encoding="utf-8"))
        kb.types = raw.get("types", {})
        kb.menus = raw.get("menus", {})
        kb.dialog = raw.get("dialog", {})
        kb.targets = {k: Running.from_json(v) for k, v in raw.get("targets", {}).items()}
        kb.healing = raw.get("healing", {})
        kb.story = raw.get("story", [])
        kb.games = raw.get("games", {})
        kb.skills = raw.get("skills", {})
        return kb

    @classmethod
    def empty(cls, path: Path) -> KnowledgeBase:
        return cls(path=path)

    def to_json(self) -> dict[str, Any]:
        return {"version": self.version, "updated": time.time(), "types": self.types, "menus": self.menus,
                "dialog": self.dialog, "targets": {k: v.to_json() for k, v in self.targets.items()},
                "healing": self.healing, "story": self.story[-200:], "games": self.games, "skills": self.skills}

    def save(self, path: Path | None = None) -> Path:
        target = path or self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=1, ensure_ascii=False), encoding="utf-8")
        for _ in range(5):
            try:
                tmp.replace(target)
                break
            except PermissionError:
                time.sleep(0.05)
        return target

    # ------------------------------------------------------------------ type chart
    @staticmethod
    def _type_key(attacker: str, defenders: tuple[str, ...]) -> str:
        return f"{attacker}>{'/'.join(sorted(set(defenders)))}"

    def observe_effectiveness(self, attacker: str, defenders: tuple[str, ...], multiplier: float,
                              generation: int, weight: float = 1.0) -> None:
        key = self._type_key(attacker, defenders)
        entry = self.types.setdefault(key, {})
        stats = Running.from_json(entry.get(str(generation), [0, 0]))
        stats.add(multiplier, weight)
        entry[str(generation)] = stats.to_json()

    def effectiveness(self, attacker: str, defenders: tuple[str, ...], generation: int,
                      use_textbook: bool = True) -> tuple[float, float]:
        """Belief about a multiplier: (estimate, evidence weight).

        Evidence from this generation counts fully, from other generations half (mechanics differ),
        and the textbook chart is a single pseudo-observation, so a few measured hits dominate it.
        """
        entry = self.types.get(self._type_key(attacker, defenders), {})
        total, weight = 0.0, 0.0
        for gen, raw in entry.items():
            stats = Running.from_json(raw)
            factor = 1.0 if int(gen) == generation else 0.5
            total += stats.mean * stats.n * factor
            weight += stats.n * factor
        if use_textbook:
            total += textbook_multi(attacker, defenders, generation)
            weight += 1.0
        if weight == 0:
            return 1.0, 0.0
        return total / weight, weight

    def type_summary(self) -> dict[str, Any]:
        learned = {}
        for key, entry in self.types.items():
            n = sum(Running.from_json(v).n for v in entry.values())
            mean = sum(Running.from_json(v).mean * Running.from_json(v).n for v in entry.values()) / max(n, 1e-9)
            learned[key] = {"x": round(mean, 2), "n": round(n, 1), "cat": category_of(round(mean * 4) / 4)}
        return learned

    # ------------------------------------------------------------------ menus
    def menu_meaning(self, label: str) -> str | None:
        entry = self.menus.get(label.strip().upper())
        return str(entry["meaning"]) if entry else None

    def learn_menu(self, label: str, meaning: str, game: str) -> None:
        key = label.strip().upper()
        entry = self.menus.setdefault(key, {"meaning": meaning, "n": 0, "games": []})
        if entry["meaning"] != meaning:
            entry.update({"meaning": meaning, "n": 0})
        entry["n"] += 1
        if game not in entry["games"]:
            entry["games"].append(game)

    # ------------------------------------------------------------------ dialog
    def answer(self, question: str) -> tuple[str, float]:
        """'yes' or 'no' for a yes/no question, with the evidence that decided it (0 = default)."""
        score = 0.0
        for word in words(question):
            entry = self.dialog.get(word)
            if entry:
                score += entry.get("yes", 0.0) - entry.get("no", 0.0)
        if score < 0:
            return "no", -score
        return "yes", score

    def learn_answer(self, question: str, good: str, strength: float = 1.0) -> None:
        for word in set(words(question)):
            entry = self.dialog.setdefault(word, {"yes": 0.0, "no": 0.0})
            entry[good] = entry.get(good, 0.0) + strength

    # ------------------------------------------------------------------ exploration returns
    def target_value(self, kind: str) -> float:
        stats = self.targets.get(kind)
        prior = TARGET_PRIOR.get(kind, 0.5)
        if stats is None or stats.n <= 0:
            return prior
        # Shrink toward the prior with 3 pseudo-observations.
        return (prior * 3.0 + stats.mean * stats.n) / (3.0 + stats.n)

    def learn_target(self, kind: str, progress: float) -> None:
        self.targets.setdefault(kind, Running()).add(progress)

    # ------------------------------------------------------------------ healing / story
    def learn_heal(self, game: str, signature: dict[str, Any], dialog_lines: list[str], via: str) -> None:
        self.healing.setdefault("events", 0)
        self.healing["events"] += 1
        self.healing.setdefault("via", {})
        self.healing["via"][via] = self.healing["via"].get(via, 0) + 1
        bag = self.healing.setdefault("words", {})
        for word in set(words(" ".join(dialog_lines))):
            bag[word] = bag.get(word, 0) + 1
        self.healing.setdefault("places", [])
        self.healing["places"].append({"game": game, **signature})
        self.healing["places"] = self.healing["places"][-30:]

    def heal_words(self, min_count: int = 1) -> set[str]:
        return {w for w, n in self.healing.get("words", {}).items() if n >= min_count}

    def record_story(self, game: str, event: str, frame: int) -> None:
        self.story.append({"game": game, "event": event, "frame": frame})

    def summary(self) -> dict[str, Any]:
        return {"type_entries": len(self.types), "menu_labels": {k: v["meaning"] for k, v in self.menus.items()},
                "dialog_words": len(self.dialog), "heal_events": self.healing.get("events", 0),
                "target_values": {k: round(self.target_value(k), 3) for k in TARGET_KINDS},
                "games": sorted(self.games)}
