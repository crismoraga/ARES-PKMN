"""ARES live dashboard: watch every agent play, its progress, its reasoning and how it gets smarter.

Read-only web server (standard library only) over the live status contract of ``core/live_status.py``
(``artifacts/live/<run_id>/{status.json, frame.png, history.jsonl}``)::

    python -m scripts.live_dashboard                 # http://127.0.0.1:8770
    python -m scripts.live_dashboard --no-browser --log-file artifacts/live_dashboard.log

Endpoints (GET; JSON unless noted)::

    /                        single-page UI (scripts/live_static)
    /api/health              {"ok": true, "pid", "live_root", ...}
    /api/runs                every run: status (knowledge summarised) + age_s, stale, has_frame, frame_etag
    /api/run/<run_id>        one full status.json (+ age_s, stale)
    /api/history/<run_id>    {"rows": history.jsonl downsampled (?limit=400), "series": stats sampled by the dashboard}
    /api/progress            the same for every run in one call (?limit=300)
    /api/intelligence        per-run learning metrics ordered by start: frames per milestone, win rate,
                             stalls per emulated hour, knowledge inherited/learned
    /api/league              artifacts/ares_u/league/report.json + live league runs, raw and normalised
    /api/knowledge           artifacts/universal/knowledge.json + per-run knowledge + the FireRed/Emerald
                             type chart learned by the legacy campaigns, normalised and checked against the real chart
    /frame/<run_id>.png      latest frame; ``Cache-Control: no-cache`` + ETag (304 when unchanged)

Robustness: files are replaced concurrently by the runners, so every read retries on torn JSON or a Windows
sharing violation and falls back to the last good copy; a run whose ``timestamp`` is older than
``--stale-after`` seconds (30) is marked ``stale``. ``history.jsonl`` is read incrementally (only the new
bytes) and downsampled keeping every badge/milestone change point.

``history.jsonl`` carries frames/badges/maps/milestones only, so the dashboard keeps its own compact series of
the battle and stall counters (``artifacts/live_dashboard/series/<run_id>.jsonl``) by sampling each status
every few seconds; that is what the "win rate over time" and "stalls per hour" charts use.

Optional status keys understood besides the contract: ``demo`` (bool, simulated data), ``started_at``
(float), ``sequence`` (int, order of the game in a transfer curriculum), ``platform``, ``league`` (dict with
the same shapes as the league report), ``knowledge.items`` (int, explicit size), ``knowledge.inherited``
(int, items carried over from previous games).
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, cast
from urllib.parse import parse_qs, unquote, urlparse
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "live_static"
LIVE_ROOT = ROOT / "artifacts" / "live"
LEAGUE_DIR = ROOT / "artifacts" / "ares_u" / "league"
KNOWLEDGE_PATH = ROOT / "artifacts" / "universal" / "knowledge.json"
LEGACY_KNOWLEDGE_PATH = ROOT / "artifacts" / "campaigns" / "native" / "shared_knowledge.json"
DASHBOARD_DIR = ROOT / "artifacts" / "live_dashboard"
DEFAULT_PORT = 8770
STALE_AFTER = 30.0
VERSION = "1.0"

RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
STATIC_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png",
                 ".json": "application/json; charset=utf-8", ".ico": "image/x-icon", ".txt": "text/plain; charset=utf-8"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_TAIL = b"IEND\xaeB`\x82"

GAMES: dict[str, dict[str, Any]] = {
    "rojo": {"title": "Pokémon Rojo", "generation": 1, "platform": "gb"},
    "cristal": {"title": "Pokémon Cristal", "generation": 2, "platform": "gbc"},
    "firered": {"title": "Pokémon Rojo Fuego", "generation": 3, "platform": "gba"},
    "emerald": {"title": "Pokémon Esmeralda", "generation": 3, "platform": "gba"},
    "platino": {"title": "Pokémon Platino", "generation": 4, "platform": "nds"},
    "heartgold": {"title": "Pokémon Oro HeartGold", "generation": 4, "platform": "nds"},
    "negro": {"title": "Pokémon Negro", "generation": 5, "platform": "nds"},
    "negro2": {"title": "Pokémon Negro 2", "generation": 5, "platform": "nds"},
    "stadium": {"title": "ARES Stadium", "generation": 3, "platform": "sim"},
}
NOMINAL_FPS = {"gb": 59.7275, "gbc": 59.7275, "gba": 59.7275, "nds": 59.8261}
KIND_ORDER = {"campaign": 0, "universal": 1, "league": 2, "experiment": 3}
SERIES_STATS = {"battles": ("battles", "battles_total", "fights"),
                "wins": ("wins", "victories", "battles_won", "won"),
                "losses": ("losses", "defeats", "battles_lost", "lost"),
                "stalls": ("stalls", "stuck", "stuck_episodes", "stalls_detected", "stagnations"),
                "recoveries": ("recoveries", "recovered", "unstuck", "recoveries_ok")}
KNOWLEDGE_META_KEYS = frozenset({"description", "saved_at", "updated", "updated_at", "timestamp", "version",
                                 "game", "games", "source", "schema", "inherited", "items", "total", "count"})

# ----------------------------------------------------------------------------------------------- type chart
TYPES = ("normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison", "ground", "flying",
         "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy")
# Gen 3 internal type ids (pokeemerald/pokefirered TYPE_* constants); 9 is the "???" type.
GEN3_TYPE_IDS = ("normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel", "mystery",
                 "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark")
TYPE_ALIASES = {"fuego": "fire", "agua": "water", "planta": "grass", "electrico": "electric", "eléctrico": "electric",
                "hielo": "ice", "lucha": "fighting", "veneno": "poison", "tierra": "ground", "volador": "flying",
                "psiquico": "psychic", "psíquico": "psychic", "bicho": "bug", "roca": "rock", "fantasma": "ghost",
                "dragon": "dragon", "dragón": "dragon", "siniestro": "dark", "acero": "steel", "hada": "fairy",
                "normal": "normal", "electr": "electric", "fight": "fighting", "psychc": "psychic"}
# Real chart for generations 2-5 (attacker -> defender -> multiplier; unlisted pairs are x1).
_CHART_GEN2_5: dict[str, dict[str, float]] = {
    "normal": {"rock": 0.5, "steel": 0.5, "ghost": 0.0},
    "fire": {"grass": 2, "ice": 2, "bug": 2, "steel": 2, "fire": 0.5, "water": 0.5, "rock": 0.5, "dragon": 0.5},
    "water": {"fire": 2, "ground": 2, "rock": 2, "water": 0.5, "grass": 0.5, "dragon": 0.5},
    "electric": {"water": 2, "flying": 2, "electric": 0.5, "grass": 0.5, "dragon": 0.5, "ground": 0.0},
    "grass": {"water": 2, "ground": 2, "rock": 2, "fire": 0.5, "grass": 0.5, "poison": 0.5, "flying": 0.5,
              "bug": 0.5, "dragon": 0.5, "steel": 0.5},
    "ice": {"grass": 2, "ground": 2, "flying": 2, "dragon": 2, "fire": 0.5, "water": 0.5, "ice": 0.5, "steel": 0.5},
    "fighting": {"normal": 2, "ice": 2, "rock": 2, "dark": 2, "steel": 2, "poison": 0.5, "flying": 0.5,
                 "psychic": 0.5, "bug": 0.5, "ghost": 0.0},
    "poison": {"grass": 2, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0.0},
    "ground": {"fire": 2, "electric": 2, "poison": 2, "rock": 2, "steel": 2, "grass": 0.5, "bug": 0.5, "flying": 0.0},
    "flying": {"grass": 2, "fighting": 2, "bug": 2, "electric": 0.5, "rock": 0.5, "steel": 0.5},
    "psychic": {"fighting": 2, "poison": 2, "psychic": 0.5, "steel": 0.5, "dark": 0.0},
    "bug": {"grass": 2, "psychic": 2, "dark": 2, "fire": 0.5, "fighting": 0.5, "poison": 0.5, "flying": 0.5,
            "ghost": 0.5, "steel": 0.5},
    "rock": {"fire": 2, "ice": 2, "flying": 2, "bug": 2, "fighting": 0.5, "ground": 0.5, "steel": 0.5},
    "ghost": {"psychic": 2, "ghost": 2, "dark": 0.5, "steel": 0.5, "normal": 0.0},
    "dragon": {"dragon": 2, "steel": 0.5},
    "dark": {"psychic": 2, "ghost": 2, "fighting": 0.5, "dark": 0.5, "steel": 0.5},
    "steel": {"ice": 2, "rock": 2, "fire": 0.5, "water": 0.5, "electric": 0.5, "steel": 0.5},
}
# Generation 1 differences (no dark/steel types).
_GEN1_OVERRIDES = {("bug", "poison"): 2.0, ("poison", "bug"): 2.0, ("ghost", "psychic"): 0.0, ("ice", "fire"): 1.0}
OUTCOME_MULT = {"super": 2.0, "super_effective": 2.0, "neutral": 1.0, "resisted": 0.5, "not_very_effective": 0.5,
                "no_effect": 0.0, "immune": 0.0}


def real_multiplier(attacker: str, defenders: Sequence[str], generation: int = 3) -> float | None:
    """Real type multiplier of the games (generations 1-5); None for types the generation does not have."""
    known = set(TYPES[:-1]) - ({"dark", "steel"} if generation <= 1 else set())
    if attacker not in known or any(defender not in known for defender in defenders):
        return None
    result = 1.0
    for defender in defenders:
        value = _CHART_GEN2_5.get(attacker, {}).get(defender, 1.0)
        if generation <= 1:
            value = _GEN1_OVERRIDES.get((attacker, defender), value)
        result *= value
    return result


def type_name(value: object) -> str | None:
    """Lower-case English type name from an English/Spanish name or a Gen 3 internal id."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return GEN3_TYPE_IDS[value] if 0 <= value < len(GEN3_TYPE_IDS) and value != 9 else None
    if isinstance(value, str):
        key = value.strip().lower()
        if key.isdigit():
            return type_name(int(key))
        key = TYPE_ALIASES.get(key, key)
        return key if key in TYPES else None
    return None


# ------------------------------------------------------------------------------------------------ file reads
_UNREADABLE = object()


def parse_json_file(path: Path, attempts: int = 5, delay: float = 0.02) -> object:
    """Parse JSON another process may be replacing right now.

    Returns the value, ``None`` when the file does not exist, or the ``_UNREADABLE`` sentinel when every
    attempt failed (torn write, Windows sharing violation while ``os.replace`` runs)."""
    for attempt in range(attempts):
        try:
            return cast(object, json.loads(path.read_bytes()))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):  # PermissionError while replaced; JSON/Unicode decode errors on a torn read
            if attempt + 1 < attempts:
                time.sleep(delay * (attempt + 1))
    return _UNREADABLE


def read_png(path: Path, attempts: int = 5, delay: float = 0.02) -> bytes | None:
    """Bytes of a complete PNG (signature and IEND present) or None."""
    for attempt in range(attempts):
        try:
            data = path.read_bytes()
            if data.startswith(PNG_SIGNATURE) and data.endswith(PNG_TAIL):
                return data
        except FileNotFoundError:
            return None
        except OSError:
            pass
        if attempt + 1 < attempts:
            time.sleep(delay * (attempt + 1))
    return None


def file_key(path: Path) -> tuple[int, int] | None:
    try:
        info = path.stat()
    except OSError:
        return None
    return info.st_mtime_ns, info.st_size


class JsonCache:
    """Parsed JSON per path, re-read only when (mtime, size) changes; keeps the last good copy on a torn read."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[Path, tuple[tuple[int, int], object]] = {}

    def get(self, path: Path) -> object:
        key = file_key(path)
        with self._lock:
            entry = self._entries.get(path)
        if key is None:
            return None
        if entry is not None and entry[0] == key:
            return entry[1]
        value = parse_json_file(path)
        if value is _UNREADABLE:
            return entry[1] if entry is not None else None
        with self._lock:
            self._entries[path] = (key, value)
        return value


def _numeric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def compact_row(row: Mapping[str, object]) -> dict[str, float]:
    """Keep the numeric fields of a history line (``t``, frames, badges, maps_visited, fps, milestones, ...)."""
    result: dict[str, float] = {}
    for key, value in row.items():
        number = _numeric(value)
        if number is not None:
            result[key] = number
    return result


def thin(rows: Sequence[Any], limit: int) -> list[Any]:
    """Uniformly downsample, always keeping the first and the last element."""
    if len(rows) <= limit:
        return list(rows)
    if limit < 2:
        return [rows[-1]] if limit == 1 else []
    step = (len(rows) - 1) / (limit - 1)
    return [rows[round(index * step)] for index in range(limit)]


def downsample(rows: Sequence[Mapping[str, float]], limit: int,
               keep: Sequence[str] = ("badges", "milestones")) -> list[Mapping[str, float]]:
    """At most ``limit`` rows: both ends, every row where a ``keep`` counter changes, the rest uniform."""
    count = len(rows)
    if count <= limit:
        return list(rows)
    changes = {0, count - 1}
    for index in range(1, count):
        previous, current = rows[index - 1], rows[index]
        if any(previous.get(key) != current.get(key) for key in keep):
            changes.add(index)
    chosen = sorted(changes)
    if len(chosen) >= limit:
        return [rows[index] for index in thin(chosen, limit)]
    budget = limit - len(chosen)
    uniform = {round(index * (count - 1) / (budget + 1)) for index in range(1, budget + 1)}
    return [rows[index] for index in sorted(changes | uniform)]  # at most len(changes) + budget == limit


@dataclass
class _Tail:
    key: tuple[int, int] = (0, 0)
    offset: int = 0
    partial: bytes = b""
    rows: list[dict[str, float]] = field(default_factory=list)


class HistoryCache:
    """Incremental reader of append-only JSONL files: only the bytes appended since the last call are parsed."""

    def __init__(self, max_rows: int = 40000, chunk: int = 4 << 20) -> None:
        self.max_rows, self.chunk = max_rows, chunk
        self._lock = threading.Lock()
        self._tails: dict[Path, _Tail] = {}

    def rows(self, path: Path) -> list[dict[str, float]]:
        with self._lock:
            tail = self._tails.setdefault(path, _Tail())
            key = file_key(path)
            if key is None:
                self._tails.pop(path, None)
                return []
            if key[1] < tail.offset:  # truncated or replaced: start again
                tail.offset, tail.partial, tail.rows = 0, b"", []
            if key[1] > tail.offset:
                self._read_new(path, tail, key[1])
            tail.key = key
            return list(tail.rows)

    def _read_new(self, path: Path, tail: _Tail, size: int) -> None:
        try:
            with path.open("rb") as handle:
                handle.seek(tail.offset)
                while tail.offset < size:
                    data = handle.read(min(self.chunk, size - tail.offset))
                    if not data:
                        break
                    tail.offset += len(data)
                    lines = (tail.partial + data).split(b"\n")
                    tail.partial = lines.pop()
                    for line in lines:
                        self._parse(line, tail)
                    if len(tail.rows) > self.max_rows:
                        tail.rows = [dict(row) for row in downsample(tail.rows, self.max_rows // 2)]
        except OSError:
            return

    @staticmethod
    def _parse(line: bytes, tail: _Tail) -> None:
        line = line.strip()
        if not line:
            return
        try:
            value = json.loads(line)
        except ValueError:
            return
        if isinstance(value, dict):
            tail.rows.append(compact_row(value))


# ------------------------------------------------------------------------------------------- status helpers
def stat_value(stats: Mapping[str, object], name: str) -> float | None:
    for key in SERIES_STATS.get(name, (name,)):
        number = _numeric(stats.get(key))
        if number is not None:
            return number
    return None


def knowledge_size(knowledge: object, depth: int = 0) -> int:
    """Number of learned items: ``items``/``total``/``count`` when given, else leaves of the structure."""
    if isinstance(knowledge, Mapping):
        for key in ("items", "total_items", "total", "count"):
            number = _numeric(knowledge.get(key))
            if number is not None and depth == 0:
                return int(number)
        total = 0
        for key, value in knowledge.items():
            if key in KNOWLEDGE_META_KEYS or str(key).startswith("_"):
                continue
            if isinstance(value, (list, tuple, set)):
                total += len(value)
            elif isinstance(value, Mapping):
                total += knowledge_size(value, depth + 1)
            elif _numeric(value) is not None and depth > 0:
                total += 1
            elif isinstance(value, str) and depth > 0:
                total += 1
        return total
    if isinstance(knowledge, (list, tuple, set)):
        return len(knowledge)
    return 0


def knowledge_summary(knowledge: object) -> dict[str, Any]:
    """Compact view of a knowledge dict for the run cards: total and per-category counts."""
    if not isinstance(knowledge, Mapping):
        return {"items": knowledge_size(knowledge), "categories": {}}
    categories: dict[str, int] = {}
    for key, value in knowledge.items():
        if key in KNOWLEDGE_META_KEYS or str(key).startswith("_"):
            continue
        if isinstance(value, (list, tuple, set)):
            categories[str(key)] = len(value)
        elif isinstance(value, Mapping):
            categories[str(key)] = knowledge_size(value, 1)
        elif _numeric(value) is not None:
            categories[str(key)] = int(cast(float, _numeric(value)))
    summary: dict[str, Any] = {"items": knowledge_size(knowledge), "categories": categories}
    inherited = _numeric(knowledge.get("inherited"))
    if inherited is not None:
        summary["inherited"] = int(inherited)
    return summary


def game_meta(status: Mapping[str, object]) -> dict[str, Any]:
    game = str(status.get("game", "") or "")
    meta = dict(GAMES.get(game, {"title": game.capitalize() or "?", "generation": None, "platform": None}))
    if isinstance(status.get("generation"), int):
        meta["generation"] = status["generation"]
    if isinstance(status.get("platform"), str):
        meta["platform"] = status["platform"]
    return meta


# ----------------------------------------------------------------------------------------------- the store
class LiveStore:
    """File-backed, read-only view of every run, league report and knowledge file."""

    def __init__(self, live_root: Path = LIVE_ROOT, *, league_dir: Path = LEAGUE_DIR,
                 knowledge_path: Path = KNOWLEDGE_PATH, legacy_knowledge_path: Path | None = LEGACY_KNOWLEDGE_PATH,
                 series_root: Path | None = DASHBOARD_DIR / "series", stale_after: float = STALE_AFTER) -> None:
        self.live_root, self.league_dir = live_root, league_dir
        self.knowledge_path, self.legacy_knowledge_path = knowledge_path, legacy_knowledge_path
        self.series_root, self.stale_after = series_root, stale_after
        self.json = JsonCache()
        self.history_cache = HistoryCache()
        self._frames: dict[str, tuple[str, bytes]] = {}
        self._series: dict[str, list[dict[str, float]]] = {}
        self._series_lock = threading.Lock()
        self.started = time.time()

    # -- runs
    def run_ids(self) -> list[str]:
        if not self.live_root.is_dir():
            return []
        return sorted(path.name for path in self.live_root.iterdir()
                      if RUN_ID.match(path.name) and (path / "status.json").is_file())

    def status(self, run_id: str) -> dict[str, Any] | None:
        if not RUN_ID.match(run_id):
            raise ValueError("invalid run id")
        value = self.json.get(self.live_root / run_id / "status.json")
        if not isinstance(value, dict):
            return None
        status = dict(cast(dict[str, Any], value))
        status.setdefault("run_id", run_id)
        now = time.time()
        stamp = _numeric(status.get("timestamp"))
        if stamp is None:
            key = file_key(self.live_root / run_id / "status.json")
            stamp = key[0] / 1e9 if key else now
        status["age_s"] = round(max(0.0, now - stamp), 1)
        status["stale"] = now - stamp > self.stale_after
        frame_key = file_key(self.live_root / run_id / "frame.png")
        status["has_frame"] = frame_key is not None
        status["frame_etag"] = f"{frame_key[0]:x}-{frame_key[1]:x}" if frame_key else None
        status["meta"] = game_meta(status)
        return status

    def runs(self) -> dict[str, Any]:
        runs: list[dict[str, Any]] = []
        for run_id in self.run_ids():
            try:
                status = self.status(run_id)
            except ValueError:
                continue
            if status is None:
                continue
            status["knowledge"] = knowledge_summary(status.get("knowledge"))
            runs.append(status)
        runs.sort(key=lambda row: (bool(row["stale"]), bool(row.get("demo")),
                                   KIND_ORDER.get(str(row.get("kind")), 9),
                                   row["meta"].get("generation") or 99, str(row["run_id"])))
        live = [row for row in runs if not row["stale"]]
        summary = {"runs": len(runs), "live": len(live),
                   "real_live": sum(1 for row in live if not row.get("demo")),
                   "badges": sum(int(_numeric(row.get("badges")) or 0) for row in runs if row.get("kind") != "league"),
                   "milestones": sum(len(row.get("milestones") or []) for row in runs),
                   "maps": sum(int(_numeric(row.get("maps_visited")) or 0) for row in runs),
                   "wins": sum(int(stat_value(row.get("stats") or {}, "wins") or 0) for row in runs
                               if isinstance(row.get("stats"), Mapping)),
                   "frames": sum(int(_numeric(row.get("frames")) or 0) for row in runs)}
        return {"server_time": time.time(), "stale_after": self.stale_after, "summary": summary, "runs": runs}

    def history(self, run_id: str) -> list[dict[str, float]]:
        if not RUN_ID.match(run_id):
            raise ValueError("invalid run id")
        return self.history_cache.rows(self.live_root / run_id / "history.jsonl")

    def frame(self, run_id: str) -> tuple[str, bytes] | None:
        """(etag, png bytes) of the latest frame; the last good one when the file is being replaced."""
        if not RUN_ID.match(run_id):
            raise ValueError("invalid run id")
        path = self.live_root / run_id / "frame.png"
        key = file_key(path)
        cached = self._frames.get(run_id)
        if key is None:
            return None
        etag = f"{key[0]:x}-{key[1]:x}"
        if cached is not None and cached[0] == etag:
            return cached
        data = read_png(path)
        if data is None:
            return cached
        self._frames[run_id] = (etag, data)
        return etag, data

    # -- dashboard-side stats series
    def _series_path(self, run_id: str) -> Path | None:
        return self.series_root / f"{run_id}.jsonl" if self.series_root is not None else None

    def series(self, run_id: str) -> list[dict[str, float]]:
        if not RUN_ID.match(run_id):
            raise ValueError("invalid run id")
        with self._series_lock:
            if run_id not in self._series:
                path = self._series_path(run_id)
                rows = HistoryCache().rows(path) if path is not None else []
                self._series[run_id] = rows
            return list(self._series[run_id])

    def sample(self) -> int:
        """Append one compact stats row per run whose status changed (the dashboard's own time series)."""
        written = 0
        for run_id in self.run_ids():
            status = self.status(run_id)
            if status is None:
                continue
            row = sample_row(status)
            if row is None:
                continue
            existing = self.series(run_id)
            last = existing[-1] if existing else None
            if last is not None:
                same = all(last.get(key) == value for key, value in row.items() if key != "t")
                if row["t"] <= last.get("t", 0.0) or (same and row["t"] - last.get("t", 0.0) < 60.0):
                    continue
            with self._series_lock:
                rows = self._series.setdefault(run_id, [])
                rows.append(row)
                if len(rows) > 20000:
                    self._series[run_id] = [dict(item) for item in downsample(rows, 10000, keep=("wins", "stalls"))]
            path = self._series_path(run_id)
            if path is not None:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row) + "\n")
                except OSError:
                    pass
            written += 1
        return written

    # -- aggregate views
    def progress(self, limit: int) -> dict[str, Any]:
        runs = []
        for run_id in self.run_ids():
            status = self.status(run_id)
            if status is None:
                continue
            runs.append({"run_id": run_id, "game": status.get("game"), "kind": status.get("kind"),
                         "demo": bool(status.get("demo")), "meta": status["meta"], "stale": status["stale"],
                         "rows": downsample(self.history(run_id), limit),
                         "series": downsample(self.series(run_id), limit, keep=("wins", "stalls"))})
        return {"server_time": time.time(), "runs": runs}

    def intelligence(self) -> dict[str, Any]:
        rows = []
        for run_id in self.run_ids():
            status = self.status(run_id)
            if status is None:
                continue
            rows.append(run_metrics(status, self.history(run_id), self.series(run_id)))
        rows.sort(key=lambda row: (row.get("sequence") if row.get("sequence") is not None else 1e9,
                                   row.get("started_at") or 1e18))
        return {"server_time": time.time(), "runs": rows, "transfer": transfer_summary(rows)}

    def league(self) -> dict[str, Any]:
        report_path = self.league_dir / "report.json"
        report = self.json.get(report_path)
        key = file_key(report_path)
        files = []
        if self.league_dir.is_dir():
            for path in sorted(self.league_dir.glob("*.json")):
                info = file_key(path)
                if info is not None:
                    files.append({"name": path.name, "bytes": info[1], "mtime": info[0] / 1e9})
        live = []
        for run_id in self.run_ids():
            status = self.status(run_id)
            if status is None or status.get("kind") != "league":
                continue
            data = status.get("league") if isinstance(status.get("league"), Mapping) else status
            live.append({"run_id": run_id, "stale": status["stale"], "age_s": status["age_s"],
                         "demo": bool(status.get("demo")), "frames": status.get("frames"),
                         "goal": status.get("goal"), "events": status.get("events", []),
                         "stats": status.get("stats", {}), "agent": status.get("agent"),
                         "normalized": normalize_league(data)})
        return {"server_time": time.time(), "report_path": str(report_path),
                "report_mtime": key[0] / 1e9 if key else None,
                "report": report if report is not None else None,
                "normalized": normalize_league(report) if isinstance(report, Mapping) else None,
                "files": files, "live": live}

    def knowledge(self) -> dict[str, Any]:
        universal = self.json.get(self.knowledge_path)
        legacy = self.json.get(self.legacy_knowledge_path) if self.legacy_knowledge_path is not None else None
        runs = []
        for run_id in self.run_ids():
            status = self.status(run_id)
            if status is None or not status.get("knowledge"):
                continue
            knowledge = status.get("knowledge")
            generation = status["meta"].get("generation") or 3
            runs.append({"run_id": run_id, "game": status.get("game"), "demo": bool(status.get("demo")),
                         "stale": status["stale"], "summary": knowledge_summary(knowledge),
                         "type_chart": type_chart_cells(knowledge, int(generation))})
        return {"server_time": time.time(),
                "universal_path": str(self.knowledge_path),
                "universal": universal,
                "universal_summary": knowledge_summary(universal) if universal is not None else None,
                "universal_type_chart": type_chart_cells(universal, 3) if universal is not None else None,
                "legacy_path": str(self.legacy_knowledge_path) if self.legacy_knowledge_path else None,
                "legacy_summary": knowledge_summary(legacy) if legacy is not None else None,
                "legacy_type_chart": type_chart_cells(legacy, 3) if legacy is not None else None,
                "runs": runs}


def sample_row(status: Mapping[str, Any]) -> dict[str, float] | None:
    stamp = _numeric(status.get("timestamp"))
    if stamp is None:
        return None
    row: dict[str, float] = {"t": round(stamp, 3)}
    frames = _numeric(status.get("frames"))
    if frames is not None:
        row["frames"] = frames
    stats = status.get("stats")
    if isinstance(stats, Mapping):
        for name in SERIES_STATS:
            value = stat_value(stats, name)
            if value is not None:
                row[name] = value
    knowledge = status.get("knowledge")
    if knowledge:
        row["knowledge"] = float(knowledge_size(knowledge))
    badges = _numeric(status.get("badges"))
    if badges is not None:
        row["badges"] = badges
    row["milestones"] = float(len(status.get("milestones") or []))
    return row


# ------------------------------------------------------------------------------------------------ metrics
def milestone_frames(rows: Sequence[Mapping[str, float]]) -> list[float]:
    """Emulated frame at which each milestone *after the first observed row* was reached."""
    result: list[float] = []
    best: float | None = None
    for row in rows:
        frames, milestones = row.get("frames"), row.get("milestones")
        if frames is None or milestones is None:
            continue
        if best is None:
            best = milestones
            continue
        while best < milestones:
            result.append(frames)
            best += 1
    return result


def run_metrics(status: Mapping[str, Any], history: Sequence[Mapping[str, float]],
                series: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    """Learning metrics of one run (everything measured from its own published data; None when unknown)."""
    meta = cast(Mapping[str, Any], status.get("meta") or game_meta(status))
    stats = cast(Mapping[str, object], status.get("stats") if isinstance(status.get("stats"), Mapping) else {})
    frames = _numeric(status.get("frames")) or (history[-1].get("frames", 0.0) if history else 0.0)
    milestones = len(status.get("milestones") or []) or int(history[-1].get("milestones", 0) if history else 0)
    first = history[0] if history else {}
    last = history[-1] if history else {}
    reached = milestone_frames(history)
    start_frames = first.get("frames", 0.0)
    intervals = []
    previous = start_frames
    for value in reached:
        if value >= previous:
            intervals.append(value - previous)
        previous = value
    gained = len(reached)
    window_frames = (last.get("frames", 0.0) - start_frames) if history else 0.0
    wins, losses = stat_value(stats, "wins"), stat_value(stats, "losses")
    decided = (wins or 0.0) + (losses or 0.0)
    stalls, recoveries = stat_value(stats, "stalls"), stat_value(stats, "recoveries")
    platform = str(meta.get("platform") or "")
    fps = NOMINAL_FPS.get(platform)
    emu_hours = frames / fps / 3600.0 if fps and frames > 0 else None
    knowledge_now = knowledge_size(status.get("knowledge")) if status.get("knowledge") else None
    inherited: float | None = None
    if isinstance(status.get("knowledge"), Mapping):
        inherited = _numeric(cast(Mapping[str, object], status["knowledge"]).get("inherited"))
    knowledge_rows = [row for row in series if "knowledge" in row]
    if inherited is None and knowledge_rows:
        inherited = knowledge_rows[0]["knowledge"]
    win_curve = []
    for index, row in enumerate(series):
        w, l_ = row.get("wins"), row.get("losses")
        if w is None or l_ is None or w + l_ <= 0:
            continue
        back = series[max(0, index - 12)]
        dw, dl = w - back.get("wins", 0.0), l_ - back.get("losses", 0.0)
        recent = dw / (dw + dl) if dw + dl > 0 else None
        win_curve.append({"t": row["t"], "frames": row.get("frames"), "cumulative": w / (w + l_), "recent": recent})
    stall_curve = [{"t": row["t"], "frames": row.get("frames"), "stalls": row.get("stalls")}
                   for row in series if "stalls" in row]
    started = _numeric(status.get("started_at")) or (first.get("t") if history else None) or \
        (series[0].get("t") if series else None)
    sequence = status.get("sequence")
    return {
        "run_id": status.get("run_id"), "game": status.get("game"), "kind": status.get("kind"),
        "agent": status.get("agent"), "demo": bool(status.get("demo")), "stale": bool(status.get("stale")),
        "meta": dict(meta), "sequence": sequence if isinstance(sequence, int) else None,
        "started_at": started, "frames": frames, "badges": _numeric(status.get("badges")),
        "milestones": milestones, "maps_visited": _numeric(status.get("maps_visited")),
        "milestones_observed": gained, "observed_frames": window_frames,
        "frames_per_milestone": window_frames / gained if gained else None,
        "frames_per_milestone_total": frames / milestones if milestones else None,
        "first_milestone_frames": (reached[0] - start_frames) if reached else None,
        "milestone_intervals": intervals[-60:],
        "battles": stat_value(stats, "battles"), "wins": wins, "losses": losses,
        "win_rate": (wins or 0.0) / decided if decided > 0 else None,
        "stalls": stalls, "recoveries": recoveries,
        "emulated_hours": emu_hours,
        "stalls_per_hour": stalls / emu_hours if stalls is not None and emu_hours and emu_hours >= 0.05 else None,
        "knowledge": knowledge_now, "knowledge_inherited": inherited,
        "knowledge_learned": (knowledge_now - inherited) if knowledge_now is not None and inherited is not None
        else None,
        "win_curve": thin(win_curve, 200), "stall_curve": thin(stall_curve, 200),
        "knowledge_curve": thin([{"t": row["t"], "frames": row.get("frames"), "knowledge": row["knowledge"]}
                                 for row in knowledge_rows], 200),
    }


def transfer_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """First vs latest real (non-demo) game in play order: the honest headline of the intelligence page."""
    real = [row for row in rows if not row.get("demo") and row.get("kind") in ("universal", "campaign")]
    pool = real if len(real) >= 2 else [row for row in rows if row.get("kind") in ("universal", "campaign")]
    measured = [row for row in pool if row.get("frames_per_milestone")]
    result: dict[str, Any] = {"games": len(pool), "demo": any(row.get("demo") for row in pool)}
    if len(measured) >= 2:
        first, latest = measured[0], measured[-1]
        result.update({"first": {"run_id": first["run_id"], "game": first["game"],
                                 "frames_per_milestone": first["frames_per_milestone"]},
                       "latest": {"run_id": latest["run_id"], "game": latest["game"],
                                  "frames_per_milestone": latest["frames_per_milestone"]},
                       "change": latest["frames_per_milestone"] / first["frames_per_milestone"] - 1.0})
    return result


# ------------------------------------------------------------------------------------------ normalisation
def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _numeric_profile(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result = {}
    for key, item in value.items():
        number = _numeric(item)
        if number is not None:
            result[str(key)] = number
    return result


TRAIT_KEYS = ("aggression", "switch_rate", "status_rate", "setup_rate", "risk", "greed", "defense", "heal_rate",
              "attack_rate", "predictability", "entropy", "temperature", "boost_rate", "protect_rate")


def normalize_league(report: object) -> dict[str, Any] | None:
    """Best-effort extraction of rivals/Elo/cross-play/champion from a league report of unknown exact shape."""
    if not isinstance(report, Mapping):
        return None
    rivals: dict[str, dict[str, Any]] = {}

    def rival(name: str) -> dict[str, Any]:
        return rivals.setdefault(name, {"name": name, "elo": None, "description": None, "profile": {},
                                        "games": None, "win_rate": None})

    source = _first(report, "rivals", "agents", "players", "population", "pool", "roster", "members")
    items: list[tuple[str, object]] = []
    if isinstance(source, Mapping):
        items = [(str(key), value) for key, value in source.items()]
    elif isinstance(source, list):
        for index, entry in enumerate(source):
            if isinstance(entry, Mapping):
                items.append((str(_first(entry, "name", "id", "agent", "label") or f"rival-{index}"), entry))
            elif isinstance(entry, str):
                items.append((entry, {}))
    for name, entry in items:
        target = rival(name)
        if isinstance(entry, Mapping):
            target["elo"] = _numeric(_first(entry, "elo", "rating", "elo_rating", "mu"))
            description = _first(entry, "description", "style", "archetype", "persona", "kind", "policy", "type",
                                 "behaviour_name")
            target["description"] = str(description) if description is not None else None
            profile = _numeric_profile(_first(entry, "profile", "behaviour", "behavior", "traits",
                                              "style_profile", "tendencies", "stats"))
            if not profile:
                profile = {key: cast(float, _numeric(entry[key])) for key in TRAIT_KEYS
                           if key in entry and _numeric(entry[key]) is not None}
            target["profile"] = profile
            target["games"] = _numeric(_first(entry, "games", "matches", "battles", "n"))
            target["win_rate"] = _numeric(_first(entry, "win_rate", "winrate", "wr", "score"))
            if isinstance(entry.get("generation"), int):
                target["generation"] = entry["generation"]
        else:
            target["elo"] = _numeric(entry)
    ratings = _first(report, "elo", "ratings", "leaderboard", "standings", "table", "elo_table")
    if isinstance(ratings, Mapping):
        for name, value in ratings.items():
            number = _numeric(value) if not isinstance(value, Mapping) else \
                _numeric(_first(value, "elo", "rating", "mu"))
            if number is not None:
                rival(str(name))["elo"] = number
    elif isinstance(ratings, list):
        for entry in ratings:
            if isinstance(entry, Mapping):
                name = _first(entry, "name", "id", "agent", "player")
                number = _numeric(_first(entry, "elo", "rating", "mu", "score"))
                if name is not None and number is not None:
                    rival(str(name))["elo"] = number
    ordered = sorted(rivals.values(), key=lambda row: -(row["elo"] if row["elo"] is not None else -1e9))
    matrix = _normalize_matrix(_first(report, "crossplay", "cross_play", "matrix", "pairwise", "head_to_head",
                                      "h2h", "win_matrix", "payoff", "winrates"), [row["name"] for row in ordered])
    champion_value = _first(report, "champion", "best", "leader")
    champion: dict[str, Any] | None = None
    if isinstance(champion_value, Mapping):
        champion = {"name": str(_first(champion_value, "name", "id", "agent") or "campeón"),
                    **{str(k): v for k, v in champion_value.items() if not isinstance(v, (Mapping, list))}}
    elif isinstance(champion_value, str):
        champion = {"name": champion_value}
    elif ordered and ordered[0]["elo"] is not None:
        champion = {"name": ordered[0]["name"], "elo": ordered[0]["elo"], "inferred": True}
    if champion is not None and champion["name"] in rivals and champion.get("elo") is None:
        champion["elo"] = rivals[champion["name"]]["elo"]
    vs_e1 = _first(report, "champion_vs_e1", "vs_e1", "e1", "vs_baseline", "baseline_eval", "e1_eval", "vs_e1_brain")
    history = _normalize_history(_first(report, "elo_history", "history", "generations", "rating_history"))
    meta = {str(key): value for key, value in report.items() if isinstance(value, (str, int, float, bool))}
    return {"rivals": ordered, "matrix": matrix, "champion": champion,
            "vs_e1": vs_e1 if isinstance(vs_e1, (Mapping, int, float, str)) else None,
            "elo_history": history, "meta": meta}


def _normalize_matrix(value: object, names: Sequence[str]) -> dict[str, Any] | None:
    if value is None:
        return None

    def scale(number: float | None) -> float | None:
        return number / 100.0 if number is not None and number > 1.0 else number

    if isinstance(value, Mapping):
        labels = _first(value, "names", "agents", "labels", "players", "rivals")
        grid = _first(value, "values", "matrix", "win_rate", "winrates", "data", "scores")
        if isinstance(labels, list) and isinstance(grid, list):
            rows = [[scale(_numeric(cell)) for cell in row] if isinstance(row, list) else [] for row in grid]
            return {"names": [str(label) for label in labels], "values": rows,
                    "metric": str(_first(value, "metric", "kind") or "win_rate")}
        nested = {str(key): item for key, item in value.items() if isinstance(item, Mapping)}
        if nested:
            labels_set = list(dict.fromkeys(list(nested) + [str(k) for item in nested.values() for k in item]))
            return {"names": labels_set,
                    "values": [[scale(_numeric(cast(Mapping[str, Any], nested.get(a, {})).get(b)))
                                for b in labels_set] for a in labels_set], "metric": "win_rate"}
        return None
    if isinstance(value, list) and value:
        if all(isinstance(row, list) for row in value) and names and len(value) == len(names):
            return {"names": list(names), "values": [[scale(_numeric(cell)) for cell in cast(list[Any], row)]
                                                     for row in value], "metric": "win_rate"}
        records = [entry for entry in value if isinstance(entry, Mapping)]
        if records:
            cells: dict[tuple[str, str], float | None] = {}
            for entry in records:
                a = _first(entry, "a", "row", "p1", "agent", "player", "home", "red")
                b = _first(entry, "b", "col", "p2", "opponent", "away", "blue")
                if a is None or b is None:
                    continue
                cells[(str(a), str(b))] = scale(_numeric(_first(entry, "win_rate", "value", "wr", "score", "p")))
            labels_list = list(dict.fromkeys([a for a, _ in cells] + [b for _, b in cells]))
            return {"names": labels_list, "values": [[cells.get((a, b)) for b in labels_list] for a in labels_list],
                    "metric": "win_rate"}
    return None


def _normalize_history(value: object) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        series = [{"name": str(name), "values": [_numeric(item) for item in values]}
                  for name, values in value.items() if isinstance(values, list)]
        if series:
            length = max(len(entry["values"]) for entry in series)
            return {"x": list(range(length)), "series": series}
        return None
    if isinstance(value, list) and value and all(isinstance(entry, Mapping) for entry in value):
        xs: list[float] = []
        columns: dict[str, list[float | None]] = {}
        for index, entry in enumerate(cast(list[Mapping[str, Any]], value)):
            xs.append(_numeric(_first(entry, "generation", "gen", "round", "step", "x")) or float(index))
            ratings = _first(entry, "elo", "ratings")
            flat = ratings if isinstance(ratings, Mapping) else \
                {k: v for k, v in entry.items() if k not in ("generation", "gen", "round", "step", "x")}
            for name, number in flat.items():
                columns.setdefault(str(name), [None] * index)
                columns[str(name)].append(_numeric(number) if not isinstance(number, Mapping) else
                                          _numeric(_first(number, "elo", "rating")))
            for name in columns:
                if len(columns[name]) < index + 1:
                    columns[name].append(None)
        series = [{"name": name, "values": values} for name, values in columns.items()
                  if any(item is not None for item in values)]
        return {"x": xs, "series": series} if series else None
    return None


def _cell(attacker: object, defender: object, multiplier: object, generation: int,
          source: str) -> dict[str, Any] | None:
    a, d = type_name(attacker), type_name(defender)
    mult = OUTCOME_MULT.get(multiplier.lower()) if isinstance(multiplier, str) else _numeric(multiplier)
    if a is None or d is None or mult is None:
        return None
    real = real_multiplier(a, [d], generation)
    return {"atk": a, "def": d, "mult": mult, "real": real,
            "correct": None if real is None else math.isclose(real, mult), "source": source}


def type_chart_cells(knowledge: object, generation: int = 3) -> dict[str, Any] | None:
    """Single-type cells of a learned type chart (several layouts accepted), checked against the real chart."""
    if not isinstance(knowledge, Mapping):
        return None
    chart = _first(knowledge, "type_chart", "type_effectiveness", "effectiveness", "matchups", "types")
    if chart is None:
        return None
    cells: list[dict[str, Any]] = []
    dual_total = dual_correct = 0
    if isinstance(chart, Mapping) and isinstance(chart.get("facts"), list):  # legacy FR/EM SharedKnowledge
        for fact in cast(list[Any], chart["facts"]):
            if not (isinstance(fact, list) and len(fact) == 2 and isinstance(fact[0], list) and fact[1]):
                continue
            ids, outcomes = fact[0], fact[1]
            outcome = outcomes[0] if isinstance(outcomes, list) else outcomes
            if len(ids) == 2:
                cell = _cell(ids[0], ids[1], outcome, generation, "legacy")
                if cell is not None:
                    cells.append(cell)
            elif len(ids) == 3:
                attacker = type_name(ids[0])
                defenders = [type_name(ids[1]), type_name(ids[2])]
                mult = OUTCOME_MULT.get(str(outcome).lower())
                if attacker and all(defenders) and mult is not None:
                    real = real_multiplier(attacker, [cast(str, d) for d in defenders], generation)
                    dual_total += 1
                    if real is not None and _outcome_class(real) == _outcome_class(mult):
                        dual_correct += 1
    elif isinstance(chart, Mapping):
        for key, value in chart.items():
            if isinstance(value, Mapping):
                for defender, mult in value.items():
                    cell = _cell(key, defender, mult, generation, "learned")
                    if cell is not None:
                        cells.append(cell)
            else:
                parts = re.split(r"\s*(?:->|>|,|\||:|/)\s*", str(key))
                if len(parts) == 2:
                    cell = _cell(parts[0], parts[1], value, generation, "learned")
                    if cell is not None:
                        cells.append(cell)
    elif isinstance(chart, list):
        for entry in chart:
            if isinstance(entry, Mapping):
                cell = _cell(_first(entry, "attacker", "atk", "move_type", "attack"),
                             _first(entry, "defender", "def", "target_type", "defense"),
                             _first(entry, "multiplier", "mult", "value", "effectiveness", "outcome"),
                             generation, "learned")
            elif isinstance(entry, list) and len(entry) == 3:
                cell = _cell(entry[0], entry[1], entry[2], generation, "learned")
            else:
                cell = None
            if cell is not None:
                cells.append(cell)
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in cells:
        unique[(cell["atk"], cell["def"])] = cell
    cells = list(unique.values())
    if not cells and not dual_total:
        return None
    checked = [cell for cell in cells if cell["correct"] is not None]
    used = {cell["atk"] for cell in cells} | {cell["def"] for cell in cells}
    types = [name for name in TYPES if name != "fairy" or "fairy" in used]
    return {"types": types, "cells": cells, "known": len(cells), "checked": len(checked),
            "correct": sum(1 for cell in checked if cell["correct"]),
            "dual_facts": dual_total, "dual_correct": dual_correct, "generation": generation}


def _outcome_class(multiplier: float) -> int:
    return 0 if multiplier == 0 else (1 if multiplier < 1 else (2 if multiplier == 1 else 3))


# --------------------------------------------------------------------------------------------- HTTP layer
class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address: tuple[str, int], store: LiveStore, static: Path = STATIC) -> None:
        self.store, self.static = store, static
        super().__init__(address, DashboardHandler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Browsers drop keep-alive connections all the time (navigation, aborted fetches): not an error."""
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = f"ARESLive/{VERSION}"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return  # the UI polls every second; per-request logs would flood the log file

    @property
    def store(self) -> LiveStore:
        return cast(DashboardServer, self.server).store

    def _send(self, status: int, body: bytes, content_type: str, *, cache: str = "no-store",
              etag: str | None = None, head: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        if etag is not None:
            self.send_header("ETag", f'"{etag}"')
        self.end_headers()
        if not head and body:
            self.wfile.write(body)

    def _json(self, value: object, status: int = HTTPStatus.OK, head: bool = False) -> None:
        body = json.dumps(value, default=str, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8", head=head)

    def do_HEAD(self) -> None:
        self._route(head=True)

    def do_GET(self) -> None:
        self._route(head=False)

    def _route(self, head: bool) -> None:
        url = urlparse(self.path)
        query = parse_qs(url.query)
        parts = [unquote(part) for part in url.path.split("/") if part]
        try:
            self._dispatch(parts, query, head)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return  # the browser cancelled (e.g. a frame request superseded by the next one)
        except ValueError as error:
            self._safe_error(HTTPStatus.BAD_REQUEST, str(error), head)
        except Exception as error:  # noqa: BLE001 - one bad file must never take the dashboard down
            print(f"[{time.strftime('%H:%M:%S')}] error {self.path}: {error!r}", file=sys.stderr, flush=True)
            self._safe_error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal error", head)

    def _safe_error(self, status: int, message: str, head: bool) -> None:
        try:
            self._json({"error": message}, status, head)
        except OSError:
            pass

    def _limit(self, query: Mapping[str, list[str]], default: int) -> int:
        try:
            return max(10, min(5000, int(query.get("limit", [str(default)])[0])))
        except ValueError:
            return default

    def _dispatch(self, parts: list[str], query: Mapping[str, list[str]], head: bool) -> None:
        store = self.store
        if not parts or parts in (["index.html"], ["live"], ["en-vivo"]):
            self._static("index.html", head)
        elif len(parts) == 2 and parts[0] == "static":
            self._static(parts[1], head)
        elif parts == ["favicon.ico"]:
            self._static("favicon.svg", head)
        elif parts == ["api", "health"]:
            self._json({"ok": True, "version": VERSION, "pid": os.getpid(), "live_root": str(store.live_root),
                        "runs": len(store.run_ids()), "uptime_s": round(time.time() - store.started, 1)}, head=head)
        elif parts == ["api", "runs"]:
            self._json(store.runs(), head=head)
        elif len(parts) == 3 and parts[:2] == ["api", "run"]:
            status = store.status(parts[2])
            if status is None:
                self._json({"error": "run not found"}, HTTPStatus.NOT_FOUND, head)
            else:
                self._json(status, head=head)
        elif len(parts) == 3 and parts[:2] == ["api", "history"]:
            run_id = parts[2]
            status = store.status(run_id)
            if status is None:
                self._json({"error": "run not found"}, HTTPStatus.NOT_FOUND, head)
                return
            rows = store.history(run_id)
            limit = self._limit(query, 400)
            self._json({"run_id": run_id, "total_rows": len(rows), "rows": downsample(rows, limit),
                        "series": downsample(store.series(run_id), limit, keep=("wins", "stalls"))}, head=head)
        elif parts == ["api", "progress"]:
            self._json(store.progress(self._limit(query, 300)), head=head)
        elif parts == ["api", "intelligence"]:
            self._json(store.intelligence(), head=head)
        elif parts == ["api", "league"]:
            self._json(store.league(), head=head)
        elif parts == ["api", "knowledge"]:
            self._json(store.knowledge(), head=head)
        elif len(parts) == 2 and parts[0] == "frame" and parts[1].endswith(".png"):
            self._frame(parts[1][:-4], head)
        else:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND, head)

    def _frame(self, run_id: str, head: bool) -> None:
        frame = self.store.frame(run_id)
        if frame is None:
            self._send(HTTPStatus.NOT_FOUND, b"", "image/png", head=head)
            return
        etag, data = frame
        if self.headers.get("If-None-Match", "").strip().strip('"') == etag:
            self._send(HTTPStatus.NOT_MODIFIED, b"", "image/png", cache="no-cache", etag=etag, head=True)
            return
        self._send(HTTPStatus.OK, data, "image/png", cache="no-cache", etag=etag, head=head)

    def _static(self, name: str, head: bool) -> None:
        static = cast(DashboardServer, self.server).static
        suffix = Path(name).suffix.lower()
        if not STATIC_NAME.match(name) or ".." in name or suffix not in CONTENT_TYPES:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND, head)
            return
        path = static / name
        if not path.is_file():
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND, head)
            return
        self._send(HTTPStatus.OK, path.read_bytes(), CONTENT_TYPES[suffix], cache="no-cache", head=head)


class Sampler(threading.Thread):
    """Background thread: the dashboard's own stats series (battles, wins, stalls, knowledge) every ``period`` s."""

    def __init__(self, store: LiveStore, period: float = 5.0) -> None:
        super().__init__(name="ares-live-sampler", daemon=True)
        self.store, self.period = store, period
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.store.sample()
            except Exception as error:  # noqa: BLE001 - sampling is best effort
                print(f"[{time.strftime('%H:%M:%S')}] sampler: {error!r}", file=sys.stderr, flush=True)
            self.stop_event.wait(self.period)


def serve(store: LiveStore, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
          static: Path = STATIC) -> DashboardServer:
    return DashboardServer((host, port), store, static)


class _Tee:
    """Duplicate stdout/stderr into the log file (the background launcher has no console)."""

    def __init__(self, stream: Any, log: Any) -> None:
        self.stream, self.log = stream, log

    def write(self, text: str) -> int:
        for target in (self.stream, self.log):
            try:
                if target is not None:
                    target.write(text)
                    target.flush()
            except (OSError, ValueError):
                pass
        return len(text)

    def flush(self) -> None:
        for target in (self.stream, self.log):
            try:
                if target is not None:
                    target.flush()
            except (OSError, ValueError):
                pass


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARES live dashboard (read-only)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--live-root", type=Path, default=LIVE_ROOT)
    parser.add_argument("--league-dir", type=Path, default=LEAGUE_DIR)
    parser.add_argument("--knowledge", type=Path, default=KNOWLEDGE_PATH)
    parser.add_argument("--series-root", type=Path, default=DASHBOARD_DIR / "series")
    parser.add_argument("--stale-after", type=float, default=STALE_AFTER)
    parser.add_argument("--sample-period", type=float, default=5.0)
    parser.add_argument("--no-sampler", action="store_true")
    parser.add_argument("--no-browser", action="store_true", help="no abrir el navegador")
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.log_file is not None:
        cast(Path, args.log_file).parent.mkdir(parents=True, exist_ok=True)
        log = cast(Path, args.log_file).open("a", encoding="utf-8")
        sys.stdout = cast(Any, _Tee(sys.stdout, log))
        sys.stderr = cast(Any, _Tee(sys.stderr, log))
    store = LiveStore(cast(Path, args.live_root).resolve(), league_dir=cast(Path, args.league_dir).resolve(),
                      knowledge_path=cast(Path, args.knowledge).resolve(),
                      series_root=cast(Path, args.series_root).resolve(), stale_after=float(args.stale_after))
    try:
        server = serve(store, str(args.host), int(args.port))
    except OSError as error:
        print(json.dumps({"error": f"cannot bind {args.host}:{args.port}: {error}"}), flush=True)
        return 2
    url = f"http://{args.host}:{server.server_address[1]}/"
    info = {"dashboard": url, "pid": os.getpid(), "live_root": str(store.live_root), "started": time.time()}
    print(json.dumps(info), flush=True)
    try:
        DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
        (DASHBOARD_DIR / "server.json").write_text(json.dumps(info), encoding="utf-8")
    except OSError:
        pass
    sampler = None
    if not args.no_sampler:
        sampler = Sampler(store, float(args.sample_period))
        sampler.start()
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        if sampler is not None:
            sampler.stop_event.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
