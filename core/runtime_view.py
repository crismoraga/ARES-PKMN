"""Read-only snapshots of the real campaign workers' atomically published frames.

This module never opens an emulator, advances hardware, loads a game savestate,
or writes campaign files. A retained image remains visible during publication
races, with its age shown instead of inventing progress.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

FRAME_WIDTH = 240
FRAME_HEIGHT = 160
RGB_BYTES = FRAME_WIDTH * FRAME_HEIGHT * 3


def process_identity(pid: int) -> str | None:
    """Return an OS process creation identity, never trusting a recycled PID."""
    if pid < 1:
        return None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
                                          ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
                                          ctypes.POINTER(wintypes.FILETIME)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                          ctypes.byref(system), ctypes.byref(user)):
                return None
            if exited.dwHighDateTime or exited.dwLowDateTime:
                return None
            return f"windows:{(created.dwHighDateTime << 32) | created.dwLowDateTime}"
        finally:
            kernel.CloseHandle(handle)
    try:
        # Linux starttime survives exec and distinguishes PID reuse. The fallback
        # supports other Unix hosts, while freshness still limits old telemetry.
        stat = Path(f"/proc/{pid}/stat")
        if stat.is_file():
            fields = stat.read_text().rsplit(")", 1)[1].split()
            return None if fields[0] == "Z" else f"linux:{fields[19]}"
        os.kill(pid, 0)
        return f"pid:{pid}"
    except (OSError, ValueError, IndexError):
        return None


def campaign_liveness(status: Mapping[str, object], *, stale_after: float = 15.0) -> dict[str, object]:
    """Decorate telemetry with current OS liveness without changing campaign files."""
    result = dict(status)
    reported = bool(status.get("reported_running", status.get("running", False)))
    result["reported_running"] = reported
    result["running"] = False
    if not reported:
        result["liveness"] = "stopped"
        return result
    pid = status_number(status, "pid")
    observed = process_identity(pid)
    expected = status.get("process_identity")
    alive = observed is not None and (not isinstance(expected, str) or expected == observed)
    result["process_alive"] = alive
    if not alive:
        result["liveness"] = "dead"
        result["liveness_message"] = "El proceso terminó sin publicar su cierre"
        return result
    published = status.get("published_at")
    age = time.time() - float(published) if isinstance(published, (int, float)) else float("inf")
    if age > stale_after:
        result["liveness"] = "unresponsive"
        result["liveness_message"] = "Proceso vivo sin actualizar su estado"
        return result
    result.update(running=True, liveness="running")
    return result


def read_campaign_status(directory: Path) -> dict[str, object]:
    """Prefer the freshest valid status, including error reports without an RGB."""
    selected: dict[str, object] = {}
    newest = -1.0
    for name in ("live.json", "status.json"):
        try:
            loaded: object = json.loads((directory / name).read_bytes())
            if not isinstance(loaded, dict):
                continue
            candidate = cast(dict[str, object], loaded)
            published = candidate.get("published_at", 0)
            stamp = float(published) if isinstance(published, (int, float)) else 0.0
            if stamp >= newest:
                selected, newest = candidate, stamp
        except (OSError, UnicodeError, ValueError):
            continue
    return selected


@dataclass(frozen=True)
class LiveCampaignSnapshot:
    game: str
    rgb: bytes | None = None
    status: Mapping[str, object] = field(default_factory=dict)
    received_at: float = 0.0
    message: str = "Esperando la primera imagen del emulador…"

    @property
    def age_seconds(self) -> float:
        published = self.status.get("published_at")
        if isinstance(published, (int, float)) and not isinstance(published, bool):
            return max(0.0, time.time() - float(published))
        return max(0.0, time.monotonic() - self.received_at) if self.received_at else 0.0


class LiveCampaignReader:
    """Bounded file reader; incomplete JSON/RGB updates keep the last good frame."""

    def __init__(self, checkpoint_root: Path, game: str, backend: str = "native") -> None:
        if game not in ("firered", "emerald") or backend not in ("native", "synthetic"):
            raise ValueError("unsupported live campaign selection")
        self.directory = checkpoint_root / backend / game
        self.game = game
        self.backend = backend
        self.snapshot = LiveCampaignSnapshot(game)
        self.events: tuple[Mapping[str, object], ...] = ()
        self._signature: tuple[int, int, int, int] | None = None
        self._event_signature: tuple[int, int] | None = None
        self._last_event_poll = 0.0
        self._last_liveness_poll = 0.0

    def _refresh_liveness(self) -> LiveCampaignSnapshot:
        now = time.monotonic()
        if now - self._last_liveness_poll >= 1.0:
            self._last_liveness_poll = now
            status = self.snapshot.status
            latest = read_campaign_status(self.directory)
            if latest:
                status = latest
            self.snapshot = LiveCampaignSnapshot(self.game, self.snapshot.rgb,
                campaign_liveness(status) if status else {}, self.snapshot.received_at, self.snapshot.message)
        return self.snapshot

    def poll(self) -> LiveCampaignSnapshot:
        """Read a complete generation without changing worker state."""
        json_path, image_path = self.directory / "live.json", self.directory / "live.rgb"
        try:
            js, rgb_stat = json_path.stat(), image_path.stat()
            signature = (js.st_mtime_ns, js.st_size, rgb_stat.st_mtime_ns, rgb_stat.st_size)
            if signature == self._signature:
                return self._refresh_liveness()
            if js.st_size > 2_000_000 or rgb_stat.st_size != RGB_BYTES:
                raise ValueError("publicación incompleta")
            raw_json = json_path.read_bytes()
            loaded: object = json.loads(raw_json)
            if not isinstance(loaded, dict):
                raise ValueError("estado del emulador inválido")
            status = cast(dict[str, object], loaded)
            rgb = image_path.read_bytes()
            if len(rgb) != RGB_BYTES or json_path.read_bytes() != raw_json:
                raise ValueError("actualización en curso")
            expected_hash = status.get("rgb_sha256")
            if isinstance(expected_hash, str) and hashlib.sha256(rgb).hexdigest() != expected_hash:
                raise ValueError("actualización en curso")
            self.snapshot = LiveCampaignSnapshot(self.game, rgb, status, time.monotonic(), "")
            self._signature = signature
            self._last_liveness_poll = 0.0
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            message = "Esperando datos del emulador…" if self.snapshot.rgb is None else "Esperando la siguiente actualización…"
            if isinstance(error, ValueError) and not isinstance(error, json.JSONDecodeError):
                message = str(error).capitalize()
            self.snapshot = LiveCampaignSnapshot(self.game, self.snapshot.rgb, self.snapshot.status,
                                                  self.snapshot.received_at, message)
        return self._refresh_liveness()

    def poll_events(self, *, limit: int = 6, maximum_bytes: int = 65536) -> tuple[Mapping[str, object], ...]:
        """Read only a small tail, at most once per second regardless of log size."""
        now = time.monotonic()
        if now - self._last_event_poll < 1:
            return self.events
        self._last_event_poll = now
        if limit < 1 or not 1024 <= maximum_bytes <= 1_000_000:
            raise ValueError("invalid bounded event-tail request")
        try:
            path = self.directory / "events.jsonl"
            info = path.stat()
            signature = (info.st_mtime_ns, info.st_size)
            if signature == self._event_signature:
                return self.events
            with path.open("rb") as stream:
                offset = max(0, info.st_size - maximum_bytes)
                stream.seek(offset)
                tail = stream.read(maximum_bytes)
            lines = tail.splitlines()
            if offset:
                lines = lines[1:]  # first record may have started before the bounded tail
            parsed: list[Mapping[str, object]] = []
            for line in lines[-max(limit * 4, 20):]:
                try:
                    value: object = json.loads(line)
                except (UnicodeError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict):
                    parsed.append(cast(dict[str, object], value))
            self.events = tuple(parsed[-limit:])
            self._event_signature = signature
        except OSError:
            return self.events
        return self.events


def nested_mapping(values: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = values.get(key)
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def status_number(values: Mapping[str, object], key: str) -> int:
    value = values.get(key, 0)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


EVENT_LABELS: dict[str, str] = {
    "battle_decision": "Decisión de combate", "battle_entered": "Inicio de combate",
    "native_battle_result": "Resultado de combate", "checkpoint": "Partida guardada",
    "campaign_blocked": "Ruta pendiente", "movement": "Movimiento",
    "milestone_observed": "Objetivo observado", "team_plan": "Equipo planificado",
    "capture_decision": "Decisión de captura", "faint": "Pokémon debilitado",
    "stall_detected": "Atasco detectado", "stall_recovery": "Recuperación",
    "stall_rewind": "Retroceso a checkpoint", "stall_rewind_failed": "Retroceso fallido",
    "agent_fault": "Fallo contenido", "scene_started": "Escena", "evolution": "Evolución",
    "level_up": "Subida de nivel", "moves_changed": "Movimientos", "party_joined": "Nuevo miembro",
    "party_healed": "Equipo curado", "route_committed": "Ruta fijada", "transition_tabu": "Salida vetada",
    "exploration_detour": "Desvío exploratorio", "oscillation_detected": "Oscilación",
    "field_move": "Movimiento de campo", "objective_defeat": "Derrota ante objetivo",
    "map_transition": "Cambio de mapa", "checkpoint_restored": "Checkpoint cargado",
    "memory_snapshot_retry": "Reintento de lectura", "native_callback": "Pantalla de inicio",
    "passage_refused": "Paso denegado (aprendido)", "gate_requirement": "Requisito de paso",
    "gate_relaxed": "Verificando paso cerrado", "strategy_shift": "Cambio de estrategia",
    "shop_opened": "Tienda abierta", "shop_closed": "Compra terminada",
    "transition_tabu_lifted": "Vetos levantados", "options_result": "Opciones del juego"}
_OUTCOMES = {1: "victoria", 2: "derrota", 3: "empate", 4: "huida", 5: "teletransporte",
             6: "el rival huyó", 7: "captura", 8: "sin Safari Balls", 9: "rendición"}


def event_caption(event: Mapping[str, object]) -> str:
    """Compact visible event summary, preserving the actual frame and result."""
    kind = str(event.get("kind", "evento"))
    details = ""
    if kind == "battle_decision":
        summary = nested_mapping(event, "summary")
        decision = nested_mapping(event, "decision")
        action = str(decision.get("action", summary.get("action", "")))
        action = {"move": "movimiento", "switch": "cambio", "ball": "captura", "run": "huida"}.get(action, action)
        name = summary.get("move")
        details = f" · {action} {name}" if name else f" · {action} {status_number(decision, 'index') + 1}"
    elif kind == "native_battle_result":
        outcome = status_number(event, "outcome")
        details = " · " + _OUTCOMES.get(outcome, str(outcome))
    elif kind in ("milestone_observed", "objective_defeat"):
        details = f" · {event.get('milestone', event.get('objective', ''))}"
    elif kind == "evolution":
        details = f" · {event.get('from_name', '')} → {event.get('to_name', '')}"
    elif kind == "level_up":
        details = f" · {event.get('name', '')} N{status_number(event, 'level')}"
    elif kind in ("stall_recovery", "stall_detected"):
        recovery = nested_mapping(event, "recovery")
        details = f" · intento {status_number(event, 'attempt') or status_number(recovery, 'attempt')}"
    elif kind == "agent_fault":
        details = f" · {str(event.get('error', ''))[:60]}"
    elif kind == "passage_refused":
        details = f" · {event.get('map', '')} {event.get('tile', '')} requiere {event.get('requirement', '?')}"
    elif kind == "gate_requirement":
        details = f" · {event.get('requirement', '')} ({event.get('target', '')})"
    elif kind == "strategy_shift":
        details = f" · {event.get('change', '')}"
    elif kind == "shop_closed":
        bought = event.get("bought")
        details = f" · {sum(int(value) for value in bought.values()) if isinstance(bought, dict) else 0} objetos"
    return f"{status_number(event, 'frame'):>8,}  {EVENT_LABELS.get(kind, kind)}{details}"


def read_jsonl_tail(path: Path, *, limit: int = 200, maximum_bytes: int = 1_000_000,
                    kinds: frozenset[str] | None = None) -> list[dict[str, object]]:
    """Parse at most ``limit`` trailing JSON objects from a bounded file tail."""
    if limit < 1 or not 1024 <= maximum_bytes <= 32_000_000:
        raise ValueError("invalid bounded JSONL tail request")
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            offset = max(0, size - maximum_bytes)
            stream.seek(offset)
            tail = stream.read(maximum_bytes)
    except OSError:
        return []
    lines = tail.splitlines()
    if offset:
        lines = lines[1:]
    parsed: list[dict[str, object]] = []
    for line in reversed(lines):
        try:
            value: object = json.loads(line)
        except (UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and (kinds is None or value.get("kind") in kinds):
            parsed.append(cast(dict[str, object], value))
            if len(parsed) >= limit:
                break
    parsed.reverse()
    return parsed


def list_checkpoints(directory: Path, limit: int = 60) -> list[dict[str, object]]:
    """Newest checkpoint manifests (frame, reason, size) without loading any state."""
    rows: list[tuple[int, dict[str, object]]] = []
    for path in directory.glob("frame-*"):
        try:
            created = int(path.name.rsplit("-", 1)[1])
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, IndexError):
            continue
        if not isinstance(manifest, dict):
            continue
        size = sum(item.stat().st_size for item in path.iterdir() if item.is_file())
        rows.append((created, {"name": path.name, "frame": manifest.get("frame"), "reason": manifest.get("reason"),
                               "created": created / 1e9, "bytes": size, "pinned": (path / "PINNED").exists()}))
    rows.sort(key=lambda row: row[0], reverse=True)
    return [row for _, row in rows[:limit]]
