"""Read-only snapshots of the real campaign workers' atomically published frames.

This module never opens an emulator, advances hardware, loads a game savestate,
or writes campaign files. A retained image remains visible during publication
races, with its age shown instead of inventing progress.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

FRAME_WIDTH = 240
FRAME_HEIGHT = 160
RGB_BYTES = FRAME_WIDTH * FRAME_HEIGHT * 3


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

    def poll(self) -> LiveCampaignSnapshot:
        """Read a complete generation without changing worker state."""
        json_path, image_path = self.directory / "live.json", self.directory / "live.rgb"
        try:
            js, rgb_stat = json_path.stat(), image_path.stat()
            signature = (js.st_mtime_ns, js.st_size, rgb_stat.st_mtime_ns, rgb_stat.st_size)
            if signature == self._signature:
                return self.snapshot
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
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            message = "Esperando datos del emulador…" if self.snapshot.rgb is None else "Esperando la siguiente actualización…"
            if isinstance(error, ValueError) and not isinstance(error, json.JSONDecodeError):
                message = str(error).capitalize()
            self.snapshot = LiveCampaignSnapshot(self.game, self.snapshot.rgb, self.snapshot.status,
                                                  self.snapshot.received_at, message)
        return self.snapshot

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


def event_caption(event: Mapping[str, object]) -> str:
    """Compact visible event summary, preserving the actual frame and result."""
    kind = str(event.get("kind", "evento"))
    labels = {"battle_decision": "Decisión de combate", "battle_entered": "Inicio de combate",
              "native_battle_result": "Resultado de combate", "checkpoint": "Partida guardada",
              "campaign_blocked": "Ruta pendiente", "movement": "Movimiento",
              "milestone_observed": "Objetivo observado", "team_plan": "Equipo planificado",
              "capture_decision": "Decisión de captura", "faint": "Pokémon debilitado"}
    details = ""
    if kind == "battle_decision":
        decision = nested_mapping(event, "decision")
        action = str(decision.get("action", ""))
        action = {"move": "movimiento", "switch": "cambio", "ball": "captura", "run": "huida"}.get(action, action)
        details = f" · {action} {status_number(decision, 'index') + 1}"
    elif kind == "native_battle_result":
        outcome = status_number(event, "outcome")
        details = " · " + {1: "victoria", 2: "derrota", 7: "captura", 4: "huida"}.get(outcome, str(outcome))
    elif kind == "milestone_observed":
        details = f" · {event.get('milestone', '')}"
    return f"{status_number(event, 'frame'):>8,}  {labels.get(kind, kind)}{details}"
