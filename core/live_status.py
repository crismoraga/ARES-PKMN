"""Live status contract shared by every running agent and the live dashboard.

Every runner (FireRed/Emerald campaigns, the universal player on Rojo/Cristal/..., Stadium league
training) publishes its state under ``artifacts/live/<run_id>/``:

* ``status.json``   latest snapshot (atomic write), schema below;
* ``frame.png``     latest emulator frame (optional, atomic write);
* ``history.jsonl`` one compact line per publish, for progress charts.

``status`` keys (all optional except ``run_id``, ``game`` and ``kind``)::

    run_id: str              unique, e.g. "campaign-firered", "universal-rojo"
    kind: str                "campaign" | "universal" | "league" | "experiment"
    game: str                "firered", "emerald", "rojo", "cristal", ... or "stadium"
    generation: int
    agent: str               short description of the agent / brain in use
    frames: int              emulator frames (or env steps) so far
    fps: float
    badges: int
    milestones: list[str]    completed story milestones, in order
    maps_visited: int
    party: list[dict]        {species, level, hp, max_hp, types, moves}
    goal: str                current objective in words ("Go to Pewter Gym", "Heal at Pokemon Center")
    battle: dict | None      {me, enemy, choice, reason, effectiveness}
    stats: dict              battles, wins, losses, stalls, recoveries, ...
    knowledge: dict          what the agent has learned (e.g. type chart entries, skills)
    events: list[str]        last ~20 human-readable events
    timestamp: float         filled automatically

    from core.live_status import publish
    publish("universal-rojo", {"kind": "universal", "game": "rojo", "frames": 1200, "badges": 0}, frame=rgb)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
LIVE_ROOT = ROOT / "artifacts" / "live"
HISTORY_KEYS = ("frames", "badges", "maps_visited", "fps")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    for _ in range(5):
        try:
            tmp.replace(path)
            return
        except PermissionError:  # a reader on Windows holds the file for a moment
            time.sleep(0.02)
    tmp.unlink(missing_ok=True)


def frame_to_rgb(frame: NDArray[Any]) -> NDArray[np.uint8]:
    """Accept RGB (H, W, 3), grey (H, W) uint8 or packed little-endian XBGR (H, W) uint32."""
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[2] >= 3:
        return np.ascontiguousarray(array[..., :3].astype(np.uint8))
    if array.ndim == 2 and array.dtype in (np.uint32, np.int32, np.int64, np.uint64):
        packed = array.astype(np.uint32)
        return np.stack([(packed & 0xFF), (packed >> 8) & 0xFF, (packed >> 16) & 0xFF], axis=-1).astype(np.uint8)
    if array.ndim == 2:
        grey = array.astype(np.uint8)
        return np.asarray(np.stack([grey, grey, grey], axis=-1), dtype=np.uint8)
    raise ValueError(f"unsupported frame shape {array.shape}")


def publish(run_id: str, status: dict[str, Any], frame: NDArray[Any] | None = None,
            root: Path | None = None, scale: int = 2) -> Path:
    """Write ``status.json`` (+ ``frame.png``) atomically and append one line to ``history.jsonl``."""
    directory = (root or LIVE_ROOT) / run_id
    directory.mkdir(parents=True, exist_ok=True)
    payload = {**status, "run_id": run_id, "timestamp": time.time()}
    payload.setdefault("events", [])
    payload["events"] = list(payload["events"])[-20:]
    _atomic_write_bytes(directory / "status.json",
                        json.dumps(payload, default=_json_default, ensure_ascii=False).encode("utf-8"))
    if frame is not None:
        from PIL import Image  # local import: Pillow is only needed when frames are published

        image = Image.fromarray(frame_to_rgb(frame))
        if scale > 1:
            image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        _atomic_write_bytes(directory / "frame.png", buffer.getvalue())
    line = {"t": payload["timestamp"], **{k: payload[k] for k in HISTORY_KEYS if k in payload},
            "milestones": len(payload.get("milestones", []) or [])}
    with (directory / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, default=_json_default) + "\n")
    return directory


def read_all(root: Path | None = None) -> list[dict[str, Any]]:
    """Latest status of every run (for the dashboard)."""
    base = root or LIVE_ROOT
    result = []
    for path in sorted(base.glob("*/status.json")):
        try:
            result.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return result


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)
