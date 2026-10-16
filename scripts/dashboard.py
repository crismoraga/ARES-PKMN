"""Local web monitoring station for both campaigns (read-only, standard library only).

The server never opens an emulator or writes game state. It reads the files the
workers already publish (live.json/live.rgb, telemetry.jsonl, events.jsonl,
supervisor-events.jsonl, checkpoints) and serves them to a single-page app.
The only writes are the same checkpoint/stop request files the CLI creates.
"""
from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import struct
import threading
import time
from typing import cast
from urllib.parse import parse_qs, urlparse
import webbrowser
import zlib

from core.game_data import ROOT
from core.runtime_view import (
    FRAME_HEIGHT, FRAME_WIDTH, RGB_BYTES, campaign_liveness, list_checkpoints,
    read_campaign_status, read_jsonl_tail,
)
from core.training_monitor import read_training

GAMES = ("firered", "emerald")
STATIC = Path(__file__).resolve().parent / "dashboard_static"
_EVENT_NOISE = frozenset({"movement", "direct_damage_observed", "opponent_move_revealed", "memory_snapshot_retry",
                          "neural_weights_loaded"})


def encode_png(rgb: bytes, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> bytes:
    """Minimal lossless PNG (8-bit RGB, filter 0) for one emulator framebuffer."""
    if len(rgb) != width * height * 3:
        raise ValueError("RGB buffer does not match the frame size")
    stride = width * 3
    raw = b"".join(b"\x00" + rgb[row * stride:(row + 1) * stride] for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload +
                struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


def thin(samples: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    """Uniformly downsample a time series, always keeping the newest sample."""
    if len(samples) <= limit or limit < 2:
        return samples
    step = (len(samples) - 1) / (limit - 1)
    return [samples[round(index * step)] for index in range(limit)]


class MonitoringStation:
    """File-backed view of every campaign directory under one checkpoint root."""

    def __init__(self, checkpoint_root: Path, backend: str = "native", *,
                 training_root: Path | None = None) -> None:
        if backend not in ("native", "synthetic"):
            raise ValueError("unsupported backend")
        self.root, self.backend = checkpoint_root, backend
        self.training_root = training_root if training_root is not None else ROOT / "artifacts/training"
        self._png_cache: dict[str, tuple[str, bytes]] = {}
        self._lock = threading.Lock()

    def directory(self, game: str) -> Path:
        if game not in GAMES:
            raise KeyError(game)
        return self.root / self.backend / game

    def training(self, limit: int = 1200) -> dict[str, object]:
        return read_training(self.training_root, limit=limit)

    def overview(self) -> dict[str, object]:
        games: dict[str, object] = {}
        for game in GAMES:
            directory = self.directory(game)
            status = read_campaign_status(directory)
            entry: dict[str, object] = campaign_liveness(status) if status else {"game": game, "liveness": "absent"}
            try:
                heartbeat = json.loads((directory / "heartbeat.json").read_bytes())
                if isinstance(heartbeat, dict):
                    published = heartbeat.get("published_at")
                    heartbeat["age_seconds"] = (round(time.time() - float(published), 1)
                                                if isinstance(published, (int, float)) else None)
                    entry["heartbeat"] = heartbeat
            except (OSError, ValueError):
                pass
            supervisor = read_jsonl_tail(directory / "supervisor-events.jsonl", limit=12, maximum_bytes=65536)
            entry["supervisor"] = supervisor
            if supervisor and supervisor[-1].get("kind") == "worker_restart" and not entry.get("running"):
                entry["liveness"] = "restarting"
            entry["has_frame"] = (directory / "live.rgb").is_file()
            games[game] = entry
        return {"backend": self.backend, "root": str(self.root), "server_time": time.time(), "games": games}

    def frame_png(self, game: str) -> bytes | None:
        path = self.directory(game) / "live.rgb"
        try:
            info = path.stat()
            key = f"{info.st_mtime_ns}:{info.st_size}"
            with self._lock:
                cached = self._png_cache.get(game)
            if cached is not None and cached[0] == key:
                return cached[1]
            rgb = path.read_bytes()
            if len(rgb) != RGB_BYTES:
                return cached[1] if cached is not None else None
            png = encode_png(rgb)
            with self._lock:
                self._png_cache[game] = (key, png)
            return png
        except OSError:
            return None

    def telemetry(self, game: str, limit: int) -> list[dict[str, object]]:
        directory = self.directory(game)
        samples = read_jsonl_tail(directory / "telemetry.jsonl", limit=50000, maximum_bytes=24_000_000)
        if len(samples) < limit:
            older = read_jsonl_tail(directory / "telemetry.1.jsonl", limit=50000, maximum_bytes=24_000_000)
            samples = older + samples
        return thin(samples, limit)

    def events(self, game: str, limit: int, kinds: frozenset[str] | None) -> list[dict[str, object]]:
        directory = self.directory(game)
        if kinds is None:
            rows = read_jsonl_tail(directory / "events.jsonl", limit=limit * 6, maximum_bytes=4_000_000)
            rows = [row for row in rows if row.get("kind") not in _EVENT_NOISE]
            return rows[-limit:]
        return read_jsonl_tail(directory / "events.jsonl", limit=limit, maximum_bytes=8_000_000, kinds=kinds)

    def request(self, game: str, action: str) -> dict[str, object]:
        names = {"checkpoint": "checkpoint.request", "stop": "stop.request"}
        if action not in names:
            raise KeyError(action)
        directory = self.directory(game)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / names[action]).write_text(str(time.time()), encoding="utf-8")
        return {"requested": action, "game": game}


def make_handler(station: MonitoringStation) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ARESMonitor/1.0"

        def log_message(self, format: str, *args: object) -> None:
            return  # Polling every few hundred milliseconds would flood the console.

        def _send(self, status: int, body: bytes, content_type: str, cache: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value: object, status: int = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(value, default=str).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            url = urlparse(self.path)
            query = parse_qs(url.query)
            parts = [part for part in url.path.split("/") if part]
            try:
                if not parts or parts == ["index.html"]:
                    self._send(HTTPStatus.OK, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif parts == ["api", "overview"]:
                    self._json(station.overview())
                elif parts == ["api", "training"]:
                    limit = max(1, min(4000, int(query.get("limit", ["1200"])[0])))
                    self._json(station.training(limit))
                elif len(parts) == 3 and parts[:2] == ["api", "frame"]:
                    png = station.frame_png(parts[2].removesuffix(".png"))
                    if png is None:
                        self._send(HTTPStatus.NOT_FOUND, b"", "image/png")
                    else:
                        self._send(HTTPStatus.OK, png, "image/png")
                elif len(parts) == 3 and parts[:2] == ["api", "telemetry"]:
                    limit = max(10, min(4000, int(query.get("limit", ["1200"])[0])))
                    self._json(station.telemetry(parts[2], limit))
                elif len(parts) == 3 and parts[:2] == ["api", "events"]:
                    limit = max(1, min(2000, int(query.get("limit", ["200"])[0])))
                    raw_kinds = query.get("kinds", [""])[0]
                    kinds = frozenset(kind for kind in raw_kinds.split(",") if kind) or None
                    self._json(station.events(parts[2], limit, kinds))
                elif len(parts) == 3 and parts[:2] == ["api", "checkpoints"]:
                    self._json(list_checkpoints(station.directory(parts[2])))
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (KeyError, ValueError) as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except OSError as error:
                self._json({"error": str(error)}, HTTPStatus.SERVICE_UNAVAILABLE)

        def do_POST(self) -> None:
            parts = [part for part in urlparse(self.path).path.split("/") if part]
            # A custom header cannot be sent by a cross-site HTML form.
            if self.headers.get("X-ARES-Control") != "1":
                self._json({"error": "missing control header"}, HTTPStatus.FORBIDDEN)
                return
            try:
                if len(parts) == 4 and parts[:2] == ["api", "control"]:
                    self._json(station.request(parts[2], parts[3]))
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except KeyError as error:
                self._json({"error": f"unknown {error}"}, HTTPStatus.BAD_REQUEST)
            except OSError as error:
                self._json({"error": str(error)}, HTTPStatus.SERVICE_UNAVAILABLE)

    return Handler


def serve(checkpoint_root: Path, backend: str, host: str, port: int, *,
          training_root: Path | None = None) -> ThreadingHTTPServer:
    station = MonitoringStation(checkpoint_root, backend, training_root=training_root)
    server = ThreadingHTTPServer((host, port), make_handler(station))
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "artifacts/campaigns")
    parser.add_argument("--backend", choices=("native", "synthetic"), default="native")
    parser.add_argument("--training-root", type=Path, default=ROOT / "artifacts/training")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="No abrir el navegador automáticamente")
    args = parser.parse_args()
    server = serve(cast(Path, args.checkpoint_root).resolve(), str(args.backend), str(args.host), int(args.port),
                   training_root=cast(Path, args.training_root).resolve())
    address = f"http://{args.host}:{server.server_address[1]}/"
    print(json.dumps({"dashboard": address, "checkpoint_root": str(args.checkpoint_root),
                      "backend": args.backend, "pid": os.getpid()}), flush=True)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
