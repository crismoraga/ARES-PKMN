"""Live dashboard: every endpoint the front-end uses, the field names it reads, static files and the JS syntax."""
from __future__ import annotations

from collections.abc import Iterator
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from typing import Any
import urllib.error
import urllib.request

import numpy as np
import pytest

from core.live_status import publish
from scripts.live_dashboard import STATIC, DashboardServer, LiveStore, serve

ROOT = Path(__file__).resolve().parents[1]
JS_FILES = ("util.js", "charts.js", "live.js", "pages.js", "app.js")


def _status(game: str, *, frames: int, badges: int, battle: bool, demo: bool = False) -> dict[str, Any]:
    mon = {"species": "Pikachu", "level": 12, "hp": 20, "max_hp": 35, "types": ["electric"],
           "moves": [{"name": "Impactrueno", "type": "electric", "power": 40}]}
    return {
        "kind": "universal", "game": game, "generation": 1, "agent": "test", "frames": frames, "fps": 900.0,
        "badges": badges, "milestones": ["Elegir inicial", "Medalla Roca"][: badges + 1], "maps_visited": 3 + badges,
        "party": [mon], "goal": "Ir al gimnasio", "demo": demo, "sequence": 1, "started_at": 1.0,
        "battle": {"me": mon, "enemy": {**mon, "species": "Geodude", "types": ["rock", "ground"]}, "choice": "Placaje",
                   "reason": "El rival resiste electricidad", "effectiveness": 0.5,
                   "scores": {"Placaje": 0.6, "Impactrueno": 0.1}} if battle else None,
        "stats": {"battles": 4, "wins": 3, "losses": 1, "stalls": 1, "recoveries": 1},
        "knowledge": {"type_chart": {"fire>grass": 2.0}, "inherited": 0},
        "events": ["f100 · Inicio", {"frame": 200, "text": "Victoria"}],
    }


@pytest.fixture(scope="module")
def base_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    tmp = tmp_path_factory.mktemp("live")
    live = tmp / "live"
    frame = np.zeros((144, 160, 3), dtype=np.uint8)
    for step in range(3):
        publish("t-rojo", _status("rojo", frames=1000 * (step + 1), badges=min(step, 1), battle=step == 2),
                frame=frame, root=live)
        publish("demo-t-cristal", _status("cristal", frames=500 * (step + 1), badges=0, battle=False, demo=True),
                root=live)
    store = LiveStore(live, league_dir=tmp / "league", knowledge_path=tmp / "none.json",
                      legacy_knowledge_path=None, series_root=tmp / "series")
    server: DashboardServer = serve(store, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _get(url: str, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def _json(url: str) -> Any:
    status, _, body = _get(url)
    assert status == 200, url
    return json.loads(body)


def test_runs_fields_used_by_cards(base_url: str) -> None:
    data = _json(base_url + "/api/runs")
    assert {"server_time", "summary", "runs"} <= set(data)
    runs = {run["run_id"]: run for run in data["runs"]}
    assert set(runs) == {"t-rojo", "demo-t-cristal"}
    run = runs["t-rojo"]
    for key in ("game", "kind", "frames", "fps", "badges", "milestones", "maps_visited", "party", "goal", "battle",
                "stats", "events", "stale", "age_s", "has_frame", "frame_etag", "meta"):
        assert key in run, key
    assert run["has_frame"] and run["frame_etag"] and run["stale"] is False
    assert run["meta"]["title"] == "Pokémon Rojo"
    assert {"me", "enemy", "choice", "reason", "effectiveness", "scores"} <= set(run["battle"])
    assert runs["demo-t-cristal"]["demo"] is True and runs["demo-t-cristal"]["has_frame"] is False


def test_run_history_progress_intelligence(base_url: str) -> None:
    assert _json(base_url + "/api/run/t-rojo")["run_id"] == "t-rojo"
    history = _json(base_url + "/api/history/t-rojo?limit=400")
    assert history["rows"] and {"frames", "badges", "milestones", "maps_visited"} <= set(history["rows"][-1])
    assert _get(base_url + "/api/history/nope")[0] == 404
    progress = _json(base_url + "/api/progress?limit=300")
    assert {"run_id", "game", "kind", "demo", "meta", "stale", "rows"} <= set(progress["runs"][0])
    intel = _json(base_url + "/api/intelligence")
    row = next(r for r in intel["runs"] if r["run_id"] == "t-rojo")
    for key in ("frames_per_milestone", "milestones_observed", "first_milestone_frames", "win_curve", "wins",
                "losses", "win_rate", "stalls", "recoveries", "stalls_per_hour", "emulated_hours", "knowledge",
                "knowledge_inherited", "knowledge_learned", "sequence", "battles"):
        assert key in row, key


def test_league_and_knowledge_degrade_without_reports(base_url: str) -> None:
    league = _json(base_url + "/api/league")
    assert league["report"] is None and league["normalized"] is None and league["live"] == []
    knowledge = _json(base_url + "/api/knowledge")
    for key in ("universal_type_chart", "universal_summary", "legacy_type_chart", "legacy_summary", "runs"):
        assert key in knowledge, key
    chart = next(r for r in knowledge["runs"] if r["run_id"] == "t-rojo")["type_chart"]
    assert chart is None or {"types", "cells", "known", "checked", "correct", "generation"} <= set(chart)


def test_frame_etag_and_304(base_url: str) -> None:
    status, headers, body = _get(base_url + "/frame/t-rojo.png")
    assert status == 200 and body.startswith(b"\x89PNG") and headers.get("ETag")
    status, _, body = _get(base_url + "/frame/t-rojo.png", {"If-None-Match": headers["ETag"]})
    assert status == 304 and body == b""


def test_static_files_served(base_url: str) -> None:
    status, headers, body = _get(base_url + "/")
    assert status == 200 and b"<main id=\"app\">" in body
    html = body.decode("utf-8")
    refs = re.findall(r'(?:src|href)="/static/([^"]+)"', html)
    assert set(JS_FILES) <= set(refs) and "favicon.svg" in refs and "app.css" in refs
    for name in refs:
        status, headers, content = _get(f"{base_url}/static/{name}")
        assert status == 200 and content, name
        if name.endswith(".js"):
            assert headers["Content-Type"].startswith("text/javascript")
        if name.endswith(".svg"):
            assert headers["Content-Type"].startswith("image/svg+xml")


def test_js_element_ids_exist() -> None:
    """Every #id the scripts look up is in index.html or created by the scripts themselves."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    sources = "\n".join((STATIC / name).read_text(encoding="utf-8") for name in JS_FILES)
    known = set(re.findall(r'id="([A-Za-z0-9_-]+)"', html)) | set(re.findall(r'id="([A-Za-z0-9_-]+)"', sources))
    wanted = set(re.findall(r"\$\('#([A-Za-z0-9_-]+)'\)", sources)) | set(re.findall(r"getElementById\('([A-Za-z0-9_-]+)'\)", sources))
    wanted |= {m for m in re.findall(r"'#([A-Za-z0-9_-]*[A-Za-z0-9])'", sources) if not re.fullmatch(r"[0-9a-fA-F]{3,8}", m)}
    assert wanted and wanted <= known, sorted(wanted - known)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_syntax() -> None:
    for name in JS_FILES:
        result = subprocess.run(["node", "--check", str(STATIC / name)], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, f"{name}: {result.stderr}"


def test_client_disconnects_are_not_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    server = serve(LiveStore(ROOT / "artifacts" / "__no_live__", series_root=None), port=0)
    try:
        stderr = io.StringIO()
        monkeypatch.setattr(sys, "stderr", stderr)
        try:
            raise ConnectionAbortedError(10053, "aborted")
        except ConnectionAbortedError:
            server.handle_error(None, ("127.0.0.1", 1))
        assert stderr.getvalue() == ""
    finally:
        server.server_close()


def test_launcher_prints_commands() -> None:
    text = (ROOT / "scripts" / "live_all.ps1").read_text(encoding="utf-8")
    for needle in ("scripts.live_dashboard", "scripts.run_campaign --game firered --resume", "scripts.ares_u_league",
                   "scripts.ares_u_universal", "8770"):
        assert needle in text, needle
