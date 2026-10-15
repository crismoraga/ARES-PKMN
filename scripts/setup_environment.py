"""Reproduce the supported Windows/Python3.12 development installation with uv."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def _command(arguments: list[str]) -> None:
    """Execute literal argument vectors and preserve nonzero failure codes."""
    print("Executing: " + subprocess.list2cmdline(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, check=True)


def _python() -> Path:
    return ROOT / ".venv" / "Scripts" / "python.exe"


def acquire_sources() -> None:
    """Acquire exact recorded commits, refusing to overwrite a different checkout."""
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git is required to acquire the pinned native sources")
    manifest: dict[str, Any] = json.loads((ROOT / "data/source_manifest.json").read_text(encoding="utf-8"))
    projects = (("mgba", "mgba"), ("firered", "pokefirered"), ("emerald", "pokeemerald"))
    for game, name in projects:
        resource = manifest[game]
        path = ROOT / "native/vendor" / name
        expected = str(resource["source_commit"])
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            _command([git, "init", str(path)])
            _command([git, "-C", str(path), "remote", "add", "origin", str(resource["repository"])])
            _command([git, "-C", str(path), "fetch", "--depth", "1", "origin", expected])
            _command([git, "-C", str(path), "checkout", "--detach", expected])
        actual = subprocess.check_output([git, "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            raise RuntimeError(f"Source commit mismatch for{name}:{actual}; expected{expected}")
        if game != "mgba":
            symbols = ROOT / "data" / f"{name}.sym"
            if not symbols.exists():
                url = f"https://raw.githubusercontent.com/pret/{name}/{resource['symbols_commit']}/{name}.sym"
                with urlopen(url, timeout=60) as response:
                    data = response.read()
                if b"gBattleMons" not in data or b"gSaveBlock1Ptr" not in data:
                    raise RuntimeError(f"Invalid symbol table returned for{game}")
                temporary = symbols.with_suffix(".sym.tmp")
                temporary.write_bytes(data)
                temporary.replace(symbols)


def environment_status() -> dict[str, object]:
    """Inspect prerequisites without creating a venv, fetching, or installing."""
    python = _python()
    result: dict[str, object] = {"workspace": str(ROOT), "platform": sys.platform,
        "python": str(python), "python_exists": python.is_file(),
        "tools": {name: shutil.which(name) for name in ("uv", "git", "cmake", "ninja", "gcc")},
        "native_dll_exists": (ROOT / "native/build/ares_mgba.dll").is_file(),
        "sources": {name: (ROOT / "native/vendor" / name / ".git").is_dir()
                    for name in ("mgba", "pokefirered", "pokeemerald")}}
    if python.is_file():
        version = subprocess.check_output([str(python), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], text=True).strip()
        result["python_version"] = version
    return result


def setup(skip_native_build: bool = False) -> dict[str, object]:
    """Install locked CPU/DirectML wheels in one isolated editable environment."""
    if sys.platform != "win32":
        raise RuntimeError("This lock and native launcher target64-bit Windows/Python3.12")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("Install uv first: winget install --id astral-sh.uv --exact")
    python = _python()
    if not python.is_file():
        _command([uv, "venv", "--python", "3.12", str(ROOT / ".venv")])
    version = subprocess.check_output([str(python), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], text=True).strip()
    if version != "3.12":
        raise RuntimeError(f"Existing venv hasPython{version}; aPython3.12 venv is required")
    _command([uv, "pip", "sync", "--python", str(python), "--extra-index-url", CPU_INDEX,
              str(ROOT / "requirements.lock")])
    _command([uv, "pip", "install", "--python", str(python), "--no-deps", "--editable", str(ROOT)])
    acquire_sources()
    if not skip_native_build:
        missing = [name for name in ("cmake", "ninja", "gcc") if shutil.which(name) is None]
        if missing:
            raise RuntimeError("Native compiler tools missing:" + ", ".join(missing))
        _command([str(python), "-m", "scripts.build_native"])
    status = environment_status()
    report = ROOT / "artifacts/setup.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-native-build", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(environment_status(), indent=2))
        return
    # Avoid replacing user environment names or globalPython packages.
    os.environ.setdefault("UV_NO_PROGRESS", "1")
    print(json.dumps(setup(args.skip_native_build), indent=2))


if __name__ == "__main__":
    main()
