"""Audit installed resources and run the four required acceptance commands."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def inspect_installation() -> dict[str, object]:
    """Check the actual lock, source commits, ROM hashes and loadable adapter."""
    packages: dict[str, dict[str, object]] = {}
    for line in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([\w.-]+)==([^\s;]+)", line.strip())
        if match:
            name, expected = match.groups()
            try:
                installed = metadata.version(name)
            except metadata.PackageNotFoundError:
                installed = None
            packages[name] = {"expected": expected, "installed": installed, "matches": installed == expected}
    manifest: dict[str, Any] = json.loads((ROOT / "data/source_manifest.json").read_text(encoding="utf-8"))
    sources: dict[str, dict[str, object]] = {}
    git = shutil.which("git")
    for game, directory in (("mgba", "mgba"), ("firered", "pokefirered"), ("emerald", "pokeemerald")):
        source = ROOT / "native/vendor" / directory
        actual: str | None = None
        if git and (source / ".git").is_dir():
            actual = subprocess.check_output([git, "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        expected = str(manifest[game]["source_commit"])
        sources[game] = {"expected": expected, "actual": actual, "matches": actual == expected}
    rom_names = {"firered": "Pokemon - FireRed Version (USA).gba",
                 "emerald": "Pokemon - Emerald Version (USA, Europe).gba"}
    roms: dict[str, dict[str, object]] = {}
    for game, name in rom_names.items():
        path = ROOT.parent / name
        digest = hashlib.sha1(path.read_bytes()).hexdigest() if path.is_file() else None
        roms[game] = {"path": str(path), "sha1": digest, "matches": digest == manifest[game]["rom_sha1"]}
    dll = ROOT / "native/build/ares_mgba.dll"
    dll_loaded = False
    dll_error = ""
    if dll.is_file():
        try:
            handle = ctypes.CDLL(str(dll))
            getattr(handle, "ares_create")
            getattr(handle, "ares_save_state")
            getattr(handle, "ares_save_battery")
            dll_loaded = True
        except (OSError, AttributeError) as error:
            dll_error = str(error)
    dependencies_ok = bool(packages) and all(row["matches"] for row in packages.values())
    result: dict[str, object] = {"python": sys.executable, "python_version": sys.version.split()[0],
        "python312": sys.version_info[:2] == (3, 12), "packages": packages,
        "sources": sources, "roms": roms, "native_dll": str(dll),
        "native_dll_loadable": dll_loaded, "native_dll_error": dll_error,
        "installation_ready": sys.version_info[:2] == (3, 12) and dependencies_ok and dll_loaded
                              and all(row["matches"] for row in sources.values())
                              and all(row["matches"] for row in roms.values()),
        "native_campaign_certified": False}
    return result


def run_verification() -> dict[str, object]:
    """Execute each gate once, saving exact stdout/stderr and its return code."""
    inspection = inspect_installation()
    directory = ROOT / "artifacts/verification" / str(time.time_ns())
    directory.mkdir(parents=True, exist_ok=True)
    gates = (
        ("pytest", ["-m", "pytest", "-v", "--tb=short"]),
        ("mypy", ["-m", "mypy", "--strict", "core", "scripts", "tests"]),
        ("training", ["-m", "scripts.train_neural_modules"]),
        ("synthetic_demo", ["-m", "scripts.run_autonomous_verification_demo"]),
    )
    results: dict[str, dict[str, object]] = {}
    for name, arguments in gates:
        log = directory / f"{name}.log"
        print(f"Running{name}; evidence:{log}", flush=True)
        started = time.monotonic()
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run([sys.executable, *arguments], cwd=ROOT,
                                     stdout=stream, stderr=subprocess.STDOUT, check=False)
        results[name] = {"exit_code": process.returncode, "seconds": time.monotonic() - started,
                         "log": str(log), "passed": process.returncode == 0}
        print(json.dumps({name: results[name]}), flush=True)
    report = {"installation": inspection, "gates": results,
              "passed": bool(inspection["installation_ready"]) and all(row["passed"] for row in results.values()),
              "native_campaign_certified": False,
              "scope": "Native Champion/Hall-of-Fame clearance is verified by run_campaign separately"}
    (directory / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ROOT / "artifacts/verification/latest.json").write_text(json.dumps({"directory": str(directory)}), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Inspect resources without retraining or rerunning tests")
    args = parser.parse_args()
    result = inspect_installation() if args.quick else run_verification()
    print(json.dumps(result, indent=2))
    if not result.get("installation_ready", result.get("passed", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
