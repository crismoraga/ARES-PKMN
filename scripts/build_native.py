"""Reproducibly build the Windows/Linux mGBA C ABI adapter."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MGBA_COMMIT = "26b7884bc25a5933960f3cdcd98bac1ae14d42e2"


def build_native(jobs: int = 8) -> Path:
    """Acquire the pinned upstream source and compile a native ABI adapter."""
    if jobs < 1:
        raise ValueError("jobs must be positive")
    vendor = ROOT / "native" / "vendor" / "mgba"
    if not vendor.exists():
        vendor.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--branch", "0.10.5",
                        "https://github.com/mgba-emu/mgba.git", str(vendor)], check=True)
    actual = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    if actual != MGBA_COMMIT:
        raise RuntimeError(f"mGBA source mismatch: {actual}; expected {MGBA_COMMIT}")
    for executable in ("cmake", "ninja", "gcc"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Required build tool missing: {executable}")
    build = ROOT / "native" / "build"
    subprocess.run(["cmake", "-S", str(ROOT / "native"), "-B", str(build), "-G", "Ninja",
                    "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_C_COMPILER={shutil.which('gcc')}"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", str(jobs)], check=True)
    candidates = tuple(build.glob("ares_mgba.*"))
    binaries = [p for p in candidates if p.suffix in {".dll", ".so", ".dylib"}]
    if len(binaries) != 1:
        raise RuntimeError(f"Expected one adapter binary, got {binaries}")
    return binaries[0]


def main() -> None:
    """Compile and print the artifact path; build errors retain nonzero status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    print(build_native(args.jobs))


if __name__ == "__main__":
    main()
