"""Build the single compressed deliverable required by the IPD434 guidelines.

Contents: every file tracked by git (code, tests, data, docs, report sources), the generated report
assets and PDFs, the E1/E2 result summaries and figures, and the E1 final checkpoints (small). It
never includes ROMs, savestates or emulator states (ROM-derived), the virtual environment, native
build outputs or the large E2 checkpoints (reproducible with the documented commands).

    python -m scripts.ares_u_package            -> entrega/ARES-U_IPD434_Moraga.zip
"""
from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".gba", ".gb", ".gbc", ".nds", ".sav", ".state", ".ss0", ".ss1"}
RESULT_GLOBS = (
    "artifacts/ares_u/battle/final/*.json", "artifacts/ares_u/battle/final/*.md",
    "artifacts/ares_u/battle/final/figures/*.png", "artifacts/ares_u/battle/final/*/seed*/config.json",
    "artifacts/ares_u/battle/final/*/seed*/metrics.jsonl", "artifacts/ares_u/battle/final/*/seed*/eval.json",
    "artifacts/ares_u/battle/final/*/seed*/final.pt", "artifacts/ares_u/battle/headroom/*.json",
    "artifacts/ares_u/battle/final.log",
    "artifacts/ares_u/nav/final/*.json", "artifacts/ares_u/nav/final/*.md", "artifacts/ares_u/nav/final/figures/*.png",
    "artifacts/ares_u/nav/final/seed*/*/metrics.jsonl", "artifacts/ares_u/nav/final/seed*/*/config.json",
    "artifacts/ares_u/nav/final/seed*/*.jsonl", "artifacts/ares_u/nav/final/seed*/eval*.json",
    "artifacts/ares_u/nav/final.log", "artifacts/ares_u/nav/states/*.json",
    "informe/generated/*.tex", "informe/figs/*.png", "informe/*.pdf",
)


def tracked_files(root: Path) -> list[Path]:
    output = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True).stdout
    return [root / name for name in output.decode("utf-8").split("\0") if name]


def allowed(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() not in FORBIDDEN_SUFFIXES


def collect(root: Path) -> list[Path]:
    files = {path for path in tracked_files(root) if allowed(path)}
    for pattern in RESULT_GLOBS:
        files.update(path for path in root.glob(pattern) if allowed(path))
    return sorted(files)


def build(root: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    files = collect(root)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, Path("ARES-U") / path.relative_to(root))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "entrega" / "ARES-U_IPD434_Moraga.zip")
    args = parser.parse_args(argv)
    path = build(ROOT, args.out)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    leaked = [name for name in names if Path(name).suffix.lower() in FORBIDDEN_SUFFIXES]
    if leaked:
        raise SystemExit(f"refusing: ROM-derived files in the archive: {leaked[:5]}")
    print(f"{path} ({path.stat().st_size / 1e6:.1f} MB, {len(names)} files)")


if __name__ == "__main__":
    main()
