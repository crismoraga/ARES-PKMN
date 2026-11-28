"""Derivation and provenance of the fixed E2 start savestates.

Both start states are early-game, post-intro, with the player in control inside the
professor's lab right after receiving the starter, so the two games begin from the
same narrative situation:

* FireRed: campaign checkpoint at frame 12772 (milestone ``RIVAL_LAB``; Oak's lab,
  map 4.3, Bulbasaur in party).  The player is already in control, so it is used as is.
* Emerald: campaign checkpoint at frame 14649 (Birch's lab, map 1.4, Mudkip just received,
  dialogue still running).  The dialogue is advanced deterministically with A pulses
  (4 frames held, 26 released) until a directional probe shows the player can walk.

Source checkpoints are never modified; the derived states are written under
``artifacts/ares_u/nav/states`` together with a provenance JSON (sha256 of source and result,
number of A pulses, final position).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.bridge.ffi_mgba import KEY_A, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP, NativeLibMGBABridge
from ares_u.nav.env import DEFAULT_DLL, DEFAULT_ROMS, NAV_ARTIFACTS, ROOT, default_state_path

CAMPAIGNS = ROOT / "artifacts" / "campaigns" / "native"


@dataclass(frozen=True)
class StateSource:
    game: str
    checkpoint: Path
    description: str


SOURCES: dict[str, StateSource] = {
    "firered": StateSource(
        "firered",
        CAMPAIGNS / "firered" / "frame-000000012772-1792069576160675700" / "emulator.state",
        "FireRed campaign checkpoint, milestone RIVAL_LAB (Oak's lab, Bulbasaur in party)"),
    "emerald": StateSource(
        "emerald",
        CAMPAIGNS / "emerald" / "frame-000000014649-1792069573261031300" / "emulator.state",
        "Emerald campaign checkpoint after MEET_RIVAL (Birch's lab, Mudkip being received)"),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def player_can_walk(bridge: NativeLibMGBABridge, hold_frames: int = 40, settle_frames: int = 20) -> bool:
    """Probe each direction from a snapshot and restore it; True if any probe changes (x, y, map)."""
    snapshot = bridge.save_state()
    try:
        before = bridge.get_player_coords_and_map()
    except ValueError:
        return False
    try:
        for key in (KEY_DOWN, KEY_UP, KEY_LEFT, KEY_RIGHT):
            bridge.load_state(snapshot)
            bridge.set_hardware_keys(key)
            for _ in range(hold_frames):
                bridge.step_frame()
            bridge.set_hardware_keys(0)
            for _ in range(settle_frames):
                bridge.step_frame()
            try:
                after = bridge.get_player_coords_and_map()
            except ValueError:
                continue
            if any(after[k] != before[k] for k in ("x", "y", "map_group", "map_num")):
                return True
        return False
    finally:
        bridge.load_state(snapshot)


def derive_start_state(game: str, out_path: Path | None = None, dll: Path = DEFAULT_DLL,
                       rom: Path | None = None, max_pulses: int = 400) -> dict[str, Any]:
    source = SOURCES[game]
    raw = source.checkpoint.read_bytes()
    out = out_path or default_state_path(game)
    bridge = NativeLibMGBABridge(dll, rom or DEFAULT_ROMS[game], is_emerald=game == "emerald")
    try:
        bridge.load_state(raw)
        pulses = 0
        while not player_can_walk(bridge):
            if pulses >= max_pulses:
                raise RuntimeError(f"{game}: player never gained control after {max_pulses} A pulses")
            bridge.inject_key_pulse(KEY_A, 4)
            for _ in range(26):
                bridge.step_frame()
            pulses += 1
        state = bridge.save_state()
        position = bridge.get_player_coords_and_map()
        frame = bridge.frame_count
    finally:
        bridge.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(state)
    provenance: dict[str, Any] = {
        "game": game, "source_checkpoint": str(source.checkpoint.relative_to(ROOT)),
        "source_description": source.description, "source_sha256": sha256(raw),
        "a_pulses": pulses, "pulse": "A held 4 frames then 26 released",
        "control_probe": "hold each direction 40 frames, settle 20, restore", "frame": frame,
        "position": position, "state_path": str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out),
        "state_sha256": sha256(state),
    }
    out.with_suffix(".json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def prepare_all(force: bool = False) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for game in SOURCES:
        path = default_state_path(game)
        meta = path.with_suffix(".json")
        if path.exists() and meta.exists() and not force:
            results[game] = json.loads(meta.read_text(encoding="utf-8"))
        else:
            results[game] = derive_start_state(game, path)
    return results


# --------------------------------------------------------------------------- autoencoder state pool

# FireRed campaign checkpoints used ONLY as start points of the sticky-random frame collection for
# the autoencoder (diverse source-game visuals: towns, routes, caves, gyms, wild and trainer battles).
# Positions were read from RAM once when the pool was chosen (2026-11-24) and are re-read into the
# provenance JSON; RL training on FireRed still starts from ``firered_start.state`` only.
AE_POOL_FIRERED: tuple[tuple[str, str], ...] = (
    ("frame-000000015410-1792069595015411400", "Viridian City interior (map 5.3), OAK_PARCEL_PICKUP"),
    ("frame-000000096434-1792069924156503000", "Pewter City interior (map 6.2), after GYM_1_BROCK"),
    ("frame-000000143910-1794889811870181300", "Mt. Moon 1F (map 1.1), wild battle"),
    ("frame-000000197610-1794889296495755100", "Mt. Moon B2F (map 1.3), trainer battle"),
    ("frame-000000240511-1792117383271844000", "Mt. Moon B2F (map 1.3), overworld"),
    ("frame-000000394170-1792119687125303100", "Route 3 (map 3.21), wild battle"),
    ("frame-000000429210-1792118639517039200", "Route 4 (map 3.22), overworld"),
    ("frame-000000430410-1792118660652948900", "Cerulean City (map 3.3), overworld"),
    ("frame-000002669429-1792084079942812400", "Cerulean City interior (map 7.3), CERULEAN_RIVAL"),
    ("frame-000002697297-1792084504230850300", "Bill's house (map 30.0), BILL_RESCUE"),
    ("frame-000002704946-1792084543772879000", "Cerulean Gym (map 7.5), GYM_2_MISTY"),
    ("frame-000002727961-1792084747295700000", "S.S. Anne (map 1.11), SS_ANNE_CUT"),
    ("frame-000002731543-1792084779088352800", "Vermilion Gym (map 9.6), SURGE_SWITCHES"),
    ("frame-000009269010-1792061093228959900", "Route 2 (map 3.20), OAK_PARCEL"),
)


def ae_pool_dir() -> Path:
    return NAV_ARTIFACTS / "states" / "firered_ae_pool"


def prepare_ae_pool(force: bool = False, dll: Path = DEFAULT_DLL, rom: Path | None = None) -> dict[str, Any]:
    """Copy the FireRed AE start-state pool next to the nav artifacts, with provenance (sha256, position).

    The copies make the protocol independent of the campaign directories; the RL start state
    ``firered_start.state`` is always part of the pool (first entry)."""
    out_dir = ae_pool_dir()
    manifest_path = out_dir / "pool.json"
    if manifest_path.exists() and not force:
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    start = default_state_path("firered")
    entries.append({"state_path": str(start.relative_to(ROOT)), "source": "firered_start.state (RL start)",
                    "description": SOURCES["firered"].description, "sha256": sha256(start.read_bytes())})
    bridge = NativeLibMGBABridge(dll, rom or DEFAULT_ROMS["firered"], is_emerald=False)
    try:
        for index, (directory, description) in enumerate(AE_POOL_FIRERED, start=1):
            source = CAMPAIGNS / "firered" / directory / "emulator.state"
            raw = source.read_bytes()
            bridge.load_state(raw)
            position = bridge.get_player_coords_and_map()
            in_battle = bridge.is_in_battle()
            target = out_dir / f"{index:02d}_{directory.split('-')[1]}.state"
            target.write_bytes(raw)
            entries.append({"state_path": str(target.relative_to(ROOT)), "source": str(source.relative_to(ROOT)),
                            "description": description, "sha256": sha256(raw), "position": position,
                            "in_battle": list(in_battle)})
    finally:
        bridge.close()
    manifest = {"game": "firered", "purpose": "start states for autoencoder frame collection only",
                "states": entries}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def ae_pool_paths() -> list[Path]:
    manifest = prepare_ae_pool()
    return [ROOT / entry["state_path"] for entry in manifest["states"]]
