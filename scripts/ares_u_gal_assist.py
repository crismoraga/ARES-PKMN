"""Assisted start-state construction: replay an explicit key sequence from a saved intro state, then
continue with the generic intro driver and save a contact sheet.

Used only when the generic driver loops in a tutorial menu (Platinum's "anything else?" menu). The
manual sequence is recorded in the provenance JSON of the resulting states; the agent never uses it.

    python -m scripts.ares_u_gal_assist --game platino --from artifacts/ares_u/gal/intro/platino/f069000.state \
        --keys DOWN DOWN A --frames 30000 --every 1500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ares_u.gal.backends import (GBA_A, GBA_B, GBA_DOWN, GBA_LEFT, GBA_RIGHT, GBA_START, GBA_UP, DeSmuMEBridge,
                                 make_bridge)
from ares_u.gal.games import GAL_ARTIFACTS, game
from scripts.ares_u_gal import NDS_TAPS, contact_sheet, intro_keys, nds_pulse, NDS_TAP_PULSE, save_png

KEYS = {"A": GBA_A, "B": GBA_B, "UP": GBA_UP, "DOWN": GBA_DOWN, "LEFT": GBA_LEFT, "RIGHT": GBA_RIGHT,
        "START": GBA_START}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--game", required=True)
    parser.add_argument("--from", dest="source", type=Path, required=True)
    parser.add_argument("--keys", nargs="+", required=True, choices=sorted(KEYS))
    parser.add_argument("--frames", type=int, default=30000)
    parser.add_argument("--every", type=int, default=1500)
    parser.add_argument("--period", type=int, default=24)
    args = parser.parse_args(argv)
    spec = game(args.game)
    out = GAL_ARTIFACTS / "intro" / f"{spec.key}_assisted"
    out.mkdir(parents=True, exist_ok=True)
    bridge = make_bridge(spec)
    images: list[tuple[str, Path]] = []
    try:
        bridge.load_state(args.source.read_bytes())
        for name in args.keys:  # one 6-frame pulse per key, 30 frames apart
            bridge.set_hardware_keys(KEYS[name])
            for _ in range(6):
                bridge.step_frame()
            bridge.set_hardware_keys(0)
            for _ in range(30):
                bridge.step_frame()
        nds = isinstance(bridge, DeSmuMEBridge)
        for frame in range(args.frames):
            phase, pulse = frame % args.period, frame // args.period
            tapping = nds and nds_pulse(pulse) == NDS_TAP_PULSE
            bridge.set_hardware_keys(0 if tapping else intro_keys(frame, args.period, nds=nds))
            if tapping:
                assert isinstance(bridge, DeSmuMEBridge)
                if phase == 0:
                    bridge.touch(*NDS_TAPS[(pulse // 7) % len(NDS_TAPS)])
                elif phase == 10:
                    bridge.release_touch()
            bridge.step_frame()
            if (frame + 1) % args.every == 0:
                tag = f"a{frame + 1:06d}"
                save_png(bridge.framebuffer(), out / f"{tag}.png")
                (out / f"{tag}.state").write_bytes(bridge.save_state())
                images.append((tag, out / f"{tag}.png"))
    finally:
        bridge.close()
    (out / "assist.json").write_text(json.dumps({"from": str(args.source), "keys": args.keys,
                                                 "then": "generic intro driver"}, indent=2), encoding="utf-8")
    print(contact_sheet(images, out / "sheet.png"))


if __name__ == "__main__":
    main()
