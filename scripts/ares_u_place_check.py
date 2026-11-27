"""Validate the pixel place detector (fade -> new scene) against RAM map changes.

The agent's place-novelty bonus uses pixels only: a step counts as a *transition* when a near-uniform
frame (the fade every generation shows on a door, cave or route change) appears during the step and the
final frame is not blank. Here the RAM map id is read *only to grade the detector*:

* precision = P(the map id changed in the last ``WINDOW`` steps | transition detected)
* recall    = P(a transition is detected within ``WINDOW`` steps | map id changed)

The RAM switches the map id when the fade *starts*; the detector fires when the new scene appears, two
steps later at 24 frames per step, hence the window.

    python -m scripts.ares_u_place_check --games rojo cristal emerald --steps 3000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ares_u.nav.env import NUM_ACTIONS, NavConfig, NavEnv
from ares_u.nav.novelty import PlaceNovelty

ROOT = Path(__file__).resolve().parents[1]
WINDOW = 3


def check(game: str, steps: int, seed: int) -> dict[str, float | int | str]:
    env = NavEnv(NavConfig(game, episode_steps=512, detect_transitions=True), seed=seed)
    rng = np.random.default_rng(seed)
    obs = env.reset()
    places = PlaceNovelty(1, obs.shape[-2], obs.shape[-1])
    places.reset(obs[None, -1])
    previous = env._read_cell()[:2]
    maps_seen = {previous}
    new_maps: list[int] = []  # first visit to a map id in the episode (the target of the bonus)
    bonuses: list[int] = []
    changes: list[int] = []
    detections: list[int] = []
    battles = 0
    for t in range(steps):
        # random walk biased to directions and A (enters doors), like an untrained explorer
        action = int(rng.choice(NUM_ACTIONS, p=_PROBS))
        obs, _, _, truncated, info = env.step(action)
        current = tuple(info["cell"][:2])
        if current != previous and min(current) >= 0 and min(previous) >= 0:
            changes.append(t)
            if current not in maps_seen:
                maps_seen.add(current)
                new_maps.append(t)
        if places.step(obs[None, -1], np.asarray([info["transition"]]), np.asarray([False]))[0] > 0:
            bonuses.append(t)
        if info["transition"]:
            detections.append(t)
            battles += int(info["in_battle"])
        previous = current
        if truncated:
            obs = env.reset()
            places.reset(obs[None, -1])
            previous = env._read_cell()[:2]
            maps_seen = {previous}
    tp = sum(1 for d in detections if any(0 <= d - c <= WINDOW for c in changes))
    found = sum(1 for c in changes if any(0 <= d - c <= WINDOW for d in detections))
    fp, fn = len(detections) - tp, len(changes) - found
    return {"game": game, "steps": steps, "map_changes": len(changes), "detections": len(detections),
            "true_positive": tp, "precision": tp / max(1, tp + fp), "recall": found / max(1, found + fn),
            "detections_in_battle": battles, "new_maps": len(new_maps), "bonuses": len(bonuses),
            "bonus_precision": sum(1 for b in bonuses if any(0 <= b - c <= WINDOW for c in changes)) / max(1, len(bonuses)),
            "bonus_recall_new_maps": sum(1 for c in new_maps if any(0 <= b - c <= WINDOW for b in bonuses)) / max(1, len(new_maps))}


_PROBS = np.full(NUM_ACTIONS, 0.02)
_PROBS[1:5] = 0.2
_PROBS[5] = 1.0 - _PROBS[1:5].sum() - 0.02 * (NUM_ACTIONS - 5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--games", nargs="+", default=["rojo", "cristal", "emerald"])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    results = [check(game, args.steps, args.seed) for game in args.games]
    out = ROOT / "artifacts" / "ares_u" / "place_check.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
