"""Pixel-only episodic novelty for ARES-Nav.

A Go-Explore style "domain-agnostic cell" (Ecoffet et al., 2021): the newest grayscale frame is
area-downscaled to a coarse ``rows x cols`` grid and quantised to ``levels`` intensities; the bytes
of that tiny image are the key.  Because the Gen-3 camera is always centred on the player, the key
changes when the player moves to another tile, so counting keys within an episode is a pixel-only
proxy for "places seen in this episode".  The bonus is 1 the first time a key is seen in the current
episode and 0 afterwards (count-based novelty with the counts reset at every episode start).

Only pixels enter this module: nothing here reads emulator RAM.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ares_u.nav.env import area_matrix

UInt8Array = NDArray[np.uint8]
Float64Array = NDArray[np.float64]


class FrameCellHasher:
    """Map (N, H, W) uint8 frames to coarse quantised byte keys."""

    def __init__(self, height: int, width: int, rows: int = 5, cols: int = 7, levels: int = 8) -> None:
        if not 1 <= rows <= height or not 1 <= cols <= width or not 2 <= levels <= 256:
            raise ValueError("Need 1 <= rows <= H, 1 <= cols <= W and 2 <= levels <= 256")
        self.rows, self.cols, self.levels = rows, cols, levels
        self._rows = area_matrix(height, rows)
        self._cols_t = np.ascontiguousarray(area_matrix(width, cols).T)

    def small(self, frames: UInt8Array) -> UInt8Array:
        """Downscaled and quantised frames, shape (N, rows, cols), values in [0, levels)."""
        if frames.ndim != 3:
            raise ValueError("frames must be (N, H, W)")
        coarse = np.einsum("rh,nhw,wc->nrc", self._rows, frames.astype(np.float32), self._cols_t)
        quantised: UInt8Array = np.clip(np.floor(coarse * (self.levels / 256.0)), 0, self.levels - 1).astype(np.uint8)
        return quantised

    def keys(self, frames: UInt8Array) -> list[bytes]:
        return [image.tobytes() for image in self.small(frames)]


class PlaceNovelty:
    """Bonus 1.0 for arriving, right after a fade, at a scene not reached after a fade this episode.

    The saga changes place (door, cave, route edge) through a fade to black or white; the first frame after
    the fade identifies the new place. Coarser than the screen keys, so walking inside one place does not
    count as a new place. Pixels only.
    """

    BLANK_STD = 8.0

    def __init__(self, num_envs: int, height: int, width: int, rows: int = 4, cols: int = 4, levels: int = 4,
                 mark_departure: bool = False) -> None:
        self.hasher = FrameCellHasher(height, width, rows, cols, levels)
        self.seen: list[set[bytes]] = [set() for _ in range(num_envs)]
        self.last: list[bytes | None] = [None] * num_envs
        # Optional (off in E6/E6b): the scene the agent leaves also counts as seen, so opening a menu and closing it again pays
        # at most once. Offline it did not raise precision and lowered recall (coarse keys collide), so it was not trained.
        self.mark_departure = mark_departure

    def reset(self, frames: UInt8Array) -> None:
        keys = self.hasher.keys(frames)
        self.seen = [{key} for key in keys]
        self.last = list(keys)

    def step(self, frames: UInt8Array, transitions: NDArray[np.bool_], dones: NDArray[np.bool_]) -> Float64Array:
        bonus = np.zeros(len(self.seen), dtype=np.float64)
        keys = self.hasher.keys(frames)
        blanks = frames.reshape(len(keys), -1).astype(np.float32).std(axis=1) < self.BLANK_STD
        for index, key in enumerate(keys):
            if dones[index]:
                self.seen[index] = {key}  # the first frame of the next episode: its starting place
                self.last[index] = key
                continue
            if transitions[index]:
                departure = self.last[index]
                if self.mark_departure and departure is not None:
                    self.seen[index].add(departure)
                if key not in self.seen[index]:
                    self.seen[index].add(key)
                    bonus[index] = 1.0
            if not blanks[index]:
                self.last[index] = key
        return bonus


class EpisodicNovelty:
    """Per-environment episodic sets of frame keys; bonus 1.0 for a key not yet seen this episode."""

    def __init__(self, num_envs: int, height: int, width: int, rows: int = 5, cols: int = 7,
                 levels: int = 8) -> None:
        self.hasher = FrameCellHasher(height, width, rows, cols, levels)
        self.seen: list[set[bytes]] = [set() for _ in range(num_envs)]
        self.finished_counts: list[int] = []

    def reset(self, frames: UInt8Array) -> None:
        """Start a new episode in every environment; the first frames count as seen."""
        for index, key in enumerate(self.hasher.keys(frames)):
            self.seen[index] = {key}

    def step(self, frames: UInt8Array, dones: NDArray[np.bool_]) -> Float64Array:
        """Bonus for arriving at ``frames``.

        For an environment whose episode just ended, ``frames`` is already the first frame of the
        next episode (the vectorised env auto-resets), so that transition gets no bonus and the
        new episode's set starts with that frame.
        """
        bonus = np.zeros(len(self.seen), dtype=np.float64)
        for index, key in enumerate(self.hasher.keys(frames)):
            if dones[index]:
                self.finished_counts.append(len(self.seen[index]))
                self.seen[index] = {key}
            elif key not in self.seen[index]:
                self.seen[index].add(key)
                bonus[index] = 1.0
        return bonus

    def pop_finished(self) -> list[int]:
        """Distinct keys of the episodes finished since the last call (a pixel-side diagnostic)."""
        finished, self.finished_counts = self.finished_counts, []
        return finished
