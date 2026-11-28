"""Torch-free pixel exploration environment over the native mGBA shim.

The observation is a stack of grayscale frames resampled from the 240x160 GBA framebuffer.
RAM is read *only* to fill ``info`` with measurement data (map, x, y, battle flag); nothing
derived from RAM ever enters the observation.  This module deliberately does not import torch
so that rollout worker processes stay light and can never call ``backward()``.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ares_u.gal.games import GAMES as GAL_GAMES
from core.bridge.ffi_mgba import (
    KEY_A, KEY_B, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_SELECT, KEY_START, KEY_UP,
)

ROOT = Path(__file__).resolve().parents[2]
GBA_HEIGHT, GBA_WIDTH = 160, 240

# Universal action space: NOOP + 8 buttons shared by GB, GBA and (the face buttons of) NDS.
ACTION_NAMES: tuple[str, ...] = ("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT")
ACTION_KEYS: tuple[int, ...] = (0, KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_A, KEY_B, KEY_START, KEY_SELECT)
DIRECTION_ACTIONS: frozenset[int] = frozenset({1, 2, 3, 4})
NUM_ACTIONS = len(ACTION_NAMES)
CELL_SAMPLE_FRAMES = 4

DEFAULT_ROMS: dict[str, Path] = {
    "firered": ROOT.parent / "Pokemon - FireRed Version (USA).gba",
    "emerald": ROOT.parent / "Pokemon - Emerald Version (USA, Europe).gba",
}
DEFAULT_DLL = ROOT / "native" / "build" / "ares_mgba.dll"
NAV_ARTIFACTS = ROOT / "artifacts" / "ares_u" / "nav"
# FireRed/Emerald keep the original native bridge (E2); every other game of the ARES-U Game
# Abstraction Layer (ares_u.gal: GB, GBC, NDS) uses its multi-console backends.
NATIVE_GAMES: tuple[str, ...] = ("firered", "emerald")
GAMES: tuple[str, ...] = NATIVE_GAMES + tuple(key for key in GAL_GAMES if key not in NATIVE_GAMES)
for _key, _spec in GAL_GAMES.items():
    DEFAULT_ROMS.setdefault(_key, _spec.rom)

Cell = tuple[int, int, int, int]  # (map_group, map_num, x, y)
UInt8Array = NDArray[np.uint8]
Float32Array = NDArray[np.float32]


def default_state_path(game: str) -> Path:
    return NAV_ARTIFACTS / "states" / f"{game}_start.state"


class BridgeLike(Protocol):
    """The subset of ``NativeLibMGBABridge`` the environment relies on."""

    def step_frame(self) -> None: ...
    def set_hardware_keys(self, key_mask: int) -> None: ...
    def framebuffer(self) -> NDArray[np.uint32]: ...
    def load_state(self, state: bytes) -> None: ...
    def get_player_coords_and_map(self) -> dict[str, int]: ...
    def is_in_battle(self) -> tuple[bool, bool]: ...
    def close(self) -> None: ...


BridgeFactory = Callable[["NavConfig"], BridgeLike]


@dataclass(frozen=True)
class NavConfig:
    game: str
    rom_path: str = ""
    dll_path: str = str(DEFAULT_DLL)
    state_path: str = ""
    action_repeat: int = 24
    button_frames: int = 6
    frame_stack: int = 4
    obs_height: int = 84
    obs_width: int = 84
    episode_steps: int = 1024
    noop_max: int = 30
    # Optional pool of start savestates; ``reset`` draws one uniformly (used for diverse AE frames).
    start_states: tuple[str, ...] = ()
    detect_transitions: bool = False  # pixel-only "screen went blank" detector (place novelty)

    def __post_init__(self) -> None:
        if self.game not in GAMES:
            raise ValueError(f"Unknown game {self.game!r}; expected one of {GAMES}")
        if self.action_repeat < 1 or not 1 <= self.button_frames <= self.action_repeat:
            raise ValueError("Need action_repeat >= 1 and 1 <= button_frames <= action_repeat")
        if self.frame_stack < 1 or self.episode_steps < 1 or self.noop_max < 0:
            raise ValueError("frame_stack and episode_steps must be positive, noop_max >= 0")
        if self.obs_height < 8 or self.obs_width < 8:
            raise ValueError("Observation must be at least 8x8")

    def resolved_rom(self) -> Path:
        return Path(self.rom_path) if self.rom_path else DEFAULT_ROMS[self.game]

    def resolved_state(self) -> Path:
        return Path(self.state_path) if self.state_path else default_state_path(self.game)

    def resolved_states(self) -> list[Path]:
        return [Path(path) for path in self.start_states] if self.start_states else [self.resolved_state()]

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        return (self.frame_stack, self.obs_height, self.obs_width)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def area_matrix(n_in: int, n_out: int) -> Float32Array:
    """Row-stochastic matrix that area-averages ``n_in`` samples into ``n_out`` bins."""
    if n_in < 1 or n_out < 1:
        raise ValueError("Sizes must be positive")
    weights = np.zeros((n_out, n_in), dtype=np.float64)
    scale = n_in / n_out
    for i in range(n_out):
        lo, hi = i * scale, (i + 1) * scale
        for j in range(int(np.floor(lo)), min(n_in, int(np.ceil(hi)))):
            weights[i, j] = max(0.0, min(hi, j + 1) - max(lo, j))
        weights[i] /= weights[i].sum()
    return weights.astype(np.float32)


class FramePreprocessor:
    """Convert a packed little-endian XBGR framebuffer into an area-resampled grayscale image.

    Any console resolution is accepted (GB/GBC 160x144, GBA 240x160, NDS screen 256x192); the
    resampling matrices are built once per input shape, so every game yields the same obs shape.
    """

    def __init__(self, height: int, width: int) -> None:
        self.height, self.width = height, width
        self._matrices: dict[tuple[int, int], tuple[Float32Array, Float32Array]] = {}
        self._matrices_for((GBA_HEIGHT, GBA_WIDTH))

    def _matrices_for(self, shape: tuple[int, int]) -> tuple[Float32Array, Float32Array]:
        if shape not in self._matrices:
            rows = area_matrix(shape[0], self.height)
            cols_t = np.ascontiguousarray(area_matrix(shape[1], self.width).T)
            self._matrices[shape] = (rows, cols_t)
        return self._matrices[shape]

    def __call__(self, framebuffer: NDArray[np.uint32]) -> UInt8Array:
        if framebuffer.ndim != 2 or min(framebuffer.shape) < 8:
            raise ValueError(f"Framebuffer shape {framebuffer.shape} is not a 2-D screen")
        rows, cols_t = self._matrices_for((int(framebuffer.shape[0]), int(framebuffer.shape[1])))
        packed = framebuffer.astype(np.uint32, copy=False)
        red = (packed & 0xFF).astype(np.float32)
        green = ((packed >> 8) & 0xFF).astype(np.float32)
        blue = ((packed >> 16) & 0xFF).astype(np.float32)
        gray = 0.299 * red + 0.587 * green + 0.114 * blue
        resized = rows @ gray @ cols_t
        image: UInt8Array = np.clip(np.rint(resized), 0, 255).astype(np.uint8)
        return image


@dataclass
class CellTracker:
    """Unique (map_group, map_num, x, y) cells; RAM-derived and therefore metric-only."""

    cells: set[Cell] = field(default_factory=set)

    def visit(self, cell: Cell) -> bool:
        if cell in self.cells:
            return False
        self.cells.add(cell)
        return True

    def __len__(self) -> int:
        return len(self.cells)

    def maps(self) -> set[tuple[int, int]]:
        return {(group, number) for group, number, _, _ in self.cells}


def stable_cells(previous: Cell, samples: Sequence[Cell]) -> tuple[list[Cell], int]:
    """Cells confirmed by two consecutive RAM samples, and the number of unconfirmed samples.

    While a warp or a SaveBlock relocation is in progress a single read can return a garbage
    (map, x, y).  A walking player stays on a tile for 16 frames, i.e. for at least three samples
    taken every ``CELL_SAMPLE_FRAMES`` frames, so requiring two equal consecutive samples drops the
    one-off garbage without missing real tiles.  Metric-only.
    """
    confirmed: list[Cell] = []
    unconfirmed = 0
    last = previous
    for index, cell in enumerate(samples):
        if cell == last:
            if cell[0] >= 0 and cell not in confirmed:
                confirmed.append(cell)
        elif index + 1 < len(samples) and cell != samples[index + 1]:
            unconfirmed += 1  # differs from both neighbours: a one-off read, never counted
        last = cell
    return confirmed, unconfirmed


SCREEN_GRID = 7      # 84x84 observation -> 7x7 blocks of 12x12 pixels
SCREEN_LEVELS = 4    # gray levels per block after averaging


def screen_key(frame: UInt8Array, grid: int = SCREEN_GRID, levels: int = SCREEN_LEVELS) -> bytes:
    """Coarse signature of a screen for the universal pixel-coverage metric.

    The observation frame is block-averaged to ``grid`` x ``grid`` and quantised to ``levels``
    gray levels, so tile animations and single-sprite changes rarely create a new key while a new
    camera position or a new room does. Used for *measurement* on games without a RAM adapter
    (it is validated against the RAM cells on the games that have one); never a reward.
    """
    height, width = frame.shape
    bh, bw = height // grid, width // grid
    blocks = frame[: bh * grid, : bw * grid].reshape(grid, bh, grid, bw).mean(axis=(1, 3))
    return bytes(np.minimum(levels - 1, (blocks * levels / 256.0).astype(np.int64)).astype(np.uint8).ravel())


def native_bridge_factory(config: NavConfig) -> BridgeLike:
    if config.game not in NATIVE_GAMES:
        from dataclasses import replace as _replace

        from ares_u.gal.backends import make_bridge
        spec = GAL_GAMES[config.game]
        if config.rom_path:
            spec = _replace(spec, rom_name=str(Path(config.rom_path)))
        return make_bridge(spec)
    from core.bridge.ffi_mgba import NativeLibMGBABridge
    return NativeLibMGBABridge(config.dll_path, config.resolved_rom(), is_emerald=config.game == "emerald")


BLANK_STD = 8.0  # grey-level std below which a frame counts as a fade (black or white)


def frame_is_blank(framebuffer: NDArray[np.uint32]) -> bool:
    """Near-uniform frame: the fade-out every Pokemon generation shows when the player changes place.

    Pixels only (a strided grey version of the framebuffer); the RAM is never read here.
    """
    sub = np.asarray(framebuffer)[::6, ::6].astype(np.uint32)
    gray = 0.299 * (sub & 0xFF) + 0.587 * ((sub >> 8) & 0xFF) + 0.114 * ((sub >> 16) & 0xFF)
    return float(gray.std()) < BLANK_STD


class NavEnv:
    """Gym-like single environment: ``reset() -> obs`` and ``step(a) -> (obs, r, term, trunc, info)``.

    The extrinsic reward is always 0.0: the task is reward-free exploration and the learner adds
    its own pixel-based intrinsic reward.  Episodes have a fixed length (truncation only).
    """

    def __init__(self, config: NavConfig, seed: int = 0,
                 bridge_factory: BridgeFactory | None = None,
                 start_state: bytes | None = None) -> None:
        self.config = config
        self._start_states = ([start_state] if start_state is not None
                              else [path.read_bytes() for path in config.resolved_states()])
        self.bridge: BridgeLike = (bridge_factory or native_bridge_factory)(config)
        self._rng = np.random.default_rng(seed)
        self._pre = FramePreprocessor(config.obs_height, config.obs_width)
        self._stack: UInt8Array = np.zeros(config.obs_shape, dtype=np.uint8)
        self._last_cell: Cell = (-1, -1, -1, -1)
        self._prev_sample: Cell = (-1, -1, -1, -1)
        self.cells = CellTracker()
        self.screens: set[bytes] = set()
        self.steps = 0
        self.episodes = 0
        self.start_index = 0
        self._held = 0
        self._ended_blank = False

    def _frame(self) -> UInt8Array:
        return self._pre(self.bridge.framebuffer())

    def _read_cell(self) -> Cell:
        try:
            pos = self.bridge.get_player_coords_and_map()
            cell = (int(pos["map_group"]), int(pos["map_num"]), int(pos["x"]), int(pos["y"]))
        except (ValueError, RuntimeError, KeyError):
            return self._last_cell  # transient invalid SaveBlock pointer during a warp
        self._last_cell = cell
        return cell

    def _in_battle(self) -> bool:
        try:
            return bool(self.bridge.is_in_battle()[0])
        except (ValueError, RuntimeError):
            return False

    def _press(self, keys: int) -> None:
        if keys != self._held:
            self.bridge.set_hardware_keys(keys)
            self._held = keys

    def reset(self) -> UInt8Array:
        count = len(self._start_states)
        self.start_index = int(self._rng.integers(0, count)) if count > 1 else 0
        self.bridge.load_state(self._start_states[self.start_index])
        self._held = -1
        self._press(0)
        for _ in range(int(self._rng.integers(0, self.config.noop_max + 1))):
            self.bridge.step_frame()
        frame = self._frame()
        self._stack[:] = frame[None]
        self.cells = CellTracker()
        self.screens = {screen_key(frame)}
        self._last_cell = (-1, -1, -1, -1)
        cell = self._read_cell()
        if cell[0] >= 0:  # the start state is a settled position, so a single read is trusted
            self.cells.visit(cell)
        self._prev_sample = cell
        self._ended_blank = False
        self.steps = 0
        self.episodes += 1
        return self._stack.copy()

    def step(self, action: int) -> tuple[UInt8Array, float, bool, bool, dict[str, Any]]:
        if not 0 <= action < NUM_ACTIONS:
            raise ValueError(f"Action {action} outside [0, {NUM_ACTIONS})")
        keys = ACTION_KEYS[action]
        held_frames = self.config.action_repeat if action in DIRECTION_ACTIONS or action == 0 \
            else self.config.button_frames
        samples: list[Cell] = []
        blank_seen = self._ended_blank  # a one-step fade (Gen 2) ends blank and resolves in the next step
        for frame_index in range(self.config.action_repeat):
            if frame_index == 0:
                self._press(keys)
            elif frame_index == held_frames:
                self._press(0)
            self.bridge.step_frame()
            # Walking takes 16 frames per tile, so sampling every 4 frames sees every tile crossed.
            if (frame_index + 1) % CELL_SAMPLE_FRAMES == 0 and frame_index + 1 < self.config.action_repeat:
                samples.append(self._read_cell())
                if self.config.detect_transitions and not blank_seen:
                    blank_seen = frame_is_blank(self.bridge.framebuffer())
        raw = self.bridge.framebuffer()
        frame = self._pre(raw)
        final_blank = bool(self.config.detect_transitions and frame_is_blank(raw))
        transition = bool(self.config.detect_transitions and blank_seen and not final_blank)
        self._ended_blank = final_blank
        self._stack[:-1] = self._stack[1:]
        self._stack[-1] = frame
        self.screens.add(screen_key(frame))
        cell = self._read_cell()
        samples.append(cell)
        step_cells, unconfirmed = stable_cells(self._prev_sample, samples)
        self._prev_sample = cell
        new_cell = False
        for visited in step_cells:
            new_cell = self.cells.visit(visited) or new_cell
        self.steps += 1
        truncated = self.steps >= self.config.episode_steps
        info: dict[str, Any] = {
            "cell": cell, "step_cells": step_cells, "new_cell": new_cell, "episode_cells": len(self.cells),
            "episode_maps": len(self.cells.maps()), "episode_screens": len(self.screens),
            "in_battle": self._in_battle(), "t": self.steps,
            "unconfirmed_samples": unconfirmed, "transition": transition,
        }
        return self._stack.copy(), 0.0, False, truncated, info

    def close(self) -> None:
        self.bridge.close()


# --------------------------------------------------------------------------- vectorisation

def _worker(conn: Connection, config: NavConfig, seed: int) -> None:
    env = NavEnv(config, seed=seed)
    try:
        while True:
            command, payload = conn.recv()
            if command == "reset":
                conn.send(env.reset())
            elif command == "step":
                obs, reward, terminated, truncated, info = env.step(int(payload))
                if terminated or truncated:
                    info["final_episode_cells"] = len(env.cells)
                    info["final_episode_maps"] = len(env.cells.maps())
                    info["final_episode_screens"] = len(env.screens)
                    obs = env.reset()
                conn.send((obs, reward, terminated or truncated, info))
            elif command == "close":
                break
            else:
                raise ValueError(f"Unknown worker command {command!r}")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        env.close()
        conn.close()


_SINGLE_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


class SubprocVecEnv:
    """One emulator per process.  Workers are torch-free and auto-reset at episode end."""

    def __init__(self, configs: Sequence[NavConfig], seeds: Sequence[int]) -> None:
        if len(configs) != len(seeds) or not configs:
            raise ValueError("Need one seed per config and at least one env")
        context = mp.get_context("spawn")
        saved = {name: os.environ.get(name) for name in _SINGLE_THREAD_VARS}
        for name in _SINGLE_THREAD_VARS:
            os.environ[name] = "1"
        self._conns: list[Any] = []  # Connection (POSIX) or PipeConnection (Windows)
        self._procs: list[mp.process.BaseProcess] = []
        try:
            for config, seed in zip(configs, seeds):
                parent, child = context.Pipe()
                proc = context.Process(target=_worker, args=(child, config, seed), daemon=True)
                proc.start()
                child.close()
                self._conns.append(parent)
                self._procs.append(proc)
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        self.num_envs = len(configs)
        self.obs_shape = configs[0].obs_shape
        self._closed = False

    def reset(self) -> UInt8Array:
        for conn in self._conns:
            conn.send(("reset", None))
        return np.stack([conn.recv() for conn in self._conns])

    def step(self, actions: NDArray[np.int64] | Sequence[int]
             ) -> tuple[UInt8Array, NDArray[np.bool_], list[dict[str, Any]]]:
        for conn, action in zip(self._conns, actions):
            conn.send(("step", int(action)))
        results = [conn.recv() for conn in self._conns]
        obs = np.stack([r[0] for r in results])
        dones = np.array([bool(r[2]) for r in results], dtype=np.bool_)
        return obs, dones, [dict(r[3]) for r in results]

    def close(self) -> None:
        if self._closed:
            return
        for conn in self._conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
        self._closed = True

    def __enter__(self) -> SubprocVecEnv:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def make_configs(config: NavConfig, num_envs: int) -> list[NavConfig]:
    return [config for _ in range(num_envs)]


# --------------------------------------------------------------------------- throughput

def bench_raw_frames(config: NavConfig, frames: int = 3000) -> dict[str, float]:
    """Frames/s of bare ``step_frame`` from the start state, plus preprocess cost."""
    bridge = native_bridge_factory(config)
    try:
        bridge.load_state(config.resolved_state().read_bytes())
        start = time.perf_counter()
        for _ in range(frames):
            bridge.step_frame()
        fps = frames / (time.perf_counter() - start)
        pre = FramePreprocessor(config.obs_height, config.obs_width)
        start = time.perf_counter()
        for _ in range(200):
            pre(bridge.framebuffer())
        pre_ms = (time.perf_counter() - start) / 200 * 1e3
        start = time.perf_counter()
        for _ in range(200):
            bridge.get_player_coords_and_map()
        ram_ms = (time.perf_counter() - start) / 200 * 1e3
    finally:
        bridge.close()
    return {"frames_per_s": fps, "preprocess_ms": pre_ms, "ram_read_ms": ram_ms}


def bench_env_steps(config: NavConfig, num_envs: int, steps: int, seed: int = 0) -> dict[str, float]:
    """Env steps/s (uniform random actions) for ``num_envs`` worker processes."""
    rng = np.random.default_rng(seed)
    with SubprocVecEnv(make_configs(config, num_envs), [seed + i for i in range(num_envs)]) as venv:
        venv.reset()
        start = time.perf_counter()
        for _ in range(steps):
            venv.step(rng.integers(0, NUM_ACTIONS, size=num_envs))
        elapsed = time.perf_counter() - start
    total = steps * num_envs
    return {"num_envs": float(num_envs), "env_steps_per_s": total / elapsed,
            "frames_per_s": total * config.action_repeat / elapsed,
            "per_env_steps_per_s": steps / elapsed}
