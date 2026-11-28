"""Tests for ARES-Nav (IPD434 experiment E2): env, preprocessing, models, PPO/RND utilities.

Tests that need the mGBA shim, the ROMs or the derived start states skip when those are absent.
All torch tests run on the CPU (this process never touches DirectML).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch.optim.adam import Adam

from ares_u.nav.env import (
    ACTION_KEYS, ACTION_NAMES, DEFAULT_DLL, DEFAULT_ROMS, DIRECTION_ACTIONS, GBA_HEIGHT, GBA_WIDTH, NUM_ACTIONS,
    CellTracker, FramePreprocessor, NavConfig, NavEnv, SubprocVecEnv, area_matrix, default_state_path, stable_cells,
)
from ares_u.nav.models import (
    VALUE_STREAMS, ActorCritic, ConvAutoencoder, NatureEncoder, RNDModel, SpaceToDepthConv,
    categorical_log_prob_entropy,
)
from ares_u.nav.novelty import EpisodicNovelty, FrameCellHasher
from ares_u.nav.train import (
    ExplorationLog, FrameStackDataset, LerpFreeAdam, RewardForwardFilter, RunningMeanStd, StickyRandomPolicy,
    StreamNormaliser, gae_episodic, gae_non_episodic,
)
from core.bridge.ffi_mgba import KEY_A, KEY_B, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_SELECT, KEY_START, KEY_UP


# --------------------------------------------------------------------------- fake bridge

class FakeBridge:
    """Deterministic stand-in: the player walks one tile per 16 frames of a held direction and the
    framebuffer paints the player's tile position, so pixels carry the same information a game would."""

    def __init__(self, ram_offset: int = 0, fail_reads: bool = False, start: tuple[int, int] = (5, 5)) -> None:
        self.start = start
        self.x, self.y, self.map = start[0], start[1], (1, 2)
        self.garbage_reads: set[int] = set()  # read indices that return a one-off garbage cell
        self.reads = 0
        self.keys = 0
        self.key_log: list[int] = []
        self.walk = 0
        self.ram_offset = ram_offset
        self.fail_reads = fail_reads
        self.closed = False
        self.loads = 0

    def step_frame(self) -> None:
        self.key_log.append(self.keys)
        moves = {KEY_UP: (0, -1), KEY_DOWN: (0, 1), KEY_LEFT: (-1, 0), KEY_RIGHT: (1, 0)}
        if self.keys in moves:
            self.walk += 1
            if self.walk % 16 == 0:
                dx, dy = moves[self.keys]
                self.x = min(14, max(0, self.x + dx))
                self.y = min(9, max(0, self.y + dy))
        else:
            self.walk = 0

    def set_hardware_keys(self, key_mask: int) -> None:
        self.keys = key_mask

    def framebuffer(self) -> NDArray[np.uint32]:
        frame = np.zeros((GBA_HEIGHT, GBA_WIDTH), dtype=np.uint32)
        frame[self.y * 16:(self.y + 1) * 16, self.x * 16:(self.x + 1) * 16] = 0x00FFFFFF
        return frame

    def load_state(self, state: bytes) -> None:
        self.x, self.y = (3, 3) if state == b"other" else self.start
        self.keys, self.walk = 0, 0
        self.loads += 1

    def get_player_coords_and_map(self) -> dict[str, int]:
        if self.fail_reads:
            raise ValueError("Uninitialized or invalid SaveBlock pointer")
        self.reads += 1
        if self.reads in self.garbage_reads:
            return {"x": 77, "y": 77, "z": 3, "map_group": 9, "map_num": 9, "tile_transition_state": 0}
        return {"x": self.x + self.ram_offset, "y": self.y + self.ram_offset, "z": 3,
                "map_group": self.map[0], "map_num": self.map[1], "tile_transition_state": 0}

    def is_in_battle(self) -> tuple[bool, bool]:
        return False, False

    def close(self) -> None:
        self.closed = True


def make_env(config: NavConfig | None = None, bridge: FakeBridge | None = None) -> tuple[NavEnv, FakeBridge]:
    fake = bridge or FakeBridge()
    env = NavEnv(config or NavConfig("firered", episode_steps=8, noop_max=0, action_repeat=24),
                 seed=0, bridge_factory=lambda _: fake, start_state=b"fake")
    return env, fake


class InProcessVecEnv:
    """Drop-in for ``SubprocVecEnv`` over fake bridges whose RAM is shifted by ``ram_offset``."""

    def __init__(self, configs: list[NavConfig], seeds: list[int], ram_offset: int = 0) -> None:
        self.envs = [NavEnv(config, seed=seed, bridge_factory=lambda _: FakeBridge(ram_offset=ram_offset),
                            start_state=b"fake") for config, seed in zip(configs, seeds)]
        self.num_envs, self.obs_shape = len(self.envs), configs[0].obs_shape

    def reset(self) -> NDArray[np.uint8]:
        return np.stack([env.reset() for env in self.envs])

    def step(self, actions: Any) -> tuple[NDArray[np.uint8], NDArray[np.bool_], list[dict[str, Any]]]:
        observations, dones, infos = [], [], []
        for env, action in zip(self.envs, actions):
            obs, _, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                info["final_episode_cells"] = len(env.cells)
                info["final_episode_maps"] = len(env.cells.maps())
                obs = env.reset()
            observations.append(obs)
            dones.append(terminated or truncated)
            infos.append(info)
        return np.stack(observations), np.array(dones, dtype=np.bool_), infos

    def __enter__(self) -> InProcessVecEnv:
        return self

    def __exit__(self, *exc: object) -> None:
        for env in self.envs:
            env.close()


# --------------------------------------------------------------------------- action space

def test_universal_action_space_is_noop_plus_eight_buttons() -> None:
    assert NUM_ACTIONS == 9 == len(ACTION_NAMES) == len(ACTION_KEYS)
    assert ACTION_NAMES[0] == "NOOP" and ACTION_KEYS[0] == 0
    mapping = dict(zip(ACTION_NAMES, ACTION_KEYS))
    assert mapping == {"NOOP": 0, "UP": KEY_UP, "DOWN": KEY_DOWN, "LEFT": KEY_LEFT, "RIGHT": KEY_RIGHT,
                       "A": KEY_A, "B": KEY_B, "START": KEY_START, "SELECT": KEY_SELECT}
    buttons = [k for k in ACTION_KEYS if k]
    assert len(set(buttons)) == 8 and all(bin(k).count("1") == 1 for k in buttons)
    assert {ACTION_NAMES[a] for a in DIRECTION_ACTIONS} == {"UP", "DOWN", "LEFT", "RIGHT"}


def test_direction_is_held_for_whole_repeat_and_button_is_pulsed() -> None:
    env, fake = make_env()
    env.reset()
    fake.key_log.clear()
    env.step(ACTION_NAMES.index("DOWN"))
    assert fake.key_log == [KEY_DOWN] * 24
    fake.key_log.clear()
    env.step(ACTION_NAMES.index("A"))
    assert fake.key_log == [KEY_A] * 6 + [0] * 18
    fake.key_log.clear()
    env.step(0)
    assert fake.key_log == [0] * 24
    with pytest.raises(ValueError):
        env.step(NUM_ACTIONS)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        NavConfig("crystal")
    with pytest.raises(ValueError):
        NavConfig("firered", action_repeat=4, button_frames=8)
    assert NavConfig("emerald").obs_shape == (4, 84, 84)


# --------------------------------------------------------------------------- preprocessing

def test_area_matrix_is_row_stochastic_and_preserves_constants() -> None:
    for n_in, n_out in ((160, 84), (240, 84), (240, 80), (160, 72), (84, 84)):
        weights = area_matrix(n_in, n_out)
        assert weights.shape == (n_out, n_in)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)
        assert (weights >= 0).all()
    np.testing.assert_allclose(area_matrix(8, 4) @ np.arange(8, dtype=np.float32), [0.5, 2.5, 4.5, 6.5])


def test_preprocessor_grayscale_from_packed_xbgr() -> None:
    pre = FramePreprocessor(84, 84)
    white = np.full((GBA_HEIGHT, GBA_WIDTH), 0x00FFFFFF, dtype=np.uint32)
    red = np.full((GBA_HEIGHT, GBA_WIDTH), 0x000000FF, dtype=np.uint32)  # R lives in the low byte
    blue = np.full((GBA_HEIGHT, GBA_WIDTH), 0x00FF0000, dtype=np.uint32)
    out = pre(white)
    assert out.shape == (84, 84) and out.dtype == np.uint8 and (out == 255).all()
    assert (pre(red) == round(0.299 * 255)).all() and (pre(blue) == round(0.114 * 255)).all()
    assert FramePreprocessor(72, 80)(white).shape == (72, 80)
    # Any console screen maps to the same observation shape (GB/GBC 160x144, NDS 256x192).
    for shape in ((144, 160), (192, 256)):
        assert (pre(np.full(shape, 0x00FFFFFF, dtype=np.uint32)) == 255).all()
    with pytest.raises(ValueError):
        pre(np.zeros(84, dtype=np.uint32))
    with pytest.raises(ValueError):
        pre(np.zeros((4, 4), dtype=np.uint32))


def test_observation_shape_stack_and_dtype() -> None:
    env, _ = make_env()
    obs = env.reset()
    assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8
    assert all((obs[i] == obs[0]).all() for i in range(4))
    new, reward, terminated, truncated, _ = env.step(ACTION_NAMES.index("RIGHT"))
    assert reward == 0.0 and not terminated and not truncated
    assert (new[:3] == obs[1:]).all() and not (new[3] == obs[3]).all()


# --------------------------------------------------------------------------- RAM never in the observation

def test_ram_is_not_in_observation() -> None:
    """Identical pixels with different RAM must give identical observations (RAM is metric-only)."""
    env_a, _ = make_env(bridge=FakeBridge(ram_offset=0))
    env_b, _ = make_env(bridge=FakeBridge(ram_offset=37))
    obs_a, obs_b = env_a.reset(), env_b.reset()
    assert (obs_a == obs_b).all()
    for action in (2, 2, 4, 5, 0, 1):
        obs_a, *_, info_a = env_a.step(action)
        obs_b, *_, info_b = env_b.step(action)
        assert (obs_a == obs_b).all()
        assert info_a["cell"] != info_b["cell"]


def test_learner_is_invariant_to_ram(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """End to end: two PPO runs whose emulators differ only in RAM learn identical weights.

    RAM still changes the logged exploration metric, which proves it reached the metric path only."""
    import ares_u.nav.train as train_module

    results: list[dict[str, torch.Tensor]] = []
    cells: list[Any] = []
    for offset in (0, 37):
        monkeypatch.setattr(train_module, "SubprocVecEnv",
                            lambda configs, seeds, o=offset: InProcessVecEnv(configs, seeds, ram_offset=o))
        out = tmp_path / f"run{offset}"
        config = train_module.TrainConfig(
            game="firered", out_dir=str(out), total_steps=48, num_envs=2, rollout_len=8, epochs=1, minibatches=2,
            norm_init_steps=4, device="cpu", torch_threads=1,
            env=NavConfig("firered", episode_steps=6, noop_max=0, action_repeat=16))
        train_module.train_ppo_rnd(config)
        results.append(torch.load(out / "final.pt", weights_only=True)["model"])
        cells.append(train_module.read_jsonl(out / "metrics.jsonl")[-1]["global_cells"])
    assert results[0].keys() == results[1].keys()
    for key in results[0]:
        torch.testing.assert_close(results[0][key], results[1][key], rtol=0, atol=0)
    assert cells[0] == cells[1] > 0  # same number of cells, but at shifted coordinates


def test_observation_only_depends_on_framebuffer(monkeypatch: pytest.MonkeyPatch) -> None:
    env, fake = make_env()
    env.reset()
    frame_before = fake.framebuffer().copy()
    monkeypatch.setattr(fake, "get_player_coords_and_map",
                        lambda: {"x": 99, "y": 99, "map_group": 9, "map_num": 9})
    obs, *_ = env.step(0)
    assert (obs[-1] == FramePreprocessor(84, 84)(frame_before)).all()


# --------------------------------------------------------------------------- metric counting

def test_unique_cell_counting_and_truncation() -> None:
    env, fake = make_env()
    env.reset()
    assert len(env.cells) == 1
    infos = [env.step(ACTION_NAMES.index("DOWN"))[4] for _ in range(2)]
    assert infos[0]["new_cell"] and infos[1]["new_cell"]
    # 48 held frames walk 3 tiles; the intermediate tile is sampled inside the action repeat.  The
    # tile entered on the very last frame is seen by one sample only, so it is confirmed next step.
    assert fake.y == 8 and infos[1]["step_cells"] == [(1, 2, 5, 6), (1, 2, 5, 7)]
    assert infos[1]["episode_cells"] == len(env.cells) == 3
    *_, info = env.step(ACTION_NAMES.index("UP"))
    assert info["new_cell"] and info["step_cells"][0] == (1, 2, 5, 8) and len(env.cells) == 4
    *_, info = env.step(ACTION_NAMES.index("UP"))
    assert not info["new_cell"]
    infos = [env.step(0)[4] for _ in range(3)]
    assert [i["t"] for i in infos] == [5, 6, 7]
    _, _, _, truncated, info = env.step(0)
    assert truncated and info["t"] == 8 and info["episode_maps"] == 1
    env.reset()
    assert len(env.cells) == 1 and env.steps == 0


def test_stable_cells_drop_one_off_reads() -> None:
    a, b, junk = (1, 2, 5, 5), (1, 2, 5, 6), (9, 9, 77, 77)
    assert stable_cells(a, [a, a, b, b, b, b]) == ([a, b], 0)
    assert stable_cells(a, [a, junk, a, a, b, b]) == ([a, b], 1)
    assert stable_cells((-1, -1, -1, -1), [(-1, -1, -1, -1), a]) == ([], 0)
    assert stable_cells(b, [a]) == ([], 0)  # the last sample of a step is judged with the next step


def test_garbage_ram_read_is_not_counted_as_a_cell() -> None:
    fake = FakeBridge()
    env, _ = make_env(bridge=fake)
    env.reset()
    fake.garbage_reads = {fake.reads + 3}
    *_, info = env.step(0)
    assert info["unconfirmed_samples"] == 1 and info["step_cells"] == [(1, 2, 5, 5)]
    assert (9, 9, 77, 77) not in env.cells.cells and len(env.cells) == 1


def test_start_state_pool_is_sampled_at_reset(tmp_path: Path) -> None:
    paths = []
    for name, payload in (("a.state", b"fake"), ("b.state", b"other")):
        path = tmp_path / name
        path.write_bytes(payload)
        paths.append(str(path))
    fake = FakeBridge()
    env = NavEnv(NavConfig("firered", episode_steps=8, noop_max=0, start_states=tuple(paths)), seed=0,
                 bridge_factory=lambda _: fake)
    starts = set()
    for _ in range(12):
        env.reset()
        starts.add((fake.x, fake.y))
    assert starts == {(5, 5), (3, 3)}
    assert NavConfig("emerald").resolved_states() == [default_state_path("emerald")]


def test_invalid_ram_read_keeps_last_cell() -> None:
    fake = FakeBridge()
    env, _ = make_env(bridge=fake)
    env.reset()
    *_, info = env.step(0)
    fake.fail_reads = True
    *_, info_fail = env.step(2)
    assert info_fail["cell"] == info["cell"] and not info_fail["new_cell"]


def test_cell_tracker_and_exploration_log() -> None:
    tracker = CellTracker()
    assert tracker.visit((1, 2, 3, 4)) and not tracker.visit((1, 2, 3, 4)) and tracker.visit((1, 3, 3, 4))
    assert len(tracker) == 2 and tracker.maps() == {(1, 2), (1, 3)}
    log = ExplorationLog()
    finished = log.observe([
        {"cell": (1, 2, 3, 4), "step_cells": [(1, 2, 3, 3), (1, 2, 3, 4)], "in_battle": False},
        {"cell": (1, 2, 3, 4), "step_cells": [(1, 2, 3, 4)], "in_battle": True, "final_episode_cells": 7,
         "final_episode_maps": 2},
        {"cell": (-1, -1, -1, -1), "step_cells": [], "in_battle": False},
    ])
    assert finished == [7] and log.episodes == 1 and len(log.global_cells) == 2
    assert log.global_maps() == 1 and log.battle_steps == 1 and log.steps == 3


# --------------------------------------------------------------------------- data utilities

def test_frame_stack_dataset_clamps_at_episode_starts() -> None:
    frames = np.arange(2 * 6, dtype=np.uint8).reshape(2, 6, 1, 1) * np.ones((1, 1, 2, 2), dtype=np.uint8)
    starts = np.zeros((2, 6), dtype=np.bool_)
    starts[:, 0] = True
    starts[0, 3] = True
    data = FrameStackDataset(frames, starts, 3)
    assert len(data) == 12
    stacks = data.batch(np.array([0, 2, 3, 4, 5, 7], dtype=np.int64))[:, :, 0, 0]
    assert stacks.tolist() == [[0, 0, 0], [0, 1, 2], [3, 3, 3], [3, 3, 4], [3, 4, 5], [6, 6, 7]]


def test_sticky_random_policy_repeats() -> None:
    policy = StickyRandomPolicy(64, seed=3, repeat_prob=0.75)
    first = policy()
    second = policy()
    same = float((first == second).mean())
    assert 0.6 < same < 0.95 and first.min() >= 0 and first.max() < NUM_ACTIONS


def test_running_mean_std_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 2.0, size=(1000, 3))
    rms = RunningMeanStd((3,))
    for chunk in np.array_split(data, 7):
        rms.update(chunk)
    np.testing.assert_allclose(rms.mean, data.mean(axis=0), rtol=1e-4)
    np.testing.assert_allclose(rms.var, data.var(axis=0), rtol=1e-3)
    clone = RunningMeanStd((3,))
    clone.load(rms.state())
    np.testing.assert_allclose(clone.mean, rms.mean)


def test_running_mean_std_ema_tracks_recent_statistics() -> None:
    rms = RunningMeanStd((), decay=0.5)
    rms.update(np.full(10, 100.0))
    assert rms.mean == pytest.approx(100.0) and rms.var == pytest.approx(0.0)
    for _ in range(40):
        rms.update(np.array([0.0, 2.0]))
    assert rms.mean == pytest.approx(1.0, abs=1e-6) and rms.var == pytest.approx(1.0, abs=1e-4)
    history = RunningMeanStd(())
    history.update(np.full(10, 100.0))
    for _ in range(40):
        history.update(np.array([0.0, 2.0]))
    assert history.var > 100.0  # all-history statistics keep the early transient
    with pytest.raises(ValueError):
        RunningMeanStd((), decay=1.0)


def test_stream_normaliser_scales_returns_to_unit_std() -> None:
    rng = np.random.default_rng(0)
    norm = StreamNormaliser(0.9, 4, decay=None)
    out = np.zeros((16, 4))
    for _ in range(50):
        out = norm(rng.random((16, 4)) * 0.01)
    filtered_std = math.sqrt(float(norm.rms.var))
    assert out.shape == (16, 4) and 0.0 < filtered_std < 1.0
    assert float(out.mean()) == pytest.approx(0.005 / filtered_std, rel=0.3)
    filt = RewardForwardFilter(0.5, 2)
    filt.update(np.array([1.0, 1.0]), dones=np.array([True, False]))
    np.testing.assert_allclose(filt.update(np.array([1.0, 1.0])), [1.0, 1.5])


def test_centered_stream_keeps_returns_small_for_constant_positive_rewards() -> None:
    rng = np.random.default_rng(3)
    centered, plain = StreamNormaliser(0.99, 4, 0.9, center=True), StreamNormaliser(0.99, 4, 0.9)
    novel = np.zeros((32, 4))
    for _ in range(30):
        rewards = 5.0 + 0.1 * rng.standard_normal((32, 4))
        rewards[:, 0] += 1.0  # env 0 keeps seeing more novel frames
        novel = centered(rewards)
        plain(rewards)
    assert abs(float(novel.mean())) < 0.5 * abs(float(novel[:, 0].mean()))  # centred around zero
    assert float(novel[:, 0].mean()) > float(novel[:, 1:].mean())  # novelty ranking preserved
    clone = StreamNormaliser(0.99, 4, 0.9, center=True)
    clone.load(centered.state())
    assert float(clone.reward_mean.mean) == pytest.approx(float(centered.reward_mean.mean))


def test_gae_episodic_cuts_at_episode_end() -> None:
    rng = np.random.default_rng(2)
    rewards, values, last = rng.random((6, 3)), rng.random((6, 3)), rng.random(3)
    no_done = np.zeros((6, 3), dtype=np.bool_)
    np.testing.assert_allclose(gae_episodic(rewards, values, last, no_done, 0.99, 0.95),
                               gae_non_episodic(rewards, values, last, 0.99, 0.95))
    dones = no_done.copy()
    dones[2, 1] = True
    adv = gae_episodic(rewards, values, last, dones, 0.99, 0.95)
    assert adv[2, 1] == pytest.approx(rewards[2, 1] - values[2, 1])
    changed = rewards.copy()
    changed[3:, 1] += 10.0  # rewards after the cut cannot leak into the finished episode
    np.testing.assert_allclose(gae_episodic(changed, values, last, dones, 0.99, 0.95)[:3, 1], adv[:3, 1])


def test_gae_non_episodic_matches_definition() -> None:
    rng = np.random.default_rng(1)
    rewards, values, last = rng.random((5, 2)), rng.random((5, 2)), rng.random(2)
    gamma, lam = 0.99, 0.95
    adv = gae_non_episodic(rewards, values, last, gamma, lam)
    extended = np.vstack([values, last[None]])
    deltas = rewards + gamma * extended[1:] - extended[:-1]
    expected = np.array([[sum((gamma * lam) ** (k - t) * deltas[k, n] for k in range(t, 5)) for n in range(2)]
                         for t in range(5)])
    np.testing.assert_allclose(adv, expected, rtol=1e-10)
    filt = RewardForwardFilter(0.5, 2)
    filt.update(np.array([1.0, 2.0]))
    np.testing.assert_allclose(filt.update(np.array([1.0, 0.0])), [1.5, 1.0])


def test_lerp_free_adam_matches_torch_adam() -> None:
    torch.manual_seed(0)
    reference = torch.nn.Linear(5, 3)
    ours = torch.nn.Linear(5, 3)
    ours.load_state_dict(reference.state_dict())
    opt_ref = Adam(reference.parameters(), lr=1e-2, eps=1e-5, foreach=False)
    opt_ours = LerpFreeAdam(list(ours.parameters()), lr=1e-2, eps=1e-5)
    for _ in range(5):
        x = torch.randn(8, 5)
        for model, opt in ((reference, opt_ref), (ours, opt_ours)):
            opt.zero_grad()
            model(x).pow(2).sum().backward()
            opt.step()
    for a, b in zip(reference.parameters(), ours.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- models

def test_space_to_depth_conv_equals_strided_8x8_conv() -> None:
    torch.manual_seed(0)
    reference = torch.nn.Conv2d(4, 32, 8, stride=4)
    s2d = SpaceToDepthConv(4, 32)
    weight = reference.weight.detach().view(32, 4, 2, 4, 2, 4).permute(0, 1, 3, 5, 2, 4).reshape(32, 64, 2, 2)
    assert s2d.conv.bias is not None and reference.bias is not None
    with torch.no_grad():
        s2d.conv.weight.copy_(weight)
        s2d.conv.bias.copy_(reference.bias)
    x = torch.rand(3, 4, 84, 84)
    torch.testing.assert_close(s2d(x), reference(x), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(("height", "width"), [(84, 84), (72, 80)])
def test_model_shapes(height: int, width: int) -> None:
    pixels = torch.rand(2, 4, height, width)
    assert NatureEncoder(4, height, width)(pixels).shape == (2, 512)
    ae = ConvAutoencoder(4, height, width)
    assert ae(pixels).shape == pixels.shape and ae.loss(pixels).ndim == 0
    logits, value = ActorCritic(NUM_ACTIONS, 4, height, width)(pixels)
    assert logits.shape == (2, NUM_ACTIONS) and value.shape == (2, len(VALUE_STREAMS))
    assert RNDModel(1, height, width)(pixels[:, -1:]).shape == (2,)
    with pytest.raises(ValueError):
        NatureEncoder(4, 82, 84)


def test_policy_head_starts_near_uniform_and_log_prob_matches_torch() -> None:
    torch.manual_seed(0)
    logits, _ = ActorCritic(NUM_ACTIONS)(torch.rand(4, 4, 84, 84))
    probs = torch.softmax(logits, -1)
    assert torch.allclose(probs, torch.full_like(probs, 1 / NUM_ACTIONS), atol=0.02)
    actions = torch.tensor([0, 3, 8, 5])
    logp, entropy = categorical_log_prob_entropy(logits, actions)
    dist = torch.distributions.Categorical(logits=logits)  # type: ignore[no-untyped-call]
    torch.testing.assert_close(logp, dist.log_prob(actions))  # type: ignore[no-untyped-call]
    torch.testing.assert_close(entropy, dist.entropy())  # type: ignore[no-untyped-call]
    assert math.isclose(float(entropy.mean()), math.log(NUM_ACTIONS), rel_tol=0.01)


def test_rnd_target_is_frozen_and_predictor_learns() -> None:
    torch.manual_seed(0)
    rnd = RNDModel(1, 84, 84)
    assert all(not p.requires_grad for p in rnd.target.parameters())
    target_before = [p.clone() for p in rnd.target.parameters()]
    frames = torch.randn(16, 1, 84, 84)
    optimizer = LerpFreeAdam(list(rnd.predictor.parameters()), lr=1e-3)
    first = float(rnd(frames).mean())
    for _ in range(30):
        optimizer.zero_grad()
        rnd(frames).mean().backward()
        optimizer.step()
    assert float(rnd(frames).mean()) < 0.5 * first
    assert all(torch.equal(a, b) for a, b in zip(target_before, rnd.target.parameters()))
    novel = torch.randn(16, 1, 84, 84) * 3
    assert float(rnd(novel).mean()) > float(rnd(frames).mean())


def test_frame_cell_hasher_quantises_and_ignores_tiny_changes() -> None:
    hasher = FrameCellHasher(84, 84, rows=5, cols=7, levels=8)
    frames = np.zeros((3, 84, 84), dtype=np.uint8)
    frames[1, 40, 40] = 255  # one pixel: below a quantisation step after area averaging
    frames[2, :42] = 255  # half the screen changes: a different key
    small = hasher.small(frames)
    assert small.shape == (3, 5, 7) and small.max() < 8
    keys = hasher.keys(frames)
    assert keys[0] == keys[1] != keys[2]


def test_episodic_novelty_counts_first_visits_and_resets_on_done() -> None:
    novelty = EpisodicNovelty(2, 84, 84)
    a = np.zeros((84, 84), dtype=np.uint8)
    b = np.full((84, 84), 200, dtype=np.uint8)
    novelty.reset(np.stack([a, a]))
    no = np.zeros(2, dtype=np.bool_)
    np.testing.assert_allclose(novelty.step(np.stack([b, a]), no), [1.0, 0.0])
    np.testing.assert_allclose(novelty.step(np.stack([b, b]), no), [0.0, 1.0])
    np.testing.assert_allclose(novelty.step(np.stack([a, a]), np.array([True, False])), [0.0, 0.0])
    np.testing.assert_allclose(novelty.step(np.stack([b, a]), no), [1.0, 0.0])  # env 0 restarted at a
    assert novelty.pop_finished() == [2] and novelty.pop_finished() == []


def test_episodic_novelty_is_pixel_only() -> None:
    """Same pixels with different RAM give the same novelty bonus (the bonus never reads RAM)."""
    env_a, _ = make_env(bridge=FakeBridge(ram_offset=0))
    env_b, _ = make_env(bridge=FakeBridge(ram_offset=11))
    obs_a, obs_b = env_a.reset(), env_b.reset()
    novelty = EpisodicNovelty(2, 84, 84)
    novelty.reset(np.stack([obs_a[-1], obs_b[-1]]))
    bonuses = []
    for action in (2, 2, 4, 0, 1, 3):
        obs_a, *_, info_a = env_a.step(action)
        obs_b, *_, info_b = env_b.step(action)
        bonus = novelty.step(np.stack([obs_a[-1], obs_b[-1]]), np.zeros(2, dtype=np.bool_))
        assert bonus[0] == bonus[1] and info_a["cell"] != info_b["cell"]
        bonuses.append(float(bonus[0]))
    assert sum(bonuses) > 0


def test_ae_encoder_transfers_into_actor_critic() -> None:
    ae = ConvAutoencoder()
    policy = ActorCritic(NUM_ACTIONS)
    policy.encoder.load_state_dict(ae.encoder.state_dict())
    x = torch.rand(2, 4, 84, 84)
    torch.testing.assert_close(policy.encoder(x), ae.encoder(x))


def test_cli_parser_builds() -> None:
    from scripts.ares_u_nav import FINAL, build_parser
    parser = build_parser()
    args = parser.parse_args(["train", "--game", "emerald", "--out-dir", "x", "--steps", "10", "--resume"])
    assert args.command == "train" and args.action_repeat == 24 and args.init == "scratch" and args.resume
    assert args.num_envs == 6 and args.epi_coef == 1.0 and args.rnd_coef == 0.5 and args.reward_norm_decay == 0.9
    args = parser.parse_args(["protocol", "--seeds", "0", "1"])
    assert args.seeds == [0, 1] and args.steps_em == args.steps_fr and args.random_run
    assert Path(args.out_dir) == FINAL


def test_summary_helpers() -> None:
    from scripts.ares_u_nav import curve_auc, hierarchical_bootstrap_diff, steps_to_reach
    rows: list[dict[str, Any]] = [{"env_steps": 100, "c": 1.0}, {"env_steps": 200, "c": 5.0},
                                  {"env_steps": 300, "c": 9.0}]
    assert steps_to_reach(rows, "c", 5.0) == 200 and steps_to_reach(rows, "c", 10.0) is None
    assert curve_auc(rows, "c", 300) == pytest.approx((50 + 300 + 700) / 300)
    diff, low, high = hierarchical_bootstrap_diff([[10.0, 12.0], [11.0, 13.0]], [[5.0, 6.0], [4.0, 7.0]], reps=2000)
    assert diff == pytest.approx(6.0) and low < 6.0 < high and low > 0


# --------------------------------------------------------------------------- native (skips without DLL/ROM/state)

def _native_available(game: str) -> bool:
    return DEFAULT_DLL.exists() and DEFAULT_ROMS[game].exists() and default_state_path(game).exists()


@pytest.mark.native
@pytest.mark.parametrize("game", ["firered", "emerald"])
def test_native_env_walks_and_counts_cells(game: str) -> None:
    if not _native_available(game):
        pytest.skip("mGBA shim, ROM or derived start state not available")
    env = NavEnv(NavConfig(game, episode_steps=16, noop_max=0))
    try:
        obs = env.reset()
        assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8 and obs.std() > 1.0
        start_cells = len(env.cells)
        infos: list[dict[str, Any]] = [env.step(ACTION_NAMES.index("DOWN"))[4] for _ in range(3)]
        assert infos[-1]["episode_cells"] > start_cells
        assert any(info["new_cell"] for info in infos)
        first = env.reset()
        assert (first == obs).all()  # deterministic restart from the fixed savestate
    finally:
        env.close()


@pytest.mark.native
def test_native_subproc_vec_env_auto_resets() -> None:
    if not _native_available("emerald"):
        pytest.skip("mGBA shim, ROM or derived start state not available")
    config = NavConfig("emerald", episode_steps=3, noop_max=0, action_repeat=8)
    with SubprocVecEnv([config, config], [0, 1]) as venv:
        obs = venv.reset()
        assert obs.shape == (2, 4, 84, 84)
        dones_seen = []
        for _ in range(3):
            obs, dones, infos = venv.step([2, 0])
            dones_seen.append(dones.tolist())
        assert dones_seen[-1] == [True, True] and "final_episode_cells" in infos[0]
        assert infos[0]["final_episode_cells"] >= infos[1]["final_episode_cells"]
        assert all(info["unconfirmed_samples"] == 0 for info in infos)


def test_paths_point_inside_repo() -> None:
    assert default_state_path("emerald").name == "emerald_start.state"
    assert Path(DEFAULT_DLL).name == "ares_mgba.dll"
