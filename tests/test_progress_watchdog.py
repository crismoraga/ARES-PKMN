"""Trace-level progress checks: moving animations do not count as campaign gains."""
from __future__ import annotations

import pytest

from core.progress_watchdog import ProgressObservation, ProgressWatchdog


def observed(frame: int, *, phase: str = "bag", xy: tuple[int, int] = (4, 6),
             xp: int = 0, objective: str = "badge1",
             strategic: tuple[object, ...] | None = None) -> ProgressObservation:
    return ProgressObservation(frame, "menu", (phase, *xy), (xp,), (0, 1, *xy), objective, strategic)


def monitor() -> ProgressWatchdog:
    return ProgressWatchdog(stationary_limits={"field": 60, "menu": 120, "battle": 240, "bootstrap": 360},
                            cycle_limit=600, cooldown=60)


def test_stationary_menu_triggers_then_escalates_at_bounded_intervals() -> None:
    watch = monitor()
    assert watch.observe(observed(0)) is None
    assert watch.observe(observed(119)) is None
    first = watch.observe(observed(120))
    assert first is not None and first.attempt == 1 and first.reason == "stationary"
    assert watch.observe(observed(179)) is None
    second = watch.observe(observed(180))
    assert second is not None and second.attempt == 2
    # Real progress, rather than a reset of frame_count, clears the attempt count.
    assert watch.observe(observed(181, xp=10, phase="overworld", xy=(5, 6))) is None
    assert watch.attempts == 0


def test_two_tile_and_menu_oscillation_is_detected_despite_changing_signature() -> None:
    watch = monitor()
    requests = []
    for frame in range(0, 721, 30):
        odd = frame // 30 % 2
        request = watch.observe(observed(frame, phase="bag" if odd else "context", xy=(4 + odd, 6)))
        if request:
            requests.append(request)
    assert requests and requests[0].reason == "repeated_state_cycle"
    assert requests[0].unproductive_frames == 600


def test_productive_training_or_new_tiles_do_not_trigger_stall() -> None:
    watch = monitor()
    for frame in range(0, 6000, 30):
        assert watch.observe(observed(frame, phase=str(frame % 4), xp=frame // 90,
                                       xy=(frame // 30 % 8, 6))) is None
    assert watch.attempts == 0


def test_resume_preserves_remaining_timeout_and_recovery_attempts() -> None:
    first, restored = monitor(), monitor()
    first.observe(observed(0))
    first.observe(observed(120))
    restored.load_state_dict(first.state_dict())
    for frame in (130, 179, 180, 240):
        assert first.observe(observed(frame)) == restored.observe(observed(frame))
    assert first.state_dict() == restored.state_dict()


def test_bad_checkpoint_is_rejected_before_modifying_monitor() -> None:
    watch = monitor()
    watch.observe(observed(0))
    original = watch.state_dict()
    invalid = dict(original, visited=[(0, 1, 2)])
    with pytest.raises(ValueError, match="location"):
        watch.load_state_dict(invalid)
    assert watch.state_dict() == original


def test_alternating_temporary_objectives_cannot_hide_a_cycle() -> None:
    watch = monitor()
    requests = []
    for frame in range(0, 751, 30):
        alternate = frame // 30 % 2
        request = watch.observe(observed(frame, phase=str(alternate),
            objective="heal" if alternate else "train"))
        if request:
            requests.append(request)
    assert requests and requests[0].reason == "repeated_state_cycle"


def test_visited_capacity_does_not_allow_unrecorded_tile_cycles() -> None:
    watch = monitor()
    watch.observe(observed(0))
    watch.visited = {(0, 1, index, 0) for index in range(16384)}
    requests = []
    for frame in range(30, 751, 30):
        request = watch.observe(observed(frame, phase=str(frame // 30 % 2), xy=(30000 + frame % 60, 6)))
        if request:
            requests.append(request)
    assert requests and requests[0].reason == "repeated_state_cycle"


def strategic_monitor() -> ProgressWatchdog:
    return ProgressWatchdog(stationary_limits={"field": 10000, "menu": 10000,
        "battle": 10000, "bootstrap": 10000}, cycle_limit=10000, cooldown=60, strategic_limit=600)


def test_xp_wins_and_temporary_objectives_cannot_hide_strategic_deadline() -> None:
    watch = strategic_monitor()
    requests = []
    for frame in range(0, 781, 30):
        request = watch.observe(observed(frame, phase=str(frame), xp=frame // 90,
            xy=(frame // 30, 6), objective="heal" if frame % 60 else "train", strategic=((), 7)))
        if request:
            requests.append((frame, request))
    assert [frame for frame, _ in requests] == [600, 660, 720, 780]
    assert [request.attempt for _, request in requests] == [1, 2, 3, 4]
    assert [request.unproductive_frames for _, request in requests] == [600, 660, 720, 780]
    assert all(request.reason == "objective_no_progress" for _, request in requests)
    assert watch.strategic_unproductive_frames == 780
    assert watch.strategic_attempts == 4


def test_new_persistent_story_flag_resets_only_strategic_progress_budget() -> None:
    watch = strategic_monitor()
    watch.observe(observed(0, strategic=((), 7)))
    request = watch.observe(observed(600, xp=100, strategic=((), 7)))
    assert request is not None and request.attempt == 1
    watch.rewinds = 2
    assert watch.observe(observed(630, xp=101, strategic=(("badge1",), 8))) is None
    assert watch.strategic_unproductive_frames == watch.strategic_attempts == watch.rewinds == 0
    assert watch.observe(observed(1229, xp=200, strategic=(("badge1",), 8))) is None
    next_request = watch.observe(observed(1230, xp=201, strategic=(("badge1",), 8)))
    assert next_request is not None and next_request.reason == "objective_no_progress"
    assert next_request.attempt == 1 and next_request.progress_frame == 630


def test_rewound_frame_and_repeated_story_flags_cannot_fake_new_history() -> None:
    watch = strategic_monitor()
    watch.observe(observed(0, strategic=(("old_milestone",), 7)))
    first = watch.observe(observed(600, xp=100, strategic=(("old_milestone",), 7)))
    assert first is not None
    watch.rewinds = 2
    assert watch.observe(observed(100, xp=0, strategic=((), 6))) is None
    assert watch.strategic_unproductive_frames == 600 and watch.strategic_attempts == 1
    assert watch.rewinds == 2
    # Relearning the abandoned timeline's already-seen history is no new gain.
    assert watch.observe(observed(130, xp=1, strategic=(("old_milestone",), 7))) is None
    second = watch.observe(observed(160, xp=2, strategic=(("old_milestone",), 7)))
    assert second is not None and second.attempt == 2 and second.rewinds == 2
    assert watch.strategic_unproductive_frames == 660


def test_checkpoint_roundtrip_preserves_strategic_clock_and_remaining_cooldown() -> None:
    first, restored = strategic_monitor(), strategic_monitor()
    first.observe(observed(0, strategic=((), 7)))
    first.observe(observed(600, strategic=((), 7)))
    restored.load_state_dict(first.state_dict())
    for frame in (600, 629, 659, 660, 720):
        observation = observed(frame, phase=str(frame), xp=frame, strategic=((), 7))
        assert first.observe(observation) == restored.observe(observation)
    assert first.state_dict() == restored.state_dict()


def test_restore_strategy_after_loading_older_checkpoint_preserves_budget_and_rearms_ladder() -> None:
    watch = strategic_monitor()
    watch.observe(observed(0, strategic=((), 7)))
    watch.observe(observed(100, strategic=((), 7)))
    earlier = watch.state_dict()
    watch.observe(observed(600, xp=100, strategic=((), 7)))
    watch.observe(observed(660, xp=101, strategic=((), 7)))
    before_rewind = watch.state_dict()
    watch.load_state_dict(earlier)
    watch.rewinds = 3
    watch.restore_strategic_progress(before_rewind)
    assert watch.last_frame == 100 and watch.rewinds == 3
    assert watch.strategic_unproductive_frames == 660 and watch.strategic_attempts == 0
    assert watch.observe(observed(100, xp=0, strategic=((), 6))) is None
    assert watch.observe(observed(159, xp=1, strategic=((), 6))) is None
    retry = watch.observe(observed(160, xp=2, strategic=((), 6)))
    assert retry is not None and retry.reason == "objective_no_progress" and retry.attempt == 1
    assert retry.rewinds == 3 and watch.strategic_unproductive_frames == 720


def test_optional_strategy_checkpoint_fields_are_backward_compatible_and_atomic() -> None:
    watch = strategic_monitor()
    watch.observe(observed(0, strategic=((), 7)))
    watch.observe(observed(600, strategic=((), 7)))
    original = watch.state_dict()
    for invalid in (dict(original, strategic_elapsed=-1), dict(original, strategic_attempts=True),
                    dict(original, strategic_recovery_at=601), dict(original, strategic_seen=["bad"]),
                    dict(original, strategic_rebase_pending="yes")):
        with pytest.raises(ValueError, match="[Ss]trategic"):
            watch.load_state_dict(invalid)
        assert watch.state_dict() == original
        with pytest.raises(ValueError, match="[Ss]trategic"):
            watch.restore_strategic_progress(invalid)
        assert watch.state_dict() == original
    legacy = {key: value for key, value in original.items() if not key.startswith("strategic_")}
    watch.load_state_dict(legacy)
    assert watch.strategic_unproductive_frames == watch.strategic_attempts == 0
    assert watch.observe(observed(630, strategic=((), 7))) is None


def test_disabled_optional_strategy_keeps_original_training_behavior() -> None:
    watch = strategic_monitor()
    for frame in range(0, 3000, 90):
        assert watch.observe(observed(frame, phase=str(frame), xp=frame // 90)) is None
    assert watch.strategic_unproductive_frames == watch.strategic_attempts == 0
