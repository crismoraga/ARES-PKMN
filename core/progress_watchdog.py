"""Detect stalled observations, including cycles that still advance frames.

Hardware frame counters and animation frames are deliberately not progress.
The caller supplies public gameplay facts and owns all recovery inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Literal

ProgressContext = Literal["field", "menu", "battle", "bootstrap"]


class CampaignStalled(RuntimeError):
    """Bounded recovery was exhausted; retain evidence instead of spinning."""


@dataclass(frozen=True)
class ProgressObservation:
    frame: int
    context: ProgressContext
    signature: tuple[object, ...]
    gains: tuple[object, ...]
    location: tuple[int, int, int, int]
    objective: str
    strategic_gains: tuple[object, ...] | None = None
    # Monotone preparation score (party levels and size). A new high earns
    # bounded strategic credit: real training toward a boss is not a stall.
    preparation: int | None = None


@dataclass(frozen=True)
class RecoveryRequest:
    attempt: int
    context: ProgressContext
    reason: str
    stationary_frames: int
    unproductive_frames: int
    rewinds: int = 0
    progress_frame: int = 0


class ProgressWatchdog:
    """Bound inactivity and repetitive input, preserving counters on resume.

    ``attempts`` counts recoveries since the last observed gain and ``rewinds``
    counts checkpoint rewinds since then. A rewind moves the frame counter
    backwards; that restarts the deadlines but keeps the visited-tile memory, so
    walking the abandoned timeline again is never mistaken for exploration.
    With strategic facts enabled, only new story history clears rewind counts;
    the independent strategic clock and ladder survive training gains and rewinds.
    """

    def __init__(self, *, stationary_limits: dict[ProgressContext, int] | None = None,
                 cycle_limit: int = 14400, cooldown: int = 600,
                 strategic_limit: int = 72000, preparation_credit: int | None = None,
                 preparation_budget: int | None = None) -> None:
        self.limits: dict[ProgressContext, int] = stationary_limits or {
            "field": 600, "menu": 1200, "battle": 2400, "bootstrap": 3600}
        if set(self.limits) != {"field", "menu", "battle", "bootstrap"} or min(self.limits.values()) < 1:
            raise ValueError("Every progress context requires a positive deadline")
        if min(cycle_limit, cooldown) < 1:
            raise ValueError("Progress deadlines must be positive")
        if type(strategic_limit) is not int or strategic_limit < 1:
            raise ValueError("Strategic progress deadline must be a positive integer")
        self.cycle_limit, self.cooldown = cycle_limit, cooldown
        self.last_frame = -1
        self.motion_at = self.gain_at = self.recovery_at = 0
        self.signature = self.gains = self.objective = ""
        self.visited: set[tuple[int, int, int, int]] = set()
        self.attempts = 0
        self.rewinds = 0
        self.strategic_limit = strategic_limit
        self.strategic_attempts = 0
        self._strategic_elapsed = self._strategic_recovery_at = self._strategic_progress_frame = 0
        self._strategic_last_frame = -1
        self._strategic_fingerprint: str | None = None
        self._strategic_seen: set[str] = set()
        self._strategic_rebase_pending = False
        self.preparation_credit = strategic_limit // 2 if preparation_credit is None else preparation_credit
        self.preparation_budget = 4 * strategic_limit if preparation_budget is None else preparation_budget
        if min(self.preparation_credit, self.preparation_budget) < 0:
            raise ValueError("Preparation credit must be nonnegative")
        self._preparation_best: int | None = None
        self._preparation_spent = 0

    @property
    def strategic_unproductive_frames(self) -> int:
        """Cumulative gameplay frames without new story facts, surviving rewinds."""
        return self._strategic_elapsed

    def _observe_strategic(self, observation: ProgressObservation) -> bool:
        """Return whether this observation proves previously unseen story gains."""
        frame = observation.frame
        if observation.strategic_gains is None:
            # An optional disabled context, such as uninitialized bootstrap RAM,
            # cannot accrue or erase a previously enabled gameplay clock.
            if self._strategic_fingerprint is not None:
                self._strategic_last_frame = frame
            return False
        fingerprint = self._digest(observation.strategic_gains)
        if self._strategic_fingerprint is None:
            self._strategic_fingerprint = fingerprint
            self._strategic_seen.add(fingerprint)
            self._strategic_last_frame = self._strategic_progress_frame = frame
            return False
        rewound = frame < self._strategic_last_frame or self._strategic_rebase_pending
        self._strategic_elapsed += max(0, frame - self._strategic_last_frame)
        self._strategic_last_frame = frame
        self._strategic_rebase_pending = False
        if rewound:
            # The loaded cartridge can lose flags. Neither that rollback nor
            # re-earning the abandoned timeline's flags constitutes new history.
            self._strategic_fingerprint = fingerprint
            if len(self._strategic_seen) < 16384:
                self._strategic_seen.add(fingerprint)
            self._strategic_progress_frame = min(self._strategic_progress_frame, frame)
            return False
        if fingerprint == self._strategic_fingerprint:
            return False
        novel = fingerprint not in self._strategic_seen and len(self._strategic_seen) < 16384
        self._strategic_fingerprint = fingerprint
        if not novel:
            return False
        self._strategic_seen.add(fingerprint)
        self._strategic_elapsed = self._strategic_recovery_at = self.strategic_attempts = 0
        self._strategic_progress_frame = frame
        self._preparation_spent = 0
        return True

    def _observe_preparation(self, score: int | None) -> None:
        """Credit a new preparation high-water mark against the strategic clock.

        Re-earning levels lost to a rewind is not new (the mark survives),
        and the total credit between story facts is bounded, so endless
        grinding still reaches the strategic deadline.
        """
        if score is None or self._strategic_fingerprint is None:
            return
        if self._preparation_best is None:
            self._preparation_best = score
            return
        if score <= self._preparation_best:
            return
        gained = score - self._preparation_best
        self._preparation_best = score
        credit = min(gained * self.preparation_credit, max(0, self.preparation_budget - self._preparation_spent))
        if credit <= 0:
            return
        self._preparation_spent += credit
        self._strategic_elapsed = max(0, self._strategic_elapsed - credit)
        self._strategic_recovery_at = min(self._strategic_recovery_at, self._strategic_elapsed)

    @staticmethod
    def _digest(value: object) -> str:
        return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()

    def observe(self, observation: ProgressObservation) -> RecoveryRequest | None:
        if observation.frame < 0 or observation.context not in self.limits:
            raise ValueError("Invalid progress observation")
        frame = observation.frame
        strategic_enabled = observation.strategic_gains is not None
        strategic_progress = self._observe_strategic(observation)
        if strategic_enabled:
            self._observe_preparation(observation.preparation)
        signature, gains = self._digest(observation.signature), self._digest(observation.gains)
        first = self.last_frame < 0
        if first or frame < self.last_frame:
            self.motion_at = self.gain_at = self.recovery_at = frame
            self.signature, self.gains, self.objective = signature, gains, observation.objective
            if first:
                self.visited = {observation.location}
            else:
                self.visited.add(observation.location)
            self.attempts = 0
        self.objective = observation.objective
        if signature != self.signature:
            self.motion_at, self.signature = frame, signature
        # A previously visited tile is not new progress: this catches patrol
        # loops, inaccessible exits and two-state menu oscillation.
        novel_location = observation.location not in self.visited and len(self.visited) < 16384
        if gains != self.gains or novel_location:
            self.gain_at, self.gains, self.attempts = frame, gains, 0
            if not strategic_enabled:
                self.rewinds = 0
            if novel_location:
                self.visited.add(observation.location)
        if strategic_progress:
            self.rewinds = self.attempts = 0
            self.gain_at = frame
        self.last_frame = frame
        stationary, unproductive = frame - self.motion_at, frame - self.gain_at
        if (strategic_enabled and self._strategic_elapsed >= self.strategic_limit
                and self._strategic_elapsed - self._strategic_recovery_at >= self.cooldown):
            self.strategic_attempts += 1
            self._strategic_recovery_at = self._strategic_elapsed
            self.recovery_at = frame
            return RecoveryRequest(self.strategic_attempts, observation.context, "objective_no_progress",
                                   stationary, self._strategic_elapsed, self.rewinds, self._strategic_progress_frame)
        reason = ("stationary" if stationary >= self.limits[observation.context] else
                  ("repeated_state_cycle" if unproductive >= self.cycle_limit else ""))
        if not reason or frame - self.recovery_at < self.cooldown:
            return None
        self.attempts += 1
        self.recovery_at = frame
        return RecoveryRequest(self.attempts, observation.context, reason, stationary, unproductive,
                               self.rewinds, self.gain_at)

    def rearm(self) -> None:
        """A new worker process is a new attempt: give its ladder a fresh start.

        The supervisor bounds how many such attempts happen without new story
        facts; without this a restarted worker inherited an exhausted ladder and
        stopped again within seconds, so restarts could never help.
        """
        self.attempts = self.rewinds = self.strategic_attempts = 0
        self._strategic_elapsed = min(self._strategic_elapsed, self.strategic_limit // 2)
        self._strategic_recovery_at = min(self._strategic_recovery_at, self._strategic_elapsed)

    def grant_strategic_grace(self, frames: int) -> bool:
        """Give a new strategy time to act; bounded by the preparation budget.

        Returns False when the budget since the last story fact is spent, so a
        stream of strategy changes cannot postpone the deadline forever.
        """
        if frames <= 0 or self._preparation_spent + frames > self.preparation_budget:
            return False
        self._preparation_spent += frames
        self._strategic_elapsed = max(0, self._strategic_elapsed - frames)
        self._strategic_recovery_at = min(self._strategic_recovery_at, self._strategic_elapsed)
        self.strategic_attempts = 0
        return True

    def stationary_frames(self, frame: int) -> int:
        return max(0, frame - self.motion_at)

    def unproductive_frames(self, frame: int) -> int:
        return max(0, frame - self.gain_at)

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "last_frame": self.last_frame, "motion_at": self.motion_at,
                "gain_at": self.gain_at, "recovery_at": self.recovery_at,
                "signature": self.signature, "gains": self.gains, "objective": self.objective,
                "visited": sorted(self.visited), "attempts": self.attempts, "rewinds": self.rewinds,
                "strategic_limit": self.strategic_limit, "strategic_attempts": self.strategic_attempts,
                "strategic_elapsed": self._strategic_elapsed, "strategic_recovery_at": self._strategic_recovery_at,
                "strategic_progress_frame": self._strategic_progress_frame,
                "strategic_last_frame": self._strategic_last_frame,
                "strategic_fingerprint": self._strategic_fingerprint,
                "strategic_seen": sorted(self._strategic_seen),
                "strategic_rebase_pending": self._strategic_rebase_pending,
                "strategic_preparation_best": self._preparation_best,
                "strategic_preparation_spent": self._preparation_spent}

    def restore_strategic_progress(self, state: dict[str, Any], *, reset_attempts: bool = True) -> None:
        """Carry abandoned-timeline strategy budgets over a loaded checkpoint.

        Call after loading the emulator/watchdog snapshot. The loaded frame is
        the new baseline, so the skipped interval is never counted twice. Only
        the recovery ladder is optionally rearmed; elapsed time and cooldown are
        retained, and the next observation cannot mistake rollback for a gain.
        The caller retains its own rewind count and ordinary watchdog state.
        """
        merged = self.state_dict()
        for name, value in state.items():
            if name.startswith("strategic_"):
                merged[name] = value
        self._validate_strategic(merged)
        merged["strategic_last_frame"] = self.last_frame
        merged["strategic_progress_frame"] = min(merged["strategic_progress_frame"], max(0, self.last_frame))
        merged["strategic_rebase_pending"] = merged["strategic_fingerprint"] is not None
        if reset_attempts:
            merged["strategic_attempts"] = 0
        self.load_state_dict(merged)

    def _validate_strategic(self, state: dict[str, Any]) -> dict[str, Any]:
        """Normalize optional version-one extension fields without mutation."""
        normalized: dict[str, Any] = {"strategic_limit": state.get("strategic_limit", self.strategic_limit),
            "strategic_attempts": state.get("strategic_attempts", 0),
            "strategic_elapsed": state.get("strategic_elapsed", 0),
            "strategic_recovery_at": state.get("strategic_recovery_at", 0),
            "strategic_progress_frame": state.get("strategic_progress_frame", 0),
            "strategic_last_frame": state.get("strategic_last_frame", -1)}
        if any(type(value) is not int for value in normalized.values()):
            raise ValueError("Invalid strategic progress counters")
        if (normalized["strategic_limit"] < 1 or normalized["strategic_last_frame"] < -1
                or any(normalized[name] < 0 for name in ("strategic_attempts", "strategic_elapsed",
                    "strategic_recovery_at", "strategic_progress_frame"))
                or normalized["strategic_recovery_at"] > normalized["strategic_elapsed"]):
            raise ValueError("Invalid strategic progress counter range")
        fingerprint, seen = state.get("strategic_fingerprint"), state.get("strategic_seen", [])
        pending = state.get("strategic_rebase_pending", False)
        best, spent = state.get("strategic_preparation_best"), state.get("strategic_preparation_spent", 0)
        if (best is not None and (type(best) is not int or best < 0)) or type(spent) is not int or spent < 0:
            raise ValueError("Invalid strategic preparation counters")
        normalized.update(strategic_preparation_best=best, strategic_preparation_spent=spent)
        if (fingerprint is not None and (not isinstance(fingerprint, str) or len(fingerprint) != 64)
                or not isinstance(seen, list) or len(seen) > 16384
                or any(not isinstance(value, str) or len(value) != 64 for value in seen)
                or type(pending) is not bool):
            raise ValueError("Invalid strategic progress fingerprints")
        if fingerprint is not None and fingerprint not in seen:
            raise ValueError("Strategic progress fingerprint is missing from history")
        normalized.update(strategic_fingerprint=fingerprint, strategic_seen=set(seen),
                          strategic_rebase_pending=pending)
        return normalized

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != 1:
            raise ValueError("Unsupported progress watchdog checkpoint")
        numbers = ("last_frame", "motion_at", "gain_at", "recovery_at", "attempts")
        if any(type(state.get(key)) is not int for key in numbers):
            raise ValueError("Invalid progress watchdog counters")
        if state["last_frame"] < -1 or any(state[key] < 0 for key in numbers[1:]):
            raise ValueError("Invalid progress watchdog counter range")
        if any(not isinstance(state.get(key), str) for key in ("signature", "gains", "objective")):
            raise ValueError("Invalid progress watchdog fingerprints")
        raw = state.get("visited")
        if not isinstance(raw, list) or len(raw) > 16384:
            raise ValueError("Invalid progress watchdog visited locations")
        visited: set[tuple[int, int, int, int]] = set()
        for row in raw:
            if not isinstance(row, (list, tuple)) or len(row) != 4 or any(type(x) is not int for x in row):
                raise ValueError("Invalid progress watchdog location")
            visited.add((row[0], row[1], row[2], row[3]))
        rewinds = state.get("rewinds", 0)
        if type(rewinds) is not int or rewinds < 0:
            raise ValueError("Invalid progress watchdog rewind counter")
        strategic = self._validate_strategic(state)
        for key in (*numbers, "signature", "gains", "objective"):
            setattr(self, key, state[key])
        self.visited = visited
        self.rewinds = rewinds
        for key, value in strategic.items():
            if key == "strategic_preparation_best":
                self._preparation_best = value
                continue
            if key == "strategic_preparation_spent":
                self._preparation_spent = value
                continue
            attribute = key if key in ("strategic_limit", "strategic_attempts") else "_" + key
            setattr(self, attribute, value)
