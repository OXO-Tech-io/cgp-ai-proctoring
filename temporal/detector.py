from __future__ import annotations

import statistics
from dataclasses import dataclass, replace
from typing import Iterable, Optional

from .hysteresis import HysteresisConfig, HysteresisMachine, State
from .signals import FrameSample


@dataclass(frozen=True)
class DeviationThresholds:
    # gaze (normalized offset units)
    gaze_yaw_enter: float = 0.06
    gaze_pitch_enter: float = 0.12    
    gaze_exit_ratio: float = 0.5
    gaze_axis_lock: bool = True
    # head (degrees away from neutral)
    head_yaw_enter: float = 18.0
    head_pitch_enter: float = 15.0
    head_exit_ratio: float = 0.67
    head_block_ratio: float = 1.0
    head_guard_start: float = 0.5
    head_guard_gain: float = 2.0
    min_confidence: float = 0.5

    def gaze_cfg(self, **kw) -> HysteresisConfig:
        return HysteresisConfig(enter=1.0, exit=self.gaze_exit_ratio, **kw)

    def head_cfg(self, **kw) -> HysteresisConfig:
        return HysteresisConfig(enter=1.0, exit=self.head_exit_ratio, **kw)

    def with_rest_noise(self, rest: Iterable[FrameSample], k: float = 8.0) -> "DeviationThresholds":
        """Baseline-relative thresholds (3.3): enter = max(default, k * resting std).
        `rest` = valid gaze samples from the quiet stretch after calibration.
        The default acts as a floor so a very steady user can't get a hair trigger."""
        rest = [s for s in rest if s.gaze_yaw is not None and s.gaze_pitch is not None]
        if len(rest) < 20:
            return self
        yaw = statistics.pstdev(s.gaze_yaw for s in rest)
        pit = statistics.pstdev(s.gaze_pitch for s in rest)
        return replace(self,
                       gaze_yaw_enter=max(self.gaze_yaw_enter, k * yaw),
                       gaze_pitch_enter=max(self.gaze_pitch_enter, k * pit))


def gaze_score(s: FrameSample, th: DeviationThresholds) -> Optional[float]:
    if not s.is_valid(th.min_confidence) or s.gaze_yaw is None or s.gaze_pitch is None:
        return None
    return max(abs(s.gaze_yaw) / th.gaze_yaw_enter, abs(s.gaze_pitch) / th.gaze_pitch_enter)

def gaze_axes(s: FrameSample, th: DeviationThresholds):
    """(yaw_score, pitch_score), both >= 0, or None when gaze is unusable."""
    if not s.is_valid(th.min_confidence) or s.gaze_yaw is None or s.gaze_pitch is None:
        return None
    return abs(s.gaze_yaw) / th.gaze_yaw_enter, abs(s.gaze_pitch) / th.gaze_pitch_enter


def head_score(s: FrameSample, th: DeviationThresholds,
               neutral_yaw: float, neutral_pitch: float) -> Optional[float]:
    if not s.is_valid(th.min_confidence) or s.head_yaw is None or s.head_pitch is None:
        return None
    return max(abs(s.head_yaw - neutral_yaw) / th.head_yaw_enter,
               abs(s.head_pitch - neutral_pitch) / th.head_pitch_enter)


class GazeDetector:
    """Gaze deviation with a settle rule (3.6), a soft head guard and axis-locked exit.

    Gaze is ignored while `block` is True and for `settle_s` afterwards (an open episode is
    closed with end_reason "blocked"). While the head is merely off-centre the gaze score is
    divided by a guard factor. Once an episode opens it is locked to the axis that opened it:
    only that axis (or the other one, if it reaches the enter level) keeps it alive."""

    def __init__(self, th: DeviationThresholds = DeviationThresholds(),
                 settle_s: float = 1.5, **hyst_kw):
        self.th = th
        self.settle_s = settle_s
        self.machine = HysteresisMachine(th.gaze_cfg(**hyst_kw))
        self._blocked_until = float("-inf")
        self._lock: Optional[int] = None      # 0 = yaw, 1 = pitch: the axis that opened the current episode

    def update(self, s: FrameSample, block: bool = False, head_score: Optional[float] = None):
        ts = s.ts
        was_blocked = ts < self._blocked_until
        if block:
            self._blocked_until = max(self._blocked_until, ts + self.settle_s)
        if ts < self._blocked_until:
            if not was_blocked or block:
                transitions, episode = self.machine.abort(ts, "blocked")
                self._lock = None
                if transitions:
                    return transitions, episode
            return [], None
        axes = gaze_axes(s, self.th)
        score = None
        if axes is not None:
            guard = 1.0
            if head_score is not None and head_score > self.th.head_guard_start:
                guard = 1.0 + self.th.head_guard_gain * (head_score - self.th.head_guard_start)
            sy, sp = axes[0] / guard, axes[1] / guard
            if self._lock is None or not self.th.gaze_axis_lock:
                score = max(sy, sp)
            else:
                mine, other = (sy, sp) if self._lock == 0 else (sp, sy)
                score = max(mine, other if other >= 1.0 else 0.0)
        transitions, episode = self.machine.update(ts, score)
        if self.machine.state is State.NORMAL:
            self._lock = None
        elif self._lock is None and axes is not None:
            self._lock = 0 if sy >= sp else 1
        return transitions, episode

    def finish(self, ts: float):
        """Close any open episode at the end of a session. Returns (transitions, episode)."""
        self._lock = None
        return self.machine.abort(ts, "session_end")


class HeadDetector:
    def __init__(self, neutral_yaw: float, neutral_pitch: float,
                 th: DeviationThresholds = DeviationThresholds(), **hyst_kw):
        self.th, self.ny, self.np = th, neutral_yaw, neutral_pitch
        self.machine = HysteresisMachine(th.head_cfg(**hyst_kw))
        self.last_score: Optional[float] = None

    def update(self, s: FrameSample):
        self.last_score = head_score(s, self.th, self.ny, self.np)
        return self.machine.update(s.ts, self.last_score)

    @property
    def state(self) -> State:
        return self.machine.state

    def finish(self, ts: float):
        """Close any open episode at the end of session.Returns (transition, episode). """
        return self.machine.abort(ts, "session_end")

    def block_gaze(self) -> bool:
        """Turn while the head is deviating, or merely far enough off-center
        (head_block_ratio x threshold) that gaze can't be trusted"""
        return self.state != State.NORMAL or (
            self.last_score is not None and self.last_score >= self.th.head_block_ratio
        )