from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from .signals import NUMERIC_SIGNALS, FrameSample


@dataclass(frozen=True, slots=True)
class WindowStats:
    signal: str
    span_s: float          # time actually covered by samples in the window
    valid_s: float         # time covered by valid samples for this signal
    coverage: float        # valid_s / span_s (0 if empty)
    mean: Optional[float]
    std: Optional[float]
    min: Optional[float]
    max: Optional[float]
    n_valid: int


class SlidingWindow:
    def __init__(self, duration_s: float, min_confidence: float = 0.5,
                 max_gap_s: float = 0.5):
        if duration_s <= 0:
            raise ValueError("duration_s must be > 0")
        self.duration_s = duration_s
        self.min_confidence = min_confidence
        self.max_gap_s = max_gap_s
        self._buf: deque[FrameSample] = deque()

    # ---- ingestion -------------------------------------------------------
    def push(self, sample: FrameSample) -> None:
        buf = self._buf
        if buf and sample.ts < buf[-1].ts:
            raise ValueError(
                f"Out-of-order sample: ts={sample.ts} < last ts={buf[-1].ts}. "
                "Use a monotonic clock in the capture loop."
            )
        buf.append(sample)
        cutoff = sample.ts - self.duration_s
        while len(buf) > 1 and buf[0].ts < cutoff:
            buf.popleft()

    def clear(self) -> None:
        self._buf.clear()

    # ---- introspection ---------------------------------------------------
    def __len__(self) -> int:
        return len(self._buf)

    @property
    def latest(self) -> Optional[FrameSample]:
        return self._buf[-1] if self._buf else None

    @property
    def is_full(self) -> bool:
        """True once the window spans (almost) its whole duration."""
        if len(self._buf) < 2:
            return False
        return (self._buf[-1].ts - self._buf[0].ts) >= self.duration_s * 0.95

    # ---- weighting -------------------------------------------------------
    def _weighted(self):
        """Yield (sample, weight_seconds). The newest sample gets the previous
        interval so it isn't ignored."""
        buf = self._buf
        n = len(buf)
        if n == 0:
            return
        last_dt = (buf[-1].ts - buf[-2].ts) if n >= 2 else 0.0
        for i, s in enumerate(buf):
            dt = (buf[i + 1].ts - s.ts) if i + 1 < n else last_dt
            yield s, min(max(dt, 0.0), self.max_gap_s)

    def _is_valid(self, s: FrameSample, signal: str) -> bool:
        return s.is_valid(self.min_confidence) and s.get(signal) is not None

    # ---- statistics ------------------------------------------------------
    def stats(self, signal: str) -> WindowStats:
        if signal not in NUMERIC_SIGNALS:
            raise KeyError(signal)
        total_w = valid_w = 0.0
        sum_x = sum_x2 = 0.0
        lo = hi = None
        n_valid = 0
        for s, w in self._weighted():
            total_w += w
            if not self._is_valid(s, signal):
                continue
            x = s.get(signal)
            n_valid += 1
            lo = x if lo is None else min(lo, x)
            hi = x if hi is None else max(hi, x)
            if w > 0:
                valid_w += w
                sum_x += w * x
                sum_x2 += w * x * x

        if valid_w > 0:
            mean = sum_x / valid_w
            var = max(sum_x2 / valid_w - mean * mean, 0.0)
            std = math.sqrt(var)
        elif n_valid:  # single sample, no duration yet
            mean, std = lo, 0.0
        else:
            mean = std = None

        return WindowStats(
            signal=signal,
            span_s=total_w,
            valid_s=valid_w,
            coverage=(valid_w / total_w) if total_w > 0 else 0.0,
            mean=mean, std=std, min=lo, max=hi, n_valid=n_valid,
        )

    def fraction_where(self, predicate: Callable[[FrameSample], bool],
                       signal: Optional[str] = None) -> Optional[float]:
        """Time-weighted fraction of VALID window time where predicate holds.

        Returns None if there is no valid time in the window (don't treat that
        as 0 - it means "unknown").
        """
        valid_w = hit_w = 0.0
        for s, w in self._weighted():
            if signal is not None:
                if not self._is_valid(s, signal):
                    continue
            elif not s.is_valid(self.min_confidence):
                continue
            valid_w += w
            if predicate(s):
                hit_w += w
        return (hit_w / valid_w) if valid_w > 0 else None

    def face_fraction(self, predicate: Callable[[int], bool]) -> Optional[float]:
        """Time-weighted fraction of ALL window time where predicate(face_count)
        holds. Ignores confidence - Phase 4 uses this for zero/multi-face checks."""
        total_w = hit_w = 0.0
        for s, w in self._weighted():
            total_w += w
            if predicate(s.face_count):
                hit_w += w
        return (hit_w / total_w) if total_w > 0 else None