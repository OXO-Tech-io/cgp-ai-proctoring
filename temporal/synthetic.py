from __future__ import annotations

import random
from typing import Optional

from .signals import FrameSample


class ScenarioBuilder:
    def __init__(self, fps: float = 30.0, seed: int = 0, noise_deg: float = 1.0,
                 fps_jitter: float = 0.0, start_ts: float = 0.0):
        """fps_jitter: fraction of frame interval randomly added/removed (0.3 = +/-30%)."""
        self.fps = fps
        self.noise = noise_deg
        self.fps_jitter = fps_jitter
        self.rng = random.Random(seed)
        self.ts = start_ts
        self.idx = 0
        self.samples: list[FrameSample] = []

    def _dt(self) -> float:
        base = 1.0 / self.fps
        if self.fps_jitter:
            base *= 1.0 + self.rng.uniform(-self.fps_jitter, self.fps_jitter)
        return base

    def _n(self) -> float:
        return self.rng.gauss(0.0, self.noise) if self.noise else 0.0

    def _emit(self, duration: float, face_count: int = 1, confidence: float = 0.95,
              head=(0.0, 0.0, 0.0), gaze: Optional[tuple[float, float]] = (0.0, 0.0)):
        end = self.ts + duration
        while self.ts < end:
            has_pose = face_count > 0
            self.samples.append(FrameSample(
                ts=round(self.ts, 6),
                frame_idx=self.idx,
                face_count=face_count,
                confidence=confidence if has_pose else 0.0,
                head_yaw=head[0] + self._n() if has_pose else None,
                head_pitch=head[1] + self._n() if has_pose else None,
                head_roll=head[2] + self._n() if has_pose else None,
                gaze_yaw=gaze[0] + self._n() if (has_pose and gaze) else None,
                gaze_pitch=gaze[1] + self._n() if (has_pose and gaze) else None,
            ))
            self.ts += self._dt()
            self.idx += 1
        return self

    def steady(self, duration: float):
        return self._emit(duration)

    def gaze_shift(self, duration: float, gaze_yaw: float = 0.0, gaze_pitch: float = 0.0):
        """Eyes move, head stays still - should register as gaze deviation."""
        return self._emit(duration, gaze=(gaze_yaw, gaze_pitch))

    def head_turn(self, duration: float, yaw: float = 0.0, pitch: float = 0.0):
        """Head turns, eyes stay centered in the head - must NOT register as gaze deviation."""
        return self._emit(duration, head=(yaw, pitch, 0.0))

    def no_face(self, duration: float):
        return self._emit(duration, face_count=0)

    def low_confidence(self, duration: float, confidence: float = 0.2):
        return self._emit(duration, confidence=confidence)

    def extra_face(self, duration: float, count: int = 2):
        return self._emit(duration, face_count=count)

    def calibrating(self, duration: float):
        """Face and head pose present, gaze not available yet (gaze calibration is running)."""
        return self._emit(duration, gaze=None)

    def reading(self, duration: float, yaw_amp: float = 0.03, pitch_amp: float = 0.06,
                head_amp: float = 1.5, blink_every: float = 4.0, blink_s: float = 0.2):
        """Normal screen reading: the gaze wanders slowly inside +-amp, the head sways a little and
        the eyes blink (gaze unavailable for blink_s) about every blink_every seconds.
        None of this is a deliberate look, so a correct detector stays silent."""
        end = self.ts + duration
        gy = gp = 0.0
        next_blink = self.ts + blink_every * (0.5 + self.rng.random())
        blink_until = -1.0
        while self.ts < end:
            gy = max(-yaw_amp, min(yaw_amp, 0.8 * gy + self.rng.gauss(0.0, 0.5 * yaw_amp)))
            gp = max(-pitch_amp, min(pitch_amp, 0.8 * gp + self.rng.gauss(0.0, 0.5 * pitch_amp)))
            if self.ts >= next_blink:
                blink_until = self.ts + blink_s
                next_blink = self.ts + blink_every * (0.5 + self.rng.random())
            gaze = None if self.ts < blink_until else (gy, gp)
            head = (self.rng.uniform(-head_amp, head_amp), self.rng.uniform(-head_amp, head_amp), 0.0)
            self._emit(0.5 * self._dt(), head=head, gaze=gaze)
        return self

    def build(self) -> list[FrameSample]:
        return list(self.samples)