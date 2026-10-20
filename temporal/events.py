from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass, replace
from typing import Callable, Deque, List, Optional,Tuple
from .hysteresis import Episode, State, Transition
from .signals import FrameSample

AXES = ("yaw", "pitch")
Vector = Tuple[float, float]

@dataclass(frozen=True)
class EventConfig:
    sustained_s: float = 3.0
    repeat_window_s: float = 30.0
    repeat_min: int = 3
    min_peak: float = 1.0
    min_run_s: float = 0.2
    tail_frac: float = 0.35

@dataclass(frozen=True)
class Event:
    source: str                      
    kind: str                         
    direction: str                   
    start_ts: float
    end_ts: float
    duration_s: float
    peak: float                      
    end_reason: str                   
    count: int = 1                 

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if self.kind == "repeated":
            return (f"{self.source.upper()} repeated {self.direction}: {self.count} glances "
                    f"in {self.duration_s:.0f}s")
        return (f"{self.source.upper()} {self.kind} {self.direction} {self.duration_s:.1f}s "
                f"peak {self.peak:.2f}x ({self.end_reason})")


def split_runs(frames: List[Tuple[float, float, float]], floor: float) -> List[list]:
    """frames = [(ts, yaw_score, pitch_score)] -> list of runs (lists of frames).

    A frame is "active" on an axis when |score| >= floor. Frames with no active
    axis (near centre) are skipped. A new run starts when a frame is active on
    the OPPOSITE sign of the run's current label axis. The label is the signed
    axis with the most accumulated signal so far, which keeps noise on a
    near-equal diagonal from flipping it."""
    runs: List[list] = []
    run: Optional[list] = None
    mass: dict = {}
    for f in frames:
        val = (f[1], f[2])
        active = [(ax, 1 if val[ax] > 0 else -1, abs(val[ax]))
                  for ax in (0, 1) if abs(val[ax]) >= floor]
        if not active:
            continue
        if run is not None:
            lab = max(mass, key=mass.get)
            if any(ax == lab[0] and sg == -lab[1] for ax, sg, _ in active):
                run = None
        if run is None:
            run, mass = [], {}
            runs.append(run)
        run.append(f)
        for ax, sg, sc in active:
            mass[(ax, sg)] = mass.get((ax, sg), 0.0) + sc
    return runs


def describe_run(run: list, floor: float, tail_frac: float = 0.0):
    """-> (axis, sign, peak, start_ts, end_ts) for one run.

    With tail_frac > 0 the run is trimmed to the frames at or above tail_frac * peak, so a slow
    decay or a resting offset after a big look (0.5-0.7x of the threshold) does not count as
    part of the look. Runs with a small peak are unaffected (the floor wins)."""
    mass: dict = {}
    for f in run:
        for ax in (0, 1):
            if abs(f[1 + ax]) >= floor:
                k = (ax, 1 if f[1 + ax] > 0 else -1)
                mass[k] = mass.get(k, 0.0) + abs(f[1 + ax])
    axis, sign = max(mass, key=mass.get)
    on = [f for f in run if f[1 + axis] * sign >= floor]
    peak = max(f[1 + axis] * sign for f in on)
    score = sorted((f[1 + axis] * sign for f in on), reverse=True)
    ref =  score[min(2, len(score) - 1)]
    cut = max(floor, tail_frac * peak)
    core = [f for f in on if f[1 + axis] * sign >= cut] or on
    crossed = [f[0] for f in core if f[1 + axis] * sign >= 1.0]
    start = crossed[0] if crossed else core[0][0]
    return axis, sign, peak, start, core[-1][0]


class EventBuilder:
    """Feed it every sample plus what the detector returned for that sample."""

    def __init__(self, source: str, vector_fn: Callable[[FrameSample], Optional[Vector]],
                 floor: float, cfg: EventConfig = EventConfig()):
        self.source, self._vec, self.floor, self.cfg = source, vector_fn, floor, cfg
        self._open = False
        self._buf: list = []

    @classmethod
    def for_gaze(cls, th, cfg: EventConfig = EventConfig()) -> "EventBuilder":
        def vec(s: FrameSample) -> Optional[Vector]:
            if not s.is_valid(th.min_confidence) or s.gaze_yaw is None or s.gaze_pitch is None:
                return None
            return s.gaze_yaw / th.gaze_yaw_enter, s.gaze_pitch / th.gaze_pitch_enter
        return cls("gaze", vec, th.gaze_exit_ratio, cfg)

    @classmethod
    def for_head(cls, neutral_yaw: float, neutral_pitch: float, th,
                 cfg: EventConfig = EventConfig()) -> "EventBuilder":
        def vec(s: FrameSample) -> Optional[Vector]:
            if not s.is_valid(th.min_confidence) or s.head_yaw is None or s.head_pitch is None:
                return None
            return ((s.head_yaw - neutral_yaw) / th.head_yaw_enter,
                    (s.head_pitch - neutral_pitch) / th.head_pitch_enter)
        return cls("head", vec, th.head_exit_ratio, cfg)

    def update(self, sample: Optional[FrameSample], transitions: List[Transition],
               episode: Optional[Episode]) -> List[Event]:
        """`sample` may be None when flushing at session end."""
        for t in transitions:
            if t.new is State.SUSPECT:
                self._open, self._buf = True, []
            elif t.new is State.NORMAL and episode is None:
                self._open, self._buf = False, []
        if sample is not None and self._open:
            v = self._vec(sample)
            if v is not None:
                self._buf.append((sample.ts, v[0], v[1]))
        if episode is None:
            return []
        frames, self._buf, self._open = self._buf, [], False
        if episode.confirmed_ts is None:                 
            return []
        return self._segment(frames, episode)

    def _segment(self, frames, episode: Episode) -> List[Event]:
        c = self.cfg
        runs = split_runs(frames, self.floor)
        events: List[Event] = []
        for i, run in enumerate(runs):
            axis, sign, peak, start, end = describe_run(run, self.floor, c.tail_frac)
            if len(runs) > 1:
                strongest = max(max(abs(f[1]), abs(f[2])) for f in run)
                if strongest < c.min_peak or end - start < c.min_run_s:
                    continue                              
            dur = end - start
            events.append(Event(
                source=self.source,
                kind="sustained" if dur >= c.sustained_s else "glance",
                direction=f"{AXES[axis]}{'+' if sign > 0 else '-'}",
                start_ts=start, end_ts=end, duration_s=dur, peak=peak,
                end_reason=episode.end_reason if i == len(runs) - 1 else "direction_change"))

        if events:
            events[-1] = replace(events[-1], end_reason=episode.end_reason)
        if not events and runs:                           
            run = max(runs, key=len)
            axis, sign, peak, start, end = describe_run(run, self.floor, c.tail_frac)
            events.append(Event(self.source, "glance", f"{AXES[axis]}{'+' if sign > 0 else '-'}",
                                start, end, end - start, peak, episode.end_reason))
        return events


class RepeatTracker:
    """Adds a "repeated" event when `repeat_min` fresh glances from one source
    fall inside `repeat_window_s`. Each glance counts toward one burst only."""

    def __init__(self, cfg: EventConfig = EventConfig()):
        self.cfg = cfg
        self._recent: dict[str, Deque[list]] = {}

    def add(self, ev: Event) -> List[Event]:
        out = [ev]
        if ev.kind != "glance":
            return out
        dq = self._recent.setdefault(ev.source, deque())
        dq.append([ev, False])
        while dq and ev.end_ts - dq[0][0].end_ts > self.cfg.repeat_window_s:
            dq.popleft()
        fresh = [e for e, flagged in dq if not flagged]
        if len(fresh) >= self.cfg.repeat_min:
            for item in dq:
                item[1] = True
            dirs = {e.direction for e in fresh}
            out.append(Event(
                source=ev.source, kind="repeated",
                direction=dirs.pop() if len(dirs) == 1 else "mixed",
                start_ts=fresh[0].start_ts, end_ts=fresh[-1].end_ts,
                duration_s=fresh[-1].end_ts - fresh[0].start_ts,
                peak=max(e.peak for e in fresh), end_reason="burst", count=len(fresh)))
        return out