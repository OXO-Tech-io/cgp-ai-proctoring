from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .detector import DeviationThresholds
from .events import Event, EventConfig
from .monitor import TemporalMonitor
from .output import events_path_for
from .recording import SessionReplayer
from .signals import FrameSample


@dataclass(frozen=True)
class Spread:
    p50: float
    p95: float
    p99: float
    max: float


@dataclass
class Report:
    recording: str
    frames: int
    duration_s: float
    start_ts: float
    fps: float
    calibration_ts: float
    neutral: Tuple[float, float]
    neutral_source: str
    window: Tuple[float, float]                  # absolute ts
    window_frames: int
    gaze_yaw: Spread
    gaze_pitch: Spread
    head_yaw: Spread
    head_pitch: Spread
    th: DeviationThresholds
    current_events: List[Event]
    rec_th: DeviationThresholds
    rec_events: List[Event]
    outside_events: List[Event] = field(default_factory=list)
    clear_yaw: float = 0.0
    clear_pitch: float = 0.0
    stretches: List[Tuple[float, float]] = field(default_factory=list) 

    @property
    def recommended_yaw(self) -> float:
        return self.rec_th.gaze_yaw_enter

    @property
    def recommended_pitch(self) -> float:
        return self.rec_th.gaze_pitch_enter


def _pct(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def _spread(vals: Sequence[float]) -> Spread:
    v = sorted(vals)
    return Spread(_pct(v, 0.5), _pct(v, 0.95), _pct(v, 0.99), v[-1] if v else 0.0)


def _ceil_to(x: float, step: float) -> float:
    return round(math.ceil(x / step - 1e-9) * step, 6)


def load(path) -> List[FrameSample]:
    return list(SessionReplayer(path).samples())


def calibration_index(samples: Sequence[FrameSample]) -> Optional[int]:
    """Index of the first frame that carries gaze = the moment calibration finished."""
    for i, s in enumerate(samples):
        if s.gaze_yaw is not None and s.gaze_pitch is not None:
            return i
    return None


def find_neutral(path, samples, cal_idx: int, override: Optional[Tuple[float, float]] = None):
    if override is not None:
        return override, "--neutral"
    ev_path = events_path_for(path)
    if ev_path.exists():
        try:
            cal = json.loads(ev_path.read_text(encoding="utf-8")).get("calibrations") or []
            if cal:
                return (cal[0]["neutral_yaw"], cal[0]["neutral_pitch"]), ev_path.name
        except (OSError, ValueError, KeyError):
            pass
    t_cal = samples[cal_idx].ts
    pts = [(s.head_yaw, s.head_pitch) for s in samples[:cal_idx]
           if s.head_yaw is not None and s.head_pitch is not None and t_cal - s.ts <= 3.0]
    if not pts:
        return (0.0, 0.0), "default 0,0"
    return (statistics.median(p[0] for p in pts), statistics.median(p[1] for p in pts)), "estimated before calibration ended"


def replay_events(samples: Sequence[FrameSample], neutral, th: DeviationThresholds,
                  cfg: Optional[EventConfig] = None, start_idx: int = 0) -> List[Event]:
    mon = TemporalMonitor(th, emit=lambda _l: None, event_cfg=cfg)
    mon.start(neutral[0], neutral[1])
    rest = list(samples[start_idx:])
    for s in rest:
        mon.update(s)
    if rest:
        mon.stop(rest[-1].ts + 0.1)
    return list(mon.events)


def clear_levels(gaze_yaw: Spread, gaze_pitch: Spread, margin: float = 1.25,
                 floor_yaw: float = 0.04, floor_pitch: float = 0.10) -> Tuple[float, float]:
    """Lowest enter levels that still clear the resting range with `margin` to spare."""
    return (max(floor_yaw, _ceil_to(margin * gaze_yaw.max, 0.005)),
            max(floor_pitch, _ceil_to(margin * gaze_pitch.max, 0.01)))


def recommend(gaze_yaw: Spread, gaze_pitch: Spread, th: DeviationThresholds, margin: float = 1.25,
              floor_yaw: float = 0.04, floor_pitch: float = 0.10) -> DeviationThresholds:
    """Raise a threshold when the resting range needs it; never lower one automatically."""
    yaw, pitch = clear_levels(gaze_yaw, gaze_pitch, margin, floor_yaw, floor_pitch)
    return replace(th, gaze_yaw_enter=max(th.gaze_yaw_enter, yaw), gaze_pitch_enter=max(th.gaze_pitch_enter, pitch))

def quiet_stretches(events: Sequence[Event], t_from: float, t_to: float,
                    min_len: float = 8.0, pad: float = 0.5) -> List[Tuple[float, float]]:
    """Longest stretches with no event (+-pad), as absolute (start, end) timestamps."""
    spans = sorted((e.start_ts - pad, e.end_ts + pad) for e in events if e.kind != "repeated")
    out, cur = [], t_from
    for a, b in spans:
        if a - cur >= min_len:
            out.append((cur, a))
        cur = max(cur, b)
    if t_to - cur >= min_len:
        out.append((cur, t_to))
    return sorted(out, key=lambda x: x[0] - x[1])[:3]



def analyze(path, baseline_from: float = 0.0, baseline_to: Optional[float] = None,
            neutral: Optional[Tuple[float, float]] = None, th: Optional[DeviationThresholds] = None,
            cfg: Optional[EventConfig] = None, margin: float = 1.25) -> Report:
    samples = load(path)
    cal = calibration_index(samples)
    if cal is None:
        raise ValueError("no gaze samples in the recording - calibration never finished")
    th = th or DeviationThresholds(gaze_yaw_enter=0.055)
    
    neutral_pose, source = find_neutral(path, samples, cal, neutral)
    t_cal = samples[cal].ts
    t0 = t_cal + baseline_from
    t1 = t_cal + baseline_to if baseline_to is not None else samples[-1].ts + 1.0

    win = [s for s in samples if t0 <= s.ts < t1 and s.is_valid(th.min_confidence)]
    gy = [abs(s.gaze_yaw) for s in win if s.gaze_yaw is not None and s.gaze_pitch is not None]
    gp = [abs(s.gaze_pitch) for s in win if s.gaze_yaw is not None and s.gaze_pitch is not None]
    hy = [abs(s.head_yaw - neutral_pose[0]) for s in win if s.head_yaw is not None]
    hp = [abs(s.head_pitch - neutral_pose[1]) for s in win if s.head_pitch is not None]
    sp_gy, sp_gp = _spread(gy), _spread(gp)

    rec_th = recommend(sp_gy, sp_gp, th, margin)
    clear_yaw, clear_pitch = clear_levels(sp_gy, sp_gp, margin)
    cur = replay_events(samples, neutral_pose, th, cfg, cal + 1)
    rec = replay_events(samples, neutral_pose, rec_th, cfg, cal + 1)
    inside = lambda e: t0 <= e.start_ts < t1                       # noqa: E731
    dur = samples[-1].ts - samples[0].ts if len(samples) > 1 else 0.0
    return Report(
        recording=Path(path).name, frames=len(samples), duration_s=dur, start_ts=samples[0].ts,
        fps=(len(samples) - 1) / dur if dur > 0 else 0.0, calibration_ts=t_cal,
        neutral=neutral_pose, neutral_source=source, window=(t0, t1), window_frames=len(gy),
        gaze_yaw=sp_gy, gaze_pitch=sp_gp, head_yaw=_spread(hy), head_pitch=_spread(hp), th=th,
        current_events=[e for e in cur if inside(e)], rec_th=rec_th,
        rec_events=[e for e in rec if inside(e)],
        outside_events=[e for e in cur if not inside(e) and e.kind != "repeated"],
        clear_yaw=clear_yaw, clear_pitch=clear_pitch,
        stretches=[(a - t_cal, b - t_cal) for a, b in quiet_stretches(cur, t_cal, samples[-1].ts)],
    )




def format_report(r: Report) -> str:
    t = r.calibration_ts
    L = [
        f"Recording {r.recording}: {r.frames} frames, {r.duration_s:.1f} s, {r.fps:.1f} fps",
        f"Calibration ended at +{t - r.start_ts:.1f} s into the recording; neutral yaw {r.neutral[0]:+.2f} "
        f"pitch {r.neutral[1]:+.2f}  [{r.neutral_source}]",
        f"Quiet window: +{r.window[0] - t:.1f} s .. +{min(r.window[1] - t, 9999):.1f} s after calibration "
        f"({r.window_frames} frames with gaze)",
            ]
    if r.stretches:
        L.append("Event-free stretches of 8 s or more (use as --baseline): "
                 + ", ".join(f"{a:.0f}:{b:.0f}" for a, b in r.stretches))
    else:
        L.append("No event-free stretch of 8 s or more after calibration: record a dedicated quiet run "
                 "(calibrate, then read for 60 s with no deliberate looks).")
    L += ["", "Resting range                     p50      p95      p99      max"]
    for name, sp, unit in (("gaze yaw   (raw)", r.gaze_yaw, ""), ("gaze pitch (raw)", r.gaze_pitch, ""),
                           ("head yaw   (deg from neutral)", r.head_yaw, ""),
                           ("head pitch (deg from neutral)", r.head_pitch, "")):
        L.append(f"  {name:<30}{sp.p50:8.3f} {sp.p95:8.3f} {sp.p99:8.3f} {sp.max:8.3f}")
    th = r.th
    L += ["",
          f"Current thresholds: gaze yaw {th.gaze_yaw_enter}, gaze pitch {th.gaze_pitch_enter}, "
          f"head {th.head_yaw_enter}/{th.head_pitch_enter} deg",
          f"  your largest resting values are {r.gaze_yaw.max / th.gaze_yaw_enter:.2f}x (yaw) and "
          f"{r.gaze_pitch.max / th.gaze_pitch_enter:.2f}x (pitch) of the gaze thresholds, "
          f"{r.head_yaw.max / th.head_yaw_enter:.2f}x / {r.head_pitch.max / th.head_pitch_enter:.2f}x of the head ones",
          f"  events inside the quiet window at the current thresholds: {len(r.current_events)}"]
    L += [f"    - +{e.start_ts - t:.1f}s  {e.summary()}" for e in r.current_events]
    if r.current_events:
        L.append("  WARNING: if any of these were deliberate looks, the 'resting range' above is really your LOOK "
                 "and the suggestion below is meaningless. Choose a --baseline window that contains none of them.")
    L += ["",
          f"Suggested: gaze_yaw_enter {r.recommended_yaw}, gaze_pitch_enter {r.recommended_pitch}  "
          f"(never lowered automatically; your resting range would allow as low as yaw {r.clear_yaw}, "
          f"pitch {r.clear_pitch})",
          f"  events inside the quiet window with the suggestion: {len(r.rec_events)}"]
    L += [f"    - {e.summary()}" for e in r.rec_events]
    if r.outside_events:
        L += ["", "Events outside the quiet window (your deliberate looks), peak in raw units:"]
        for e in r.outside_events:
            enter = th.gaze_yaw_enter if e.source == "gaze" else th.head_yaw_enter
            L.append(f"    - {e.summary()}   (~{e.peak * enter:.3f})")
        weakest = min((e.peak * th.gaze_yaw_enter for e in r.outside_events if e.source == "gaze"), default=None)
        if weakest is not None:
            ok = weakest >= 1.15 * r.recommended_yaw
            L.append(f"  weakest deliberate gaze look {weakest:.3f} vs suggested {r.recommended_yaw}: "
                     + ("clear margin" if ok else "TOO CLOSE - look further to the side, or accept missing weak looks"))
    return "\n".join(L)


def _window(text: str) -> Tuple[float, Optional[float]]:
    a, _, b = text.partition(":")
    return float(a or 0.0), (float(b) if b else None)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--baseline", default="0:", help="quiet window in seconds after calibration, e.g. 6:66")
    ap.add_argument("--neutral", help="yaw,pitch of the calibrated neutral head pose")
    ap.add_argument("--yaw", type=float, help="gaze_yaw_enter to evaluate instead of the current one")
    ap.add_argument("--pitch", type=float, help="gaze_pitch_enter to evaluate instead of the current one")
    a = ap.parse_args(argv)
    th = DeviationThresholds(gaze_yaw_enter=a.yaw or 0.055)
    if a.pitch:
        th = replace(th, gaze_pitch_enter=a.pitch)
    neutral = tuple(float(x) for x in a.neutral.split(",")) if a.neutral else None
    b0, b1 = _window(a.baseline)
    print(format_report(analyze(a.recording, b0, b1, neutral, th)))


if __name__ == "__main__":
    main()