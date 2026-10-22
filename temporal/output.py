from __future__ import annotations

import json
import time
from bisect import bisect_left
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from .detector import DeviationThresholds
from .events import Event, EventConfig

SCHEMA_VERSION = 1

# Head signs come from turn_state() in main.py (the on-screen text).
HEAD_LABELS = {"yaw-": "left", "yaw+": "right", "pitch-": "up", "pitch+": "down"}
# Eye signs have not been verified against the camera yet. Fill in e.g. {"yaw+": "right", ...}.
GAZE_LABELS: dict = {}


def events_path_for(recording_path) -> Path:
    p = Path(recording_path)
    return p.with_name(p.stem + ".events.json")


def _load_frames(recording_path) -> list:
    """[(ts, frame_idx)] for every sample in the recording, in order."""
    out = []
    try:
        with Path(recording_path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue            # a half-written last line must not break the report
                if rec.get("type") == "sample":
                    out.append((rec["ts"], rec["frame_idx"]))
    except OSError:
        pass
    return out


def _nearest_frame(frames: list, ts: float) -> Optional[int]:
    if not frames:
        return None
    i = bisect_left(frames, (ts, -1))
    if i == 0:
        return frames[0][1]
    if i == len(frames):
        return frames[-1][1]
    before, after = frames[i - 1], frames[i]
    return before[1] if ts - before[0] <= after[0] - ts else after[1]


class EventLog:
    def __init__(self, recording_path, session_id: str,
                 thresholds: Optional[DeviationThresholds] = None,
                 event_cfg: Optional[EventConfig] = None, meta: Optional[dict] = None):
        self.recording_path = Path(recording_path)
        self.session_id = session_id
        self.th = thresholds or DeviationThresholds()
        self.cfg = event_cfg or EventConfig()
        self.meta = meta or {}
        self.events: List[Event] = []
        self.calibrations: List[dict] = []
        self.started_at = time.time()

    def add(self, event: Event) -> None:
        self.events.append(event)

    def add_calibration(self, neutral_yaw: float, neutral_pitch: float, ts: Optional[float] = None) -> None:
        self.calibrations.append({"neutral_yaw": neutral_yaw, "neutral_pitch": neutral_pitch, "ts": ts})

    def build(self) -> dict:
        frames = _load_frames(self.recording_path)
        warnings: List[str] = []
        if not self.recording_path.exists():
            warnings.append(f"recording file not found: {self.recording_path}")
        elif not frames:
            warnings.append("recording has no samples (file empty, or read before it was closed)")
        t0 = frames[0][0] if frames else (min((e.start_ts for e in self.events), default=0.0))
        t1 = frames[-1][0] if frames else (max((e.end_ts for e in self.events), default=t0))
        rows = []
        for i, e in enumerate(sorted(self.events, key=lambda ev: (ev.start_ts, ev.source))):
            labels = HEAD_LABELS if e.source == "head" else GAZE_LABELS
            rows.append({
                "id": i,
                "source": e.source,
                "kind": e.kind,
                "direction": e.direction,
                "label": labels.get(e.direction),
                "start_s": round(e.start_ts - t0, 3),
                "end_s": round(e.end_ts - t0, 3),
                "duration_s": round(e.duration_s, 3),
                "peak": round(e.peak, 3),
                "count": e.count,
                "end_reason": e.end_reason,
                "start_ts": e.start_ts,
                "end_ts": e.end_ts,
                "start_frame": _nearest_frame(frames, e.start_ts),
                "end_frame": _nearest_frame(frames, e.end_ts),
            })
        by_source: dict = {}
        by_kind: dict = {}
        for r in rows:
            by_source[r["source"]] = by_source.get(r["source"], 0) + 1
            by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "recording": self.recording_path.name,
            "written_at": time.time(),
            "duration_s": round(t1 - t0, 3),
            "frames": len(frames),
            "meta": self.meta,
            "thresholds": asdict(self.th),
            "event_config": asdict(self.cfg),
            "calibrations": self.calibrations,
            "warnings": warnings,
            "summary": {"total": len(rows), "by_source": by_source, "by_kind": by_kind},
            "events": rows,
        }

    def write(self) -> Path:
        """Call AFTER the recorder is closed so the frame numbers can be looked up."""
        path = events_path_for(self.recording_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        doc = self.build()
        for w in doc["warnings"]:
            print(f"[WARN] {w}")
        tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        tmp.replace(path)
        return path