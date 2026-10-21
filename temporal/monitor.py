from typing import Callable, List, Optional

from .detector import DeviationThresholds, GazeDetector, HeadDetector
from .events import Event, EventBuilder, EventConfig, RepeatTracker
from .signals import FrameSample


class TemporalMonitor:
    def __init__(self, thresholds: Optional[DeviationThresholds] = None,
                 emit: Callable[[str], None] = print, event_cfg: Optional[EventConfig] = None):
        self.th = thresholds or DeviationThresholds()
        self.cfg = event_cfg or EventConfig()
        self.emit = emit
        self.repeats = RepeatTracker(self.cfg)
        self.events: List[Event] = []          # everything emitted this session (3.7 will write this out)
        self.reset()

    def reset(self) -> None:
        self.head = self.gaze = self.head_events = self.gaze_events = None
        self.repeats = RepeatTracker(self.cfg)

    @property
    def ready(self) -> bool:
        return self.head is not None and self.gaze is not None

    def start(self, neutral_yaw: float, neutral_pitch: float) -> None:
        self.head = HeadDetector(neutral_yaw, neutral_pitch, self.th)
        self.gaze = GazeDetector(self.th)
        self.head_events = EventBuilder.for_head(neutral_yaw, neutral_pitch, self.th, self.cfg)
        self.gaze_events = EventBuilder.for_gaze(self.th, self.cfg)

    def update(self, sample: FrameSample) -> List[Event]:
        if not self.ready:
            return []
        blocks = getattr(self.head, "block_gaze", None) or self.head.blocks_gaze
        h = self.head.update(sample)
        g = self.gaze.update(sample, block=blocks(), head_score=self.head.last_score)
        return self._handle((("HEAD", h, self.head_events), ("GAZE", g, self.gaze_events)), sample)

    def stop(self, ts: float) -> List[Event]:
        if not self.ready:
            return []
        out = self._handle((("HEAD", self.head.finish(ts), self.head_events),
                            ("GAZE", self.gaze.finish(ts), self.gaze_events)), None)
        self.reset()
        return out

    def _handle(self, parts, sample) -> List[Event]:
        emitted: List[Event] = []
        for name, (transitions, episode), builder in parts:
            for t in transitions:
                self.emit(f"[{name}] {t.old.name} -> {t.new.name} ({t.reason})")
            if episode is not None:
                self.emit(f"[{name}] EPISODE {episode.end_reason}: {episode.deviating_s:.1f}s deviating, "
                          f"peak {episode.peak:.2f}x threshold")
            for ev in builder.update(sample, transitions, episode):
                for out in self.repeats.add(ev):
                    self.emit(f"[EVENT] {out.summary()}")
                    self.events.append(out)
                    emitted.append(out)
        return emitted