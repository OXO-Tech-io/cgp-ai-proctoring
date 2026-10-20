from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Any, Optional

SCHEMA_VERSION = 1

NUMERIC_SIGNALS = ("head_yaw", "head_pitch", "head_roll", "gaze_yaw", "gaze_pitch")

@dataclass(frozen=True, slots=True)
class FrameSample:
    ts: float
    frame_idx: int
    face_count: int
    confidence: float
    head_yaw: Optional[float] = None
    head_pitch: Optional[float] = None
    head_roll: Optional[float] = None
    gaze_yaw: Optional[float] = None
    gaze_pitch: Optional[float] = None

    @property
    def face_present(self) -> bool:
        return self.face_count > 0

    def is_valid(self, min_confidence: float = 0.5) -> bool:
        """True if this frame carries trustworthy pose data for the primary face."""
        return self.face_present and self.confidence >= min_confidence

    def get(self, name: str) -> Optional[float]:
        if name not in NUMERIC_SIGNALS:
            raise KeyError(f"Unknown signal '{name}'. Expcted one of {NUMERIC_SIGNALS}")
        return getattr(self, name)
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FrameSample":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})