"""Scripted eye-movement session: prompts the subject and tags each row."""

from typing import Sequence

# (target, seconds). TRANSITION rows are the first SETTLE_S of every step.
SEQUENCE = [
    ("CENTER", 3.0),
    ("LEFT", 4.0),
    ("CENTER", 3.0),
    ("RIGHT", 4.0),
    ("CENTER", 3.0),
    ("UP", 4.0),
    ("CENTER", 3.0),
    ("DOWN", 4.0),
    ("CENTER", 3.0),
]
SETTLE_S = 1.0


class GuidedSession:
    def __init__(
        self,
        sequence: Sequence[tuple[str, float]] = SEQUENCE,
        settle_s: float = SETTLE_S,
    ) -> None:
        self.sequence = sequence
        self.settle_s = settle_s
        self._start: float | None = None

    @property
    def active(self) -> bool:
        return self._start is not None

    def start(self, now: float) -> None:
        self._start = now

    def stop(self) -> None:
        self._start = None

    def update(self, now: float) -> tuple[str, str] | None:
        """Return (target, prompt) for this instant, or None when finished/inactive."""
        if self._start is None:
            return None
        t = now - self._start
        for target, dur in self.sequence:
            if t < dur:
                remaining = dur - t
                prompt = f"Look {target}  {remaining:.0f}s"
                if t < self.settle_s:
                    return "TRANSITION", prompt
                return target, prompt
            t -= dur
        self._start = None
        return None
