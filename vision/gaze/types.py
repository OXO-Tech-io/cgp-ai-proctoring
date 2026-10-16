from typing import Protocol


class Landmark(Protocol):
    """Anything with normalized image coordinates (MediaPipe landmark, test fake)."""

    x: float
    y: float
