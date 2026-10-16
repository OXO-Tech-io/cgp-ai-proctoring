from collections import deque
from statistics import median


class GazeFilter:
    """Median + EMA filter for compensated gaze with divergence guard."""

    def __init__(
        self,
        window: int = 5,
        alpha: float = 0.4,
        max_step: float = 0.05,
        max_divergence: float = 0.08,
    ) -> None:
        self.window = window
        self.alpha = alpha
        self.max_step = max_step
        self.max_divergence = max_divergence
        self._dx = deque(maxlen=window)
        self._dy = deque(maxlen=window)
        self._smooth_x: float | None = None
        self._smooth_y: float | None = None

    def reset(self) -> None:
        self._dx.clear()
        self._dy.clear()
        self._smooth_x = None
        self._smooth_y = None

    def update(
        self,
        dx: float,
        dy: float,
        left_dx: float | None = None,
        right_dx: float | None = None,
    ) -> tuple[float, float, bool]:
        """Filter one sample. Returns (filt_x, filt_y, reliable)."""

        reliable = True

        if (
            left_dx is not None
            and right_dx is not None
            and abs(left_dx - right_dx) > self.max_divergence
        ):
            reliable = False
            if self._smooth_x is not None:
                return self._smooth_x, self._smooth_y or 0.0, False

        self._dx.append(dx)
        self._dy.append(dy)

        med_x = median(self._dx)
        med_y = median(self._dy)

        if self._smooth_x is None:
            self._smooth_x = med_x
            self._smooth_y = med_y
        else:
            assert self._smooth_y is not None
            if abs(med_x - self._smooth_x) > self.max_step:
                reliable = False
            if abs(med_y - self._smooth_y) > self.max_step:
                reliable = False
            self._smooth_x = (
                self.alpha * med_x + (1.0 - self.alpha) * self._smooth_x
            )
            self._smooth_y = (
                self.alpha * med_y + (1.0 - self.alpha) * self._smooth_y
            )

        assert self._smooth_x is not None and self._smooth_y is not None
        return self._smooth_x, self._smooth_y, reliable
