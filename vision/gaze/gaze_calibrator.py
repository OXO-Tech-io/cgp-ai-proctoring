from collections import deque
from statistics import median, stdev

class GazeCalibrator:
    """Collect neutral gaze samples and calculate a personal baseline."""

    def __init__(self, required_samples: int = 50) -> None:
        self.required_samples = required_samples
        self._samples = deque(maxlen=required_samples)

    def add_samples(
        self,
        left_x: float,
        left_y: float,
        right_x: float,
        right_y: float,
        yaw: float = 0.0,
        pitch: float = 0.0,
    ) -> None:
        """Add one neutral gaze sample. """

        self._samples.append(
            {
                "left_x": left_x,
                "left_y": left_y,
                "right_x": right_x,
                "right_y": right_y,
                "yaw": yaw,
                "pitch": pitch,
            }
        )

    def is_ready(self) -> bool:
        """Return True when enough samples have been collected."""

        return len(self._samples) >= self.required_samples

    def sample_count(self) -> int:
        """Return the number of collected samples."""

        return len(self._samples)

    def get_calibration(self) -> dict[str, float]:
        """Calculate the median neutral gaze position."""

        if not self.is_ready():
            raise ValueError(
                f"Calibration requires {self.required_samples} samples. "
                f"Only {len(self._samples)} collected."
            )

        return {
            "left_neutral_x": median(
                sample["left_x"] for sample in self._samples
            ),
            "left_neutral_y": median(
                sample["left_y"] for sample in self._samples
            ),
            "right_neutral_x": median(
                sample["right_x"] for sample in self._samples
            ),
            "right_neutral_y": median(
                sample["right_y"] for sample in self._samples
            ),
            "neutral_yaw": median(
                sample.get("yaw", 0.0) for sample in self._samples
            ),
            "neutral_pitch": median(
                sample.get("pitch", 0.0) for sample in self._samples
            ),
        }

    def get_spread(self) -> dict[str, float]:
        """Return per-channel stdev of collected samples."""

        if len(self._samples) < 2:
            return {
                "left_x": 0.0,
                "left_y": 0.0,
                "right_x": 0.0,
                "right_y": 0.0,
                "max": 0.0,
            }

        spreads = {
            "left_x": stdev(
                sample["left_x"] for sample in self._samples
            ),
            "left_y": stdev(
                sample["left_y"] for sample in self._samples
            ),
            "right_x": stdev(
                sample["right_x"] for sample in self._samples
            ),
            "right_y": stdev(
                sample["right_y"] for sample in self._samples
            ),
        }
        spreads["max"] = max(
            spreads["left_x"],
            spreads["left_y"],
            spreads["right_x"],
            spreads["right_y"],
        )
        return spreads

    def get_head_spread(self) -> dict[str, float]:
        """Return stdev (degrees) of yaw and pitch of collected samples."""

        if len(self._samples) < 2:
            return {"yaw": 0.0, "pitch": 0.0, "max": 0.0}

        yaw = stdev(sample["yaw"] for sample in self._samples)
        pitch = stdev(sample["pitch"] for sample in self._samples)
        return {"yaw": yaw, "pitch": pitch, "max": max(yaw, pitch)}

    def is_stable(
        self,
        threshold: float = 0.02,
        head_threshold: float | None = None,
    ) -> bool:
        """Return True when iris spread (and head-pose spread, if a
        head_threshold in degrees is given) is within threshold."""

        if self.get_spread()["max"] > threshold:
            return False
        if head_threshold is not None:
            return self.get_head_spread()["max"] <= head_threshold
        return True