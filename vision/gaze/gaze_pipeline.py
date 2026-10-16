import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from config import gaze_config as cfg
from vision.gaze.eye_normalizer import EyeNormalizer, is_eye_open
from vision.gaze.gaze_calibrator import GazeCalibrator
from vision.gaze.gaze_filter import GazeFilter
from vision.gaze.head_pose_compensator import HeadPoseCompensator
from vision.gaze.iris_tracker import IrisExtractor
from vision.gaze.types import Landmark
from vision.gaze.landmark_mapping import (
    LEFT_EYE_INNER_CORNER,
    LEFT_EYE_OUTER_CORNER,
    LEFT_EYE_TOP,
    LEFT_EYE_BOTTOM,
    RIGHT_EYE_INNER_CORNER,
    RIGHT_EYE_OUTER_CORNER,
    RIGHT_EYE_TOP,
    RIGHT_EYE_BOTTOM,
)
from vision.landmark_indices import LEFT_IRIS_INDICES, RIGHT_IRIS_INDICES


def classify_gaze(
    dx: float,
    dy: float,
    tx: float = cfg.TX,
    ty: float = cfg.TY,
    enable_vertical: bool = cfg.ENABLE_VERTICAL,
) -> str:
    """Map filtered gaze displacement to CENTER/LEFT/RIGHT/UP/DOWN."""
    if not enable_vertical:
        dy = 0.0
    if abs(dx) < tx and abs(dy) < ty:
        return "CENTER"
    if abs(dx) / tx >= abs(dy) / ty:
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


@dataclass
class CalibrationStatus:
    """Where calibration is right now; the caller decides how to show it.

    phase: WAITING (countdown before sampling) or COLLECTING.
    """

    phase: str
    remaining_s: float = 0.0
    samples: int = 0
    required: int = 0


@dataclass
class CalibrationEvent:
    """One-off calibration milestone for the caller to log or print.

    kind: SAMPLE, SPREAD, UNSTABLE, or COMPLETE. `data` holds the numbers:
      SAMPLE   samples, required
      SPREAD   left_x, left_y, right_x, right_y, head_yaw, head_pitch
      UNSTABLE iris_threshold, head_threshold
      COMPLETE the baseline dict (neutral iris positions and neutral pose)
    """

    kind: str
    data: dict[str, float] = field(default_factory=dict)


@dataclass
class GazeResult:
    """Output of one pipeline update.

    status: CALIBRATING (waiting or collecting), BLINK, HEAD_AWAY, or TRACKING.
    The numeric fields are None until gaze is actually estimated.
    """

    status: str
    calibrated: bool
    calibration: CalibrationStatus | None = None
    events: list[CalibrationEvent] = field(default_factory=list)
    label: str | None = None
    reliable: bool = False
    left_iris: tuple[float, float] | None = None
    right_iris: tuple[float, float] | None = None
    left_normalized: tuple[float, float] | None = None
    right_normalized: tuple[float, float] | None = None
    left_offset: tuple[float, float] | None = None
    right_offset: tuple[float, float] | None = None
    left_compensated: tuple[float, float] | None = None
    right_compensated: tuple[float, float] | None = None
    gaze: tuple[float, float] | None = None
    filtered: tuple[float, float] | None = None
    delta_yaw: float | None = None
    delta_pitch: float | None = None

    @property
    def has_gaze(self) -> bool:
        """True when numeric gaze values were produced (loggable)."""
        return self.status in ("TRACKING", "HEAD_AWAY") and self.gaze is not None


class GazePipeline:
    """Landmarks + head pose -> calibrated, filtered gaze label.

    Holds all calibration/filter state so it can run without a webcam.
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._iris = IrisExtractor(LEFT_IRIS_INDICES, RIGHT_IRIS_INDICES)
        self._filter = GazeFilter(
            window=cfg.FILTER_WINDOW,
            alpha=cfg.FILTER_ALPHA,
            max_step=cfg.FILTER_MAX_STEP,
            max_divergence=cfg.FILTER_MAX_DIVERGENCE,
        )
        self._compensator = HeadPoseCompensator(
            yaw_gain_pos=cfg.YAW_GAIN_POS,
            yaw_gain_neg=cfg.YAW_GAIN_NEG,
            pitch_gain_pos=cfg.PITCH_GAIN_POS,
            pitch_gain_neg=cfg.PITCH_GAIN_NEG,
            max_angle=cfg.COMP_MAX_ANGLE_DEG,
        )
        self._calibrator = GazeCalibrator(required_samples=cfg.CALIB_SAMPLES)
        self._baseline: dict[str, float] | None = None
        self._calib_start_at = self._clock() + cfg.CALIB_DELAY_S

    @property
    def calibrated(self) -> bool:
        return self._baseline is not None

    # ------------------------------------------------------------------
    # Calibration control
    # ------------------------------------------------------------------

    def start_calibration(self) -> None:
        """Discard the baseline and restart the countdown."""
        self._calibrator = GazeCalibrator(required_samples=cfg.CALIB_SAMPLES)
        self._calib_start_at = self._clock() + cfg.CALIB_DELAY_S
        self._filter.reset()
        self._baseline = None

    def on_face_lost(self) -> None:
        """Drop partial calibration samples when the face disappears."""
        if not self.calibrated and self._calibrator.sample_count() > 0:
            self.start_calibration()

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(
        self, landmarks: Sequence[Landmark], yaw: float, pitch: float
    ) -> GazeResult:
        left_inner = landmarks[LEFT_EYE_INNER_CORNER]
        left_outer = landmarks[LEFT_EYE_OUTER_CORNER]
        left_top = landmarks[LEFT_EYE_TOP]
        left_bottom = landmarks[LEFT_EYE_BOTTOM]
        right_inner = landmarks[RIGHT_EYE_INNER_CORNER]
        right_outer = landmarks[RIGHT_EYE_OUTER_CORNER]
        right_top = landmarks[RIGHT_EYE_TOP]
        right_bottom = landmarks[RIGHT_EYE_BOTTOM]

        calibration = self._calibration_status()

        if not (
            is_eye_open(left_inner.x, left_outer.x, left_top.y, left_bottom.y)
            and is_eye_open(right_inner.x, right_outer.x, right_top.y, right_bottom.y)
        ):
            return GazeResult("BLINK", self.calibrated, calibration=calibration)

        left_iris, right_iris = self._iris.extract(landmarks)
        left_center = self._iris.calculate_center(left_iris)
        right_center = self._iris.calculate_center(right_iris)

        left_norm = EyeNormalizer.normalize(
            left_center[0], left_center[1],
            left_inner.x, left_outer.x, left_top.y, left_bottom.y,
        )
        right_norm = EyeNormalizer.normalize(
            right_center[0], right_center[1],
            right_inner.x, right_outer.x, right_top.y, right_bottom.y,
        )

        events: list[CalibrationEvent] = []
        if not self.calibrated:
            if self._clock() >= self._calib_start_at:
                events = self._collect_calibration(left_norm, right_norm, yaw, pitch)
            if not self.calibrated:
                return GazeResult(
                    "CALIBRATING",
                    False,
                    calibration=self._calibration_status(),
                    events=events,
                )
            # Calibration just finished; start estimating on the next frame.
            return GazeResult("CALIBRATING", True, events=events)

        return self._estimate(
            left_center, right_center, left_norm, right_norm, yaw, pitch
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _calibration_status(self) -> CalibrationStatus | None:
        if self.calibrated:
            return None
        remaining = self._calib_start_at - self._clock()
        if remaining > 0:
            return CalibrationStatus("WAITING", remaining_s=remaining)
        return CalibrationStatus(
            "COLLECTING",
            samples=self._calibrator.sample_count(),
            required=cfg.CALIB_SAMPLES,
        )

    def _collect_calibration(
        self,
        left_norm: tuple[float, float],
        right_norm: tuple[float, float],
        yaw: float,
        pitch: float,
    ) -> list[CalibrationEvent]:
        cal = self._calibrator
        cal.add_samples(
            left_norm[0], left_norm[1], right_norm[0], right_norm[1], yaw, pitch
        )
        events = [
            CalibrationEvent(
                "SAMPLE",
                {"samples": cal.sample_count(), "required": cfg.CALIB_SAMPLES},
            )
        ]

        if not cal.is_ready():
            return events

        spread = cal.get_spread()
        head = cal.get_head_spread()
        events.append(
            CalibrationEvent(
                "SPREAD",
                {
                    "left_x": spread["left_x"],
                    "left_y": spread["left_y"],
                    "right_x": spread["right_x"],
                    "right_y": spread["right_y"],
                    "head_yaw": head["yaw"],
                    "head_pitch": head["pitch"],
                },
            )
        )

        if not cal.is_stable(
            threshold=cfg.CALIB_MAX_IRIS_STD,
            head_threshold=cfg.CALIB_MAX_HEAD_STD_DEG,
        ):
            events.append(
                CalibrationEvent(
                    "UNSTABLE",
                    {
                        "iris_threshold": cfg.CALIB_MAX_IRIS_STD,
                        "head_threshold": cfg.CALIB_MAX_HEAD_STD_DEG,
                    },
                )
            )
            self.start_calibration()
            return events

        self._baseline = cal.get_calibration()
        events.append(CalibrationEvent("COMPLETE", dict(self._baseline)))
        return events

    def _estimate(
        self,
        left_center: tuple[float, float],
        right_center: tuple[float, float],
        left_norm: tuple[float, float],
        right_norm: tuple[float, float],
        yaw: float,
        pitch: float,
    ) -> GazeResult:
        b = self._baseline
        assert b is not None

        left_off = (left_norm[0] - b["left_neutral_x"], left_norm[1] - b["left_neutral_y"])
        right_off = (right_norm[0] - b["right_neutral_x"], right_norm[1] - b["right_neutral_y"])
        raw = ((left_off[0] + right_off[0]) / 2, (left_off[1] + right_off[1]) / 2)

        delta_yaw = yaw - b["neutral_yaw"]
        delta_pitch = pitch - b["neutral_pitch"]

        common: dict[str, Any] = dict(
            calibrated=True,
            left_iris=left_center,
            right_iris=right_center,
            left_normalized=left_norm,
            right_normalized=right_norm,
            left_offset=left_off,
            right_offset=right_off,
            delta_yaw=delta_yaw,
            delta_pitch=delta_pitch,
        )

        if abs(delta_yaw) > cfg.MAX_YAW_DEG or abs(delta_pitch) > cfg.MAX_PITCH_DEG:
            self._filter.reset()
            return GazeResult(
                "HEAD_AWAY",
                label="HEAD_AWAY",
                reliable=False,
                left_compensated=left_off,   # uncompensated: gate skips compensation
                right_compensated=right_off,
                gaze=raw,
                filtered=(0.0, 0.0),
                **common,
            )

        left_comp = self._compensator.compensate(*left_off, delta_yaw, delta_pitch)
        right_comp = self._compensator.compensate(*right_off, delta_yaw, delta_pitch)
        gaze = ((left_comp[0] + right_comp[0]) / 2, (left_comp[1] + right_comp[1]) / 2)

        fx, fy, reliable = self._filter.update(gaze[0], gaze[1], left_comp[0], right_comp[0])

        return GazeResult(
            "TRACKING",
            label=classify_gaze(fx, fy),
            reliable=reliable,
            left_compensated=left_comp,
            right_compensated=right_comp,
            gaze=gaze,
            filtered=(fx, fy),
            **common,
        )
