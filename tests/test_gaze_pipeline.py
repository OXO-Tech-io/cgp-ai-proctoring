from types import SimpleNamespace as P

from vision.gaze import landmark_mapping as m
from vision.landmark_indices import LEFT_IRIS_INDICES, RIGHT_IRIS_INDICES
from vision.gaze.gaze_calibrator import GazeCalibrator
from vision.gaze.gaze_pipeline import GazePipeline, classify_gaze
from vision.gaze.head_pose_compensator import HeadPoseCompensator


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_landmarks(iris_dx=0.0, open_h=0.02):
    """Fake 478 landmarks: eye width 0.05, iris at box center + iris_dx*width."""
    lm = [P(x=0.0, y=0.0) for _ in range(478)]

    def eye(inner, outer, top, bottom, iris, cx):
        lm[inner] = P(x=cx - 0.025, y=0.5)
        lm[outer] = P(x=cx + 0.025, y=0.5)
        lm[top] = P(x=cx, y=0.5 - open_h / 2)
        lm[bottom] = P(x=cx, y=0.5 + open_h / 2)
        for i in iris:
            lm[i] = P(x=cx + iris_dx * 0.05, y=0.5)

    eye(m.LEFT_EYE_INNER_CORNER, m.LEFT_EYE_OUTER_CORNER, m.LEFT_EYE_TOP,
        m.LEFT_EYE_BOTTOM, LEFT_IRIS_INDICES, 0.4)
    eye(m.RIGHT_EYE_INNER_CORNER, m.RIGHT_EYE_OUTER_CORNER, m.RIGHT_EYE_TOP,
        m.RIGHT_EYE_BOTTOM, RIGHT_IRIS_INDICES, 0.6)
    return lm


def calibrated_pipeline():
    clock = Clock()
    p = GazePipeline(clock=clock)
    assert p.update(make_landmarks(), 0, 0).message.startswith("Calibrating in")
    clock.t = 5.0
    for _ in range(50):
        p.update(make_landmarks(), 0.0, 0.0)
    assert p.calibrated
    return p


def test_calibration_countdown_then_complete():
    calibrated_pipeline()


def test_tracking_center_and_right():
    p = calibrated_pipeline()
    r = p.update(make_landmarks(0.0), 0, 0)
    assert r.status == "TRACKING" and r.label == "CENTER"
    for _ in range(6):
        r = p.update(make_landmarks(0.4), 0, 0)
    assert r.label == "RIGHT"


def test_head_away_gate():
    p = calibrated_pipeline()
    r = p.update(make_landmarks(), 30.0, 0)
    assert r.status == "HEAD_AWAY" and r.label == "HEAD_AWAY" and not r.reliable


def test_blink_skipped():
    p = calibrated_pipeline()
    assert p.update(make_landmarks(open_h=0.001), 0, 0).status == "BLINK"


def test_head_drift_fails_calibration():
    clock = Clock()
    clock.t = 5.0
    p = GazePipeline(clock=clock)
    p._calib_start_at = 0.0
    for i in range(50):
        p.update(make_landmarks(), i * 0.5, 0.0)  # yaw drifts 25 deg
    assert not p.calibrated


def test_calibrator_head_spread():
    c = GazeCalibrator(required_samples=3)
    for y in (0.0, 10.0, 20.0):
        c.add_samples(0.5, 0.5, 0.5, 0.5, y, 0.0)
    assert c.is_stable(0.02)
    assert not c.is_stable(0.02, head_threshold=2.0)


def test_compensator_sign_and_clamp():
    c = HeadPoseCompensator(yaw_gain=0.01, pitch_gain=0.0, max_angle=10.0)
    assert c.compensate(0.0, 0.0, 0.0, 0.0) == (0.0, 0.0)
    assert c.compensate(0.0, 0.0, 5.0, 0.0)[0] == -0.05
    assert c.compensate(0.0, 0.0, 50.0, 0.0)[0] == -0.1


def test_classify():
    assert classify_gaze(0, 0) == "CENTER"
    assert classify_gaze(0.1, 0) == "RIGHT"
    assert classify_gaze(-0.1, 0) == "LEFT"
    assert classify_gaze(0, 0.5) == "CENTER"  # vertical disabled
    assert classify_gaze(0, 0.5, enable_vertical=True) == "DOWN"
    assert classify_gaze(0, -0.5, enable_vertical=True) == "UP"
