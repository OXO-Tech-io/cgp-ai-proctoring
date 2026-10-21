from temporal import FrameSample
from temporal.detector import DeviationThresholds, GazeDetector, HeadDetector
from temporal.events import EventBuilder, EventConfig

TH = DeviationThresholds(gaze_yaw_enter=0.045)


def _events(cfg):
    rows = [(0.1 * k, 0.0, 0.0) for k in range(20)]
    look = [(2.0 + 0.1 * k, -0.0585, 0.0) for k in range(10)]
    look[4] = (look[4][0], -0.192, 0.0)                                  # single-frame spike, 4.27x
    rows += look + [(3.0 + 0.1 * k, 0.0, 0.0) for k in range(25)]
    head, gaze, gb = HeadDetector(0.0, 0.0, TH), GazeDetector(TH), EventBuilder.for_gaze(TH, cfg)
    blocks = getattr(head, "block_gaze", None) or head.blocks_gaze
    out = []
    for i, (t, gy, gp) in enumerate(rows):
        s = FrameSample(ts=t, frame_idx=i, face_count=1, confidence=1.0, head_yaw=0.0, head_pitch=0.0,
                        head_roll=0.0, gaze_yaw=gy, gaze_pitch=gp)
        head.update(s)
        tr, ep = gaze.update(s, block=blocks(), head_score=head.last_score)
        out += gb.update(s, tr, ep)
    return out


def test_a_single_frame_spike_does_not_collapse_the_look():
    ev = _events(EventConfig())
    assert len(ev) == 1 and ev[0].peak > 4.0
    assert 0.7 < ev[0].duration_s < 1.1, ev[0]