from temporal import FrameSample
from temporal.detector import DeviationThresholds, GazeDetector, HeadDetector
from temporal.events import EventBuilder, EventConfig

TH = DeviationThresholds(gaze_yaw_enter=0.045)
N = None

# frames 257-302: ts, gaze_yaw, gaze_pitch
LOOK_ROWS = [
 (116104.1948,-0.173496,0.074219),(116104.2764,-0.181509,0.078837),(116104.3946,-0.186317,0.081288),(116104.4897,N,N),
 (116104.6009,-0.186641,0.082758),(116104.6918,-0.186835,0.083441),(116104.7943,-0.184742,0.083851),(116104.8999,N,N),
 (116104.9780,N,N),(116105.0965,-0.181395,0.087570),(116105.1991,-0.179386,0.089801),(116105.2873,-0.181068,0.090112),
 (116105.4028,-0.179190,0.091327),(116105.4970,-0.178064,0.092055),(116105.5967,-0.178605,0.091465),(116105.7001,-0.175904,0.094134),
 (116105.8040,-0.167831,0.095736),(116105.8849,-0.161188,0.091803),(116105.9980,N,N),(116106.1001,N,N),(116106.1969,N,N),
 (116106.2883,-0.052054,0.022228),(116106.4007,-0.038891,0.017984),(116106.4950,-0.030893,0.015440),(116106.5902,-0.026195,0.014641),
 (116106.7014,-0.023375,0.014162),(116106.7899,-0.023443,0.013147),(116106.9038,-0.023484,0.014120),(116107.0113,-0.024523,0.011222),
 (116107.0735,-0.025146,0.008427),(116107.1736,-0.025520,0.003512),(116107.2729,-0.025745,0.000333),(116107.3733,-0.025879,-0.001574),
 (116107.4761,-0.025573,-0.001054)] + [(116107.5 + 0.1 * k, N, N) for k in range(9)] + [
 (116108.5129,-0.024327,-0.000743),(116108.5991,N,N),(116108.7033,-0.005046,0.015487)]


def _events(rows, cfg):
    t0, t1 = rows[0][0], rows[-1][0]
    allrows = ([(t0 - 2 + 0.1 * k, 0.0, 0.0) for k in range(20)] + rows
               + [(t1 + 0.1 * (k + 1), 0.0, 0.0) for k in range(25)])
    head, gaze, gb = HeadDetector(-2.67, 1.17, TH), GazeDetector(TH), EventBuilder.for_gaze(TH, cfg)
    blocks = getattr(head, "block_gaze", None) or head.blocks_gaze
    out = []
    for i, (t, gy, gp) in enumerate(allrows):
        s = FrameSample(ts=t, frame_idx=i, face_count=1, confidence=1.0, head_yaw=-2.67, head_pitch=1.17,
                        head_roll=0.0, gaze_yaw=gy, gaze_pitch=gp)
        head.update(s)
        tr, ep = gaze.update(s, block=blocks(), head_score=head.last_score)
        out += gb.update(s, tr, ep)
    return out


def test_resting_offset_after_a_look_is_not_part_of_the_look():
    old = _events(LOOK_ROWS, EventConfig(tail_frac=0.0))
    assert len(old) == 1 and old[0].kind == "sustained" and old[0].duration_s > 4.0      # what you saw
    new = _events(LOOK_ROWS, EventConfig())
    assert len(new) == 1 and new[0].direction == "yaw-" and new[0].kind == "glance"
    assert 1.4 < new[0].duration_s < 2.2