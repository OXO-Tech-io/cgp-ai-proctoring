from dataclasses import replace

from temporal import FrameSample
from temporal.detector import DeviationThresholds, GazeDetector, HeadDetector
from temporal.hysteresis import State

NY, NP = -3.42, -1.55
TH = DeviationThresholds()
N = None

TILT_GLANCE = [ 
 (87439.4371,-3.922,-9.974,-0.00141,-0.09101),(87439.5357,-4.560,-8.893,0.00076,-0.09480),
 (87439.6492,-4.439,-9.493,0.00206,-0.09687),(87439.7292,-3.943,-9.633,-0.00066,-0.08744),
 (87439.8454,-4.079,-9.479,-0.00241,-0.08023),(87439.9352,-4.593,-8.822,-0.00346,-0.07591),
 (87440.0481,-4.648,-8.413,-0.00478,-0.07308),(87440.1310,-4.680,-8.602,-0.00558,-0.07161),
 (87440.2431,-5.036,-9.543,-0.00836,-0.06944),(87440.3461,-5.856,-10.275,-0.01045,-0.06606),
 (87440.4456,-5.823,-11.134,-0.01170,-0.06403),(87440.5295,-6.448,-10.814,-0.02592,-0.06281),
 (87440.6421,-6.620,-11.763,-0.04199,-0.05729),(87440.7346,-6.253,-9.959,-0.05473,-0.04856),
 (87440.8467,-6.383,-10.975,-0.06336,-0.04875),(87440.9456,-5.941,-9.883,-0.06928,-0.04886),
 (87441.0345,-6.248,-10.858,-0.07648,-0.04981),(87441.1414,-5.939,-10.529,-0.08081,-0.05038),
 (87441.2260,-5.496,-9.008,-0.08341,-0.05072),(87441.3483,-6.615,-9.819,-0.08505,-0.04819),
 (87441.4374,-6.209,-9.583,-0.08603,-0.04667),(87441.5363,-5.977,-8.672,-0.08662,-0.04266),
 (87441.6460,-5.031,-8.821,-0.08412,-0.03736),(87441.7375,-5.111,-8.990,-0.08262,-0.03418),
 (87441.8387,-4.944,-9.107,N,N),(87441.9360,-4.704,-8.295,-0.04774,-0.04731),
 (87442.0376,-4.654,-8.425,-0.03897,-0.05100),(87442.1485,-5.077,-9.350,-0.03115,-0.05349),
 (87442.2328,-5.303,-8.663,-0.02512,-0.05497),(87442.3482,-5.387,-9.179,-0.02149,-0.05360),
 (87442.4386,-4.774,-10.830,-0.01932,-0.05277),(87442.5391,-5.088,-10.839,-0.01454,-0.05556)]


def _samples(rows, pre_quiet=2.0):
    out, i, t0 = [], 0, rows[0][0]
    for k in range(int(pre_quiet * 10)):
        out.append(FrameSample(ts=t0 - pre_quiet + k * .1, frame_idx=i, face_count=1, confidence=1,
                               head_yaw=NY, head_pitch=NP, head_roll=0, gaze_yaw=0.0, gaze_pitch=0.0)); i += 1
    for ts, hy, hp, gy, gp in rows:
        out.append(FrameSample(ts=ts, frame_idx=i, face_count=1, confidence=1, head_yaw=hy,
                               head_pitch=hp, head_roll=0, gaze_yaw=gy, gaze_pitch=gp)); i += 1
    return out


def _gaze_transitions(samples, th):
    head, gaze = HeadDetector(NY, NP, th), GazeDetector(th)
    out = []
    for x in samples:
        head.update(x)
        tr, _ = gaze.update(x, block=head.block_gaze(), head_score=head.last_score)
        out += tr
    return out


def test_eye_movement_with_mild_head_tilt_is_not_swallowed():
    s = _samples(TILT_GLANCE)
    old = _gaze_transitions(s, replace(TH, head_block_ratio=0.6))
    assert State.DEVIATING not in [t.new for t in old]       # reproduces the swallowed movement
    new = _gaze_transitions(s, TH)
    assert State.DEVIATING in [t.new for t in new] 