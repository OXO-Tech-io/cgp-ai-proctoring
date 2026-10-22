from temporal import FrameSample
from temporal.detector import DeviationThresholds, GazeDetector, HeadDetector
from temporal.hysteresis import State

NY, NP = -6.23, 6.04         
TH = DeviationThresholds()

TAIL = [  
 (64213.5138,-25.358,9.508,-0.06328,0.01168),(64213.5747,-22.182,6.857,-0.05529,0.00779),
 (64213.6751,-13.517,6.344,-0.04636,0.00157),(64213.7758,-8.925,5.022,-0.04260,0.00172),
 (64213.8843,-6.821,5.509,-0.03875,0.00570),(64213.9845,-6.710,4.745,-0.03310,0.00367),
 (64214.0752,-6.429,4.930,-0.02972,0.00732),(64214.1739,-5.714,4.863,-0.02363,0.00986),
 (64214.2804,-5.992,5.907,-0.01943,0.01139),(64214.4040,-5.722,5.253,-0.01648,0.01169),
 (64214.5054,-6.329,5.549,-0.01414,0.01187),(64214.6049,-5.486,4.332,-0.01236,0.01042),
 (64214.7148,-5.197,4.786,-0.01083,0.00890),(64214.8204,-5.326,4.118,-0.00888,0.00484),
 (64214.9058,-5.040,4.592,-0.00606,0.00241),(64215.0187,-4.653,6.040,-0.00432,-0.00086),
 (64215.1092,-4.392,4.976,-0.00328,-0.00588),(64215.2147,-3.736,4.037,-0.00270,-0.00889),
 (64215.3153,-1.088,2.802,-0.00364,-0.00561)]


def head_turn_then_tail():
    t0 = TAIL[0][0]
    s, i = [], 0
    for k in range(25):                       
        s.append(FrameSample(ts=t0 - 7.0 + k * .1, frame_idx=i, face_count=1, confidence=1,
                             head_yaw=NY, head_pitch=NP, head_roll=0, gaze_yaw=0.0, gaze_pitch=0.0)); i += 1
    for k in range(47):                       # 4.7 s head at -35, gaze gated off
        s.append(FrameSample(ts=t0 - 4.7 + k * .1, frame_idx=i, face_count=1, confidence=1,
                             head_yaw=-35.0, head_pitch=5.0, head_roll=0)); i += 1
    for ts, hy, hp, gy, gp in TAIL:
        s.append(FrameSample(ts=ts, frame_idx=i, face_count=1, confidence=1, head_yaw=hy,
                             head_pitch=hp, head_roll=0, gaze_yaw=gy, gaze_pitch=gp)); i += 1
    return s


def run(samples, use_block, settle_s=1.5):
    head, gaze = HeadDetector(NY, NP, TH), GazeDetector(TH, settle_s=settle_s)
    g_eps, h_eps = [], []
    for x in samples:
        _, he = head.update(x)
        _, ge = gaze.update(x, block=(head.state != State.NORMAL) if use_block else False)
        if he: h_eps.append(he)
        if ge: g_eps.append(ge)
    return h_eps, g_eps


def test_old_behaviour_reproduces_false_alarm():
    h, g = run(head_turn_then_tail(), use_block=False, settle_s=0.0)  
    assert len(h) == 1 and len(g) == 1
    assert abs(g[0].peak - 1.05) < 0.02       


def test_settle_rule_suppresses_tail():
    h, g = run(head_turn_then_tail(), use_block=True)
    assert len(h) == 1 and g == []


def test_real_glance_after_settle_still_detected():
    s = head_turn_then_tail()
    last = s[-1]
    i = last.frame_idx
    for k in range(1, 60):                   
        i += 1
        s.append(FrameSample(ts=last.ts + k * .1, frame_idx=i, face_count=1, confidence=1,
                             head_yaw=NY, head_pitch=NP, head_roll=0, gaze_yaw=0.0, gaze_pitch=0.0))
    base = s[-1].ts
    for k in range(1, 31):                    
        i += 1
        s.append(FrameSample(ts=base + k * .1, frame_idx=i, face_count=1, confidence=1,
                             head_yaw=NY, head_pitch=NP, head_roll=0, gaze_yaw=0.15, gaze_pitch=0.0))
    for k in range(31, 51):
        i += 1
        s.append(FrameSample(ts=base + k * .1, frame_idx=i, face_count=1, confidence=1,
                             head_yaw=NY, head_pitch=NP, head_roll=0, gaze_yaw=0.0, gaze_pitch=0.0))
    h, g = run(s, use_block=True)
    assert len(h) == 1 and len(g) == 1 and g[0].deviating_s > 2.5


def test_open_gaze_episode_is_closed_when_head_takes_over():
    gd = GazeDetector(TH)
    eps = []
    for k in range(40):
        s = FrameSample(ts=k * .1, frame_idx=k, face_count=1, confidence=1, head_yaw=NY,
                        head_pitch=NP, head_roll=0, gaze_yaw=0.15, gaze_pitch=0.0)
        _, e = gd.update(s, block=(k >= 20))
        if e: eps.append(e)
    assert len(eps) == 1 and eps[0].end_reason == "blocked"