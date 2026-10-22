"""Gaze episode boundaries (real session, neutral yaw +0.43 / pitch +4.07):
 * a look followed by a pitch sag (0.07-0.11) was reported as 4.8 s instead of ~2 s
 * two looks separated by a long blind gap were glued into one 11.4 s episode"""
from dataclasses import replace

from temporal import FrameSample, ScenarioBuilder
from temporal.detector import DeviationThresholds, GazeDetector, HeadDetector
from temporal.events import EventBuilder

TH = DeviationThresholds()
NY, NP = 0.43, 4.07
N = None

# frames 553-610: eyes to -yaw (peak -0.156), then gaze_pitch sags to 0.05-0.11 for ~3 s
SAG_ROWS = [  # ts, gaze_yaw, gaze_pitch
 (111937.4316,N,N),(111937.5453,-0.092515,0.026378),(111937.6153,N,N),(111937.7459,N,N),(111937.8251,-0.102288,0.031283),
 (111937.9488,-0.108152,0.044175),(111938.0272,-0.113169,0.052551),(111938.1413,-0.132913,0.058736),(111938.2381,-0.144759,0.072652),
 (111938.3450,-0.151867,0.081001),(111938.4479,-0.156132,0.086011),(111938.5268,-0.149852,0.086445),(111938.6478,-0.146083,0.085020),
 (111938.7246,-0.143822,0.084166),(111938.8313,-0.142808,0.084056),(111938.9416,-0.144531,0.083990),(111939.0434,-0.145564,0.083950),
 (111939.1472,-0.146185,0.085094),(111939.2380,-0.146221,0.084613),(111939.3473,-0.141537,0.082085),(111939.4385,N,N),(111939.5349,N,N),
 (111939.6427,-0.048326,0.071215),(111939.7276,-0.036275,0.068781),(111939.8440,-0.028636,0.066851),(111939.9287,-0.022965,0.068170),
 (111940.0401,-0.019562,0.065861),(111940.1328,-0.015641,0.066516),(111940.2516,0.002929,0.066909),(111940.3418,0.015999,0.068205),
 (111940.4359,0.027129,0.070560),(111940.5484,0.033853,0.081634),(111940.6266,0.037888,0.092814),(111940.7588,0.040262,0.100572),
 (111940.8260,0.038824,0.105546),(111940.9467,0.037505,0.108531),(111941.0289,0.036363,0.110321),(111941.1527,0.034980,0.110280),
 (111941.2414,0.034149,0.105853),(111941.3482,0.033258,0.101062),(111941.4421,0.032722,0.098186),(111941.5298,0.016307,0.091813),
 (111941.6533,0.003308,0.086654),(111941.7295,-0.007066,0.082087),(111941.8328,-0.013532,0.079347),(111941.9634,-0.018492,0.077586),
 (111942.0273,-0.022471,0.075511),(111942.1578,-0.024859,0.074266),(111942.2322,-0.026292,0.073121),(111942.3537,-0.025103,0.070218),
 (111942.4440,-0.021233,0.067554),(111942.5585,-0.018102,0.064409),(111942.6354,-0.015905,0.060401),(111942.7474,-0.014587,0.053501),
 (111942.8391,-0.013438,0.049361),(111942.9285,-0.011457,0.046877),(111943.0505,-0.011560,0.048191),(111943.1450,-0.012575,0.048979)]


def _run(samples, th=TH, ny=NY, np_=NP, **gaze_kw):
    head, gaze, gb = HeadDetector(ny, np_, th), GazeDetector(th, **gaze_kw), EventBuilder.for_gaze(th)
    blocks = getattr(head, "block_gaze", None) or head.blocks_gaze      # your method name vs mine
    eps, evs = [], []
    for s in samples:
        head.update(s)
        tr, ep = gaze.update(s, block=blocks(), head_score=head.last_score)
        if ep:
            eps.append(ep)
        evs += gb.update(s, tr, ep)
    return eps, evs


def _from_rows(rows):
    t0, t1 = rows[0][0], rows[-1][0]
    allrows = ([(t0 - 2 + 0.1 * k, 0.0, 0.0) for k in range(20)] + rows
               + [(t1 + 0.1 * (k + 1), 0.0, 0.0) for k in range(25)])
    return [FrameSample(ts=t, frame_idx=i, face_count=1, confidence=1.0, head_yaw=NY, head_pitch=NP,
                        head_roll=0.0, gaze_yaw=gy, gaze_pitch=gp) for i, (t, gy, gp) in enumerate(allrows)]


def test_pitch_sag_after_a_look_does_not_stretch_the_episode():
    s = _from_rows(SAG_ROWS)
    eps_old, _ = _run(s, replace(TH, gaze_axis_lock=False))
    assert eps_old[0].deviating_s > 4.0                               # the 4.8 s episode
    eps, evs = _run(s)
    assert len(eps) == 1 and eps[0].deviating_s < 2.5
    assert len(evs) == 1 and evs[0].direction == "yaw-" and 1.8 < evs[0].duration_s < 2.6


def test_blind_gap_after_a_look_splits_the_next_look_into_its_own_episode():
    s = (ScenarioBuilder(fps=10, noise_deg=0.0).steady(2).gaze_shift(3.0, gaze_yaw=0.14)
         .steady(0.3).gaze_shift(0.3, gaze_yaw=-0.057)                 # return to centre is cut short by the next look
         .low_confidence(2.7)                                          # no usable gaze for 2.7 s
         .gaze_shift(2.0, gaze_yaw=-0.12).steady(2.5).build())
    eps_old, _ = _run(s, ny=0.0, np_=0.0, split_after_gap=False)
    assert len(eps_old) == 1                                          # glued together ("resumed")
    eps, evs = _run(s, ny=0.0, np_=0.0)
    assert len(eps) == 2 and eps[0].end_reason == "gap"
    assert [e.direction for e in evs] == ["yaw+", "yaw-"]