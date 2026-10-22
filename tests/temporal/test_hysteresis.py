import random

from temporal import FrameSample, ScenarioBuilder
from temporal.detector import DeviationThresholds, GazeDetector, HeadDetector
from temporal.hysteresis import HysteresisConfig, HysteresisMachine, State


def run(m, scores, dt=0.1):
    trans, eps = [], []
    for i, sc in enumerate(scores):
        t, e = m.update(i * dt, sc)
        trans += t
        if e:
            eps.append(e)
    return trans, eps


def test_jitter_at_threshold_does_not_flicker():
    rnd = random.Random(1)
    scores = [0.0] * 10 + [rnd.uniform(0.85, 1.15) for _ in range(100)]
    trans, _ = run(HysteresisMachine(), scores)
    assert [t.new for t in trans] == [State.SUSPECT, State.DEVIATING]


def test_brief_spike_never_deviates():
    trans, eps = run(HysteresisMachine(), [0] * 5 + [1.6, 1.6] + [0] * 10)
    assert State.DEVIATING not in [t.new for t in trans]
    assert eps[0].end_reason == "dropped" and eps[0].confirmed_ts is None


def test_sustained_deviation_episode():
    trans, eps = run(HysteresisMachine(), [0] * 5 + [1.5] * 30 + [0] * 15)
    assert [t.new for t in trans] == [State.SUSPECT, State.DEVIATING, State.RETURNING, State.NORMAL]
    assert abs(eps[0].deviating_s - 3.0) < 0.35 and eps[0].end_reason == "returned"
    assert eps[0].peak == 1.5


def test_resume_from_returning():
    scores = [1.5] * 10 + [0.0] * 3 + [1.5] * 10 + [0.0] * 10
    trans, eps = run(HysteresisMachine(), scores)
    assert len(eps) == 1 and State.DEVIATING in [t.new for t in trans[3:]]


def test_short_gap_is_bridged():
    scores = [1.5] * 10 + [None] * 5 + [1.5] * 10 + [0.0] * 10
    _, eps = run(HysteresisMachine(), scores)
    assert len(eps) == 1


def test_long_gap_freezes_then_loses():
    m = HysteresisMachine(HysteresisConfig(lose_s=2.0))
    trans, eps = run(m, [1.5] * 10 + [None] * 40)
    assert eps and eps[0].end_reason == "signal_lost"


def test_gap_during_suspect_drops():
    trans, eps = run(HysteresisMachine(), [1.5, 1.5] + [None] * 15)
    assert eps and eps[0].end_reason == "dropped"


def test_head_turn_below_threshold_is_not_gaze():
    th = DeviationThresholds()
    s = ScenarioBuilder(fps=10, seed=2, noise_deg=0.0).steady(3).head_turn(5, yaw=15).steady(3).build()
    g, h = GazeDetector(th), HeadDetector(0.0, 0.0, th)
    ge = [e for x in s for e in [g.update(x)[1]] if e]
    he = [e for x in s for e in [h.update(x)[1]] if e]
    assert not ge   # gaze stays 0 while the head turns 15 deg
    assert not he   # 15 deg < 18 deg head threshold


def test_head_past_threshold_flags():
    th = DeviationThresholds()
    s = ScenarioBuilder(fps=10, seed=2, noise_deg=0.0).steady(3).head_turn(5, yaw=30).steady(3).build()
    h = HeadDetector(0.0, 0.0, th)
    he = [e for x in s for e in [h.update(x)[1]] if e]
    assert len(he) == 1 and he[0].deviating_s > 4


def test_rest_noise_floor():
    th = DeviationThresholds()
    rnd = random.Random(0)
    rest = [FrameSample(ts=i * .1, frame_idx=i, face_count=1, confidence=1,
                        gaze_yaw=rnd.gauss(0, .02), gaze_pitch=rnd.gauss(0, .02)) for i in range(60)]
    assert th.with_rest_noise(rest).gaze_yaw_enter > th.gaze_yaw_enter   # noisy user -> higher bar
    assert th.with_rest_noise(rest[:5]) == th                            # too little data -> unchanged