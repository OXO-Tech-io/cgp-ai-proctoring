from temporal import FrameSample, ScenarioBuilder
from temporal.detector import DeviationThresholds, GazeDetector, HeadDetector
from temporal.events import EventBuilder, EventConfig, RepeatTracker

TH = DeviationThresholds()
N = None

# Real frames (session with neutral yaw -2.40 / pitch +0.36): eyes to the +yaw side
# (peak 0.0927), then 0.6 s later to the -yaw side (peak -0.1138). Head stays still.
REAL_GLANCES = [  # ts, gaze_yaw, gaze_pitch
 (89512.0293,0.005809,-0.007859),(89512.1262,0.006529,-0.001127),(89512.2271,N,N),
 (89512.3274,0.008239,0.005231),(89512.4244,0.015347,0.011458),(89512.5370,0.031010,0.016679),
 (89512.6247,N,N),(89512.7321,0.046316,0.022466),(89512.8403,0.059362,0.027581),
 (89512.9270,0.067189,0.032301),(89513.0309,0.074277,0.035134),(89513.1274,0.078529,0.039179),
 (89513.2316,0.081081,0.041606),(89513.3264,0.084290,0.039178),(89513.4343,0.088227,0.037615),
 (89513.5388,0.090588,0.036784),(89513.6315,N,N),(89513.7276,0.092005,0.036285),
 (89513.8453,0.092670,0.035879),(89513.9458,0.090171,0.035636),(89514.0234,0.086208,0.032845),
 (89514.1440,0.072603,0.030688),(89514.2302,N,N),(89514.3321,0.037544,0.011649),
 (89514.4344,0.028188,0.004732),(89514.5334,0.022344,0.000582),(89514.6333,0.017877,-0.001908),
 (89514.7336,0.015198,-0.002319),(89514.8509,0.010806,-0.002502),(89514.9253,0.007683,-0.001373),
 (89515.0597,-0.004477,0.001240),(89515.1249,N,N),(89515.2471,N,N),
 (89515.3276,-0.072912,0.009090),(89515.4532,-0.082280,0.008470),(89515.5299,-0.090800,0.000886),
 (89515.6538,-0.095912,-0.003664),(89515.7329,-0.104333,-0.007371),(89515.8554,-0.109474,-0.008618),
 (89515.9400,-0.112471,-0.009366),(89516.0303,-0.113302,-0.009588),(89516.1271,-0.113801,-0.009722),
 (89516.2467,-0.111998,-0.006903),(89516.3341,-0.110916,-0.003566),(89516.4469,-0.109710,-0.001564),
 (89516.5376,-0.108986,-0.002008),(89516.6369,-0.109109,-0.005102),(89516.7360,-0.110507,-0.006958),
 (89516.8450,-0.109465,-0.008071),(89516.9392,-0.110720,-0.008401),(89517.0365,-0.111474,-0.008599),
 (89517.1418,-0.109088,-0.008718),(89517.2494,N,N),(89517.3486,N,N),
 (89517.4425,-0.039643,-0.019526),(89517.5629,-0.029678,-0.019952),(89517.6387,-0.022163,-0.020208),
 (89517.7378,-0.016910,-0.017581),(89517.8431,-0.014369,-0.013928),(89517.9343,-0.011937,-0.011437),
 (89518.0535,-0.010478,-0.008758),(89518.1437,-0.009677,-0.006983),(89518.2556,-0.009195,-0.005918),
 (89518.3520,-0.008614,-0.005278),(89518.4529,-0.008186,-0.004895),(89518.5431,-0.007208,-0.006470),
 (89518.6526,-0.005379,-0.007415),(89518.7394,-0.001424,-0.007733),(89518.8618,0.005472,-0.007924)]


def _run(samples, ny=0.0, np_=0.0, cfg=EventConfig()):
    head, gaze = HeadDetector(ny, np_, TH), GazeDetector(TH)
    hb, gb = EventBuilder.for_head(ny, np_, TH, cfg), EventBuilder.for_gaze(TH, cfg)
    rt = RepeatTracker(cfg)
    eps, evs = {"head": [], "gaze": []}, []
    for s in samples:
        ht, he = head.update(s)
        gt, ge = gaze.update(s, block=head.block_gaze(), head_score=head.last_score)
        for src, ep in (("head", he), ("gaze", ge)):
            if ep is not None:
                eps[src].append(ep)
        for ev in hb.update(s, ht, he) + gb.update(s, gt, ge):
            evs += rt.add(ev)
    return eps, evs, (head, gaze, hb, gb)


def test_two_real_glances_become_two_events():
    s = [FrameSample(ts=t, frame_idx=i, face_count=1, confidence=1.0, head_yaw=-2.4, head_pitch=0.36,
                     head_roll=0.0, gaze_yaw=gy, gaze_pitch=gp)
         for i, (t, gy, gp) in enumerate(REAL_GLANCES)]
    eps, evs, _ = _run(s, -2.4, 0.36)
    assert len(eps["gaze"]) == 2 and not eps["head"]
    assert [e.direction for e in evs] == ["yaw+", "yaw-"]
    assert all(e.kind == "glance" and e.source == "gaze" for e in evs)
    assert abs(evs[0].peak - 1.54) < 0.02 and abs(evs[1].peak - 1.90) < 0.02   # matches terminal
    assert 1.0 < evs[0].duration_s < 2.2 and 1.5 < evs[1].duration_s < 2.8


def test_merged_head_turn_is_split_by_direction():
    s = (ScenarioBuilder(fps=10, seed=1, noise_deg=0.0).steady(2)
         .head_turn(2.0, yaw=-35).head_turn(0.2, yaw=0)            # left, straight through centre...
         .head_turn(5.0, yaw=25, pitch=-17).steady(2).build())     # ...then right and up
    eps, evs, _ = _run(s)
    assert len(eps["head"]) == 1                                   # the detector merges them
    assert [(e.direction, e.kind) for e in evs] == [("yaw-", "glance"), ("yaw+", "sustained")]
    assert evs[0].end_reason == "direction_change" and evs[1].end_reason == "returned"


def test_glance_vs_sustained_boundary():
    s = (ScenarioBuilder(fps=10, noise_deg=0.0).steady(2).gaze_shift(1.0, gaze_yaw=0.1)
         .steady(3).gaze_shift(4.0, gaze_yaw=0.1).steady(2).build())
    _, evs, _ = _run(s)
    assert [e.kind for e in evs] == ["glance", "sustained"]


def test_three_glances_in_30s_make_one_repeated_event():
    b = ScenarioBuilder(fps=10, noise_deg=0.0).steady(2)
    for _ in range(3):
        b.gaze_shift(1.0, gaze_yaw=0.1).steady(4)
    _, evs, _ = _run(b.build())
    assert [e.kind for e in evs] == ["glance", "glance", "glance", "repeated"]
    assert evs[-1].count == 3 and evs[-1].direction == "yaw+"


def test_two_glances_or_slow_glances_are_not_repeated():
    two = ScenarioBuilder(fps=10, noise_deg=0.0).steady(2)
    for _ in range(2):
        two.gaze_shift(1.0, gaze_yaw=0.1).steady(4)
    assert "repeated" not in [e.kind for e in _run(two.build())[1]]
    slow = ScenarioBuilder(fps=10, noise_deg=0.0).steady(2)
    for _ in range(3):
        slow.gaze_shift(1.0, gaze_yaw=0.1).steady(20)             # 21 s apart: three span 42 s > 30 s
    assert "repeated" not in [e.kind for e in _run(slow.build())[1]]


def _head_noise_only(samples):
    return [FrameSample(**{**x.to_dict(), "gaze_yaw": 0.0, "gaze_pitch": 0.0}) for x in samples]


def test_diagonal_head_pose_does_not_flicker_into_pieces():
    for seed in range(1, 8):
        s = _head_noise_only(ScenarioBuilder(fps=10, seed=seed, noise_deg=1.5).steady(2)
                             .head_turn(6.0, yaw=20, pitch=-16).steady(2).build())   # 1.11x vs 1.07x
        _, evs, _ = _run(s)
        assert len(evs) == 1 and evs[0].kind == "sustained", (seed, [e.summary() for e in evs])


def test_pitch_bobbing_inside_a_yaw_turn_does_not_split_it():
    s = _head_noise_only(ScenarioBuilder(fps=10, seed=2, noise_deg=4.0).steady(2)
                         .head_turn(6.0, yaw=-30).steady(2).build())   # noisy pitch crosses both signs
    _, evs, _ = _run(s)
    assert len(evs) == 1 and evs[0].direction == "yaw-"


def test_open_episode_is_flushed_with_an_event_at_session_end():
    s = ScenarioBuilder(fps=10, noise_deg=0.0).steady(2).head_turn(6.0, yaw=30).build()
    eps, evs, (head, gaze, hb, gb) = _run(s)
    assert eps["head"] == [] and evs == []
    tr, ep = head.finish(s[-1].ts)
    out = hb.update(None, tr, ep)
    assert len(out) == 1 and out[0].kind == "sustained" and out[0].end_reason == "session_end"


def test_unconfirmed_crossing_makes_no_event():
    s = ScenarioBuilder(fps=10, noise_deg=0.0).steady(2).gaze_shift(0.2, gaze_yaw=0.1).steady(3).build()
    eps, evs, _ = _run(s)
    assert evs == []