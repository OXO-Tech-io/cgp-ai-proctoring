from temporal import ScenarioBuilder, TemporalMonitor
from temporal.detector import DeviationThresholds

TH = DeviationThresholds(gaze_yaw_enter=0.055)
FPS, NOISE = 10, 0.003


def _run(samples, th=TH):
    lines = []
    m = TemporalMonitor(th, emit=lines.append)
    m.start(0.0, 0.0)
    for s in samples:
        m.update(s)
    m.stop(samples[-1].ts + 0.1)
    return m.events, lines


def _b():
    return ScenarioBuilder(fps=FPS, noise_deg=NOISE)


def _kinds(events, source=None):
    return [(e.source, e.kind, e.direction) for e in events if source in (None, e.source)]


def test_three_one_second_eye_flicks_make_three_glances_and_a_burst():
    s = (_b().steady(3).gaze_shift(1.0, gaze_yaw=0.12).steady(3).gaze_shift(1.0, gaze_yaw=0.12).steady(3)
         .gaze_shift(1.0, gaze_yaw=0.12).steady(3).build())
    ev, _ = _run(s)
    assert _kinds(ev) == [("gaze", "glance", "yaw+")] * 3 + [("gaze", "repeated", "yaw+")]
    assert all(0.8 < e.duration_s < 1.4 for e in ev if e.kind == "glance")


def test_a_flick_shorter_than_0_3_s_is_ignored():
    ev, lines = _run(_b().steady(3).gaze_shift(0.2, gaze_yaw=0.12).steady(3).build())
    assert ev == []
    assert any("too_brief" in l for l in lines)


def test_a_half_second_look_is_reported():
    ev, _ = _run(_b().steady(3).gaze_shift(0.5, gaze_yaw=0.12).steady(3).build())
    assert _kinds(ev) == [("gaze", "glance", "yaw+")]


def test_a_ten_degree_head_hold_for_five_seconds_is_not_a_head_event():
    ev, _ = _run(_b().steady(3).head_turn(5.0, yaw=10.0).steady(3).build())
    assert _kinds(ev, "head") == []


def test_a_head_turn_past_the_limit_is_a_head_event_and_never_a_gaze_event():
    ev, _ = _run(_b().steady(3).head_turn(2.0, yaw=30.0).steady(3).build())
    assert _kinds(ev) == [("head", "glance", "yaw+")]


def test_a_long_head_turn_is_sustained():
    ev, _ = _run(_b().steady(3).head_turn(5.0, yaw=30.0).steady(3).build())
    assert _kinds(ev) == [("head", "sustained", "yaw+")]


def test_an_eye_look_with_the_head_slightly_off_centre_is_still_detected():
    s = _b().steady(3)
    s._emit(1.5, head=(9.0, 0.0, 0.0), gaze=(0.12, 0.0))                  # head at 0.5x of its limit
    ev, _ = _run(s.steady(3).build())
    assert ("gaze", "glance", "yaw+") in _kinds(ev) and _kinds(ev, "head") == []


def test_sixty_seconds_of_perfectly_still_resting_is_silent():
    ev, lines = _run(_b().steady(60).build())
    assert ev == [] and not any("EPISODE" in l for l in lines)


def test_sixty_seconds_of_reading_is_silent():
    ev, lines = _run(ScenarioBuilder(fps=FPS, seed=7, noise_deg=NOISE).reading(60).build())
    assert ev == [] and not any("EPISODE" in l for l in lines), lines


def test_reading_with_a_deliberate_look_in_the_middle_reports_only_that_look():
    b = ScenarioBuilder(fps=FPS, seed=3, noise_deg=NOISE)
    ev, _ = _run(b.reading(20).gaze_shift(1.5, gaze_yaw=-0.15).reading(20).build())
    assert _kinds(ev) == [("gaze", "glance", "yaw-")]


def test_the_reading_scenario_has_blinks_and_head_sway():
    s = ScenarioBuilder(fps=FPS, seed=7, noise_deg=NOISE).reading(30).build()
    assert any(x.gaze_yaw is None for x in s) and len({round(x.head_yaw, 1) for x in s}) > 10