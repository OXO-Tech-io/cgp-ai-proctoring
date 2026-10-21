import pytest

from temporal import ScenarioBuilder, TemporalMonitor
from temporal.detector import DeviationThresholds, GazeDetector, HeadDetector
from temporal.events import Event, EventBuilder, EventConfig, RepeatTracker

TH = DeviationThresholds(gaze_yaw_enter=0.055)


def _inline_like_old_main(samples, ny=0.0, np_=0.0):
    head, gaze = HeadDetector(ny, np_, TH), GazeDetector(TH)
    hb, gb = EventBuilder.for_head(ny, np_, TH), EventBuilder.for_gaze(TH)
    rep, out = RepeatTracker(), []
    blocks = getattr(head, "block_gaze", None) or head.blocks_gaze
    for s in samples:
        h = head.update(s)
        g = gaze.update(s, block=blocks(), head_score=head.last_score)
        for (tr, ep), b in ((h, hb), (g, gb)):
            for ev in b.update(s, tr, ep):
                out += rep.add(ev)
    end = samples[-1].ts + 0.1
    for det, b in ((head, hb), (gaze, gb)):
        tr, ep = det.finish(end)
        for ev in b.update(None, tr, ep):
            out += rep.add(ev)
    return out


def _with_monitor(samples, ny=0.0, np_=0.0):
    m = TemporalMonitor(TH, emit=lambda _l: None)
    m.start(ny, np_)
    out = []
    for s in samples:
        out += m.update(s)
    out += m.stop(samples[-1].ts + 0.1)
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_monitor_reports_exactly_what_the_old_inline_wiring_did(seed):
    s = (ScenarioBuilder(fps=10, seed=seed, noise_deg=0.003).reading(10)
         .gaze_shift(1.5, gaze_yaw=0.15).reading(5).gaze_shift(1.0, gaze_yaw=-0.15).reading(5)
         .gaze_shift(1.0, gaze_yaw=0.15).steady(2).head_turn(2.0, yaw=30).steady(3)
         .head_turn(6.0, pitch=-25).build())
    old, new = _inline_like_old_main(s), _with_monitor(s)
    assert new == old and len(new) >= 5


def test_reset_forgets_glance_history_so_old_glances_cannot_form_a_burst():
    m = TemporalMonitor(TH, emit=lambda _l: None, event_cfg=EventConfig())
    g = lambda t: Event("gaze", "glance", "yaw+", t, t + 1.0, 1.0, 2.0, "returned", 1)       # noqa: E731
    m.repeats.add(g(0)); m.repeats.add(g(5))
    m.reset()
    assert [e.kind for e in m.repeats.add(g(9))] == ["glance"]