from temporal import ScenarioBuilder
from temporal.detector import DeviationThresholds,GazeDetector, HeadDetector

TH = DeviationThresholds()

def test_open_head_episode_is_flushed_at_session_end():
    s = ScenarioBuilder(fps=10, seed=1, noise_deg=0.0).steady(3).head_turn(6, yaw=30).build()
    head = HeadDetector(0.0, 0.0,TH)
    eps = [e for x in s for e in [head.update(x)[1]] if e]
    assert eps == []
    _, ep = head.finish(s[-1].ts)
    assert ep is not None and ep.end_reason == "session_end"
    assert ep.deviating_s > 5.0 and ep.peak > 1.5

def test_finish_with_nothing_open_returns_no_episode():
    s = ScenarioBuilder(fps=10, seed=1, noise_deg=0.0).steady(3).build()
    head, gaze = HeadDetector(0.0, 0.0, TH), GazeDetector(TH)
    for x in s:
        head.update(x); gaze.update(x)
    assert head.finish(s[-1].ts) == ([], None)
    assert gaze.finish(s[-1].ts) == ([], None)