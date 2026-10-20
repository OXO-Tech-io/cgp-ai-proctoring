import json

from temporal import ScenarioBuilder, SessionRecorder
from temporal.detector import DeviationThresholds
from temporal.events import Event
from temporal.tune import analyze, calibration_index, format_report, load, main, quiet_stretches

def _record(path, wide_reading=False):
    b = ScenarioBuilder(fps=10, seed=5, noise_deg=0.003)
    b.calibrating(5)
    b.reading(30, yaw_amp=0.07 if wide_reading else 0.03, pitch_amp=0.06)
    b.gaze_shift(1.5, gaze_yaw=0.2).steady(3)
    with SessionRecorder(path, session_id="t") as r:
        for s in b.build():
            r.write(s)
    return path


def test_calibration_is_the_first_frame_with_gaze(tmp_path):
    p = _record(tmp_path / "a.jsonl")
    s = load(p)
    assert abs(s[calibration_index(s)].ts - 5.0) < 0.15


def test_quiet_baseline_keeps_the_current_threshold_and_finds_the_deliberate_look(tmp_path):
    p = _record(tmp_path / "a.jsonl")
    r = analyze(p, baseline_from=0.0, baseline_to=30.0, neutral=(0.0, 0.0))
    assert r.gaze_yaw.max < 0.045 and r.current_events == []
    assert 0.045 <= r.recommended_yaw <= 0.055 and r.rec_events == []
    assert [e.kind for e in r.outside_events] == ["glance"]
    text = format_report(r)
    assert "Resting range" in text and "clear margin" in text


def test_wide_reading_makes_the_tool_recommend_a_higher_threshold(tmp_path):
    p = _record(tmp_path / "w.jsonl", wide_reading=True)
    r = analyze(p, baseline_from=0.0, baseline_to=30.0, neutral=(0.0, 0.0))
    assert r.recommended_yaw > 0.07
    assert len(r.rec_events) <= len(r.current_events)


def test_neutral_is_read_from_the_events_json_next_to_the_recording(tmp_path):
    p = _record(tmp_path / "n.jsonl")
    (tmp_path / "n.events.json").write_text(json.dumps({"calibrations": [{"neutral_yaw": -3.0, "neutral_pitch": 1.5}]}))
    r = analyze(p, 0.0, 30.0)
    assert r.neutral == (-3.0, 1.5) and r.neutral_source == "n.events.json"


def test_cli_prints_a_report(tmp_path, capsys):
    p = _record(tmp_path / "c.jsonl")
    main([str(p), "--baseline", "0:30", "--neutral", "0,0"])
    assert "Suggested: gaze_yaw_enter" in capsys.readouterr().out

def _ev(a, b):
    return Event("gaze", "glance", "yaw+", a, b, b - a, 2.0, "returned", 1)


def test_quiet_stretches_are_the_long_gaps_between_events():
    ev = [_ev(2, 4), _ev(5, 6), _ev(20, 22)]
    assert quiet_stretches(ev, 0.0, 40.0) == [(22.5, 40.0), (6.5, 19.5)]
    assert quiet_stretches([_ev(1, 2), _ev(6, 7)], 0.0, 12.0) == []


def test_report_warns_when_the_quiet_window_contains_events(tmp_path):
    p = _record(tmp_path / "w2.jsonl")
    r = analyze(p, baseline_from=0.0, baseline_to=40.0, neutral=(0.0, 0.0))     # window includes the deliberate look
    text = format_report(r)
    assert r.current_events and "WARNING" in text