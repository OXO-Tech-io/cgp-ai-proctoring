import json

import pytest

from temporal import (
    FrameSample, ScenarioBuilder, SessionRecorder, SessionReplayer, SlidingWindow,
)



def _session(seed=7):
    return (ScenarioBuilder(fps=30, seed=seed)
            .steady(5).gaze_shift(1, gaze_yaw=25).steady(2)
            .no_face(1).steady(2).build())


def test_roundtrip_preserves_samples(tmp_path):
    samples = _session()
    path = tmp_path / "s.jsonl"
    with SessionRecorder(path, "sess-1", meta={"fps": 30}) as rec:
        for s in samples:
            rec.write(s)
    rp = SessionReplayer(path)
    assert rp.session_id == "sess-1"
    assert rp.meta == {"fps": 30}
    assert list(rp.samples()) == samples


def test_replay_is_deterministic(tmp_path):
    path = tmp_path / "s.jsonl"
    with SessionRecorder(path, "sess-2") as rec:
        for s in _session():
            rec.write(s)

    def run():
        win = SlidingWindow(2.0)

        def step(s):
            win.push(s)
            return win.stats("gaze_yaw").mean

        return SessionReplayer(path).replay(step)

    assert run() == run()


def test_rejects_wrong_schema(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"type": "header", "schema_version": 999,
                                "session_id": "x"}) + "\n")
    with pytest.raises(ValueError, match="Schema version"):
        SessionReplayer(path)


def test_none_gaze_survives_roundtrip(tmp_path):
    """Phase 2 not ready yet -> gaze is None; must not turn into 0."""
    s = FrameSample(ts=0.0, frame_idx=0, face_count=1, confidence=0.9,
                    head_yaw=1.0, head_pitch=0.0, head_roll=0.0)
    path = tmp_path / "s.jsonl"
    with SessionRecorder(path, "x") as rec:
        rec.write(s)
    (back,) = SessionReplayer(path).samples()
    assert back.gaze_yaw is None



def _window_after(samples, duration=3.0):
    win = SlidingWindow(duration)
    for s in samples:
        win.push(s)
    return win


@pytest.mark.parametrize("fps,jitter", [(15, 0.0), (30, 0.0), (30, 0.4), (60, 0.2)])
def test_stats_are_fps_invariant(fps, jitter):
    samples = (ScenarioBuilder(fps=fps, seed=3, noise_deg=0.0, fps_jitter=jitter)
               .steady(4).gaze_shift(1.5, gaze_yaw=30).build())
    st = _window_after(samples).stats("gaze_yaw")
    assert st.mean == pytest.approx(15.0, abs=1.5)
    assert st.coverage == pytest.approx(1.0)


def test_window_evicts_by_time():
    samples = ScenarioBuilder(fps=30, seed=0).steady(10).build()
    win = _window_after(samples, duration=2.0)
    assert win.latest.ts - min(s.ts for s in win._buf) <= 2.0
    assert win.is_full


def test_no_face_reduces_coverage_not_mean():
    samples = (ScenarioBuilder(fps=30, seed=0, noise_deg=0.0)
               .steady(2).no_face(1).build())
    st = _window_after(samples).stats("gaze_yaw")
    assert st.mean == pytest.approx(0.0)
    assert st.coverage == pytest.approx(2 / 3, abs=0.05)


def test_low_confidence_excluded():
    samples = (ScenarioBuilder(fps=30, seed=0, noise_deg=0.0)
               .steady(2).build())
    samples += [FrameSample(ts=samples[-1].ts + 0.033 * (i + 1), frame_idx=1000 + i,
                            face_count=1, confidence=0.1, gaze_yaw=80.0, gaze_pitch=0.0)
                for i in range(30)]
    st = _window_after(samples).stats("gaze_yaw")
    assert st.max == pytest.approx(0.0)


def test_fraction_where():
    samples = (ScenarioBuilder(fps=30, seed=0, noise_deg=0.0)
               .steady(2).gaze_shift(1, gaze_yaw=25).build())
    win = _window_after(samples)
    frac = win.fraction_where(lambda s: abs(s.gaze_yaw) > 20, signal="gaze_yaw")
    assert frac == pytest.approx(1 / 3, abs=0.05)


def test_fraction_where_unknown_when_no_valid_data():
    samples = ScenarioBuilder(fps=30, seed=0).no_face(3).build()
    win = _window_after(samples)
    assert win.fraction_where(lambda s: True, signal="gaze_yaw") is None
    assert win.face_fraction(lambda n: n == 0) == pytest.approx(1.0)


def test_frame_stall_is_capped():
    a = FrameSample(ts=0.0, frame_idx=0, face_count=1, confidence=0.9, gaze_yaw=40.0, gaze_pitch=0)
    b = FrameSample(ts=5.0, frame_idx=1, face_count=1, confidence=0.9, gaze_yaw=0.0, gaze_pitch=0)
    win = SlidingWindow(10.0, max_gap_s=0.5)
    win.push(a)
    win.push(b)
    st = win.stats("gaze_yaw")
    assert st.span_s == pytest.approx(1.0)


def test_out_of_order_rejected():
    win = SlidingWindow(1.0)
    win.push(FrameSample(ts=1.0, frame_idx=0, face_count=1, confidence=1))
    with pytest.raises(ValueError, match="Out-of-order"):
        win.push(FrameSample(ts=0.5, frame_idx=1, face_count=1, confidence=1))