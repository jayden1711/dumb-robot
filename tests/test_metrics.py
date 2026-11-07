import pytest

from dumbrobot import scoring
from dumbrobot.sim import EpisodeResult, FlipCounter, SegStat, motor_power
from dumbrobot.worlds import flat_course, get_course


def test_flip_counter_hysteresis():
    f = FlipCounter()
    seq = [1.0, 0.3, 0.1, 0.1, 0.3, 0.4, 0.6, 1.0, -0.5, -0.9, 0.7, 0.19]
    events = [f.update(u, 0.01) for u in seq]
    assert f.flips == 3
    assert events == [False, False, True, False, False, False, False, False, True, False, False, True]


def test_inverted_clock_resets_when_recovered():
    f = FlipCounter()
    for _ in range(100):
        f.update(-1.0, 0.01)
    assert f.inverted_s == pytest.approx(1.0)
    f.update(0.9, 0.01)
    assert f.inverted_s == 0.0


def test_motor_power_dc_model():
    stall, w0 = 0.15, 20.0
    assert motor_power(stall, 0.0, stall, w0) == pytest.approx(stall * w0)  # V * I_stall
    assert motor_power(0.0, w0, stall, w0) == 0.0  # free spin draws nothing in this model
    assert motor_power(-0.05, 10.0, stall, w0) == pytest.approx(0.05 ** 2 * w0 / stall)  # braking: loss only
    assert motor_power(0.075, 10.0, stall, w0) == pytest.approx(0.75 + 0.075 ** 2 * w0 / stall)


def _res(**kw):
    base = dict(status="timeout", done=False, progress=0.5, time_s=10.0, energy_j=0.0, flips=0, segments=[], ending={},
                brain="A", course="bumps")
    base.update(kw)
    return EpisodeResult(**base)


def test_score_formula_by_hand():
    c = get_course("bumps")
    s = scoring.score_episode(_res(progress=0.4), c)
    assert s["score"] == pytest.approx(0.4)
    s = scoring.score_episode(_res(done=True, status="finished", progress=1.0, time_s=c.t_limit / 2, energy_j=c.e_ref / 2),
                              c)
    assert s["score"] == pytest.approx(1.0 + 0.30 * 0.5 + 0.15 * 0.5)
    s = scoring.score_episode(_res(progress=0.6, flips=7), c)
    assert s["score"] == pytest.approx(0.6 - 0.05 * 4)


def test_score_ordering_finish_beats_fast_partial():
    c = get_course("bumps")
    fin = scoring.score_episode(_res(done=True, status="finished", progress=1.0, time_s=c.t_limit, energy_j=c.e_ref), c)
    part = scoring.score_episode(_res(progress=0.99), c)
    assert fin["score"] > part["score"]


def test_exploded_scores_zero_and_invalid():
    s = scoring.score_episode(_res(status="exploded", progress=1.0), get_course("bumps"))
    assert s["score"] == 0.0 and not s["valid"]


def test_segments_cover_course_in_order():
    c = get_course("bumps")
    xs = [(s.x0, s.x1) for s in c.segments]
    for (a0, a1), (b0, b1) in zip(xs, xs[1:]):
        assert a1 == pytest.approx(b0)
    assert c.segments[-1].x1 == pytest.approx(c.finish_x)
    assert c.segment_index(-1.0) == 0
    assert c.segments[c.segment_index(1.0)].name == "bump_1"
    assert c.segments[c.segment_index(99.0)].name == "run_out"
    assert c.progress(c.start_x) == 0.0 and c.length == pytest.approx(c.finish_x - c.start_x)
    assert flat_course().length > 0
