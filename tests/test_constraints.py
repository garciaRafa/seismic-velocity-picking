"""
Tests for the time constraints of formulation v2: minimum separation
between picks and allowed time window.
"""

import numpy as np
import pytest

from src.optimizers.hill_climbing import HillClimbing

DT = 2.0


def _opt(n_samples=1501, n_picks=4, **kw):
    traces  = np.zeros((5, n_samples), dtype=np.float32)
    offsets = np.linspace(100.0, 3000.0, 5, dtype=np.float32)
    return HillClimbing(traces, offsets, DT, vel_min=1400.0, vel_max=3000.0,
                        n_picks=n_picks, seed=0, **kw)


def test_conversion_from_ms_to_samples():
    o = _opt(min_sep_ms=100.0, t_min_ms=600.0, t_max_ms=2700.0)
    assert (o.min_sep, o.t_min, o.t_max) == (50, 300, 1350)


def test_defaults_are_formulation_v1():
    o = _opt()
    assert (o.min_sep, o.t_min, o.t_max) == (1, 0, 1500)


def test_minimum_separation_is_enforced():
    o = _opt(min_sep_ms=100.0)                  # 50 samples
    ok  = [(300, 1500.0), (350, 1600.0), (400, 1700.0), (450, 1800.0)]
    bad = [(300, 1500.0), (349, 1600.0), (400, 1700.0), (450, 1800.0)]
    assert o._is_valid(ok)
    assert not o._is_valid(bad)


def test_time_window_is_enforced():
    o = _opt(t_min_ms=600.0, t_max_ms=2700.0)   # samples 300..1350
    assert o._is_valid([(300, 1500.0), (500, 1600.0), (900, 1700.0), (1350, 1800.0)])
    assert not o._is_valid([(299, 1500.0), (500, 1600.0), (900, 1700.0), (1300, 1800.0)])
    assert not o._is_valid([(300, 1500.0), (500, 1600.0), (900, 1700.0), (1351, 1800.0)])


def test_random_picks_respect_all_constraints():
    o = _opt(n_picks=10, min_sep_ms=100.0, t_min_ms=600.0, t_max_ms=2700.0)
    for _ in range(500):
        picks = o._random_picks()
        times = [t for t, _ in picks]
        assert o._is_valid(picks)
        assert min(times) >= 300 and max(times) <= 1350
        assert min(np.diff(times)) >= 50


def test_tightest_feasible_case_has_a_single_solution():
    # 4 picks, 100 ms apart, in a 300 ms window: only one placement
    o = _opt(min_sep_ms=100.0, t_min_ms=1000.0, t_max_ms=1300.0)
    for _ in range(20):
        assert [t for t, _ in o._random_picks()] == [500, 550, 600, 650]


def test_infeasible_constraints_raise():
    with pytest.raises(ValueError):
        _opt(n_picks=10, min_sep_ms=300.0, t_min_ms=600.0, t_max_ms=2700.0)


def test_hill_climbing_result_respects_constraints(layered_gather):
    g, off, dt = (layered_gather[k] for k in ('gather', 'offsets', 'dt_ms'))
    hc = HillClimbing(g, off, dt, vel_min=1400.0, vel_max=3000.0, n_picks=3,
                      max_iter=40, restarts=2, n_neighbors=10, patience=10,
                      seed=5, min_sep_ms=100.0, t_min_ms=600.0,
                      t_max_ms=2700.0).run()
    times = [t for t, _ in hc.best_picks]
    assert hc._is_valid(hc.best_picks)
    assert min(times) >= 300 and max(times) <= 1350
    assert min(np.diff(sorted(times))) >= 50
