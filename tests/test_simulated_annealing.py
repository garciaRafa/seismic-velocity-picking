"""Tests for src/optimizers/simulated_annealing.py."""

import numpy as np
import pytest

from src.optimizers.simulated_annealing import SimulatedAnnealing


def _sa(d, **kw):
    params = dict(vel_min=1400.0, vel_max=3000.0, min_sep_ms=100.0,
                  objective='threshold', tau=0.2, seed=5, max_evals=3000,
                  step_time=5, step_vel=50.0)
    params.update(kw)
    return SimulatedAnnealing(d['gather'], d['offsets'], d['dt_ms'], **params)


def test_needs_a_budget(layered_gather):
    with pytest.raises(ValueError):
        _sa(layered_gather, max_evals=None)


def test_uses_the_whole_budget_and_is_reproducible(layered_gather):
    a = _sa(layered_gather).run()
    b = _sa(layered_gather).run()
    assert a.n_evals == 3000
    assert a.best_score == b.best_score and a.best_picks == b.best_picks
    assert a._is_valid(a.best_picks)


def test_t0_from_tau_in_v3(layered_gather):
    opt = _sa(layered_gather, tau=0.3, p0=0.8).run()
    assert opt.t0 == pytest.approx(0.3 / -np.log(0.8))


def test_explicit_t0_and_cooling(layered_gather):
    opt = _sa(layered_gather, t0=2.0, t_final_ratio=1e-2)
    opt.run()
    assert opt.t0 == 2.0
    assert opt.temperature == pytest.approx(2.0 * 1e-2)      # budget used up


def test_accepts_worse_moves_when_hot_and_few_when_cold(layered_gather):
    hot = _sa(layered_gather, t0=10.0, t_final_ratio=0.9).run()
    cold = _sa(layered_gather, t0=1e-6, t_final_ratio=0.5).run()
    rate = lambda o: o.accept_stats['worse_accepted'] / max(
        1, o.accept_stats['worse_accepted'] + o.accept_stats['worse_rejected'])
    assert rate(hot) > 0.5 and rate(cold) < 0.05


def test_finds_the_two_reflections(layered_gather):
    """With a sensible budget the run ends on both reflections. (Some seeds
    end with one: a missing event can only come back through an insertion
    that draws time and velocity uniformly, which rarely lands on a blob
    once the temperature is low.)"""
    d = layered_gather
    opt = _sa(d, seed=9, max_evals=20000, vel_max=3000.0).run()
    times = np.array([p[0] for p in opt.best_picks]) * d['dt_ms']
    for t_ev in d['event_times_ms']:
        assert np.min(np.abs(times - t_ev)) <= 40.0


def test_calibration_without_tau(layered_gather):
    opt = _sa(layered_gather, objective='mean', tau=None, n_picks=2,
              min_sep_ms=100.0).run()
    assert opt.t0 > 0 and opt.n_evals == 3000


def test_trajectory_fields(layered_gather):
    opt = _sa(layered_gather, record_trajectory=True, record_every=10).run()
    rec = opt.trajectory[-1]
    assert {'temperature', 'accepted', 'delta'} <= set(rec)
    temps = [r['temperature'] for r in opt.trajectory]
    assert all(t2 <= t1 + 1e-12 for t1, t2 in zip(temps, temps[1:]))