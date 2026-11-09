"""Tests for Particle Swarm Optimization (with and without Lévy flights)."""

import numpy as np
import pytest

from src.optimizers.pso import ParticleSwarm, ParticleSwarmLevy, levy_steps


def _make(d, cls=ParticleSwarm, **kw):
    opts = dict(vel_min=1400.0, vel_max=3000.0, min_sep_ms=100.0, objective='threshold',
                tau=0.2, seed=3, max_evals=2000, min_vel_gradient=100.0, swarm_size=10)
    opts.update(kw)
    return cls(d['gather'], d['offsets'], d['dt_ms'], **opts)


@pytest.mark.parametrize('cls', [ParticleSwarm, ParticleSwarmLevy])
def test_budget_seed_and_validity(layered_gather, cls):
    a = _make(layered_gather, cls).run()
    b = _make(layered_gather, cls).run()
    assert a.n_evals == 2000
    assert a.best_score == b.best_score and a.best_picks == b.best_picks
    assert a._is_valid(a.best_picks)
    assert (a.stats['levy_steps'] > 0) == (cls is ParticleSwarmLevy)


def test_decode_is_always_valid(layered_gather):
    opt = _make(layered_gather)
    opt.rng = np.random.default_rng(0)
    for _ in range(300):
        x = np.column_stack([opt.rng.uniform(opt.t_min, opt.t_max, opt.n_slots),
                             opt.rng.uniform(1400, 3000, opt.n_slots),
                             opt.rng.random(opt.n_slots)])
        assert opt._is_valid(opt.decode(x))


def test_repair_keeps_the_longest_valid_chain(layered_gather):
    opt = _make(layered_gather)                    # min_sep 50 samples, 100 m/s per s
    # a very fast early pick must not remove the good chain after it
    picks = [(100, 2900.0), (200, 1600.0), (300, 1700.0), (400, 1800.0)]
    assert opt._repair(picks) == [(200, 1600.0), (300, 1700.0), (400, 1800.0)]


def test_encode_then_decode_returns_the_solution(layered_gather):
    opt = _make(layered_gather)
    opt.rng = np.random.default_rng(1)
    picks = [(100, 1500.0), (300, 1700.0), (500, 1900.0)]
    assert opt.decode(opt._encode(picks)) == picks


def test_levy_steps_are_heavy_tailed():
    s = levy_steps(np.random.default_rng(0), 20000, 1.5)
    q50, q999 = np.percentile(np.abs(s), [50, 99.9])
    assert q999 / q50 > 50                         # a Gaussian gives ~5


def test_fixed_k_is_rejected(layered_gather):
    with pytest.raises(ValueError):
        _make(layered_gather, objective='mean', tau=None, n_picks=3)


def test_cache_does_not_change_the_run(layered_gather):
    a = _make(layered_gather).run()
    b = _make(layered_gather, cache_semblance=True, record_trajectory=True).run()
    assert a.best_score == b.best_score and a.best_picks == b.best_picks
    assert b.trajectory and 'swarm_spread' in b.trajectory[-1]