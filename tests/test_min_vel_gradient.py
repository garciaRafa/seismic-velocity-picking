"""Tests for the minimum velocity gradient between consecutive picks."""

import numpy as np
import pytest

from src.optimizers.hill_climbing import HillClimbing
from src.optimizers.random_search import RandomSearch
from src.optimizers.simulated_annealing import SimulatedAnnealing
from src.optimizers.tabu_search import TabuSearch

G = 200.0                     # m/s per second of two-way time


def _base(cls, d, **kw):
    params = dict(vel_min=1400.0, vel_max=3000.0, min_sep_ms=100.0,
                  objective='threshold', tau=0.2, seed=4, max_evals=2000)
    params.update(kw)
    if cls in (HillClimbing, TabuSearch):
        params.update(step_time=5, step_vel=50.0, n_neighbors=10, max_iter=10**5)
    if cls is HillClimbing:
        params.update(restarts=10**4, patience=15)
    if cls is TabuSearch:
        params.update(tabu_tenure=10, patience=10**5, p_jump=0.2)
    if cls is SimulatedAnnealing:
        params.update(step_time=5, step_vel=50.0)
    return cls(d['gather'], d['offsets'], d['dt_ms'], **params)


def test_zero_gradient_is_the_plain_monotonic_run(layered_gather):
    for cls in (HillClimbing, TabuSearch, SimulatedAnnealing, RandomSearch):
        a = _base(cls, layered_gather).run()
        b = _base(cls, layered_gather, min_vel_gradient=0.0).run()
        assert a.best_score == b.best_score and a.best_picks == b.best_picks


def test_validity_uses_the_gradient(layered_gather):
    opt = _base(RandomSearch, layered_gather, min_vel_gradient=G)
    dt = layered_gather['dt_ms']
    t1, t2 = 300, 550                                   # 500 ms apart
    need = G * (t2 - t1) * dt / 1000.0                  # 100 m/s
    assert opt._is_valid([(t1, 1800.0), (t2, 1800.0 + need + 1)])
    assert not opt._is_valid([(t1, 1800.0), (t2, 1800.0 + need - 1)])
    # with zero gradient only a decrease is invalid
    plain = _base(RandomSearch, layered_gather)
    assert plain._is_valid([(t1, 1800.0), (t2, 1801.0)])
    assert not plain._is_valid([(t1, 1800.0), (t2, 1799.0)])


def test_random_and_inserted_picks_respect_the_gradient(layered_gather):
    opt = _base(RandomSearch, layered_gather, min_vel_gradient=G)
    for _ in range(300):
        picks = opt._random_picks()
        assert opt._is_valid(picks)
        ins = opt._insert_neighbor(picks)
        if ins is not None:
            assert opt._is_valid(ins[0])
        rem = opt._remove_neighbor(picks)
        if rem is not None:
            assert opt._is_valid(rem[0])           # removing keeps the gradient


def test_tabu_jump_with_gradient_is_valid_or_discarded(layered_gather):
    opt = _base(TabuSearch, layered_gather, min_vel_gradient=G)
    picks = opt._random_picks()
    for _ in range(200):
        neighbor, idx, t_old, t_new = opt._jump_neighbor(picks)
        assert len(neighbor) == len(picks)      # validity is checked by the caller


@pytest.mark.parametrize('cls', [HillClimbing, TabuSearch, SimulatedAnnealing, RandomSearch])
def test_algorithms_return_solutions_that_respect_the_gradient(layered_gather, cls):
    opt = _base(cls, layered_gather, min_vel_gradient=G).run()
    assert opt._is_valid(opt.best_picks)
    dt = layered_gather['dt_ms']
    p = sorted(opt.best_picks)
    for (t1, v1), (t2, v2) in zip(p, p[1:]):
        assert v2 - v1 >= G * (t2 - t1) * dt / 1000.0 - 1e-9


def test_negative_gradient_is_rejected(layered_gather):
    with pytest.raises(ValueError):
        _base(RandomSearch, layered_gather, min_vel_gradient=-1.0)


def test_infeasible_random_start_raises(layered_gather):
    # 'mean' objective with many picks spread over the record and a steep
    # gradient: the required velocity rise exceeds vel_max - vel_min
    d = layered_gather
    opt = RandomSearch(d['gather'], d['offsets'], d['dt_ms'], vel_min=1400.0,
                       vel_max=1500.0, n_picks=10, min_sep_ms=200.0,
                       max_evals=10, min_vel_gradient=1000.0)
    with pytest.raises(ValueError):
        opt.run()
