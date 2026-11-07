"""The per-pick semblance cache gives exactly the same runs, only faster."""

import pytest

from src.optimizers.hill_climbing import HillClimbing
from src.optimizers.random_search import RandomSearch
from src.optimizers.simulated_annealing import SimulatedAnnealing
from src.optimizers.tabu_search import TabuSearch
from src.seismic.semblance import ROBUST_SEMBLANCE


def _make(cls, d, cache, objective):
    common = dict(vel_min=1400.0, vel_max=3000.0, min_sep_ms=100.0, seed=7, max_evals=1500,
                  cache_semblance=cache, min_vel_gradient=100.0, **ROBUST_SEMBLANCE)
    if objective == 'threshold':
        common.update(objective='threshold', tau=0.2)
    else:
        common.update(n_picks=3)
    local = dict(step_time=5, step_vel=50.0)
    if cls is RandomSearch:
        return cls(d['gather'], d['offsets'], d['dt_ms'], **common)
    if cls is SimulatedAnnealing:
        return cls(d['gather'], d['offsets'], d['dt_ms'], **common, **local)
    local.update(n_neighbors=10, max_iter=10**5)
    if cls is HillClimbing:
        return cls(d['gather'], d['offsets'], d['dt_ms'], **common, **local,
                   restarts=10**4, patience=15)
    return cls(d['gather'], d['offsets'], d['dt_ms'], **common, **local,
               tabu_tenure=10, patience=10**5, p_jump=0.1)


@pytest.mark.parametrize('objective', ['threshold', 'mean'])
@pytest.mark.parametrize('cls', [RandomSearch, HillClimbing, TabuSearch, SimulatedAnnealing])
def test_cache_gives_identical_runs(layered_gather, cls, objective):
    if cls is SimulatedAnnealing and objective == 'mean':
        pytest.skip('covered by threshold; SA mean calibration is the same code path')
    a = _make(cls, layered_gather, False, objective).run()
    b = _make(cls, layered_gather, True, objective).run()
    assert a.best_score == b.best_score and a.best_picks == b.best_picks
    assert a.n_evals == b.n_evals and a.history == b.history
    if cls is not RandomSearch:                  # random solutions never repeat a pick
        assert b.cache_stats['hits'] > 0
    assert a.cache_stats == {'hits': 0, 'misses': 0}


def test_cached_value_equals_direct_evaluation(layered_gather):
    opt = _make(TabuSearch, layered_gather, True, 'threshold')
    picks = [(300, 1600.0), (500, 2000.0), (700, 2400.0)]
    direct = _make(TabuSearch, layered_gather, False, 'threshold')._evaluate(picks)
    assert opt._evaluate(picks) == direct
    assert opt._evaluate(picks) == direct            # second time from the cache
    assert opt.cache_stats == {'hits': 3, 'misses': 3}
