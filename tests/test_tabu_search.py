"""Tests for src/optimizers/tabu_search.py."""

import numpy as np
import pytest

from src.optimizers.tabu_search import TabuSearch

V2 = dict(min_sep_ms=100.0, t_min_ms=600.0, t_max_ms=2700.0)


def _ts(data, seed=0, budget=600, max_iter=1000, patience=1000, **kw):
    return TabuSearch(data['gather'], data['offsets'], data['dt_ms'],
                      vel_min=1400.0, vel_max=3000.0, n_picks=3,
                      seed=seed, max_evals=budget, max_iter=max_iter,
                      patience=patience, n_neighbors=10, **kw)


# ---------------------------------------------------------------------- #
# Reproducibility                                                         #
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize('p_jump', [0.0, 0.3])
def test_same_seed_same_result(layered_gather, p_jump):
    a = _ts(layered_gather, seed=3, p_jump=p_jump, **V2).run()
    b = _ts(layered_gather, seed=3, p_jump=p_jump, **V2).run()
    assert a.best_picks == b.best_picks and a.best_score == b.best_score


def test_rerunning_the_same_object_is_reproducible(layered_gather):
    ts = _ts(layered_gather, seed=4, p_jump=0.3, **V2)
    first = (list(ts.run().best_picks), ts.best_score, dict(ts.jump_stats))
    second = (list(ts.run().best_picks), ts.best_score, dict(ts.jump_stats))
    assert first == second


def test_global_random_state_is_untouched(layered_gather):
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    _ts(layered_gather, seed=5, p_jump=0.3, **V2).run()
    assert np.array_equal(before, np.random.get_state()[1])


# ---------------------------------------------------------------------- #
# Budget and stopping                                                     #
# ---------------------------------------------------------------------- #

def test_uses_exactly_the_budget(layered_gather):
    ts = _ts(layered_gather, budget=400, **V2).run()
    assert ts.n_evals == 400
    assert ts.stop_reason == 'budget'


def test_patience_stops_the_search(layered_gather):
    ts = _ts(layered_gather, budget=None, patience=5, **V2).run()
    assert ts.stop_reason == 'patience'
    # The last `patience` iterations did not improve the global best
    assert len(set(ts.history[-5:])) == 1


def test_max_iter_stops_the_search(layered_gather):
    ts = _ts(layered_gather, budget=None, max_iter=7, **V2).run()
    assert ts.stop_reason == 'max_iter'
    assert len(ts.history) == 7


# ---------------------------------------------------------------------- #
# Solution quality bookkeeping                                            #
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize('p_jump', [0.0, 0.3, 1.0])
def test_result_respects_formulation_v2(layered_gather, p_jump):
    ts = _ts(layered_gather, seed=6, p_jump=p_jump, **V2).run()
    times = [t for t, _ in ts.best_picks]
    assert ts._is_valid(ts.best_picks)
    assert min(times) >= 300 and max(times) <= 1350
    assert min(np.diff(times)) >= 50


def test_best_score_never_decreases(layered_gather):
    ts = _ts(layered_gather, seed=7, p_jump=0.3, **V2).run()
    assert all(b >= a for a, b in zip(ts.history, ts.history[1:]))
    assert ts.best_score == max(s for _, s in ts.eval_history)


# ---------------------------------------------------------------------- #
# Tabu list                                                               #
# ---------------------------------------------------------------------- #

def test_active_tabu_list_never_exceeds_tenure(layered_gather):
    ts = _ts(layered_gather, seed=8, tabu_tenure=4, **V2)
    ts._reset_run_state()
    ts._initialize()
    for _ in range(30):
        ts._iterate()
        assert ts.active_tabu_size <= 4
    assert ts.n_tabu_attributes >= ts.active_tabu_size


# ---------------------------------------------------------------------- #
# Jump move                                                               #
# ---------------------------------------------------------------------- #

def test_invalid_p_jump_is_rejected(layered_gather):
    with pytest.raises(ValueError):
        _ts(layered_gather, p_jump=1.5)


def test_no_jumps_when_p_jump_is_zero(layered_gather):
    ts = _ts(layered_gather, seed=9, p_jump=0.0, **V2).run()
    assert ts.jump_stats == {'proposed': 0, 'accepted': 0}


def test_jump_neighbors_are_always_valid(layered_gather):
    ts = _ts(layered_gather, seed=10, **V2)
    ts.rng = np.random.default_rng(10)
    current = ts._random_picks()
    for _ in range(2000):
        neighbor, idx, t_old, t_new = ts._jump_neighbor(current)
        assert ts._is_valid(neighbor)
        assert neighbor[idx][0] == t_new and current[idx][0] == t_old
        # Only the chosen pick changes; order is preserved
        assert [p for i, p in enumerate(neighbor) if i != idx] == \
               [p for i, p in enumerate(current) if i != idx]


def test_all_neighbors_are_jumps_when_p_jump_is_one(layered_gather):
    ts = _ts(layered_gather, seed=11, p_jump=1.0, budget=300, **V2).run()
    # Every evaluation after the initial solution was a (valid) jump
    assert ts.jump_stats['proposed'] == ts.n_evals - 1
    assert 0 < ts.jump_stats['accepted'] <= len(ts.history)


def test_jump_explores_the_whole_feasible_box(layered_gather):
    """Jumps of the middle pick cover its whole time window."""
    ts = _ts(layered_gather, seed=12, **V2)
    ts.rng = np.random.default_rng(12)
    current = [(400, 1500.0), (800, 2000.0), (1200, 2500.0)]
    new_times = [t_new for _ in range(3000)
                 for neighbor, idx, _, t_new in [ts._jump_neighbor(current)]
                 if idx == 1]
    # Middle pick: between 400 + 50 and 1200 - 50 samples
    assert min(new_times) >= 450 and max(new_times) <= 1150
    assert min(new_times) < 500 and max(new_times) > 1100