"""Tests for src/optimizers/random_search.py."""

import numpy as np
import pytest

from src.optimizers.random_search import RandomSearch


def _rs(data, seed=0, budget=200, **kw):
    return RandomSearch(data['gather'], data['offsets'], data['dt_ms'],
                        vel_min=1400.0, vel_max=3000.0, n_picks=3,
                        seed=seed, max_evals=budget, **kw)


def test_uses_exactly_the_budget(layered_gather):
    rs = _rs(layered_gather, budget=150).run()
    assert rs.n_evals == 150


def test_budget_is_required(layered_gather):
    with pytest.raises(ValueError):
        _rs(layered_gather, budget=None)


def test_same_seed_same_result(layered_gather):
    a = _rs(layered_gather, seed=3).run()
    b = _rs(layered_gather, seed=3).run()
    assert a.best_picks == b.best_picks and a.best_score == b.best_score


def test_rerunning_the_same_object_is_reproducible(layered_gather):
    rs = _rs(layered_gather, seed=4)
    first = (list(rs.run().best_picks), rs.best_score)
    second = (list(rs.run().best_picks), rs.best_score)
    assert first == second


def test_result_respects_formulation_v2(layered_gather):
    rs = _rs(layered_gather, seed=5, min_sep_ms=100.0,
             t_min_ms=600.0, t_max_ms=2700.0).run()
    times = [t for t, _ in rs.best_picks]
    assert rs._is_valid(rs.best_picks)
    assert min(times) >= 300 and max(times) <= 1350
    assert min(np.diff(times)) >= 50


def test_best_score_is_the_best_evaluated(layered_gather):
    rs = _rs(layered_gather, seed=6).run()
    scores = [s for _, s in rs.eval_history]
    assert rs.best_score == max(scores)
    assert all(b > a for a, b in zip(scores, scores[1:]))
