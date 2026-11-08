"""Tests for the Memetic Algorithm."""

import numpy as np
import pytest

from src.optimizers.memetic import MemeticAlgorithm


def _make(d, **kw):
    opts = dict(vel_min=1400.0, vel_max=3000.0, min_sep_ms=100.0, objective='threshold',
                tau=0.2, seed=5, max_evals=3000, min_vel_gradient=100.0, pop_size=8)
    opts.update(kw)
    return MemeticAlgorithm(d['gather'], d['offsets'], d['dt_ms'], **opts)


def test_budget_is_respected_exactly(layered_gather):
    opt = _make(layered_gather).run()
    assert opt.n_evals == 3000
    assert opt.best_picks is not None and opt._is_valid(opt.best_picks)


def test_same_seed_same_run(layered_gather):
    a = _make(layered_gather).run()
    b = _make(layered_gather).run()
    assert a.best_score == b.best_score and a.best_picks == b.best_picks
    assert a.history == b.history


def test_population_stays_valid_and_scores_are_right(layered_gather):
    opt = _make(layered_gather).run()
    for picks, score in opt.population:
        assert opt._is_valid(picks)
        assert score == opt._evaluate(picks)
    assert opt.best_score >= max(s for _, s in opt.population)


def test_crossover_takes_a_before_the_cut_and_b_after(layered_gather):
    opt = _make(layered_gather)
    opt.rng = np.random.default_rng(0)
    a = [(100, 1500.0), (300, 1700.0), (500, 1900.0)]
    b = [(150, 1550.0), (350, 1750.0), (550, 1950.0)]
    for _ in range(50):
        child = opt._crossover(a, b)
        assert child == sorted(child)
        # every pick from a comes before every pick from b (one cut)
        from_a = [p for p in child if p in a]
        from_b = [p for p in child if p in b]
        assert len(from_a) + len(from_b) == len(child)
        if from_a and from_b:
            assert max(t for t, _ in from_a) < min(t for t, _ in from_b)


def test_two_cuts_give_a_b_a(layered_gather):
    opt = _make(layered_gather, n_cuts=2)
    opt.rng = np.random.default_rng(1)
    a = [(t, 1500.0 + t) for t in range(100, 1000, 100)]
    b = [(t + 50, 1500.0 + t) for t in range(100, 1000, 100)]
    seen_aba = False
    for _ in range(100):
        src = ''.join('a' if p in a else 'b' for p in opt._crossover(a, b))
        runs = [ch for i, ch in enumerate(src) if i == 0 or ch != src[i - 1]]
        assert ''.join(runs) in ('', 'a', 'b', 'ab', 'ba', 'aba')
        seen_aba |= ''.join(runs) == 'aba'
    assert seen_aba


def test_repair_drops_picks_that_break_the_constraints(layered_gather):
    opt = _make(layered_gather)                       # min_sep 100 ms = 50 samples
    picks = [(100, 1500.0), (120, 1600.0), (300, 1450.0), (400, 1800.0)]
    fixed = opt._repair(picks)
    assert fixed == [(100, 1500.0), (400, 1800.0)]
    assert opt._is_valid(fixed)


def test_n_unmatched(layered_gather):
    opt = _make(layered_gather)                       # tolerance 40 ms = 20 samples, 2%
    a = [(100, 1500.0), (300, 1700.0)]
    assert opt.n_unmatched(a, a) == 0
    assert opt.n_unmatched(a, [(110, 1510.0), (305, 1690.0)]) == 0
    assert opt.n_unmatched(a, [(100, 1500.0)]) == 1
    assert opt.n_unmatched(a, [(100, 1500.0), (300, 1900.0)]) == 2
    assert opt.n_unmatched(a, [(100, 1500.0), (300, 1700.0), (600, 2000.0)]) == 1


def test_population_has_no_twins_after_a_run(layered_gather):
    opt = _make(layered_gather, max_evals=4000).run()
    pop = [p for p, _ in opt.population]
    twins = sum(opt.n_unmatched(pop[i], pop[j]) == 0
                for i in range(len(pop)) for j in range(i + 1, len(pop)))
    # children never add a twin; only the initial population or a restart
    # (random solutions) could
    assert twins <= len(pop)
    assert opt.stats['children'] > 0


def test_cache_and_trajectory_do_not_change_the_run(layered_gather):
    a = _make(layered_gather).run()
    b = _make(layered_gather, cache_semblance=True, record_trajectory=True).run()
    assert a.best_score == b.best_score and a.best_picks == b.best_picks
    assert b.trajectory and {'child', 'accepted', 'pop_best'} <= set(b.trajectory[-1])


def test_bad_parameters(layered_gather):
    with pytest.raises(ValueError):
        _make(layered_gather, max_evals=None)
    with pytest.raises(ValueError):
        _make(layered_gather, pop_size=1)
    with pytest.raises(ValueError):
        _make(layered_gather, n_cuts=0)