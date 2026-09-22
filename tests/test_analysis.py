"""Tests for src/analysis (metrics and statistics)."""

import numpy as np
import pytest

from src.analysis.metrics import count_correct_picks
from src.analysis.stats import summarize, wilcoxon_signed_rank


def test_correct_picks_uses_relative_tolerance():
    t_grid = np.array([0.0, 1000.0, 2000.0])
    v_true = np.array([1500.0, 2000.0, 2500.0])
    # true velocities at the picks: 2000 and 2250 m/s -> tolerance 40 and 45
    out = count_correct_picks([1000.0, 1500.0], [2039.0, 2300.0],
                              t_grid, v_true, tol=0.02)
    assert out['correct'] == [True, False]
    assert out['n_correct'] == 1 and out['frac_correct'] == 0.5


def test_summarize():
    s = summarize([1.0, 2.0, 3.0, 4.0])
    assert s['n'] == 4 and s['mean'] == 2.5 and s['median'] == 2.5
    assert s['min'] == 1.0 and s['max'] == 4.0
    assert np.isclose(s['std'], np.std([1, 2, 3, 4], ddof=1))


def test_wilcoxon_matches_scipy_normal_approximation():
    # Reference values computed with scipy.stats.wilcoxon(a, b,
    # method='approx', correction=True): 0.000945 for this data
    a = [8,5,7,6,1,4,1,3,5,3,1,3,2,0,5,4,5,0,6,5,1,3,3,5,2,2,7,1,3,3]
    b = [7,4,3,7,2,4,7,3,8,5,2,10,5,4,6,5,6,2,3,8,1,6,7,9,3,6,8,3,5,5]
    assert wilcoxon_signed_rank(a, b)['p_value'] == pytest.approx(0.000945, abs=2e-6)


def test_wilcoxon_identical_samples():
    out = wilcoxon_signed_rank([1, 2, 3], [1, 2, 3])
    assert out['n'] == 0 and out['p_value'] == 1.0


def test_wilcoxon_is_symmetric():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=25), rng.normal(size=25)
    assert np.isclose(wilcoxon_signed_rank(a, b)['p_value'],
                      wilcoxon_signed_rank(b, a)['p_value'])
