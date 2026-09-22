"""
src/analysis/stats.py
---------------------
Descriptive statistics over independent runs.
"""

import math

import numpy as np


def summarize(values):
    """
    Summary statistics of a list of values (one per independent run).

    Returns
    -------
    dict with n, mean, std (sample, ddof=1), median, min, max
    """
    x = np.asarray(values, dtype=float)
    return {
        'n':      int(x.size),
        'mean':   float(np.mean(x)),
        'std':    float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        'median': float(np.median(x)),
        'min':    float(np.min(x)),
        'max':    float(np.max(x)),
    }


def wilcoxon_signed_rank(a, b):
    """
    Two-sided Wilcoxon signed-rank test for paired samples, NumPy only.

    Used to compare two algorithms run with the same seeds (run i of
    algorithm A is paired with run i of algorithm B).

    Follows the usual convention: pairs with zero difference are dropped,
    tied absolute differences get average ranks, and the p-value uses the
    normal approximation with tie correction and continuity correction.
    For n >= 20 pairs this is very close to the exact test.

    Parameters
    ----------
    a, b : sequences of the same length (paired values)

    Returns
    -------
    dict with
        n         : number of non-zero differences used
        w_plus    : sum of ranks of positive differences (a > b)
        w_minus   : sum of ranks of negative differences (a < b)
        statistic : min(w_plus, w_minus)
        p_value   : two-sided p-value (1.0 when n == 0)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same length (paired samples)")

    d = a - b
    d = d[d != 0]
    n = d.size
    if n == 0:
        return {'n': 0, 'w_plus': 0.0, 'w_minus': 0.0,
                'statistic': 0.0, 'p_value': 1.0}

    # Average ranks of |d| (ties share the mean rank)
    absd  = np.abs(d)
    order = np.argsort(absd, kind='mergesort')
    ranks = np.empty(n)
    sorted_abs = absd[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1

    w_plus  = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    stat    = min(w_plus, w_minus)

    # Normal approximation with tie correction
    _, counts = np.unique(sorted_abs, return_counts=True)
    mean = n * (n + 1) / 4.0
    var  = n * (n + 1) * (2 * n + 1) / 24.0
    var -= np.sum(counts ** 3 - counts) / 48.0
    if var <= 0:
        return {'n': int(n), 'w_plus': w_plus, 'w_minus': w_minus,
                'statistic': stat, 'p_value': 1.0}

    z = (abs(w_plus - mean) - 0.5) / np.sqrt(var)   # continuity correction
    z = max(z, 0.0)
    p = float(np.clip(math.erfc(z / np.sqrt(2.0)), 0.0, 1.0))

    return {'n': int(n), 'w_plus': w_plus, 'w_minus': w_minus,
            'statistic': stat, 'p_value': p}
