"""
src/analysis/events.py
----------------------
Reflection events along a semblance curve.

Used to validate the variable-K objective (formulation v3): along the TRUE
RMS velocity of a synthetic case, the local maxima of semblance that rise
above the threshold tau are the events an ideal velocity function would
pick. Their number is the "natural K" of the case. This uses the ground
truth, so it belongs to the evaluation, never to the optimizers.
"""

import numpy as np


def select_events(s_curve, tau, min_sep):
    """
    Local maxima of a semblance curve above a threshold, at least
    `min_sep` samples apart.

    Greedy non-maximum suppression: candidates (local maxima with value
    > tau) are visited from the strongest to the weakest, and a candidate
    is kept only if it is at least `min_sep` samples away from every
    event already kept. Within a cluster of nearby maxima, the strongest
    one wins.

    Parameters
    ----------
    s_curve : np.ndarray, shape (n_samples,) — e.g. semblance_along_curve()
    tau     : float — threshold; only maxima strictly above it count
    min_sep : int   — minimum separation between events, in samples
                      (same meaning as BaseOptimizer.min_sep)

    Returns
    -------
    events : np.ndarray of int — sample indices, in increasing order
    """
    s = np.asarray(s_curve, dtype=np.float64)
    if s.size < 3:
        return np.array([], dtype=int)

    inner = np.arange(1, s.size - 1)
    is_peak = (s[inner] >= s[inner - 1]) & (s[inner] >= s[inner + 1]) \
        & (s[inner] > tau)
    candidates = inner[is_peak]
    candidates = candidates[np.argsort(-s[candidates], kind='stable')]

    kept = []
    for i in candidates:
        if all(abs(int(i) - j) >= min_sep for j in kept):
            kept.append(int(i))
    return np.array(sorted(kept), dtype=int)