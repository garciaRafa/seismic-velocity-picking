"""
src/analysis/stats.py
---------------------
Descriptive statistics over independent runs.
"""

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
