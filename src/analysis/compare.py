"""
src/analysis/compare.py
-----------------------
Paired comparison of two algorithms run with the same seeds.
"""

import numpy as np

from .stats import wilcoxon_signed_rank

# metric -> True if higher is better
METRICS = {
    'best_score':  True,
    'n_correct':   True,
    'rmse':        False,
    'mae':         False,
    'exec_time_s': False,
    'n_evals':     False,
}


def paired_table(runs_a, runs_b, metrics=METRICS):
    """
    Compare two lists of run records paired by position (run i of A with
    run i of B). The seeds must be the same, in the same order.

    Returns
    -------
    dict metric -> median_a, median_b, mean_a, mean_b, a_wins, b_wins,
                   wilcoxon_p
    """
    if [r['seed'] for r in runs_a] != [r['seed'] for r in runs_b]:
        raise ValueError("The runs do not use the same seeds; "
                         "a paired comparison is not valid.")

    table = {}
    for key, higher_is_better in metrics.items():
        if key not in runs_a[0] or key not in runs_b[0]:
            continue
        a = np.array([r[key] for r in runs_a], dtype=float)
        b = np.array([r[key] for r in runs_b], dtype=float)
        a_better = a > b if higher_is_better else a < b
        b_better = b > a if higher_is_better else b < a
        table[key] = {
            'median_a': float(np.median(a)), 'median_b': float(np.median(b)),
            'mean_a':   float(a.mean()),     'mean_b':   float(b.mean()),
            'a_wins':   int(a_better.sum()), 'b_wins':   int(b_better.sum()),
            'wilcoxon_p': wilcoxon_signed_rank(a, b)['p_value'],
        }
    return table
