"""
scripts/compare_experiments.py
------------------------------
Paired comparison of two experiments run with the same seeds
(same experiment.master_seed and n_runs): run i of A is compared with
run i of B.

Usage (from the project root):
    python scripts/compare_experiments.py <folder_A> <folder_B>

Prints, for each metric, the median and mean of A and B, how many runs
each one wins, and the two-sided Wilcoxon signed-rank p-value. The same
table is saved as comparison.json inside folder A.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.analysis.stats import wilcoxon_signed_rank    # noqa: E402

# metric -> True if higher is better
METRICS = {
    'best_score':  True,
    'n_correct':   True,
    'rmse':        False,
    'mae':         False,
    'exec_time_s': False,
    'n_evals':     False,
}


def load(folder):
    with open(os.path.join(folder, 'runs.json')) as f:
        runs = json.load(f)
    with open(os.path.join(folder, 'config.json')) as f:
        cfg = json.load(f)
    return runs, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder_a')
    parser.add_argument('folder_b')
    args = parser.parse_args()

    runs_a, cfg_a = load(args.folder_a)
    runs_b, cfg_b = load(args.folder_b)

    # Paired comparison only makes sense with the same seeds, in order
    seeds_a = [r['seed'] for r in runs_a]
    seeds_b = [r['seed'] for r in runs_b]
    if seeds_a != seeds_b:
        sys.exit("The experiments do not use the same seeds; "
                 "a paired comparison is not valid.")

    name_a = cfg_a['optimizer']['name']
    name_b = cfg_b['optimizer']['name']
    print(f"A = {name_a:14s} ({os.path.basename(os.path.normpath(args.folder_a))})")
    print(f"B = {name_b:14s} ({os.path.basename(os.path.normpath(args.folder_b))})")
    print(f"{len(runs_a)} paired runs\n")

    header = (f"{'metric':12s} {'median A':>10s} {'median B':>10s} "
              f"{'mean A':>10s} {'mean B':>10s} {'A wins':>7s} "
              f"{'B wins':>7s} {'p-value':>9s}")
    print(header)
    print('-' * len(header))

    table = {}
    for key, higher_is_better in METRICS.items():
        if key not in runs_a[0] or key not in runs_b[0]:
            continue
        a = np.array([r[key] for r in runs_a], dtype=float)
        b = np.array([r[key] for r in runs_b], dtype=float)
        better_a = a > b if higher_is_better else a < b
        better_b = b > a if higher_is_better else b < a
        test = wilcoxon_signed_rank(a, b)
        table[key] = {
            'median_a': float(np.median(a)), 'median_b': float(np.median(b)),
            'mean_a':   float(a.mean()),     'mean_b':   float(b.mean()),
            'a_wins':   int(better_a.sum()), 'b_wins':   int(better_b.sum()),
            'wilcoxon_p': test['p_value'],
        }
        t = table[key]
        print(f"{key:12s} {t['median_a']:10.3f} {t['median_b']:10.3f} "
              f"{t['mean_a']:10.3f} {t['mean_b']:10.3f} {t['a_wins']:7d} "
              f"{t['b_wins']:7d} {t['wilcoxon_p']:9.4f}")

    out = {'a': args.folder_a, 'b': args.folder_b,
           'optimizer_a': name_a, 'optimizer_b': name_b,
           'n_runs': len(runs_a), 'metrics': table}
    path = os.path.join(args.folder_a, 'comparison.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
