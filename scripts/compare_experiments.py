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

from src.analysis.compare import paired_table    # noqa: E402


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
    if [r['seed'] for r in runs_a] != [r['seed'] for r in runs_b]:
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

    table = paired_table(runs_a, runs_b)
    for key, t in table.items():
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
