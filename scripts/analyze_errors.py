"""
scripts/analyze_errors.py
-------------------------
Per-pick error analysis of two experiments (e.g. hill climbing vs random
search), producing grayscale figures for the report and a JSON summary.

Usage (from the project root):
    python scripts/analyze_errors.py <folder_A> <folder_B> [--out <folder>]

Outputs (default: inside folder A, subfolder error_analysis/):
    fig_error_hist.pdf/.png    — distribution of |error| per pick
    fig_error_vs_time.pdf/.png — signed error of every pick vs pick time
    fig_convergence.pdf/.png   — median best score vs evaluations
    error_analysis.json        — error classes and statistics per algorithm

Error classes (relative to the true RMS velocity at the pick time):
    correct : |error| <= 2%
    small   : 2% < |error| <= 5%
    large   : |error| > 5%
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

CLASSES = (('correct', 0.00, 0.02), ('small', 0.02, 0.05), ('large', 0.05, np.inf))
GRAYS = ('0.15', '0.60')        # A, B
MARKERS = ('o', 's')


def load(folder):
    with open(os.path.join(folder, 'runs.json')) as f:
        runs = json.load(f)
    with open(os.path.join(folder, 'config.json')) as f:
        cfg = json.load(f)
    return runs, cfg


def pick_table(runs):
    """Flatten all picks of all runs: time (ms), signed error, relative error."""
    t, e, rel = [], [], []
    for r in runs:
        for ti, ei, vt in zip(r['picks_time_ms'], r['errors'], r['v_true']):
            t.append(ti)
            e.append(ei)
            rel.append(ei / vt)
    return np.array(t), np.array(e), np.array(rel)


def classify(rel):
    """Count picks in each error class (lower bound exclusive, upper inclusive)."""
    a = np.abs(rel)
    counts = {}
    for name, lo, hi in CLASSES:
        inside = (a <= hi) if lo == 0 else ((a > lo) & (a <= hi))
        counts[name] = int(inside.sum())
    return counts


def median_convergence(runs, budget, n_points=200):
    """Median over runs of the best score after e evaluations."""
    grid = np.unique(np.linspace(1, budget, n_points).astype(int))
    curves = []
    for r in runs:
        hist = np.array(r['eval_history'], dtype=float)   # (n_evals, best)
        idx = np.searchsorted(hist[:, 0], grid, side='right') - 1
        curve = np.where(idx >= 0, hist[np.clip(idx, 0, None), 1], np.nan)
        curves.append(curve)
    curves = np.array(curves)
    return grid, np.nanmedian(curves, axis=0), \
        np.nanpercentile(curves, 25, axis=0), np.nanpercentile(curves, 75, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder_a')
    parser.add_argument('folder_b')
    parser.add_argument('--out', default=None, help='output folder')
    args = parser.parse_args()

    runs_a, cfg_a = load(args.folder_a)
    runs_b, cfg_b = load(args.folder_b)
    names = (cfg_a['optimizer']['name'], cfg_b['optimizer']['name'])
    out = args.out or os.path.join(args.folder_a, 'error_analysis')
    os.makedirs(out, exist_ok=True)

    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
    tables = [pick_table(runs_a), pick_table(runs_b)]

    # ---------------------------------------------------------------- #
    # Summary                                                          #
    # ---------------------------------------------------------------- #
    summary = {}
    for name, (t, e, rel), runs in zip(names, tables, (runs_a, runs_b)):
        n = len(e)
        cls = classify(rel)
        large = np.abs(rel) > 0.05
        summary[name] = {
            'n_picks': n,
            'classes': {k: {'count': v, 'fraction': v / n} for k, v in cls.items()},
            'abs_error_m_s': {
                'median': float(np.median(np.abs(e))),
                'p90':    float(np.percentile(np.abs(e), 90)),
                'max':    float(np.max(np.abs(e))),
            },
            'large_errors_positive_fraction':
                float((e[large] > 0).mean()) if large.any() else None,
            'large_errors_in_deepest_pick':
                int(sum(1 for r in runs
                        if abs(r['errors'][-1] / r['v_true'][-1]) > 0.05)),
            'runs_with_large_error': int(sum(
                1 for r in runs
                if any(abs(ei / vt) > 0.05 for ei, vt in zip(r['errors'], r['v_true'])))),
        }
    with open(os.path.join(out, 'error_analysis.json'), 'w') as f:
        json.dump({'a': args.folder_a, 'b': args.folder_b, 'summary': summary}, f, indent=2)

    print(f"{'':28s} {names[0]:>14s} {names[1]:>14s}")
    for key in ('correct', 'small', 'large'):
        vals = [summary[n]['classes'][key] for n in names]
        print(f"picks {key:22s} " + " ".join(
            f"{v['count']:5d} ({100 * v['fraction']:4.1f}%)" for v in vals))
    for key in ('median', 'p90', 'max'):
        print(f"|error| {key:20s} " + " ".join(
            f"{summary[n]['abs_error_m_s'][key]:13.1f} " for n in names))
    print("large errors that are too fast " + " ".join(
        f"{100 * (summary[n]['large_errors_positive_fraction'] or 0):12.1f}% " for n in names))
    print("runs with a large error        " + " ".join(
        f"{summary[n]['runs_with_large_error']:13d} " for n in names))
    print("deepest pick with large error  " + " ".join(
        f"{summary[n]['large_errors_in_deepest_pick']:13d} " for n in names))

    # ---------------------------------------------------------------- #
    # Figure 1: distribution of |error|                                #
    # ---------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(6, 3))
    bins = np.arange(0, max(np.abs(tables[0][1]).max(), np.abs(tables[1][1]).max()) + 50, 50)
    for (t, e, rel), name, g in zip(tables, names, GRAYS):
        ax.hist(np.abs(e), bins=bins, histtype='stepfilled' if g == GRAYS[1] else 'step',
                color=g, edgecolor=g, linewidth=1.4, label=name, alpha=0.9)
    ax.set_xlabel('|Error| per pick (m/s)')
    ax.set_ylabel('Number of picks')
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'fig_error_hist.{ext}'), dpi=200)
    plt.close(fig)

    # ---------------------------------------------------------------- #
    # Figure 2: signed error vs pick time                              #
    # ---------------------------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    for ax, (t, e, rel), name, g, m in zip(axes, tables, names, GRAYS, MARKERS):
        ax.axhline(0, color='0.5', lw=0.8)
        ax.scatter(t, e, s=8, color=g if g != GRAYS[1] else '0.35', marker=m, alpha=0.6)
        ax.set_title(name)
        ax.set_xlabel('Pick time (ms)')
    axes[0].set_ylabel('Error (m/s)')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'fig_error_vs_time.{ext}'), dpi=200)
    plt.close(fig)

    # ---------------------------------------------------------------- #
    # Figure 3: convergence (median best score vs evaluations)         #
    # ---------------------------------------------------------------- #
    budget = max(max(r['n_evals'] for r in runs_a), max(r['n_evals'] for r in runs_b))
    fig, ax = plt.subplots(figsize=(6, 3))
    for runs, name, g, ls in zip((runs_a, runs_b), names, GRAYS, ('-', '--')):
        x, med, q1, q3 = median_convergence(runs, budget)
        ax.plot(x, med, color=g, ls=ls, lw=1.5, label=f'{name} (median)')
        ax.fill_between(x, q1, q3, color=g, alpha=0.15, lw=0)
    ax.set_xlabel('Objective evaluations')
    ax.set_ylabel('Best score')
    ax.legend(frameon=False, loc='lower right')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'fig_convergence.{ext}'), dpi=200)
    plt.close(fig)

    print(f"\nSaved figures and error_analysis.json in: {out}")


if __name__ == '__main__':
    main()
