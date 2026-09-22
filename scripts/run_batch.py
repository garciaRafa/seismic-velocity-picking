"""
scripts/run_batch.py
--------------------
Run several algorithms on several test cases (N runs each, same seeds for
every algorithm and case) and compare them, case by case and overall.

Usage (from the project root):
    python scripts/run_batch.py configs/batch_t1_all_cases.json
    python scripts/run_batch.py configs/batch_t1_all_cases.json --workers 4
    python scripts/run_batch.py configs/batch_t1_all_cases.json --resume <batch_folder>

Output folder: results/experiments/<name>_<timestamp>/
    config.json, metadata.json   — batch config and machine/git metadata
    <case>/<Algorithm>/          — runs.json, summary.json and config.json of
                                   each experiment (same format as
                                   run_experiment.py, so compare_experiments.py
                                   and analyze_errors.py work on them)
    batch_summary.md / .json / .csv — comparison tables

--workers N runs N executions in parallel. Results are identical to a
sequential run (each run depends only on its seed), but execution times
are measured under load and are not comparable to sequential timings.

--resume continues an interrupted batch: runs already saved are skipped.
"""

import argparse
import csv
import itertools
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_experiment import (OPTIMIZERS, collect_metadata,    # noqa: E402
                            resolve_params, run_seeds, run_single,
                            summarize_runs)
from src.analysis.compare import paired_table               # noqa: E402
from src.analysis.stats import wilcoxon_signed_rank         # noqa: E402
from src.seismic.cases import list_cases, load_case         # noqa: E402

_CASE_CACHE = {}


def case_data(path):
    """Case in the dict format used by run_single (cached per process)."""
    if path not in _CASE_CACHE:
        c = load_case(path)
        _CASE_CACHE[path] = {
            'name': c['name'], 'gather': c['gather'], 'offsets': c['offsets'],
            't_grid': c['t_grid'], 'v_rms': c['v_rms_true'], 'dt_ms': c['dt_ms'],
            'water_bottom_ms': c['meta']['water_bottom_ms'],
        }
    return _CASE_CACHE[path]


def _task(args):
    """One run of one algorithm on one case (executed by a worker)."""
    case_path, algorithm, raw_params, seed, run_index, tol = args
    data   = case_data(case_path)
    params = resolve_params(raw_params, data)
    record = run_single(OPTIMIZERS[algorithm], params, data, seed, run_index, tol)
    return data['name'], algorithm, record


def experiment_config(batch, case_name, case_path, algorithm):
    """Config of one (case, algorithm) experiment, in run_experiment.py format."""
    params = dict(batch['formulation'])
    params.update(batch['algorithms'][algorithm])
    return {
        'name': f"{batch['name']}__{case_name}__{algorithm}",
        'model': {'source': 'case', 'path': os.path.relpath(case_path, ROOT)},
        'optimizer': {'name': algorithm, 'params': params},
        'experiment': {'n_runs': batch['n_runs'],
                       'master_seed': batch['master_seed'],
                       'correct_tol': batch.get('correct_tol', 0.02)},
    }


def load_runs(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_experiment(folder, cfg, runs):
    os.makedirs(folder, exist_ok=True)
    runs = sorted(runs, key=lambda r: r['run'])
    with open(os.path.join(folder, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(folder, 'runs.json'), 'w') as f:
        json.dump(runs, f, indent=2)
    if runs:
        with open(os.path.join(folder, 'summary.json'), 'w') as f:
            json.dump(summarize_runs(runs), f, indent=2)


# ------------------------------------------------------------------ #
# Summary tables                                                       #
# ------------------------------------------------------------------ #

def batch_summary(out_dir, cases, algorithms, results):
    """
    Per-case medians of every algorithm, pairwise Wilcoxon tests per case,
    and pooled tests over all (case, run) pairs.
    """
    pairs = list(itertools.combinations(algorithms, 2))
    per_case, pooled = [], {}

    for name in cases:
        row = {'case': name}
        for alg in algorithms:
            runs = results[name][alg]
            for key in ('n_correct', 'rmse', 'best_score'):
                row[f'{alg}.{key}.median'] = float(np.median([r[key] for r in runs]))
        for a, b in pairs:
            table = paired_table(results[name][a], results[name][b])
            for key in ('n_correct', 'rmse', 'best_score'):
                row[f'{a}_vs_{b}.{key}.p'] = table[key]['wilcoxon_p']
        per_case.append(row)

    for a, b in pairs:
        pooled[f'{a}_vs_{b}'] = {}
        for key in ('n_correct', 'rmse', 'best_score'):
            va = [r[key] for n in cases for r in results[n][a]]
            vb = [r[key] for n in cases for r in results[n][b]]
            test = wilcoxon_signed_rank(va, vb)
            pooled[f'{a}_vs_{b}'][key] = {
                'median_a': float(np.median(va)), 'median_b': float(np.median(vb)),
                'n_pairs': len(va), 'wilcoxon_p': test['p_value']}

    with open(os.path.join(out_dir, 'batch_summary.json'), 'w') as f:
        json.dump({'per_case': per_case, 'pooled': pooled}, f, indent=2)

    with open(os.path.join(out_dir, 'batch_summary.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_case[0].keys()))
        writer.writeheader()
        writer.writerows(per_case)

    # Markdown / screen table: correct picks and RMSE per case
    lines = ["# Batch summary", "",
             "Medians over runs. p = two-sided Wilcoxon signed-rank (paired by seed).", ""]
    for key, label in (('n_correct', 'Correct picks'), ('rmse', 'RMSE (m/s)'),
                       ('best_score', 'Score')):
        header = f"| Case | " + " | ".join(algorithms) + " | " + \
                 " | ".join(f"p {a} vs {b}" for a, b in pairs) + " |"
        lines += [f"## {label}", "", header,
                  "|" + "---|" * (1 + len(algorithms) + len(pairs))]
        for row in per_case:
            fmt = '.0f' if key == 'n_correct' else ('.1f' if key == 'rmse' else '.3f')
            vals = [format(row[f'{alg}.{key}.median'], fmt) for alg in algorithms]
            ps = [f"{row[f'{a}_vs_{b}.{key}.p']:.4f}" for a, b in pairs]
            lines.append(f"| {row['case']} | " + " | ".join(vals + ps) + " |")
        for a, b in pairs:
            pl = pooled[f'{a}_vs_{b}'][key]
            lines.append(f"| **all cases** ({pl['n_pairs']} pairs) | "
                         f"{format(pl['median_a'], fmt)} | {format(pl['median_b'], fmt)} | "
                         f"{pl['wilcoxon_p']:.4f} |" if len(algorithms) == 2 else
                         f"| **all cases**, {a} vs {b} | {format(pl['median_a'], fmt)} "
                         f"| {format(pl['median_b'], fmt)} | {pl['wilcoxon_p']:.4f} |")
        lines.append("")
    text = "\n".join(lines)
    with open(os.path.join(out_dir, 'batch_summary.md'), 'w') as f:
        f.write(text + "\n")
    return text


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', help='batch config (JSON)')
    parser.add_argument('--workers', type=int, default=1,
                        help='parallel runs (default 1 = sequential)')
    parser.add_argument('--resume', default=None, help='batch folder to continue')
    parser.add_argument('--out', default=os.path.join(ROOT, 'results', 'experiments'))
    args = parser.parse_args()

    with open(args.config) as f:
        batch = json.load(f)

    cases_dir = os.path.join(ROOT, batch.get('cases_dir', 'cases'))
    paths = list_cases(cases_dir)
    if batch.get('cases', 'all') != 'all':
        wanted = set(batch['cases'])
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in wanted]
    if not paths:
        sys.exit("No test cases found. Run: python scripts/generate_cases.py")
    cases = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    algorithms = list(batch['algorithms'])

    metadata = collect_metadata()
    metadata['workers'] = args.workers
    if metadata['git_dirty']:
        print('WARNING: uncommitted changes — results may not match git_commit.')

    if args.resume:
        out_dir = args.resume
    else:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join(args.out, f"{batch['name']}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(batch, f, indent=2)
    with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    seeds = run_seeds(batch['master_seed'], batch['n_runs'])
    tol   = batch.get('correct_tol', 0.02)

    # Existing results (resume) and pending tasks
    results, cfgs, tasks = {}, {}, []
    for case_name, path in zip(cases, paths):
        results[case_name] = {}
        for alg in algorithms:
            cfg = experiment_config(batch, case_name, path, alg)
            cfgs[(case_name, alg)] = cfg
            folder = os.path.join(out_dir, case_name, alg)
            done = {r['run']: r for r in load_runs(os.path.join(folder, 'runs.json'))}
            results[case_name][alg] = list(done.values())
            for i, seed in enumerate(seeds):
                if i not in done:
                    tasks.append((path, alg, cfg['optimizer']['params'], seed, i, tol))

    total = len(cases) * len(algorithms) * len(seeds)
    print(f"Batch      : {batch['name']}")
    print(f"Output     : {out_dir}")
    print(f"Cases      : {len(cases)}   Algorithms: {', '.join(algorithms)}   "
          f"Runs: {len(seeds)}   Workers: {args.workers}")
    print(f"Pending    : {len(tasks)} of {total} runs\n")

    t0 = time.perf_counter()
    completed = total - len(tasks)

    def handle(result):
        nonlocal completed
        name, alg, record = result
        results[name][alg].append(record)
        save_experiment(os.path.join(out_dir, name, alg), cfgs[(name, alg)],
                        results[name][alg])
        completed += 1
        elapsed = time.perf_counter() - t0
        print(f"  [{completed:4d}/{total}] {name:28s} {alg:13s} run {record['run'] + 1:3d}  "
              f"correct={record['n_correct']:2d}  rmse={record['rmse']:7.1f}  "
              f"({elapsed / 60:5.1f} min)", flush=True)

    if args.workers > 1:
        with Pool(args.workers) as pool:
            for result in pool.imap_unordered(_task, tasks):
                handle(result)
    else:
        for task in tasks:
            handle(_task(task))

    for name in cases:
        for alg in algorithms:
            results[name][alg].sort(key=lambda r: r['run'])

    print()
    print(batch_summary(out_dir, cases, algorithms, results))
    print(f"Saved: {os.path.join(out_dir, 'batch_summary.md')}")


if __name__ == '__main__':
    main()
