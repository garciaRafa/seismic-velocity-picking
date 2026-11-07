"""
scripts/rescore_batch.py
------------------------
Re-evaluate a finished batch (formulation v3, finite-difference cases)
with another reference, without running the algorithms again: the picks
saved in every runs.json are matched against the new reference and a new
batch summary is written to a subfolder. The original files are not
changed.

Typical use: score an old batch with the alternate maxima of the
reference events (src.analysis.reference, alt_ratio), so that it can be
compared with a batch run with "reference_alt_ratio" in its formulation.

Usage, from the project root:

    python scripts/rescore_batch.py results/experiments/<batch folder> --alt-ratio 0.9

Output: <batch folder>/rescored_alt<ratio>/batch_summary.{md,json,csv}
"""

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_batch import batch_summary, load_runs                 # noqa: E402
from run_experiment import semblance_options                   # noqa: E402
from src.analysis.reference import get_reference, match_reference  # noqa: E402
from src.seismic.cases import load_case                        # noqa: E402


def rescore_record(record, reference, tol=0.02, time_tol_ms=40.0):
    """Copy of a run record with the detection scores recomputed."""
    m = match_reference(record['picks_time_ms'], record['picks_vel_ms'], reference,
                        time_tol_ms=time_tol_ms, vel_tol=tol)
    out = dict(record)
    out.update({'n_picks': m['n_picks'], 'n_events': m['n_primaries'],
                'precision': m['precision'], 'recall': m['recall'], 'f1': m['f1'],
                'n_on_multiple': m['n_on_multiple'], 'pick_kind': m['kind']})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('batch_dir', help='folder of a finished batch (with config.json)')
    parser.add_argument('--alt-ratio', type=float, default=0.9,
                        help='accept picks on maxima with S >= ratio * S_peak (default 0.9)')
    args = parser.parse_args()

    with open(os.path.join(args.batch_dir, 'config.json')) as f:
        batch = json.load(f)
    if batch['formulation'].get('objective') != 'threshold':
        sys.exit('Only formulation v3 batches (objective "threshold") have detection scores.')
    opts = semblance_options(batch['formulation'])
    tol = batch.get('correct_tol', 0.02)
    cases_dir = os.path.join(ROOT, batch.get('cases_dir', 'cases'))

    cases = sorted(d for d in os.listdir(args.batch_dir)
                   if os.path.isdir(os.path.join(args.batch_dir, d)) and
                   os.path.exists(os.path.join(cases_dir, f'{d}.npz')))
    algorithms = list(batch['algorithms'])

    results = {}
    for name in cases:
        case = load_case(os.path.join(cases_dir, f'{name}.npz'))
        if 'primaries' not in case['meta']:
            sys.exit(f'{name}: no expected events in meta (only FD cases can be rescored).')
        ref = get_reference(case, alt_ratio=args.alt_ratio, **opts)
        results[name] = {}
        for alg in algorithms:
            runs = load_runs(os.path.join(args.batch_dir, name, alg, 'runs.json'))
            results[name][alg] = [rescore_record(r, ref, tol) for r in runs]
        print(f'  {name}: rescored')

    out_dir = os.path.join(args.batch_dir, f'rescored_alt{args.alt_ratio:g}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'rescore.json'), 'w') as f:
        json.dump({'source': os.path.abspath(args.batch_dir), 'alt_ratio': args.alt_ratio,
                   'semblance_options': opts, 'vel_tol': tol}, f, indent=2)
    print(batch_summary(out_dir, cases, algorithms, results, groups=batch.get('groups')))
    print(f'\nWritten to {out_dir}')


if __name__ == '__main__':
    main()