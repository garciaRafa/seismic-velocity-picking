"""
scripts/add_reference_picks.py
------------------------------
Add (or refresh) the semblance reference picks of the finite-difference
cases, without running the modeling again.

For every case whose meta lists the expected events (`primaries`, and
`surface_multiples` with a free surface), it finds the semblance peak of
each event (src.analysis.reference) and stores the list in
meta['reference_picks']. The gather and the other arrays are not touched.

Usage, from the project root:

    python scripts/add_reference_picks.py                  # every fd_* case
    python scripts/add_reference_picks.py cases/fd_layered_cdp2500_fs.npz
"""

import argparse
import glob
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.analysis.reference import reference_picks          # noqa: E402
from src.seismic.cases import load_case, save_case          # noqa: E402

# Search box around each expected event (see src/analysis/reference.py)
TIME_WIN_MS = 40.0       # same as the time tolerance of the metrics
VEL_WIN     = 0.08       # +- 8% of the model velocity
VEL_STEP    = 5.0        # m/s


def add_reference(path, verbose=True):
    """Compute and store meta['reference_picks'] of one case. Returns the list."""
    case = load_case(path)
    meta = case['meta']
    if 'primaries' not in meta:
        if verbose:
            print(f"  {case['name']}: no expected events in meta, skipped")
        return None

    refs = reference_picks(case['gather'], case['offsets'], case['dt_ms'],
                           meta['primaries'], meta.get('surface_multiples', []),
                           time_win_ms=TIME_WIN_MS, vel_win=VEL_WIN, vel_step=VEL_STEP)
    meta['reference_picks'] = refs
    meta['reference_picks_note'] = (
        'Semblance peak (local maximum) of each expected event, searched within '
        f'+-{TIME_WIN_MS:g} ms and +-{100 * VEL_WIN:g}% of the model (t0, velocity). '
        'This is the stacking velocity a picker would choose; use it as the '
        'evaluation reference (src.analysis.reference.match_reference).')
    save_case(path, case['gather'], case['offsets'], case['t_grid'],
              case['v_rms_true'], case['dt_ms'], meta)

    if verbose:
        print(f"  {case['name']}")
        for r in refs:
            flag = '  (on the box edge — check)' if r['on_edge'] else ''
            print(f"    {r['kind']:8s} {r['name']:26s} model ({r['t0_ms']:5.0f} ms, "
                  f"{r['v_model']:5.0f} m/s) -> peak ({r['t_ms']:5.0f} ms, "
                  f"{r['v']:5.0f} m/s)  S = {r['s']:.2f}{flag}")
    return refs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='*', help='case files (default: cases/fd_*.npz)')
    args = parser.parse_args()
    paths = args.paths or sorted(glob.glob(os.path.join(ROOT, 'cases', 'fd_*.npz')))
    for path in paths:
        add_reference(path)


if __name__ == '__main__':
    main()