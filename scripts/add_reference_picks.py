"""
scripts/add_reference_picks.py
------------------------------
Add (or refresh) the semblance reference picks of the finite-difference
cases, without running the modeling again.

For every case whose meta lists the expected events (`primaries`, and
`surface_multiples` with a free surface), it finds the semblance peak of
each event (src.analysis.reference) twice and stores both lists in meta:

    reference_picks         — plain semblance (formulation v3)
    reference_picks_robust  — robust semblance (stretch mute, energy floor,
                              minimum fold; options in
                              reference_picks_robust_options, default
                              src.seismic.semblance.ROBUST_SEMBLANCE)

The reference must come from the same semblance as the objective being
evaluated; src.analysis.reference.get_reference picks the right one. The
gather and the other arrays are not touched.

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
from src.seismic.semblance import ROBUST_SEMBLANCE          # noqa: E402

# Search box around each expected event (see src/analysis/reference.py)
TIME_WIN_MS = 40.0       # same as the time tolerance of the metrics
VEL_WIN     = 0.08       # +- 8% of the model velocity
VEL_STEP    = 5.0        # m/s


def _print(title, refs):
    print(f"    {title}")
    for r in refs:
        flag = '  (on the box edge — check)' if r['on_edge'] else ''
        print(f"      {r['kind']:8s} {r['name']:26s} model ({r['t0_ms']:5.0f} ms, "
              f"{r['v_model']:5.0f} m/s) -> peak ({r['t_ms']:5.0f} ms, "
              f"{r['v']:5.0f} m/s)  S = {r['s']:.2f}{flag}")


def add_reference(path, robust_options=None, verbose=True):
    """
    Compute and store meta['reference_picks'] (plain semblance) and
    meta['reference_picks_robust'] (robust semblance) of one case.
    Returns (plain, robust) or None when the case has no expected events.
    """
    case = load_case(path)
    meta = case['meta']
    if 'primaries' not in meta:
        if verbose:
            print(f"  {case['name']}: no expected events in meta, skipped")
        return None

    opts = dict(ROBUST_SEMBLANCE if robust_options is None else robust_options)
    box = dict(time_win_ms=TIME_WIN_MS, vel_win=VEL_WIN, vel_step=VEL_STEP)
    args = (case['gather'], case['offsets'], case['dt_ms'],
            meta['primaries'], meta.get('surface_multiples', []))
    plain  = reference_picks(*args, **box)
    robust = reference_picks(*args, **box, **opts)

    meta['reference_picks'] = plain
    meta['reference_picks_robust'] = robust
    meta['reference_picks_robust_options'] = opts
    meta['reference_picks_note'] = (
        'Semblance peak (local maximum) of each expected event, searched within '
        f'+-{TIME_WIN_MS:g} ms and +-{100 * VEL_WIN:g}% of the model (t0, velocity): '
        'the stacking velocity a picker would choose. reference_picks uses the plain '
        'semblance, reference_picks_robust the robust one (options in '
        'reference_picks_robust_options). Evaluate with the one that matches the '
        'objective (src.analysis.reference.get_reference / match_reference).')
    save_case(path, case['gather'], case['offsets'], case['t_grid'],
              case['v_rms_true'], case['dt_ms'], meta)

    if verbose:
        print(f"  {case['name']}")
        _print('plain semblance', plain)
        _print(f'robust semblance {opts}', robust)
    return plain, robust


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