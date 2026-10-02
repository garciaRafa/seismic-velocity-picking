"""
scripts/plot_case.py
--------------------
Grayscale figure of one test case for the report: the CDP gather and its
semblance panel with the true RMS velocity.

Usage (from the project root):
    python scripts/plot_case.py marmousi2_cdp06800
    python scripts/plot_case.py marmousi2_cdp06800 --out results/figures

Saves fig_case_<name>.pdf and .png.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.seismic.cases import load_case                 # noqa: E402
from src.seismic.semblance import compute_semblance      # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('case', help='case name (file cases/<name>.npz)')
    parser.add_argument('--out', default=os.path.join(ROOT, 'results', 'figures'))
    parser.add_argument('--vmin', type=float, default=1400.0)
    parser.add_argument('--vmax', type=float, default=3000.0)
    args = parser.parse_args()

    c = load_case(os.path.join(ROOT, 'cases', f'{args.case}.npz'))
    vels = np.arange(args.vmin, args.vmax + 1, 20.0, dtype=np.float32)
    panel = compute_semblance(c['gather'], c['offsets'], vels, c['dt_ms'])
    t_max = c['t_grid'][-1]

    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4.2), sharey=True)

    clip = np.percentile(np.abs(c['gather']), 99)
    ax1.imshow(c['gather'].T, aspect='auto', cmap='gray', vmin=-clip, vmax=clip,
               extent=[c['offsets'][0], c['offsets'][-1], t_max, 0])
    ax1.set_xlabel('Offset (m)')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('CDP gather')

    ax2.imshow(panel.T, aspect='auto', cmap='gray_r', vmin=0, vmax=1,
               extent=[vels[0], vels[-1], t_max, 0])
    ax2.plot(c['v_rms_true'], c['t_grid'], color='white', lw=2.2)
    ax2.plot(c['v_rms_true'], c['t_grid'], color='black', lw=1.0, ls='--',
             label='True $V_{rms}$')
    ax2.set_xlabel('Velocity (m/s)')
    ax2.set_title('Semblance panel')
    ax2.legend(frameon=True, loc='upper right', fontsize=8)

    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = os.path.join(args.out, f'fig_case_{args.case}.{ext}')
        fig.savefig(path, dpi=200)
    print(f"Saved: {os.path.relpath(path[:-4], ROOT)}.pdf/.png")


if __name__ == '__main__':
    main()
