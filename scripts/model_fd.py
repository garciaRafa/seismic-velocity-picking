"""
scripts/model_fd.py
-------------------
Finite-difference modeling of the 2D synthetic model described in a
config file (default configs/synthetic_fd_model.json).

For every CDP position and every boundary option (with / without free
surface) it:
  1. simulates one shot per offset (source at cdp_x - h/2) with Devito;
  2. keeps the trace at receiver cdp_x + h/2 of each shot -> CDP gather;
  3. applies a direct-wave mute and a time gain;
  4. saves the CDP gather as a test case in cases/ (same format as the
     other cases, so every algorithm runs on it unchanged), with the
     column V_rms as ground truth and the expected primaries and surface
     multiples in its meta;
  5. saves three full shot records (near, middle and far offset shots)
     for display in data/synthetic_fd/ (line_idx selects the receivers of
     the regular line, with uniform spacing).

The model is deeper than the record needs: the absorbing layer still
reflects a small part of the energy (~0.5%), so the bottom of the grid is
placed where that reflection arrives after the end of the record.

Needs Devito (pip install devito). Usage, from the project root:

    python scripts/model_fd.py
    python scripts/model_fd.py --config configs/synthetic_fd_model.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.seismic.cases import save_case                                    # noqa: E402
from src.seismic.fd_modeling import (AcousticModeler, cdp_geometry,         # noqa: E402
                                     direct_wave_mute, receiver_line,
                                     time_gain)
from src.seismic.model2d import (build_layered_model, column_event_times,  # noqa: E402
                                 column_vrms, surface_multiples)

# Imported here (not at the top of generate_cases) to refresh cases/README.md
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from generate_cases import write_readme                                    # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=os.path.join(ROOT, 'configs',
                                                         'synthetic_fd_model.json'))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    acq, fd, proc = cfg['acquisition'], cfg['modeling'], cfg['processing']

    model   = build_layered_model(cfg['model'])
    offsets = np.linspace(acq['offset_min_m'], acq['offset_max_m'],
                          acq['n_offsets']).astype(np.float32)
    t_out   = np.arange(0.0, acq['t_max_ms'] + acq['dt_ms'] / 2,
                        acq['dt_ms']).astype(np.float32)

    # One receiver line for all shots: regular line + every CDP receiver
    cdp_receivers = np.concatenate([cdp_geometry(x, offsets)[1]
                                    for x in acq['cdp_x_m']])
    rec_x, _ = receiver_line(cfg['model']['width_m'],
                             acq['receiver_spacing_m'], cdp_receivers)
    # Receivers of the regular line only (for displaying shot records with a
    # uniform x axis; the extra CDP receivers make the spacing irregular)
    line = np.arange(0.0, cfg['model']['width_m'] + 1e-9, acq['receiver_spacing_m'])
    line_idx = np.searchsorted(rec_x, line)

    out_dir = os.path.join(ROOT, 'data', 'synthetic_fd')
    os.makedirs(out_dir, exist_ok=True)

    for free_surface in fd['free_surface']:
        tag = 'fs' if free_surface else 'nofs'
        print(f"\n=== free surface: {free_surface} ===")
        t0 = time.perf_counter()
        modeler = AcousticModeler(
            model['vp'], model['dx'], rec_x, acq['t_max_ms'], acq['f0_hz'],
            free_surface=free_surface,
            rec_depth_m=acq['receiver_depth_m'], src_depth_m=acq['source_depth_m'],
            space_order=fd['space_order'], absorbing_cells=fd['absorbing_cells'],
            courant=fd['courant'])
        print(f"operator ready: dt = {1000 * modeler.dt_s:.3f} ms, "
              f"{modeler.nt} steps per shot")

        for cdp_x in acq['cdp_x_m']:
            src, rec = cdp_geometry(cdp_x, offsets)
            rec_idx = np.searchsorted(rec_x, rec)
            gather = np.zeros((len(offsets), len(t_out)), dtype=np.float32)
            keep = {0: None, len(offsets) // 2: None, len(offsets) - 1: None}

            for i, (sx, ir) in enumerate(zip(src, rec_idx)):
                record = modeler.shot(float(sx), t_out)
                gather[i] = record[ir]
                if i in keep:
                    keep[i] = record
                print(f"  CDP {cdp_x:.0f} m — shot {i + 1:2d}/{len(offsets)} "
                      f"(source {sx:7.1f} m)   {time.perf_counter() - t0:6.0f} s",
                      end='\r')
            print()

            processed = direct_wave_mute(gather, offsets, t_out,
                                         proc['mute_velocity_m_s'], proc['mute_pad_ms'])
            processed = time_gain(processed, t_out, proc['gain_power'])

            _, v_rms = column_vrms(model, cdp_x, acq['dt_ms'], acq['t_max_ms'])
            primaries = column_event_times(model, cdp_x)
            multiples = surface_multiples(primaries) if free_surface else []

            name = f"{cfg['name']}_cdp{int(cdp_x):04d}_{tag}"
            meta = {
                'name': name,
                'description': (f"FD acoustic, {cfg['name']} model at x = {cdp_x / 1000:.1f} km, "
                                + ('free surface (with multiples)' if free_surface
                                   else 'absorbing top (no surface multiples)')),
                'model': {'source': 'fd_modeling', 'config': os.path.relpath(args.config, ROOT),
                          'cdp_x_m': cdp_x, 'free_surface': free_surface},
                'acquisition': acq,
                'modeling': {**fd, 'dt_fd_ms': 1000 * modeler.dt_s, 'nt': modeler.nt},
                'processing': proc,
                'noise_level': 0.0,
                'seed': None,
                'water_bottom_ms': primaries[0]['t0_ms'],
                'primaries': primaries,
                'surface_multiples': multiples,
                'ground_truth_note': ('v_rms_true is the RMS velocity of the vertical '
                                      'column under the CDP; for dipping events the best '
                                      'stacking velocity is slightly higher.'),
            }
            path = os.path.join(ROOT, 'cases', f'{name}.npz')
            save_case(path, processed, offsets, t_out, v_rms, acq['dt_ms'], meta)
            print(f"  saved {os.path.relpath(path, ROOT)}")

            np.savez_compressed(
                os.path.join(out_dir, f'{name}_shots.npz'),
                shots=np.stack([keep[k] for k in sorted(keep)]),
                shot_x=np.array([src[k] for k in sorted(keep)], dtype=np.float32),
                rec_x=rec_x.astype(np.float32), line_idx=line_idx, t_ms=t_out,
                raw_cdp_gather=gather)
            print(f"  saved data/synthetic_fd/{name}_shots.npz")

    write_readme(os.path.join(ROOT, 'cases'))
    print("\ncases/README.md updated.")


if __name__ == '__main__':
    main()