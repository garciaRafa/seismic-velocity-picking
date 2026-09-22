"""
scripts/run_experiment.py
-------------------------
Run N independent executions of an optimizer on one synthetic CDP gather,
fully described by a JSON config file, and save everything needed to
reproduce and analyze the experiment.

Usage (from the project root):
    python scripts/run_experiment.py configs/hill_climbing_cdp6800.json
    python scripts/run_experiment.py configs/hill_climbing_cdp6800.json --n-runs 5

Output folder: results/experiments/<name>_<timestamp>/
    config.json    — exact copy of the config used
    metadata.json  — git commit, library versions, machine, command line
    runs.json      — one record per run (seed, score, errors, picks, ...)
    summary.json   — statistics over the runs

Seeds:
    The data (noise) uses experiment.data_seed, fixed for all runs.
    The run seeds are derived from experiment.master_seed with
    np.random.SeedSequence, so they are independent and reproducible.
    Any single run can be repeated by creating the optimizer with the
    seed stored in runs.json.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.analysis.metrics import count_correct_picks        # noqa: E402
from src.analysis.metrics import velocity_errors            # noqa: E402
from src.analysis.stats import summarize                    # noqa: E402
from src.optimizers.hill_climbing import HillClimbing       # noqa: E402
from src.optimizers.random_search import RandomSearch       # noqa: E402
from src.seismic.cases import (build_velocity_profile,      # noqa: E402
                               load_case, water_bottom_time_ms)
from src.seismic.synthetic import generate_cdp_gather       # noqa: E402

# Optimizers available to configs (add new algorithms here)
OPTIMIZERS = {
    'HillClimbing': HillClimbing,
    'RandomSearch': RandomSearch,
}


# ------------------------------------------------------------------ #
# Data                                                                 #
# ------------------------------------------------------------------ #

def build_gather(cfg):
    """
    Data of one experiment.

    model.source == "case": load a saved test case (model.path, relative
    to the project root). Otherwise the gather is generated from the model
    config ("marmousi2" or "layers") with the acquisition parameters.

    Returns
    -------
    dict with gather, offsets, t_grid, v_rms, dt_ms, water_bottom_ms
    """
    model = cfg['model']

    if model['source'] == 'case':
        c = load_case(os.path.join(ROOT, model['path']))
        return {'gather': c['gather'], 'offsets': c['offsets'],
                't_grid': c['t_grid'], 'v_rms': c['v_rms_true'],
                'dt_ms': c['dt_ms'],
                'water_bottom_ms': c['meta']['water_bottom_ms']}

    acq = cfg['acquisition']
    profile, spacing = build_velocity_profile(model, ROOT)
    offsets = np.linspace(acq['offset_min_m'], acq['offset_max_m'],
                          acq['n_offsets'], dtype=np.float32)
    gather, t_grid, v_rms = generate_cdp_gather(
        profile, spacing, offsets,
        dt_ms=acq['dt_ms'], t_max_ms=acq['t_max_ms'],
        f0_hz=acq['f0_hz'], noise_level=acq['noise_level'],
        seed=cfg['experiment']['data_seed'],
    )
    return {'gather': gather, 'offsets': offsets, 't_grid': t_grid,
            'v_rms': v_rms, 'dt_ms': acq['dt_ms'],
            'water_bottom_ms': water_bottom_time_ms(profile, spacing)}


def resolve_params(params, data):
    """
    Replace "auto" values that depend on the data:
    t_min_ms = "auto" -> two-way time of the water bottom of the case.
    """
    params = dict(params)
    if params.get('t_min_ms') == 'auto':
        params['t_min_ms'] = float(data['water_bottom_ms'])
    return params


# ------------------------------------------------------------------ #
# Metadata                                                             #
# ------------------------------------------------------------------ #

def _git(*args):
    try:
        return subprocess.check_output(['git', *args], cwd=ROOT,
                                       stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def _cpu_name():
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def collect_metadata():
    status = _git('status', '--porcelain')
    return {
        'timestamp':    datetime.now().isoformat(timespec='seconds'),
        'git_commit':   _git('rev-parse', 'HEAD'),
        'git_dirty':    bool(status) if status is not None else None,
        'python':       sys.version.split()[0],
        'numpy':        np.__version__,
        'platform':     platform.platform(),
        'cpu':          _cpu_name(),
        'command':      ' '.join(sys.argv),
    }


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', help='path to the JSON config')
    parser.add_argument('--n-runs', type=int, default=None,
                        help='override experiment.n_runs')
    parser.add_argument('--out', default=os.path.join(ROOT, 'results', 'experiments'),
                        help='base output folder')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    if args.n_runs is not None:
        cfg['experiment']['n_runs'] = args.n_runs

    exp      = cfg['experiment']
    opt_cfg  = cfg['optimizer']
    opt_cls  = OPTIMIZERS[opt_cfg['name']]

    metadata = collect_metadata()
    if metadata['git_dirty']:
        print('WARNING: uncommitted changes — results may not match git_commit.')

    stamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.out, f"{cfg['name']}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Experiment : {cfg['name']}")
    print(f"Output     : {out_dir}")
    data = build_gather(cfg)
    gather, offsets = data['gather'], data['offsets']
    t_grid, v_rms, dt_ms = data['t_grid'], data['v_rms'], data['dt_ms']
    params = resolve_params(opt_cfg['params'], data)
    source = cfg['model'].get('path', f"data_seed={exp.get('data_seed')}")
    print(f"Gather     : {gather.shape}, {source}")

    run_seeds = np.random.SeedSequence(exp['master_seed']).generate_state(exp['n_runs'])

    runs = []
    t_all = time.perf_counter()
    for i, seed in enumerate(run_seeds):
        opt = opt_cls(gather, offsets, dt_ms, seed=int(seed), **params)
        opt.run()

        times_s, vels = opt.get_result()
        times_ms = times_s * dt_ms
        err = velocity_errors(times_ms, vels, t_grid, v_rms)
        ok  = count_correct_picks(times_ms, vels, t_grid, v_rms,
                                  tol=exp.get('correct_tol', 0.02))

        record = {
            'run':         i,
            'seed':        int(seed),
            'best_score':  float(opt.best_score),
            'rmse':        err['rmse'],
            'mae':         err['mae'],
            'mape':        err['mape'],
            'n_correct':   ok['n_correct'],
            'frac_correct': ok['frac_correct'],
            'exec_time_s': float(opt.exec_time_s),
            'n_evals':     int(opt.n_evals),
            'picks_time_ms':  [float(t) for t in times_ms],
            'picks_vel_ms':   [float(v) for v in vels],
            'v_true':         err['v_true'],
            'errors':         err['errors'],
            'correct':        ok['correct'],
            'eval_history':   [[int(e), float(s)] for e, s in opt.eval_history],
        }
        # Algorithm-specific extras (present only in some optimizers)
        if hasattr(opt, 'restart_scores'):
            record['restart_scores'] = [float(s) for s in opt.restart_scores]
            record['restart_iters']  = [len(h) - 1 for h in opt.restart_histories]
        runs.append(record)

        print(f"  run {i + 1:3d}/{exp['n_runs']}  seed={int(seed):10d}  "
              f"score={record['best_score']:.4f}  correct={record['n_correct']:2d}  "
              f"rmse={record['rmse']:7.2f}  "
              f"evals={record['n_evals']:6d}  time={record['exec_time_s']:6.1f}s")

        # Save after every run, so an interrupted experiment keeps its data
        with open(os.path.join(out_dir, 'runs.json'), 'w') as f:
            json.dump(runs, f, indent=2)

    summary = {
        key: summarize([r[key] for r in runs])
        for key in ('best_score', 'n_correct', 'rmse', 'mae', 'mape',
                    'exec_time_s', 'n_evals')
    }
    summary['total_time_s'] = time.perf_counter() - t_all
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('\nSummary (mean ± std, [min, max])')
    for key in ('best_score', 'n_correct', 'rmse', 'exec_time_s', 'n_evals'):
        s = summary[key]
        print(f"  {key:12s} {s['mean']:10.4f} ± {s['std']:<10.4f} "
              f"[{s['min']:.4f}, {s['max']:.4f}]")


if __name__ == '__main__':
    main()
