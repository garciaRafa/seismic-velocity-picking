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

Formulation v3 (objective = "threshold"):
    tau = "auto" computes the blind threshold from the case's own semblance
    panel (robust_threshold over vel_min..vel_max, step tau_vel_step,
    factor tau_factor), with the same robust-semblance options as the
    objective (stretch_mute, energy_floor, min_fold, if given). Each run
    is then scored on detection: against the semblance peaks of the
    expected events for the finite-difference cases (meta['primaries'],
    src.analysis.reference), or against the natural events along the true
    V_rms for the other cases (src.analysis.events). Fields added to the
    run record: n_picks, precision, recall, f1 (and n_on_multiple for the
    finite-difference cases).

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

from src.analysis.events import select_events               # noqa: E402
from src.analysis.metrics import count_correct_picks        # noqa: E402
from src.analysis.metrics import precision_recall           # noqa: E402
from src.analysis.metrics import velocity_errors            # noqa: E402
from src.analysis.reference import get_reference, match_reference  # noqa: E402
from src.analysis.stats import summarize                    # noqa: E402
from src.optimizers.hill_climbing import HillClimbing       # noqa: E402
from src.optimizers.random_search import RandomSearch       # noqa: E402
from src.optimizers.tabu_search import TabuSearch           # noqa: E402
from src.seismic.cases import (build_velocity_profile,      # noqa: E402
                               load_case, water_bottom_time_ms)
from src.seismic.semblance import (compute_semblance,       # noqa: E402
                                   robust_threshold, semblance_along_curve)
from src.seismic.synthetic import generate_cdp_gather       # noqa: E402

# Optimizers available to configs (add new algorithms here)
OPTIMIZERS = {
    'HillClimbing': HillClimbing,
    'RandomSearch': RandomSearch,
    'TabuSearch':   TabuSearch,
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
        return case_to_data(c)

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
    return {'name': cfg['name'], 'gather': gather, 'offsets': offsets, 't_grid': t_grid,
            'v_rms': v_rms, 'dt_ms': acq['dt_ms'],
            'water_bottom_ms': water_bottom_time_ms(profile, spacing),
            'meta': None, 'case': None}


def case_to_data(c):
    """A loaded test case (src.seismic.cases.load_case) in the data format of run_single."""
    return {'name': c['name'], 'gather': c['gather'], 'offsets': c['offsets'],
            't_grid': c['t_grid'], 'v_rms': c['v_rms_true'], 'dt_ms': c['dt_ms'],
            'water_bottom_ms': c['meta']['water_bottom_ms'],
            'meta': c['meta'], 'case': c}


SEMBLANCE_KEYS = ('stretch_mute', 'energy_floor', 'min_fold')
_TAU_CACHE = {}


def semblance_options(params):
    """Robust-semblance options present in the optimizer params."""
    return {k: params[k] for k in SEMBLANCE_KEYS if k in params}


def resolve_params(params, data):
    """
    Replace "auto" values that depend on the data:
    t_min_ms = "auto" -> two-way time of the water bottom of the case;
    tau      = "auto" -> blind threshold from the case's semblance panel
               (tau_factor, default 10; tau_vel_step, default 20 m/s; same
               robust-semblance options as the objective).
    tau_factor and tau_vel_step are consumed here (not optimizer params).
    """
    params = dict(params)
    if params.get('t_min_ms') == 'auto':
        params['t_min_ms'] = float(data['water_bottom_ms'])
    factor = float(params.pop('tau_factor', 10.0))
    step   = float(params.pop('tau_vel_step', 20.0))
    if params.get('tau') == 'auto':
        opts = semblance_options(params)
        key = (data['name'], params['vel_min'], params['vel_max'], step, factor,
               tuple(sorted(opts.items())))
        if key not in _TAU_CACHE:
            vels = np.arange(params['vel_min'], params['vel_max'] + step / 2, step,
                             dtype=np.float32)
            panel = compute_semblance(data['gather'], data['offsets'], vels,
                                      data['dt_ms'], **opts)
            _TAU_CACHE[key] = robust_threshold(panel, factor=factor)
        params['tau'] = float(_TAU_CACHE[key])
    return params


_EVENT_CACHE = {}


def detection_scores(data, params, times_ms, vels, tol=0.02, time_tol_ms=40.0):
    """
    Detection scores of a variable-K solution (formulation v3).

    Finite-difference cases (meta['primaries']): against the semblance peaks
    of the expected events, computed with the same semblance as the
    objective (src.analysis.reference). Other cases: against the natural
    events along the true V_rms (src.analysis.events), as in
    notebooks/v3_threshold_free_k.
    """
    opts = semblance_options(params)
    meta = data.get('meta') or {}
    if 'primaries' in meta:
        ref = get_reference(data['case'], **opts)
        m = match_reference(times_ms, vels, ref, time_tol_ms=time_tol_ms, vel_tol=tol)
        return {'n_picks': m['n_picks'], 'n_events': m['n_primaries'],
                'precision': m['precision'], 'recall': m['recall'], 'f1': m['f1'],
                'n_on_multiple': m['n_on_multiple'], 'pick_kind': m['kind'],
                'reference': 'semblance_peaks'}

    min_sep = max(1, int(np.ceil(params.get('min_sep_ms', 0.0) / data['dt_ms'] - 1e-9)))
    key = (data['name'], params['tau'], min_sep, tuple(sorted(opts.items())))
    if key not in _EVENT_CACHE:
        s_true = semblance_along_curve(data['gather'], data['offsets'], data['v_rms'],
                                       data['dt_ms'], **opts)
        _EVENT_CACHE[key] = select_events(s_true, params['tau'], min_sep) * data['dt_ms']
    pr = precision_recall(times_ms, vels, data['t_grid'], data['v_rms'],
                          _EVENT_CACHE[key], tol=tol, time_tol_ms=time_tol_ms)
    return {'n_picks': pr['n_picks'], 'n_events': pr['n_events'],
            'precision': pr['precision'], 'recall': pr['recall'], 'f1': pr['f1'],
            'reference': 'natural_events'}


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
# One run                                                              #
# ------------------------------------------------------------------ #

SUMMARY_KEYS = ('best_score', 'n_correct', 'rmse', 'mae', 'mape',
                'exec_time_s', 'n_evals')
DETECTION_KEYS = ('n_picks', 'precision', 'recall', 'f1', 'n_on_multiple')


def run_single(opt_cls, params, data, seed, run_index, tol=0.02):
    """
    Run one optimizer once and return the record saved in runs.json.

    The result depends only on (data, params, seed), so runs can be
    executed in any order or in parallel with identical results.
    """
    opt = opt_cls(data['gather'], data['offsets'], data['dt_ms'],
                  seed=int(seed), **params)
    opt.run()

    times_s, vels = opt.get_result()
    times_ms = times_s * data['dt_ms']
    err = velocity_errors(times_ms, vels, data['t_grid'], data['v_rms'])
    ok  = count_correct_picks(times_ms, vels, data['t_grid'], data['v_rms'], tol=tol)

    record = {
        'run':          int(run_index),
        'seed':         int(seed),
        'best_score':   float(opt.best_score),
        'rmse':         err['rmse'],
        'mae':          err['mae'],
        'mape':         err['mape'],
        'n_correct':    ok['n_correct'],
        'frac_correct': ok['frac_correct'],
        'exec_time_s':  float(opt.exec_time_s),
        'n_evals':      int(opt.n_evals),
        'picks_time_ms': [float(t) for t in times_ms],
        'picks_vel_ms':  [float(v) for v in vels],
        'v_true':        err['v_true'],
        'errors':        err['errors'],
        'correct':       ok['correct'],
        'eval_history':  [[int(e), float(s)] for e, s in opt.eval_history],
    }
    # Formulation v3: detection scores (variable number of picks)
    if params.get('objective') == 'threshold':
        record.update(detection_scores(data, params, times_ms, vels, tol=tol))

    # Algorithm-specific extras (present only in some optimizers)
    if hasattr(opt, 'restart_scores'):
        record['restart_scores'] = [float(s) for s in opt.restart_scores]
        record['restart_iters']  = [len(h) - 1 for h in opt.restart_histories]
    return record


def summarize_runs(runs):
    """Statistics over the runs of one experiment (summary.json)."""
    keys = SUMMARY_KEYS + tuple(k for k in DETECTION_KEYS if k in runs[0])
    return {key: summarize([r[key] for r in runs]) for key in keys}


def run_line(record):
    """One-line progress text of a run (detection scores for v3)."""
    if 'f1' in record:
        return (f"K={record['n_picks']:2d}  recall={record['recall']:4.0%}  "
                f"f1={record['f1']:.2f}")
    return f"correct={record['n_correct']:2d}  rmse={record['rmse']:7.1f}"


def run_seeds(master_seed, n_runs):
    """Independent, reproducible run seeds derived from the master seed."""
    return [int(s) for s in np.random.SeedSequence(master_seed).generate_state(n_runs)]


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
    gather = data['gather']
    params = resolve_params(opt_cfg['params'], data)
    source = cfg['model'].get('path', f"data_seed={exp.get('data_seed')}")
    print(f"Gather     : {gather.shape}, {source}")

    seeds = run_seeds(exp['master_seed'], exp['n_runs'])

    runs = []
    t_all = time.perf_counter()
    for i, seed in enumerate(seeds):
        record = run_single(opt_cls, params, data, seed, i,
                            tol=exp.get('correct_tol', 0.02))
        runs.append(record)

        print(f"  run {i + 1:3d}/{exp['n_runs']}  seed={int(seed):10d}  "
              f"score={record['best_score']:.4f}  {run_line(record)}  "
              f"evals={record['n_evals']:6d}  time={record['exec_time_s']:6.1f}s")

        # Save after every run, so an interrupted experiment keeps its data
        with open(os.path.join(out_dir, 'runs.json'), 'w') as f:
            json.dump(runs, f, indent=2)

    summary = summarize_runs(runs)
    summary['total_time_s'] = time.perf_counter() - t_all
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('\nSummary (mean ± std, [min, max])')
    keys = ('best_score', 'n_picks', 'precision', 'recall', 'f1') if 'f1' in summary \
        else ('best_score', 'n_correct', 'rmse')
    for key in keys + ('exec_time_s', 'n_evals'):
        s = summary[key]
        print(f"  {key:12s} {s['mean']:10.4f} ± {s['std']:<10.4f} "
              f"[{s['min']:.4f}, {s['max']:.4f}]")


if __name__ == '__main__':
    main()