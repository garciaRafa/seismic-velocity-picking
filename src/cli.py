"""
src/cli.py
----------
Logic of the command-line interface (run.py): choose a test case and an
algorithm, run it once, show the report on screen and save it to files.

The report shows the best solution found (every pick: time and velocity)
and its objective value, so the result can be checked. A saved solution
can be re-evaluated later with verify_solution().
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

from src.analysis.metrics import count_correct_picks, velocity_errors
from src.optimizers.hill_climbing import HillClimbing
from src.optimizers.random_search import RandomSearch
from src.seismic.cases import list_cases, load_case
from src.seismic.semblance import picking_objective

ALGORITHMS = {
    'HillClimbing': HillClimbing,
    'RandomSearch': RandomSearch,
}


def load_defaults(path):
    with open(path) as f:
        return json.load(f)


def algorithm_params(defaults, algorithm, case, overrides=None):
    """Formulation + algorithm parameters, with "auto" values resolved."""
    params = dict(defaults['formulation'])
    params.update(defaults['algorithms'][algorithm])
    params.update(overrides or {})
    if params.get('t_min_ms') == 'auto':
        params['t_min_ms'] = float(case['meta']['water_bottom_ms'])
    return params


def _git_commit(root):
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                       cwd=root, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def run_once(case_path, algorithm, seed, defaults, overrides=None, root='.'):
    """
    Run one algorithm once on one test case.

    Returns
    -------
    dict with everything the report needs (case, parameters, best
    solution, objective value, accuracy metrics, cost)
    """
    case   = load_case(case_path)
    params = algorithm_params(defaults, algorithm, case, overrides)
    opt    = ALGORITHMS[algorithm](case['gather'], case['offsets'],
                                   case['dt_ms'], seed=seed, **params)
    opt.run()

    samples, vels = opt.get_result()
    times_ms = samples * case['dt_ms']
    err = velocity_errors(times_ms, vels, case['t_grid'], case['v_rms_true'])
    ok  = count_correct_picks(times_ms, vels, case['t_grid'], case['v_rms_true'],
                              tol=defaults.get('correct_tol', 0.02))

    picks = [{
        'pick': i + 1,
        'time_sample': int(s),
        'time_ms': float(t),
        'velocity_m_s': float(v),
        'true_velocity_m_s': float(vt),
        'error_m_s': float(e),
        'correct': bool(c),
    } for i, (s, t, v, vt, e, c) in enumerate(zip(
        samples, times_ms, vels, err['v_true'], err['errors'], ok['correct']))]

    return {
        'timestamp':   datetime.now().isoformat(timespec='seconds'),
        'git_commit':  _git_commit(root),
        'python':      sys.version.split()[0],
        'numpy':       np.__version__,
        'case':        case['name'],
        'case_file':   os.path.relpath(case_path, root),
        'case_description': case['meta']['description'],
        'algorithm':   algorithm,
        'seed':        int(seed),
        'params':      params,
        'objective_value': float(opt.best_score),
        'n_evals':     int(opt.n_evals),
        'exec_time_s': float(opt.exec_time_s),
        'n_correct':   ok['n_correct'],
        'n_picks':     len(picks),
        'correct_tol': defaults.get('correct_tol', 0.02),
        'rmse':        err['rmse'],
        'mae':         err['mae'],
        'mape':        err['mape'],
        'best_solution': picks,
    }


def format_report(r):
    """Human-readable report of one run."""
    p = r['params']
    lines = [
        "=" * 72,
        "SEISMIC VELOCITY PICKING — EXPERIMENT REPORT",
        "=" * 72,
        f"Date            : {r['timestamp']}",
        f"Code (git)      : {r['git_commit'] or 'unknown'}   "
        f"Python {r['python']}, NumPy {r['numpy']}",
        f"Test case       : {r['case']}  ({r['case_file']})",
        f"                  {r['case_description']}",
        f"Algorithm       : {r['algorithm']}",
        f"Seed            : {r['seed']}",
        "Parameters      : " + ", ".join(f"{k}={v}" for k, v in p.items()),
        "",
        "RESULT",
        "-" * 72,
        f"Objective value (mean semblance) : {r['objective_value']:.6f}",
        f"Objective evaluations            : {r['n_evals']}",
        f"Execution time                   : {r['exec_time_s']:.2f} s",
        f"Correct picks (error <= {100 * r['correct_tol']:.0f}%)     : "
        f"{r['n_correct']} of {r['n_picks']}",
        f"RMSE / MAE / MAPE                : {r['rmse']:.1f} m/s / "
        f"{r['mae']:.1f} m/s / {r['mape']:.2f} %",
        "",
        "BEST SOLUTION (picks)",
        "-" * 72,
        f"{'Pick':>4s} {'Time (ms)':>10s} {'Velocity':>10s} {'True V_rms':>11s} "
        f"{'Error':>9s}  {'Correct':>7s}",
        f"{'':>4s} {'':>10s} {'(m/s)':>10s} {'(m/s)':>11s} {'(m/s)':>9s}",
    ]
    for pk in r['best_solution']:
        lines.append(
            f"{pk['pick']:4d} {pk['time_ms']:10.1f} {pk['velocity_m_s']:10.1f} "
            f"{pk['true_velocity_m_s']:11.1f} {pk['error_m_s']:+9.1f}  "
            f"{'yes' if pk['correct'] else 'no':>7s}")
    lines.append("=" * 72)
    return "\n".join(lines)


def save_report(r, folder):
    """Save the report as text (.txt) and the full result as JSON (.json)."""
    os.makedirs(folder, exist_ok=True)
    stamp = r['timestamp'].replace(':', '').replace('-', '').replace('T', '_')
    base  = os.path.join(folder, f"{r['case']}_{r['algorithm']}_seed{r['seed']}_{stamp}")
    with open(base + '.txt', 'w') as f:
        f.write(format_report(r) + "\n")
    with open(base + '.json', 'w') as f:
        json.dump(r, f, indent=2)
    return base + '.txt', base + '.json'


def verify_solution(json_path, root='.'):
    """
    Re-evaluate a saved solution on its test case.

    Returns
    -------
    (stored_value, recomputed_value)
    """
    with open(json_path) as f:
        r = json.load(f)
    case  = load_case(os.path.join(root, r['case_file']))
    picks = [(pk['time_sample'], pk['velocity_m_s']) for pk in r['best_solution']]
    value = picking_objective(case['gather'], case['offsets'], picks, case['dt_ms'])
    return r['objective_value'], float(value)


def available_cases(folder):
    """(path, name, description) of every case in the folder."""
    out = []
    for path in list_cases(folder):
        c = load_case(path)
        out.append((path, c['name'], c['meta']['description']))
    return out
