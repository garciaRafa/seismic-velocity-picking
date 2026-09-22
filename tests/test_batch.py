"""Tests for scripts/run_batch.py (one run task and the summary tables)."""

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import run_batch                                              # noqa: E402
from src.seismic.cases import build_velocity_profile, save_case, water_bottom_time_ms  # noqa: E402
from src.seismic.synthetic import generate_cdp_gather        # noqa: E402

LAYERS = {"source": "layers", "velocities": [1500.0, 2000.0, 2600.0],
          "thicknesses_m": [600.0, 700.0], "total_depth_m": 3500.0}
BATCH = {
    "name": "t", "n_runs": 3, "master_seed": 1, "correct_tol": 0.02,
    "formulation": {"vel_min": 1400.0, "vel_max": 3000.0, "n_picks": 3,
                    "min_sep_ms": 100.0, "t_min_ms": "auto", "t_max_ms": 2700.0},
    "algorithms": {"HillClimbing": {"max_iter": 10, "restarts": 2, "step_time": 5,
                                    "step_vel": 50.0, "n_neighbors": 5,
                                    "patience": 5, "max_evals": 200},
                   "RandomSearch": {"max_evals": 200}},
}


def _case(tmp_path):
    profile, spacing = build_velocity_profile(LAYERS)
    offsets = np.linspace(100.0, 3000.0, 20, dtype=np.float32)
    g, t, v = generate_cdp_gather(profile, spacing, offsets, seed=0)
    path = str(tmp_path / 'layers3.npz')
    save_case(path, g, offsets, t, v, 2.0,
              {'description': 'x', 'noise_level': 0.05, 'seed': 0,
               'water_bottom_ms': water_bottom_time_ms(profile, spacing)})
    return path


def test_task_is_deterministic_and_resolves_auto(tmp_path):
    path = _case(tmp_path)
    cfg = run_batch.experiment_config(BATCH, 'layers3', path, 'RandomSearch')
    task = (path, 'RandomSearch', cfg['optimizer']['params'], 123, 0, 0.02)
    a = run_batch._task(task)[2]
    b = run_batch._task(task)[2]
    assert a['picks_vel_ms'] == b['picks_vel_ms']
    assert min(a['picks_time_ms']) >= 800.0          # t_min_ms = "auto"


def test_batch_summary_tables(tmp_path):
    path = _case(tmp_path)
    seeds = run_batch.run_seeds(BATCH['master_seed'], BATCH['n_runs'])
    results = {'layers3': {}}
    for alg in BATCH['algorithms']:
        cfg = run_batch.experiment_config(BATCH, 'layers3', path, alg)
        results['layers3'][alg] = [
            run_batch._task((path, alg, cfg['optimizer']['params'], s, i, 0.02))[2]
            for i, s in enumerate(seeds)]
    text = run_batch.batch_summary(str(tmp_path), ['layers3'],
                                   list(BATCH['algorithms']), results)
    assert 'Correct picks' in text and 'all cases' in text
    for name in ('batch_summary.md', 'batch_summary.json', 'batch_summary.csv'):
        assert os.path.exists(tmp_path / name)
