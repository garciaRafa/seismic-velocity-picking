"""Tests for the test case bank (src/seismic/cases.py) and the interface (src/cli.py)."""

import json
import os

import numpy as np
import pytest

from src.cli import format_report, run_once, save_report, verify_solution
from src.seismic.cases import (build_velocity_profile, list_cases, load_case,
                               save_case, water_bottom_time_ms)
from src.seismic.synthetic import generate_cdp_gather

LAYERS = {"source": "layers", "velocities": [1500.0, 2000.0, 2600.0],
          "thicknesses_m": [600.0, 700.0], "total_depth_m": 3500.0,
          "spacing_m": 1.25}

DEFAULTS = {
    "formulation": {"vel_min": 1400.0, "vel_max": 3000.0, "n_picks": 3,
                    "min_sep_ms": 100.0, "t_min_ms": "auto", "t_max_ms": 2700.0},
    "algorithms": {"HillClimbing": {"max_iter": 20, "restarts": 2, "step_time": 5,
                                    "step_vel": 50.0, "n_neighbors": 5,
                                    "patience": 5, "max_evals": 300},
                   "RandomSearch": {"max_evals": 300}},
    "correct_tol": 0.02,
}


@pytest.fixture(scope='module')
def case_path(tmp_path_factory):
    folder = tmp_path_factory.mktemp('cases')
    profile, spacing = build_velocity_profile(LAYERS)
    offsets = np.linspace(100.0, 3000.0, 20, dtype=np.float32)
    gather, t_grid, v_rms = generate_cdp_gather(profile, spacing, offsets,
                                                noise_level=0.05, seed=1)
    meta = {'name': 'layers3', 'description': 'test case', 'noise_level': 0.05,
            'seed': 1, 'water_bottom_ms': water_bottom_time_ms(profile, spacing)}
    path = str(folder / 'layers3.npz')
    save_case(path, gather, offsets, t_grid, v_rms, 2.0, meta)
    return path


def test_water_bottom_of_layered_model():
    profile, spacing = build_velocity_profile(LAYERS)
    assert water_bottom_time_ms(profile, spacing) == pytest.approx(800.0, abs=0.01)


def test_case_roundtrip(case_path):
    c = load_case(case_path)
    assert c['name'] == 'layers3'
    assert c['gather'].shape == (20, 1501)
    assert c['dt_ms'] == 2.0
    assert c['meta']['water_bottom_ms'] == pytest.approx(800.0, abs=0.01)
    assert list_cases(os.path.dirname(case_path)) == [case_path]


@pytest.mark.parametrize('algorithm', ['HillClimbing', 'RandomSearch'])
def test_run_report_save_and_verify(case_path, tmp_path, algorithm):
    r = run_once(case_path, algorithm, seed=3, defaults=DEFAULTS,
                 root=os.path.dirname(case_path))
    assert r['params']['t_min_ms'] == pytest.approx(800.0, abs=0.01)   # "auto"
    assert len(r['best_solution']) == 3
    assert all(pk['time_ms'] >= 800.0 for pk in r['best_solution'])

    text = format_report(r)
    assert 'BEST SOLUTION' in text and f"{r['objective_value']:.6f}" in text

    txt, js = save_report(r, str(tmp_path))
    assert os.path.exists(txt) and os.path.exists(js)
    with open(js) as f:
        assert json.load(f)['objective_value'] == r['objective_value']

    stored, recomputed = verify_solution(js, root=os.path.dirname(case_path))
    assert stored == pytest.approx(recomputed, abs=1e-9)


def test_same_seed_same_report(case_path):
    a = run_once(case_path, 'RandomSearch', 5, DEFAULTS, root=os.path.dirname(case_path))
    b = run_once(case_path, 'RandomSearch', 5, DEFAULTS, root=os.path.dirname(case_path))
    assert a['best_solution'] == b['best_solution']
