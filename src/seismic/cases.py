"""
src/seismic/cases.py
--------------------
Test case bank: each test case is one CDP gather saved in its own file
(cases/<name>.npz), together with the true RMS velocity used to score
the picks and a description of how it was generated.

A case file contains:
    gather     : (n_offsets, n_samples) float32 — the CDP gather
    offsets    : (n_offsets,) float32 — offsets in m
    t_grid     : (n_samples,) float32 — time axis in ms
    v_rms_true : (n_samples,) float32 — true RMS velocity in m/s
    dt_ms      : float — sampling interval in ms
    meta       : JSON string with the generation parameters, including
                 water_bottom_ms (two-way time of the first reflector)

Cases are plain NumPy files, so running the algorithms on them does not
require the Marmousi2 model (which is only needed to regenerate them).
"""

import glob
import json
import os

import numpy as np

from .models import layered_velocity_profile
from .synthetic import depth_to_time

_MARMOUSI_CACHE = {}


def water_bottom_time_ms(vel_profile, spacing_m):
    """
    Two-way time (ms) of the first velocity change in a depth profile,
    i.e. the bottom of the top layer (the water layer in marine models).

    Returns 0.0 if the profile has no velocity change.
    """
    vel_profile = np.asarray(vel_profile, dtype=np.float32)
    change = np.nonzero(np.abs(vel_profile - vel_profile[0]) > 1.0)[0]
    if change.size == 0:
        return 0.0
    t0_ms, _ = depth_to_time(vel_profile, spacing_m)
    return round(float(t0_ms[change[0] - 1]), 2) if change[0] > 0 else 0.0


def build_velocity_profile(model_cfg, root='.'):
    """
    Velocity profile in depth described by a model config.

    model_cfg["source"] == "marmousi2": one column of the Marmousi2 model
        (keys: models_dir, relative to root, and cdp_idx)
    model_cfg["source"] == "layers": flat layers (keys: velocities,
        thicknesses_m, optional total_depth_m and spacing_m)

    Returns
    -------
    (profile, spacing_m)
    """
    source = model_cfg['source']

    if source == 'marmousi2':
        models_dir = os.path.join(root, model_cfg['models_dir'])
        if models_dir not in _MARMOUSI_CACHE:        # load the model once
            from .io import load_marmousi2_velocity
            _MARMOUSI_CACHE[models_dir] = load_marmousi2_velocity(
                models_dir, verbose=False)
        vel_model, info = _MARMOUSI_CACHE[models_dir]
        return vel_model[model_cfg['cdp_idx'], :], info['spacing_m']

    if source == 'layers':
        spacing = model_cfg.get('spacing_m', 1.25)
        profile = layered_velocity_profile(
            model_cfg['velocities'], model_cfg['thicknesses_m'],
            spacing_m=spacing, total_depth_m=model_cfg.get('total_depth_m'))
        return profile, spacing

    raise ValueError(f"Unknown model source: {source}")


def save_case(path, gather, offsets, t_grid, v_rms_true, dt_ms, meta):
    """Save one test case to a compressed .npz file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        gather=np.asarray(gather, dtype=np.float32),
        offsets=np.asarray(offsets, dtype=np.float32),
        t_grid=np.asarray(t_grid, dtype=np.float32),
        v_rms_true=np.asarray(v_rms_true, dtype=np.float32),
        dt_ms=np.float32(dt_ms),
        meta=np.array(json.dumps(meta)),
    )


def load_case(path):
    """
    Load one test case.

    Returns
    -------
    dict with gather, offsets, t_grid, v_rms_true, dt_ms, meta and name
    """
    with np.load(path, allow_pickle=False) as f:
        case = {
            'gather':     f['gather'],
            'offsets':    f['offsets'],
            't_grid':     f['t_grid'],
            'v_rms_true': f['v_rms_true'],
            'dt_ms':      float(f['dt_ms']),
            'meta':       json.loads(str(f['meta'])),
        }
    case['name'] = os.path.splitext(os.path.basename(path))[0]
    return case


def list_cases(folder):
    """Sorted list of the .npz case files in a folder."""
    return sorted(glob.glob(os.path.join(folder, '*.npz')))
