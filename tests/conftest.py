"""
tests/conftest.py
-----------------
Shared test setup: puts the project root on sys.path and provides a
small layered synthetic gather used by several tests.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.seismic.models import layered_velocity_profile   # noqa: E402
from src.seismic.synthetic import generate_cdp_gather     # noqa: E402

# Three-layer model: water, then two sediment layers
LAYER_VELOCITIES   = [1500.0, 2000.0, 2600.0]
LAYER_THICKNESSES  = [600.0, 700.0]          # meters
SPACING_M          = 1.25
DT_MS              = 2.0
T_MAX_MS           = 3000.0


@pytest.fixture(scope='session')
def layered_gather():
    """
    CDP gather generated from the three-layer model.

    Returns a dict with gather, offsets, dt_ms, t_grid, v_rms and the
    expected zero-offset times (ms) of the two reflections.
    """
    profile = layered_velocity_profile(
        LAYER_VELOCITIES, LAYER_THICKNESSES,
        spacing_m=SPACING_M, total_depth_m=3500.0
    )
    offsets = np.linspace(100.0, 3000.0, 50, dtype=np.float32)

    gather, t_grid, v_rms = generate_cdp_gather(
        profile, SPACING_M, offsets,
        dt_ms=DT_MS, t_max_ms=T_MAX_MS,
        f0_hz=30.0, noise_level=0.02, seed=0
    )

    # Expected two-way times of the two interfaces (ms)
    t1 = 2.0 * 600.0 / 1500.0 * 1000.0                     # 800 ms
    t2 = t1 + 2.0 * 700.0 / 2000.0 * 1000.0                # 1500 ms

    return {
        'gather':  gather,
        'offsets': offsets,
        'dt_ms':   DT_MS,
        't_grid':  t_grid,
        'v_rms':   v_rms,
        'event_times_ms': [t1, t2],
    }
