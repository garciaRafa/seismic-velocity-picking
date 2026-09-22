"""Tests for src/seismic/synthetic.py."""

import numpy as np

from src.seismic.models import layered_velocity_profile
from src.seismic.synthetic import (
    _accumulate_on_time_grid, compute_reflectivity, depth_to_time
)


def test_accumulation_preserves_total_and_position():
    times  = np.array([10.0, 11.0, 13.4])
    values = np.array([1.0, -0.5, 2.0])
    out = _accumulate_on_time_grid(times, values, dt_ms=2.0, n_samples=10)
    assert np.isclose(out.sum(), values.sum())
    # A value exactly on a sample goes entirely to that sample
    single = _accumulate_on_time_grid(np.array([8.0]), np.array([1.0]), 2.0, 10)
    assert single[4] == 1.0 and single.sum() == 1.0


def test_no_reflection_is_lost_in_fast_layers():
    # In a 2600 m/s layer, 1.25 m of rock takes < 1 ms (two-way), so the
    # old interpolation could drop the reflection entirely.
    profile = layered_velocity_profile([1500.0, 2000.0, 2600.0],
                                       [600.0, 700.0], total_depth_m=3500.0)
    t0_ms, _ = depth_to_time(profile, 1.25)
    refl     = compute_reflectivity(profile)
    out = _accumulate_on_time_grid(t0_ms, refl, dt_ms=2.0, n_samples=1501)
    assert np.isclose(out.sum(), refl.sum(), rtol=1e-5)
    # Both interfaces are present near their expected times
    for t_ms in (800.0, 1500.0):
        i = int(t_ms / 2.0)
        assert np.abs(out[i - 2:i + 3]).sum() > 0.1
