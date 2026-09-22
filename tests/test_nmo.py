"""Tests for src/seismic/nmo.py."""

import numpy as np

from src.seismic.nmo import nmo_times


def test_nmo_analytic_value():
    # t = sqrt(1000^2 + (1000 * 3000 / 2000)^2) = sqrt(1000^2 + 1500^2)
    t = nmo_times(np.array([1000.0]), np.array([3000.0]), 2000.0)
    assert np.isclose(t[0, 0], 1802.7756, rtol=1e-5)


def test_nmo_zero_offset_returns_t0():
    t0 = np.array([0.0, 500.0, 1234.0], dtype=np.float32)
    t = nmo_times(t0, np.array([0.0]), 2500.0)
    assert np.allclose(t[0], t0)


def test_nmo_output_shape():
    t0 = np.arange(100, dtype=np.float32)
    offsets = np.linspace(100.0, 3000.0, 7, dtype=np.float32)
    t = nmo_times(t0, offsets, 2000.0)
    assert t.shape == (7, 100)


def test_nmo_moveout_grows_with_offset_and_shrinks_with_velocity():
    offsets = np.array([500.0, 1500.0, 3000.0], dtype=np.float32)
    slow = nmo_times(np.array([1000.0]), offsets, 1500.0)[:, 0]
    fast = nmo_times(np.array([1000.0]), offsets, 3000.0)[:, 0]
    assert np.all(np.diff(slow) > 0)          # later arrival at far offsets
    assert np.all(slow > fast)                # slower medium, more moveout
