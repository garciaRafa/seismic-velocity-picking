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


# ---------------------------------------------------------------------- #
# Time-variant NMO                                                        #
# ---------------------------------------------------------------------- #

from src.seismic.nmo import (apply_nmo, apply_nmo_variable, stack_live,  # noqa: E402
                             velocity_function)

_DT = 2.0
_OFF = np.linspace(100.0, 2500.0, 25)


def _event(t0_ms, v, n_t=800, f0=30.0):
    t = _DT * np.arange(n_t)
    th = np.sqrt(t0_ms ** 2 + (1000.0 * _OFF / v) ** 2)
    a = (np.pi * f0 * (t[None, :] - th[:, None]) / 1000.0) ** 2
    return ((1 - 2 * a) * np.exp(-a)).astype(np.float32)


def test_velocity_function_interpolates_and_holds_the_ends():
    v = velocity_function([1000.0, 500.0], [2000.0, 1500.0], np.array([0.0, 500.0, 750.0, 1000.0, 2000.0]))
    assert np.allclose(v, [1500.0, 1500.0, 1750.0, 2000.0, 2000.0])


def test_constant_velocity_matches_apply_nmo():
    g = _event(600.0, 2000.0)
    a = apply_nmo(g, _OFF, 2000.0, _DT)
    b, live = apply_nmo_variable(g, _OFF, 2000.0, _DT)
    assert np.allclose(a, b, atol=1e-5) and live.any()


def test_right_velocity_flattens_two_events():
    g = _event(500.0, 1800.0) + _event(1100.0, 2400.0)
    t0 = _DT * np.arange(g.shape[1])
    # stretch mute: the far traces of the shallow event, stretched into the deep
    # part, would otherwise dominate there (the reason for the mute)
    corr, live = apply_nmo_variable(g, _OFF, velocity_function([500.0, 1100.0], [1800.0, 2400.0], t0),
                                    _DT, stretch_mute=0.5)
    near = _OFF <= 2000.0      # beyond, stretched shallow energy still reaches ~1000 ms
    peaks = _DT * np.argmax(np.abs(corr[near, 400:]), axis=1) + 800.0
    assert np.all(np.abs(peaks - 1100.0) <= _DT)
    stack = stack_live(corr, live)
    assert abs(_DT * np.argmax(stack[200:300]) + 400.0 - 500.0) <= _DT


def test_wrong_velocity_leaves_the_event_curved():
    g = _event(1100.0, 2400.0)
    corr, _ = apply_nmo_variable(g, _OFF, 2000.0, _DT)              # too slow: over-corrected
    peaks = _DT * np.argmax(np.abs(corr), axis=1)
    assert peaks[-1] < peaks[0] - 20.0


def test_stretch_mute_zeroes_far_shallow_samples():
    g = np.ones((len(_OFF), 400), dtype=np.float32)
    corr, live = apply_nmo_variable(g, _OFF, 1500.0, _DT, stretch_mute=0.5)
    t0 = _DT * 50                                                   # 100 ms
    stretch = (np.sqrt(t0 ** 2 + (1000 * _OFF / 1500.0) ** 2) - t0) / t0
    assert np.array_equal(live[:, 50], stretch <= 0.5)
    assert np.all(corr[~live] == 0.0)