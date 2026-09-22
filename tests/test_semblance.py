"""Tests for src/seismic/semblance.py."""

import numpy as np

from src.seismic.semblance import (
    compute_semblance, semblance_at_pick, picking_objective
)


def _random_trace(n_samples=400, seed=0):
    return np.random.default_rng(seed).standard_normal(n_samples).astype(np.float32)


def test_identical_traces_give_semblance_one():
    # With zero offsets there is no moveout, so every trace reads the
    # same amplitudes and semblance must be exactly 1.
    trace   = _random_trace()
    traces  = np.tile(trace, (10, 1))
    offsets = np.zeros(10, dtype=np.float32)
    s = semblance_at_pick(traces, offsets, 2000.0, 200, dt_ms=2.0)
    assert np.isclose(s, 1.0, atol=1e-5)


def test_cancelling_traces_give_semblance_zero():
    # Pairs of traces with opposite sign cancel in the stack.
    trace   = _random_trace()
    traces  = np.vstack([trace, -trace] * 5)
    offsets = np.zeros(10, dtype=np.float32)
    s = semblance_at_pick(traces, offsets, 2000.0, 200, dt_ms=2.0)
    assert s < 1e-6


def test_semblance_values_are_in_unit_interval():
    rng     = np.random.default_rng(1)
    traces  = rng.standard_normal((20, 300)).astype(np.float32)
    offsets = np.linspace(100.0, 3000.0, 20, dtype=np.float32)
    panel   = compute_semblance(traces, offsets,
                                np.array([1500.0, 2000.0, 2500.0]), dt_ms=2.0)
    assert panel.min() >= 0.0
    assert panel.max() <= 1.0 + 1e-6


def test_panel_and_pick_are_consistent():
    # The objective must read exactly the same value as the panel cell.
    rng     = np.random.default_rng(2)
    traces  = rng.standard_normal((20, 300)).astype(np.float32)
    offsets = np.linspace(100.0, 3000.0, 20, dtype=np.float32)
    vels    = np.array([1800.0, 2200.0], dtype=np.float32)
    panel   = compute_semblance(traces, offsets, vels, dt_ms=2.0)
    for iv, v in enumerate(vels):
        for t in (50, 150, 250):
            s = semblance_at_pick(traces, offsets, float(v), t, dt_ms=2.0)
            assert np.isclose(panel[iv, t], s, atol=1e-5)


def test_objective_is_mean_of_pick_semblances():
    rng     = np.random.default_rng(3)
    traces  = rng.standard_normal((20, 300)).astype(np.float32)
    offsets = np.linspace(100.0, 3000.0, 20, dtype=np.float32)
    picks   = [(60, 1600.0), (150, 2000.0), (240, 2400.0)]
    expected = np.mean([
        semblance_at_pick(traces, offsets, v, t, dt_ms=2.0) for t, v in picks
    ])
    assert np.isclose(picking_objective(traces, offsets, picks, 2.0), expected)
