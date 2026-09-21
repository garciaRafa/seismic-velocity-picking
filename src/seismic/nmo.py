"""
src/seismic/nmo.py
------------------
Normal Moveout (NMO) correction.
Based on the hyperbolic travel time equation (Yilmaz, 2001):

    t^2 = t0^2 + (1000 * x / v)^2

where:
    t  = NMO travel time (ms)
    t0 = zero-offset two-way travel time (ms)
    x  = source-receiver offset (m)
    v  = stacking velocity (m/s)

The factor 1000 converts the offset term x / v from seconds to ms, so
that both terms of the sum are in the same unit.
"""

import numpy as np


def nmo_times(t0_array, offsets, velocity):
    """
    Compute NMO travel times for all offset-time pairs.

    Parameters
    ----------
    t0_array : np.ndarray, shape (n_samples,) — zero-offset times in ms
    offsets  : np.ndarray, shape (n_traces,)  — source-receiver offsets in m
    velocity : float — stacking velocity in m/s

    Returns
    -------
    t_nmo : np.ndarray, shape (n_traces, n_samples) — NMO times in ms
    """
    t0 = t0_array[np.newaxis, :]        # (1, n_samples)
    x  = offsets[:, np.newaxis]         # (n_traces, 1)

    # Offset term is (x / v) in seconds, so convert it to ms before summing
    t_nmo = np.sqrt(t0**2 + (x / velocity * 1000.0) ** 2)

    return t_nmo.astype(np.float32)


def apply_nmo(traces, offsets, velocity, dt_ms, t0_ms=0.0):
    """
    Apply NMO correction to a CDP gather using linear interpolation.

    Parameters
    ----------
    traces   : np.ndarray, shape (n_traces, n_samples)
    offsets  : np.ndarray, shape (n_traces,) in meters
    velocity : float — stacking velocity in m/s
    dt_ms    : float — sampling interval in ms
    t0_ms    : float — start time in ms (default 0)

    Returns
    -------
    corrected : np.ndarray, shape (n_traces, n_samples)
    """
    n_traces, n_samples = traces.shape
    t0_array = np.arange(n_samples, dtype=np.float32) * dt_ms + t0_ms

    t_nmo = nmo_times(t0_array, offsets, velocity)

    corrected = np.zeros_like(traces)

    for i in range(n_traces):
        # Convert NMO times to sample indices
        idx = (t_nmo[i] - t0_ms) / dt_ms

        # Linear interpolation for each sample
        idx_floor = np.floor(idx).astype(int)
        idx_ceil  = idx_floor + 1
        frac      = idx - idx_floor

        valid = (idx_floor >= 0) & (idx_ceil < n_samples)

        corrected[i, valid] = (
            (1 - frac[valid]) * traces[i, idx_floor[valid]] +
            frac[valid]       * traces[i, idx_ceil[valid]]
        )

    return corrected


def stack_gather(corrected_traces):
    """
    Stack NMO-corrected traces into a single trace.

    Parameters
    ----------
    corrected_traces : np.ndarray, shape (n_traces, n_samples)

    Returns
    -------
    stacked : np.ndarray, shape (n_samples,)
    """
    return np.mean(corrected_traces, axis=0)
