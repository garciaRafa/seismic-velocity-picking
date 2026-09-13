"""
src/seismic/semblance.py
------------------------
Semblance coherence measure for velocity analysis.
Based on equation 2.6 from Araújo (2018) and Yilmaz (2001):

    S = (sum_t (sum_i f_i(t))^2) / (M * sum_t sum_i f_i(t)^2)

where:
    f_i(t) = amplitude of i-th trace at NMO time t
    M      = number of traces in the CDP gather
"""

import numpy as np
from .nmo import nmo_times


def compute_semblance(traces, offsets, velocities, dt_ms, t0_ms=0.0):
    """
    Compute semblance panel for a CDP gather.

    Parameters
    ----------
    traces     : np.ndarray, shape (n_traces, n_samples)
    offsets    : np.ndarray, shape (n_traces,) in meters
    velocities : np.ndarray, shape (n_vel,) in m/s
    dt_ms      : float — sampling interval in ms
    t0_ms      : float — start time in ms

    Returns
    -------
    semblance : np.ndarray, shape (n_vel, n_samples)
        Values in [0, 1] — higher means better coherence
    """
    n_traces, n_samples = traces.shape
    n_vel = len(velocities)

    t0_array = np.arange(n_samples, dtype=np.float32) * dt_ms + t0_ms
    semblance = np.zeros((n_vel, n_samples), dtype=np.float32)

    for iv, vel in enumerate(velocities):
        # NMO times for all traces: shape (n_traces, n_samples)
        t_nmo = nmo_times(t0_array, offsets, vel)

        # Convert to sample indices
        idx = (t_nmo - t0_ms) / dt_ms
        idx_floor = np.floor(idx).astype(int)
        idx_ceil  = idx_floor + 1
        frac      = idx - idx_floor

        # Interpolated amplitudes: shape (n_traces, n_samples)
        valid = (idx_floor >= 0) & (idx_ceil < n_samples)
        amps  = np.zeros((n_traces, n_samples), dtype=np.float32)

        for i in range(n_traces):
            v = valid[i]
            amps[i, v] = (
                (1 - frac[i, v]) * traces[i, idx_floor[i, v]] +
                frac[i, v]       * traces[i, idx_ceil[i, v]]
            )

        # Semblance numerator and denominator
        numerator   = np.sum(amps, axis=0) ** 2
        denominator = n_traces * np.sum(amps ** 2, axis=0)

        # Avoid division by zero
        mask = denominator > 1e-10
        semblance[iv, mask] = numerator[mask] / denominator[mask]

    return semblance


def semblance_at_pick(traces, offsets, velocity, time_sample, dt_ms,
                      t0_ms=0.0, window=5):
    """
    Compute semblance at a single (velocity, time) pick.
    Used as objective function by metaheuristic algorithms.

    Parameters
    ----------
    traces      : np.ndarray, shape (n_traces, n_samples)
    offsets     : np.ndarray, shape (n_traces,)
    velocity    : float — stacking velocity in m/s
    time_sample : int   — sample index of the pick
    dt_ms       : float — sampling interval in ms
    t0_ms       : float — start time in ms
    window      : int   — number of samples around the pick to average

    Returns
    -------
    score : float in [0, 1]
    """
    n_traces, n_samples = traces.shape
    t0_array = np.arange(n_samples, dtype=np.float32) * dt_ms + t0_ms

    t_nmo     = nmo_times(t0_array, offsets, velocity)
    idx       = (t_nmo - t0_ms) / dt_ms
    idx_floor = np.floor(idx).astype(int)
    idx_ceil  = idx_floor + 1
    frac      = idx - idx_floor

    # Window around the pick
    t_start = max(0, time_sample - window // 2)
    t_end   = min(n_samples, time_sample + window // 2 + 1)

    numerator   = 0.0
    denominator = 0.0

    for t in range(t_start, t_end):
        amps = np.zeros(n_traces, dtype=np.float32)
        for i in range(n_traces):
            f = idx_floor[i, t]
            c = idx_ceil[i, t]
            if 0 <= f and c < n_samples:
                amps[i] = (1 - frac[i, t]) * traces[i, f] + \
                          frac[i, t]        * traces[i, c]

        numerator   += np.sum(amps) ** 2
        denominator += n_traces * np.sum(amps ** 2)

    if denominator < 1e-10:
        return 0.0

    return float(numerator / denominator)


def picking_objective(traces, offsets, picks, dt_ms, t0_ms=0.0, window=5):
    """
    Objective function for a full set of velocity picks.
    Returns the mean semblance across all picks.
    Used directly by metaheuristic algorithms.

    Parameters
    ----------
    traces  : np.ndarray, shape (n_traces, n_samples)
    offsets : np.ndarray, shape (n_traces,)
    picks   : list of (time_sample, velocity) tuples
    dt_ms   : float
    t0_ms   : float
    window  : int

    Returns
    -------
    score : float in [0, 1] — higher is better
    """
    scores = []
    for time_sample, velocity in picks:
        s = semblance_at_pick(
            traces, offsets, velocity, int(time_sample),
            dt_ms, t0_ms, window
        )
        scores.append(s)

    return float(np.mean(scores))
