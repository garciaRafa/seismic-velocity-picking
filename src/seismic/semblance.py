"""
src/seismic/semblance.py
------------------------
Semblance coherence measure for velocity analysis.
Based on equation 2.6 from Araújo (2018) and Yilmaz (2001):

    S = (sum_t (sum_i f_i(t))^2) / (M * sum_t sum_i f_i(t)^2)

where:
    f_i(t) = amplitude of i-th trace at NMO time t
    M      = number of traces in the CDP gather
    sum_t  = sum over a time gate (window) centered on the analysis time

The same time gate is used by the semblance panel and by the objective
function, so the panel seen in the notebooks is exactly the landscape
that the metaheuristics optimize.

NOTE: nmo_times() must return times in ms, i.e. the offset term must be
converted with (offset / velocity * 1000). Otherwise the NMO correction
is ~1000x too small and the semblance no longer depends on velocity.
"""

import numpy as np
from .nmo import nmo_times

# Time gate length in samples. With dt = 2 ms, 15 samples = 30 ms, which is
# about one period of a 30 Hz Ricker wavelet. A gate shorter than one
# period gives a noisy semblance.
DEFAULT_WINDOW = 15


def _odd_window(window):
    """Force an odd window length so the gate is centered on the sample."""
    window = max(1, int(window))
    return window if window % 2 == 1 else window + 1


def _nmo_amplitudes(traces, offsets, velocity, t_samples, dt_ms, t0_ms=0.0):
    """
    Interpolate the amplitudes of all traces along the NMO trajectory
    of a given velocity, only for the requested zero-offset samples.

    Parameters
    ----------
    traces    : np.ndarray, shape (n_traces, n_samples)
    offsets   : np.ndarray, shape (n_traces,) in meters
    velocity  : float — stacking velocity in m/s
    t_samples : np.ndarray of int, shape (n_t,) — zero-offset sample indices
    dt_ms     : float — sampling interval in ms
    t0_ms     : float — start time in ms

    Returns
    -------
    amps : np.ndarray, shape (n_traces, n_t)
        Amplitudes at the NMO times. Samples whose NMO time falls outside
        the record are set to zero (muted).
    """
    n_traces, n_samples = traces.shape

    # Zero-offset times (ms) of the requested samples
    t0_array = t_samples.astype(np.float32) * dt_ms + t0_ms

    # NMO times for all traces: shape (n_traces, n_t)
    t_nmo = nmo_times(t0_array, offsets, velocity)

    # Convert to fractional sample indices
    idx       = (t_nmo - t0_ms) / dt_ms
    idx_floor = np.floor(idx).astype(int)
    idx_ceil  = idx_floor + 1
    frac      = idx - idx_floor

    # Samples inside the record (both interpolation neighbors must exist)
    valid = (idx_floor >= 0) & (idx_ceil < n_samples)

    # Clip only to keep the fancy indexing legal; invalid samples are
    # zeroed by the mask below
    f_idx = np.clip(idx_floor, 0, n_samples - 1)
    c_idx = np.clip(idx_ceil,  0, n_samples - 1)
    rows  = np.arange(n_traces)[:, None]

    # Linear interpolation, fully vectorized: shape (n_traces, n_t)
    amps = (1.0 - frac) * traces[rows, f_idx] + frac * traces[rows, c_idx]
    amps = np.where(valid, amps, 0.0).astype(np.float32)

    return amps


def compute_semblance(traces, offsets, velocities, dt_ms, t0_ms=0.0,
                      window=DEFAULT_WINDOW):
    """
    Compute semblance panel for a CDP gather.

    Parameters
    ----------
    traces     : np.ndarray, shape (n_traces, n_samples)
    offsets    : np.ndarray, shape (n_traces,) in meters
    velocities : np.ndarray, shape (n_vel,) in m/s
    dt_ms      : float — sampling interval in ms
    t0_ms      : float — start time in ms
    window     : int   — time gate length in samples (forced to be odd)

    Returns
    -------
    semblance : np.ndarray, shape (n_vel, n_samples)
        Values in [0, 1] — higher means better coherence
    """
    n_traces, n_samples = traces.shape
    n_vel  = len(velocities)
    window = _odd_window(window)

    all_samples = np.arange(n_samples)
    kernel      = np.ones(window)
    semblance   = np.zeros((n_vel, n_samples), dtype=np.float32)

    for iv, vel in enumerate(velocities):
        # Amplitudes along the NMO trajectory: shape (n_traces, n_samples)
        amps = _nmo_amplitudes(traces, offsets, vel, all_samples,
                               dt_ms, t0_ms)

        # Per-sample numerator and denominator
        num_t = np.sum(amps, axis=0) ** 2
        den_t = n_traces * np.sum(amps ** 2, axis=0)

        # Sum both terms inside the time gate (zero padded at the borders,
        # which matches the clipped gate of semblance_at_pick)
        numerator   = np.convolve(num_t, kernel, mode='same')
        denominator = np.convolve(den_t, kernel, mode='same')

        # Avoid division by zero
        mask = denominator > 1e-10
        semblance[iv, mask] = numerator[mask] / denominator[mask]

    return semblance


def semblance_at_pick(traces, offsets, velocity, time_sample, dt_ms,
                      t0_ms=0.0, window=DEFAULT_WINDOW):
    """
    Compute semblance at a single (velocity, time) pick.
    Used as objective function by metaheuristic algorithms.

    Only the samples inside the time gate are computed, so the cost does
    not depend on the record length. The result is identical to reading
    the (velocity, time) cell of compute_semblance() with the same window.

    Parameters
    ----------
    traces      : np.ndarray, shape (n_traces, n_samples)
    offsets     : np.ndarray, shape (n_traces,)
    velocity    : float — stacking velocity in m/s
    time_sample : int   — sample index of the pick
    dt_ms       : float — sampling interval in ms
    t0_ms       : float — start time in ms
    window      : int   — time gate length in samples (forced to be odd)

    Returns
    -------
    score : float in [0, 1]
    """
    n_traces, n_samples = traces.shape
    half = _odd_window(window) // 2

    # Time gate around the pick, clipped to the record
    t_start = max(0, int(time_sample) - half)
    t_end   = min(n_samples, int(time_sample) + half + 1)
    if t_end <= t_start:
        return 0.0

    t_samples = np.arange(t_start, t_end)

    # Amplitudes inside the gate only: shape (n_traces, n_gate)
    amps = _nmo_amplitudes(traces, offsets, velocity, t_samples,
                           dt_ms, t0_ms)

    numerator   = np.sum(np.sum(amps, axis=0) ** 2)
    denominator = n_traces * np.sum(amps ** 2)

    if denominator < 1e-10:
        return 0.0

    return float(numerator / denominator)


def picking_objective(traces, offsets, picks, dt_ms, t0_ms=0.0,
                      window=DEFAULT_WINDOW):
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
    window  : int — time gate length in samples

    Returns
    -------
    score : float in [0, 1] — higher is better
    """
    scores = [
        semblance_at_pick(traces, offsets, velocity, int(time_sample),
                          dt_ms, t0_ms, window)
        for time_sample, velocity in picks
    ]

    return float(np.mean(scores))