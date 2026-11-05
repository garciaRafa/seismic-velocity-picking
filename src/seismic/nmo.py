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


# ---------------------------------------------------------------------- #
# Time-variant NMO (a velocity function from picks)                       #
# ---------------------------------------------------------------------- #

def velocity_function(times_ms, vels, t0_array):
    """
    Stacking velocity at every zero-offset time, from a set of picks:
    linear between picks, constant above the first and below the last
    (the usual interpretation of a picked velocity function).

    Parameters
    ----------
    times_ms : array-like — pick times (ms), any order
    vels     : array-like — pick velocities (m/s)
    t0_array : np.ndarray — zero-offset times (ms) where it is evaluated

    Returns
    -------
    v_t0 : np.ndarray, shape like t0_array (float32)
    """
    t = np.asarray(times_ms, dtype=float).ravel()
    v = np.asarray(vels, dtype=float).ravel()
    if t.size == 0 or t.size != v.size:
        raise ValueError("velocity_function needs at least one pick, with one velocity per time.")
    order = np.argsort(t)
    return np.interp(np.asarray(t0_array, dtype=float), t[order], v[order]).astype(np.float32)


def apply_nmo_variable(traces, offsets, v_t0, dt_ms, t0_ms=0.0, stretch_mute=None):
    """
    NMO correction with a time-variant velocity: the sample at zero-offset
    time t0 takes the amplitude at t = sqrt(t0^2 + (1000 x / v(t0))^2)
    (linear interpolation). With the right velocity function every
    primary becomes flat; an event whose velocity differs from v(t0)
    stays curved (down if its velocity is lower, as for a multiple).

    Parameters
    ----------
    traces       : np.ndarray, shape (n_traces, n_samples)
    offsets      : np.ndarray, shape (n_traces,) in meters
    v_t0         : float or np.ndarray, shape (n_samples,) — velocity at
                   each zero-offset time (m/s), e.g. from velocity_function
    dt_ms        : float — sampling interval in ms
    t0_ms        : float — start time in ms (default 0)
    stretch_mute : float or None — mute samples whose NMO stretch
                   (t - t0) / t0 exceeds this value (e.g. 0.5, as in the
                   robust semblance); None keeps every sample

    Returns
    -------
    corrected : np.ndarray, shape (n_traces, n_samples) — muted samples are 0
    live      : np.ndarray of bool, same shape — samples that were kept
    """
    traces = np.asarray(traces)
    n_traces, n_samples = traces.shape
    t0 = (np.arange(n_samples, dtype=np.float64) * dt_ms + t0_ms)[None, :]
    v = np.broadcast_to(np.asarray(v_t0, dtype=np.float64), (n_samples,))[None, :]
    x = np.asarray(offsets, dtype=np.float64)[:, None]

    t_nmo = np.sqrt(t0 ** 2 + (1000.0 * x / v) ** 2)
    idx = (t_nmo - t0_ms) / dt_ms
    i0 = np.floor(idx).astype(np.int64)
    frac = idx - i0
    live = (i0 >= 0) & (i0 + 1 < n_samples)
    if stretch_mute is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            stretch = np.where(t0 > 0, (t_nmo - t0) / t0, np.where(x == 0, 0.0, np.inf))
        live &= stretch <= stretch_mute

    i0c = np.clip(i0, 0, n_samples - 2)
    a = np.take_along_axis(traces, i0c, axis=1)
    b = np.take_along_axis(traces, i0c + 1, axis=1)
    corrected = np.where(live, (1.0 - frac) * a + frac * b, 0.0).astype(traces.dtype)
    return corrected, live


def stack_live(corrected, live):
    """
    Stack of an NMO-corrected gather averaging only the samples that were
    kept (live), so the muted zone does not dim the shallow part.

    Returns
    -------
    stacked : np.ndarray, shape (n_samples,)
    """
    n = np.asarray(live).sum(axis=0)
    return np.where(n > 0, np.asarray(corrected).sum(axis=0) / np.maximum(n, 1), 0.0)