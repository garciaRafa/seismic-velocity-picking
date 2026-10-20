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

Robust semblance (optional)
---------------------------
Three options, all off by default (with the defaults every function
returns exactly the values of the plain semblance above):

    stretch_mute : float or None — NMO stretch mute. A trace enters the
                   sum at (t0, v) only when its stretch
                   (t_nmo - t0) / t0 <= stretch_mute (e.g. 0.5) and its
                   NMO time is inside the record. Far offsets at shallow
                   times are distorted by NMO (post-critical reflections,
                   refractions, a stretched wavelet) and pull the semblance
                   maxima away from the reflections.
    energy_floor : float — stabilization of the denominator,
                   floor = energy_floor * E_ref * N_live^2, with E_ref the
                   mean energy of the gather over one gate (gather_energy).
                   Where the data are almost zero (e.g. between the
                   direct-wave mute and the first reflection) the plain
                   semblance normalizes numerical noise up to high values;
                   the floor sends it to 0 and leaves real events
                   untouched.
    min_fold     : int — minimum number of live traces at the analysis
                   time; with fewer, S = 0 (a handful of traces is
                   coherent too easily, e.g. at the end of the record).

With any option on, M in the formula becomes N_live(t), the number of
live traces at each time:

    S = sum_t (sum_live f_i)^2 / ( sum_t N_live(t) * sum_live f_i^2
                                   + energy_floor * E_ref * N_live(t0)^2 )
"""

import numpy as np
from .nmo import nmo_times

# Time gate length in samples. With dt = 2 ms, 15 samples = 30 ms, which is
# about one period of a 30 Hz Ricker wavelet. A gate shorter than one
# period gives a noisy semblance.
DEFAULT_WINDOW = 15

# Robust-semblance setting used with the finite-difference cases
# ("v3 + stretch mute"): stretch up to 50%, energy floor of 1% of the
# gather energy, at least 10 live traces (20% of a 50-trace gather).
# Validated on fd_layered_cdp2500_fs / _nofs: removes the mute-zone
# artifact and brings the far-offset-distorted maxima back to the
# reflections, while keeping the water bottom.
ROBUST_SEMBLANCE = {'stretch_mute': 0.5, 'energy_floor': 1e-2, 'min_fold': 10}


def _odd_window(window):
    """Force an odd window length so the gate is centered on the sample."""
    window = max(1, int(window))
    return window if window % 2 == 1 else window + 1


def _nmo_amplitudes(traces, offsets, velocity, t_samples, dt_ms, t0_ms=0.0,
                    return_valid=False):
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
    With return_valid=True, also returns valid (bool, same shape: NMO time
    inside the record), t_nmo (NMO times, ms) and t0_array (zero-offset
    times of the requested samples, ms).
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

    if return_valid:
        return amps, valid, t_nmo, t0_array
    return amps


# ---------------------------------------------------------------------- #
# Robust semblance helpers                                                #
# ---------------------------------------------------------------------- #

def _is_robust(stretch_mute, energy_floor, min_fold):
    """True when any robust-semblance option is on."""
    return stretch_mute is not None or energy_floor > 0 or min_fold > 1


def gather_energy(traces, window=DEFAULT_WINDOW):
    """
    Reference energy E_ref of the energy floor: mean squared amplitude of
    the gather times the gate length. Compute it once per gather and pass
    it as e_ref to avoid recomputing it at every evaluation.
    """
    return float(np.mean(np.asarray(traces, dtype=np.float64) ** 2)) * _odd_window(window)


def _live_traces(valid, t_nmo, t0_array, stretch_mute, dt_ms):
    """Live samples: NMO time inside the record and stretch within the limit."""
    live = valid
    if stretch_mute is not None:
        t0 = np.maximum(t0_array, dt_ms)[None, :]
        live = live & ((t_nmo - t0_array[None, :]) / t0 <= stretch_mute)
    return live


def compute_semblance(traces, offsets, velocities, dt_ms, t0_ms=0.0,
                      window=DEFAULT_WINDOW, stretch_mute=None,
                      energy_floor=0.0, min_fold=1, e_ref=None):
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
    stretch_mute, energy_floor, min_fold : robust options (module
                 docstring); off by default
    e_ref      : float or None — gather_energy(traces, window), computed
                 when None and the energy floor is on

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

    if _is_robust(stretch_mute, energy_floor, min_fold):
        if energy_floor > 0 and e_ref is None:
            e_ref = gather_energy(traces, window)
        for iv, vel in enumerate(velocities):
            amps, valid, t_nmo, t0_arr = _nmo_amplitudes(
                traces, offsets, vel, all_samples, dt_ms, t0_ms, return_valid=True)
            live = _live_traces(valid, t_nmo, t0_arr, stretch_mute, dt_ms)
            amps = np.where(live, amps, 0.0)
            n_live = live.sum(axis=0).astype(np.float64)

            numerator   = np.convolve(np.sum(amps, axis=0) ** 2, kernel, mode='same')
            denominator = np.convolve(n_live * np.sum(amps ** 2, axis=0), kernel, mode='same')
            ok = (denominator > 1e-10) & (n_live >= min_fold)
            den_ok = denominator[ok]
            if energy_floor > 0:
                den_ok = den_ok + energy_floor * e_ref * n_live[ok] ** 2
            semblance[iv, ok] = numerator[ok] / den_ok
        return semblance

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
                      t0_ms=0.0, window=DEFAULT_WINDOW, stretch_mute=None,
                      energy_floor=0.0, min_fold=1, e_ref=None):
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
    stretch_mute, energy_floor, min_fold, e_ref : robust options, as in
                  compute_semblance() (same value as the panel cell)

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

    if _is_robust(stretch_mute, energy_floor, min_fold):
        amps, valid, t_nmo, t0_arr = _nmo_amplitudes(
            traces, offsets, velocity, t_samples, dt_ms, t0_ms, return_valid=True)
        live = _live_traces(valid, t_nmo, t0_arr, stretch_mute, dt_ms)
        amps = np.where(live, amps, 0.0)
        n_live = live.sum(axis=0).astype(np.float64)
        n_center = n_live[int(time_sample) - t_start]
        if n_center < min_fold:
            return 0.0
        numerator   = np.sum(np.sum(amps, axis=0) ** 2)
        denominator = np.sum(n_live * np.sum(amps ** 2, axis=0))
        if denominator < 1e-10:
            return 0.0
        if energy_floor > 0:
            if e_ref is None:
                e_ref = gather_energy(traces, window)
            denominator += energy_floor * e_ref * n_center ** 2
        return float(numerator / denominator)

    # Amplitudes inside the gate only: shape (n_traces, n_gate)
    amps = _nmo_amplitudes(traces, offsets, velocity, t_samples,
                           dt_ms, t0_ms)

    numerator   = np.sum(np.sum(amps, axis=0) ** 2)
    denominator = n_traces * np.sum(amps ** 2)

    if denominator < 1e-10:
        return 0.0

    return float(numerator / denominator)


def _robust_kwargs(traces, window, stretch_mute, energy_floor, min_fold, e_ref):
    """Robust options for semblance_at_pick, with E_ref computed once."""
    if energy_floor > 0 and e_ref is None:
        e_ref = gather_energy(traces, window)
    return dict(stretch_mute=stretch_mute, energy_floor=energy_floor,
                min_fold=min_fold, e_ref=e_ref)


def picking_objective(traces, offsets, picks, dt_ms, t0_ms=0.0,
                      window=DEFAULT_WINDOW, stretch_mute=None,
                      energy_floor=0.0, min_fold=1, e_ref=None):
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
    opts = _robust_kwargs(traces, window, stretch_mute, energy_floor, min_fold, e_ref)
    scores = [
        semblance_at_pick(traces, offsets, velocity, int(time_sample),
                          dt_ms, t0_ms, window, **opts)
        for time_sample, velocity in picks
    ]

    return float(np.mean(scores))

def semblance_along_curve(traces, offsets, v_curve, dt_ms, t0_ms=0.0,
                          window=DEFAULT_WINDOW, stretch_mute=None,
                          energy_floor=0.0, min_fold=1, e_ref=None):
    """
    Semblance along a velocity function v(t): one value per time sample,
    S(t_i, v(t_i)), with the same time gate as the objective function.

    Typical use is validation only: evaluated along the TRUE RMS velocity
    of a synthetic case, it shows which reflections an ideal velocity
    function would see. It is not used by the optimizers.

    Parameters
    ----------
    traces  : np.ndarray, shape (n_traces, n_samples)
    offsets : np.ndarray, shape (n_traces,)
    v_curve : np.ndarray, shape (n_samples,) — velocity (m/s) at each sample
    dt_ms   : float
    t0_ms   : float
    window  : int — time gate length in samples

    Returns
    -------
    s_curve : np.ndarray, shape (n_samples,), values in [0, 1]
    """
    v_curve = np.asarray(v_curve, dtype=np.float64)
    if v_curve.shape[0] != traces.shape[1]:
        raise ValueError("v_curve must have one velocity per time sample.")
    opts = _robust_kwargs(traces, window, stretch_mute, energy_floor, min_fold, e_ref)
    return np.array([
        semblance_at_pick(traces, offsets, float(v), i, dt_ms, t0_ms, window, **opts)
        for i, v in enumerate(v_curve)
    ])


def robust_threshold(panel, factor=10.0):
    """
    Blind semblance threshold, computed from the panel alone (no model or
    ground-truth information):

        tau = median(panel) + factor * sigma_rob,  sigma_rob = 1.4826 * MAD(panel)

    The semblance ridge (coherent reflections) covers a small fraction of
    the panel, so the median and the median absolute deviation (MAD)
    describe the incoherent background, and are barely moved by the ridge.
    The factor 1.4826 makes sigma_rob match the standard deviation for a
    normal distribution; semblance is skewed (bounded at 0), so `factor`
    should be read as "number of robust spreads above the background", not
    as a Gaussian z-score with a probability attached. (It is called
    `factor`, not k, to avoid confusion with K, the number of picks.)

    Parameters
    ----------
    panel : np.ndarray — semblance values (any shape), e.g. the output of
            compute_semblance over the velocity range of the formulation
    factor : float — number of robust spreads above the median

    Returns
    -------
    tau : float
    """
    values = np.asarray(panel, dtype=np.float64).ravel()
    med = np.median(values)
    sigma_rob = 1.4826 * np.median(np.abs(values - med))
    return float(med + factor * sigma_rob)


def picking_objective_threshold(traces, offsets, picks, dt_ms, tau,
                                t0_ms=0.0, window=DEFAULT_WINDOW, stretch_mute=None,
                                energy_floor=0.0, min_fold=1, e_ref=None):
    """
    Objective of formulation v3 (variable number of picks):

        f = sum_k ( S(t_k, v_k) - tau )

    Each pick adds its semblance minus a fixed cost tau, so a pick only
    improves the solution when its semblance is above the threshold.
    Equivalently f = sum_k S_k - tau * K: total semblance with a penalty
    proportional to the number of picks K (a cardinality penalty).

    Parameters
    ----------
    traces, offsets, dt_ms, t0_ms, window : as in picking_objective()
    picks : list of (time_sample, velocity) — any number of picks (an
            empty list gives 0)
    tau   : float — threshold, e.g. from robust_threshold() (computed on a
            panel with the SAME robust options)
    stretch_mute, energy_floor, min_fold, e_ref : robust options, as in
            compute_semblance()

    Returns
    -------
    score : float — higher is better; not bounded to [0, 1]
    """
    opts = _robust_kwargs(traces, window, stretch_mute, energy_floor, min_fold, e_ref)
    return float(sum(
        semblance_at_pick(traces, offsets, velocity, int(time_sample),
                          dt_ms, t0_ms, window, **opts) - tau
        for time_sample, velocity in picks
    ))