"""
src/seismic/synthetic.py
------------------------
Synthetic seismic data generation using ray tracing.
Generates CDP gathers from a velocity model using the
hyperbolic NMO equation and a Ricker wavelet source.

Physics:
    t(x)^2 = t0^2 + x^2 / v_rms^2

where:
    t0    = zero-offset two-way travel time
    x     = source-receiver offset
    v_rms = RMS velocity at time t0
"""

import numpy as np


# ------------------------------------------------------------------ #
# Wavelet                                                             #
# ------------------------------------------------------------------ #

def ricker_wavelet(dt_ms, f0_hz, duration_ms=200.0):
    """
    Generate a Ricker (Mexican hat) wavelet.

    Parameters
    ----------
    dt_ms       : float — sampling interval in ms
    f0_hz       : float — dominant frequency in Hz
    duration_ms : float — wavelet duration in ms

    Returns
    -------
    wavelet : np.ndarray, float32
    t       : np.ndarray — time axis in ms
    """
    t_half = duration_ms / 2.0
    t = np.arange(-t_half, t_half, dt_ms, dtype=np.float32)

    # Convert frequency to angular
    f0_ms = f0_hz / 1000.0           # Hz to cycles/ms
    u = (np.pi * f0_ms * t) ** 2

    wavelet = (1.0 - 2.0 * u) * np.exp(-u)
    wavelet /= np.abs(wavelet).max()  # normalize

    return wavelet, t


# ------------------------------------------------------------------ #
# Velocity model utilities                                            #
# ------------------------------------------------------------------ #

def depth_to_time(vel_profile, spacing_m):
    """
    Convert depth velocity profile to time domain using
    the vertical travel time integral.

    Parameters
    ----------
    vel_profile : np.ndarray, shape (n_depth,) — velocity in m/s
    spacing_m   : float — depth sampling in meters

    Returns
    -------
    t0_ms  : np.ndarray — two-way travel times in ms for each depth sample
    v_rms  : np.ndarray — RMS velocity at each t0
    """
    n = len(vel_profile)
    dt_one_way = spacing_m / vel_profile  # one-way time per sample (s)

    # Cumulative one-way time
    t_one_way = np.cumsum(dt_one_way)     # seconds

    # Two-way time in ms
    t0_ms = 2.0 * t_one_way * 1000.0

    # RMS velocity at each depth
    v_rms = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        v_rms[i] = np.sqrt(
            np.sum(vel_profile[:i+1]**2 * dt_one_way[:i+1]) /
            t_one_way[i]
        )
    v_rms[0] = vel_profile[0]

    return t0_ms.astype(np.float32), v_rms.astype(np.float32)


def resample_to_regular_time(t0_ms, v_rms, dt_ms, t_max_ms):
    """
    Resample v_rms from irregular depth grid to regular time grid.

    Parameters
    ----------
    t0_ms   : np.ndarray — two-way times at depth samples (ms)
    v_rms   : np.ndarray — RMS velocities at depth samples (m/s)
    dt_ms   : float      — output time sampling in ms
    t_max_ms: float      — maximum output time in ms

    Returns
    -------
    t_grid  : np.ndarray — regular time axis in ms
    v_grid  : np.ndarray — v_rms interpolated onto regular grid
    """
    t_grid = np.arange(0, t_max_ms + dt_ms, dt_ms, dtype=np.float32)
    v_grid = np.interp(t_grid, t0_ms, v_rms).astype(np.float32)
    return t_grid, v_grid


# ------------------------------------------------------------------ #
# Reflectivity                                                        #
# ------------------------------------------------------------------ #

def compute_reflectivity(vel_profile):
    """
    Compute reflection coefficients from velocity contrasts.

    R = (v2 - v1) / (v2 + v1)

    Parameters
    ----------
    vel_profile : np.ndarray, shape (n_depth,)

    Returns
    -------
    reflectivity : np.ndarray, shape (n_depth,)
    """
    r = np.zeros(len(vel_profile), dtype=np.float32)
    v1 = vel_profile[:-1]
    v2 = vel_profile[1:]
    r[1:] = (v2 - v1) / (v2 + v1)
    return r


# ------------------------------------------------------------------ #
# Trace generation                                                    #
# ------------------------------------------------------------------ #

def generate_trace(t_grid, v_rms, reflectivity_time,
                   offset_m, wavelet, dt_ms):
    """
    Generate a single synthetic seismic trace for a given offset.

    Steps:
        1. Apply NMO shift to reflectivity (hyperbolic moveout)
        2. Convolve with Ricker wavelet

    Parameters
    ----------
    t_grid           : np.ndarray — time axis in ms (n_samples,)
    v_rms            : np.ndarray — RMS velocity at each sample (m/s)
    reflectivity_time: np.ndarray — reflectivity on time grid (n_samples,)
    offset_m         : float      — source-receiver offset in meters
    wavelet          : np.ndarray — Ricker wavelet
    dt_ms            : float      — sampling interval in ms

    Returns
    -------
    trace : np.ndarray, float32, shape (n_samples,)
    """
    n = len(t_grid)
    shifted = np.zeros(n, dtype=np.float32)

    for i, t0 in enumerate(t_grid):
        if reflectivity_time[i] == 0:
            continue

        # NMO time: t^2 = t0^2 + x^2/v^2
        v = v_rms[i]
        if v < 1.0:
            continue

        t_nmo_ms = np.sqrt(t0**2 + (offset_m**2) / (v**2))
        idx = int(round(t_nmo_ms / dt_ms))

        if 0 <= idx < n:
            shifted[idx] += reflectivity_time[i]

    # Convolve with wavelet
    trace = np.convolve(shifted, wavelet, mode='same').astype(np.float32)
    return trace


# ------------------------------------------------------------------ #
# CDP gather generation                                               #
# ------------------------------------------------------------------ #

def generate_cdp_gather(vel_profile, spacing_m, offsets_m,
                        dt_ms=2.0, t_max_ms=3000.0,
                        f0_hz=30.0, noise_level=0.05,
                        seed=None):
    """
    Generate a full CDP gather from a 1D velocity profile.

    Parameters
    ----------
    vel_profile : np.ndarray — velocity in m/s at each depth sample
    spacing_m   : float      — depth sampling in meters
    offsets_m   : np.ndarray — source-receiver offsets in meters
    dt_ms       : float      — time sampling in ms (default 2.0)
    t_max_ms    : float      — maximum record time in ms (default 3000)
    f0_hz       : float      — Ricker wavelet frequency in Hz (default 30)
    noise_level : float      — Gaussian noise amplitude (default 0.05)
    seed        : int        — random seed for reproducibility

    Returns
    -------
    gather  : np.ndarray, shape (n_offsets, n_samples), float32
    t_grid  : np.ndarray — time axis in ms
    v_rms   : np.ndarray — RMS velocity on time grid (m/s)
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Depth to time conversion
    t0_ms, v_rms_depth = depth_to_time(vel_profile, spacing_m)

    # 2. Resample to regular time grid
    t_grid, v_rms = resample_to_regular_time(t0_ms, v_rms_depth,
                                              dt_ms, t_max_ms)
    n_samples = len(t_grid)

    # 3. Reflectivity in depth → resample to time
    refl_depth = compute_reflectivity(vel_profile)
    refl_time  = np.interp(t_grid, t0_ms, refl_depth).astype(np.float32)

    # 4. Ricker wavelet
    wavelet, _ = ricker_wavelet(dt_ms, f0_hz)

    # 5. Generate one trace per offset
    n_offsets = len(offsets_m)
    gather    = np.zeros((n_offsets, n_samples), dtype=np.float32)

    for i, offset in enumerate(offsets_m):
        trace = generate_trace(t_grid, v_rms, refl_time,
                               offset, wavelet, dt_ms)
        # Add Gaussian noise
        if noise_level > 0:
            noise  = np.random.randn(n_samples).astype(np.float32)
            noise *= noise_level * np.abs(trace).max()
            trace += noise

        gather[i] = trace

    return gather, t_grid, v_rms


# ------------------------------------------------------------------ #
# Multi-CDP generation                                                #
# ------------------------------------------------------------------ #

def generate_dataset(vel_model, spacing_m, cdp_indices,
                     offsets_m, dt_ms=2.0, t_max_ms=3000.0,
                     f0_hz=30.0, noise_level=0.05,
                     verbose=True, seed=42):
    """
    Generate synthetic CDP gathers for multiple CDPs.

    Parameters
    ----------
    vel_model   : np.ndarray, shape (n_cdp, n_depth) — velocity model
    spacing_m   : float      — spatial sampling in meters
    cdp_indices : list/array — which CDP columns to generate
    offsets_m   : np.ndarray — offsets in meters
    dt_ms       : float      — time sampling in ms
    t_max_ms    : float      — max record time in ms
    f0_hz       : float      — wavelet frequency in Hz
    noise_level : float      — noise amplitude
    verbose     : bool       — print progress
    seed        : int        — base random seed

    Returns
    -------
    dataset : dict
        keys   : CDP index (int)
        values : dict with 'gather', 'offsets', 't_grid', 'v_rms_true'
    """
    dataset = {}

    for k, cdp_idx in enumerate(cdp_indices):
        if verbose and k % max(1, len(cdp_indices) // 10) == 0:
            print(f'  Generating CDP {cdp_idx} '
                  f'({k+1}/{len(cdp_indices)})...')

        vel_profile = vel_model[cdp_idx, :]

        gather, t_grid, v_rms = generate_cdp_gather(
            vel_profile, spacing_m, offsets_m,
            dt_ms=dt_ms, t_max_ms=t_max_ms,
            f0_hz=f0_hz, noise_level=noise_level,
            seed=seed + cdp_idx
        )

        dataset[int(cdp_idx)] = {
            'gather':    gather,
            'offsets':   offsets_m.copy(),
            't_grid':    t_grid,
            'v_rms_true': v_rms,
        }

    if verbose:
        print(f'Dataset generated: {len(dataset)} CDPs')
        print(f'  Gather shape: {gather.shape}')
        print(f'  T max       : {t_grid[-1]:.1f} ms')

    return dataset