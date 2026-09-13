"""
src/seismic/io.py
-----------------
Seismic data I/O utilities.
Supports SEG-Y (.segy) and raw binary (.bin) formats.
"""

import numpy as np
import os


def load_segy(path, max_traces=None, verbose=True):
    """
    Load a SEG-Y file and return data as numpy array.

    Parameters
    ----------
    path       : str  — path to .segy file
    max_traces : int  — limit number of traces (None = all)
    verbose    : bool — print file info

    Returns
    -------
    data : np.ndarray, shape (n_traces, n_samples), float32
    info : dict with metadata
    """
    import segyio

    if not os.path.exists(path):
        raise FileNotFoundError(f"SEG-Y file not found: {path}")

    with segyio.open(path, ignore_geometry=True) as f:
        n_traces  = f.tracecount
        n_samples = len(f.samples)
        dt_ms     = segyio.dt(f) / 1000.0  # microseconds to ms

        if max_traces is not None:
            n_traces = min(n_traces, max_traces)

        data = np.zeros((n_traces, n_samples), dtype=np.float32)
        for i in range(n_traces):
            data[i] = f.trace[i]

        info = {
            'n_traces':  n_traces,
            'n_samples': n_samples,
            'dt_ms':     dt_ms,
            't_max_ms':  dt_ms * (n_samples - 1),
            'path':      path,
            'format':    'segy',
        }

    if verbose:
        print(f"Loaded : {os.path.basename(path)}")
        print(f"  Traces  : {info['n_traces']}")
        print(f"  Samples : {info['n_samples']}")
        print(f"  dt      : {info['dt_ms']:.4f} ms")
        print(f"  T max   : {info['t_max_ms']:.1f} ms")

    return data, info


def load_bin(path, n1, n2, dtype=np.float32,
             spacing_m=1.25, verbose=True):
    """
    Load a raw binary file as a 2D numpy array.
    Used for Marmousi2 .bin velocity models.

    The file is stored as (n2, n1) — horizontal x vertical.
    Returned array is transposed to (n2, n1) with shape
    (n_horizontal, n_vertical) for consistency with seismic convention.

    Parameters
    ----------
    path      : str            — path to .bin file
    n1        : int            — number of vertical samples (depth)
    n2        : int            — number of horizontal samples (distance)
    dtype     : np.dtype       — data type (default float32)
    spacing_m : float          — spatial sampling in meters (default 1.25)
    verbose   : bool           — print file info

    Returns
    -------
    data : np.ndarray, shape (n2, n1), float32
           axis 0 = horizontal distance
           axis 1 = vertical depth
    info : dict with metadata
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Binary file not found: {path}")

    raw = np.fromfile(path, dtype=dtype)
    expected = n1 * n2

    if len(raw) != expected:
        raise ValueError(
            f"File has {len(raw)} values, expected {expected} "
            f"(n1={n1} x n2={n2})."
        )

    data = raw.reshape(n2, n1)

    info = {
        'n1':         n1,
        'n2':         n2,
        'n_vertical':    n1,
        'n_horizontal':  n2,
        'spacing_m':  spacing_m,
        'x_max_km':   n2 * spacing_m / 1000.0,
        'z_max_km':   n1 * spacing_m / 1000.0,
        'val_min':    float(data.min()),
        'val_max':    float(data.max()),
        'val_mean':   float(data.mean()),
        'path':       path,
        'format':     'bin',
    }

    if verbose:
        print(f"Loaded : {os.path.basename(path)}")
        print(f"  Shape    : {data.shape}  (n2 x n1)")
        print(f"  X extent : {info['x_max_km']:.2f} km")
        print(f"  Z extent : {info['z_max_km']:.2f} km")
        print(f"  Min      : {info['val_min']:.1f}")
        print(f"  Max      : {info['val_max']:.1f}")
        print(f"  Mean     : {info['val_mean']:.1f}")

    return data, info


def load_marmousi2_velocity(models_dir, verbose=True):
    """
    Convenience function to load the Marmousi2 P-wave velocity model.

    Parameters
    ----------
    models_dir : str  — path to data/models/marmousi2/
    verbose    : bool — print info

    Returns
    -------
    vel  : np.ndarray, shape (13601, 2801), float32  — velocity in m/s
    info : dict with metadata
    """
    path = os.path.join(models_dir, 'MODEL_P-WAVE_VELOCITY_1.25m.bin')

    return load_bin(
        path,
        n1=2801,
        n2=13601,
        dtype=np.float32,
        spacing_m=1.25,
        verbose=verbose
    )


def load_headers(path, keys=None):
    """
    Load trace headers from a SEG-Y file.

    Parameters
    ----------
    path : str  — path to .segy file
    keys : list — header keys to extract (None = common defaults)

    Returns
    -------
    headers : dict of np.ndarray, one entry per key
    """
    import segyio

    if keys is None:
        keys = ['cdp', 'offset', 'iline', 'xline',
                'source_x', 'source_y', 'groupx', 'groupy']

    headers = {k: [] for k in keys}

    with segyio.open(path, ignore_geometry=True) as f:
        for i in range(f.tracecount):
            h = f.header[i]
            for k in keys:
                try:
                    headers[k].append(
                        h[getattr(segyio.TraceField,
                                  k.upper(), None) or k]
                    )
                except Exception:
                    headers[k].append(0)

    return {k: np.array(v) for k, v in headers.items()}