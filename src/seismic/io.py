"""
src/seismic/io.py
-----------------
SEG-Y file reading utilities for the Marmousi2 dataset.
"""

import numpy as np
import segyio
import os


def load_segy(path, max_traces=None, verbose=True):
    """
    Load a SEG-Y file and return data as numpy array.

    Parameters
    ----------
    path       : str   — path to .segy file
    max_traces : int   — limit number of traces (None = all)
    verbose    : bool  — print file info

    Returns
    -------
    data : np.ndarray, shape (n_traces, n_samples), float32
    info : dict with metadata
    """
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
        }

    if verbose:
        print(f"Loaded: {os.path.basename(path)}")
        print(f"  Traces  : {info['n_traces']}")
        print(f"  Samples : {info['n_samples']}")
        print(f"  dt      : {info['dt_ms']:.4f} ms")
        print(f"  T max   : {info['t_max_ms']:.1f} ms")

    return data, info


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
    if keys is None:
        keys = ['cdp', 'offset', 'iline', 'xline',
                'source_x', 'source_y',
                'groupx',   'groupy']

    headers = {k: [] for k in keys}

    with segyio.open(path, ignore_geometry=True) as f:
        for i in range(f.tracecount):
            h = f.header[i]
            for k in keys:
                try:
                    headers[k].append(h[getattr(segyio.TraceField,
                                                 k.upper(), None) or k])
                except Exception:
                    headers[k].append(0)

    return {k: np.array(v) for k, v in headers.items()}
