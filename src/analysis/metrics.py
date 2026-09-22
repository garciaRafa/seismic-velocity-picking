"""
src/analysis/metrics.py
-----------------------
Accuracy metrics of velocity picks against the true RMS velocity.
"""

import numpy as np


def velocity_errors(times_ms, vels, t_grid, v_rms_true):
    """
    Compare picked velocities with the true RMS velocity at the pick times.

    Parameters
    ----------
    times_ms   : np.ndarray — pick times in ms
    vels       : np.ndarray — picked velocities in m/s
    t_grid     : np.ndarray — time axis of the true velocity (ms)
    v_rms_true : np.ndarray — true RMS velocity on t_grid (m/s)

    Returns
    -------
    dict with
        rmse, mae (m/s), mape (%) — summary errors
        v_true  : true velocity at each pick time
        errors  : picked - true, per pick (m/s)
    """
    times_ms = np.asarray(times_ms, dtype=float)
    vels     = np.asarray(vels, dtype=float)
    v_true   = np.interp(times_ms, t_grid, v_rms_true)
    errors   = vels - v_true

    return {
        'rmse':   float(np.sqrt(np.mean(errors ** 2))),
        'mae':    float(np.mean(np.abs(errors))),
        'mape':   float(np.mean(np.abs(errors / v_true)) * 100.0),
        'v_true': v_true.tolist(),
        'errors': errors.tolist(),
    }


def count_correct_picks(times_ms, vels, t_grid, v_rms_true, tol=0.02):
    """
    Count picks whose velocity error is within a relative tolerance.

    A pick is "correct" if |picked - true| <= tol * true, where true is the
    RMS velocity at the pick time. With tol = 0.02 (2%), a pick at
    2000 m/s is correct if it is within 40 m/s of the true velocity.

    Parameters
    ----------
    times_ms, vels, t_grid, v_rms_true : as in velocity_errors()
    tol : float — relative tolerance (default 0.02)

    Returns
    -------
    dict with
        n_correct    : number of correct picks
        frac_correct : n_correct / number of picks
        correct      : list of booleans, one per pick
    """
    times_ms = np.asarray(times_ms, dtype=float)
    vels     = np.asarray(vels, dtype=float)
    v_true   = np.interp(times_ms, t_grid, v_rms_true)
    correct  = np.abs(vels - v_true) <= tol * v_true

    return {
        'n_correct':    int(correct.sum()),
        'frac_correct': float(correct.mean()),
        'correct':      [bool(c) for c in correct],
    }
