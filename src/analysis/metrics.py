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


def precision_recall(times_ms, vels, t_grid, v_rms_true, event_times_ms,
                     tol=0.02, time_tol_ms=40.0):
    """
    Detection scores for a variable number of picks (formulation v3).

    A pick MATCHES an event when it is within `time_tol_ms` of the event
    time AND its velocity is correct (|picked - true| <= tol * true at
    the pick time). With time_tol_ms below half the minimum separation
    between picks (and between events), each pick can match at most one
    event and each event at most one pick.

        precision = matched picks / picks    (how many picks are real)
        recall    = matched events / events  (how many events were found)
        f1        = harmonic mean of precision and recall

    Parameters
    ----------
    times_ms, vels       : picks (ms, m/s)
    t_grid, v_rms_true   : true RMS velocity on its time axis
    event_times_ms       : times of the reference events, e.g. the
                           natural events of src.analysis.events
                           (ground truth — evaluation only)
    tol                  : relative velocity tolerance (default 2%)
    time_tol_ms          : time tolerance between a pick and an event

    Returns
    -------
    dict with n_picks, n_events, n_vel_correct (picks with a correct
    velocity, wherever they are), n_matched_picks, n_found_events,
    precision, recall, f1, matched (bool per pick), found (bool per event)
    """
    times_ms = np.asarray(times_ms, dtype=float)
    vels     = np.asarray(vels, dtype=float)
    events   = np.asarray(event_times_ms, dtype=float)

    v_true   = np.interp(times_ms, t_grid, v_rms_true)
    vel_ok   = np.abs(vels - v_true) <= tol * v_true

    if times_ms.size and events.size:
        close = np.abs(times_ms[:, None] - events[None, :]) <= time_tol_ms
        hit   = close & vel_ok[:, None]                # (picks, events)
    else:
        hit = np.zeros((times_ms.size, events.size), dtype=bool)

    matched = hit.any(axis=1)
    found   = hit.any(axis=0)
    precision = float(matched.mean()) if times_ms.size else 0.0
    recall    = float(found.mean()) if events.size else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)

    return {
        'n_picks':          int(times_ms.size),
        'n_events':         int(events.size),
        'n_vel_correct':    int(vel_ok.sum()),
        'n_matched_picks':  int(matched.sum()),
        'n_found_events':   int(found.sum()),
        'precision':        precision,
        'recall':           recall,
        'f1':               float(f1),
        'matched':          [bool(m) for m in matched],
        'found':            [bool(f) for f in found],
    }