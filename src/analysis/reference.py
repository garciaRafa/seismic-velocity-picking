"""
src/analysis/reference.py
-------------------------
Reference picks taken from the semblance panel ("the blob") instead of
the model V_rms, and the detection scores against them.

Why
    The model gives, for each reflector, its vertical time t0 and the
    column RMS velocity. On finite-difference data the semblance maximum
    of that event is not exactly there: the real moveout is not a
    perfect hyperbola (long offsets, post-critical reflections), so the
    best-fitting hyperbola is 1-3% faster and slightly shifted in time.
    What a picker (human or algorithm) chooses is that best hyperbola,
    the stacking velocity. The reference is therefore the semblance peak
    of each expected event: the best answer the data allow.

How
    For every expected event (t0, v) the semblance is computed on a
    small box around it (time +- time_win_ms, velocity +- vel_win) and the
    highest LOCAL maximum inside the box is kept. A local maximum (and not
    simply the largest value of the box) avoids taking the flank of a
    stronger neighboring event that enters the box; when the box has no
    interior local maximum, its largest value is used and `on_edge` is
    set, so the case can be inspected.

The model is used only to say WHERE to look (evaluation only); the
algorithms never see these picks.
"""

import numpy as np

from src.seismic.semblance import compute_semblance


def semblance_peak(gather, offsets, dt_ms, t0_ms, v0, time_win_ms=40.0,
                   vel_win=0.08, vel_step=5.0, t_start_ms=0.0):
    """
    Semblance peak of one event near (t0_ms, v0).

    Returns
    -------
    dict with t_ms, v (peak position), s (semblance at the peak) and
    on_edge (True when no interior local maximum exists in the box)
    """
    n_samples = gather.shape[1]
    t_grid = t_start_ms + dt_ms * np.arange(n_samples)
    rows = np.flatnonzero(np.abs(t_grid - t0_ms) <= time_win_ms)
    vels = np.arange(v0 * (1 - vel_win), v0 * (1 + vel_win) + vel_step / 2,
                     vel_step, dtype=np.float32)

    panel = np.asarray(compute_semblance(gather, offsets, vels, dt_ms, t_start_ms))
    box = panel[:, rows].T                     # (time, velocity)

    # Interior local maxima (3 x 3 neighborhood)
    c = box[1:-1, 1:-1]
    is_max = np.ones_like(c, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                is_max &= c >= box[1 + di:box.shape[0] - 1 + di,
                                   1 + dj:box.shape[1] - 1 + dj]
    if is_max.any():
        i, j = np.unravel_index(np.argmax(np.where(is_max, c, -np.inf)), c.shape)
        i, j, on_edge = i + 1, j + 1, False
    else:
        i, j = np.unravel_index(np.argmax(box), box.shape)
        on_edge = True

    return {'t_ms': float(t_grid[rows[i]]), 'v': float(vels[j]),
            's': float(box[i, j]), 'on_edge': bool(on_edge)}


def reference_picks(gather, offsets, dt_ms, primaries, multiples=(), **kwargs):
    """
    Semblance peaks of every expected event.

    primaries : list of dicts with name, t0_ms, v_rms (case meta)
    multiples : list of dicts with name, t0_ms, v_nmo (case meta)
    kwargs    : passed to semblance_peak (time_win_ms, vel_win, vel_step)

    Returns a list of dicts: name, kind ('primary' / 'multiple'),
    t0_ms and v_model (from the model) and t_ms, v, s, on_edge (peak).
    """
    out = []
    for kind, events, vkey in (('primary', primaries, 'v_rms'),
                               ('multiple', multiples, 'v_nmo')):
        for ev in events:
            peak = semblance_peak(gather, offsets, dt_ms, ev['t0_ms'], ev[vkey], **kwargs)
            out.append({'name': ev['name'], 'kind': kind,
                        't0_ms': float(ev['t0_ms']), 'v_model': float(ev[vkey]),
                        **peak})
    return out


def match_reference(times_ms, vels, reference, time_tol_ms=40.0, vel_tol=0.02):
    """
    Detection scores against reference picks (see reference_picks).

    A pick MATCHES a reference point when it is within time_tol_ms of its
    time AND within vel_tol (relative) of its velocity. Each pick is
    labeled with the closest matching point (in time), primary or
    multiple.

        precision = picks on a primary / picks
        recall    = primaries found   / primaries
        f1        = harmonic mean of precision and recall

    Picks on a multiple are counted apart (n_on_multiple): they are wrong
    picks, but a different kind of wrong than a pick on nothing.

    Returns
    -------
    dict with n_picks, n_primaries, n_on_primary, n_on_multiple,
    n_found, precision, recall, f1, label (per pick: event name or None),
    kind (per pick: 'primary' / 'multiple' / None), found (per primary)
    """
    times_ms = np.asarray(times_ms, dtype=float)
    vels = np.asarray(vels, dtype=float)
    ref_t = np.array([r['t_ms'] for r in reference], dtype=float)
    ref_v = np.array([r['v'] for r in reference], dtype=float)
    kinds = np.array([r['kind'] for r in reference])

    if times_ms.size and ref_t.size:
        dt = np.abs(times_ms[:, None] - ref_t[None, :])
        hit = (dt <= time_tol_ms) & (np.abs(vels[:, None] - ref_v[None, :])
                                     <= vel_tol * ref_v[None, :])
        dt = np.where(hit, dt, np.inf)
    else:
        hit = np.zeros((times_ms.size, ref_t.size), dtype=bool)
        dt = np.full(hit.shape, np.inf)

    label, kind = [], []
    for row_hit, row_dt in zip(hit, dt):
        if row_hit.any():
            k = int(np.argmin(row_dt))
            label.append(reference[k]['name'])
            kind.append(str(kinds[k]))
        else:
            label.append(None)
            kind.append(None)

    is_primary = kinds == 'primary'
    found = hit[:, is_primary].any(axis=0) if times_ms.size else np.zeros(is_primary.sum(), bool)
    n_on_primary = sum(k == 'primary' for k in kind)
    n_on_multiple = sum(k == 'multiple' for k in kind)

    precision = n_on_primary / times_ms.size if times_ms.size else 0.0
    recall = float(found.mean()) if found.size else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        'n_picks': int(times_ms.size),
        'n_primaries': int(is_primary.sum()),
        'n_on_primary': int(n_on_primary),
        'n_on_multiple': int(n_on_multiple),
        'n_found': int(found.sum()),
        'precision': float(precision),
        'recall': recall,
        'f1': float(f1),
        'label': label,
        'kind': kind,
        'found': [bool(f) for f in found],
    }