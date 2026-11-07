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

Alternates (optional, alt_ratio)
    Some events have two (or more) comparable maxima inside the box — the
    water bottom, for instance, has one near 506 ms / 1520 m/s and one
    near 544 ms / 1490 m/s, the same reflection. A picker may choose
    either; with a single point and a 2% tolerance the other one counts as
    a wrong pick AND a missed event. With alt_ratio set, every other
    interior local maximum of the box with S >= alt_ratio * S_peak is kept
    in 'alternates', and match_reference accepts a pick near the peak or
    near any alternate. Off by default (the reference is the single peak,
    as before).

The model is used only to say WHERE to look (evaluation only); the
algorithms never see these picks.
"""

import numpy as np

from src.seismic.semblance import _is_robust, compute_semblance

ROBUST_OPTION_KEYS = ('stretch_mute', 'energy_floor', 'min_fold')


def semblance_peak(gather, offsets, dt_ms, t0_ms, v0, time_win_ms=40.0,
                   vel_win=0.08, vel_step=5.0, t_start_ms=0.0, alt_ratio=None,
                   **semblance_opts):
    """
    Semblance peak of one event near (t0_ms, v0).

    semblance_opts : robust-semblance options (stretch_mute, energy_floor,
                     min_fold, e_ref), passed to compute_semblance — use the
                     same ones as the objective being evaluated
    alt_ratio      : None (default) or a fraction in (0, 1]: also return the
                     other interior local maxima with S >= alt_ratio * S_peak

    Returns
    -------
    dict with t_ms, v (peak position), s (semblance at the peak) and
    on_edge (True when no interior local maximum exists in the box); with
    alt_ratio, also alternates: list of {t_ms, v, s}, strongest first
    """
    n_samples = gather.shape[1]
    t_grid = t_start_ms + dt_ms * np.arange(n_samples)
    rows = np.flatnonzero(np.abs(t_grid - t0_ms) <= time_win_ms)
    vels = np.arange(v0 * (1 - vel_win), v0 * (1 + vel_win) + vel_step / 2,
                     vel_step, dtype=np.float32)

    panel = np.asarray(compute_semblance(gather, offsets, vels, dt_ms, t_start_ms,
                                         **semblance_opts))
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

    out = {'t_ms': float(t_grid[rows[i]]), 'v': float(vels[j]),
           's': float(box[i, j]), 'on_edge': bool(on_edge)}
    if alt_ratio is not None:
        if not 0.0 < alt_ratio <= 1.0:
            raise ValueError("alt_ratio must be in (0, 1].")
        alts = []
        for a, b in zip(*np.nonzero(is_max)):
            a, b = a + 1, b + 1
            if (a, b) != (i, j) and box[a, b] >= alt_ratio * box[i, j]:
                alts.append({'t_ms': float(t_grid[rows[a]]), 'v': float(vels[b]),
                             's': float(box[a, b])})
        out['alternates'] = sorted(alts, key=lambda d: -d['s'])
    return out


def reference_picks(gather, offsets, dt_ms, primaries, multiples=(), **kwargs):
    """
    Semblance peaks of every expected event.

    primaries : list of dicts with name, t0_ms, v_rms (case meta)
    multiples : list of dicts with name, t0_ms, v_nmo (case meta)
    kwargs    : passed to semblance_peak (time_win_ms, vel_win, vel_step and
                the robust-semblance options)

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
    multiple. When a reference point has 'alternates' (reference_picks
    with alt_ratio), a pick near any of them matches that event too.

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
    kinds = np.array([r['kind'] for r in reference])
    # Every acceptable point of every event: the peak and its alternates
    pts = [(k, r['t_ms'], r['v']) for k, r in enumerate(reference)]
    pts += [(k, a['t_ms'], a['v']) for k, r in enumerate(reference)
            for a in r.get('alternates', ())]
    owner = np.array([p[0] for p in pts], dtype=int)
    ref_t = np.array([p[1] for p in pts], dtype=float)
    ref_v = np.array([p[2] for p in pts], dtype=float)

    if times_ms.size and len(reference):
        d = np.abs(times_ms[:, None] - ref_t[None, :])
        ok = (d <= time_tol_ms) & (np.abs(vels[:, None] - ref_v[None, :])
                                   <= vel_tol * ref_v[None, :])
        d = np.where(ok, d, np.inf)
        # per event: closest acceptable point
        dt = np.full((times_ms.size, len(reference)), np.inf)
        for k in range(len(reference)):
            dt[:, k] = d[:, owner == k].min(axis=1)
        hit = np.isfinite(dt)
    else:
        hit = np.zeros((times_ms.size, len(reference)), dtype=bool)
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


def _robust_options(semblance_opts):
    """Only the robust options that change the semblance, with defaults filled."""
    opts = {'stretch_mute': None, 'energy_floor': 0.0, 'min_fold': 1}
    opts.update({k: semblance_opts[k] for k in ROBUST_OPTION_KEYS if k in semblance_opts})
    return opts


def get_reference(case, alt_ratio=None, **semblance_opts):
    """
    Reference picks of a case for a given semblance.

    alt_ratio : None (default: stored reference, single peak per event) or
                a fraction: recomputed with alternates (see reference_picks;
                a few seconds per case)

    The reference must come from the same semblance the algorithms see:
    - plain semblance (no robust option on): meta['reference_picks'];
    - robust semblance: meta['reference_picks_robust'] when it was stored
      with exactly these options (scripts/add_reference_picks.py), else
      it is recomputed here (a few seconds).

    case : dict from src.seismic.cases.load_case (needs meta['primaries'])
    """
    meta = case['meta']
    opts = _robust_options(semblance_opts)
    if alt_ratio is not None:
        return reference_picks(case['gather'], case['offsets'], case['dt_ms'],
                               meta['primaries'], meta.get('surface_multiples', []),
                               alt_ratio=alt_ratio,
                               **(opts if _is_robust(**opts) else {}))
    if not _is_robust(**opts):
        return meta['reference_picks']
    if meta.get('reference_picks_robust_options') == opts:
        return meta['reference_picks_robust']
    return reference_picks(case['gather'], case['offsets'], case['dt_ms'],
                           meta['primaries'], meta.get('surface_multiples', []),
                           **opts)