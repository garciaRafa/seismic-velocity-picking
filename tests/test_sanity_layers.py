"""
Sanity tests on a three-layer model.

If these fail, something is wrong in the modeling / NMO / semblance chain,
and no optimizer result on Marmousi2 can be trusted.
"""

import numpy as np

from src.seismic.semblance import compute_semblance, picking_objective


def test_gather_has_hyperbolic_moveout(layered_gather):
    # On the far-offset trace, the first reflection must appear at the
    # hyperbolic time, not at its zero-offset time (flat events would
    # indicate a moveout bug).
    g, dt  = layered_gather['gather'], layered_gather['dt_ms']
    x_far  = float(layered_gather['offsets'][-1])
    t0     = layered_gather['event_times_ms'][0]          # 800 ms, water
    t_far  = np.sqrt(t0 ** 2 + (x_far / 1500.0 * 1000.0) ** 2)

    def energy_around(trace, t_ms, half_ms=20.0):
        i0 = int((t_ms - half_ms) / dt)
        i1 = int((t_ms + half_ms) / dt) + 1
        return float(np.sum(trace[i0:i1] ** 2))

    far = g[-1]
    assert energy_around(far, t_far) > 10.0 * energy_around(far, t0)


def test_semblance_peaks_at_true_rms_velocity(layered_gather):
    g, off, dt = (layered_gather[k] for k in ('gather', 'offsets', 'dt_ms'))
    v_rms = layered_gather['v_rms']
    vels  = np.arange(1400.0, 3001.0, 20.0, dtype=np.float32)
    panel = compute_semblance(g, off, vels, dt_ms=dt)

    for t_ms in layered_gather['event_times_ms']:
        # Strongest coherence around the event (the wavelet may shift it
        # by a few samples)
        idx   = int(round(t_ms / dt))
        rows  = slice(idx - 5, idx + 6)
        iv, _ = np.unravel_index(np.argmax(panel[:, rows]), panel[:, rows].shape)
        v_true = float(v_rms[idx])
        assert abs(vels[iv] - v_true) <= 60.0, (t_ms, vels[iv], v_true)


def test_true_picks_score_higher_than_wrong_picks(layered_gather):
    g, off, dt = (layered_gather[k] for k in ('gather', 'offsets', 'dt_ms'))
    v_rms = layered_gather['v_rms']
    idx   = [int(round(t / dt)) for t in layered_gather['event_times_ms']]
    true_picks  = [(i, float(v_rms[i])) for i in idx]
    wrong_picks = [(i, 2900.0) for i in idx]
    assert (picking_objective(g, off, true_picks, dt)
            > picking_objective(g, off, wrong_picks, dt) + 0.3)
