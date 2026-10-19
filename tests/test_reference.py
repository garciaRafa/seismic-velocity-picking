"""Tests for src/analysis/reference.py (semblance reference picks)."""

import numpy as np
import pytest

from src.analysis.reference import match_reference, reference_picks, semblance_peak


# ---------------------------------------------------------------------- #
# semblance_peak / reference_picks                                        #
# ---------------------------------------------------------------------- #

def test_peak_on_hyperbolic_data_is_at_the_true_vrms(layered_gather):
    """On data built with perfect hyperbolas the blob is at the model V_rms."""
    d = layered_gather
    for t0 in d['event_times_ms']:
        v0 = float(np.interp(t0 + 10.0, d['t_grid'], d['v_rms']))   # just below the interface
        p = semblance_peak(d['gather'], d['offsets'], d['dt_ms'], t0, v0)
        assert abs(p['t_ms'] - t0) <= 30.0
        assert abs(p['v'] - v0) <= 0.02 * v0
        assert not p['on_edge']


def test_peak_starting_from_a_wrong_guess_moves_to_the_blob(layered_gather):
    """A guess 5% off in velocity and 20 ms off in time still lands on the blob."""
    d = layered_gather
    t0 = d['event_times_ms'][1]
    v0 = float(np.interp(t0 + 10.0, d['t_grid'], d['v_rms']))
    p = semblance_peak(d['gather'], d['offsets'], d['dt_ms'], t0 - 20.0, 1.05 * v0)
    assert abs(p['v'] - v0) <= 0.02 * v0


def _fake_panel(monkeypatch, panel_tv):
    """Make compute_semblance return a given (time, velocity) panel."""
    import src.analysis.reference as ref
    monkeypatch.setattr(ref, 'compute_semblance',
                        lambda g, h, v, dt, t0=0.0: panel_tv.T[:len(v)])


def test_peak_ignores_the_flank_of_a_stronger_neighbor(monkeypatch):
    """The box max sits on the box edge (flank of a stronger event below);
    the interior local maximum is kept instead."""
    dt, n_t = 2.0, 200
    t = dt * np.arange(n_t)
    v = np.arange(1840.0, 2160.0 + 2.5, 5.0)            # what semblance_peak builds
    T, V = np.meshgrid(t, v, indexing='ij')
    weak = 0.4 * np.exp(-((T - 200) / 8) ** 2 - ((V - 2000) / 30) ** 2)
    strong = 0.9 * np.exp(-((T - 250) / 8) ** 2 - ((V - 2050) / 30) ** 2)
    _fake_panel(monkeypatch, weak + strong)

    gather = np.zeros((3, n_t))
    p = semblance_peak(gather, np.array([1.0, 2.0, 3.0]), dt, 200.0, 2000.0,
                       time_win_ms=40.0, vel_win=0.08, vel_step=5.0)
    assert p['t_ms'] == pytest.approx(200.0, abs=2.0)
    assert p['v'] == pytest.approx(2000.0, abs=5.0)
    assert not p['on_edge']


def test_peak_flags_a_box_without_interior_maximum(monkeypatch):
    dt, n_t = 2.0, 200
    t = dt * np.arange(n_t)
    v = np.arange(1840.0, 2160.0 + 2.5, 5.0)
    T, V = np.meshgrid(t, v, indexing='ij')
    _fake_panel(monkeypatch, T / t[-1] + 0 * V)         # grows with time: max on the edge
    p = semblance_peak(np.zeros((3, n_t)), np.array([1.0, 2.0, 3.0]), dt, 200.0, 2000.0)
    assert p['on_edge'] and p['t_ms'] == pytest.approx(240.0)


def test_reference_picks_labels_kinds(layered_gather):
    d = layered_gather
    prim = [{'name': 'a', 't0_ms': d['event_times_ms'][0], 'v_rms': 1500.0}]
    mult = [{'name': 'm', 't0_ms': 2 * d['event_times_ms'][0], 'v_nmo': 1500.0}]
    refs = reference_picks(d['gather'], d['offsets'], d['dt_ms'], prim, mult)
    assert [r['kind'] for r in refs] == ['primary', 'multiple']
    assert {'name', 't0_ms', 'v_model', 't_ms', 'v', 's', 'on_edge'} <= set(refs[0])


# ---------------------------------------------------------------------- #
# match_reference                                                         #
# ---------------------------------------------------------------------- #

REF = [
    {'name': 'P1', 'kind': 'primary',  't_ms': 500.0,  'v': 1520.0},
    {'name': 'P2', 'kind': 'primary',  't_ms': 900.0,  'v': 1650.0},
    {'name': 'M1', 'kind': 'multiple', 't_ms': 1000.0, 'v': 1505.0},
]


def test_match_reference_counts():
    times = [505.0, 990.0, 1300.0]          # on P1, on M1, on nothing
    vels  = [1530.0, 1500.0, 2000.0]
    m = match_reference(times, vels, REF)
    assert m['n_on_primary'] == 1 and m['n_on_multiple'] == 1
    assert m['kind'] == ['primary', 'multiple', None]
    assert m['label'] == ['P1', 'M1', None]
    assert m['found'] == [True, False]
    assert m['precision'] == pytest.approx(1 / 3)
    assert m['recall'] == pytest.approx(1 / 2)
    assert m['f1'] == pytest.approx(2 * (1 / 3) * (1 / 2) / (1 / 3 + 1 / 2))


def test_match_reference_needs_time_and_velocity():
    # right time, velocity 5% off -> no match; right velocity, 60 ms off -> no match
    m = match_reference([900.0, 960.0], [1650.0 * 1.05, 1650.0], REF)
    assert m['n_on_primary'] == 0 and m['recall'] == 0.0


def test_match_reference_empty():
    m = match_reference([], [], REF)
    assert m['precision'] == 0.0 and m['recall'] == 0.0 and m['found'] == [False, False]