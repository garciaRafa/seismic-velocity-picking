"""Tests for src/seismic/model2d.py and src/seismic/fd_modeling.py."""

import numpy as np
import pytest

from src.seismic.fd_modeling import (cdp_geometry, direct_wave_mute,
                                     receiver_line, resample_traces, ricker,
                                     time_gain)
from src.seismic.model2d import (build_layered_model, column_event_times,
                                 column_vrms, interface_depths,
                                 surface_multiples)

MODEL = {
    'width_m': 2000.0, 'depth_m': 1500.0, 'dx_m': 5.0,
    'velocities': [1500.0, 2000.0, 2500.0],
    'interfaces': [
        {'name': 'wb', 'z0_m': 400.0},
        {'name': 'deep', 'z0_m': 900.0, 'dip_deg': 5.0, 'x_ref_m': 1000.0,
         'fault': {'x_m': 1500.0, 'throw_m': 50.0}},
    ],
}


# ---------------------------------------------------------------------- #
# Model                                                                   #
# ---------------------------------------------------------------------- #

def test_interface_with_dip_and_fault():
    itf = MODEL['interfaces'][1]
    z = interface_depths(itf, [1000.0, 1200.0, 1499.0, 1500.0])
    slope = np.tan(np.deg2rad(5.0))
    assert z[0] == pytest.approx(900.0)
    assert z[1] == pytest.approx(900.0 + 200.0 * slope)
    assert z[3] - z[2] == pytest.approx(50.0 + 1.0 * slope)   # fault throw


def test_model_assigns_one_velocity_per_layer():
    m = build_layered_model(MODEL)
    assert m['vp'].shape == (401, 301)
    ix = int(1000 / 5)
    assert m['vp'][ix, int(300 / 5)] == 1500.0     # water
    assert m['vp'][ix, int(600 / 5)] == 2000.0     # middle layer
    assert m['vp'][ix, int(1200 / 5)] == 2500.0    # below the deep interface


def test_model_rejects_crossing_interfaces():
    bad = {**MODEL, 'interfaces': [{'z0_m': 800.0}, {'z0_m': 500.0}]}
    with pytest.raises(ValueError):
        build_layered_model(bad)


def test_model_needs_one_velocity_per_layer():
    with pytest.raises(ValueError):
        build_layered_model({**MODEL, 'velocities': [1500.0, 2000.0]})


def test_column_event_times_and_vrms():
    m = build_layered_model(MODEL)
    ev = column_event_times(m, 1000.0)
    t_wb = 2000.0 * 400.0 / 1500.0
    t_deep = t_wb + 2000.0 * 500.0 / 2000.0
    assert ev[0]['t0_ms'] == pytest.approx(t_wb, abs=0.5)
    assert ev[0]['v_rms'] == pytest.approx(1500.0, abs=0.1)
    assert ev[1]['t0_ms'] == pytest.approx(t_deep, abs=0.5)
    v_expected = np.sqrt((1500.0 ** 2 * t_wb + 2000.0 ** 2 * (t_deep - t_wb)) / t_deep)
    assert ev[1]['v_rms'] == pytest.approx(v_expected, rel=1e-3)

    t, v = column_vrms(m, 1000.0, dt_ms=2.0, t_max_ms=1500.0)
    assert np.allclose(v[t < t_wb - 5], 1500.0, atol=0.5)


def test_surface_multiples():
    prim = [{'name': 'wb', 't0_ms': 500.0, 'v_rms': 1500.0},
            {'name': 'A', 't0_ms': 900.0, 'v_rms': 1800.0}]
    mult = {m['name']: m for m in surface_multiples(prim, n_water=3)}
    assert mult['water-bottom multiple x2']['t0_ms'] == 1000.0
    assert mult['water-bottom multiple x3']['t0_ms'] == 1500.0
    assert mult['water-bottom multiple x2']['v_nmo'] == 1500.0
    peg = mult['peg-leg of A']
    assert peg['t0_ms'] == 1400.0
    assert peg['v_nmo'] == pytest.approx(np.sqrt((1800 ** 2 * 900 + 1500 ** 2 * 500) / 1400))


# ---------------------------------------------------------------------- #
# Geometry and processing                                                 #
# ---------------------------------------------------------------------- #

def test_cdp_geometry_puts_every_pair_on_the_cdp():
    offsets = np.array([100.0, 500.0, 3000.0])
    s, r = cdp_geometry(2500.0, offsets)
    assert np.allclose((s + r) / 2, 2500.0)
    assert np.allclose(r - s, offsets)


def test_receiver_line_contains_the_extra_receivers():
    extra = [2550.0, 2603.7, 4000.0]
    allx, idx = receiver_line(5000.0, 10.0, extra)
    assert np.allclose(allx[idx], extra)
    assert np.all(np.diff(allx) > 0)
    assert allx[0] == 0.0 and allx[-1] == 5000.0


def test_direct_wave_mute():
    t = np.arange(0.0, 1000.0, 2.0)
    g = np.ones((2, t.size), dtype=np.float32)
    out = direct_wave_mute(g, [300.0, 600.0], t, velocity=1500.0, pad_ms=50.0,
                           taper_ms=20.0)
    for row, h in zip(out, (300.0, 600.0)):
        cut = 1000.0 * h / 1500.0 + 50.0
        assert np.all(row[t <= cut] == 0.0)
        assert np.all(row[t >= cut + 20.0] == 1.0)


def test_time_gain():
    t = np.array([0.0, 500.0, 1000.0])
    out = time_gain(np.ones((1, 3)), t, power=1.0)
    assert np.allclose(out, [[0.0, 0.5, 1.0]])


def test_resample_traces_is_linear_and_zero_outside():
    t_in = np.linspace(-10.0, 100.0, 111)
    trace = 3.0 * t_in
    out = resample_traces(trace, t_in, np.array([0.0, 50.0, 200.0]))
    assert np.allclose(out[0, :2], [0.0, 150.0])
    assert out[0, 2] == 0.0


def test_ricker_peaks_at_its_delay():
    t = np.linspace(0, 0.2, 2001)
    w = ricker(t, 30.0, delay_s=0.05)
    assert t[np.argmax(w)] == pytest.approx(0.05, abs=1e-4)


# ---------------------------------------------------------------------- #
# Finite-difference modeling (only when Devito is installed)              #
# ---------------------------------------------------------------------- #

def test_direct_wave_arrives_at_distance_over_velocity():
    pytest.importorskip('devito')
    from src.seismic.fd_modeling import AcousticModeler

    vp = np.full((241, 121), 2000.0, dtype=np.float32)   # 1.2 km x 0.6 km
    rec_x = np.array([600.0, 900.0])
    t = np.arange(0.0, 300.1, 0.5)
    m = AcousticModeler(vp, 5.0, rec_x, 300.0, f0_hz=30.0, free_surface=False,
                        rec_depth_m=300.0, src_depth_m=300.0, absorbing_cells=40)
    record = m.shot(300.0, t)
    for trace, dist in zip(record, (300.0, 600.0)):
        # The 2D Green's function lags slightly behind the geometric arrival:
        # check the peak lies within a few ms of distance / velocity
        assert abs(t[np.argmax(np.abs(trace))] - 1000.0 * dist / 2000.0) < 12.0