"""Tests for the robust-semblance options (stretch mute, energy floor, min fold)."""

import numpy as np
import pytest

from src.analysis.reference import get_reference
from src.optimizers.random_search import RandomSearch
from src.seismic.semblance import (ROBUST_SEMBLANCE, compute_semblance, gather_energy,
                                   picking_objective, picking_objective_threshold,
                                   semblance_at_pick)

DT = 2.0
OFFSETS = np.linspace(100.0, 3000.0, 30)
N_T = 700


def _ricker_event(t0_ms, v, amp=1.0, offsets=OFFSETS, n_t=N_T, dt=DT, f0=30.0):
    t = dt * np.arange(n_t)
    th = np.sqrt(t0_ms ** 2 + (1000.0 * offsets / v) ** 2)
    a = (np.pi * f0 * (t[None, :] - th[:, None]) / 1000.0) ** 2
    return amp * (1 - 2 * a) * np.exp(-a)


@pytest.fixture(scope='module')
def gather():
    rng = np.random.default_rng(1)
    g = _ricker_event(600.0, 2000.0) + _ricker_event(1000.0, 2400.0, 0.8)
    return (g + 0.01 * rng.standard_normal(g.shape)).astype(np.float32)


VELS = np.arange(1500.0, 3001.0, 50.0, dtype=np.float32)


# ---------------------------------------------------------------------- #
# Defaults: plain semblance unchanged                                     #
# ---------------------------------------------------------------------- #

def test_defaults_are_the_plain_semblance(gather):
    plain = compute_semblance(gather, OFFSETS, VELS, DT)
    same = compute_semblance(gather, OFFSETS, VELS, DT, stretch_mute=None,
                             energy_floor=0.0, min_fold=1)
    assert np.array_equal(plain, same)
    picks = [(300, 2000.0), (500, 2400.0)]
    assert picking_objective(gather, OFFSETS, picks, DT) == \
        picking_objective(gather, OFFSETS, picks, DT, stretch_mute=None)
    assert picking_objective_threshold(gather, OFFSETS, picks, DT, 0.2) == \
        picking_objective_threshold(gather, OFFSETS, picks, DT, 0.2, energy_floor=0.0)


# ---------------------------------------------------------------------- #
# Panel and point evaluation agree (the optimizer sees the panel)         #
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize('opts', [
    {'stretch_mute': 0.5},
    {'energy_floor': 1e-2},
    {'min_fold': 10},
    ROBUST_SEMBLANCE,
])
def test_panel_matches_semblance_at_pick(gather, opts):
    panel = compute_semblance(gather, OFFSETS, VELS, DT, **opts)
    rng = np.random.default_rng(0)
    for _ in range(200):
        iv, it = rng.integers(len(VELS)), rng.integers(N_T)
        s = semblance_at_pick(gather, OFFSETS, float(VELS[iv]), int(it), DT, **opts)
        assert s == pytest.approx(float(panel[iv, it]), abs=1e-6)


# ---------------------------------------------------------------------- #
# Each option does what it says                                           #
# ---------------------------------------------------------------------- #

def test_stretch_mute_ignores_stretched_traces(gather):
    """Traces whose stretch exceeds the limit over the whole gate do not
    change the value, whatever they contain."""
    t_sample, v = 300, 2000.0                        # t0 = 600 ms
    t0 = DT * t_sample
    stretch = (np.sqrt(t0 ** 2 + (1000 * OFFSETS / v) ** 2) - t0) / t0
    far = stretch > 0.5 * 1.3                        # beyond the limit with margin (gate)
    assert far.any() and (~far).sum() >= 4
    garbage = gather.copy()
    garbage[far] = np.random.default_rng(5).standard_normal((far.sum(), N_T))
    a = semblance_at_pick(gather, OFFSETS, v, t_sample, DT, stretch_mute=0.5)
    b = semblance_at_pick(garbage, OFFSETS, v, t_sample, DT, stretch_mute=0.5)
    assert a == pytest.approx(b, abs=1e-6)
    # without the mute the garbage does change it
    assert semblance_at_pick(gather, OFFSETS, v, t_sample, DT) != pytest.approx(
        semblance_at_pick(garbage, OFFSETS, v, t_sample, DT), abs=1e-3)


def test_energy_floor_silences_near_zero_coherent_noise():
    """A perfectly coherent but tiny event (numerical noise) gets a high
    plain semblance; the floor sends it to ~0 and keeps the real event."""
    near = np.linspace(100.0, 1000.0, 20)            # the two events never cross
    g = (_ricker_event(300.0, 1500.0, amp=1e-5, offsets=near)
         + _ricker_event(900.0, 2000.0, offsets=near)).astype(np.float32)
    tiny_plain = semblance_at_pick(g, near, 1500.0, 150, DT)
    tiny_floor = semblance_at_pick(g, near, 1500.0, 150, DT, energy_floor=1e-2)
    real_plain = semblance_at_pick(g, near, 2000.0, 450, DT)
    real_floor = semblance_at_pick(g, near, 2000.0, 450, DT, energy_floor=1e-2)
    assert tiny_plain > 0.5 and tiny_floor < 0.01
    assert real_floor > 0.9 * real_plain


def test_min_fold_zeroes_cells_with_few_live_traces(gather):
    """At a shallow time the stretch mute leaves few traces: below min_fold, S = 0."""
    t_sample, v = 40, 1500.0                         # t0 = 80 ms: only near traces live
    t0 = DT * t_sample
    live = ((np.sqrt(t0 ** 2 + (1000 * OFFSETS / v) ** 2) - t0) / t0 <= 0.5).sum()
    assert 0 < live < 10
    assert semblance_at_pick(gather, OFFSETS, v, t_sample, DT, stretch_mute=0.5,
                             min_fold=live + 1) == 0.0


def test_gather_energy():
    g = np.full((3, 5), 2.0)
    assert gather_energy(g, window=15) == pytest.approx(4.0 * 15)


# ---------------------------------------------------------------------- #
# Optimizer and reference                                                 #
# ---------------------------------------------------------------------- #

def test_optimizer_objective_uses_the_options(gather):
    opt = RandomSearch(gather, OFFSETS, DT, vel_min=1500, vel_max=3000, min_sep_ms=100,
                       objective='threshold', tau=0.2, seed=1, max_evals=10,
                       **ROBUST_SEMBLANCE)
    picks = [(300, 2000.0), (500, 2400.0)]
    assert opt._evaluate(picks) == pytest.approx(
        picking_objective_threshold(gather, OFFSETS, picks, DT, 0.2, **ROBUST_SEMBLANCE))


def test_optimizer_rejects_bad_options(gather):
    with pytest.raises(ValueError):
        RandomSearch(gather, OFFSETS, DT, stretch_mute=0.0)
    with pytest.raises(ValueError):
        RandomSearch(gather, OFFSETS, DT, min_fold=0)


def test_get_reference_matches_the_semblance(gather):
    prim = [{'name': 'a', 't0_ms': 600.0, 'v_rms': 2000.0}]
    case = {'gather': gather, 'offsets': OFFSETS, 'dt_ms': DT,
            'meta': {'primaries': prim, 'reference_picks': ['plain'],
                     'reference_picks_robust': ['stored'],
                     'reference_picks_robust_options': dict(ROBUST_SEMBLANCE)}}
    assert get_reference(case) == ['plain']
    assert get_reference(case, **ROBUST_SEMBLANCE) == ['stored']
    other = get_reference(case, stretch_mute=0.3)          # not stored: recomputed
    assert other[0]['name'] == 'a' and abs(other[0]['t_ms'] - 600.0) <= 30