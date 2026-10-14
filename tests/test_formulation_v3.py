"""Tests for formulation v3: threshold objective with a free number of picks."""

import numpy as np
import pytest

from src.analysis.metrics import precision_recall
from src.optimizers.hill_climbing import HillClimbing
from src.optimizers.random_search import RandomSearch
from src.optimizers.tabu_search import TabuSearch
from src.seismic.semblance import picking_objective_threshold, semblance_at_pick

TAU = 0.1
V3 = dict(vel_min=1400.0, vel_max=3000.0, min_sep_ms=100.0,
          objective='threshold', tau=TAU)


def _rs(d, seed=0, budget=200, **kw):
    return RandomSearch(d['gather'], d['offsets'], d['dt_ms'], seed=seed,
                        max_evals=budget, **{**V3, **kw})


def _hc(d, seed=0, budget=600, **kw):
    return HillClimbing(d['gather'], d['offsets'], d['dt_ms'], seed=seed,
                        max_iter=10**6, restarts=1000, n_neighbors=10,
                        patience=20, max_evals=budget, **{**V3, **kw})


def _ts(d, seed=0, budget=600, **kw):
    return TabuSearch(d['gather'], d['offsets'], d['dt_ms'], seed=seed,
                      max_iter=10**6, n_neighbors=10, patience=10**6,
                      max_evals=budget, **{**V3, **kw})


# ---------------------------------------------------------------------- #
# Objective                                                               #
# ---------------------------------------------------------------------- #

def test_threshold_objective_is_sum_of_semblance_minus_tau(layered_gather):
    d = layered_gather
    picks = [(400, 1500.0), (750, 1750.0), (1100, 1900.0)]
    expected = sum(semblance_at_pick(d['gather'], d['offsets'], v, t, d['dt_ms']) - TAU
                   for t, v in picks)
    got = picking_objective_threshold(d['gather'], d['offsets'], picks,
                                      d['dt_ms'], TAU)
    assert got == pytest.approx(expected)


def test_threshold_objective_of_no_picks_is_zero(layered_gather):
    d = layered_gather
    assert picking_objective_threshold(d['gather'], d['offsets'], [],
                                       d['dt_ms'], TAU) == 0.0


def test_optimizer_evaluates_with_the_threshold_objective(layered_gather):
    d = layered_gather
    rs = _rs(d)
    picks = [(400, 1500.0), (750, 1750.0)]
    assert rs._evaluate(picks) == pytest.approx(picking_objective_threshold(
        d['gather'], d['offsets'], picks, d['dt_ms'], TAU))


# ---------------------------------------------------------------------- #
# Constructor                                                             #
# ---------------------------------------------------------------------- #

def test_threshold_needs_tau(layered_gather):
    with pytest.raises(ValueError):
        _rs(layered_gather, tau=None)


def test_unknown_objective_is_rejected(layered_gather):
    with pytest.raises(ValueError):
        _rs(layered_gather, objective='median')


def test_move_probabilities_must_add_up_to_at_most_one(layered_gather):
    with pytest.raises(ValueError):
        _hc(layered_gather, p_insert=0.7, p_remove=0.5)


def test_k_max_defaults_to_and_is_clipped_by_what_fits(layered_gather):
    d = layered_gather
    rs = _rs(d)            # whole record: samples 0..1500, sep 50 samples
    assert rs.k_max == (rs.t_max - rs.t_min) // rs.min_sep + 1 == 31
    assert _rs(d, k_max=10).k_max == 10
    assert _rs(d, k_max=500).k_max == 31


def test_k_min_that_does_not_fit_is_rejected(layered_gather):
    with pytest.raises(ValueError):
        _rs(layered_gather, k_min=32)


# ---------------------------------------------------------------------- #
# Random solutions and validity                                           #
# ---------------------------------------------------------------------- #

def test_random_solutions_have_a_random_valid_number_of_picks(layered_gather):
    rs = _rs(layered_gather, k_min=2, k_max=12)
    rs.rng = np.random.default_rng(0)
    sizes = []
    for _ in range(300):
        picks = rs._random_picks()
        assert rs._is_valid(picks)
        sizes.append(len(picks))
    assert min(sizes) == 2 and max(sizes) == 12


def test_validity_checks_the_number_of_picks(layered_gather):
    rs = _rs(layered_gather, k_min=2, k_max=3)
    assert not rs._is_valid([(400, 1500.0)])
    assert rs._is_valid([(400, 1500.0), (600, 1600.0)])
    assert not rs._is_valid([(400, 1500.0), (600, 1600.0),
                             (800, 1700.0), (1000, 1800.0)])


# ---------------------------------------------------------------------- #
# Insert / remove moves                                                   #
# ---------------------------------------------------------------------- #

def test_insert_and_remove_are_valid_by_construction(layered_gather):
    hc = _hc(layered_gather, k_min=1, k_max=30)
    hc.rng = np.random.default_rng(1)
    for _ in range(300):
        current = hc._random_picks()
        ins = hc._insert_neighbor(current)
        if ins is not None:
            assert hc._is_valid(ins[0]) and len(ins[0]) == len(current) + 1
            assert ins[1] in [t for t, _ in ins[0]]
        rem = hc._remove_neighbor(current)
        if rem is not None:
            assert hc._is_valid(rem[0]) and len(rem[0]) == len(current) - 1
            assert rem[1] in [t for t, _ in current]


def test_insert_stops_at_k_max_and_remove_at_k_min(layered_gather):
    hc = _hc(layered_gather, k_min=2, k_max=3)
    hc.rng = np.random.default_rng(2)
    three = [(400, 1500.0), (700, 1600.0), (1000, 1700.0)]
    two   = three[:2]
    assert hc._insert_neighbor(three) is None
    assert hc._remove_neighbor(two) is None


def test_insert_returns_none_when_no_gap_fits(layered_gather):
    hc = _hc(layered_gather)
    hc.rng = np.random.default_rng(3)
    # Picks every min_sep samples over the whole window: no room left
    full = [(t, 1500.0 + t) for t in range(hc.t_min, hc.t_max + 1, hc.min_sep)]
    assert hc._insert_neighbor(full) is None


def test_structural_move_draws_nothing_with_the_mean_objective(layered_gather):
    d = layered_gather
    hc = HillClimbing(d['gather'], d['offsets'], d['dt_ms'], seed=0,
                      vel_min=1400.0, vel_max=3000.0, n_picks=3)
    hc.rng = np.random.default_rng(4)
    before = hc.rng.bit_generator.state
    assert hc._structural_move([(400, 1500.0)]) is None
    assert hc.rng.bit_generator.state == before


# ---------------------------------------------------------------------- #
# Full runs                                                               #
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize('make', [_rs, _hc, _ts])
def test_v3_runs_use_the_budget_and_return_valid_solutions(layered_gather, make):
    opt = make(layered_gather, seed=5, budget=300).run()
    assert opt.n_evals == 300
    assert opt._is_valid(opt.best_picks)
    assert opt.k_min <= len(opt.best_picks) <= opt.k_max


@pytest.mark.parametrize('make', [_rs, _hc, _ts])
def test_v3_runs_are_reproducible(layered_gather, make):
    a = make(layered_gather, seed=6, budget=250).run()
    b = make(layered_gather, seed=6, budget=250).run()
    assert a.best_picks == b.best_picks and a.best_score == b.best_score


def test_local_search_changes_the_number_of_picks(layered_gather):
    ts = _ts(layered_gather, seed=7, budget=800, p_jump=0.1).run()
    assert ts.move_stats['insert'] + ts.move_stats['remove'] > 0


def test_tabu_attributes_are_time_based_in_v3(layered_gather):
    ts = _ts(layered_gather, seed=8)
    ts._reset_run_state()
    ts._initialize()
    for _ in range(20):
        ts._iterate()
    assert ts._tabu
    assert all(kind in ('place', 'remove') and isinstance(t, int)
               for kind, t in ts._tabu)


def test_v3_finds_the_two_reflections_of_the_layered_model(layered_gather):
    """
    With a sensible threshold the optimum has exactly the 2 real events,
    each picked at that event's velocity.

    The pick time can sit a few tens of ms off the event time: inside the
    30 ms semblance gate plus the wavelet, a slightly later and slower
    trajectory still sees the same event (often with a higher semblance).
    Right below the water bottom V_rms rises quickly, so such a pick can
    miss the 2% tolerance of precision_recall (which compares against
    V_rms at the PICK time) while still carrying the event's velocity —
    which is what this test checks.
    """
    d = layered_gather
    ts = _ts(d, seed=9, budget=4000, tau=0.3, p_jump=0.1).run()
    picks = sorted((t * d['dt_ms'], v) for t, v in ts.best_picks)
    assert len(picks) == 2
    for (t_ms, v), t_event in zip(picks, d['event_times_ms']):
        v_event = np.interp(t_event, d['t_grid'], d['v_rms'])
        assert abs(t_ms - t_event) <= 40.0
        assert abs(v - v_event) <= 0.02 * v_event


# ---------------------------------------------------------------------- #
# Detection metrics                                                       #
# ---------------------------------------------------------------------- #

def test_precision_recall_counts_matches():
    t_grid, v_true = np.array([0.0, 1000.0]), np.array([1500.0, 2500.0])
    events = [100.0, 500.0, 900.0]
    # pick 1: on event 1, correct velocity -> match
    # pick 2: on event 2, wrong velocity   -> no match
    # pick 3: correct velocity but far from any event -> no match
    picks_t = [110.0, 500.0, 700.0]
    picks_v = [1610.0, 2500.0, 2200.0]
    r = precision_recall(picks_t, picks_v, t_grid, v_true, events)
    assert r['matched'] == [True, False, False]
    assert r['found'] == [True, False, False]
    assert r['n_vel_correct'] == 2
    assert r['precision'] == pytest.approx(1 / 3)
    assert r['recall'] == pytest.approx(1 / 3)
    assert r['f1'] == pytest.approx(1 / 3)


def test_precision_recall_with_no_picks():
    r = precision_recall([], [], [0.0, 1000.0], [1500.0, 2500.0], [100.0])
    assert r['precision'] == 0.0 and r['recall'] == 0.0 and r['f1'] == 0.0