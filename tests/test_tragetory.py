"""Tests for the search-trajectory recording and its figures."""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

from src.analysis import search_viz
from src.optimizers.hill_climbing import HillClimbing
from src.optimizers.random_search import RandomSearch
from src.optimizers.tabu_search import TabuSearch


def _make(cls, gather, record, **extra):
    d = gather
    common = dict(vel_min=1400.0, vel_max=3000.0, min_sep_ms=100.0,
                  objective='threshold', tau=0.2, seed=3, max_evals=1500,
                  record_trajectory=record, **extra)
    if cls is RandomSearch:
        return cls(d['gather'], d['offsets'], d['dt_ms'], **common)
    local = dict(step_time=5, step_vel=50.0, n_neighbors=10, max_iter=10**5)
    if cls is HillClimbing:
        return cls(d['gather'], d['offsets'], d['dt_ms'], **common, **local,
                   restarts=10**4, patience=15)
    return cls(d['gather'], d['offsets'], d['dt_ms'], **common, **local,
               tabu_tenure=10, patience=10**5, p_jump=0.1)


@pytest.mark.parametrize('cls', [HillClimbing, TabuSearch, RandomSearch])
def test_recording_does_not_change_the_run(layered_gather, cls):
    a = _make(cls, layered_gather, False).run()
    b = _make(cls, layered_gather, True).run()
    assert a.best_score == b.best_score and a.best_picks == b.best_picks
    assert a.n_evals == b.n_evals
    assert a.trajectory == [] and len(a.proposals) == 0


def test_hill_climbing_trajectory_marks_restarts(layered_gather):
    opt = _make(HillClimbing, layered_gather, True).run()
    events = [r['event'] for r in opt.trajectory]
    assert events[0] == 'start' and 'restart' in events
    assert all('restart' in r for r in opt.trajectory)
    # best score never decreases, n_evals never decreases
    best = [r['best_score'] for r in opt.trajectory]
    assert all(b2 >= b1 for b1, b2 in zip(best, best[1:]))
    ev = [r['n_evals'] for r in opt.trajectory]
    assert all(e2 >= e1 for e1, e2 in zip(ev, ev[1:]))


def test_tabu_trajectory_has_moves_and_tabu_size(layered_gather):
    opt = _make(TabuSearch, layered_gather, True).run()
    rec = opt.trajectory[-1]
    assert {'move', 'tabu_size'} <= set(rec)
    assert any(r['move'] is not None for r in opt.trajectory)
    # Tabu Search accepts worse moves: the current score sometimes drops
    cur = [r['current_score'] for r in opt.trajectory]
    assert any(c2 < c1 for c1, c2 in zip(cur, cur[1:]))


def test_record_every_thins_the_trajectory(layered_gather):
    full = _make(TabuSearch, layered_gather, True).run()
    thin = _make(TabuSearch, layered_gather, True, record_every=5).run()
    assert len(thin.trajectory) < len(full.trajectory)
    assert thin.best_score == full.best_score


def test_proposals_are_inside_the_bounds(layered_gather):
    opt = _make(HillClimbing, layered_gather, True).run()
    p = opt.proposals
    assert p.shape[1] == 3 and len(p) > 0
    assert p[:, 1].min() >= 1400.0 and p[:, 1].max() <= 3000.0
    assert np.all(np.diff(p[:, 2]) >= 0)          # in evaluation order


def test_figures_run(layered_gather, tmp_path):
    import matplotlib.pyplot as plt
    opt = _make(TabuSearch, layered_gather, True).run()
    panel = np.random.default_rng(0).random((20, layered_gather['gather'].shape[1]))
    extent = [1400.0, 3000.0, layered_gather['t_grid'][-1], 0.0]
    dt = layered_gather['dt_ms']
    search_viz.plot_score_trace(opt)
    search_viz.plot_n_picks(opt)
    search_viz.plot_snapshots(opt, panel, extent, dt, n=3)
    search_viz.plot_proposals(opt, panel, extent, dt)
    path = search_viz.animate_search(opt, panel, extent, dt, str(tmp_path / 'a.gif'),
                                     max_frames=5, fps=5)
    assert (tmp_path / 'a.gif').stat().st_size > 0 and path.endswith('a.gif')
    plt.close('all')


def test_figures_need_a_recorded_run(layered_gather):
    opt = _make(TabuSearch, layered_gather, False).run()
    with pytest.raises(ValueError):
        search_viz.plot_score_trace(opt)