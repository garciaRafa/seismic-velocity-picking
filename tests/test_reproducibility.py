"""
Reproducibility and evaluation-budget tests.

These guarantee that a run is fully determined by its seed, that the
optimizers never touch the global np.random state, and that the
evaluation counter / budget work as documented.
"""

import numpy as np
import pytest

from src.optimizers.hill_climbing import HillClimbing
from src.seismic.models import layered_velocity_profile
from src.seismic.synthetic import generate_cdp_gather


def _hc(data, seed, **kw):
    params = dict(vel_min=1400.0, vel_max=3000.0, n_picks=3, max_iter=30,
                  step_time=5, step_vel=50.0, n_neighbors=8, restarts=2,
                  patience=10)
    params.update(kw)
    return HillClimbing(data['gather'], data['offsets'], data['dt_ms'],
                        seed=seed, **params)


def test_same_seed_gives_identical_run(layered_gather):
    a = _hc(layered_gather, seed=123).run()
    b = _hc(layered_gather, seed=123).run()
    assert a.best_picks == b.best_picks
    assert a.best_score == b.best_score
    assert a.restart_scores == b.restart_scores
    assert a.n_evals == b.n_evals


def test_different_seeds_explore_differently(layered_gather):
    a = _hc(layered_gather, seed=1).run()
    b = _hc(layered_gather, seed=2).run()
    assert a.best_picks != b.best_picks


def test_rerunning_the_same_object_is_reproducible(layered_gather):
    hc = _hc(layered_gather, seed=7)
    first = (list(hc.run().best_picks), hc.best_score, hc.n_evals)
    second = (list(hc.run().best_picks), hc.best_score, hc.n_evals)
    assert first == second


def test_optimizer_does_not_touch_global_random_state(layered_gather):
    np.random.seed(0)
    expected = np.random.rand()
    np.random.seed(0)
    _hc(layered_gather, seed=5).run()
    assert np.random.rand() == expected


def test_run_result_does_not_depend_on_global_random_state(layered_gather):
    np.random.seed(0)
    a = _hc(layered_gather, seed=9).run()
    np.random.seed(999)
    np.random.rand(1000)
    b = _hc(layered_gather, seed=9).run()
    assert a.best_picks == b.best_picks


def test_evaluation_counter_matches_objective_calls(layered_gather, monkeypatch):
    hc = _hc(layered_gather, seed=3)
    calls = {'n': 0}
    original = HillClimbing._evaluate

    def counting(self, picks):
        calls['n'] += 1
        return original(self, picks)

    monkeypatch.setattr(HillClimbing, '_evaluate', counting)
    hc.run()
    assert hc.n_evals == calls['n'] > 0


@pytest.mark.parametrize('budget', [1, 25, 100])
def test_evaluation_budget_is_respected(layered_gather, budget):
    hc = _hc(layered_gather, seed=4, max_evals=budget).run()
    assert hc.n_evals <= budget
    assert hc.best_picks is not None


def test_eval_history_is_increasing(layered_gather):
    hc = _hc(layered_gather, seed=8).run()
    evals  = [e for e, _ in hc.eval_history]
    scores = [s for _, s in hc.eval_history]
    assert evals == sorted(evals)
    assert all(b > a for a, b in zip(scores, scores[1:]))
    assert scores[-1] == hc.best_score


def test_synthetic_noise_is_reproducible():
    profile = layered_velocity_profile([1500.0, 2000.0], [600.0],
                                       total_depth_m=1500.0)
    offsets = np.linspace(100.0, 3000.0, 10, dtype=np.float32)
    g1, _, _ = generate_cdp_gather(profile, 1.25, offsets, t_max_ms=1000.0,
                                   noise_level=0.1, seed=11)
    g2, _, _ = generate_cdp_gather(profile, 1.25, offsets, t_max_ms=1000.0,
                                   noise_level=0.1, seed=11)
    g3, _, _ = generate_cdp_gather(profile, 1.25, offsets, t_max_ms=1000.0,
                                   noise_level=0.1, seed=12)
    assert np.array_equal(g1, g2)
    assert not np.array_equal(g1, g3)
