"""
src/optimizers/hill_climbing.py
--------------------------------
Hill Climbing for seismic velocity picking.

Strategy:
    Start from a random valid solution (set of velocity picks).
    At each iteration, generate neighbors by perturbing one pick
    at a time (small changes in time and/or velocity).
    Move to the best neighbor only if it improves the semblance score.
    Stop a restart when no improvement is found for `patience`
    consecutive iterations (local optimum) or max_iter is reached.

This is the simplest local search heuristic and serves as
a baseline for comparison with more advanced metaheuristics.
"""

import numpy as np
from src.optimizers.base import BaseOptimizer


class HillClimbing(BaseOptimizer):
    """
    Hill Climbing optimizer for seismic velocity picking.

    Parameters
    ----------
    traces      : np.ndarray, shape (n_traces, n_samples)
    offsets     : np.ndarray, shape (n_traces,) in meters
    dt_ms       : float — sampling interval in ms
    vel_min     : float — minimum velocity in m/s
    vel_max     : float — maximum velocity in m/s
    n_picks     : int   — number of velocity picks to find
    max_iter    : int   — maximum number of iterations per restart
    seed        : int   — random seed
    step_time   : int   — max perturbation in time samples
    step_vel    : float — max perturbation in velocity (m/s)
    n_neighbors : int   — neighbors evaluated per iteration
    restarts    : int   — number of random restarts
    patience    : int   — iterations without improvement before a
                          restart is considered stuck in a local optimum
    max_evals   : int or None — budget of objective evaluations for the
                          whole run (all restarts). None = no budget.

    Reproducibility
    ---------------
    Each restart gets its own random generator, derived from `seed` with
    np.random.SeedSequence(seed).spawn(restarts). The streams of different
    restarts (and of runs with different seeds) are statistically
    independent, unlike consecutive seeds such as seed + restart.
    """

    def __init__(self, traces, offsets, dt_ms,
                 vel_min=1400.0, vel_max=3000.0,
                 n_picks=10, max_iter=200, seed=42,
                 step_time=5, step_vel=50.0,
                 n_neighbors=20, restarts=5, patience=20,
                 max_evals=None):

        super().__init__(traces, offsets, dt_ms,
                         vel_min, vel_max,
                         n_picks, max_iter, seed, max_evals)

        self.step_time   = step_time
        self.step_vel    = step_vel
        self.n_neighbors = n_neighbors
        self.restarts    = restarts
        self.patience    = patience

        # Internal state
        self._current_picks = None
        self._current_score = -np.inf

        # Per-restart results — filled after run()
        self.restart_histories = []   # current score per iteration, per restart
        self.restart_scores    = []   # final score of each restart

    # ------------------------------------------------------------------ #
    # Abstract method implementations                                      #
    # ------------------------------------------------------------------ #

    def _initialize(self):
        """Start from a random valid solution."""
        self._current_picks = self._random_picks()
        self._current_score = self._evaluate(self._current_picks)
        self._update_best(self._current_picks, self._current_score)

    def _iterate(self):
        """
        One iteration: evaluate neighbors and move to best improvement.
        """
        improved = False
        best_neighbor       = None
        best_neighbor_score = self._current_score

        for _ in range(self.n_neighbors):
            if self.budget_exhausted:
                break

            neighbor = self._generate_neighbor(self._current_picks)

            if not self._is_valid(neighbor):
                continue

            score = self._evaluate(neighbor)

            if score > best_neighbor_score:
                best_neighbor_score = score
                best_neighbor       = neighbor
                improved            = True

        if improved:
            self._current_picks = best_neighbor
            self._current_score = best_neighbor_score
            self._update_best(self._current_picks, self._current_score)

    # ------------------------------------------------------------------ #
    # Neighbor generation                                                  #
    # ------------------------------------------------------------------ #

    def _generate_neighbor(self, picks):
        """
        Generate a neighbor by perturbing one randomly chosen pick.
        Perturbation is a small random change in time and/or velocity.
        The neighbor is kept sorted by time, so list order always
        matches temporal order.
        """
        neighbor = list(picks)
        idx      = int(self.rng.integers(0, self.n_picks))

        t, v = neighbor[idx]

        # Perturb time
        dt = int(self.rng.integers(-self.step_time, self.step_time + 1))
        t_new = int(np.clip(t + dt, 0, self.n_samples - 1))

        # Perturb velocity
        dv = float(self.rng.uniform(-self.step_vel, self.step_vel))
        v_new = float(np.clip(v + dv, self.vel_min, self.vel_max))

        neighbor[idx] = (t_new, v_new)
        neighbor.sort(key=lambda p: p[0])
        return neighbor

    # ------------------------------------------------------------------ #
    # Run with restarts                                                    #
    # ------------------------------------------------------------------ #

    def run(self):
        """
        Run Hill Climbing with multiple random restarts.
        Each restart begins from a new random solution and stops when it
        has no improvement for `patience` iterations. The best solution
        across all restarts is kept.

        self.history           : best score so far (global), one entry
                                 per iteration, all restarts concatenated
        self.restart_histories : current score of each restart, one list
                                 per restart (use this to plot restarts)
        """
        import time
        t_start = time.perf_counter()

        self._reset_run_state()
        self.restart_histories = []
        self.restart_scores    = []

        # One independent random generator per restart
        restart_seeds = np.random.SeedSequence(self.seed).spawn(self.restarts)

        for restart, restart_seed in enumerate(restart_seeds):
            if self.budget_exhausted:
                break

            self.rng = np.random.default_rng(restart_seed)

            self._initialize()

            restart_history = [self._current_score]
            stagnant        = 0

            for it in range(self.max_iter):
                if self.budget_exhausted:
                    break

                prev_score = self._current_score
                self._iterate()

                restart_history.append(self._current_score)
                self.history.append(self.best_score)

                # Early stop: count iterations without improvement of the
                # CURRENT solution of this restart (not of the global best)
                if self._current_score > prev_score:
                    stagnant = 0
                else:
                    stagnant += 1
                    if stagnant >= self.patience:
                        break

            self.restart_histories.append(restart_history)
            self.restart_scores.append(self._current_score)

        self.exec_time_s = time.perf_counter() - t_start
        return self