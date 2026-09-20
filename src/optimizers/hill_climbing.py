"""
src/optimizers/hill_climbing.py
--------------------------------
Hill Climbing for seismic velocity picking.

Strategy:
    Start from a random valid solution (set of velocity picks).
    At each iteration, generate neighbors by perturbing one pick
    at a time (small changes in time and/or velocity).
    Move to a neighbor only if it improves the semblance score.
    Stop when no improvement is found (local optimum) or
    max_iter is reached.

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
    traces       : np.ndarray, shape (n_traces, n_samples)
    offsets      : np.ndarray, shape (n_traces,) in meters
    dt_ms        : float — sampling interval in ms
    vel_min      : float — minimum velocity in m/s
    vel_max      : float — maximum velocity in m/s
    n_picks      : int   — number of velocity picks to find
    max_iter     : int   — maximum number of iterations
    seed         : int   — random seed
    step_time    : int   — max perturbation in time samples
    step_vel     : float — max perturbation in velocity (m/s)
    n_neighbors  : int   — neighbors evaluated per iteration
    restarts     : int   — number of random restarts
    """

    def __init__(self, traces, offsets, dt_ms,
                 vel_min=1400.0, vel_max=3000.0,
                 n_picks=10, max_iter=200, seed=42,
                 step_time=5, step_vel=50.0,
                 n_neighbors=20, restarts=5):

        super().__init__(traces, offsets, dt_ms,
                         vel_min, vel_max,
                         n_picks, max_iter, seed)

        self.step_time   = step_time
        self.step_vel    = step_vel
        self.n_neighbors = n_neighbors
        self.restarts    = restarts

        # Internal state
        self._current_picks = None
        self._current_score = -np.inf

    # ------------------------------------------------------------------ #
    # Abstract method implementations                                      #
    # ------------------------------------------------------------------ #

    def _initialize(self):
        """Start from a random valid solution."""
        self._current_picks = self._random_picks()
        self._current_score = self._evaluate(self._current_picks)

        # Update global best
        if self._current_score > self.best_score:
            self.best_score = self._current_score
            self.best_picks = list(self._current_picks)

    def _iterate(self):
        """
        One iteration: evaluate neighbors and move to best improvement.
        """
        improved = False
        best_neighbor       = None
        best_neighbor_score = self._current_score

        for _ in range(self.n_neighbors):
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

            if self._current_score > self.best_score:
                self.best_score = self._current_score
                self.best_picks = list(self._current_picks)

    # ------------------------------------------------------------------ #
    # Neighbor generation                                                  #
    # ------------------------------------------------------------------ #

    def _generate_neighbor(self, picks):
        """
        Generate a neighbor by perturbing one randomly chosen pick.
        Perturbation is a small random change in time and/or velocity.
        """
        neighbor = list(picks)
        idx      = np.random.randint(0, self.n_picks)

        t, v = neighbor[idx]

        # Perturb time
        dt = np.random.randint(-self.step_time, self.step_time + 1)
        t_new = int(np.clip(t + dt, 0, self.n_samples - 1))

        # Perturb velocity
        dv = np.random.uniform(-self.step_vel, self.step_vel)
        v_new = float(np.clip(v + dv, self.vel_min, self.vel_max))

        neighbor[idx] = (t_new, v_new)
        return neighbor

    # ------------------------------------------------------------------ #
    # Run with restarts                                                    #
    # ------------------------------------------------------------------ #

    def run(self):
        """
        Run Hill Climbing with multiple random restarts.
        Each restart begins from a new random solution.
        The best solution across all restarts is kept.
        """
        import time
        t_start = time.perf_counter()

        for restart in range(self.restarts):
            np.random.seed(self.seed + restart)

            self._initialize()

            for it in range(self.max_iter):
                prev_score = self._current_score
                self._iterate()

                self.history.append(self.best_score)

                # Early stop if stuck
                if self._current_score == prev_score:
                    stagnant = sum(
                        1 for s in self.history[-20:]
                        if s == self.history[-1]
                    )
                    if stagnant >= 20:
                        break

        self.exec_time_s = time.perf_counter() - t_start
        return self