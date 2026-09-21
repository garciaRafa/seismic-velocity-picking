"""
src/optimizers/base.py
----------------------
Abstract base class for all metaheuristic optimizers.
All algorithm implementations must inherit from this class
to ensure a uniform interface for comparison.
"""

import time
import numpy as np
from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """
    Base class for velocity picking metaheuristics.

    Every optimizer receives a CDP gather and searches for the
    set of (time, velocity) pairs that maximizes semblance.

    Parameters
    ----------
    traces     : np.ndarray, shape (n_traces, n_samples)
    offsets    : np.ndarray, shape (n_traces,) in meters
    dt_ms      : float — sampling interval in ms
    vel_min    : float — minimum velocity in m/s
    vel_max    : float — maximum velocity in m/s
    n_picks    : int   — number of velocity picks to find
    max_iter   : int   — maximum number of iterations
    seed       : int   — random seed for reproducibility
    """

    def __init__(self, traces, offsets, dt_ms,
                 vel_min=1500.0, vel_max=5000.0,
                 n_picks=10, max_iter=200, seed=42):

        self.traces   = traces
        self.offsets  = offsets
        self.dt_ms    = dt_ms
        self.vel_min  = vel_min
        self.vel_max  = vel_max
        self.n_picks  = n_picks
        self.max_iter = max_iter
        self.seed     = seed

        self.n_traces, self.n_samples = traces.shape

        # Results — filled after run()
        self.history       = []   # best fitness per iteration
        self.best_picks    = None # list of (time_sample, velocity)
        self.best_score    = -np.inf
        self.exec_time_s   = None

        np.random.seed(seed)

    # ------------------------------------------------------------------ #
    # Abstract methods — must be implemented by each algorithm            #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _initialize(self):
        """Initialize the search (population, current solution, etc.)."""
        pass

    @abstractmethod
    def _iterate(self):
        """Execute one iteration of the algorithm."""
        pass

    # ------------------------------------------------------------------ #
    # Shared utilities                                                     #
    # ------------------------------------------------------------------ #

    def _evaluate(self, picks):
        """
        Evaluate a candidate solution using semblance as objective.

        Parameters
        ----------
        picks : list of (time_sample, velocity) tuples

        Returns
        -------
        score : float in [0, 1]
        """
        from src.seismic.semblance import picking_objective
        return picking_objective(
            self.traces, self.offsets, picks, self.dt_ms
        )

    def _is_valid(self, picks):
        """
        Check physical plausibility constraints:
        - velocities within [vel_min, vel_max]
        - times within valid sample range and distinct
        - velocities increase monotonically with time
        """
        # Sort by time so the check does not depend on list order
        ordered = sorted(picks, key=lambda p: p[0])
        times = [p[0] for p in ordered]
        vels  = [p[1] for p in ordered]

        # Velocity bounds
        if any(v < self.vel_min or v > self.vel_max for v in vels):
            return False

        # Time bounds
        if any(t < 0 or t >= self.n_samples for t in times):
            return False

        # Two picks at the same time are redundant
        if len(set(times)) != len(times):
            return False

        # Monotonic velocity increase with time
        for i in range(1, len(vels)):
            if vels[i] < vels[i - 1]:
                return False

        return True

    def _random_picks(self):
        """Generate a random valid set of picks."""
        times = sorted(np.random.choice(
            self.n_samples, self.n_picks, replace=False
        ))
        vels = sorted(np.random.uniform(
            self.vel_min, self.vel_max, self.n_picks
        ))
        return list(zip(times, vels))

    # ------------------------------------------------------------------ #
    # Main execution                                                       #
    # ------------------------------------------------------------------ #

    def run(self):
        """
        Execute the optimization and return self.

        Usage:
            optimizer = MyOptimizer(traces, offsets, dt_ms)
            optimizer.run()
            times, vels = optimizer.get_result()
        """
        self._initialize()

        t_start = time.perf_counter()

        for it in range(self.max_iter):
            self._iterate()

            if self.best_score not in self.history or \
               (self.history and self.best_score > self.history[-1]):
                self.history.append(self.best_score)
            else:
                self.history.append(
                    self.history[-1] if self.history else self.best_score
                )

        self.exec_time_s = time.perf_counter() - t_start
        return self

    def get_result(self):
        """
        Return the best picks found.

        Returns
        -------
        times      : np.ndarray of time samples
        velocities : np.ndarray of velocities in m/s
        """
        if self.best_picks is None:
            raise RuntimeError("Run the optimizer first with .run()")

        times = np.array([p[0] for p in self.best_picks])
        vels  = np.array([p[1] for p in self.best_picks])

        order = np.argsort(times)
        return times[order], vels[order]

    def get_metrics(self, true_velocities=None):
        """
        Return performance metrics.

        Parameters
        ----------
        true_velocities : np.ndarray or None
            If provided, computes RMSE and MAE against ground truth.

        Returns
        -------
        metrics : dict
        """
        metrics = {
            'algorithm':   self.__class__.__name__,
            'best_score':  self.best_score,
            'exec_time_s': self.exec_time_s,
            'n_iter':      len(self.history),
            'seed':        self.seed,
        }

        if true_velocities is not None:
            _, vels = self.get_result()
            n = min(len(vels), len(true_velocities))
            rmse = np.sqrt(np.mean((vels[:n] - true_velocities[:n]) ** 2))
            mae  = np.mean(np.abs(vels[:n] - true_velocities[:n]))
            metrics['rmse'] = float(rmse)
            metrics['mae']  = float(mae)

        return metrics