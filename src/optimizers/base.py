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
    max_evals  : int or None — budget of objective evaluations (one
                 evaluation = one full solution). None means no budget.
    min_sep_ms : float — minimum time separation between consecutive
                 picks, in ms. 0 (default) only requires distinct times.
    t_min_ms   : float — earliest time allowed for a pick, in ms
                 (default 0). Use it to exclude trace segments without
                 reflections, such as the water layer.
    t_max_ms   : float or None — latest time allowed for a pick, in ms
                 (default: end of the record). Use it to exclude the end
                 of the record, where far-offset traces leave the record
                 and semblance is limited by the fraction of live traces.

    Formulation
    -----------
    With the defaults (min_sep_ms=0, t_min_ms=0, t_max_ms=None) the
    constraints and the random draws are exactly those of formulation v1,
    so earlier results are reproduced. Setting the three parameters gives
    formulation v2.

    Randomness
    ----------
    Every optimizer draws random numbers ONLY from its own generator,
    self.rng (numpy Generator). The global np.random state is never used,
    so other code (notebook cells, data generation, other optimizers)
    cannot change the sequence of draws, and the same seed always
    reproduces the same run.
    """

    def __init__(self, traces, offsets, dt_ms,
                 vel_min=1500.0, vel_max=5000.0,
                 n_picks=10, max_iter=200, seed=42, max_evals=None,
                 min_sep_ms=0.0, t_min_ms=0.0, t_max_ms=None):

        self.traces   = traces
        self.offsets  = offsets
        self.dt_ms    = dt_ms
        self.vel_min  = vel_min
        self.vel_max  = vel_max
        self.n_picks  = n_picks
        self.max_iter  = max_iter
        self.seed      = seed
        self.max_evals = max_evals

        self.n_traces, self.n_samples = traces.shape

        # Time constraints, converted from ms to sample indices
        self.min_sep_ms = float(min_sep_ms)
        self.t_min_ms   = float(t_min_ms)
        self.t_max_ms   = t_max_ms
        self.min_sep = max(1, int(np.ceil(self.min_sep_ms / dt_ms - 1e-9)))
        self.t_min   = max(0, int(np.ceil(self.t_min_ms / dt_ms - 1e-9)))
        self.t_max   = (self.n_samples - 1 if t_max_ms is None else
                        min(self.n_samples - 1,
                            int(np.floor(t_max_ms / dt_ms + 1e-9))))

        if (self.n_picks - 1) * self.min_sep > self.t_max - self.t_min:
            raise ValueError(
                f"{self.n_picks} picks separated by {self.min_sep_ms} ms "
                f"do not fit between {self.t_min_ms} and {t_max_ms} ms."
            )

        # Own random generator (see "Randomness" above)
        self.rng = np.random.default_rng(seed)

        # Results — filled after run()
        self._reset_run_state()

    def _reset_run_state(self):
        """Clear results and counters before a new run."""
        self.history       = []   # best fitness per iteration
        self.eval_history  = []   # (n_evals, best_score) at each improvement
        self.best_picks    = None # list of (time_sample, velocity)
        self.best_score    = -np.inf
        self.exec_time_s   = None
        self.n_evals       = 0    # objective evaluations performed

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
        self.n_evals += 1
        return picking_objective(
            self.traces, self.offsets, picks, self.dt_ms
        )

    @property
    def budget_exhausted(self):
        """True when the evaluation budget (max_evals) has been used up."""
        return self.max_evals is not None and self.n_evals >= self.max_evals

    def _update_best(self, picks, score):
        """
        Keep the global best solution and record when it improved,
        measured in number of evaluations (for fair comparisons).
        """
        if score > self.best_score:
            self.best_score = score
            self.best_picks = list(picks)
            self.eval_history.append((self.n_evals, score))

    def _is_valid(self, picks):
        """
        Check physical plausibility constraints:
        - velocities within [vel_min, vel_max]
        - times within [t_min, t_max] (sample indices)
        - consecutive times separated by at least min_sep samples
          (min_sep = 1 means only distinct times)
        - velocities increase monotonically with time
        """
        # Sort by time so the check does not depend on list order
        ordered = sorted(picks, key=lambda p: p[0])
        times = [p[0] for p in ordered]
        vels  = [p[1] for p in ordered]

        # Velocity bounds
        if any(v < self.vel_min or v > self.vel_max for v in vels):
            return False

        # Time bounds (allowed time window)
        if any(t < self.t_min or t > self.t_max for t in times):
            return False

        # Minimum separation between consecutive picks (also rejects
        # repeated times, which are always redundant)
        for i in range(1, len(times)):
            if times[i] - times[i - 1] < self.min_sep:
                return False

        # Monotonic velocity increase with time
        for i in range(1, len(vels)):
            if vels[i] < vels[i - 1]:
                return False

        return True

    def _random_picks(self):
        """
        Generate a random valid set of picks.

        Times are drawn uniformly among all sets that respect the time
        window and the minimum separation: K distinct values are drawn
        from a reduced range and then spread apart by (min_sep - 1)
        samples each. With min_sep = 1 and the full window this is
        exactly the formulation v1 draw.
        """
        gap    = self.min_sep - 1
        n_free = (self.t_max - self.t_min) - (self.n_picks - 1) * gap + 1
        base   = sorted(int(t) for t in self.rng.choice(
            n_free, self.n_picks, replace=False
        ))
        times = [self.t_min + b + i * gap for i, b in enumerate(base)]
        vels = sorted(float(v) for v in self.rng.uniform(
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
        self._reset_run_state()
        self.rng = np.random.default_rng(self.seed)   # same seed, same run
        self._initialize()

        t_start = time.perf_counter()

        for it in range(self.max_iter):
            if self.budget_exhausted:
                break
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
            'n_evals':     self.n_evals,
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