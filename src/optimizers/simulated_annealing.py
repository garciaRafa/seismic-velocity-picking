"""
src/optimizers/simulated_annealing.py
-------------------------------------
Simulated Annealing for seismic velocity picking.

Same neighborhood as Hill Climbing and Tabu Search (shift one pick; with
formulation v3 also insert / remove one pick, see BaseOptimizer), so any
difference in the results comes from the acceptance rule, not from the
moves:

    better or equal neighbor -> always accepted
    worse neighbor (delta < 0) -> accepted with probability exp(delta / T)

The temperature T starts high (many worse moves accepted: exploration)
and decreases geometrically with the number of EVALUATIONS, from T0 to
T0 * t_final_ratio when the budget max_evals is used up:

    T(n) = T0 * t_final_ratio ** (n / max_evals)

Tying the schedule to the budget (not to iterations) makes every run use
the whole budget with the same cooling, whatever the share of invalid
neighbors, so comparisons with the other algorithms are fair.

Initial temperature (t0=None): set so that, at the start, a worse move of
a reference size delta_ref is accepted with probability p0 (default 0.8):

    T0 = delta_ref / -ln(p0)

With formulation v3 the natural unit of the objective is tau, the price
of one pick (f = sum(S - tau)), so delta_ref = tau: losing one pick's
worth of value is accepted 80% of the time at the start. With the 'mean'
objective (v1/v2) there is no tau, and delta_ref is the 90th percentile
of |delta| over a few worse random neighbors of the initial solution
(these calibration evaluations count toward the budget). A percentile,
not the mean: the initial random solution has picks in empty regions,
where most moves change f by almost nothing and the mean would give a
temperature too low to explore.
"""

import time

import numpy as np

from .base import BaseOptimizer


class SimulatedAnnealing(BaseOptimizer):
    """
    Parameters (besides those of BaseOptimizer)
    -------------------------------------------
    step_time, step_vel : max perturbation of a shift move (samples, m/s)
    t0            : initial temperature, or None to set it from tau (v3) or
                    from a calibration (v1/v2), see the module docstring
    p0            : acceptance probability, at T0, of a worse move of the
                    reference size (used only when t0 is None)
    t_final_ratio : final temperature = T0 * t_final_ratio
    n_calib       : neighbors evaluated to calibrate T0
    """

    def __init__(self, traces, offsets, dt_ms,
                 vel_min=1400.0, vel_max=3000.0,
                 n_picks=10, max_iter=10**9, seed=42,
                 step_time=5, step_vel=50.0,
                 t0=None, p0=0.8, t_final_ratio=1e-3, n_calib=100,
                 max_evals=None, min_sep_ms=0.0, t_min_ms=0.0,
                 t_max_ms=None, **formulation):

        if max_evals is None or max_evals < 1:
            raise ValueError("SimulatedAnnealing needs a positive max_evals budget "
                             "(the cooling schedule is defined over it).")
        if not 0.0 < p0 < 1.0:
            raise ValueError("p0 must be in (0, 1).")
        if not 0.0 < t_final_ratio < 1.0:
            raise ValueError("t_final_ratio must be in (0, 1).")

        super().__init__(traces, offsets, dt_ms,
                         vel_min, vel_max,
                         n_picks, max_iter, seed, max_evals,
                         min_sep_ms, t_min_ms, t_max_ms, **formulation)

        self.step_time     = step_time
        self.step_vel      = step_vel
        self.t0_param      = t0
        self.p0            = float(p0)
        self.t_final_ratio = float(t_final_ratio)
        self.n_calib       = int(n_calib)

        self._current_picks = None
        self._current_score = -np.inf

    # ------------------------------------------------------------------ #
    # Neighborhood (shared with Hill Climbing)                             #
    # ------------------------------------------------------------------ #

    def _propose(self):
        """One neighbor of the current solution and the kind of move."""
        move = self._structural_move(self._current_picks)
        if move is not None:
            return move[0], move[1]
        neighbor, _, _, _ = self._generate_neighbor(
            self._current_picks, self.step_time, self.step_vel)
        return neighbor, 'shift'

    # ------------------------------------------------------------------ #
    # Temperature                                                          #
    # ------------------------------------------------------------------ #

    def _calibrate_t0(self):
        """T0 when t0 is None (see the module docstring)."""
        if self.variable_k:
            return float(self.tau / -np.log(self.p0))
        worse = []
        for _ in range(self.n_calib):
            if self.budget_exhausted:
                break
            neighbor, _ = self._propose()
            if not self._is_valid(neighbor):
                continue
            delta = self._evaluate(neighbor) - self._current_score
            if delta < 0:
                worse.append(-delta)
        if not worse:
            return 1.0
        return float(np.percentile(worse, 90) / -np.log(self.p0))

    @property
    def temperature(self):
        """Current temperature, from the share of the budget already used."""
        frac = min(1.0, self.n_evals / self.max_evals)
        return self.t0 * self.t_final_ratio ** frac

    # ------------------------------------------------------------------ #
    # BaseOptimizer interface                                              #
    # ------------------------------------------------------------------ #

    def _initialize(self):
        self._current_picks = self._random_picks()
        self._current_score = self._evaluate(self._current_picks)
        self._update_best(self._current_picks, self._current_score)
        self.t0 = (float(self.t0_param) if self.t0_param is not None
                   else self._calibrate_t0())
        self.move_stats   = {'shift': 0, 'insert': 0, 'remove': 0}
        self.accept_stats = {'better': 0, 'worse_accepted': 0, 'worse_rejected': 0}
        self._last_accepted = None
        self._last_delta    = 0.0

    def _iterate(self):
        """Propose one neighbor and accept it by the Metropolis rule."""
        neighbor, kind = self._propose()
        self._last_accepted = None
        if not self._is_valid(neighbor):
            return
        T = self.temperature
        score = self._evaluate(neighbor)
        delta = score - self._current_score
        self._last_delta = float(delta)
        if delta >= 0:
            accept = True
            self.accept_stats['better'] += 1
        else:
            accept = self.rng.random() < np.exp(delta / T)
            self.accept_stats['worse_accepted' if accept else 'worse_rejected'] += 1
        self._last_accepted = bool(accept)
        if accept:
            self._current_picks = neighbor
            self._current_score = score
            self.move_stats[kind] += 1
            self._update_best(neighbor, score)

    def _record_extra(self):
        """Trajectory fields of Simulated Annealing."""
        return {'temperature': float(self.temperature),
                'accepted': self._last_accepted,
                'delta': float(getattr(self, '_last_delta', 0.0))}

    def run(self):
        """
        Single trajectory, cooled over the whole budget. self.history: best
        score after each iteration.
        """
        t_start = time.perf_counter()
        self._reset_run_state()
        self.rng = np.random.default_rng(self.seed)
        self._initialize()
        self._record('start', iteration_done=False)

        for _ in range(self.max_iter):
            if self.budget_exhausted:
                break
            self._iterate()
            self._record('iter')
            self.history.append(self.best_score)

        self.exec_time_s = time.perf_counter() - t_start
        return self