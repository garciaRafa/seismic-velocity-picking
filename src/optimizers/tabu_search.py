"""
src/optimizers/tabu_search.py
------------------------------
Tabu Search for seismic velocity picking.

Strategy:
    Like Hill Climbing, explores the neighborhood of the current
    solution by perturbing one pick at a time (the exact same move
    operator, inherited from BaseOptimizer._generate_neighbor). Unlike
    Hill Climbing, it does NOT only accept improving moves: at each
    iteration it takes the best NON-TABU neighbor, even if it is worse
    than the current solution. This is what lets Tabu Search escape
    local optima without restarting from a new random solution.

Tabu attribute
---------------
A move is identified by (pick_index, time_sample) — NOT by the full
solution. The full solution is quasi-continuous (velocity is a float,
perturbed by a uniform draw), so two solutions are essentially never
bit-for-bit identical again; a tabu on the whole solution would almost
never trigger. Time, on the other hand, is already discrete (a sample
index), and the typical degenerate cycle in this kind of local search
is a pick oscillating in TIME between two nearby, similarly strong
reflectors — so the attribute that matters for cycling is time, not
velocity.

When a move is accepted (pick `idx` moves from `t_old` to `t_new`),
the attribute `(idx, t_old)` becomes tabu for `tabu_tenure` iterations:
the search cannot set pick `idx`'s time back to `t_old` during that
window, which is exactly what would undo the move just taken.

Why tracking by position (idx) is safe here: a single-pick
perturbation moves a pick by at most `step_time` samples, while picks
are kept `min_sep` samples apart (formulation v2). As long as
step_time is small relative to min_sep (the project's defaults are 5
and 50 samples respectively), a perturbation can never cross over a
neighboring pick, so each position in the time-sorted list keeps
referring to "the same" pick across iterations.

Aspiration criterion
----------------------
A tabu move is allowed anyway if it produces a new global best score
(classic objective-based aspiration) — otherwise the tabu list could
permanently block the best solution found so far, which would make
the algorithm strictly worse than doing nothing.

No restarts
------------
Unlike Hill Climbing, Tabu Search does not use random restarts: its
memory of recently-undone moves is itself the mechanism for escaping
local optima, so adding restarts on top would mix two different
escape strategies and make it harder to attribute any effect to
either one. Early stopping instead tracks stagnation of the GLOBAL
best score directly (`patience` iterations without improvement),
rather than per-restart stagnation as in HillClimbing.
"""

import numpy as np
from src.optimizers.base import BaseOptimizer


class TabuSearch(BaseOptimizer):
    """
    Tabu Search optimizer for seismic velocity picking.

    Parameters
    ----------
    traces      : np.ndarray, shape (n_traces, n_samples)
    offsets     : np.ndarray, shape (n_traces,) in meters
    dt_ms       : float — sampling interval in ms
    vel_min     : float — minimum velocity in m/s
    vel_max     : float — maximum velocity in m/s
    n_picks     : int   — number of velocity picks to find
    max_iter    : int   — maximum number of iterations (single trajectory,
                          no restarts)
    seed        : int   — random seed
    step_time   : int   — max perturbation in time samples
    step_vel    : float — max perturbation in velocity (m/s)
    n_neighbors : int   — neighbors evaluated per iteration
    tabu_tenure : int   — iterations a forbidden (pick_index, time_sample)
                          attribute stays tabu after its move is undone
    patience    : int   — iterations without improvement of the GLOBAL
                          best score before stopping
    max_evals   : int or None — budget of objective evaluations. None =
                          no budget.
    min_sep_ms, t_min_ms, t_max_ms : time constraints of the formulation
                          (see BaseOptimizer). Defaults = formulation v1.
    """

    def __init__(self, traces, offsets, dt_ms,
                 vel_min=1400.0, vel_max=3000.0,
                 n_picks=10, max_iter=300, seed=42,
                 step_time=5, step_vel=50.0, n_neighbors=30,
                 tabu_tenure=15, patience=60,
                 max_evals=None, min_sep_ms=0.0, t_min_ms=0.0,
                 t_max_ms=None):

        super().__init__(traces, offsets, dt_ms,
                         vel_min, vel_max,
                         n_picks, max_iter, seed, max_evals,
                         min_sep_ms, t_min_ms, t_max_ms)

        self.step_time   = step_time
        self.step_vel    = step_vel
        self.n_neighbors = n_neighbors
        self.tabu_tenure = tabu_tenure
        self.patience    = patience

        # Internal state — (re)initialized in _initialize()
        self._current_picks = None
        self._current_score = -np.inf
        self._tabu          = {}   # {(pick_index, time_sample): expires_after_iter}
        self._iteration      = 0
        self._no_improve     = 0

    # ------------------------------------------------------------------ #
    # Abstract method implementations                                      #
    # ------------------------------------------------------------------ #

    def _initialize(self):
        """Start from a random valid solution. Empty tabu list."""
        self._current_picks = self._random_picks()
        self._current_score = self._evaluate(self._current_picks)
        self._tabu       = {}
        self._iteration  = 0
        self._no_improve = 0
        self._update_best(self._current_picks, self._current_score)

    def _iterate(self):
        """
        One iteration: evaluate n_neighbors candidates, discard the
        ones that are both invalid and the ones that are tabu (unless
        aspiration applies), and move to the BEST remaining candidate
        — even if it is worse than the current solution. Register the
        move just taken as tabu (forbid undoing it for `tabu_tenure`
        iterations).
        """
        self._iteration += 1
        prev_best = self.best_score

        best_neighbor       = None
        best_neighbor_score = -np.inf
        best_move           = None   # (idx, t_old): what becomes tabu

        for _ in range(self.n_neighbors):
            if self.budget_exhausted:
                break

            neighbor, idx, t_old, t_new = self._generate_neighbor(
                self._current_picks, self.step_time, self.step_vel
            )
            if not self._is_valid(neighbor):
                continue

            move    = (idx, t_new)
            is_tabu = move in self._tabu and self._tabu[move] > self._iteration

            score   = self._evaluate(neighbor)
            aspired = score > self.best_score

            if is_tabu and not aspired:
                continue

            if score > best_neighbor_score:
                best_neighbor_score = score
                best_neighbor       = neighbor
                best_move           = (idx, t_old)

        if best_neighbor is not None:
            self._current_picks = best_neighbor
            self._current_score = best_neighbor_score
            self._tabu[best_move] = self._iteration + self.tabu_tenure
            self._update_best(self._current_picks, self._current_score)

        # Early-stop bookkeeping tracks the GLOBAL best (there is no
        # per-restart "current solution" concept here, unlike HC).
        if self.best_score > prev_best:
            self._no_improve = 0
        else:
            self._no_improve += 1

    # ------------------------------------------------------------------ #
    # Run — single trajectory, no restarts                                 #
    # ------------------------------------------------------------------ #

    def run(self):
        """
        Run Tabu Search: a single trajectory from one random start,
        escaping local optima via the tabu list + aspiration instead
        of restarting from a new random solution.

        self.history : best score so far (global), one entry per
                       iteration actually executed.
        """
        import time
        t_start = time.perf_counter()

        self._reset_run_state()
        self.rng = np.random.default_rng(self.seed)   # same seed, same run
        self._initialize()

        for it in range(self.max_iter):
            if self.budget_exhausted or self._no_improve >= self.patience:
                break
            self._iterate()
            self.history.append(self.best_score)

        self.exec_time_s = time.perf_counter() - t_start
        return self