"""
src/optimizers/random_search.py
-------------------------------
Random search for seismic velocity picking.

Strategy:
    Repeatedly draw a random valid solution (same sampler used to start
    the other algorithms) and keep the best one found, until the budget
    of objective evaluations is used up.

This is the minimum reference any search method must beat: an algorithm
is only useful if, with the same number of evaluations, it finds better
solutions than sampling at random.
"""

from src.optimizers.base import BaseOptimizer


class RandomSearch(BaseOptimizer):
    """
    Random search optimizer for seismic velocity picking.

    Parameters
    ----------
    traces, offsets, dt_ms, vel_min, vel_max, n_picks, seed,
    min_sep_ms, t_min_ms, t_max_ms : see BaseOptimizer
    max_evals : int — number of random solutions to evaluate (required;
                it is the only stopping criterion of the method)
    """

    def __init__(self, traces, offsets, dt_ms,
                 vel_min=1400.0, vel_max=3000.0,
                 n_picks=10, seed=42, max_evals=None,
                 min_sep_ms=0.0, t_min_ms=0.0, t_max_ms=None):

        if max_evals is None or max_evals < 1:
            raise ValueError("RandomSearch needs a positive max_evals budget.")

        # One evaluation per iteration, so max_iter = max_evals
        super().__init__(traces, offsets, dt_ms,
                         vel_min, vel_max,
                         n_picks, max_evals, seed, max_evals,
                         min_sep_ms, t_min_ms, t_max_ms)

    def _initialize(self):
        """Evaluate the first random solution."""
        picks = self._random_picks()
        self._update_best(picks, self._evaluate(picks))

    def _iterate(self):
        """Evaluate one new random solution and keep it if it is better."""
        picks = self._random_picks()
        self._update_best(picks, self._evaluate(picks))
