"""
src/optimizers/memetic.py
-------------------------
Memetic Algorithm for seismic velocity picking: a steady-state genetic
algorithm whose children are improved by a short Hill Climbing.

    1. population of pop_size random valid solutions, each evaluated and
       (ls_initial=True) improved by the local search
    2. until the budget (max_evals) is used up, one child per iteration:
       a. two parents by tournament
       b. crossover by time cut(s): the child takes the picks of parent A
          before the cut and those of parent B after it (n_cuts=2: A, B, A
          — the middle of the record comes from B)
       c. repair: picks that break min_sep or the velocity rule at a cut
          are dropped (left to right)
       d. mutation, with probability p_mutation: ONE move of the shared
          neighborhood (insert / remove with p_insert / p_remove, else
          shift); a move that gives an invalid solution is ignored
       e. evaluation (1 evaluation) and local search: best-improvement
          Hill Climbing with the same moves, n_neighbors per iteration,
          stopped after ls_patience iterations without improvement
       f. replacement: the child replaces the worst member if it is
          better — unless it is equivalent to a member already in the
          population (same number of picks, each within div_time_ms and
          div_vel of the other's), then it can only replace that member
       g. partial restart: after restart_after children in a row that
          brought no new solution (only twins), the population has converged; all members but the
          best are replaced by new random solutions

Why: Hill Climbing with many restarts and Tabu Search with one long
trajectory each fail in a different way on the free-surface cases — the
first does not refine, the second can stay on a configuration it only
leaves through an invalid state (a primary dropped, a deep multiple kept).
The crossover combines the shallow part of one solution with the deep part
of another in one step, which no single local move does.

Budget: only objective evaluations count, as for every other algorithm
(crossover, repair, mutation and selection draw no evaluation; invalid
children are discarded without evaluation; members keep their score).
The budget is checked at every evaluation, including inside the local
search.
"""

import time

import numpy as np

from .base import BaseOptimizer


class MemeticAlgorithm(BaseOptimizer):
    """
    Parameters (besides those of BaseOptimizer)
    -------------------------------------------
    pop_size        : number of solutions in the population
    tournament_size : solutions drawn per tournament (the best one is a parent)
    n_cuts          : time cuts of the crossover (1: A|B, 2: A|B|A)
    p_mutation      : probability of one mutation move per child
    step_time, step_vel : max perturbation of a shift move (samples, m/s)
    n_neighbors     : neighbors evaluated per local-search iteration
    ls_patience     : local-search iterations without improvement before it stops
    ls_initial      : also apply the local search to the initial population
    div_time_ms, div_vel : tolerance for two solutions to be equivalent
                      (time in ms, velocity relative), as in the evaluation
    restart_after   : when this many children in a row bring no new solution
                      into the population (none replaces the worst member: it
                      has converged), every member except the
                      best is replaced by a new random solution (with local
                      search); None disables it
    """

    def __init__(self, traces, offsets, dt_ms,
                 vel_min=1400.0, vel_max=3000.0,
                 n_picks=10, max_iter=10**9, seed=42,
                 pop_size=20, tournament_size=2, n_cuts=1, p_mutation=0.5,
                 step_time=5, step_vel=50.0, n_neighbors=10, ls_patience=5,
                 ls_initial=True, div_time_ms=40.0, div_vel=0.02, restart_after=100,
                 max_evals=None, min_sep_ms=0.0, t_min_ms=0.0,
                 t_max_ms=None, **formulation):

        if max_evals is None or max_evals < 1:
            raise ValueError("MemeticAlgorithm needs a positive max_evals budget.")
        if pop_size < 2 or not 1 <= tournament_size <= pop_size:
            raise ValueError("pop_size must be >= 2 and 1 <= tournament_size <= pop_size.")
        if n_cuts < 1:
            raise ValueError("n_cuts must be >= 1.")
        if not 0.0 <= p_mutation <= 1.0:
            raise ValueError("p_mutation must be in [0, 1].")

        super().__init__(traces, offsets, dt_ms,
                         vel_min, vel_max,
                         n_picks, max_iter, seed, max_evals,
                         min_sep_ms, t_min_ms, t_max_ms, **formulation)

        self.pop_size        = int(pop_size)
        self.tournament_size = int(tournament_size)
        self.n_cuts          = int(n_cuts)
        self.p_mutation      = float(p_mutation)
        self.step_time       = step_time
        self.step_vel        = step_vel
        self.n_neighbors     = int(n_neighbors)
        self.ls_patience     = int(ls_patience)
        self.ls_initial      = bool(ls_initial)
        self.div_samples     = div_time_ms / dt_ms
        self.div_vel         = float(div_vel)
        self.restart_after   = None if restart_after is None else int(restart_after)

        self.population = []          # list of (picks, score)
        self._current_picks = None
        self._current_score = -np.inf

    # ------------------------------------------------------------------ #
    # Operators                                                            #
    # ------------------------------------------------------------------ #

    def _tournament(self):
        """Index of the best of tournament_size members drawn without replacement."""
        idx = self.rng.choice(len(self.population), self.tournament_size, replace=False)
        return int(max(idx, key=lambda i: self.population[i][1]))

    def _crossover(self, a, b):
        """
        Child of a and b by n_cuts time cuts: segments alternate between
        a and b, starting with a. Cut times are drawn uniformly in the
        allowed time window.
        """
        cuts = np.sort(self.rng.integers(self.t_min, self.t_max + 1, self.n_cuts))
        child = []
        for parent_is_a, picks in ((True, a), (False, b)):
            for t, v in picks:
                segment = int(np.searchsorted(cuts, t, side='right'))
                if (segment % 2 == 0) == parent_is_a:
                    child.append((t, v))
        child.sort(key=lambda p: p[0])
        return child

    def _repair(self, picks):
        """
        Drop, from left to right, every pick that breaks min_sep or the
        velocity rule (monotonicity / minimum gradient) with the last kept
        pick; then trim to k_max. Never adds picks.
        """
        g = self._dv_per_sample
        kept = []
        for t, v in sorted(picks, key=lambda p: p[0]):
            if kept:
                t0, v0 = kept[-1]
                if t - t0 < self.min_sep or v - v0 < g * (t - t0):
                    continue
            kept.append((t, v))
        if self.variable_k and len(kept) > self.k_max:
            drop = set(self.rng.choice(len(kept), len(kept) - self.k_max, replace=False))
            kept = [p for i, p in enumerate(kept) if i not in drop]
        return kept

    def _neighbor(self, picks):
        """One move of the shared neighborhood (as Hill Climbing)."""
        move = self._structural_move(picks)
        if move is not None:
            return move[0]
        neighbor, _, _, _ = self._generate_neighbor(picks, self.step_time, self.step_vel)
        return neighbor

    def _mutate(self, picks):
        """With probability p_mutation, one move; ignored if it is invalid."""
        if self.rng.random() >= self.p_mutation:
            return picks
        mutant = self._neighbor(picks)
        return mutant if self._is_valid(mutant) else picks

    def _local_search(self, picks, score):
        """
        Best-improvement Hill Climbing from (picks, score): n_neighbors
        per iteration, stops after ls_patience iterations without
        improvement or when the budget is used up.
        """
        stagnant = 0
        while stagnant < self.ls_patience and not self.budget_exhausted:
            best_n, best_s = None, score
            for _ in range(self.n_neighbors):
                if self.budget_exhausted:
                    break
                n = self._neighbor(picks)
                if not self._is_valid(n):
                    continue
                s = self._evaluate(n)
                self._update_best(n, s)
                if s > best_s:
                    best_n, best_s = n, s
            if best_n is None:
                stagnant += 1
            else:
                picks, score, stagnant = best_n, best_s, 0
        return picks, score

    # ------------------------------------------------------------------ #
    # Diversity                                                            #
    # ------------------------------------------------------------------ #

    def n_unmatched(self, a, b):
        """
        Distance between two solutions: picks of a and b left without a
        partner after matching them in time order (a partner is within
        div_time_ms and div_vel). 0 means equivalent solutions.
        """
        i = j = matched = 0
        while i < len(a) and j < len(b):
            (ta, va), (tb, vb) = a[i], b[j]
            if abs(ta - tb) <= self.div_samples and abs(va - vb) <= self.div_vel * max(va, vb):
                matched += 1
                i += 1
                j += 1
            elif ta < tb:
                i += 1
            else:
                j += 1
        return len(a) + len(b) - 2 * matched

    def diversity(self):
        """Mean distance (n_unmatched) over all pairs of the population."""
        pop = [p for p, _ in self.population]
        d = [self.n_unmatched(pop[i], pop[j])
             for i in range(len(pop)) for j in range(i + 1, len(pop))]
        return float(np.mean(d)) if d else 0.0

    def _replace(self, child, score):
        """
        Steady-state elitist replacement with a diversity rule. Returns
        'replaced_worst', 'replaced_twin', 'rejected' (not better than the
        worst) or 'rejected_twin' (equivalent to a better member).
        """
        twins = [i for i, (p, _) in enumerate(self.population)
                 if self.n_unmatched(p, child) == 0]
        if twins:
            i = min(twins, key=lambda k: self.population[k][1])
            if score > self.population[i][1]:
                self.population[i] = (child, score)
                return 'replaced_twin'
            return 'rejected_twin'
        worst = min(range(len(self.population)), key=lambda k: self.population[k][1])
        if score > self.population[worst][1]:
            self.population[worst] = (child, score)
            return 'replaced_worst'
        return 'rejected'

    # ------------------------------------------------------------------ #
    # BaseOptimizer interface                                              #
    # ------------------------------------------------------------------ #

    def _fill_population(self):
        """Add random valid solutions (evaluated, improved if ls_initial) up to pop_size."""
        while len(self.population) < self.pop_size and not self.budget_exhausted:
            picks = self._random_picks()
            score = self._evaluate(picks)
            self._update_best(picks, score)
            if self.ls_initial:
                picks, score = self._local_search(picks, score)
            self.population.append((picks, score))

    def _initialize(self):
        """Random valid population, evaluated (and improved if ls_initial)."""
        self.population = []
        self._fill_population()
        best = max(self.population, key=lambda m: m[1])
        self._current_picks, self._current_score = best
        self.generation = 0
        self._since_accept = 0
        self.stats = {'children': 0, 'mutated': 0, 'repaired_empty': 0,
                      'replaced_worst': 0, 'replaced_twin': 0, 'rejected': 0,
                      'rejected_twin': 0, 'restarts': 0}

    def _iterate(self):
        """One child: selection, crossover, repair, mutation, local search, replacement."""
        a = self.population[self._tournament()][0]
        b = self.population[self._tournament()][0]
        child = self._repair(self._crossover(a, b))
        if self.variable_k and len(child) < self.k_min:
            self.stats['repaired_empty'] += 1
            child = list(a)
        mutant = self._mutate(child)
        self.stats['mutated'] += mutant is not child
        child = mutant
        if not self._is_valid(child):
            return
        score = self._evaluate(child)
        self._update_best(child, score)
        child, score = self._local_search(child, score)
        self.stats['children'] += 1
        outcome = self._replace(child, score)
        self.stats[outcome] += 1
        self._current_picks, self._current_score = child, score
        self._last_accepted = outcome.startswith('replaced')
        self.generation += 1
        # only a NEW solution entering counts (a twin replacing its twin is a
        # small refinement of a solution already there)
        self._since_accept = 0 if outcome == 'replaced_worst' else self._since_accept + 1
        if self.restart_after is not None and self._since_accept >= self.restart_after:
            # Converged: keep the best member, new random solutions for the rest
            self.population = [max(self.population, key=lambda m: m[1])]
            self._fill_population()
            self._since_accept = 0
            self.stats['restarts'] += 1

    def _record_extra(self):
        """Trajectory fields of the Memetic Algorithm."""
        scores = [s for _, s in self.population]
        stats = getattr(self, 'stats', {})
        return {'child': int(getattr(self, 'generation', 0)),
                'accepted': getattr(self, '_last_accepted', None),
                'pop_best': float(max(scores)) if scores else None,
                'pop_worst': float(min(scores)) if scores else None,
                'diversity': self.diversity(),
                'restarts': int(stats.get('restarts', 0))}

    def run(self):
        """
        Steady-state memetic search until the budget is used up.
        self.history: best score after each child.
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