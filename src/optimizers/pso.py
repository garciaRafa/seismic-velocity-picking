"""
src/optimizers/pso.py
---------------------
Particle Swarm Optimization for seismic velocity picking, with optional
Lévy flights.

Encoding
    PSO moves real vectors of fixed length, but a picking solution has a
    variable number of picks K (formulation v3). Each particle holds
    n_slots slots (default k_max, the most picks that fit), and each slot
    three coordinates:

        time (samples), velocity (m/s), presence in [0, 1]

    A slot is a pick when its presence is above 0.5. Decoding: active
    slots -> picks (time rounded, both clipped to the bounds), sorted in
    time, then repaired: the largest subset of them that respects min_sep
    and the velocity rule is kept (longest valid chain, by dynamic
    programming over the few slots). If fewer than k_min picks remain, the slots with the
    highest presence are used. The decoded solution is always valid and is
    evaluated once (1 evaluation per particle per iteration).

    The slots of every particle are kept sorted by time (the slots and
    their velocities are reordered together after each move), so slot i
    is roughly "the i-th pick" in every particle and the attraction to the
    personal / global best pulls corresponding picks toward each other.
    With K fixed (formulation v1 / v2) presence is ignored and every slot
    is a pick (n_slots = n_picks) — not supported yet: PSO runs only with
    formulation v3 (objective='threshold').

Update (global-best PSO, constriction-type coefficients)
    v <- w v + c1 r1 (pbest - x) + c2 r2 (gbest - x),   clipped to
         v_max_frac of each coordinate's range
    x <- x + v,  then clipped to the bounds

Lévy flights (levy=True, off by default)
    After the update, each particle takes, with probability p_levy, a
    Lévy-flight step: x <- x + levy_scale * L * range, L drawn coordinate
    by coordinate from a Lévy distribution of index levy_beta (Mantegna's
    algorithm). Most steps are small, a few are very large: the swarm
    keeps exploring after it starts to converge, the usual remedy for the
    premature convergence of PSO.

Initialization
    Each particle starts from a random valid solution (as the other
    algorithms): its picks fill the first slots with presence 0.75; the
    remaining slots get random times and velocities and presence 0.25
    (inactive, but ready to be switched on).

Budget: only objective evaluations count (one per decoded particle).
"""

import time

import numpy as np

from .base import BaseOptimizer


def levy_steps(rng, size, beta=1.5):
    """
    Lévy-distributed steps of index beta (1 < beta <= 2) by Mantegna's
    algorithm: u / |v|^(1/beta), u ~ N(0, sigma_u^2), v ~ N(0, 1).
    """
    from math import gamma, pi, sin
    sigma_u = (gamma(1 + beta) * sin(pi * beta / 2)
               / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0.0, sigma_u, size)
    v = rng.normal(0.0, 1.0, size)
    return u / np.abs(v) ** (1 / beta)


class ParticleSwarm(BaseOptimizer):
    """
    Parameters (besides those of BaseOptimizer)
    -------------------------------------------
    swarm_size  : number of particles
    n_slots     : slots per particle (None: k_max with 'threshold', n_picks otherwise)
    w, c1, c2   : inertia, cognitive and social coefficients
    v_max_frac  : max speed per coordinate, as a fraction of its range
    levy        : add Lévy-flight steps (see the module docstring)
    p_levy      : probability of a Lévy step per particle and iteration
    levy_scale  : size of a Lévy step, as a fraction of each coordinate's range
    levy_beta   : index of the Lévy distribution (1 < beta <= 2)
    restart_after : iterations without improvement of the global best
                  before every particle except the best one is re-drawn at
                  random (positions, speeds and personal bests); None
                  disables it. Default 50 (chosen on the development CDP:
                  without it the swarm stays on its first good solution)
    """

    def __init__(self, traces, offsets, dt_ms,
                 vel_min=1400.0, vel_max=3000.0,
                 n_picks=10, max_iter=10**9, seed=42,
                 swarm_size=30, n_slots=None, w=0.72, c1=1.49, c2=1.49,
                 v_max_frac=0.2, levy=False, p_levy=0.2, levy_scale=0.01, levy_beta=1.5,
                 restart_after=50,
                 max_evals=None, min_sep_ms=0.0, t_min_ms=0.0,
                 t_max_ms=None, **formulation):

        if max_evals is None or max_evals < 1:
            raise ValueError("ParticleSwarm needs a positive max_evals budget.")
        if swarm_size < 2:
            raise ValueError("swarm_size must be >= 2.")
        if not 1.0 < levy_beta <= 2.0:
            raise ValueError("levy_beta must be in (1, 2].")

        super().__init__(traces, offsets, dt_ms,
                         vel_min, vel_max,
                         n_picks, max_iter, seed, max_evals,
                         min_sep_ms, t_min_ms, t_max_ms, **formulation)

        if not self.variable_k:
            # with a fixed K the repair could leave fewer than n_picks picks
            # (an invalid solution that is never evaluated): v3 only
            raise ValueError("ParticleSwarm supports only objective='threshold' (formulation v3).")
        default_slots = self.k_max if self.variable_k else self.n_picks
        self.n_slots    = int(default_slots if n_slots is None else n_slots)
        if self.variable_k and self.n_slots < self.k_min:
            raise ValueError("n_slots must be >= k_min.")
        if not self.variable_k and self.n_slots != self.n_picks:
            raise ValueError("With a fixed K, n_slots must equal n_picks.")
        self.swarm_size = int(swarm_size)
        self.w, self.c1, self.c2 = float(w), float(c1), float(c2)
        self.v_max_frac = float(v_max_frac)
        self.levy       = bool(levy)
        self.p_levy     = float(p_levy)
        self.levy_scale = float(levy_scale)
        self.levy_beta  = float(levy_beta)
        self.restart_after = None if restart_after is None else int(restart_after)

        # Bounds of the three coordinates of a slot: time, velocity, presence
        self._lo = np.array([self.t_min, self.vel_min, 0.0])
        self._hi = np.array([self.t_max, self.vel_max, 1.0])
        self._range = self._hi - self._lo

        self._current_picks = None
        self._current_score = -np.inf

    # ------------------------------------------------------------------ #
    # Encoding                                                             #
    # ------------------------------------------------------------------ #

    def _encode(self, picks):
        """Position (n_slots, 3) holding a valid solution, sorted in time."""
        x = np.empty((self.n_slots, 3))
        x[:, 0] = self.rng.uniform(self.t_min, self.t_max, self.n_slots)
        x[:, 1] = self.rng.uniform(self.vel_min, self.vel_max, self.n_slots)
        x[:, 2] = 0.25
        for i, (t, v) in enumerate(picks[:self.n_slots]):
            x[i] = (t, v, 0.75)
        return x[np.argsort(x[:, 0], kind='stable')]

    def _repair(self, picks):
        """
        Largest valid subset of picks (sorted in time): the longest chain in
        which every consecutive pair respects min_sep and the velocity rule
        (dynamic programming over at most n_slots picks). Dropping picks
        greedily from the left would let one bad early pick (e.g. a very
        high velocity) remove every pick after it.
        """
        n = len(picks)
        if n <= 1:
            return list(picks)
        g = self._dv_per_sample
        length = [1] * n
        prev = [-1] * n
        for j in range(n):
            tj, vj = picks[j]
            for i in range(j):
                ti, vi = picks[i]
                if (tj - ti >= self.min_sep and vj - vi >= g * (tj - ti)
                        and length[i] + 1 > length[j]):
                    length[j], prev[j] = length[i] + 1, i
        j = int(np.argmax(length))
        chain = []
        while j >= 0:
            chain.append(picks[j])
            j = prev[j]
        return chain[::-1]

    def decode(self, x):
        """Valid list of picks [(t_sample, velocity)] from a position."""
        if self.variable_k:
            active = np.nonzero(x[:, 2] > 0.5)[0]
            if len(active) < self.k_min:
                active = np.argsort(-x[:, 2], kind='stable')[:self.k_min]
        else:
            active = np.arange(self.n_slots)
        picks = sorted((int(round(np.clip(x[i, 0], self.t_min, self.t_max))),
                        float(np.clip(x[i, 1], self.vel_min, self.vel_max))) for i in active)
        picks = self._repair(picks)
        if self.variable_k:
            if len(picks) < self.k_min:            # the repair can only drop picks
                picks = picks[:1] or [(int(round(x[active[0], 0])), float(x[active[0], 1]))]
            picks = picks[:self.k_max]
        return picks

    # ------------------------------------------------------------------ #
    # Swarm                                                                #
    # ------------------------------------------------------------------ #

    def _evaluate_particle(self, i):
        """Decode and evaluate particle i; update its personal best and the global best."""
        picks = self.decode(self.x[i])
        if not self._is_valid(picks):          # should not happen: decode repairs
            return
        score = self._evaluate(picks)
        self._update_best(picks, score)
        self._current_picks, self._current_score = picks, score
        if score > self.pbest_score[i]:
            self.pbest_score[i] = score
            self.pbest[i] = self.x[i].copy()
            if score > self.gbest_score:
                self.gbest_score = score
                self.gbest = self.x[i].copy()

    def _sort_slots(self, i):
        """Keep the slots of particle i sorted by time (with their velocities)."""
        order = np.argsort(self.x[i][:, 0], kind='stable')
        self.x[i] = self.x[i][order]
        self.vel[i] = self.vel[i][order]

    def _initialize(self):
        self.x = np.array([self._encode(self._random_picks()) for _ in range(self.swarm_size)])
        vmax = self.v_max_frac * self._range
        self.vel = self.rng.uniform(-1.0, 1.0, self.x.shape) * vmax
        self.pbest = self.x.copy()
        self.pbest_score = np.full(self.swarm_size, -np.inf)
        self.gbest = self.x[0].copy()
        self.gbest_score = -np.inf
        self.iteration = 0
        self._stagnant = 0
        self.stats = {'levy_steps': 0, 'restarts': 0}
        for i in range(self.swarm_size):
            if self.budget_exhausted:
                break
            self._evaluate_particle(i)
        self._last_gbest = self.gbest_score

    def _iterate(self):
        """One synchronous PSO iteration (every particle moves and is evaluated)."""
        vmax = self.v_max_frac * self._range
        for i in range(self.swarm_size):
            if self.budget_exhausted:
                break
            r1 = self.rng.random(self.x[i].shape)
            r2 = self.rng.random(self.x[i].shape)
            self.vel[i] = (self.w * self.vel[i]
                           + self.c1 * r1 * (self.pbest[i] - self.x[i])
                           + self.c2 * r2 * (self.gbest - self.x[i]))
            self.vel[i] = np.clip(self.vel[i], -vmax, vmax)
            self.x[i] = self.x[i] + self.vel[i]
            if self.levy and self.rng.random() < self.p_levy:
                step = levy_steps(self.rng, self.x[i].shape, self.levy_beta)
                self.x[i] = self.x[i] + self.levy_scale * step * self._range
                self.stats['levy_steps'] += 1
            self.x[i] = np.clip(self.x[i], self._lo, self._hi)
            self._sort_slots(i)
            self._evaluate_particle(i)
        self.iteration += 1
        self._stagnant = 0 if self.gbest_score > self._last_gbest else self._stagnant + 1
        self._last_gbest = self.gbest_score
        if self.restart_after is not None and self._stagnant >= self.restart_after:
            self._restart()

    def _restart(self):
        """Re-draw every particle except the one at the global best."""
        keep = int(np.argmax(self.pbest_score))
        vmax = self.v_max_frac * self._range
        for i in range(self.swarm_size):
            if i == keep or self.budget_exhausted:
                continue
            self.x[i] = self._encode(self._random_picks())
            self.vel[i] = self.rng.uniform(-1.0, 1.0, self.x[i].shape) * vmax
            self.pbest_score[i] = -np.inf
            self._evaluate_particle(i)
        self._stagnant = 0
        self.stats['restarts'] += 1

    def _record_extra(self):
        """Trajectory fields of PSO."""
        if not hasattr(self, 'x'):
            return {}
        k = (self.x[:, :, 2] > 0.5).sum(axis=1) if self.variable_k else None
        return {'iteration': int(getattr(self, 'iteration', 0)),
                'gbest_score': float(self.gbest_score),
                'mean_active_slots': None if k is None else float(k.mean()),
                'swarm_spread': float(np.mean(np.std(self.x[:, :, :2], axis=0) / self._range[:2]))}

    def run(self):
        """Global-best PSO until the budget is used up. self.history: best score per iteration."""
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


class ParticleSwarmLevy(ParticleSwarm):
    """ParticleSwarm with Lévy flights on by default (a separate name for batch configs)."""

    def __init__(self, traces, offsets, dt_ms, levy=True, **kwargs):
        super().__init__(traces, offsets, dt_ms, levy=levy, **kwargs)