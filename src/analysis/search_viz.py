"""
src/analysis/search_viz.py
--------------------------
Figures of a search trajectory, recorded with record_trajectory=True
(src/optimizers/base.py). They show HOW an algorithm moves through the
search space, not only where it ends:

    plot_score_trace  current and best objective along the run
    plot_n_picks      number of picks of the current solution along the run
    plot_snapshots    current solution over the semblance panel at a few moments
    plot_proposals    density of the new picks the algorithm evaluated
    animate_search    GIF of the current solution moving over the panel

Every function takes the optimizer after run(), the semblance panel
(n_vel, n_samples) and the extent [v_min, v_max, t_max, t_min] used with
imshow. Times in the trajectory are sample indices; dt_ms converts them.
"""

import numpy as np
import matplotlib.pyplot as plt


def _check(opt):
    if not getattr(opt, 'trajectory', None):
        raise ValueError("No trajectory: create the optimizer with record_trajectory=True.")


def _panel(ax, panel, extent):
    ax.imshow(panel.T, aspect='auto', cmap='jet', vmin=0, vmax=1, extent=extent)
    ax.set_xlabel('Velocity (m/s)')


def _draw_picks(ax, picks, dt_ms, **style):
    if not picks:
        return
    t = np.array([p[0] for p in picks]) * dt_ms
    v = np.array([p[1] for p in picks])
    ax.plot(v, t, '-', color=style.get('color', 'white'), lw=0.8, alpha=0.7)
    ax.plot(v, t, style.get('marker', 'o'), color=style.get('color', 'white'),
            mec='black', ms=style.get('ms', 7))


def restart_evals(opt):
    """Evaluation counts at which a restart happened (Hill Climbing)."""
    return [r['n_evals'] for r in opt.trajectory if r['event'] == 'restart']


def plot_score_trace(opt, ax=None, show_restarts=True):
    """
    Current-solution objective (thin line) and best-so-far (thick line)
    against evaluations. A current curve that drops shows accepted worse
    moves (Tabu Search); vertical lines mark restarts (Hill Climbing).
    """
    _check(opt)
    ax = ax or plt.gca()
    ev = [r['n_evals'] for r in opt.trajectory]
    ax.plot(ev, [r['current_score'] for r in opt.trajectory], lw=0.7,
            color='tab:blue', label='current solution')
    ax.plot(ev, [r['best_score'] for r in opt.trajectory], lw=2.0,
            color='tab:red', label='best so far')
    if show_restarts:
        for i, e in enumerate(restart_evals(opt)):
            ax.axvline(e, color='gray', lw=0.5, alpha=0.5,
                       label='restart' if i == 0 else None)
    ax.set_xlabel('Objective evaluations')
    ax.set_ylabel('f')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def plot_n_picks(opt, ax=None):
    """Number of picks K of the current solution along the run (formulation v3)."""
    _check(opt)
    ax = ax or plt.gca()
    ax.step([r['n_evals'] for r in opt.trajectory],
            [len(r['current']) for r in opt.trajectory], where='post', lw=1.0)
    ax.set_xlabel('Objective evaluations')
    ax.set_ylabel('K (current solution)')
    ax.grid(True, alpha=0.3)
    return ax


def snapshot_indices(opt, n=6):
    """n trajectory records spread over the run (always the first and last)."""
    return np.unique(np.linspace(0, len(opt.trajectory) - 1, n).round().astype(int))


def plot_snapshots(opt, panel, extent, dt_ms, n=6, reference=None, axes=None):
    """
    Current solution (white) and best so far (black X) over the panel at
    n moments of the run. reference: optional list of reference picks
    (src.analysis.reference) drawn as '+'.
    """
    _check(opt)
    idx = snapshot_indices(opt, n)
    if axes is None:
        _, axes = plt.subplots(1, len(idx), figsize=(3.2 * len(idx), 6), sharey=True)
    axes = np.atleast_1d(axes)
    best_picks = None
    for ax, i in zip(axes, idx):
        rec = opt.trajectory[i]
        # best solution at that moment = current of the last record whose
        # current score equals the best score up to there
        for r in opt.trajectory[:i + 1]:
            if r['current_score'] >= r['best_score'] - 1e-12:
                best_picks = r['current']
        _panel(ax, panel, extent)
        if reference:
            for ref in reference:
                ax.plot(ref['v'], ref['t_ms'], '+', ms=12, mew=2,
                        color='white' if ref['kind'] == 'primary' else 'magenta')
        if best_picks and best_picks != rec['current']:
            _draw_picks(ax, best_picks, dt_ms, color='black', marker='X', ms=6)
        _draw_picks(ax, rec['current'], dt_ms)
        ax.set_title(f"evals {rec['n_evals']}\nf = {rec['current_score']:.2f} "
                     f"(best {rec['best_score']:.2f})", fontsize=9)
    axes[0].set_ylabel('Time (ms)')
    return axes


def plot_proposals(opt, panel, extent, dt_ms, ax=None, v_bin=20.0, t_bin_ms=10.0,
                   reference=None):
    """
    Where the algorithm looked: 2D histogram (log scale) of every new pick
    it evaluated, over the semblance panel (gray). Dense regions are where
    the search spent its evaluations; empty regions were never tried.
    """
    _check(opt)
    ax = ax or plt.gca()
    prop = opt.proposals
    ax.imshow(panel.T, aspect='auto', cmap='gray', vmin=0, vmax=1, extent=extent)
    v_edges = np.arange(extent[0], extent[1] + v_bin, v_bin)
    t_edges = np.arange(extent[3], extent[2] + t_bin_ms, t_bin_ms)
    if len(prop):
        H, _, _ = np.histogram2d(prop[:, 1], prop[:, 0] * dt_ms, bins=[v_edges, t_edges])
        H = np.ma.masked_equal(H, 0)
        mesh = ax.pcolormesh(v_edges, t_edges, np.ma.log10(H.T), cmap='hot', alpha=0.75,
                             shading='flat')
        plt.colorbar(mesh, ax=ax, label='log10(new picks evaluated)')
    if reference:
        for ref in reference:
            ax.plot(ref['v'], ref['t_ms'], '+', ms=12, mew=2,
                    color='cyan' if ref['kind'] == 'primary' else 'magenta')
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'{len(prop)} new picks evaluated')
    return ax


def animate_search(opt, panel, extent, dt_ms, path, max_frames=200, fps=10,
                   reference=None, title=''):
    """
    Save a GIF of the current solution moving over the semblance panel.
    The trajectory is subsampled to at most max_frames frames (restarts are
    always kept, so the jumps of Hill Climbing are visible). Returns path.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    _check(opt)
    traj = opt.trajectory
    keep = set(np.linspace(0, len(traj) - 1, max_frames).round().astype(int))
    keep |= {i for i, r in enumerate(traj) if r['event'] in ('start', 'restart')}
    frames = sorted(keep)

    fig, ax = plt.subplots(figsize=(5, 7))
    _panel(ax, panel, extent)
    ax.set_ylabel('Time (ms)')
    if reference:
        for ref in reference:
            ax.plot(ref['v'], ref['t_ms'], '+', ms=12, mew=2,
                    color='white' if ref['kind'] == 'primary' else 'magenta')
    line, = ax.plot([], [], '-', color='white', lw=0.8, alpha=0.7)
    pts,  = ax.plot([], [], 'o', color='white', mec='black', ms=7)
    best, = ax.plot([], [], 'X', color='black', ms=6)
    label = ax.set_title('')
    best_picks = []

    def update(k):
        nonlocal best_picks
        rec = traj[frames[k]]
        if rec['current_score'] >= rec['best_score'] - 1e-12:
            best_picks = rec['current']
        t = np.array([p[0] for p in rec['current']]) * dt_ms
        v = np.array([p[1] for p in rec['current']])
        line.set_data(v, t)
        pts.set_data(v, t)
        pts.set_color('yellow' if rec['event'] == 'restart' else 'white')
        if best_picks:
            best.set_data([p[1] for p in best_picks], [p[0] * dt_ms for p in best_picks])
        label.set_text(f"{title}\nevals {rec['n_evals']}  f = {rec['current_score']:.2f}"
                       f"  best = {rec['best_score']:.2f}")
        return line, pts, best, label

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path