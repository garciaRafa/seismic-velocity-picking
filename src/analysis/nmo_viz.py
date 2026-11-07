"""
src/analysis/nmo_viz.py
-----------------------
Qualitative check of a set of picks by NMO correction, as an interpreter
does it: the picks define a velocity function (linear between picks), the
gather is corrected with it, and

    - a primary picked at the right velocity becomes flat;
    - an event picked too slow is over-corrected (curves up), too fast is
      under-corrected (curves down);
    - an event that was not picked takes the velocity interpolated from
      its neighbors and stays curved if that velocity is wrong — for a
      surface multiple below a primary it curves down (the multiple is
      slower than the primary trend).

The same correction with the reference velocity function (semblance peaks
of the primaries) is shown next to it, and the two stacks side by side.

    nmo_correct_picks  corrected gather, live samples and stack of one pick set
    plot_nmo_check     one row of four panels: gather with the picked
                       moveout curves, NMO with the picks, NMO with the
                       reference, and the two stacks
"""

import numpy as np

from src.seismic.nmo import apply_nmo_variable, stack_live, velocity_function

KIND_COLOR = {'primary': 'cyan', 'multiple': 'magenta', None: 'red'}


def nmo_correct_picks(gather, offsets, dt_ms, times_ms, vels, stretch_mute=0.5):
    """
    NMO-correct a gather with the velocity function of a pick set.

    Returns (corrected, live, stack), or (None, None, None) without picks.
    """
    if len(times_ms) == 0:
        return None, None, None
    t0 = dt_ms * np.arange(gather.shape[1])
    v_t0 = velocity_function(times_ms, vels, t0)
    corrected, live = apply_nmo_variable(gather, offsets, v_t0, dt_ms, stretch_mute=stretch_mute)
    return corrected, live, stack_live(corrected, live)


def _seismic(ax, data, offsets, t_max, clip):
    ax.imshow(data.T, aspect='auto', cmap='gray', vmin=-clip, vmax=clip,
              extent=[offsets[0], offsets[-1], t_max, 0.0])
    ax.set_xlabel('Offset (m)')


def _ticks(ax, times_ms, colors, offsets):
    """Short marks on both edges of a corrected gather at the picked times."""
    span = offsets[-1] - offsets[0]
    for t, c in zip(times_ms, colors):
        ax.plot([offsets[0], offsets[0] + 0.06 * span], [t, t], color=c, lw=2.5)
        ax.plot([offsets[-1] - 0.06 * span, offsets[-1]], [t, t], color=c, lw=2.5)


def plot_nmo_check(axes, gather, offsets, dt_ms, times_ms, vels, reference=None, kinds=None,
                   stretch_mute=0.5, clip_pct=99.0, t_max_ms=None):
    """
    Draw one NMO check row on four axes:

      [0] gather with the moveout curve of every pick (color = kind)
      [1] gather NMO-corrected with the picked velocity function
      [2] gather NMO-corrected with the reference (primaries only)
      [3] stacks: picks (blue) and reference (black, dashed)

    times_ms, vels : the picks; kinds : 'primary' / 'multiple' / None for
    each pick (from src.analysis.reference.match_reference), used only for
    the colors; reference : list of reference picks (kind, t_ms, v).
    Returns {'picks': (corrected, live, stack), 'reference': (...)}.
    """
    times_ms = np.asarray(times_ms, dtype=float)
    vels = np.asarray(vels, dtype=float)
    kinds = list(kinds) if kinds is not None else ['primary'] * len(times_ms)
    colors = [KIND_COLOR.get(k, 'red') for k in kinds]
    n_t = gather.shape[1]
    t_axis = dt_ms * np.arange(n_t)
    t_max = t_axis[-1] if t_max_ms is None else t_max_ms
    clip = np.percentile(np.abs(gather), clip_pct) or 1.0
    x = np.linspace(offsets[0], offsets[-1], 100)

    # [0] gather + picked moveout curves
    _seismic(axes[0], gather, offsets, t_axis[-1], clip)
    for t0, v, c in zip(times_ms, vels, colors):
        axes[0].plot(x, np.sqrt(t0 ** 2 + (1000.0 * x / v) ** 2), color=c, lw=1.2)
    axes[0].set_ylim(t_max, 0)
    axes[0].set_title('Gather + picked moveout', fontsize=9)

    # [1] NMO with the picks
    out = {'picks': nmo_correct_picks(gather, offsets, dt_ms, times_ms, vels, stretch_mute)}
    if out['picks'][0] is not None:
        _seismic(axes[1], out['picks'][0], offsets, t_axis[-1], clip)
        _ticks(axes[1], times_ms, colors, offsets)
    axes[1].set_ylim(t_max, 0)
    axes[1].set_title('NMO, picked velocities', fontsize=9)

    # [2] NMO with the reference primaries
    prim = [r for r in (reference or []) if r['kind'] == 'primary']
    if prim:
        rt = np.array([r['t_ms'] for r in prim])
        rv = np.array([r['v'] for r in prim])
        out['reference'] = nmo_correct_picks(gather, offsets, dt_ms, rt, rv, stretch_mute)
        _seismic(axes[2], out['reference'][0], offsets, t_axis[-1], clip)
        _ticks(axes[2], rt, ['cyan'] * len(rt), offsets)
    else:
        out['reference'] = (None, None, None)
        axes[2].text(0.5, 0.5, 'no reference', ha='center', va='center',
                     transform=axes[2].transAxes)
    axes[2].set_ylim(t_max, 0)
    axes[2].set_title('NMO, reference velocities', fontsize=9)

    # [3] stacks, same scale
    stacks = [s for s in (out['picks'][2], out['reference'][2]) if s is not None]
    scale = max([np.max(np.abs(s)) for s in stacks] + [1e-12])
    if out['reference'][2] is not None:
        axes[3].plot(out['reference'][2] / scale, t_axis, 'k--', lw=0.9, label='reference')
    if out['picks'][2] is not None:
        axes[3].plot(out['picks'][2] / scale, t_axis, color='tab:blue', lw=0.9, label='picks')
    for t, c in zip(times_ms, colors):
        axes[3].axhline(t, color=c, lw=0.6, alpha=0.8)
    axes[3].set_xlim(-1.1, 1.1)
    axes[3].set_ylim(t_max, 0)
    axes[3].set_xticks([])
    axes[3].set_title('Stack', fontsize=9)
    axes[3].legend(fontsize=7, loc='lower right')
    return out