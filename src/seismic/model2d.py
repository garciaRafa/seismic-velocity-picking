"""
src/seismic/model2d.py
----------------------
2D layered velocity models built from a simple description: a stack of
interfaces, each one a straight line (optionally dipping) that can be
offset by a vertical normal fault. Used as input for finite-difference
modeling (src/seismic/fd_modeling.py).

Pure NumPy — building and inspecting a model does not need Devito.

Model description (see configs/synthetic_fd_model.json):

    width_m, depth_m, dx_m       grid size and spacing (dx = dz)
    velocities                   one per layer, top to bottom
                                 (len = len(interfaces) + 1)
    interfaces                   top to bottom, each with
        z0_m      depth at x_ref_m
        dip_deg   dip, positive = deeper to the right
        x_ref_m   reference position of z0_m
        fault     optional {"x_m", "throw_m"}: to the right of x_m the
                  interface is shifted down by throw_m

Array convention: vp[ix, iz] (x first), the layout used by Devito.
"""

import numpy as np

from .synthetic import depth_to_time, resample_to_regular_time


def interface_depths(interface, x):
    """
    Depth (m) of one interface at the horizontal positions x (m).

    z(x) = z0 + tan(dip) * (x - x_ref)  [+ throw for x >= x_fault]
    """
    x  = np.asarray(x, dtype=float)
    z0 = float(interface['z0_m'])
    dip = np.deg2rad(float(interface.get('dip_deg', 0.0)))
    x_ref = float(interface.get('x_ref_m', 0.0))
    z = z0 + np.tan(dip) * (x - x_ref)
    fault = interface.get('fault')
    if fault:
        z = z + np.where(x >= float(fault['x_m']), float(fault['throw_m']), 0.0)
    return z


def build_layered_model(model_cfg):
    """
    Build the P-wave velocity grid of a layered model.

    Returns
    -------
    dict with
        vp         : (nx, nz) float32 — velocity in m/s, vp[ix, iz]
        x, z       : 1D axes in m
        dx         : grid spacing in m
        interfaces : (n_interfaces, nx) depth of each interface along x
        names      : interface names
    """
    width, depth, dx = (float(model_cfg[k]) for k in ('width_m', 'depth_m', 'dx_m'))
    vels = [float(v) for v in model_cfg['velocities']]
    ifaces = model_cfg['interfaces']
    if len(vels) != len(ifaces) + 1:
        raise ValueError("Need one velocity per layer: len(velocities) == "
                         "len(interfaces) + 1.")

    x = np.arange(int(round(width / dx)) + 1) * dx
    z = np.arange(int(round(depth / dx)) + 1) * dx
    depths = np.array([interface_depths(itf, x) for itf in ifaces])

    # Interfaces must not cross: each one stays below the previous one
    if np.any(np.diff(depths, axis=0) <= 0):
        raise ValueError("Interfaces cross each other: check depths, dips "
                         "and fault throws.")

    # Layer index of each grid point = number of interfaces above it
    layer = (z[None, None, :] >= depths[:, :, None]).sum(axis=0)   # (nx, nz)
    vp = np.asarray(vels, dtype=np.float32)[layer]

    return {
        'vp': vp, 'x': x, 'z': z, 'dx': dx,
        'interfaces': depths,
        'names': [itf.get('name', f'interface {i}') for i, itf in enumerate(ifaces)],
    }


def column_vrms(model, x_m, dt_ms, t_max_ms):
    """
    RMS velocity on a regular time grid for the vertical column under
    position x_m — the 1D ground truth used for a CDP at that position.

    With dipping interfaces the best stacking (NMO) velocity of an event
    is slightly higher than this column V_rms (about V / cos(dip) for a
    single dipping layer), so for dipping events it is an approximation.

    Returns
    -------
    t_grid : time axis in ms
    v_rms  : RMS velocity on t_grid (m/s)
    """
    ix = int(round((x_m - model['x'][0]) / model['dx']))
    if not 0 <= ix < model['vp'].shape[0]:
        raise ValueError(f"x = {x_m} m is outside the model.")
    profile = model['vp'][ix].astype(np.float64)
    t0_ms, v_rms_depth = depth_to_time(profile, model['dx'])
    return resample_to_regular_time(t0_ms, v_rms_depth, dt_ms, t_max_ms)


def column_event_times(model, x_m):
    """
    Zero-offset two-way time (ms) of each interface under position x_m,
    and the RMS velocity down to it — the primary reflections expected in
    a CDP gather at x_m (flat-layer approximation).

    Returns
    -------
    list of dicts with name, depth_m, t0_ms, v_rms
    """
    ix = int(round((x_m - model['x'][0]) / model['dx']))
    profile = model['vp'][ix].astype(np.float64)
    dz = model['dx']
    t_one_way = np.cumsum(dz / profile)                       # s, at each depth sample
    events = []
    for name, z_itf in zip(model['names'], model['interfaces'][:, ix]):
        iz = int(round(z_itf / dz)) - 1                       # last sample above it
        t1 = t_one_way[iz]
        v_rms = np.sqrt(np.sum(profile[:iz + 1] ** 2 * dz / profile[:iz + 1]) / t1)
        events.append({'name': name, 'depth_m': float(z_itf),
                       't0_ms': float(2000.0 * t1), 'v_rms': float(v_rms)})
    return events


def surface_multiples(primaries, n_water=3):
    """
    Expected zero-offset time and NMO velocity of the main free-surface
    multiples, for interpretation of the panels:

    - water-bottom multiples of order 2..n_water: t0 = n * t_wb, at the
      water velocity (they travel only in the water layer)
    - first-order peg-legs of every deeper reflector: the primary plus
      one extra bounce in the water layer, t0 = t_k + t_wb, with the RMS
      velocity of the combined path

    `primaries` is the output of column_event_times; the first entry must
    be the water bottom.
    """
    wb = primaries[0]
    t_wb, v_w = wb['t0_ms'], wb['v_rms']
    mult = [{'name': f'water-bottom multiple x{n}', 't0_ms': n * t_wb, 'v_nmo': v_w}
            for n in range(2, n_water + 1)]
    for p in primaries[1:]:
        t0 = p['t0_ms'] + t_wb
        v = np.sqrt((p['v_rms'] ** 2 * p['t0_ms'] + v_w ** 2 * t_wb) / t0)
        mult.append({'name': f"peg-leg of {p['name']}", 't0_ms': t0, 'v_nmo': float(v)})
    return mult