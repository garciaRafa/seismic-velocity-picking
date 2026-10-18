"""
src/seismic/fd_modeling.py
--------------------------
Acoustic finite-difference modeling of shot records on a 2D velocity
model, and their sorting into CDP gathers.

The wave equation (constant density)

    (1 / v^2) d2u/dt2 - laplacian(u) = source

is solved with Devito (https://www.devitoproject.org), which generates
and compiles optimized C code from this symbolic description. Unlike the
1D generator (src/seismic/synthetic.py), which draws one hyperbola per
reflector, the simulation propagates the wavefield through the model, so
every wave phenomenon appears on its own: primaries, free-surface and
interbed multiples, diffractions at the fault edge, head waves and the
direct wave.

Boundaries
    top     free surface (u = 0) when free_surface=True — this is what
            generates the surface multiples; otherwise absorbing
    sides, bottom   absorbing layer of `absorbing_cells` cells

Only the AcousticModeler class needs Devito, and it imports it when it is
created. Everything else here (geometry, resampling, mute, gain) is
plain NumPy, so the rest of the project and its tests run without Devito.
"""

import numpy as np


# ---------------------------------------------------------------------- #
# Wavelet and resampling                                                  #
# ---------------------------------------------------------------------- #

def ricker(t_s, f0_hz, delay_s):
    """Ricker wavelet of peak frequency f0_hz, centered at delay_s."""
    a = (np.pi * f0_hz * (np.asarray(t_s) - delay_s)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def resample_traces(traces, t_in_ms, t_out_ms):
    """
    Linear interpolation of every trace from the time axis t_in_ms (the
    fine finite-difference time step, already corrected for the wavelet
    delay) to t_out_ms (the regular output sampling, e.g. 2 ms). Samples
    outside t_in_ms are set to 0. The modeled signal is band-limited well
    below the output Nyquist frequency (250 Hz at 2 ms), so linear
    interpolation is enough.
    """
    traces = np.atleast_2d(traces)
    out = np.empty((traces.shape[0], len(t_out_ms)), dtype=np.float32)
    for i, tr in enumerate(traces):
        out[i] = np.interp(t_out_ms, t_in_ms, tr, left=0.0, right=0.0)
    return out


# ---------------------------------------------------------------------- #
# Acquisition geometry                                                    #
# ---------------------------------------------------------------------- #

def cdp_geometry(cdp_x, offsets):
    """
    Source and receiver positions that put every offset on the same CDP:
    source at cdp_x - h/2 and receiver at cdp_x + h/2, so the midpoint of
    each pair is cdp_x. One shot per offset.
    """
    offsets = np.asarray(offsets, dtype=float)
    return cdp_x - offsets / 2.0, cdp_x + offsets / 2.0


def receiver_line(width_m, spacing_m, extra_x=()):
    """
    Receiver positions for every shot: a regular line over the whole
    model (for displaying shot records) plus the exact CDP receiver
    positions. Returns (positions, index of each extra_x in positions).
    """
    line = np.arange(0.0, width_m + 1e-9, spacing_m)
    allx = np.unique(np.concatenate([line, np.asarray(extra_x, dtype=float)]))
    idx = np.searchsorted(allx, np.asarray(extra_x, dtype=float))
    return allx, idx


# ---------------------------------------------------------------------- #
# Basic processing of the CDP gather                                      #
# ---------------------------------------------------------------------- #

def direct_wave_mute(gather, offsets, t_ms, velocity, pad_ms, taper_ms=20.0):
    """
    Top mute: zero every sample earlier than offset / velocity + pad_ms,
    with a cosine taper of taper_ms after that line. With the water
    velocity it removes the direct wave, which otherwise is the strongest
    event of a marine record. Uses only the acquisition geometry and the
    water velocity (no model information).
    """
    g = np.array(gather, dtype=np.float32, copy=True)
    for i, h in enumerate(offsets):
        t_cut = 1000.0 * h / velocity + pad_ms
        w = np.clip((t_ms - t_cut) / taper_ms, 0.0, 1.0)
        g[i] *= 0.5 - 0.5 * np.cos(np.pi * w)
    return g


def time_gain(gather, t_ms, power=1.0):
    """
    Multiply each sample by (t / t_max) ** power — a simple correction
    for the amplitude decay with time (geometrical spreading), so deep
    reflections are not hidden by shallow ones in the displays.
    """
    t = np.asarray(t_ms, dtype=np.float32)
    return (np.asarray(gather, dtype=np.float32) * (t / t[-1]) ** power).astype(np.float32)


# ---------------------------------------------------------------------- #
# Finite-difference modeling (needs Devito)                               #
# ---------------------------------------------------------------------- #

class AcousticModeler:
    """
    Acoustic 2D finite-difference modeler for one velocity model and one
    receiver line. The Devito operator is built and compiled once, in
    the constructor, and reused for every shot.

    Parameters
    ----------
    vp             : (nx, nz) velocity in m/s, vp[ix, iz]
    dx             : grid spacing in m (dx = dz)
    rec_x          : receiver x positions (m), the same for every shot
    t_max_ms       : record length after the wavelet-delay correction
    f0_hz          : Ricker peak frequency
    free_surface   : True = free surface on top (surface multiples)
    rec_depth_m, src_depth_m : depth of receivers and sources
    space_order    : spatial order of the finite differences
    absorbing_cells: width of the absorbing layer (sides and bottom; also
                     top when free_surface is False)
    courant        : time step = courant * dx / max(vp)
    """

    def __init__(self, vp, dx, rec_x, t_max_ms, f0_hz=30.0, free_surface=True,
                 rec_depth_m=5.0, src_depth_m=5.0, space_order=8,
                 absorbing_cells=80, courant=0.38):
        from devito import (Grid, TimeFunction, Function, SparseTimeFunction,
                            Eq, Operator, solve, configuration)
        configuration['log-level'] = 'WARNING'

        self.free_surface = bool(free_surface)
        self.f0 = float(f0_hz)
        self.delay_s = 1.0 / self.f0                 # Ricker centered at 1/f0
        self.src_depth = float(src_depth_m)

        nbl = int(absorbing_cells)
        top = 0 if self.free_surface else nbl
        vp = np.asarray(vp, dtype=np.float32)
        nx, nz = vp.shape
        vp_pad = np.pad(vp, ((nbl, nbl), (top, nbl)), mode='edge')

        self.dt_s = courant * dx / float(vp.max())
        t_total = t_max_ms / 1000.0 + self.delay_s
        self.nt = int(np.ceil(t_total / self.dt_s)) + 1

        shape = vp_pad.shape
        grid = Grid(shape=shape,
                    extent=((shape[0] - 1) * dx, (shape[1] - 1) * dx),
                    origin=(-nbl * dx, -top * dx))
        x, z = grid.dimensions
        t = grid.stepping_dim

        self.vp = Function(name='vp', grid=grid, space_order=space_order)
        self.vp.data[:] = vp_pad
        self.damp = Function(name='damp', grid=grid)
        self.damp.data[:] = self._damping(shape, nbl, top, dx)

        self.u = TimeFunction(name='u', grid=grid, time_order=2,
                              space_order=space_order)
        pde = self.u.dt2 / self.vp ** 2 - self.u.laplace + self.damp * self.u.dt
        eqs = [Eq(self.u.forward, solve(pde, self.u.forward))]
        if self.free_surface:
            eqs.append(Eq(self.u[t + 1, x, 0], 0.0))

        self.src = SparseTimeFunction(name='src', grid=grid, npoint=1, nt=self.nt)
        self.rec = SparseTimeFunction(name='rec', grid=grid, npoint=len(rec_x),
                                      nt=self.nt)
        self.rec.coordinates.data[:, 0] = rec_x
        self.rec.coordinates.data[:, 1] = rec_depth_m
        self.rec_x = np.asarray(rec_x, dtype=float)

        dt_sym = grid.stepping_dim.spacing
        eqs += self.src.inject(field=self.u.forward,
                               expr=self.src * dt_sym ** 2 * self.vp ** 2)
        eqs += self.rec.interpolate(expr=self.u)
        self.op = Operator(eqs)

        t_fd = np.arange(self.nt) * self.dt_s
        self.src.data[:, 0] = ricker(t_fd, self.f0, self.delay_s)
        self.t_fd_ms = 1000.0 * (t_fd - self.delay_s)   # 0 = wavelet peak

    @staticmethod
    def _damping(shape, nbl, top, dx):
        """Damping profile of the absorbing layer (as in the Devito examples)."""
        coeff = 1.5 * np.log(1.0 / 0.001) / (nbl * dx)
        damp = np.zeros(shape, dtype=np.float32)
        ramp = (np.arange(nbl, 0, -1) / nbl)                 # 1 at the outer edge
        val = coeff * (ramp - np.sin(2 * np.pi * ramp) / (2 * np.pi))
        damp[:nbl, :] = np.maximum(damp[:nbl, :], val[:, None])
        damp[-nbl:, :] = np.maximum(damp[-nbl:, :], val[::-1][:, None])
        damp[:, -nbl:] = np.maximum(damp[:, -nbl:], val[::-1][None, :])
        if top:
            damp[:, :top] = np.maximum(damp[:, :top], val[None, :top])
        return damp

    def shot(self, src_x, t_out_ms):
        """
        Simulate one shot at src_x and return its record on the receiver
        line, resampled to t_out_ms: array (n_receivers, len(t_out_ms)).
        """
        self.u.data[:] = 0.0
        self.rec.data[:] = 0.0
        self.src.coordinates.data[:] = [[src_x, self.src_depth]]
        self.op(time=self.nt - 2, dt=self.dt_s)
        return resample_traces(self.rec.data.T, self.t_fd_ms, t_out_ms)