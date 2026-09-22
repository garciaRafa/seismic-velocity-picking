"""
src/seismic/models.py
---------------------
Simple velocity models for controlled experiments and sanity tests.

A layered model has a few thick, flat layers with known velocities, so the
reflection times and the true RMS velocities are easy to predict. It is used
to check that the whole pipeline (modeling, NMO, semblance, optimizers)
behaves as expected before running on Marmousi2.
"""

import numpy as np


def layered_velocity_profile(velocities, thicknesses_m, spacing_m=1.25,
                             total_depth_m=None):
    """
    Build a 1D velocity profile (velocity vs depth) made of flat layers.

    Parameters
    ----------
    velocities    : sequence of float — velocity of each layer in m/s
    thicknesses_m : sequence of float — thickness of each layer in meters,
                    one value per layer except the last, which extends
                    down to total_depth_m (len = len(velocities) - 1)
    spacing_m     : float — depth sampling in meters (default 1.25)
    total_depth_m : float — total depth of the profile in meters; defaults
                    to the sum of the thicknesses plus 1000 m

    Returns
    -------
    profile : np.ndarray, float32, shape (n_depth,)
        Velocity at each depth sample, in m/s.
    """
    velocities    = np.asarray(velocities, dtype=np.float32)
    thicknesses_m = np.asarray(thicknesses_m, dtype=np.float64)

    if len(thicknesses_m) != len(velocities) - 1:
        raise ValueError("thicknesses_m must have len(velocities) - 1 values")

    if total_depth_m is None:
        total_depth_m = float(np.sum(thicknesses_m)) + 1000.0

    n_depth = int(round(total_depth_m / spacing_m))
    depth   = np.arange(n_depth) * spacing_m

    # Depth of the bottom of each layer (the last layer has no bottom)
    interfaces = np.cumsum(thicknesses_m)

    # Layer index of each depth sample
    layer_idx = np.searchsorted(interfaces, depth, side='right')

    return velocities[layer_idx].astype(np.float32)
