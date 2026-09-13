"""
src/seismic/cdp.py
------------------
CDP gather organization from seismic shot records.
"""

import numpy as np
from collections import defaultdict


def sort_cdp_gathers(data, headers):
    """
    Sort seismic traces into CDP gathers.

    Parameters
    ----------
    data    : np.ndarray, shape (n_traces, n_samples)
    headers : dict with at least 'cdp' and 'offset' keys

    Returns
    -------
    cdp_gathers : dict
        keys   : CDP numbers
        values : dict with 'traces' and 'offsets'
    """
    cdp_ids = headers['cdp']
    offsets = headers.get('offset', np.zeros(len(cdp_ids)))

    cdp_dict = defaultdict(lambda: {'traces': [], 'offsets': []})

    for i, cdp_id in enumerate(cdp_ids):
        cdp_dict[int(cdp_id)]['traces'].append(data[i])
        cdp_dict[int(cdp_id)]['offsets'].append(float(offsets[i]))

    # Convert lists to arrays
    gathers = {}
    for cdp_id, content in cdp_dict.items():
        gathers[cdp_id] = {
            'traces':  np.array(content['traces'],  dtype=np.float32),
            'offsets': np.array(content['offsets'], dtype=np.float32),
        }

    return gathers


def get_cdp_gather(gathers, cdp_id):
    """
    Retrieve a single CDP gather.

    Parameters
    ----------
    gathers : dict — output of sort_cdp_gathers
    cdp_id  : int  — CDP number to retrieve

    Returns
    -------
    traces  : np.ndarray, shape (n_traces, n_samples)
    offsets : np.ndarray, shape (n_traces,)
    """
    if cdp_id not in gathers:
        raise KeyError(f"CDP {cdp_id} not found.")

    return gathers[cdp_id]['traces'], gathers[cdp_id]['offsets']


def get_cdp_ids(gathers):
    """Return sorted list of available CDP numbers."""
    return sorted(gathers.keys())


def cdp_coverage(gathers):
    """
    Return coverage statistics across all CDP gathers.

    Returns
    -------
    dict with min, max, mean coverage and total CDPs
    """
    coverages = [len(g['offsets']) for g in gathers.values()]
    return {
        'total_cdps': len(gathers),
        'min_coverage': int(np.min(coverages)),
        'max_coverage': int(np.max(coverages)),
        'mean_coverage': float(np.mean(coverages)),
    }
