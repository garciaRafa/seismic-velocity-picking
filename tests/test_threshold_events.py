"""Tests for robust_threshold, semblance_along_curve and select_events."""

import numpy as np
import pytest

from src.analysis.events import select_events
from src.seismic.semblance import (robust_threshold, semblance_along_curve,
                                   semblance_at_pick)


# ---------------------------------------------------------------------- #
# robust_threshold                                                        #
# ---------------------------------------------------------------------- #

def test_robust_threshold_formula():
    # median = 3, |x - 3| = [2, 1, 0, 1, 2] -> MAD = 1
    panel = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert robust_threshold(panel, factor=2.0) == pytest.approx(3.0 + 2 * 1.4826)


def test_robust_threshold_ignores_a_small_ridge():
    """A few strong values (the ridge) barely move the threshold."""
    rng = np.random.default_rng(0)
    background = rng.uniform(0.0, 0.05, size=10000)
    with_ridge = np.concatenate([background, np.full(100, 0.9)])  # 1% ridge
    a = robust_threshold(background, factor=10)
    b = robust_threshold(with_ridge, factor=10)
    assert abs(a - b) < 0.01
    assert b < 0.5


def test_robust_threshold_accepts_2d_panels():
    panel = np.arange(12, dtype=float).reshape(3, 4)
    assert robust_threshold(panel, factor=1) == robust_threshold(panel.ravel(), factor=1)


# ---------------------------------------------------------------------- #
# semblance_along_curve                                                   #
# ---------------------------------------------------------------------- #

def test_along_curve_matches_pointwise_semblance(layered_gather):
    d = layered_gather
    s = semblance_along_curve(d['gather'], d['offsets'], d['v_rms'], d['dt_ms'])
    for i in (0, 400, 750, 1200):
        assert s[i] == pytest.approx(semblance_at_pick(
            d['gather'], d['offsets'], float(d['v_rms'][i]), i, d['dt_ms']))


def test_along_curve_rejects_wrong_length(layered_gather):
    d = layered_gather
    with pytest.raises(ValueError):
        semblance_along_curve(d['gather'], d['offsets'], d['v_rms'][:-1],
                              d['dt_ms'])


# ---------------------------------------------------------------------- #
# select_events                                                           #
# ---------------------------------------------------------------------- #

def test_select_events_keeps_strongest_and_respects_separation():
    s = np.zeros(100)
    s[[20, 24, 60]] = [0.5, 0.8, 0.6]          # 20 and 24 are too close
    events = select_events(s, tau=0.1, min_sep=10)
    assert events.tolist() == [24, 60]


def test_select_events_ignores_values_below_threshold():
    s = np.zeros(50)
    s[[10, 30]] = [0.05, 0.4]
    assert select_events(s, tau=0.1, min_sep=5).tolist() == [30]


def test_select_events_on_short_or_flat_curves():
    assert select_events(np.array([0.9, 0.9]), tau=0.1, min_sep=1).size == 0
    assert select_events(np.zeros(20), tau=0.1, min_sep=1).size == 0


def test_layered_model_has_exactly_its_two_reflections(layered_gather):
    """Along the true V_rms, the 3-layer model shows its 2 interfaces."""
    d = layered_gather
    s = semblance_along_curve(d['gather'], d['offsets'], d['v_rms'], d['dt_ms'])
    events = select_events(s, tau=0.3, min_sep=50)
    times = events * d['dt_ms']
    assert len(times) == 2
    for found, expected in zip(times, d['event_times_ms']):
        assert abs(found - expected) <= 20.0