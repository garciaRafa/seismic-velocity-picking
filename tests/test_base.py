"""Tests for the shared utilities of src/optimizers/base.py."""

import numpy as np
import pytest

from src.optimizers.hill_climbing import HillClimbing


@pytest.fixture
def optimizer():
    traces  = np.zeros((5, 500), dtype=np.float32)
    offsets = np.linspace(100.0, 3000.0, 5, dtype=np.float32)
    return HillClimbing(traces, offsets, dt_ms=2.0,
                        vel_min=1400.0, vel_max=3000.0, n_picks=3, seed=0)


def test_valid_solution_is_accepted(optimizer):
    assert optimizer._is_valid([(100, 1500.0), (200, 1800.0), (300, 2100.0)])


def test_equal_velocities_are_accepted(optimizer):
    assert optimizer._is_valid([(100, 1500.0), (200, 1500.0), (300, 1500.0)])


def test_decreasing_velocity_is_rejected(optimizer):
    assert not optimizer._is_valid([(100, 1800.0), (200, 1500.0), (300, 2100.0)])


def test_validity_uses_time_order_not_list_order(optimizer):
    # Same valid solution, listed out of time order
    assert optimizer._is_valid([(300, 2100.0), (100, 1500.0), (200, 1800.0)])
    # Monotonic in list order, but not in time order
    assert not optimizer._is_valid([(300, 1500.0), (100, 1800.0), (200, 2100.0)])


def test_duplicate_times_are_rejected(optimizer):
    assert not optimizer._is_valid([(100, 1500.0), (100, 1800.0), (300, 2100.0)])


def test_out_of_bounds_values_are_rejected(optimizer):
    assert not optimizer._is_valid([(100, 1300.0), (200, 1800.0), (300, 2100.0)])
    assert not optimizer._is_valid([(100, 1500.0), (200, 1800.0), (300, 3100.0)])
    assert not optimizer._is_valid([(-1, 1500.0), (200, 1800.0), (300, 2100.0)])
    assert not optimizer._is_valid([(100, 1500.0), (200, 1800.0), (500, 2100.0)])


def test_random_picks_are_always_valid(optimizer):
    for _ in range(200):
        picks = optimizer._random_picks()
        assert len(picks) == 3
        assert optimizer._is_valid(picks)
