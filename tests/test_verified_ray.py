"""Adversarial and analytic checks for verified finite-grid ray candidates."""

import math

import numpy as np
import pytest

from evasive_acceleration._paper import numerical_solver as frozen
from evasive_acceleration._paper import verified_solver as verified


def _axis_aligned_data(relx, times):
    relx = np.asarray(relx, dtype=float)
    times = np.asarray(times, dtype=float)
    n = len(relx)
    axes_x = np.tile([1.0, 0.0, 1.0, 0.0], (n, 1))
    axes_y = np.tile([0.0, 1.0, 0.0, 1.0], (n, 1))
    return (
        relx, np.zeros(n), axes_x, axes_y, np.ones((n, 4)),
        times, 0.5 * times ** 2, 2.0,
    )


def _collides(data, value, ux=1.0, uy=0.0):
    return frozen.will_collide_with_extra_accel(
        *data[:7], value * ux, value * uy, data[7]
    )


def test_one_nextafter_is_not_a_collision_free_certificate():
    zeros = np.zeros(1)
    times = np.array([0.37])
    geometry = frozen.prepare_support_sat_data(
        zeros, zeros, np.array([-2.709076465371118]),
        np.array([0.3460850238112636]), np.array([0.1759457235078592]),
        np.array([-1.1922037918569737]), 4.0, 2.0, 4.0, 2.0,
    )
    data = (*geometry[:5], times, 0.5 * times ** 2, geometry[5])
    theta = -0.5754899058490901
    ux, uy = math.cos(theta), math.sin(theta)
    old = frozen.solve_direction_min_acceleration(*data, ux, uy, 100.0, 0.001)
    assert _collides(data, old, ux, uy)
    repaired = verified.solve_direction_min_acceleration(
        *data, ux, uy, 100.0, 0.001
    )
    assert math.isfinite(repaired)
    assert repaired >= old
    assert repaired - old < 1e-10
    assert not _collides(data, repaired, ux, uy)

    # A cap a few ulps above the computed boundary must still be considered,
    # even if the conservative roundoff proposal would step past that cap.
    tight_cap = old
    for _ in range(100):
        tight_cap = np.nextafter(tight_cap, np.inf)
        if not _collides(data, tight_cap, ux, uy):
            break
    else:
        pytest.fail("Could not construct a feasible tight cap")
    tight_result = verified.solve_direction_min_acceleration(
        *data, ux, uy, tight_cap, 0.001
    )
    assert math.isfinite(tight_result)
    assert tight_result <= tight_cap
    assert not _collides(data, tight_result, ux, uy)


def test_nonmonotone_ray_finds_first_gap_even_when_cap_collides():
    # Forbidden intervals are [0, 2] and [4, 5], with the second clipped at 5.
    # The cap collides although the gap just above 2 is safe.
    data = _axis_aligned_data([0.0, 9.0], [1.0, 2.0])
    assert _collides(data, 5.0)
    result = verified.solve_direction_min_acceleration(*data, 1.0, 0.0, 5.0, 0.001)
    assert 2.0 < result < 4.0
    assert not _collides(data, result)


def test_direction_is_not_pruned_for_colliding_at_an_existing_best_value():
    data = _axis_aligned_data([0.0, 6.0], [1.0, 2.0])
    diagonal = math.sqrt(0.5)
    diagonal_value = verified.solve_direction_min_acceleration(
        *data, diagonal, diagonal, 10.0, 0.001
    )
    # The x ray collides at the diagonal ray's best value, but has a smaller
    # safe gap at (2, 2.5).  Collision at the incumbent is not a valid bound.
    assert _collides(data, diagonal_value)
    value, index, hit = verified.compute_ea_on_directions(
        *data, np.array([diagonal, 1.0]), np.array([diagonal, 0.0]), 10.0, 0.001
    )
    assert hit
    assert index == 1
    assert 2.0 < value < 2.5
    assert value < diagonal_value
    assert not _collides(data, value)


def test_tolerance_does_not_remove_a_feasible_point_near_cap():
    data = _axis_aligned_data([0.0], [1.0])
    result = verified.solve_direction_min_acceleration(
        *data, 1.0, 0.0, 2.0001, 0.001
    )
    assert 2.0 < result <= 2.0001
    assert not _collides(data, result)


@pytest.mark.parametrize("cap", [0.0, 1.0, 2.0])
def test_no_candidate_is_returned_if_cap_only_reaches_contact(cap):
    data = _axis_aligned_data([0.0], [1.0])
    assert math.isnan(verified.solve_direction_min_acceleration(
        *data, 1.0, 0.0, cap, 0.001
    ))


def test_current_contact_cannot_be_removed_by_future_acceleration():
    data = _axis_aligned_data([1.0, 0.0], [0.0, 1.0])
    value, direction, hit = verified.compute_ea_on_directions(
        *data, np.array([1.0, -1.0]), np.zeros(2), 100.0, 0.001
    )
    assert hit
    assert direction == -1
    assert value == np.inf


def test_tiny_nonzero_time_does_not_masquerade_as_zero_ea():
    data = _axis_aligned_data([0.0], [1e-8])
    result = verified.solve_direction_min_acceleration(
        *data, 1.0, 0.0, 100.0, 0.001
    )
    assert math.isnan(result)


def test_separated_baseline_is_zero_and_verified():
    data = _axis_aligned_data([10.0], [1.0])
    result = verified.solve_direction_min_acceleration(
        *data, 1.0, 0.0, 100.0, 0.001
    )
    assert result == 0.0
    assert not _collides(data, result)


def test_every_finite_random_direction_passes_frozen_collision_predicate():
    rng = np.random.default_rng(9041)
    times = np.linspace(0.1, 3.0, 31)
    zeros = np.zeros(len(times))
    finite_count = 0
    for _ in range(30):
        geometry = frozen.prepare_support_sat_data(
            zeros, zeros, rng.normal(0, 0.5, len(times)),
            rng.normal(0, 5, len(times)), rng.normal(0, 2, len(times)),
            rng.normal(0, 0.5, len(times)), 4.0, 2.0, 4.0, 2.0,
        )
        data = (*geometry[:5], times, 0.5 * times ** 2, geometry[5])
        theta = rng.uniform(-math.pi, math.pi)
        ux, uy = math.cos(theta), math.sin(theta)
        result = verified.solve_direction_min_acceleration(
            *data, ux, uy, 100.0, 0.001
        )
        if math.isfinite(result):
            finite_count += 1
            assert 0.0 <= result <= 100.0
            assert not _collides(data, result, ux, uy)
    assert finite_count > 0
