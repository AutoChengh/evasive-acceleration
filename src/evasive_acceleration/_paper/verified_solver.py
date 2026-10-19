"""Verified finite-grid EA ray search, keeping the frozen paper SAT geometry.

The reference solver is deliberately left unchanged.  This module repairs its
boundary handling: interval arithmetic locates candidate free gaps, but only
the *actual* frozen SAT predicate can accept a finite acceleration.  In
particular, one ``nextafter`` is not assumed to guarantee separation because
forming displaced coordinates and projecting them can round differently from
the algebra used to calculate an interval endpoint.

The tolerance joins nearby forbidden intervals; it is not an acceleration
offset.  Boundary retries use a floating-point forward-error scale instead.
No binary search or monotonic-collision assumption is used on an entire ray.
"""

import math

import numpy as np

from .numerical_solver import (
    _check_collision_with_extra_accel_support_jit,
    _direction_collision_interval_one_time_jit,
    njit,
    will_collide_with_extra_accel,
)


@njit(cache=True, fastmath=False)
def _roundoff_escape_for_contact(
    relx, rely, axes_x_k, axes_y_k, sum_radius_k, t2, ux, uy, candidate
):
    """Smallest increasing SAT-axis exit with an arithmetic-roundoff margin.

    For each moving axis, the analytic right endpoint is inflated by a bound
    proportional to the magnitudes of its floating-point products and sums.
    This is normally a handful of ulps, rather than the search tolerance.
    The returned proposal still requires an actual collision check.
    """
    eps = np.finfo(np.float64).eps
    best = np.inf
    for j in range(4):
        nx = axes_x_k[j]
        ny = axes_y_k[j]
        s = sum_radius_k[j]
        q = t2 * (ux * nx + uy * ny)
        if q == 0.0:
            continue
        p = relx * nx + rely * ny
        if q > 0.0:
            endpoint = (p + s) / q
        else:
            endpoint = (p - s) / q

        # The SAT predicate first translates x/y, then projects.  Include both
        # product/sum orders in the error scale.  The factor 32 is a
        # conservative rounding budget for these elementary operations, not a
        # physical clearance or an EA-resolution margin.
        scale = (
            abs(relx * nx)
            + abs(rely * ny)
            + abs(candidate * ux * t2 * nx)
            + abs(candidate * uy * t2 * ny)
            + abs(s)
        )
        margin = 32.0 * eps * scale / abs(q)
        acceleration_ulp = np.nextafter(candidate, np.inf) - candidate
        if margin < acceleration_ulp:
            margin = acceleration_ulp
        proposal = max(endpoint, candidate) + margin
        proposal = np.nextafter(proposal, np.inf)
        if proposal > candidate and proposal < best:
            best = proposal
    return best


@njit(cache=True, fastmath=False)
def _verified_candidate_in_gap(
    relx_arr, rely_arr, axes_x, axes_y, sum_radius, t_arr, t2_arr, R2,
    ux, uy, left, right,
):
    """Find a verified point immediately after a forbidden component.

    A failure does not license skipping a possibly nonmonotone ray.  The
    caller continues with subsequent *interval components*, rather than
    assuming that collision at this or a larger acceleration rules out all
    smaller values.  The right endpoint is also tested: floating-point SAT
    may resolve an analytic boundary differently, and a feasible cap must not
    be missed merely because a roundoff retry would have stepped past it.
    """
    candidate = np.nextafter(left, np.inf)
    # At a clipped cap, its exact value can still be feasible under the actual
    # floating-point predicate, so check it rather than imposing a tol margin.
    if candidate > right:
        candidate = right

    for _ in range(relx_arr.shape[0] + 64):
        if not math.isfinite(candidate) or candidate > right:
            return np.nan
        collision, collision_time = _check_collision_with_extra_accel_support_jit(
            relx_arr, rely_arr, axes_x, axes_y, sum_radius, t_arr, t2_arr,
            candidate * ux, candidate * uy, R2,
        )
        if not collision:
            # Deliberately call the public frozen predicate as the final
            # acceptance gate; no finite return bypasses this test.
            if not will_collide_with_extra_accel(
                relx_arr, rely_arr, axes_x, axes_y, sum_radius, t_arr, t2_arr,
                candidate * ux, candidate * uy, R2,
            ):
                return candidate

        proposal = np.inf
        for k in range(t_arr.shape[0]):
            if t_arr[k] == collision_time:
                proposal = _roundoff_escape_for_contact(
                    relx_arr[k], rely_arr[k], axes_x[k], axes_y[k],
                    sum_radius[k], t2_arr[k], ux, uy, candidate,
                )
                break
        if proposal <= candidate or not math.isfinite(proposal):
            return np.nan
        if proposal > right:
            if candidate < right:
                candidate = right
            else:
                return np.nan
        else:
            candidate = proposal

    # Refuse an unverified value even in pathological ill-conditioned data.
    return np.nan


@njit(cache=True, fastmath=False)
def solve_direction_min_acceleration(
    relx_arr, rely_arr,
    axes_x, axes_y,
    sum_radius,
    t_arr, t2_arr,
    R2,
    ux, uy,
    a_max,
    tol,
):
    """Return a SAT-verified finite-grid ray minimum, or NaN if unresolved.

    The signature matches the frozen paper direction solver.  Collision
    intervals are sorted and their union is traversed from zero.  Small gaps
    of width at most ``tol`` are conservatively joined as in the reference.
    """
    if not will_collide_with_extra_accel(
        relx_arr, rely_arr, axes_x, axes_y, sum_radius, t_arr, t2_arr,
        0.0, 0.0, R2,
    ):
        return 0.0

    count = relx_arr.shape[0]
    lower = np.empty(count, dtype=np.float64)
    upper = np.empty(count, dtype=np.float64)
    interval_count = 0
    for k in range(count):
        has_interval, left, right = _direction_collision_interval_one_time_jit(
            relx_arr[k], rely_arr[k], axes_x[k], axes_y[k], sum_radius[k],
            t2_arr[k], ux, uy, a_max, 0.0,
        )
        if has_interval:
            lower[interval_count] = left
            upper[interval_count] = right
            interval_count += 1

    order = np.argsort(lower[:interval_count])
    reach = 0.0
    cursor = 0
    while True:
        # Compute the entire connected forbidden component before attempting
        # its outgoing gap.  A component that reaches the cap is not rejected
        # merely because it comes within tol of the cap.
        while cursor < interval_count:
            index = order[cursor]
            if lower[index] > reach + tol:
                break
            if upper[index] > reach:
                reach = upper[index]
            cursor += 1

        next_start = a_max
        has_next = cursor < interval_count
        if has_next:
            next_start = lower[order[cursor]]
        candidate = _verified_candidate_in_gap(
            relx_arr, rely_arr, axes_x, axes_y, sum_radius, t_arr, t2_arr, R2,
            ux, uy, reach, next_start,
        )
        if math.isfinite(candidate):
            return candidate
        if not has_next:
            return np.nan

        # If rounding closes this nominal gap, continue after the next
        # exclusion component.  We do not bisect across a nonmonotone union.
        index = order[cursor]
        if upper[index] > reach:
            reach = upper[index]
        cursor += 1


@njit(cache=True, fastmath=False)
def compute_ea_on_directions(
    relx_arr, rely_arr,
    axes_x, axes_y,
    sum_radius,
    t_arr, t2_arr,
    R2,
    ux_arr, uy_arr,
    a_max,
    tol,
):
    """Search every supplied direction and return (EA, index, baseline hit)."""
    baseline_collision = will_collide_with_extra_accel(
        relx_arr, rely_arr, axes_x, axes_y, sum_radius, t_arr, t2_arr,
        0.0, 0.0, R2,
    )
    if not baseline_collision:
        return 0.0, -1, False

    best_value = np.inf
    best_index = -1
    for index in range(ux_arr.shape[0]):
        candidate = solve_direction_min_acceleration(
            relx_arr, rely_arr, axes_x, axes_y, sum_radius, t_arr, t2_arr, R2,
            ux_arr[index], uy_arr[index], a_max, tol,
        )
        if math.isfinite(candidate) and candidate < best_value:
            best_value = candidate
            best_index = index
    return best_value, best_index, True


__all__ = ["solve_direction_min_acceleration", "compute_ea_on_directions"]
