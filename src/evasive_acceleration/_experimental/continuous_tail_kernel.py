"""Tail-optimization kernels: same exclusion sets, less scratch work.

These are opt-in helpers; the frozen continuous v5 code is not edited.
A local witness-time search only proposes exact constraints. Acceptance still
requires the original continuous certificate and angular lower/upper gap.
"""
import math
import numpy as np
from .continuous import (
    njit, ContinuousResult, ContinuousBudget, solve_pair, certify as _full_certify, separation,
    _store_time,
)


@njit(cache=True, fastmath=False)
def acceleration_tail_start(state, ax, ay, guard):
    """Analytic circle-separation certificate for every t >= returned time.

    ||r(t)-.5*a*t²|| >= .5*||a||*t² - (vA+vB)*t - ||r(0)||.
    Once the increasing quadratic exceeds both circumradii, no rectangle
    collision can occur again, regardless of headings or yaw rates.
    """
    acceleration = math.hypot(ax, ay)
    if acceleration == 0.:
        return np.inf
    distance = math.hypot(state[7] - state[0], state[8] - state[1])
    radius = .5 * (math.hypot(state[4], state[5]) + math.hypot(state[11], state[12]))
    speed = state[2] + state[9]
    margin = 8. * guard + 1e-9 * (distance + radius + 1.)
    root = (speed + math.sqrt(speed * speed + 2. * acceleration * (distance + radius + margin))) / acceleration
    return np.nextafter(root * (1. + 1e-12) + 1e-12, np.inf)


@njit(cache=True, fastmath=False)
def certify(state, wa, wb, ax, ay, horizon, guard=1e-10, max_nodes=30000):
    # The tail is analytically certified, not ignored or discretized away.
    checked_horizon = min(horizon, acceleration_tail_start(state, ax, ay, guard))
    return _full_certify(state, wa, wb, ax, ay, checked_horizon, guard, max_nodes)



@njit(cache=True, fastmath=False)
def _union_reach(lower, upper, size, cap):
    """Exact connected interval union without sorting or allocating indices."""
    reach = 0.
    while True:
        previous = reach
        for k in range(size):
            if lower[k] <= reach and upper[k] > reach:
                reach = upper[k]
                if reach >= cap:
                    return cap
        if reach == previous:
            return reach

@njit(cache=True, fastmath=False)
def _ray(projection, radii, nx, ny, squared, count, ux, uy, cap, clearance, lower, upper):
    size = 0
    for k in range(count):
        lo, hi = 0., cap
        for j in range(4):
            p, s = projection[k, j], radii[k, j] + clearance
            q = squared[k] * (ux * nx[k, j] + uy * ny[k, j])
            if q == 0.:
                if abs(p) > s:
                    hi = -1.
                    break
            else:
                a, b = (p - s) / q, (p + s) / q
                lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
            if lo > hi:
                break
        if lo <= hi:
            lower[size], upper[size] = lo, hi
            size += 1
    reach = _union_reach(lower, upper, size, cap)
    return np.inf if reach >= cap else np.nextafter(reach, np.inf)

@njit(cache=True, fastmath=False)
def sector_lower_bound(projection, radii, nx, ny, squared, count, theta, half, cap, guard, lower, upper):
    """Radius excluded for EVERY direction in an angular sector.

    Use exact ranges of n dot u over the sector, then intersect the resulting
    constraints in radius. A union of common collision intervals excludes all
    smaller vectors in that sector. Shrinking the bodies makes this a lower
    bound, unlike an unsafe collision-at-incumbent pruning rule.
    """
    ux, uy, ch, sh = math.cos(theta), math.sin(theta), math.cos(half), math.sin(half)
    size = 0
    for k in range(count):
        lo, hi = 0., cap
        for j in range(4):
            p, s = projection[k, j], radii[k, j] - guard
            if s < 0.:
                hi = -1.
                break
            d = ux * nx[k, j] + uy * ny[k, j]
            e = -uy * nx[k, j] + ux * ny[k, j]
            qlo = d * ch - abs(e) * sh
            qhi = d * ch + abs(e) * sh
            norm = math.hypot(nx[k, j], ny[k, j])
            if d >= norm * ch: qhi = norm
            if d <= -norm * ch: qlo = -norm
            err = 64 * np.finfo(np.float64).eps * (norm + 1.)
            qlo = squared[k] * (qlo - err)
            qhi = squared[k] * (qhi + err)
            # a*qlo >= p-s and a*qhi <= p+s, with a >= 0.
            if qlo > 0.: lo = max(lo, (p - s) / qlo)
            elif qlo < 0.: hi = min(hi, (p - s) / qlo)
            elif p - s > 0.: hi = -1.
            if qhi > 0.: hi = min(hi, (p + s) / qhi)
            elif qhi < 0.: lo = max(lo, (p + s) / qhi)
            elif p + s < 0.: hi = -1.
            if lo > hi: break
        if lo <= hi:
            lower[size], upper[size] = lo, hi
            size += 1
    return _union_reach(lower, upper, size, cap)

@njit(cache=True, fastmath=False)
def _refine_witness_time(state, wa, wb, ax, ay, witness, horizon):
    """Local constraint proposal; never used as a safety certificate."""
    center, width = witness, min(.05, horizon / 200.)
    best = separation(state, wa, wb, center, ax, ay)
    for _ in range(5):
        origin = center
        for j in range(9):
            t = max(0., min(horizon, origin - width + j * width / 4.))
            value = separation(state, wa, wb, t, ax, ay)
            if value < best:
                center, best = t, value
        width /= 4.
    return center

