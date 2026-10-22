"""Opt-in continuous-time, angular-bound EA research candidate.

Adaptive time constraints give lower bounds on the continuous problem. A
Taylor swept-OBB certificate supplies a safe upper bound. Angular subdivision
stops only when that gap closes; exhausted budgets are explicitly unresolved.
No released solver, default input adapter or yaw-rate estimator is changed.
"""
from dataclasses import dataclass
import math
import numpy as np

from .._paper import numerical_solver as geometry
from .._paper.config import EAConfig
from .._paper.four_modes import FOUR_PAIRS

njit = geometry.njit


@dataclass(frozen=True)
class ContinuousBudget:
    # The paper's numerical acceleration tolerance. Callers can request 1e-4
    # explicitly; precision is reported as a certified upper/lower bracket.
    acceleration_tolerance: float = .001
    clearance_m: float = 1e-7
    floating_guard_m: float = 1e-10
    initial_time_intervals: int = 20
    initial_sectors: int = 36
    max_angular_nodes: int = 4096
    max_time_constraints: int = 512
    max_exchange_iterations: int = 80
    max_certificate_nodes: int = 30000


@dataclass(frozen=True)
class ContinuousResult:
    value: float
    direction_x: float
    direction_y: float
    lower_bound: float
    continuous_verified: bool
    angular_converged: bool
    iterations: int
    time_constraints: int
    direction_evaluations: int
    sector_nodes: int
    certificate_nodes: int
    status: str


@njit(cache=True, fastmath=False)
def _pose(x, y, v, h, w, t):
    angle = h + w * t
    c, s = math.cos(angle), math.sin(angle)
    if w == 0.:
        return x + v * c * t, y + v * s * t, c, s
    return x + v / w * (s - math.sin(h)), y - v / w * (c - math.cos(h)), c, s


@njit(cache=True, fastmath=False)
def _kinematics(state, wa, wb, t, ax, ay):
    xa, ya, ca, sa = _pose(0., 0., state[2], state[3], wa, t)
    xb, yb, cb, sb = _pose(state[7] - state[0], state[8] - state[1], state[9], state[10], wb, t)
    return (xb - xa - .5 * ax * t * t, yb - ya - .5 * ay * t * t,
            state[9] * cb - state[2] * ca - ax * t,
            state[9] * sb - state[2] * sa - ay * t, ca, sa, cb, sb)


@njit(cache=True, fastmath=False)
def _axis_gap(state, dx, dy, ca, sa, cb, sb, nx, ny):
    da, ea, db, eb = nx * ca + ny * sa, -nx * sa + ny * ca, nx * cb + ny * sb, -nx * sb + ny * cb
    radius = .5 * (state[4] * abs(da) + state[5] * abs(ea) + state[11] * abs(db) + state[12] * abs(eb))
    return abs(dx * nx + dy * ny) - radius


@njit(cache=True, fastmath=False)
def separation(state, wa, wb, t, ax, ay):
    dx, dy, _vx, _vy, ca, sa, cb, sb = _kinematics(state, wa, wb, t, ax, ay)
    return max(_axis_gap(state, dx, dy, ca, sa, cb, sb, ca, sa),
               _axis_gap(state, dx, dy, ca, sa, cb, sb, -sa, ca),
               _axis_gap(state, dx, dy, ca, sa, cb, sb, cb, sb),
               _axis_gap(state, dx, dy, ca, sa, cb, sb, -sb, cb))


@njit(cache=True, fastmath=False)
def _linearized_gap_bound(state, dx, dy, vx, vy, ca, sa, cb, sb, wa, wb, nx, ny, half, second):
    """Lower bound on a fixed-axis separating gap throughout an interval.

    Every pair of box corners has a linear Taylor term. Its minimum over
    time occurs at an interval endpoint. Supports of the linearized rotating
    corner vectors calculate the minimum over ALL corner pairs at each end.
    A second-derivative norm bound covers both translational and rotational
    remainders. No differentiability of abs/support functions is assumed.
    """
    p = dx * nx + dy * ny
    sign = 1. if p >= 0. else -1.
    speed = sign * (vx * nx + vy * ny)
    da, ea = nx * ca + ny * sa, -nx * sa + ny * ca
    db, eb = nx * cb + ny * sb, -nx * sb + ny * cb
    best = np.inf
    for tau in (-half, half):
        support_a = .5 * (state[4] * abs(da + tau * wa * ea) + state[5] * abs(ea - tau * wa * da))
        support_b = .5 * (state[11] * abs(db + tau * wb * eb) + state[12] * abs(eb - tau * wb * db))
        best = min(best, abs(p) + tau * speed - support_a - support_b)
    return best - .5 * second * half * half


@njit(cache=True, fastmath=False)
def certify(state, wa, wb, ax, ay, horizon, guard=1e-10, max_nodes=30000):
    """Return (1=safe, 0=penetration, -1=unresolved, time, visited nodes)."""
    radius_a, radius_b = .5 * math.hypot(state[4], state[5]), .5 * math.hypot(state[11], state[12])
    trans_second = state[2] * abs(wa) + state[9] * abs(wb) + math.hypot(ax, ay)
    second = trans_second + radius_a * wa * wa + radius_b * wb * wb
    for t in (0., horizon):
        if separation(state, wa, wb, t, ax, ay) < -guard:
            return 0, t, 1
    left, right = np.empty(128), np.empty(128)
    left[0], right[0] = 0., horizon
    stack, visited = 1, 0
    while stack:
        stack -= 1
        lo, hi = left[stack], right[stack]
        mid, half = .5 * (lo + hi), .5 * (hi - lo)
        dx, dy, vx, vy, ca, sa, cb, sb = _kinematics(state, wa, wb, mid, ax, ay)
        visited += 1
        if math.hypot(dx, dy) - math.hypot(vx, vy) * half - .5 * trans_second * half * half > radius_a + radius_b + guard:
            continue
        g1 = _axis_gap(state, dx, dy, ca, sa, cb, sb, ca, sa)
        g2 = _axis_gap(state, dx, dy, ca, sa, cb, sb, -sa, ca)
        g3 = _axis_gap(state, dx, dy, ca, sa, cb, sb, cb, sb)
        g4 = _axis_gap(state, dx, dy, ca, sa, cb, sb, -sb, cb)
        if max(g1, g2, g3, g4) < -guard:
            return 0, mid, visited
        passed = False
        for j in range(4):
            nx = ca if j == 0 else -sa if j == 1 else cb if j == 2 else -sb
            ny = sa if j == 0 else ca if j == 1 else sb if j == 2 else cb
            if _linearized_gap_bound(state, dx, dy, vx, vy, ca, sa, cb, sb, wa, wb, nx, ny, half, second) > guard:
                passed = True
                break
        if passed:
            continue
        if half < 1e-11 or visited >= max_nodes or stack + 2 >= 128:
            return -1, mid, visited
        left[stack], right[stack] = lo, mid
        left[stack + 1], right[stack + 1] = mid, hi
        stack += 2
    return 1, np.nan, visited


@njit(cache=True, fastmath=False)
def _store_time(state, wa, wb, t, times, squared, projection, radii, nx, ny, count):
    for k in range(count):
        if abs(times[k] - t) < 1e-11:
            return count
    dx, dy, _vx, _vy, ca, sa, cb, sb = _kinematics(state, wa, wb, t, 0., 0.)
    times[count], squared[count] = t, .5 * t * t
    for j in range(4):
        x = ca if j == 0 else -sa if j == 1 else cb if j == 2 else -sb
        y = sa if j == 0 else ca if j == 1 else sb if j == 2 else cb
        nx[count, j], ny[count, j] = x, y
        projection[count, j] = dx * x + dy * y
        radii[count, j] = .5 * (state[4] * abs(x * ca + y * sa) + state[5] * abs(-x * sa + y * ca)
                                     + state[11] * abs(x * cb + y * sb) + state[12] * abs(-x * sb + y * cb))
    return count + 1


@njit(cache=True, fastmath=False)
def _union_reach(lower, upper, size, cap):
    if size == 0:
        return 0.
    order = np.argsort(lower[:size])
    reach = 0.
    for j in range(size):
        k = order[j]
        if lower[k] > reach:
            break
        reach = max(reach, upper[k])
    return min(reach, cap)


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
def _angular_solve(projection, radii, nx, ny, squared, count, cap, tolerance, clearance, guard, initial, max_nodes):
    lower, upper = np.empty(count), np.empty(count)
    bounds, centers, widths = np.empty(max_nodes + initial + 4), np.empty(max_nodes + initial + 4), np.empty(max_nodes + initial + 4)
    pending, nodes, rays = 0, 0, 0
    best, bx, by = np.inf, np.nan, np.nan
    half = math.pi / initial
    for j in range(initial):
        center = (j + .5) * 2 * math.pi / initial
        bound = sector_lower_bound(projection, radii, nx, ny, squared, count, center, half, cap, guard, lower, upper)
        nodes += 1
        if bound < min(best, cap) - tolerance:
            ux, uy = math.cos(center), math.sin(center)
            value = _ray(projection, radii, nx, ny, squared, count, ux, uy, cap, clearance, lower, upper)
            rays += 1
            if value < best: best, bx, by = value, ux, uy
            bounds[pending], centers[pending], widths[pending] = bound, center, half
            pending += 1
    while pending:
        at = np.argmin(bounds[:pending])
        bound, center, half = bounds[at], centers[at], widths[at]
        pending -= 1
        bounds[at], centers[at], widths[at] = bounds[pending], centers[pending], widths[pending]
        if math.isfinite(best) and best - bound <= tolerance:
            # Previously discarded sectors were pruned at best_at_that_time
            # minus tolerance, not at the smallest STILL-ACTIVE bound. Include
            # that threshold in the reported global lower bound.
            return best, bx, by, min(bound, best - tolerance), True, rays, nodes
        if nodes + 2 > max_nodes:
            return best, bx, by, bound, False, rays, nodes
        for sign in (-1., 1.):
            child, width = center + sign * half / 2., half / 2.
            lb = sector_lower_bound(projection, radii, nx, ny, squared, count, child, width, cap, guard, lower, upper)
            nodes += 1
            if lb < min(best, cap) - tolerance:
                ux, uy = math.cos(child), math.sin(child)
                value = _ray(projection, radii, nx, ny, squared, count, ux, uy, cap, clearance, lower, upper)
                rays += 1
                if value < best: best, bx, by = value, ux, uy
                bounds[pending], centers[pending], widths[pending] = lb, child, width
                pending += 1
    return best, bx, by, min(best, cap) - tolerance, True, rays, nodes


@njit(cache=True, fastmath=False)
def _refine_witness_time(state, wa, wb, ax, ay, witness, horizon):
    """Batch-separation oracle: find a deep contact, not the first entry point.

    A scan only proposes constraints; safety still requires certify(). It can
    miss a very narrow well without licensing a false safe return. Starting
    from the certified penetration witness preserves such narrow contacts.
    """
    center, width = witness, min(.05, horizon / 200.)
    best = separation(state, wa, wb, center, ax, ay)
    for k in range(201):
        t = horizon * k / 200.
        value = separation(state, wa, wb, t, ax, ay)
        if value < best: center, best = t, value
    for _ in range(5):
        origin = center
        for j in range(9):
            t = max(0., min(horizon, origin - width + j * width / 4.))
            value = separation(state, wa, wb, t, ax, ay)
            if value < best: center, best = t, value
        width /= 4.
    return center


@njit(cache=True, fastmath=False)
def _solve(state, wa, wb, horizon, cap, tolerance, clearance, guard, initial_times, sectors, max_nodes, max_times, max_iterations, cert_limit):
    status, witness, cert_nodes = certify(state, wa, wb, 0., 0., horizon, guard, cert_limit)
    if status == 1:
        return 0., np.nan, np.nan, 0., True, True, 0, 0, 0, 0, cert_nodes, 0
    if separation(state, wa, wb, 0., 0., 0.) <= guard:
        return np.inf, np.nan, np.nan, cap, False, False, 0, 0, 0, 0, cert_nodes, 3
    times, squared = np.empty(max_times), np.empty(max_times)
    projection, radii, nx, ny = np.empty((max_times, 4)), np.empty((max_times, 4)), np.empty((max_times, 4)), np.empty((max_times, 4))
    count = 0
    for j in range(initial_times + 1):
        count = _store_time(state, wa, wb, horizon * j / initial_times, times, squared, projection, radii, nx, ny, count)
    count = _store_time(state, wa, wb, witness, times, squared, projection, radii, nx, ny, count)
    rays, sectors_used = 0, 0
    value, ux, uy, bound, converged = np.inf, np.nan, np.nan, 0., False
    for iteration in range(max_iterations):
        value, ux, uy, bound, converged, ray_count, node_count = _angular_solve(
            projection, radii, nx, ny, squared, count, cap, tolerance / 4., clearance, guard, sectors, max_nodes,
        )
        rays += ray_count
        sectors_used += node_count
        if not math.isfinite(value):
            return value, ux, uy, bound, False, converged, iteration + 1, count, rays, sectors_used, cert_nodes, 4
        status, witness, visited = certify(state, wa, wb, value * ux, value * uy, horizon, guard, cert_limit)
        cert_nodes += visited
        if status == 1:
            return value, ux, uy, bound, True, converged, iteration + 1, count, rays, sectors_used, cert_nodes, 1 if converged else 5
        # Do NOT insist that the minimum of a sampled relaxation is itself
        # continuously safe. A verified feasible upper bound within the
        # requested global error bracket is sufficient. Each magnitude is
        # checked independently; no monotone feasibility assumption is made.
        trial = min(cap, max(value, max(0., bound) + tolerance * (1. - 1e-10)))
        if trial > value:
            trial_status, trial_witness, visited = certify(state, wa, wb, trial * ux, trial * uy, horizon, guard, cert_limit)
            cert_nodes += visited
            if trial_status == 1 and trial - bound <= tolerance + 1e-12:
                return trial, ux, uy, max(0., bound), True, True, iteration + 1, count, rays, sectors_used, cert_nodes, 1
        if count + 8 >= max_times:
            break
        old_count = count
        count = _store_time(state, wa, wb, witness, times, squared, projection, radii, nx, ny, count)
        refined = _refine_witness_time(state, wa, wb, value * ux, value * uy, witness, horizon)
        count = _store_time(state, wa, wb, refined, times, squared, projection, radii, nx, ny, count)
        # Neighbouring exact constraints close the angular loopholes around a
        # single critical time without globally densifying the whole horizon.
        for shift in (-.02, -.005, .005, .02):
            t = max(0., min(horizon, refined + shift))
            count = _store_time(state, wa, wb, t, times, squared, projection, radii, nx, ny, count)
        if count == old_count:
            return value, ux, uy, bound, False, converged, iteration + 1, count, rays, sectors_used, cert_nodes, 6
    return value, ux, uy, bound, False, converged, max_iterations, count, rays, sectors_used, cert_nodes, 2


def solve_pair(state_values, pair, config=EAConfig(), budget=ContinuousBudget()):
    state = np.asarray(state_values, dtype=np.float64).copy()
    state[2], state[9] = max(0., state[2]), max(0., state[9])
    wa = float(state[6]) if int(pair[0]) else 0.
    wb = float(state[13]) if int(pair[1]) else 0.
    wa = 0. if abs(wa) < 1e-6 else wa
    wb = 0. if abs(wb) < 1e-6 else wb
    result = _solve(state, wa, wb, config.prediction_horizon_s, config.max_extra_acceleration_mps2,
                    budget.acceleration_tolerance, budget.clearance_m, budget.floating_guard_m,
                    budget.initial_time_intervals, budget.initial_sectors, budget.max_angular_nodes,
                    budget.max_time_constraints, budget.max_exchange_iterations, budget.max_certificate_nodes)
    value, ux, uy, lower, verified, converged, iterations, times, rays, nodes, cert_nodes, status = result
    return ContinuousResult(float(value), float(ux), float(uy), float(lower), bool(verified), bool(converged),
                            int(iterations), int(times), int(rays), int(nodes), int(cert_nodes),
                            ("certified-zero", "certified-continuous", "iteration-limit", "current-contact-or-numerical-uncertainty",
                             "no-candidate-within-cap", "safe-angular-budget-exhausted", "time-constraint-stagnation")[status])


def compute_selected_modes_with_diagnostics(state_values, config=EAConfig(), pairs=FOUR_PAIRS, budget=ContinuousBudget()):
    results, identical = {}, {}
    for pair in pairs:
        wa = float(state_values[6]) if int(pair[0]) else 0.
        wb = float(state_values[13]) if int(pair[1]) else 0.
        key = (0. if abs(wa) < 1e-6 else wa, 0. if abs(wb) < 1e-6 else wb)
        if key not in identical:
            identical[key] = solve_pair(state_values, pair, config, budget)
        results[pair] = identical[key]
    return results


def compute_selected_modes(state_values, config=EAConfig(), pairs=FOUR_PAIRS, budget=ContinuousBudget()):
    results = compute_selected_modes_with_diagnostics(state_values, config, pairs, budget)
    if any(not r.continuous_verified and math.isfinite(r.value) for r in results.values()):
        raise RuntimeError("Continuous candidate unresolved; inspect diagnostic results, do not publish an unsafe scalar")
    return {pair: r.value for pair, r in results.items()}
