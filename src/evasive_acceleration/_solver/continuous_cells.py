"""EA continuous-time exclusion intervals with inner/outer OBB envelopes.

An affine relative centre and fixed strip polygon over a TIME INTERVAL give a
connected forbidden acceleration interval on a ray. Compute its extrema
analytically. Taylor/Hausdorff bounds enclose CTRV with inner/outer polygons;
there are no unexamined gaps between time samples. Angular lower bounds and a
continuously safe upper bound expose the remaining numerical error explicitly.
"""
from dataclasses import dataclass
import math
import numpy as np
from . import continuous as base
from .._model.config import EAConfig
from .._model.four_modes import FOUR_PAIRS

njit = base.njit


@dataclass(frozen=True)
class CellBudget:
    acceleration_tolerance: float = .001
    clearance_m: float = 1e-7
    floating_guard_m: float = 1e-9
    initial_time_cells: int = 20
    initial_sectors: int = 36
    angular_nodes_per_round: int = 256
    max_rounds: int = 256
    max_time_cells: int = 512
    max_certificate_nodes: int = 30000


@njit(cache=True, fastmath=False)
def _tighten(lo, hi, intercept, slope):
    """Intersect a time interval with intercept + slope*t <= 0."""
    if slope > 0.: hi = min(hi, -intercept / slope)
    elif slope < 0.: lo = max(lo, -intercept / slope)
    elif intercept > 0.: hi = -np.inf
    return lo, hi


@njit(cache=True, fastmath=False)
def _envelope_value(intercepts, slopes, count, t, maximum):
    value = -np.inf if maximum else np.inf
    for i in range(count):
        q = intercepts[i] + slopes[i] * t
        value = max(value, q) if maximum else min(value, q)
    return max(0., value) * 2. / (t * t)


@njit(cache=True, fastmath=False)
def _envelope_extreme(intercepts, slopes, count, lo, hi, maximum_envelope, find_maximum):
    best = -np.inf if find_maximum else np.inf
    for t in (lo, hi):
        if t > 0.:
            value = _envelope_value(intercepts, slopes, count, t, maximum_envelope)
            best = max(best, value) if find_maximum else min(best, value)
    for i in range(count):
        if slopes[i] != 0.:
            t = -2. * intercepts[i] / slopes[i]
            if lo <= t <= hi and t > 0.:
                value = _envelope_value(intercepts, slopes, count, t, maximum_envelope)
                best = max(best, value) if find_maximum else min(best, value)
        for j in range(i):
            den = slopes[i] - slopes[j]
            if den != 0.:
                t = (intercepts[j] - intercepts[i]) / den
                if lo <= t <= hi and t > 0.:
                    value = _envelope_value(intercepts, slopes, count, t, maximum_envelope)
                    best = max(best, value) if find_maximum else min(best, value)
    return best


@njit(cache=True, fastmath=False)
def affine_interval(p0, pv, radii, nx, ny, lo, hi, theta, half, cap, radius_change, inner_sector):
    """A common forbidden radial interval for every angle in a sector.

    Write y=.5*a*t^2. All strip constraints are affine in (t,y).
    Their feasible time projection is one interval. The radial projection is
    connected; extrema occur at endpoints, active-line intersections, or a
    stationary point of 2*(alpha+beta*t)/t^2. No time discretization is used.
    """
    lower_a, lower_b, upper_a, upper_b = [np.empty(9) for _ in range(4)]
    lower_a[0], lower_b[0] = 0., 0.
    nl, nu = 1, 0
    ux, uy, ch, sh = math.cos(theta), math.sin(theta), math.cos(half), math.sin(half)
    for j in range(4):
        s = radii[j] + radius_change
        if s < 0.: return False, 0., 0.
        d, e = ux * nx[j] + uy * ny[j], -uy * nx[j] + ux * ny[j]
        qlo, qhi = d * ch - abs(e) * sh, d * ch + abs(e) * sh
        if inner_sector:
            norm = math.hypot(nx[j], ny[j])
            if d >= norm * ch: qhi = norm
            if d <= -norm * ch: qlo = -norm
            error = 64 * np.finfo(np.float64).eps * (norm + 1.)
            qlo -= error
            qhi += error
        # qlo*y >= p0+pv*t-s; qhi*y <= p0+pv*t+s.
        if qlo > 0.:
            lower_a[nl], lower_b[nl] = (p0[j] - s) / qlo, pv[j] / qlo
            nl += 1
        elif qlo < 0.:
            upper_a[nu], upper_b[nu] = (p0[j] - s) / qlo, pv[j] / qlo
            nu += 1
        else: lo, hi = _tighten(lo, hi, p0[j] - s, pv[j])
        if qhi > 0.:
            upper_a[nu], upper_b[nu] = (p0[j] + s) / qhi, pv[j] / qhi
            nu += 1
        elif qhi < 0.:
            lower_a[nl], lower_b[nl] = (p0[j] + s) / qhi, pv[j] / qhi
            nl += 1
        else: lo, hi = _tighten(lo, hi, -p0[j] - s, -pv[j])
        if lo > hi: return False, 0., 0.
    for i in range(nl):
        for j in range(nu):
            lo, hi = _tighten(lo, hi, lower_a[i] - upper_a[j], lower_b[i] - upper_b[j])
            if lo > hi: return False, 0., 0.
    if hi < 0.: return False, 0., 0.
    if nu == 0: return False, 0., 0.
    singular_upper = False
    if lo == 0.:
        # Feasibility of (t=0,y>0) is NOT feasibility at finite acceleration:
        # y=.5*a*t^2 must equal zero at t=0. Otherwise only the t->0 radial
        # upper limit diverges, and the positive lower radial limit remains.
        lower_zero = np.max(lower_a[:nl])
        upper_zero = np.min(upper_a[:nu])
        if lower_zero <= 0. <= upper_zero:
            return True, 0., cap
        if upper_zero > 0.:
            singular_upper = True
        elif upper_zero == 0.:
            slope = np.inf
            for j in range(nu):
                if upper_a[j] == 0.: slope = min(slope, upper_b[j])
            singular_upper = slope > 0.
    left = _envelope_extreme(lower_a, lower_b, nl, lo, hi, True, False)
    right = cap if singular_upper else _envelope_extreme(upper_a, upper_b, nu, lo, hi, False, True)
    if left > cap or left > right: return False, 0., 0.
    return True, max(0., left), min(cap, right)


@njit(cache=True, fastmath=False)
def _build_cell(state, wa, wb, lo, hi, k, left, right, p0, pv, radii, nx, ny, error):
    mid, half = .5 * (lo + hi), .5 * (hi - lo)
    dx, dy, vx, vy, ca, sa, cb, sb = base._kinematics(state, wa, wb, mid, 0., 0.)
    left[k], right[k] = lo, hi
    error[k] = (.5 * (state[2] * abs(wa) + state[9] * abs(wb)) * half * half
                + .5 * math.hypot(state[4], state[5]) * abs(wa) * half
                + .5 * math.hypot(state[11], state[12]) * abs(wb) * half)
    for j in range(4):
        x = ca if j == 0 else -sa if j == 1 else cb if j == 2 else -sb
        y = sa if j == 0 else ca if j == 1 else sb if j == 2 else cb
        nx[k, j], ny[k, j] = x, y
        pv[k, j] = vx * x + vy * y
        p0[k, j] = dx * x + dy * y - mid * pv[k, j]
        radii[k, j] = .5 * (state[4] * abs(x * ca + y * sa) + state[5] * abs(-x * sa + y * ca)
                                 + state[11] * abs(x * cb + y * sb) + state[12] * abs(-x * sb + y * cb))


@njit(cache=True, fastmath=False)
def _radial_union(left, right, p0, pv, radii, nx, ny, error, count, theta, half, cap, guard, inner):
    lower, upper, owner = np.empty(count), np.empty(count), np.empty(count, dtype=np.int64)
    size = 0
    for k in range(count):
        change = -error[k] - guard if inner else error[k] + guard
        hit, lo, hi = affine_interval(p0[k], pv[k], radii[k], nx[k], ny[k], left[k], right[k], theta, half, cap, change, inner)
        if hit:
            lower[size], upper[size], owner[size] = lo, hi, k
            size += 1
    if size == 0: return 0., -1
    order = np.argsort(lower[:size])
    reach, active = 0., -1
    for j in range(size):
        k = order[j]
        if lower[k] > reach: break
        if upper[k] > reach: reach, active = upper[k], owner[k]
    return reach, active


@njit(cache=True, fastmath=False)
def _bridge_cell(left, right, p0, pv, radii, nx, ny, error, count, theta, cap, guard, tolerance):
    """Refine the cell hiding the FIRST inner radial gap, not a distant exit.

    Refining the outer union's largest endpoint cannot close a missing inner
    interval near zero. Locate the first actual gap and ask which outer-only
    cell covers a point inside it. This is proposal logic, not a safety gate.
    """
    lower, upper = np.empty(count), np.empty(count)
    size = 0
    for k in range(count):
        hit, lo, hi = affine_interval(p0[k], pv[k], radii[k], nx[k], ny[k], left[k], right[k], theta, 0., cap, -error[k] - guard, True)
        if hit:
            lower[size], upper[size] = lo, hi
            size += 1
    order = np.argsort(lower[:size])
    reach, next_start = 0., cap
    for j in range(size):
        k = order[j]
        if lower[k] > reach:
            next_start = lower[k]
            break
        reach = max(reach, upper[k])
    probe = reach + min(tolerance / 4., max(0., next_start - reach) / 2.)
    chosen, score = -1, -1.
    for k in range(count):
        if error[k] == 0. or right[k] - left[k] < 1e-8: continue
        hit, lo, hi = affine_interval(p0[k], pv[k], radii[k], nx[k], ny[k], left[k], right[k], theta, 0., cap, error[k] + guard, False)
        if hit and lo <= probe <= hi:
            rank = error[k] / max(.5 * (.5 * (left[k] + right[k])) ** 2, 1e-8)
            if rank > score: chosen, score = k, rank
    return chosen


@njit(cache=True, fastmath=False)
def _angular(left, right, p0, pv, radii, nx, ny, error, count, cap, tolerance, guard, initial, max_nodes):
    bounds, centers, widths, ids = np.empty(max_nodes + initial + 4), np.empty(max_nodes + initial + 4), np.empty(max_nodes + initial + 4), np.empty(max_nodes + initial + 4, dtype=np.int64)
    pending, nodes, rays = 0, 0, 0
    best, bx, by, best_id = np.inf, np.nan, np.nan, -1
    half = math.pi / initial
    for j in range(initial):
        theta = (j + .5) * 2 * math.pi / initial
        lb, lid = _radial_union(left, right, p0, pv, radii, nx, ny, error, count, theta, half, cap, guard, True)
        nodes += 1
        if lb < min(best, cap) - tolerance:
            value, uid = _radial_union(left, right, p0, pv, radii, nx, ny, error, count, theta, 0., cap, guard, False)
            rays += 1
            if value < cap and value < best:
                best, bx, by, best_id = np.nextafter(value, np.inf), math.cos(theta), math.sin(theta), uid
            bounds[pending], centers[pending], widths[pending], ids[pending] = lb, theta, half, lid
            pending += 1
    low, low_id, low_theta = 0., -1, 0.
    while pending:
        at = np.argmin(bounds[:pending])
        low, theta, half, low_id = bounds[at], centers[at], widths[at], ids[at]
        low_theta = theta
        pending -= 1
        bounds[at], centers[at], widths[at], ids[at] = bounds[pending], centers[pending], widths[pending], ids[pending]
        if math.isfinite(best) and best - low <= tolerance:
            return best, bx, by, max(0., min(low, best - tolerance)), True, best_id, low_id, -1, rays, nodes
        if nodes + 2 > max_nodes or (half < 1e-4 and (best_id >= 0 and error[best_id] > 0.)):
            _v, other_id = _radial_union(left, right, p0, pv, radii, nx, ny, error, count, theta, 0., cap, guard, False)
            return best, bx, by, max(0., min(low, best - tolerance)), False, best_id, low_id, other_id, rays + 1, nodes
        for sign in (-1., 1.):
            child, width = theta + sign * half / 2., half / 2.
            lb, lid = _radial_union(left, right, p0, pv, radii, nx, ny, error, count, child, width, cap, guard, True)
            nodes += 1
            if lb < min(best, cap) - tolerance:
                value, uid = _radial_union(left, right, p0, pv, radii, nx, ny, error, count, child, 0., cap, guard, False)
                rays += 1
                if value < cap and value < best:
                    best, bx, by, best_id = np.nextafter(value, np.inf), math.cos(child), math.sin(child), uid
                bounds[pending], centers[pending], widths[pending], ids[pending] = lb, child, width, lid
                pending += 1
    return best, bx, by, max(0., min(best, cap) - tolerance), math.isfinite(best), best_id, -1, -1, rays, nodes


@njit(cache=True, fastmath=False)
def _solve(state, wa, wb, horizon, cap, tolerance, clearance, guard, initial_cells, sectors, angular_nodes, max_rounds, max_cells, cert_limit):
    baseline, _time, cert_nodes = base.certify(state, wa, wb, 0., 0., horizon, guard, cert_limit)
    if baseline == 1:
        return 0., np.nan, np.nan, 0., True, True, 0, 0, 0, 0, cert_nodes, 0
    if base.separation(state, wa, wb, 0., 0., 0.) <= guard:
        return np.inf, np.nan, np.nan, cap, False, False, 0, 0, 0, 0, cert_nodes, 3
    left, right, error = np.empty(max_cells), np.empty(max_cells), np.empty(max_cells)
    p0, pv, radii, nx, ny = [np.empty((max_cells, 4)) for _ in range(5)]
    count = 1 if wa == 0. and wb == 0. else initial_cells
    for j in range(count):
        _build_cell(state, wa, wb, horizon * j / count, horizon * (j + 1) / count, j, left, right, p0, pv, radii, nx, ny, error)
    rays, nodes = 0, 0
    value, ux, uy, lower, converged = np.inf, np.nan, np.nan, 0., False
    for iteration in range(max_rounds):
        value, ux, uy, lower, converged, upper_id, lower_id, other_id, nr, nn = _angular(
            left, right, p0, pv, radii, nx, ny, error, count, cap, tolerance, max(guard, clearance), sectors, angular_nodes,
        )
        rays += nr
        nodes += nn
        if converged:
            check, _time, visited = base.certify(state, wa, wb, value * ux, value * uy, horizon, guard, cert_limit)
            cert_nodes += visited
            return value, ux, uy, lower, check == 1, True, iteration + 1, count, rays, nodes, cert_nodes, 1 if check == 1 else 6
        refined = False
        chosen = np.array([upper_id, lower_id, other_id])
        for j in range(3):
            k = chosen[j]
            if k < 0 or count >= max_cells: continue
            duplicate = False
            for q in range(j):
                if chosen[q] == k: duplicate = True
            if duplicate or error[k] == 0.: continue
            lo, hi = left[k], right[k]
            if hi - lo < 1e-8: continue
            mid = .5 * (lo + hi)
            _build_cell(state, wa, wb, lo, mid, k, left, right, p0, pv, radii, nx, ny, error)
            _build_cell(state, wa, wb, mid, hi, count, left, right, p0, pv, radii, nx, ny, error)
            count += 1
            refined = True
        if not refined:
            return value, ux, uy, lower, False, False, iteration + 1, count, rays, nodes, cert_nodes, 5
    return value, ux, uy, lower, False, converged, max_rounds, count, rays, nodes, cert_nodes, 2


@njit(cache=True, fastmath=False)
def _solve_joint(state, wa, wb, horizon, cap, tolerance, clearance, guard, initial_cells, sectors, angular_nodes, max_rounds, max_cells, cert_limit):
    """Retain valid angular bounds/feasible incumbents across time refinements.

    Old lower bounds remain valid bounds on the SAME continuous problem, even
    when a new time envelope is constructed. Old outer-cover solutions remain
    feasible upper bounds. No restarting a 36-sector search after each split.
    """
    baseline, _t, cn = base.certify(state, wa, wb, 0., 0., horizon, guard, cert_limit)
    if baseline == 1: return 0., np.nan, np.nan, 0., True, True, 0, 0, 0, 0, cn, 0
    if base.separation(state, wa, wb, 0., 0., 0.) <= guard:
        return np.inf, np.nan, np.nan, cap, False, False, 0, 0, 0, 0, cn, 3
    left, right, error = np.empty(max_cells), np.empty(max_cells), np.empty(max_cells)
    p0, pv, radii, nx, ny = [np.empty((max_cells, 4)) for _ in range(5)]
    count = 1 if wa == 0. and wb == 0. else initial_cells
    for j in range(count):
        _build_cell(state, wa, wb, horizon * j / count, horizon * (j + 1) / count, j, left, right, p0, pv, radii, nx, ny, error)
    capacity, limit = angular_nodes * 32 + sectors + 4, max_rounds * 32
    bounds, centers, halves = np.empty(capacity), np.empty(capacity), np.empty(capacity)
    pending, nodes, rays = 0, 0, 0
    best, bx, by = np.inf, np.nan, np.nan
    geom_guard = max(guard, clearance)
    for j in range(sectors):
        theta, half = (j + .5) * 2 * math.pi / sectors, math.pi / sectors
        lower, _id = _radial_union(left, right, p0, pv, radii, nx, ny, error, count, theta, half, cap, geom_guard, True)
        bounds[pending], centers[pending], halves[pending] = lower, theta, half
        pending += 1
        nodes += 1
    low, steps = 0., 0
    converged = False
    while pending and steps < limit:
        at = np.argmin(bounds[:pending])
        low, theta, half = bounds[at], centers[at], halves[at]
        pending -= 1
        bounds[at], centers[at], halves[at] = bounds[pending], centers[pending], halves[pending]
        steps += 1
        if math.isfinite(best) and best - low <= tolerance:
            converged = True
            break
        new_lower, lower_id = _radial_union(left, right, p0, pv, radii, nx, ny, error, count, theta, half, cap, geom_guard, True)
        low = max(low, new_lower)
        if low >= cap: continue
        if math.isfinite(best) and best - low <= tolerance: continue
        outer, upper_id = _radial_union(left, right, p0, pv, radii, nx, ny, error, count, theta, 0., cap, geom_guard, False)
        rays += 1
        if outer < cap and outer < best:
            best, bx, by = np.nextafter(outer, np.inf), math.cos(theta), math.sin(theta)
        if math.isfinite(best) and best - low <= tolerance: continue
        inner, point_id = _radial_union(left, right, p0, pv, radii, nx, ny, error, count, theta, 0., cap, geom_guard, True)
        angular_gap, temporal_gap = max(0., inner - low), max(0., outer - inner)
        refined = False
        if temporal_gap >= .5 * angular_gap and temporal_gap > tolerance / 4.:
            bridge = _bridge_cell(left, right, p0, pv, radii, nx, ny, error, count, theta, cap, geom_guard, tolerance)
            chosen = np.array([bridge, point_id if bridge >= 0 else upper_id])
            for j in range(2):
                k = chosen[j]
                if k < 0 or count >= max_cells: continue
                duplicate = False
                for q in range(j):
                    if chosen[q] == k: duplicate = True
                if duplicate or error[k] == 0. or right[k] - left[k] < 1e-8: continue
                lo_t, hi_t = left[k], right[k]
                mid = .5 * (lo_t + hi_t)
                _build_cell(state, wa, wb, lo_t, mid, k, left, right, p0, pv, radii, nx, ny, error)
                _build_cell(state, wa, wb, mid, hi_t, count, left, right, p0, pv, radii, nx, ny, error)
                count += 1
                refined = True
        if refined:
            bounds[pending], centers[pending], halves[pending] = low, theta, half
            pending += 1
            continue
        if pending + 2 >= capacity or half < 1e-12:
            bounds[pending], centers[pending], halves[pending] = low, theta, half
            pending += 1
            break
        for sign in (-1., 1.):
            child, width = theta + sign * half / 2., half / 2.
            lb, _id = _radial_union(left, right, p0, pv, radii, nx, ny, error, count, child, width, cap, geom_guard, True)
            lb = max(low, lb)
            nodes += 1
            threshold = best - tolerance if math.isfinite(best) else cap
            if lb < threshold:
                bounds[pending], centers[pending], halves[pending] = lb, child, width
                pending += 1
    if not pending:
        if math.isfinite(best): converged = True
        else: return np.inf, np.nan, np.nan, cap, False, True, steps, count, rays, nodes, cn, 7
    lower = max(0., best - tolerance) if converged else max(0., min(low, best - tolerance)) if math.isfinite(best) else low
    if math.isfinite(best):
        verified, _t, visited = base.certify(state, wa, wb, best * bx, best * by, horizon, guard, cert_limit)
        cn += visited
    else: verified = -1
    status = 1 if verified == 1 and converged else 5 if verified == 1 else 6
    return best, bx, by, lower, verified == 1, converged, steps, count, rays, nodes, cn, status


def solve_pair(state_values, pair, config=EAConfig(), budget=CellBudget()):
    state = np.asarray(state_values, dtype=np.float64).copy()
    state[2], state[9] = max(0., state[2]), max(0., state[9])
    wa, wb = float(state[6]) if int(pair[0]) else 0., float(state[13]) if int(pair[1]) else 0.
    wa, wb = (0. if abs(wa) < 1e-6 else wa), (0. if abs(wb) < 1e-6 else wb)
    r = _solve_joint(state, wa, wb, config.prediction_horizon_s, config.max_extra_acceleration_mps2,
               budget.acceleration_tolerance, budget.clearance_m, budget.floating_guard_m,
               budget.initial_time_cells, budget.initial_sectors, budget.angular_nodes_per_round,
               budget.max_rounds, budget.max_time_cells, budget.max_certificate_nodes)
    value, ux, uy, lower, verified, converged, iterations, times, rays, nodes, cn, status = r
    return base.ContinuousResult(float(value), float(ux), float(uy), float(lower), bool(verified), bool(converged),
                                 int(iterations), int(times), int(rays), int(nodes), int(cn),
                                 ("certified-zero", "certified-continuous", "round-limit", "current-contact-or-uncertainty",
                                  "no-candidate", "needs-angular-or-temporal-refinement", "certificate-unresolved", "certified-cap-infeasible")[status])


def compute_selected_modes_with_diagnostics(state_values, config=EAConfig(), pairs=FOUR_PAIRS, budget=CellBudget()):
    results, identical = {}, {}
    for pair in pairs:
        wa, wb = float(state_values[6]) if int(pair[0]) else 0., float(state_values[13]) if int(pair[1]) else 0.
        key = (0. if abs(wa) < 1e-6 else wa, 0. if abs(wb) < 1e-6 else wb)
        if key not in identical: identical[key] = solve_pair(state_values, pair, config, budget)
        results[pair] = identical[key]
    return results


def compute_selected_modes(state_values, config=EAConfig(), pairs=FOUR_PAIRS, budget=CellBudget()):
    result = compute_selected_modes_with_diagnostics(state_values, config, pairs, budget)
    if any(not (r.continuous_verified and r.angular_converged) and math.isfinite(r.value) for r in result.values()):
        raise RuntimeError("EA cell refinement unresolved; inspect diagnostics; no uncertified scalar is returned")
    return {pair: r.value for pair, r in result.items()}
