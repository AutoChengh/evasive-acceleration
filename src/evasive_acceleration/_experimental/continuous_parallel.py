"""Retained angular lower bounds plus continuous separation-oracle cuts.

Bounds on the unchanged continuous problem remain valid after adding exact
time constraints. Keep the angular frontier instead of restarting a global
sampled optimization after every contact. CV-CV uses the analytic continuous
affine-cell primitive; all reported finite values require swept certification.
"""
from dataclasses import dataclass
import math
import numpy as np
from . import continuous_compact_kernel as base, continuous_cells as cells
from .._paper.config import EAConfig
from .._paper.four_modes import FOUR_PAIRS

njit = base.njit


@dataclass(frozen=True)
class RetainedBudget:
    acceleration_tolerance: float = .001
    clearance_m: float = 1e-7
    floating_guard_m: float = 1e-9
    initial_time_intervals: int = 4
    initial_sectors: int = 36
    max_frontier_nodes: int = 8192
    max_time_constraints: int = 56
    max_steps: int = 8192
    max_certificate_nodes: int = 30000

    def __post_init__(self):
        if not (math.isfinite(self.acceleration_tolerance) and self.acceleration_tolerance > 0.):
            raise ValueError("acceleration_tolerance must be finite and positive")
        if not (0. < self.floating_guard_m < self.clearance_m < math.inf):
            raise ValueError("Require 0 < floating_guard_m < clearance_m")
        if self.initial_time_intervals < 1 or self.initial_sectors < 8:
            raise ValueError("Require positive time intervals and at least 8 sectors")
        if self.max_time_constraints < self.initial_time_intervals + 12:
            raise ValueError("Time-constraint cache is too small for protected observations")
        if self.max_frontier_nodes < self.initial_sectors + 4 or self.max_steps < 1 or self.max_certificate_nodes < 1:
            raise ValueError("Search and certificate budgets must be positive and large enough")


@njit(cache=True, fastmath=False)
def _make_room(times, squared, p, r, nx, ny, count, protected, capacity):
    """Drop old scratch cuts while retaining their VALID frontier bounds.

    The true problem never changes. Existing sector lower bounds remain
    valid after forgetting a scratch constraint, and descendants inherit
    them. Every proposed upper bound still needs whole-horizon certification.
    No inputs, audit evidence or experiment records are removed.
    """
    if count + 6 < capacity: return count
    drop = max(1, (count - protected) // 2)
    for j in range(protected, count - drop):
        times[j], squared[j] = times[j + drop], squared[j + drop]
        p[j], r[j], nx[j], ny[j] = p[j + drop], r[j + drop], nx[j + drop], ny[j + drop]
    return count - drop


@njit(cache=True, fastmath=False)
def _nearby_upper_bounds(state, wa, wb, theta, horizon, cap, tolerance, clearance, guard, cert_limit,
                         times, squared, p, r, nx, ny, count, protected, max_times, lo_scratch, hi_scratch, best, bx, by):
    """Multiscale upper-bound proposals for a narrow sampled-relaxation well.

    These proposals can NEVER prune by themselves. A swept certificate is
    mandatory and the retained global lower frontier must still close the gap.
    """
    rays, certificates = 0, 0
    for offset in (.001, -.001, .0001, -.0001, .01, -.01):
        angle = theta + offset
        ux, uy = math.cos(angle), math.sin(angle)
        for _ in range(8):
            value = base._ray(p, r, nx, ny, squared, count, ux, uy, cap, clearance, lo_scratch, hi_scratch)
            rays += 1
            if not math.isfinite(value) or value >= best: break
            value = min(cap, value + tolerance / 4.)
            check, witness, visited = base.certify(state, wa, wb, value * ux, value * uy, horizon, guard, cert_limit)
            certificates += visited
            if check == 1:
                if value < best: best, bx, by = value, ux, uy
                break
            count = _make_room(times, squared, p, r, nx, ny, count, protected, max_times)
            old = count
            count = base._store_time(state, wa, wb, witness, times, squared, p, r, nx, ny, count)
            critical = base._refine_witness_time(state, wa, wb, value * ux, value * uy, witness, horizon)
            count = base._store_time(state, wa, wb, critical, times, squared, p, r, nx, ny, count)
            if count == old: break
    return best, bx, by, count, rays, certificates


@njit(cache=True, fastmath=False, nogil=True)
def _solve(state, wa, wb, horizon, cap, tolerance, clearance, guard, time_intervals, sectors, max_nodes, max_times, max_steps, cert_limit):
    status, witness, cn = base.certify(state, wa, wb, 0., 0., horizon, guard, cert_limit)
    if status == 1: return 0., np.nan, np.nan, 0., True, True, 0, 0, 0, 0, cn, 0
    if base.separation(state, wa, wb, 0., 0., 0.) <= guard:
        return np.inf, np.nan, np.nan, cap, False, False, 0, 0, 0, 0, cn, 3
    times, squared = np.empty(max_times), np.empty(max_times)
    p, r, nx, ny = [np.empty((max_times, 4)) for _ in range(4)]
    count = 0
    for j in range(time_intervals + 1):
        count = base._store_time(state, wa, wb, horizon * j / time_intervals, times, squared, p, r, nx, ny, count)
    count = base._store_time(state, wa, wb, witness, times, squared, p, r, nx, ny, count)
    protected = count
    bounds, centers, halves = np.empty(max_nodes), np.empty(max_nodes), np.empty(max_nodes)
    versions = np.empty(max_nodes, dtype=np.int64)
    epoch = 0
    lo_scratch, hi_scratch = np.empty(max_times), np.empty(max_times)
    pending, nodes, rays = 0, 0, 0
    for j in range(sectors):
        theta, half = (j + .5) * 2 * math.pi / sectors, math.pi / sectors
        lb = base.sector_lower_bound(p, r, nx, ny, squared, count, theta, half, cap, guard, lo_scratch, hi_scratch)
        bounds[pending], centers[pending], halves[pending] = lb, theta, half
        versions[pending] = epoch
        pending += 1
        nodes += 1
    best, bx, by, low, steps = np.inf, np.nan, np.nan, 0., 0
    converged = False
    recovered = False
    while pending and steps < max_steps:
        at = np.argmin(bounds[:pending])
        low, theta, half = bounds[at], centers[at], halves[at]
        version = versions[at]
        pending -= 1
        bounds[at], centers[at], halves[at] = bounds[pending], centers[pending], halves[pending]
        versions[at] = versions[pending]
        steps += 1
        if math.isfinite(best) and best - low <= tolerance:
            converged = True
            break
        lb = low if version == epoch else base.sector_lower_bound(p, r, nx, ny, squared, count, theta, half, cap, guard, lo_scratch, hi_scratch)
        low = max(low, lb)
        if low >= cap or (math.isfinite(best) and best - low <= tolerance): continue
        ux, uy = math.cos(theta), math.sin(theta)
        value = base._ray(p, r, nx, ny, squared, count, ux, uy, cap, clearance, lo_scratch, hi_scratch)
        rays += 1
        added = False
        old_count = count
        if math.isfinite(value) and value < best:
            proposal = min(cap, value + tolerance * .25)
            check, witness, visited = base.certify(state, wa, wb, proposal * ux, proposal * uy, horizon, guard, cert_limit)
            cn += visited
            if check == 1:
                if proposal < best: best, bx, by = proposal, ux, uy
            else:
                count = _make_room(times, squared, p, r, nx, ny, count, protected, max_times)
                old_count = count
                count = base._store_time(state, wa, wb, witness, times, squared, p, r, nx, ny, count)
                critical = base._refine_witness_time(state, wa, wb, proposal * ux, proposal * uy, witness, horizon)
                count = base._store_time(state, wa, wb, critical, times, squared, p, r, nx, ny, count)
                trial = min(cap, value + tolerance * .5)
                trial_check, trial_t, visited = base.certify(state, wa, wb, trial * ux, trial * uy, horizon, guard, cert_limit)
                cn += visited
                if trial_check == 1:
                    if trial < best: best, bx, by = trial, ux, uy
                else:
                    count = base._store_time(state, wa, wb, trial_t, times, squared, p, r, nx, ny, count)
                added = count > old_count
                if added: epoch += 1
        # Children will be bounded against all current cuts immediately below.
        # Recomputing their parent's bound first, then repeatedly reinserting
        # that same sector, adds work without being necessary for correctness.
        # Keep the existing valid parent bound and cover it by both children.
        new_lower = low
        if half < 1e-5 and not recovered and (not math.isfinite(best) or best - new_lower > tolerance):
            best, bx, by, count, nr, nc = _nearby_upper_bounds(
                state, wa, wb, theta, horizon, cap, tolerance, clearance, guard, cert_limit,
                times, squared, p, r, nx, ny, count, protected, max_times, lo_scratch, hi_scratch, best, bx, by,
            )
            rays += nr
            cn += nc
            recovered = True
            epoch += 1
        if math.isfinite(best) and best - new_lower <= tolerance: continue
        if pending + 2 >= max_nodes or half < 1e-12:
            # The popped sector has NOT been excluded. Preserve it on budget
            # exit; an empty queue must never masquerade as a completed proof.
            bounds[pending], centers[pending], halves[pending] = new_lower, theta, half
            versions[pending] = -1
            pending += 1
            break
        for sign in (-1., 1.):
            child, width = theta + sign * half / 2., half / 2.
            lb = base.sector_lower_bound(p, r, nx, ny, squared, count, child, width, cap, guard, lo_scratch, hi_scratch)
            lb = max(new_lower, lb)
            nodes += 1
            threshold = best - tolerance if math.isfinite(best) else cap
            if lb < threshold:
                bounds[pending], centers[pending], halves[pending] = lb, child, width
                versions[pending] = epoch
                pending += 1
    if not pending:
        if math.isfinite(best): converged = True
        else: return np.inf, np.nan, np.nan, cap, False, True, steps, count, rays, nodes, cn, 7
    lower = max(0., best - tolerance) if converged else max(0., min(low, best - tolerance)) if math.isfinite(best) else low
    return best, bx, by, lower, math.isfinite(best), converged, steps, count, rays, nodes, cn, 1 if converged else 5


def solve_pair(state_values, pair, config=EAConfig(), budget=RetainedBudget()):
    state = np.asarray(state_values, dtype=np.float64).copy()
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("Expected fourteen finite canonical state values")
    if any(state[i] <= 0. for i in (4, 5, 11, 12)):
        raise ValueError("Both road users must have positive length and width")
    if len(pair) != 2 or any(int(mode) not in (0, 1) for mode in pair):
        raise ValueError("The continuous EA candidate supports CV and CTRV pairings only")
    if not (0. < config.prediction_horizon_s < math.inf and 0. <= config.max_extra_acceleration_mps2 < math.inf):
        raise ValueError("Require a positive finite horizon and nonnegative finite acceleration cap")
    state[2], state[9] = max(0., state[2]), max(0., state[9])
    wa, wb = float(state[6]) if int(pair[0]) else 0., float(state[13]) if int(pair[1]) else 0.
    wa, wb = (0. if abs(wa) < 1e-6 else wa), (0. if abs(wb) < 1e-6 else wb)
    if wa == 0. and wb == 0.:
        clearance = budget.clearance_m
        # A fixed spatial roundoff buffer can dominate the requested
        # acceleration error on very short horizons. Tighten that buffer only
        # when the explicitly reported bracket fails to close; never relax
        # acceleration accuracy or silently accept an unresolved result.
        for _ in range(4):
            affine = cells.solve_pair(state, pair, config, cells.CellBudget(
                acceleration_tolerance=budget.acceleration_tolerance,
                clearance_m=clearance, floating_guard_m=budget.floating_guard_m,
                max_certificate_nodes=budget.max_certificate_nodes))
            if affine.continuous_verified and affine.angular_converged:
                return affine
            next_clearance = max(4 * budget.floating_guard_m, clearance / 10.)
            if next_clearance >= clearance:
                break
            clearance = next_clearance
        return affine
    r = _solve(state, wa, wb, config.prediction_horizon_s, config.max_extra_acceleration_mps2,
               budget.acceleration_tolerance, budget.clearance_m, budget.floating_guard_m,
               budget.initial_time_intervals, budget.initial_sectors, budget.max_frontier_nodes,
               budget.max_time_constraints, budget.max_steps, budget.max_certificate_nodes)
    value, ux, uy, lower, verified, converged, steps, times, rays, nodes, cn, status = r
    result = base.ContinuousResult(float(value), float(ux), float(uy), float(lower), bool(verified), bool(converged),
                                 int(steps), int(times), int(rays), int(nodes), int(cn),
                                 ("certified-zero", "certified-continuous", "step-limit", "current-contact-or-uncertainty",
                                  "no-candidate", "safe-budget-exhausted", "unresolved", "certified-cap-infeasible")[status])
    if math.isfinite(result.value) and not (result.continuous_verified and result.angular_converged):
        # Complementary separation-oracle restart only for a stalled frontier.
        # Combine independently valid lower bounds and certified upper bounds;
        # a fallback cannot make an uncertified value acceptable.
        fallback = base.solve_pair(state, pair, config, base.ContinuousBudget(
            acceleration_tolerance=budget.acceleration_tolerance))
        valid = [v for v in (result, fallback) if v.continuous_verified and math.isfinite(v.value)]
        if valid:
            upper = min(valid, key=lambda v: v.value)
            bound = max(result.lower_bound, fallback.lower_bound)
            if bound > upper.value + 1e-8:
                raise RuntimeError("Inconsistent continuous lower/upper bounds")
            if upper.value - bound <= budget.acceleration_tolerance + 1e-12:
                return base.ContinuousResult(upper.value, upper.direction_x, upper.direction_y, bound,
                                              True, True, result.iterations + fallback.iterations,
                                              max(result.time_constraints, fallback.time_constraints),
                                              result.direction_evaluations + fallback.direction_evaluations,
                                              result.sector_nodes + fallback.sector_nodes,
                                              result.certificate_nodes + fallback.certificate_nodes,
                                              "certified-continuous-fallback")
    if (budget.max_steps >= 8192 and
        not (result.continuous_verified and result.angular_converged) and
        result.status not in ("certified-cap-infeasible", "current-contact-or-uncertainty")):
        # A small scratch cache is a latency optimization, not permission to
        # weaken a result. Retry the preserved larger-cache solver when needed.
        from . import continuous_tail as robust
        fallback = robust.solve_pair(state, pair, config, robust.RetainedBudget(
            acceleration_tolerance=budget.acceleration_tolerance,
            clearance_m=budget.clearance_m, floating_guard_m=budget.floating_guard_m,
            max_certificate_nodes=budget.max_certificate_nodes))
        if fallback.continuous_verified and fallback.angular_converged:
            return base.ContinuousResult(fallback.value, fallback.direction_x, fallback.direction_y,
                fallback.lower_bound, True, True, result.iterations + fallback.iterations,
                max(result.time_constraints, fallback.time_constraints),
                result.direction_evaluations + fallback.direction_evaluations,
                result.sector_nodes + fallback.sector_nodes, result.certificate_nodes + fallback.certificate_nodes,
                "certified-continuous-larger-cache-fallback")
        if fallback.status in ("certified-cap-infeasible", "current-contact-or-uncertainty"):
            return fallback
    return result


def compute_selected_modes_with_diagnostics(state_values, config=EAConfig(), pairs=FOUR_PAIRS, budget=RetainedBudget()):
    result, cache = {}, {}
    for pair in pairs:
        if len(pair) != 2 or any(int(mode) not in (0, 1) for mode in pair):
            raise ValueError("The continuous EA candidate supports CV and CTRV pairings only")
        wa, wb = float(state_values[6]) if int(pair[0]) else 0., float(state_values[13]) if int(pair[1]) else 0.
        key = (0. if abs(wa) < 1e-6 else wa, 0. if abs(wb) < 1e-6 else wb)
        if key not in cache: cache[key] = solve_pair(state_values, pair, config, budget)
        result[pair] = cache[key]
    return result


def compute_selected_modes(state_values, config=EAConfig(), pairs=FOUR_PAIRS, budget=RetainedBudget()):
    result = compute_selected_modes_with_diagnostics(state_values, config, pairs, budget)
    if any(not (r.continuous_verified and r.angular_converged)
           and r.status not in ("certified-cap-infeasible", "current-contact-or-uncertainty")
           for r in result.values()):
        raise RuntimeError("Retained continuous search unresolved; inspect diagnostics")
    return {pair: r.value for pair, r in result.items()}



class ParallelEvaluator:
    """Explicit multi-core latency mode; NOT a single-core speed claim.

    Keep the executor alive across frames. Inputs, scratch arrays and Numba
    workspaces are local to each independent pairing. Shared-equivalent zero
    yaw-rate pairings are evaluated once. No worker changes data or tolerance.
    """

    def __init__(self, workers=3, config=EAConfig(), budget=RetainedBudget(), cpu_ids=None):
        from concurrent.futures import ThreadPoolExecutor
        import itertools
        import os
        if not isinstance(workers, int) or not 1 <= workers <= 4:
            raise ValueError("workers must be an integer in [1, 4]")
        self.config, self.budget = config, budget
        self.workers = workers
        self.cpu_ids = None if cpu_ids is None else tuple(cpu_ids)
        initialize = None
        if self.cpu_ids is not None:
            if not hasattr(os, "sched_setaffinity"):
                raise ValueError("Explicit worker CPU affinity is unavailable on this platform")
            if len(self.cpu_ids) != workers or len(set(self.cpu_ids)) != workers:
                raise ValueError("Provide one distinct CPU per worker")
            if any(cpu not in os.sched_getaffinity(0) for cpu in self.cpu_ids):
                raise ValueError("Worker CPUs must be inside the calling thread's allowed CPU set")
            counter = itertools.count()
            def initialize():
                os.sched_setaffinity(0, {self.cpu_ids[next(counter)]})
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ea-pair", initializer=initialize)
        self._closed = False

    def compute_with_diagnostics(self, state_values, pairs=FOUR_PAIRS):
        if self._closed:
            raise RuntimeError("ParallelEvaluator is closed")
        state = np.asarray(state_values, dtype=np.float64).copy()
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Expected fourteen finite canonical state values")
        if any(state[i] <= 0. for i in (4, 5, 11, 12)):
            raise ValueError("Both road users must have positive length and width")
        jobs, keys = {}, []
        for pair in pairs:
            if len(pair) != 2 or any(int(mode) not in (0, 1) for mode in pair):
                raise ValueError("CV and CTRV pairings only")
            wa = state[6] if int(pair[0]) else 0.
            wb = state[13] if int(pair[1]) else 0.
            key = (0. if abs(wa) < 1e-6 else wa, 0. if abs(wb) < 1e-6 else wb)
            keys.append((pair, key))
            if key not in jobs:
                jobs[key] = self._executor.submit(solve_pair, state, pair, self.config, self.budget)
        completed = {key: future.result() for key, future in jobs.items()}
        return {pair: completed[key] for pair, key in keys}

    def compute(self, state_values, pairs=FOUR_PAIRS):
        results = self.compute_with_diagnostics(state_values, pairs)
        if any(not (r.continuous_verified and r.angular_converged)
               and r.status not in ("certified-cap-infeasible", "current-contact-or-uncertainty")
               for r in results.values()):
            raise RuntimeError("Parallel continuous search unresolved; inspect diagnostics")
        return {pair: result.value for pair, result in results.items()}

    def close(self):
        if not self._closed:
            self._executor.shutdown(wait=True)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

