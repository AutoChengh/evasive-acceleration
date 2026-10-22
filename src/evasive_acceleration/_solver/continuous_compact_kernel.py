"""Compact EA constraint proposals governed by safety and global error checks."""
import math
from .continuous_tail_kernel import (
    njit, ContinuousResult, ContinuousBudget, solve_pair, certify, separation,
    _store_time, _ray, sector_lower_bound, _full_certify,
)


@njit(cache=True, fastmath=False)
def _refine_witness_time(state, wa, wb, ax, ay, witness, horizon):
    """Cheap local time PROPOSAL, not an assumption of global unimodality.

    Every proposed time defines an exact necessary constraint. The original
    witness is retained if none of these proposals improves it. Multimodality
    can hurt convergence but cannot certify an unsafe or inaccurate answer:
    final acceptance still requires continuous safety and a full-angle bound.
    """
    width = min(.05, horizon / 200.)
    lo, hi = max(0., witness - width), min(horizon, witness + width)
    best_time = witness
    best = separation(state, wa, wb, witness, ax, ay)
    ratio = (math.sqrt(5.) - 1.) / 2.
    x = hi - ratio * (hi - lo)
    y = lo + ratio * (hi - lo)
    fx = separation(state, wa, wb, x, ax, ay)
    fy = separation(state, wa, wb, y, ax, ay)
    for _ in range(16):
        if fx < best: best, best_time = fx, x
        if fy < best: best, best_time = fy, y
        if fx < fy:
            hi, y, fy = y, x, fx
            x = hi - ratio * (hi - lo)
            fx = separation(state, wa, wb, x, ax, ay)
        else:
            lo, x, fx = x, y, fy
            y = lo + ratio * (hi - lo)
            fy = separation(state, wa, wb, y, ax, ay)
    if fx < best: best, best_time = fx, x
    if fy < best: best_time = y
    return best_time
