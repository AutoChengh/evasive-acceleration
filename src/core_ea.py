import math
import time
from functools import lru_cache
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Optional JIT acceleration. Numerical results are intended to be identical
# with or without numba; numba only affects runtime.
try:
    from numba import njit

    _NUMBA_OK = True
except Exception:
    _NUMBA_OK = False

    def njit(*args, **kwargs):
        def _wrap(fn):
            return fn

        return _wrap


__all__ = [
    "aggregate_ea_modes",
    "compute_single_mode_ea",
    "compute_single_mode_ea_timed",
    "compute_ea_modes",
    "compute_ea_modes_timed",
    "compute_ea",
    "compute_ea_timed",
]


# ============================================================================
# Default numerical configuration
# ============================================================================
DEFAULT_T_TOTAL = 10.0
DEFAULT_DT_COARSE = 0.1
DEFAULT_DT_FINE = 0.02
DEFAULT_A_MAX = 100.0
DEFAULT_TOL = 1e-3

DEFAULT_COARSE_SECTOR_NUM = 72
DEFAULT_LOCAL_FINE_HALF_WINDOW_DEG = 5.0
DEFAULT_LOCAL_FINE_DIR_NUM = 101
DEFAULT_FINE_REFINE_RATIO = 1.15


# ============================================================================
# Formatting helpers for the runnable demo
# ============================================================================
def _format_float_3(x: float) -> str:
    """Format a float for compact console display."""
    try:
        xf = float(x)
    except Exception:
        return str(x)

    if math.isnan(xf):
        return "nan"
    if math.isinf(xf):
        return "inf" if xf > 0 else "-inf"

    s = f"{xf:.3f}".rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return s


def _format_seconds(sec: float) -> str:
    """Format a duration in a human-readable unit."""
    sec = float(sec)
    if sec < 1e-3:
        return f"{sec * 1e6:.3f} us"
    if sec < 1.0:
        return f"{sec * 1e3:.3f} ms"
    return f"{sec:.6f} s"


# ============================================================================
# Public aggregation function
# ============================================================================
def aggregate_ea_modes(mode_dict: Dict[str, float]) -> float:
    """
    Aggregate the four mode-specific EA values into the final EA value.

    The final EA is defined as the arithmetic mean of:
        EA_CTCT, EA_CTCV, EA_CVCT, EA_CVCV

    By design, if any of the four mode-specific values is NaN, the final EA is
    reported as NaN as well. This keeps the final definition strict and
    unambiguous.
    """
    required_keys = ("EA_CTCT", "EA_CTCV", "EA_CVCT", "EA_CVCV")

    try:
        vals = [float(mode_dict[k]) for k in required_keys]
    except Exception:
        return float("nan")

    if any(math.isnan(v) for v in vals):
        return float("nan")

    return float(sum(vals) / 4.0)


# ============================================================================
# Cached grids / direction sets
# ============================================================================
@lru_cache(maxsize=None)
def _get_time_grids(T_total: float, dt_pred: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Build the prediction time grids.

    Returns:
        t_arr:  time array
        t2_arr: 0.5 * t^2, so extra displacement from constant acceleration a is
                a * t2_arr
        num_steps: number of sampled time steps
    """
    T_total = float(T_total)
    dt_pred = float(dt_pred)
    num_steps = int(round(T_total / dt_pred)) + 1
    t_arr = np.linspace(0.0, T_total, num_steps)
    t2_arr = 0.5 * (t_arr ** 2)
    return t_arr.astype(np.float64), t2_arr.astype(np.float64), num_steps


@lru_cache(maxsize=None)
def _get_sector_unit_vectors(sector_num: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the coarse search directions as unit vectors.

    The i-th direction corresponds to angle:
        (i + 0.5) * dtheta
    """
    sector_num = int(sector_num)
    dtheta = 2.0 * math.pi / sector_num
    thetas = (np.arange(sector_num, dtype=np.float64) + 0.5) * dtheta
    ux = np.cos(thetas)
    uy = np.sin(thetas)
    return ux.astype(np.float64), uy.astype(np.float64)


def _build_local_unit_vectors(
    theta_center: float,
    half_window_deg: float,
    local_dir_num: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the refined search directions around one coarse-search direction.
    """
    half_window_rad = math.radians(float(half_window_deg))
    local_dir_num = int(local_dir_num)
    thetas = np.linspace(
        theta_center - half_window_rad,
        theta_center + half_window_rad,
        local_dir_num,
    )
    ux = np.cos(thetas).astype(np.float64)
    uy = np.sin(thetas).astype(np.float64)
    return thetas.astype(np.float64), ux, uy


# ============================================================================
# CTRV propagation
# ============================================================================
def step_ctrv(
    x: float,
    y: float,
    v: float,
    heading: float,
    omega: float,
    dt: float,
    eps_omega: float = 1e-6,
) -> Tuple[float, float, float, float]:
    """
    Propagate one CTRV step under:
      - constant speed v
      - constant yaw rate omega
    """
    if abs(omega) < eps_omega:
        x_new = x + v * math.cos(heading) * dt
        y_new = y + v * math.sin(heading) * dt
        heading_new = heading
    else:
        R = v / omega
        dpsi = omega * dt
        x_new = x + R * (math.sin(heading + dpsi) - math.sin(heading))
        y_new = y - R * (math.cos(heading + dpsi) - math.cos(heading))
        heading_new = heading + dpsi

    return x_new, y_new, v, heading_new


# ============================================================================
# Support-function SAT preparation
# ============================================================================
@njit(cache=True, fastmath=False)
def _prepare_support_sat_data_jit(
    xA_arr,
    yA_arr,
    hA_arr,
    xB_arr,
    yB_arr,
    hB_arr,
    lengthA,
    widthA,
    lengthB,
    widthB,
):
    """
    Precompute exact OBB-vs-OBB SAT data in support-function form.
    """
    N = xA_arr.shape[0]

    relx_arr = np.empty(N, dtype=np.float64)
    rely_arr = np.empty(N, dtype=np.float64)

    axes_x = np.empty((N, 4), dtype=np.float64)
    axes_y = np.empty((N, 4), dtype=np.float64)
    sum_radius = np.empty((N, 4), dtype=np.float64)

    halfLA = 0.5 * lengthA
    halfWA = 0.5 * widthA
    halfLB = 0.5 * lengthB
    halfWB = 0.5 * widthB

    for k in range(N):
        relx = xB_arr[k] - xA_arr[k]
        rely = yB_arr[k] - yA_arr[k]
        relx_arr[k] = relx
        rely_arr[k] = rely

        cA = math.cos(hA_arr[k])
        sA = math.sin(hA_arr[k])
        a1x, a1y = cA, sA
        a2x, a2y = -sA, cA

        cB = math.cos(hB_arr[k])
        sB = math.sin(hB_arr[k])
        b1x, b1y = cB, sB
        b2x, b2y = -sB, cB

        axes_x[k, 0], axes_y[k, 0] = a1x, a1y
        axes_x[k, 1], axes_y[k, 1] = a2x, a2y
        axes_x[k, 2], axes_y[k, 2] = b1x, b1y
        axes_x[k, 3], axes_y[k, 3] = b2x, b2y

        n0x, n0y = a1x, a1y
        sum_radius[k, 0] = (
            halfLA * abs(a1x * n0x + a1y * n0y)
            + halfWA * abs(a2x * n0x + a2y * n0y)
            + halfLB * abs(b1x * n0x + b1y * n0y)
            + halfWB * abs(b2x * n0x + b2y * n0y)
        )

        n1x, n1y = a2x, a2y
        sum_radius[k, 1] = (
            halfLA * abs(a1x * n1x + a1y * n1y)
            + halfWA * abs(a2x * n1x + a2y * n1y)
            + halfLB * abs(b1x * n1x + b1y * n1y)
            + halfWB * abs(b2x * n1x + b2y * n1y)
        )

        n2x, n2y = b1x, b1y
        sum_radius[k, 2] = (
            halfLA * abs(a1x * n2x + a1y * n2y)
            + halfWA * abs(a2x * n2x + a2y * n2y)
            + halfLB * abs(b1x * n2x + b1y * n2y)
            + halfWB * abs(b2x * n2x + b2y * n2y)
        )

        n3x, n3y = b2x, b2y
        sum_radius[k, 3] = (
            halfLA * abs(a1x * n3x + a1y * n3y)
            + halfWA * abs(a2x * n3x + a2y * n3y)
            + halfLB * abs(b1x * n3x + b1y * n3y)
            + halfWB * abs(b2x * n3x + b2y * n3y)
        )

    rA = math.hypot(halfLA, halfWA)
    rB = math.hypot(halfLB, halfWB)
    R2 = (rA + rB) ** 2

    return relx_arr, rely_arr, axes_x, axes_y, sum_radius, R2


# ============================================================================
# Exact OBB collision checks
# ============================================================================
@njit(cache=True, fastmath=False)
def _check_collision_base_support_jit(
    relx_arr,
    rely_arr,
    axes_x,
    axes_y,
    sum_radius,
    t_arr,
    R2,
):
    N = relx_arr.shape[0]

    for k in range(N):
        relx = relx_arr[k]
        rely = rely_arr[k]

        if relx * relx + rely * rely > R2:
            continue

        sep = False
        for j in range(4):
            proj = relx * axes_x[k, j] + rely * axes_y[k, j]
            if abs(proj) > sum_radius[k, j]:
                sep = True
                break

        if not sep:
            return True, float(t_arr[k])

    return False, 0.0


@njit(cache=True, fastmath=False)
def _check_collision_with_extra_accel_support_jit(
    relx_arr,
    rely_arr,
    axes_x,
    axes_y,
    sum_radius,
    t_arr,
    t2_arr,
    ax_extra,
    ay_extra,
    R2,
):
    N = relx_arr.shape[0]

    for k in range(N):
        relx = relx_arr[k] - ax_extra * t2_arr[k]
        rely = rely_arr[k] - ay_extra * t2_arr[k]

        if relx * relx + rely * rely > R2:
            continue

        sep = False
        for j in range(4):
            proj = relx * axes_x[k, j] + rely * axes_y[k, j]
            if abs(proj) > sum_radius[k, j]:
                sep = True
                break

        if not sep:
            return True, float(t_arr[k])

    return False, 0.0


@njit(cache=True, fastmath=False)
def _will_collide_with_extra_accel_support_jit(
    relx_arr,
    rely_arr,
    axes_x,
    axes_y,
    sum_radius,
    t_arr,
    t2_arr,
    ax_extra,
    ay_extra,
    R2,
):
    will_collide, _ = _check_collision_with_extra_accel_support_jit(
        relx_arr,
        rely_arr,
        axes_x,
        axes_y,
        sum_radius,
        t_arr,
        t2_arr,
        ax_extra,
        ay_extra,
        R2,
    )
    return will_collide


# ============================================================================
# Interval solver on one direction
# ============================================================================
@njit(cache=True, fastmath=False)
def _direction_collision_interval_one_time_jit(
    relx,
    rely,
    axes_x_k,
    axes_y_k,
    sum_radius_k,
    t2,
    ux,
    uy,
    a_max,
    eps_q,
):
    L = 0.0
    U = a_max

    for j in range(4):
        nx = axes_x_k[j]
        ny = axes_y_k[j]
        s = sum_radius_k[j]

        p = relx * nx + rely * ny
        q = t2 * (ux * nx + uy * ny)

        if abs(q) <= eps_q:
            if abs(p) > s:
                return False, 0.0, 0.0
            else:
                continue

        a1 = (p - s) / q
        a2 = (p + s) / q

        lo = a1 if a1 < a2 else a2
        hi = a2 if a2 > a1 else a1

        if lo > L:
            L = lo
        if hi < U:
            U = hi

        if L > U:
            return False, 0.0, 0.0

    if L < 0.0:
        L = 0.0
    if U > a_max:
        U = a_max

    if L > U:
        return False, 0.0, 0.0

    return True, L, U


@njit(cache=True, fastmath=False)
def _solve_direction_min_accel_interval_jit(
    relx_arr,
    rely_arr,
    axes_x,
    axes_y,
    sum_radius,
    t_arr,
    t2_arr,
    R2,
    ux,
    uy,
    a_max,
    tol,
):
    if _will_collide_with_extra_accel_support_jit(
        relx_arr,
        rely_arr,
        axes_x,
        axes_y,
        sum_radius,
        t_arr,
        t2_arr,
        a_max * ux,
        a_max * uy,
        R2,
    ):
        return np.nan

    N = relx_arr.shape[0]
    eps_q = 1e-15

    Ls = np.empty(N, dtype=np.float64)
    Us = np.empty(N, dtype=np.float64)
    m = 0

    for k in range(N):
        has_itv, L, U = _direction_collision_interval_one_time_jit(
            relx_arr[k],
            rely_arr[k],
            axes_x[k],
            axes_y[k],
            sum_radius[k],
            t2_arr[k],
            ux,
            uy,
            a_max,
            eps_q,
        )
        if has_itv:
            Ls[m] = L
            Us[m] = U
            m += 1

    if m == 0:
        return 0.0

    idx = np.argsort(Ls[:m])
    reach = 0.0
    covered_zero = False

    for ii in range(m):
        i = idx[ii]
        L = Ls[i]
        U = Us[i]

        if not covered_zero:
            if L <= tol and U >= -tol:
                covered_zero = True
                if U > reach:
                    reach = U
            else:
                if L > tol:
                    return 0.0
        else:
            if L <= reach + tol:
                if U > reach:
                    reach = U
            else:
                break

    if not covered_zero:
        return 0.0

    if reach >= a_max - tol:
        return np.nan

    candidate = np.nextafter(reach, np.inf)
    if candidate > a_max:
        return np.nan

    return candidate


@njit(cache=True, fastmath=False)
def _compute_ea_on_directions_interval_jit(
    relx_arr,
    rely_arr,
    axes_x,
    axes_y,
    sum_radius,
    t_arr,
    t2_arr,
    R2,
    ux_arr,
    uy_arr,
    a_max,
    tol,
):
    base_collide, _ = _check_collision_base_support_jit(
        relx_arr, rely_arr, axes_x, axes_y, sum_radius, t_arr, R2
    )
    if not base_collide:
        return 0.0, -1, False

    found = False
    best_idx = -1
    ea_min = 1e30

    n_dir = ux_arr.shape[0]
    for i in range(n_dir):
        uxi = ux_arr[i]
        uyi = uy_arr[i]

        if found:
            if _will_collide_with_extra_accel_support_jit(
                relx_arr,
                rely_arr,
                axes_x,
                axes_y,
                sum_radius,
                t_arr,
                t2_arr,
                ea_min * uxi,
                ea_min * uyi,
                R2,
            ):
                continue

        m_dir = _solve_direction_min_accel_interval_jit(
            relx_arr,
            rely_arr,
            axes_x,
            axes_y,
            sum_radius,
            t_arr,
            t2_arr,
            R2,
            uxi,
            uyi,
            a_max,
            tol,
        )

        if np.isnan(m_dir):
            continue

        if (not found) or (m_dir < ea_min):
            found = True
            ea_min = m_dir
            best_idx = i

    if not found:
        return np.nan, -1, True

    return ea_min, best_idx, True


# ============================================================================
# Trajectory preparation
# ============================================================================
def prepare_ctrv_prediction(
    xA: float,
    yA: float,
    vA: float,
    hA: float,
    xB: float,
    yB: float,
    vB: float,
    hB: float,
    lA: float,
    wA: float,
    lB: float,
    wB: float,
    yawA: float,
    yawB: float,
    T_total: float,
    dt_pred: float,
) -> Dict[str, np.ndarray]:
    """
    Simulate CTRV forward and precompute support-function SAT data.
    """
    t_arr, t2_arr, num_steps = _get_time_grids(float(T_total), float(dt_pred))

    xA_arr = np.zeros(num_steps, dtype=np.float64)
    yA_arr = np.zeros(num_steps, dtype=np.float64)
    hA_arr = np.zeros(num_steps, dtype=np.float64)

    xB_arr = np.zeros(num_steps, dtype=np.float64)
    yB_arr = np.zeros(num_steps, dtype=np.float64)
    hB_arr = np.zeros(num_steps, dtype=np.float64)

    _xA, _yA, _vA, _hA = float(xA), float(yA), float(vA), float(hA)
    _xB, _yB, _vB, _hB = float(xB), float(yB), float(vB), float(hB)
    dt_pred = float(dt_pred)

    for k in range(num_steps):
        xA_arr[k] = _xA
        yA_arr[k] = _yA
        hA_arr[k] = _hA

        xB_arr[k] = _xB
        yB_arr[k] = _yB
        hB_arr[k] = _hB

        if k < num_steps - 1:
            _xA, _yA, _vA, _hA = step_ctrv(_xA, _yA, _vA, _hA, float(yawA), dt_pred)
            _xB, _yB, _vB, _hB = step_ctrv(_xB, _yB, _vB, _hB, float(yawB), dt_pred)

    relx_arr, rely_arr, axes_x, axes_y, sum_radius, R2 = _prepare_support_sat_data_jit(
        xA_arr,
        yA_arr,
        hA_arr,
        xB_arr,
        yB_arr,
        hB_arr,
        float(lA),
        float(wA),
        float(lB),
        float(wB),
    )

    return {
        "t_arr": t_arr,
        "t2_arr": t2_arr,
        "xA_arr": xA_arr,
        "yA_arr": yA_arr,
        "xB_arr": xB_arr,
        "yB_arr": yB_arr,
        "hA_arr": hA_arr,
        "hB_arr": hB_arr,
        "relx_arr": relx_arr,
        "rely_arr": rely_arr,
        "axes_x": axes_x,
        "axes_y": axes_y,
        "sum_radius": sum_radius,
        "R2": float(R2),
    }


# ============================================================================
# Analytical EA_CVCV prerequisites and infinite-horizon solver
# ============================================================================

def _point_to_obb_local(
    px: float,
    py: float,
    cx: float,
    cy: float,
    cos_h: float,
    sin_h: float,
) -> Tuple[float, float]:
    dx = px - cx
    dy = py - cy
    local_x = dx * cos_h + dy * sin_h
    local_y = -dx * sin_h + dy * cos_h
    return local_x, local_y


def _is_point_in_obb(
    px: float,
    py: float,
    cx: float,
    cy: float,
    h: float,
    length: float,
    width: float,
    eps: float = 1e-12,
) -> bool:
    cos_h = math.cos(h)
    sin_h = math.sin(h)
    local_x, local_y = _point_to_obb_local(px, py, cx, cy, cos_h, sin_h)
    return (
        abs(local_x) <= 0.5 * length + eps
        and abs(local_y) <= 0.5 * width + eps
    )


def _segments_intersect(
    p1: Sequence[float],
    p2: Sequence[float],
    q1: Sequence[float],
    q2: Sequence[float],
    eps: float = 1e-12,
) -> bool:
    def cross(ax: float, ay: float, bx: float, by: float) -> float:
        return ax * by - ay * bx

    def on_segment(a: Sequence[float], b: Sequence[float], p: Sequence[float]) -> bool:
        return (
            min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
        )

    ax, ay = p1
    bx, by = p2
    cx, cy = q1
    dx, dy = q2

    abx, aby = bx - ax, by - ay
    acx, acy = cx - ax, cy - ay
    adx, ady = dx - ax, dy - ay
    cdx, cdy = dx - cx, dy - cy
    cax, cay = ax - cx, ay - cy
    cbx, cby = bx - cx, by - cy

    d1 = cross(abx, aby, acx, acy)
    d2 = cross(abx, aby, adx, ady)
    d3 = cross(cdx, cdy, cax, cay)
    d4 = cross(cdx, cdy, cbx, cby)

    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and (
        (d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)
    ):
        return True

    if abs(d1) <= eps and on_segment(p1, p2, q1):
        return True
    if abs(d2) <= eps and on_segment(p1, p2, q2):
        return True
    if abs(d3) <= eps and on_segment(q1, q2, p1):
        return True
    if abs(d4) <= eps and on_segment(q1, q2, p2):
        return True

    return False


def _check_current_collision_obb(
    xA: float,
    yA: float,
    hA: float,
    lA: float,
    wA: float,
    xB: float,
    yB: float,
    hB: float,
    lB: float,
    wB: float,
    eps: float = 1e-12,
) -> bool:
    ca, sa = math.cos(hA), math.sin(hA)
    cb, sb = math.cos(hB), math.sin(hB)
    hlA, hwA = 0.5 * lA, 0.5 * wA
    hlB, hwB = 0.5 * lB, 0.5 * wB

    Ax0 = xA - hlA * ca - hwA * sa
    Ay0 = yA - hlA * sa + hwA * ca
    Ax1 = xA + hlA * ca - hwA * sa
    Ay1 = yA + hlA * sa + hwA * ca
    Ax2 = xA + hlA * ca + hwA * sa
    Ay2 = yA + hlA * sa - hwA * ca
    Ax3 = xA - hlA * ca + hwA * sa
    Ay3 = yA - hlA * sa - hwA * ca

    Bx0 = xB - hlB * cb - hwB * sb
    By0 = yB - hlB * sb + hwB * cb
    Bx1 = xB + hlB * cb - hwB * sb
    By1 = yB + hlB * sb + hwB * cb
    Bx2 = xB + hlB * cb + hwB * sb
    By2 = yB + hlB * sb - hwB * cb
    Bx3 = xB - hlB * cb + hwB * sb
    By3 = yB - hlB * sb - hwB * cb

    A_pts = ((Ax0, Ay0), (Ax1, Ay1), (Ax2, Ay2), (Ax3, Ay3))
    B_pts = ((Bx0, By0), (Bx1, By1), (Bx2, By2), (Bx3, By3))

    for px, py in A_pts:
        if _is_point_in_obb(px, py, xB, yB, hB, lB, wB, eps=eps):
            return True

    for px, py in B_pts:
        if _is_point_in_obb(px, py, xA, yA, hA, lA, wA, eps=eps):
            return True

    for i in range(4):
        a1 = A_pts[i]
        a2 = A_pts[(i + 1) & 3]
        for j in range(4):
            b1 = B_pts[j]
            b2 = B_pts[(j + 1) & 3]
            if _segments_intersect(a1, a2, b1, b2, eps=eps):
                return True

    return False


def _get_rect_corners_for_gate(x: float, y: float, h: float, l: float, w: float) -> List[List[float]]:
    ch, sh = math.cos(h), math.sin(h)
    corners = [
        [-l / 2.0 * ch - w / 2.0 * sh, -l / 2.0 * sh + w / 2.0 * ch],
        [l / 2.0 * ch - w / 2.0 * sh, l / 2.0 * sh + w / 2.0 * ch],
        [l / 2.0 * ch + w / 2.0 * sh, l / 2.0 * sh - w / 2.0 * ch],
        [-l / 2.0 * ch + w / 2.0 * sh, -l / 2.0 * sh - w / 2.0 * ch],
    ]
    corners = [[x + cx, y + cy] for cx, cy in corners]
    return corners


def _distance_point(p1: Sequence[float], p2: Sequence[float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _point_to_segment_distance_for_gate(
    p: Sequence[float],
    v1: Sequence[float],
    v2: Sequence[float],
) -> Tuple[float, List[float]]:
    line_len = _distance_point(v1, v2)
    if line_len == 0:
        return _distance_point(p, v1), [v1[0], v1[1]]

    t = max(
        0.0,
        min(
            1.0,
            ((p[0] - v1[0]) * (v2[0] - v1[0]) + (p[1] - v1[1]) * (v2[1] - v1[1]))
            / (line_len ** 2),
        ),
    )
    closest = [v1[0] + t * (v2[0] - v1[0]), v1[1] + t * (v2[1] - v1[1])]
    return _distance_point(p, closest), closest


def _get_shortest_distance_for_gate(
    corners_A: Sequence[Sequence[float]],
    corners_B: Sequence[Sequence[float]],
) -> Tuple[float, Optional[Sequence[float]], Optional[Sequence[float]]]:
    min_distance = float("inf")
    closest_A = None
    closest_B = None

    for i in range(4):
        p1 = corners_A[i]
        for k in range(4):
            p2 = corners_B[k]
            p3 = corners_B[(k + 1) % 4]
            dist, closest = _point_to_segment_distance_for_gate(p1, p2, p3)
            if dist < min_distance:
                min_distance = dist
                closest_A = p1
                closest_B = closest

        p1 = corners_B[i]
        for k in range(4):
            p2 = corners_A[k]
            p3 = corners_A[(k + 1) % 4]
            dist, closest = _point_to_segment_distance_for_gate(p1, p2, p3)
            if dist < min_distance:
                min_distance = dist
                closest_A = closest
                closest_B = p1

    return min_distance, closest_A, closest_B


def _compute_shortest_distance_for_gate(
    x_A: float,
    y_A: float,
    v_A: float,
    h_A: float,
    l_A: float,
    w_A: float,
    x_B: float,
    y_B: float,
    v_B: float,
    h_B: float,
    l_B: float,
    w_B: float,
) -> Tuple[float, Tuple[Sequence[float], Sequence[float]], float]:
    corners_A = _get_rect_corners_for_gate(x_A, y_A, h_A, l_A, w_A)
    corners_B = _get_rect_corners_for_gate(x_B, y_B, h_B, l_B, w_B)

    min_distance, closest_A, closest_B = _get_shortest_distance_for_gate(corners_A, corners_B)

    delta_x = closest_B[0] - closest_A[0]
    delta_y = closest_B[1] - closest_A[1]
    norm_delta = math.sqrt(delta_x ** 2 + delta_y ** 2)

    if norm_delta != 0:
        unit_vector = np.array([delta_x / norm_delta, delta_y / norm_delta], dtype=float)
        velocity_diff = np.array(
            [
                v_B * math.cos(h_B) - v_A * math.cos(h_A),
                v_B * math.sin(h_B) - v_A * math.sin(h_A),
            ],
            dtype=float,
        )
        v_closest = -float(np.dot(unit_vector, velocity_diff))
    else:
        v_closest = 0.0

    return min_distance, (closest_A, closest_B), v_closest


def _compute_tdm_indepth_for_gate(
    x_A: float,
    y_A: float,
    v_A: float,
    h_A: float,
    l_A: float,
    w_A: float,
    x_B: float,
    y_B: float,
    v_B: float,
    h_B: float,
    l_B: float,
    w_B: float,
) -> Tuple[Optional[float], Optional[float]]:
    D_SAFE = 0.0

    v_diff = np.array(
        [
            v_B * math.cos(h_B) - v_A * math.cos(h_A),
            v_B * math.sin(h_B) - v_A * math.sin(h_A),
        ],
        dtype=float,
    )
    v_diff_norm = np.linalg.norm(v_diff)
    if v_diff_norm < 1e-12:
        return None, None

    theta_B_prime = v_diff / v_diff_norm
    delta = np.array([x_B - x_A, y_B - y_A], dtype=float)
    d_t1 = np.linalg.norm(delta - np.dot(delta, theta_B_prime) * theta_B_prime)

    chA, shA = math.cos(h_A), math.sin(h_A)
    chB, shB = math.cos(h_B), math.sin(h_B)

    AA1 = np.array([l_A / 2 * chA - w_A / 2 * -shA, l_A / 2 * shA - w_A / 2 * chA], dtype=float)
    AA2 = np.array([l_A / 2 * chA + w_A / 2 * -shA, l_A / 2 * shA + w_A / 2 * chA], dtype=float)
    AA3 = np.array([-l_A / 2 * chA - w_A / 2 * -shA, -l_A / 2 * shA - w_A / 2 * chA], dtype=float)
    AA4 = np.array([-l_A / 2 * chA + w_A / 2 * -shA, -l_A / 2 * shA + w_A / 2 * chA], dtype=float)

    d_A_max = np.max(np.array([
        np.linalg.norm(AA1 - np.dot(AA1, theta_B_prime) * theta_B_prime),
        np.linalg.norm(AA2 - np.dot(AA2, theta_B_prime) * theta_B_prime),
        np.linalg.norm(AA3 - np.dot(AA3, theta_B_prime) * theta_B_prime),
        np.linalg.norm(AA4 - np.dot(AA4, theta_B_prime) * theta_B_prime),
    ], dtype=float))

    BB1 = np.array([l_B / 2 * chB - w_B / 2 * -shB, l_B / 2 * shB - w_B / 2 * chB], dtype=float)
    BB2 = np.array([l_B / 2 * chB + w_B / 2 * -shB, l_B / 2 * shB + w_B / 2 * chB], dtype=float)
    BB3 = np.array([-l_B / 2 * chB - w_B / 2 * -shB, -l_B / 2 * shB - w_B / 2 * chB], dtype=float)
    BB4 = np.array([-l_B / 2 * chB + w_B / 2 * -shB, -l_B / 2 * shB + w_B / 2 * chB], dtype=float)

    d_B_max = np.max(np.array([
        np.linalg.norm(BB1 - np.dot(BB1, theta_B_prime) * theta_B_prime),
        np.linalg.norm(BB2 - np.dot(BB2, theta_B_prime) * theta_B_prime),
        np.linalg.norm(BB3 - np.dot(BB3, theta_B_prime) * theta_B_prime),
        np.linalg.norm(BB4 - np.dot(BB4, theta_B_prime) * theta_B_prime),
    ], dtype=float))

    mfd = d_t1 - (d_A_max + d_B_max)
    d_B_prime = -float(np.dot(delta, theta_B_prime))
    tdm = d_B_prime / v_diff_norm if v_diff_norm != 0 else None
    indepth = D_SAFE - mfd

    return tdm, indepth


def _compute_ttc2d_for_analytical_cvcv(
    x_A: float,
    y_A: float,
    v_A: float,
    h_A: float,
    l_A: float,
    w_A: float,
    x_B: float,
    y_B: float,
    v_B: float,
    h_B: float,
    l_B: float,
    w_B: float,
) -> Tuple[float, float, float]:
    def is_ray_intersect_segment(
        ray_origin_x: float,
        ray_origin_y: float,
        ray_direction_x: float,
        ray_direction_y: float,
        segment_start_x: float,
        segment_start_y: float,
        segment_end_x: float,
        segment_end_y: float,
    ) -> Optional[float]:
        ray_origin = np.array([ray_origin_x, ray_origin_y], dtype=float)
        ray_direction = np.array([ray_direction_x, ray_direction_y], dtype=float)
        segment_start = np.array([segment_start_x, segment_start_y], dtype=float)
        segment_end = np.array([segment_end_x, segment_end_y], dtype=float)

        ray_norm = np.linalg.norm(ray_direction)
        if ray_norm < 1e-12:
            return None

        v1 = ray_origin - segment_start
        v2 = segment_end - segment_start
        v3 = np.array([-ray_direction[1], ray_direction[0]], dtype=float)
        v3_norm = np.linalg.norm(v3)
        if v3_norm < 1e-12:
            return None
        v3 = v3 / v3_norm

        dot = float(np.dot(v2, v3))
        if abs(dot) < 1e-10:
            if abs(np.cross(v1, v2)) < 1e-10:
                t0 = float(np.dot(segment_start - ray_origin, ray_direction) / (ray_norm ** 2))
                t1 = float(np.dot(segment_end - ray_origin, ray_direction) / (ray_norm ** 2))
                if t0 >= 0 and t1 >= 0:
                    return min(t0, t1) * ray_norm
                if t0 < 0 and t1 < 0:
                    return None
                return 0.0
            return None

        t1 = float(np.cross(v2, v1) / dot)
        t2 = float(np.dot(v1, v3) / dot)
        if 0 <= t2 <= 1:
            return t1
        return None

    cA, sA = math.cos(h_A), math.sin(h_A)
    cB, sB = math.cos(h_B), math.sin(h_B)

    rot_A = np.array([[cA, sA], [-sA, cA]], dtype=float)
    rot_B = np.array([[cB, sB], [-sB, cB]], dtype=float)

    bbox_A = np.array([x_A, y_A], dtype=float) + np.dot(
        np.array(
            [
                [l_A / 2, w_A / 2],
                [l_A / 2, -w_A / 2],
                [-l_A / 2, -w_A / 2],
                [-l_A / 2, w_A / 2],
            ],
            dtype=float,
        ),
        rot_A,
    )

    bbox_B = np.array([x_B, y_B], dtype=float) + np.dot(
        np.array(
            [
                [l_B / 2, w_B / 2],
                [l_B / 2, -w_B / 2],
                [-l_B / 2, -w_B / 2],
                [-l_B / 2, w_B / 2],
            ],
            dtype=float,
        ),
        rot_B,
    )

    v_A_vec = np.array([v_A * cA, v_A * sA], dtype=float)
    v_B_vec = np.array([v_B * cB, v_B * sB], dtype=float)
    v_rel = v_A_vec - v_B_vec
    dtc = np.nan

    for i in range(4):
        neg_flag = False
        for j in range(4):
            dist = is_ray_intersect_segment(
                *bbox_A[i],
                *v_rel,
                *bbox_B[j],
                *bbox_B[(j + 1) % 4],
            )
            if dist is not None:
                if np.isnan(dtc) or (0 < dist < dtc):
                    dtc = dist
                if dist < 0:
                    neg_flag = True
                if neg_flag and dist > 0:
                    return 0.0, 0.0, float(np.linalg.norm(v_rel))

    for i in range(4):
        neg_flag = False
        for j in range(4):
            dist = is_ray_intersect_segment(
                *bbox_B[i],
                *(-v_rel),
                *bbox_A[j],
                *bbox_A[(j + 1) % 4],
            )
            if dist is not None:
                if np.isnan(dtc) or (0 <= dist < dtc):
                    dtc = dist
                if dist < 0:
                    neg_flag = True
                if neg_flag and dist > 0:
                    return 0.0, 0.0, float(np.linalg.norm(v_rel))

    v_rel_norm = float(np.linalg.norm(v_rel))
    if not np.isnan(dtc) and v_rel_norm > 1e-12:
        ttc2d = dtc / v_rel_norm
        return ttc2d, dtc, v_rel_norm

    return np.nan, np.nan, np.nan


def _compute_radial_solution_analytical(d_R: float, v_R: float) -> Tuple[str, float, float, float, float]:
    a_R = v_R ** 2 / (2.0 * d_R)
    a_T = 0.0
    f = a_R ** 2 + a_T ** 2
    sqrt_f = math.sqrt(f)
    label = "radial segment (a_T = 0)"
    return label, a_R, a_T, f, sqrt_f


def _compute_tangential_solutions_analytical(
    d_R: float,
    d_T: float,
    v_R: float,
) -> Tuple[List[Tuple[str, float, float, float, float]], Tuple[str, float, float, float, float]]:
    c1 = v_R * math.sqrt(2.0 / d_T)
    c2 = -d_R / d_T
    A = 2.0 * (c2 ** 2 + 1.0)
    B = 3.0 * c1 * c2
    C = c1 ** 2
    delta = B ** 2 - 4.0 * A * C
    tangents = []

    if d_R ** 2 >= 8.0 * d_T ** 2 and delta >= 0.0:
        sqrt_delta = math.sqrt(delta)
        for i, sign in enumerate([1.0, -1.0], start=1):
            x = (-B + sign * sqrt_delta) / (2.0 * A)
            if x > 0.0:
                a_T = x ** 2
                a_R = c1 * x + c2 * a_T
                f_val = a_R ** 2 + a_T ** 2
                sqrt_f = math.sqrt(f_val)
                label = f"analytical local-minimum candidate {i}"
                tangents.append((label, a_R, a_T, f_val, sqrt_f))

    a_R_geo = v_R ** 2 / (2.0 * d_R)
    a_T_geo = d_T * v_R ** 2 / (2.0 * d_R ** 2)
    f_geo = (v_R ** 4) / (4.0 * d_R ** 2) * (1.0 + (d_T ** 2) / (d_R ** 2))
    sqrt_f_geo = math.sqrt(f_geo)
    tangents.append(("geometric candidate", a_R_geo, a_T_geo, f_geo, sqrt_f_geo))

    best = min(tangents, key=lambda t: t[3])
    return tangents, best


def _compute_intersection_analytical(
    d_R1: float,
    d_T1: float,
    d_R2: float,
    d_T2: float,
    v_R: float,
) -> Optional[Tuple[float, float]]:
    if d_R1 > d_R2:
        d_R1, d_R2 = d_R2, d_R1
        d_T1, d_T2 = d_T2, d_T1

    if abs(d_R1 - d_R2) < 1e-8:
        return None

    ratio1 = d_T1 / (d_R1 ** 2)
    ratio2 = d_T2 / (d_R2 ** 2)
    if ratio1 >= ratio2:
        return None

    denom = (d_R1 / d_T1) - (d_R2 / d_T2)
    sqrt_diff = math.sqrt(2.0 / d_T1) - math.sqrt(2.0 / d_T2)
    if denom == 0.0 or sqrt_diff == 0.0:
        return None

    a_T_star = (v_R * sqrt_diff / denom) ** 2
    lower_bound_1 = d_T1 * v_R ** 2 / (2.0 * d_R1 ** 2)
    lower_bound_2 = d_T2 * v_R ** 2 / (2.0 * d_R2 ** 2)
    lower_bound = max(lower_bound_1, lower_bound_2)

    if a_T_star >= lower_bound:
        sqrt_term = math.sqrt(2.0 * a_T_star / d_T1)
        a_R_star = v_R * sqrt_term - (d_R1 / d_T1) * a_T_star
        return a_R_star, a_T_star

    t2_const = v_R ** 2 / (2.0 * d_R2)
    A = d_R1 / d_T1
    B = -v_R * math.sqrt(2.0 / d_T1)
    C = t2_const
    delta = B ** 2 - 4.0 * A * C
    if delta < 0.0:
        return None

    sqrt_delta = math.sqrt(delta)
    for x in [(-B + sqrt_delta) / (2.0 * A), (-B - sqrt_delta) / (2.0 * A)]:
        if x > 0.0:
            a_T_alt = x ** 2
            if lower_bound_1 <= a_T_alt < lower_bound_2:
                a_R_alt = t2_const
                return a_R_alt, a_T_alt

    return None


def _is_outer_point_except_analytical(
    aR: float,
    aT: float,
    d_Rs: np.ndarray,
    d_Ts: np.ndarray,
    v_R: float,
    idxs: Sequence[int],
    excl: Sequence[int],
) -> bool:
    for idx in idxs:
        if idx in excl:
            continue

        dR, dT = d_Rs[idx], d_Ts[idx]
        if dT == 0:
            aR_cmp = v_R ** 2 / (2.0 * dR)
        else:
            if aR >= v_R ** 2 / (2.0 * dR) + 1e-12:
                continue
            aR_cmp = v_R * math.sqrt(2.0 * aT / dT) - (dR / dT) * aT

        if aR <= aR_cmp + 1e-12:
            return False

    return True


def _compute_case_general_analytical(
    d_Rs: np.ndarray,
    d_Ts: np.ndarray,
    v_R: float,
    has_SB0: bool,
) -> Tuple[str, float, float, float, float]:
    offset = 1 if has_SB0 else 0
    idxs = list(range(offset, len(d_Rs)))

    rad_lbl, aR_rad, aT_rad, f_rad, sqrt_rad = _compute_radial_solution_analytical(d_Rs[0], v_R)

    candidates = []
    for k_idx, sb_idx in enumerate(idxs, start=1):
        _, best = _compute_tangential_solutions_analytical(d_Rs[sb_idx], d_Ts[sb_idx], v_R)
        _, aR_t, aT_t, f_t, sqrt_t = best
        candidates.append((f"tan_SB{k_idx}", aR_t, aT_t, f_t, sqrt_t, {sb_idx}))

    for i, j in combinations(idxs, 2):
        pt = _compute_intersection_analytical(d_Rs[i], d_Ts[i], d_Rs[j], d_Ts[j], v_R)
        if pt is None:
            raise RuntimeError(
                f"An intersection between SB{i} and SB{j} was expected but not found. "
                "Please check the upstream filtering logic."
            )
        aR_ij, aT_ij = pt
        f_ij = aR_ij ** 2 + aT_ij ** 2
        candidates.append((f"i{i}-{j}", aR_ij, aT_ij, f_ij, math.sqrt(f_ij), {i, j}))

    outer = [
        c for c in candidates
        if _is_outer_point_except_analytical(c[1], c[2], d_Rs, d_Ts, v_R, idxs, c[5])
    ]
    if not outer:
        raise RuntimeError(
            "No outer candidate point was found. Please check the step-barrier data "
            "or the selection logic."
        )

    best_outer = min(outer, key=lambda x: x[3])
    best = (rad_lbl, aR_rad, aT_rad, f_rad, sqrt_rad) if f_rad <= best_outer[3] else best_outer[:5]
    return best


def _process_step_barrier_list_analytical(
    d_Rs: np.ndarray,
    d_Ts: np.ndarray,
    label: str,
    v_R: float,
) -> Tuple[str, float, float, float, float]:
    if len(d_Rs) == 0:
        return (f"{label}-no-risk", 0.0, 0.0, 0.0, 0.0)

    has_SB0 = d_Ts[0] == 0
    sb_count = len(d_Rs) - (1 if has_SB0 else 0)

    if sb_count == 0:
        return _compute_radial_solution_analytical(d_Rs[0], v_R)
    if sb_count == 1:
        radial = _compute_radial_solution_analytical(d_Rs[0], v_R)
        idx1 = 1 if (has_SB0 and len(d_Rs) > 1) else 0
        _, aR_t, aT_t, f_t, _ = _compute_tangential_solutions_analytical(d_Rs[idx1], d_Ts[idx1], v_R)[1]
        tangential = ("tangential optimum", aR_t, aT_t, f_t, math.sqrt(f_t))
        return radial if radial[3] <= f_t else tangential

    return _compute_case_general_analytical(d_Rs, d_Ts, v_R, has_SB0)


def _select_critical_step_barrier_analytical(
    d_R_list: Sequence[float],
    d_T_list: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    if len(d_R_list) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    unique_pairs: Dict[float, float] = {}
    for dR, dT in zip(d_R_list, d_T_list):
        dR = float(dR)
        dT = float(dT)
        prev = unique_pairs.get(dR)
        if prev is None or dT > prev:
            unique_pairs[dR] = dT

    items = sorted(unique_pairs.items(), key=lambda x: x[0])
    d_Rs = np.array([it[0] for it in items], dtype=float)
    d_Ts = np.array([it[1] for it in items], dtype=float)

    keep_idx = []
    max_ratio = -np.inf
    for i in range(len(d_Rs)):
        ratio = d_Ts[i] / (d_Rs[i] ** 2)
        if ratio > max_ratio + 1e-15:
            keep_idx.append(i)
            max_ratio = ratio

    return d_Rs[keep_idx], d_Ts[keep_idx]


def _ea_sampled_points_from_vehicle_analytical(
    x: float,
    y: float,
    h: float,
    L: float,
    W: float,
    light_side: str = "right",
    density: int = 24,
) -> np.ndarray:
    c, s = math.cos(h), math.sin(h)
    hl, hw = 0.5 * L, 0.5 * W

    p1 = (x + hl * c - hw * s, y + hl * s + hw * c)
    p2 = (x + hl * c + hw * s, y + hl * s - hw * c)
    p3 = (x - hl * c + hw * s, y - hl * s - hw * c)
    p4 = (x - hl * c - hw * s, y - hl * s + hw * c)
    p5 = ((p1[0] + p4[0]) * 0.5, (p1[1] + p4[1]) * 0.5)
    p6 = ((p2[0] + p3[0]) * 0.5, (p2[1] + p3[1]) * 0.5)
    p7 = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
    p8 = ((p2[0] + p6[0]) * 0.5, (p2[1] + p6[1]) * 0.5)
    p9 = ((p6[0] + p3[0]) * 0.5, (p6[1] + p3[1]) * 0.5)
    p10 = ((p3[0] + p4[0]) * 0.5, (p3[1] + p4[1]) * 0.5)
    p11 = ((p4[0] + p5[0]) * 0.5, (p4[1] + p5[1]) * 0.5)
    p12 = ((p5[0] + p1[0]) * 0.5, (p5[1] + p1[1]) * 0.5)

    edges = (
        (p1, p7, p2),
        (p2, p8, p6, p9, p3),
        (p3, p10, p4),
        (p4, p11, p5, p12, p1),
    )
    normals = ((c, s), (s, -c), (-c, -s), (-s, c))

    lx = 1.0 if light_side == "right" else -1.0
    illum = [i for i, n in enumerate(normals) if n[0] * lx > 1e-12]

    if density not in (12, 24, 48):
        raise ValueError("density must be one of 12, 24, or 48.")

    pts = []

    def add_pt(pt: Sequence[float]) -> None:
        pts.append((pt[0], pt[1]))

    def sample_edge(edge_pts: Sequence[Sequence[float]]) -> None:
        if density == 12:
            for pt in edge_pts[:-1]:
                add_pt(pt)
            add_pt(edge_pts[-1])

        elif density == 24:
            for i in range(len(edge_pts) - 1):
                a = edge_pts[i]
                b = edge_pts[i + 1]
                m = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
                add_pt(a)
                add_pt(m)
            add_pt(edge_pts[-1])

        else:  # density == 48
            for i in range(len(edge_pts) - 1):
                a = edge_pts[i]
                b = edge_pts[i + 1]
                m = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
                q = ((a[0] + m[0]) * 0.5, (a[1] + m[1]) * 0.5)
                r = ((m[0] + b[0]) * 0.5, (m[1] + b[1]) * 0.5)
                add_pt(a)
                add_pt(q)
                add_pt(m)
                add_pt(r)
            add_pt(edge_pts[-1])

    for idx in illum:
        sample_edge(edges[idx])

    seen = set()
    out = []
    for px, py in pts:
        key = (round(px, 12), round(py, 12))
        if key not in seen:
            seen.add(key)
            out.append((px, py))

    return np.array(out, dtype=float)


def _compute_global_ea_cv_analytical_core(
    xA: float,
    yA: float,
    vA: float,
    hA: float,
    lA: float,
    wA: float,
    xB: float,
    yB: float,
    vB: float,
    hB: float,
    lB: float,
    wB: float,
    DTC: Optional[float] = None,
    v_rel_mag: Optional[float] = None,
) -> Tuple[str, float, float, float, float]:
    phi1 = math.atan2(
        vB * math.sin(hB) - vA * math.sin(hA),
        vB * math.cos(hB) - vA * math.cos(hA),
    )

    cphi = math.cos(phi1)
    sphi = math.sin(phi1)

    dx = xA - xB
    dy = yA - yB
    xA_p = dx * cphi + dy * sphi
    yA_p = -dx * sphi + dy * cphi
    hA_p = hA - phi1
    xB_p, yB_p, hB_p = 0.0, 0.0, hB - phi1

    A_corners_new = _ea_sampled_points_from_vehicle_analytical(
        xA_p, yA_p, hA_p, lA, wA, light_side="left", density=24
    )
    B_corners_new = _ea_sampled_points_from_vehicle_analytical(
        xB_p, yB_p, hB_p, lB, wB, light_side="right", density=24
    )

    xA_vals = A_corners_new[:, 0]
    yA_vals = A_corners_new[:, 1]
    xB_vals = B_corners_new[:, 0]
    yB_vals = B_corners_new[:, 1]

    upper_exists = False
    lower_exists = False

    if xA_vals.max() > xB_vals.min():
        if xA_vals.min() > xB_vals.max():
            upper_exists = True
            lower_exists = True
        else:
            idx_min_xA = int(np.argmin(xA_vals))
            idx_max_xB = int(np.argmax(xB_vals))
            if yA_vals[idx_min_xA] > yB_vals[idx_max_xB]:
                upper_exists = True
            elif yA_vals[idx_min_xA] < yB_vals[idx_max_xB]:
                lower_exists = True

    if DTC is None or v_rel_mag is None:
        _, dtc_local, v_rel_mag_local = _compute_ttc2d_for_analytical_cvcv(
            xA, yA, vA, hA, lA, wA,
            xB, yB, vB, hB, lB, wB,
        )
        if DTC is None:
            DTC = dtc_local
        if v_rel_mag is None:
            v_rel_mag = 0.0 if np.isnan(v_rel_mag_local) else v_rel_mag_local

    if v_rel_mag is None or np.isnan(v_rel_mag):
        v_rel_mag = 0.0

    upper_pairs_dR = np.array([], dtype=float)
    upper_pairs_dT = np.array([], dtype=float)
    lower_pairs_dR = np.array([], dtype=float)
    lower_pairs_dT = np.array([], dtype=float)

    XA = xA_vals[:, None]
    YA = yA_vals[:, None]
    XB = xB_vals[None, :]
    YB = yB_vals[None, :]

    if upper_exists:
        mask_up = (XA > XB) & (YA < YB)
        if np.any(mask_up):
            upper_pairs_dR = np.round((XA - XB)[mask_up], 7).astype(float)
            upper_pairs_dT = np.round((YB - YA)[mask_up], 7).astype(float)

    if lower_exists:
        mask_lo = (XA > XB) & (YA > YB)
        if np.any(mask_lo):
            lower_pairs_dR = np.round((XA - XB)[mask_lo], 7).astype(float)
            lower_pairs_dT = np.round((YA - YB)[mask_lo], 7).astype(float)

    if DTC is not None and not np.isnan(DTC):
        upper_pairs_dR = np.insert(upper_pairs_dR, 0, float(DTC))
        upper_pairs_dT = np.insert(upper_pairs_dT, 0, 0.0)
        lower_pairs_dR = np.insert(lower_pairs_dR, 0, float(DTC))
        lower_pairs_dT = np.insert(lower_pairs_dT, 0, 0.0)

    if len(upper_pairs_dR) == 0 and len(lower_pairs_dR) == 0:
        return ("no-risk", 0.0, 0.0, 0.0, 0.0)

    d_Rs_up, d_Ts_up = _select_critical_step_barrier_analytical(upper_pairs_dR, upper_pairs_dT)
    d_Rs_lo, d_Ts_lo = _select_critical_step_barrier_analytical(lower_pairs_dR, lower_pairs_dT)

    upper_best = _process_step_barrier_list_analytical(d_Rs_up, d_Ts_up, "Upper", v_rel_mag)
    lower_best = _process_step_barrier_list_analytical(d_Rs_lo, d_Ts_lo, "Lower", v_rel_mag)

    final = upper_best if upper_best[3] <= lower_best[3] else lower_best
    return final


def compute_ea_cvcv_analytical_with_prerequisites(
    xA: float,
    yA: float,
    vA: float,
    hA: float,
    xB: float,
    yB: float,
    vB: float,
    hB: float,
    lA: float,
    wA: float,
    lB: float,
    wB: float,
) -> float:
    """
    Analytical EA_CVCV with the same prerequisite logic as the old analytical EA_CV path.

    Prerequisites:
      1) current collision -> 0
      2) v_closest <= 0 -> 0
      3) indepth <= 0 or invalid -> 0
      4) otherwise solve infinite-horizon analytical CVCV EA
    """
    if _check_current_collision_obb(
        xA, yA, hA, lA, wA,
        xB, yB, hB, lB, wB,
    ):
        return 0.0

    _, _, v_closest = _compute_shortest_distance_for_gate(
        xA, yA, vA, hA, lA, wA,
        xB, yB, vB, hB, lB, wB,
    )
    if not (v_closest > 0.0):
        return 0.0

    _, indepth = _compute_tdm_indepth_for_gate(
        xA, yA, vA, hA, lA, wA,
        xB, yB, vB, hB, lB, wB,
    )
    if indepth is None or (not np.isfinite(indepth)) or indepth <= 0.0:
        return 0.0

    try:
        res = _compute_global_ea_cv_analytical_core(
            xA, yA, vA, hA, lA, wA,
            xB, yB, vB, hB, lB, wB,
            DTC=None,
            v_rel_mag=None,
        )
    except RuntimeError:
        return float("nan")

    ea_val = res[4]
    if np.isnan(ea_val):
        return 0.0
    return float(ea_val)


# ============================================================================
# Two-stage search for one mode
# ============================================================================
def _compute_single_mode_ea_from_precomputed(
    coarse_data: Dict[str, np.ndarray],
    fine_data: Dict[str, np.ndarray] = None,
    *,
    coarse_sector_num: int = DEFAULT_COARSE_SECTOR_NUM,
    local_fine_half_window_deg: float = DEFAULT_LOCAL_FINE_HALF_WINDOW_DEG,
    local_fine_dir_num: int = DEFAULT_LOCAL_FINE_DIR_NUM,
    fine_refine_ratio: float = DEFAULT_FINE_REFINE_RATIO,
    a_max: float = DEFAULT_A_MAX,
    tol: float = DEFAULT_TOL,
) -> float:
    coarse_base_collide, _ = _check_collision_base_support_jit(
        coarse_data["relx_arr"],
        coarse_data["rely_arr"],
        coarse_data["axes_x"],
        coarse_data["axes_y"],
        coarse_data["sum_radius"],
        coarse_data["t_arr"],
        float(coarse_data["R2"]),
    )
    if not coarse_base_collide:
        return 0.0

    ux_coarse, uy_coarse = _get_sector_unit_vectors(int(coarse_sector_num))
    coarse_ea, coarse_best_idx, _ = _compute_ea_on_directions_interval_jit(
        coarse_data["relx_arr"],
        coarse_data["rely_arr"],
        coarse_data["axes_x"],
        coarse_data["axes_y"],
        coarse_data["sum_radius"],
        coarse_data["t_arr"],
        coarse_data["t2_arr"],
        float(coarse_data["R2"]),
        np.asarray(ux_coarse, dtype=np.float64),
        np.asarray(uy_coarse, dtype=np.float64),
        float(a_max),
        float(tol),
    )

    coarse_ea = float(coarse_ea)

    if np.isnan(coarse_ea):
        return float("nan")
    if coarse_ea == 0.0:
        return 0.0
    if coarse_best_idx < 0:
        return coarse_ea

    dtheta_coarse = 2.0 * math.pi / int(coarse_sector_num)
    theta_center = (coarse_best_idx + 0.5) * dtheta_coarse

    _, ux_local, uy_local = _build_local_unit_vectors(
        theta_center=theta_center,
        half_window_deg=float(local_fine_half_window_deg),
        local_dir_num=int(local_fine_dir_num),
    )

    best_local = float("nan")

    if fine_data is not None:
        fine_base_collide, _ = _check_collision_base_support_jit(
            fine_data["relx_arr"],
            fine_data["rely_arr"],
            fine_data["axes_x"],
            fine_data["axes_y"],
            fine_data["sum_radius"],
            fine_data["t_arr"],
            float(fine_data["R2"]),
        )
        if not fine_base_collide:
            return 0.0

    for i in range(len(ux_local)):
        uxi = float(ux_local[i])
        uyi = float(uy_local[i])

        m_coarse = float(
            _solve_direction_min_accel_interval_jit(
                coarse_data["relx_arr"],
                coarse_data["rely_arr"],
                coarse_data["axes_x"],
                coarse_data["axes_y"],
                coarse_data["sum_radius"],
                coarse_data["t_arr"],
                coarse_data["t2_arr"],
                float(coarse_data["R2"]),
                uxi,
                uyi,
                float(a_max),
                float(tol),
            )
        )

        if np.isnan(m_coarse):
            continue

        m_use = m_coarse

        if fine_data is not None and m_coarse <= coarse_ea * float(fine_refine_ratio) + float(tol):
            m_fine = float(
                _solve_direction_min_accel_interval_jit(
                    fine_data["relx_arr"],
                    fine_data["rely_arr"],
                    fine_data["axes_x"],
                    fine_data["axes_y"],
                    fine_data["sum_radius"],
                    fine_data["t_arr"],
                    fine_data["t2_arr"],
                    float(fine_data["R2"]),
                    uxi,
                    uyi,
                    float(a_max),
                    float(tol),
                )
            )
            if not np.isnan(m_fine):
                m_use = m_fine
            else:
                continue

        if np.isnan(best_local) or (m_use < best_local):
            best_local = float(m_use)

    if np.isnan(best_local):
        return coarse_ea

    return float(best_local)


# ============================================================================
# Public function: single mode
# ============================================================================
def compute_single_mode_ea(
    xA: float,
    yA: float,
    vA: float,
    hA: float,
    xB: float,
    yB: float,
    vB: float,
    hB: float,
    lA: float,
    wA: float,
    lB: float,
    wB: float,
    yawA: float,
    yawB: float,
    *,
    coarse_sector_num: int = DEFAULT_COARSE_SECTOR_NUM,
    local_fine_half_window_deg: float = DEFAULT_LOCAL_FINE_HALF_WINDOW_DEG,
    local_fine_dir_num: int = DEFAULT_LOCAL_FINE_DIR_NUM,
    fine_refine_ratio: float = DEFAULT_FINE_REFINE_RATIO,
    a_max: float = DEFAULT_A_MAX,
    tol: float = DEFAULT_TOL,
    T_total: float = DEFAULT_T_TOTAL,
    dt_coarse: float = DEFAULT_DT_COARSE,
    dt_fine: float = DEFAULT_DT_FINE,
) -> float:
    """
    Compute EA for one specific motion mode.

    This function does not perform the four-mode aggregation. It returns the EA
    value for the supplied CTRV/CV motion setting only.
    """
    vals = [xA, yA, vA, hA, xB, yB, vB, hB, lA, wA, lB, wB, yawA, yawB]
    try:
        vals = [float(v) for v in vals]
    except Exception:
        return float("nan")

    if any(math.isnan(v) for v in vals):
        return float("nan")

    xA, yA, vA, hA, xB, yB, vB, hB, lA, wA, lB, wB, yawA, yawB = vals

    coarse_sector_num = int(coarse_sector_num)
    local_fine_dir_num = int(local_fine_dir_num)

    if coarse_sector_num < 4 or local_fine_dir_num < 3:
        return float("nan")
    if not (dt_coarse > 0 and dt_fine > 0 and T_total > 0 and a_max > 0 and tol > 0):
        return float("nan")
    if not (lA > 0 and wA > 0 and lB > 0 and wB > 0):
        return float("nan")

    coarse_data = prepare_ctrv_prediction(
        xA,
        yA,
        vA,
        hA,
        xB,
        yB,
        vB,
        hB,
        lA,
        wA,
        lB,
        wB,
        yawA,
        yawB,
        T_total=float(T_total),
        dt_pred=float(dt_coarse),
    )

    fine_data = None
    if dt_fine < dt_coarse:
        fine_data = prepare_ctrv_prediction(
            xA,
            yA,
            vA,
            hA,
            xB,
            yB,
            vB,
            hB,
            lA,
            wA,
            lB,
            wB,
            yawA,
            yawB,
            T_total=float(T_total),
            dt_pred=float(dt_fine),
        )

    ea_val = _compute_single_mode_ea_from_precomputed(
        coarse_data=coarse_data,
        fine_data=fine_data,
        coarse_sector_num=coarse_sector_num,
        local_fine_half_window_deg=float(local_fine_half_window_deg),
        local_fine_dir_num=local_fine_dir_num,
        fine_refine_ratio=float(fine_refine_ratio),
        a_max=float(a_max),
        tol=float(tol),
    )

    if ea_val is None or (isinstance(ea_val, float) and np.isnan(ea_val)):
        return float("nan")

    return float(ea_val)


def compute_single_mode_ea_timed(
    xA: float,
    yA: float,
    vA: float,
    hA: float,
    xB: float,
    yB: float,
    vB: float,
    hB: float,
    lA: float,
    wA: float,
    lB: float,
    wB: float,
    yawA: float,
    yawB: float,
    *,
    coarse_sector_num: int = DEFAULT_COARSE_SECTOR_NUM,
    local_fine_half_window_deg: float = DEFAULT_LOCAL_FINE_HALF_WINDOW_DEG,
    local_fine_dir_num: int = DEFAULT_LOCAL_FINE_DIR_NUM,
    fine_refine_ratio: float = DEFAULT_FINE_REFINE_RATIO,
    a_max: float = DEFAULT_A_MAX,
    tol: float = DEFAULT_TOL,
    T_total: float = DEFAULT_T_TOTAL,
    dt_coarse: float = DEFAULT_DT_COARSE,
    dt_fine: float = DEFAULT_DT_FINE,
) -> Tuple[float, float]:
    """Timed wrapper for one specific motion mode."""
    t0 = time.perf_counter()
    val = compute_single_mode_ea(
        xA,
        yA,
        vA,
        hA,
        xB,
        yB,
        vB,
        hB,
        lA,
        wA,
        lB,
        wB,
        yawA,
        yawB,
        coarse_sector_num=coarse_sector_num,
        local_fine_half_window_deg=local_fine_half_window_deg,
        local_fine_dir_num=local_fine_dir_num,
        fine_refine_ratio=fine_refine_ratio,
        a_max=a_max,
        tol=tol,
        T_total=T_total,
        dt_coarse=dt_coarse,
        dt_fine=dt_fine,
    )
    elapsed = time.perf_counter() - t0
    return val, elapsed


# ============================================================================
# Public function: four mode-specific values
# ============================================================================
def compute_ea_modes(
    xA: float,
    yA: float,
    vA: float,
    hA: float,
    xB: float,
    yB: float,
    vB: float,
    hB: float,
    lA: float,
    wA: float,
    lB: float,
    wB: float,
    yawA: float,
    yawB: float,
    *,
    coarse_sector_num: int = DEFAULT_COARSE_SECTOR_NUM,
    local_fine_half_window_deg: float = DEFAULT_LOCAL_FINE_HALF_WINDOW_DEG,
    local_fine_dir_num: int = DEFAULT_LOCAL_FINE_DIR_NUM,
    fine_refine_ratio: float = DEFAULT_FINE_REFINE_RATIO,
    a_max: float = DEFAULT_A_MAX,
    tol: float = DEFAULT_TOL,
    T_total: float = DEFAULT_T_TOTAL,
    dt_coarse: float = DEFAULT_DT_COARSE,
    dt_fine: float = DEFAULT_DT_FINE,
    size_scale: float = 1.0,
) -> Dict[str, float]:
    """
    Compute the four mode-specific EA values used to define the final EA.

    The returned dictionary contains:
        EA_CTCT
        EA_CTCV
        EA_CVCT
        EA_CVCV

    The optional `size_scale` multiplies both agents' length and width before
    the computation.
    """
    lA_s = float(lA) * float(size_scale)
    wA_s = float(wA) * float(size_scale)
    lB_s = float(lB) * float(size_scale)
    wB_s = float(wB) * float(size_scale)

    out: Dict[str, float] = {}

    # 1) CTCT
    out["EA_CTCT"] = compute_single_mode_ea(
        xA,
        yA,
        vA,
        hA,
        xB,
        yB,
        vB,
        hB,
        lA_s,
        wA_s,
        lB_s,
        wB_s,
        yawA,
        yawB,
        coarse_sector_num=coarse_sector_num,
        local_fine_half_window_deg=local_fine_half_window_deg,
        local_fine_dir_num=local_fine_dir_num,
        fine_refine_ratio=fine_refine_ratio,
        a_max=a_max,
        tol=tol,
        T_total=T_total,
        dt_coarse=dt_coarse,
        dt_fine=dt_fine,
    )

    # 2) CTCV: agent B uses CV (yawB = 0)
    out["EA_CTCV"] = compute_single_mode_ea(
        xA,
        yA,
        vA,
        hA,
        xB,
        yB,
        vB,
        hB,
        lA_s,
        wA_s,
        lB_s,
        wB_s,
        yawA,
        0.0,
        coarse_sector_num=coarse_sector_num,
        local_fine_half_window_deg=local_fine_half_window_deg,
        local_fine_dir_num=local_fine_dir_num,
        fine_refine_ratio=fine_refine_ratio,
        a_max=a_max,
        tol=tol,
        T_total=T_total,
        dt_coarse=dt_coarse,
        dt_fine=dt_fine,
    )

    # 3) CVCT: agent A uses CV (yawA = 0)
    out["EA_CVCT"] = compute_single_mode_ea(
        xA,
        yA,
        vA,
        hA,
        xB,
        yB,
        vB,
        hB,
        lA_s,
        wA_s,
        lB_s,
        wB_s,
        0.0,
        yawB,
        coarse_sector_num=coarse_sector_num,
        local_fine_half_window_deg=local_fine_half_window_deg,
        local_fine_dir_num=local_fine_dir_num,
        fine_refine_ratio=fine_refine_ratio,
        a_max=a_max,
        tol=tol,
        T_total=T_total,
        dt_coarse=dt_coarse,
        dt_fine=dt_fine,
    )

    # 4) CVCV: both agents use CV (yawA = yawB = 0)
    # Keep the old analytical EA_CV prerequisite logic, and solve CVCV
    # analytically on the infinite horizon.
    out["EA_CVCV"] = compute_ea_cvcv_analytical_with_prerequisites(
        xA,
        yA,
        vA,
        hA,
        xB,
        yB,
        vB,
        hB,
        lA_s,
        wA_s,
        lB_s,
        wB_s,
    )

    return out


def compute_ea_modes_timed(
    xA: float,
    yA: float,
    vA: float,
    hA: float,
    xB: float,
    yB: float,
    vB: float,
    hB: float,
    lA: float,
    wA: float,
    lB: float,
    wB: float,
    yawA: float,
    yawB: float,
    *,
    coarse_sector_num: int = DEFAULT_COARSE_SECTOR_NUM,
    local_fine_half_window_deg: float = DEFAULT_LOCAL_FINE_HALF_WINDOW_DEG,
    local_fine_dir_num: int = DEFAULT_LOCAL_FINE_DIR_NUM,
    fine_refine_ratio: float = DEFAULT_FINE_REFINE_RATIO,
    a_max: float = DEFAULT_A_MAX,
    tol: float = DEFAULT_TOL,
    T_total: float = DEFAULT_T_TOTAL,
    dt_coarse: float = DEFAULT_DT_COARSE,
    dt_fine: float = DEFAULT_DT_FINE,
    size_scale: float = 1.0,
) -> Tuple[Dict[str, float], float]:
    """Timed wrapper for the four mode-specific EA values."""
    t0 = time.perf_counter()
    out = compute_ea_modes(
        xA,
        yA,
        vA,
        hA,
        xB,
        yB,
        vB,
        hB,
        lA,
        wA,
        lB,
        wB,
        yawA,
        yawB,
        coarse_sector_num=coarse_sector_num,
        local_fine_half_window_deg=local_fine_half_window_deg,
        local_fine_dir_num=local_fine_dir_num,
        fine_refine_ratio=fine_refine_ratio,
        a_max=a_max,
        tol=tol,
        T_total=T_total,
        dt_coarse=dt_coarse,
        dt_fine=dt_fine,
        size_scale=size_scale,
    )
    elapsed = time.perf_counter() - t0
    return out, elapsed


# ============================================================================
# Public function: final EA
# ============================================================================
def compute_ea(
    xA: float,
    yA: float,
    vA: float,
    hA: float,
    xB: float,
    yB: float,
    vB: float,
    hB: float,
    lA: float,
    wA: float,
    lB: float,
    wB: float,
    yawA: float,
    yawB: float,
    *,
    coarse_sector_num: int = DEFAULT_COARSE_SECTOR_NUM,
    local_fine_half_window_deg: float = DEFAULT_LOCAL_FINE_HALF_WINDOW_DEG,
    local_fine_dir_num: int = DEFAULT_LOCAL_FINE_DIR_NUM,
    fine_refine_ratio: float = DEFAULT_FINE_REFINE_RATIO,
    a_max: float = DEFAULT_A_MAX,
    tol: float = DEFAULT_TOL,
    T_total: float = DEFAULT_T_TOTAL,
    dt_coarse: float = DEFAULT_DT_COARSE,
    dt_fine: float = DEFAULT_DT_FINE,
    size_scale: float = 1.0,
) -> float:
    """
    Compute the final EA value, defined as the arithmetic mean of the four
    mode-specific EA values.
    """
    mode_dict = compute_ea_modes(
        xA,
        yA,
        vA,
        hA,
        xB,
        yB,
        vB,
        hB,
        lA,
        wA,
        lB,
        wB,
        yawA,
        yawB,
        coarse_sector_num=coarse_sector_num,
        local_fine_half_window_deg=local_fine_half_window_deg,
        local_fine_dir_num=local_fine_dir_num,
        fine_refine_ratio=fine_refine_ratio,
        a_max=a_max,
        tol=tol,
        T_total=T_total,
        dt_coarse=dt_coarse,
        dt_fine=dt_fine,
        size_scale=size_scale,
    )
    return aggregate_ea_modes(mode_dict)


def compute_ea_timed(
    xA: float,
    yA: float,
    vA: float,
    hA: float,
    xB: float,
    yB: float,
    vB: float,
    hB: float,
    lA: float,
    wA: float,
    lB: float,
    wB: float,
    yawA: float,
    yawB: float,
    *,
    coarse_sector_num: int = DEFAULT_COARSE_SECTOR_NUM,
    local_fine_half_window_deg: float = DEFAULT_LOCAL_FINE_HALF_WINDOW_DEG,
    local_fine_dir_num: int = DEFAULT_LOCAL_FINE_DIR_NUM,
    fine_refine_ratio: float = DEFAULT_FINE_REFINE_RATIO,
    a_max: float = DEFAULT_A_MAX,
    tol: float = DEFAULT_TOL,
    T_total: float = DEFAULT_T_TOTAL,
    dt_coarse: float = DEFAULT_DT_COARSE,
    dt_fine: float = DEFAULT_DT_FINE,
    size_scale: float = 1.0,
) -> Tuple[float, Dict[str, float], float]:
    """
    Timed wrapper for the final EA.

    Returns:
        ea_value, mode_dict, elapsed_seconds
    """
    t0 = time.perf_counter()
    mode_dict = compute_ea_modes(
        xA,
        yA,
        vA,
        hA,
        xB,
        yB,
        vB,
        hB,
        lA,
        wA,
        lB,
        wB,
        yawA,
        yawB,
        coarse_sector_num=coarse_sector_num,
        local_fine_half_window_deg=local_fine_half_window_deg,
        local_fine_dir_num=local_fine_dir_num,
        fine_refine_ratio=fine_refine_ratio,
        a_max=a_max,
        tol=tol,
        T_total=T_total,
        dt_coarse=dt_coarse,
        dt_fine=dt_fine,
        size_scale=size_scale,
    )
    ea_value = aggregate_ea_modes(mode_dict)
    elapsed = time.perf_counter() - t0
    return ea_value, mode_dict, elapsed


# ============================================================================
# Minimal single-frame runnable demo
# ============================================================================
if __name__ == "__main__":
    test_kwargs = dict(
        xA=0.0,
        yA=0.0,
        vA=10.0,
        hA=0.0,
        xB=20.0,
        yB=0.0,
        vB=8.0,
        hB=math.pi,
        lA=4.5,
        wA=1.8,
        lB=4.7,
        wB=1.9,
        yawA=0.0,
        yawB=0.0,
        coarse_sector_num=72,
        local_fine_half_window_deg=5.0,
        local_fine_dir_num=101,
        fine_refine_ratio=1.15,
        a_max=100.0,
        tol=1e-3,
        T_total=10.0,
        dt_coarse=0.1,
        dt_fine=0.02,
        size_scale=1.0,
    )

    # Optional warm-up for a fairer runtime display, especially when numba is available.
    _ = compute_ea(**test_kwargs)
    _ = compute_ea_modes(**test_kwargs)

    ea_value, mode_dict, elapsed = compute_ea_timed(**test_kwargs)

    print("EA single-frame demo")
    print("--------------------")
    print("EA =", _format_float_3(ea_value))
    print("Runtime =", _format_seconds(elapsed))
    print("Mode-specific values:")
    for k, v in mode_dict.items():
        print(f"  {k} = {_format_float_3(v)}")