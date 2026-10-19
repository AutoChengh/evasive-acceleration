"""Low-level collision geometry and minimum-EA numerical search."""

import math
import numpy as np
from functools import lru_cache

# Optional numba acceleration
try:
    from numba import njit
except Exception:

    def njit(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap


# ==========================
# Default configs
# ==========================
DEFAULT_DT_COARSE = 0.1
DEFAULT_DT_FINE = 0.02
DEFAULT_A_MAX = 100.0
DEFAULT_TOL = 1e-3

DEFAULT_COARSE_SECTOR_NUM = 72
DEFAULT_LOCAL_FINE_HALF_WINDOW_DEG = 5
DEFAULT_LOCAL_FINE_DIR_NUM = 101
DEFAULT_FINE_REFINE_RATIO = 1.15
CURRENT_COLLISION_EPS = 1e-12


# ==========================
# Cached grids / directions
# ==========================
@lru_cache(maxsize=None)
def _get_time_grids(T_total: float, dt_pred: float):
    """
    Returns (t_arr, t2_arr, num_steps).
    ``t2_arr = 0.5 * t^2`` so constant acceleration displacement is
    ``a * t2_arr``.
    """
    T_total = float(T_total)
    dt_pred = float(dt_pred)
    num_steps = int(round(T_total / dt_pred)) + 1
    t_arr = np.linspace(0.0, T_total, num_steps)
    t2_arr = 0.5 * (t_arr ** 2)
    return t_arr.astype(np.float64), t2_arr.astype(np.float64), num_steps


@lru_cache(maxsize=None)
def _get_sector_unit_vectors(sector_num: int):
    """
    Returns (ux, uy) for directions at (i+0.5)*dtheta.
    Cached to avoid trig recompute per call.
    """
    sector_num = int(sector_num)
    dtheta = 2.0 * math.pi / sector_num
    thetas = (np.arange(sector_num, dtype=np.float64) + 0.5) * dtheta
    ux = np.cos(thetas)
    uy = np.sin(thetas)
    return ux.astype(np.float64), uy.astype(np.float64)


def _build_local_unit_vectors(theta_center: float,
                              half_window_deg: float,
                              local_dir_num: int):
    """
    Build local refined directions around theta_center.
    """
    half_window_rad = math.radians(float(half_window_deg))
    local_dir_num = int(local_dir_num)
    thetas = np.linspace(theta_center - half_window_rad,
                         theta_center + half_window_rad,
                         local_dir_num)
    ux = np.cos(thetas).astype(np.float64)
    uy = np.sin(thetas).astype(np.float64)
    return thetas.astype(np.float64), ux, uy


# ==========================
# Support-function SAT data
# ==========================
@njit(cache=True, fastmath=False)
def _prepare_support_sat_data_jit(
    xA_arr, yA_arr, hA_arr,
    xB_arr, yB_arr, hB_arr,
    lengthA, widthA,
    lengthB, widthB,
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
            halfLA * abs(a1x * n0x + a1y * n0y) +
            halfWA * abs(a2x * n0x + a2y * n0y) +
            halfLB * abs(b1x * n0x + b1y * n0y) +
            halfWB * abs(b2x * n0x + b2y * n0y)
        )

        n1x, n1y = a2x, a2y
        sum_radius[k, 1] = (
            halfLA * abs(a1x * n1x + a1y * n1y) +
            halfWA * abs(a2x * n1x + a2y * n1y) +
            halfLB * abs(b1x * n1x + b1y * n1y) +
            halfWB * abs(b2x * n1x + b2y * n1y)
        )

        n2x, n2y = b1x, b1y
        sum_radius[k, 2] = (
            halfLA * abs(a1x * n2x + a1y * n2y) +
            halfWA * abs(a2x * n2x + a2y * n2y) +
            halfLB * abs(b1x * n2x + b1y * n2y) +
            halfWB * abs(b2x * n2x + b2y * n2y)
        )

        n3x, n3y = b2x, b2y
        sum_radius[k, 3] = (
            halfLA * abs(a1x * n3x + a1y * n3y) +
            halfWA * abs(a2x * n3x + a2y * n3y) +
            halfLB * abs(b1x * n3x + b1y * n3y) +
            halfWB * abs(b2x * n3x + b2y * n3y)
        )

    rA = math.hypot(halfLA, halfWA)
    rB = math.hypot(halfLB, halfWB)
    R2 = (rA + rB) ** 2

    return relx_arr, rely_arr, axes_x, axes_y, sum_radius, R2


# ==========================
# Exact OBB collision checks
# ==========================
def _check_collision_obb_fast_params(xA, yA, hA, lA, wA, xB, yB, hB, lB, wB):
    vals = [xA, yA, hA, lA, wA, xB, yB, hB, lB, wB]
    try:
        vals = [float(v) for v in vals]
    except Exception:
        return False

    if any(not math.isfinite(v) for v in vals):
        return False

    xA, yA, hA, lA, wA, xB, yB, hB, lB, wB = vals
    if lA <= 0.0 or wA <= 0.0 or lB <= 0.0 or wB <= 0.0:
        return False

    ca = math.cos(hA)
    sa = math.sin(hA)
    cb = math.cos(hB)
    sb = math.sin(hB)

    A_long = (ca, sa)
    A_lat = (-sa, ca)
    B_long = (cb, sb)
    B_lat = (-sb, cb)

    dx = xB - xA
    dy = yB - yA

    hlA = 0.5 * lA
    hwA = 0.5 * wA
    hlB = 0.5 * lB
    hwB = 0.5 * wB

    axes = (A_long, A_lat, B_long, B_lat)

    for ax, ay in axes:
        center_dist = abs(dx * ax + dy * ay)

        rA = (
            hlA * abs(A_long[0] * ax + A_long[1] * ay) +
            hwA * abs(A_lat[0] * ax + A_lat[1] * ay)
        )
        rB = (
            hlB * abs(B_long[0] * ax + B_long[1] * ay) +
            hwB * abs(B_lat[0] * ax + B_lat[1] * ay)
        )

        if center_dist > rA + rB + CURRENT_COLLISION_EPS:
            return False

    return True


@njit(cache=True, fastmath=False)
def _check_collision_base_support_jit(
    relx_arr, rely_arr,
    axes_x, axes_y,
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
    relx_arr, rely_arr,
    axes_x, axes_y,
    sum_radius,
    t_arr, t2_arr,
    ax_extra, ay_extra,
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
    relx_arr, rely_arr,
    axes_x, axes_y,
    sum_radius,
    t_arr, t2_arr,
    ax_extra, ay_extra,
    R2,
):
    will_collide, _ = _check_collision_with_extra_accel_support_jit(
        relx_arr, rely_arr,
        axes_x, axes_y,
        sum_radius,
        t_arr, t2_arr,
        ax_extra, ay_extra,
        R2,
    )
    return will_collide


# ==========================
# Interval solver on one direction
# ==========================
@njit(cache=True, fastmath=False)
def _direction_collision_interval_one_time_jit(
    relx, rely,
    axes_x_k, axes_y_k,
    sum_radius_k,
    t2,
    ux, uy,
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
    relx_arr, rely_arr,
    axes_x, axes_y,
    sum_radius,
    t_arr, t2_arr,
    R2,
    ux, uy,
    a_max,
    tol,
):
    N = relx_arr.shape[0]
    eps_q = 1e-15

    Ls = np.empty(N, dtype=np.float64)
    Us = np.empty(N, dtype=np.float64)
    m = 0

    for k in range(N):
        has_itv, L, U = _direction_collision_interval_one_time_jit(
            relx_arr[k], rely_arr[k],
            axes_x[k], axes_y[k],
            sum_radius[k],
            t2_arr[k],
            ux, uy,
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
def _compute_EA_on_directions_interval_jit(
    relx_arr, rely_arr,
    axes_x, axes_y,
    sum_radius,
    t_arr, t2_arr,
    R2,
    ux_arr, uy_arr,
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
    EA_min = 1e30

    n_dir = ux_arr.shape[0]
    for i in range(n_dir):
        uxi = ux_arr[i]
        uyi = uy_arr[i]

        m_dir = _solve_direction_min_accel_interval_jit(
            relx_arr, rely_arr,
            axes_x, axes_y,
            sum_radius,
            t_arr, t2_arr,
            R2,
            uxi, uyi,
            a_max,
            tol,
        )

        if np.isnan(m_dir):
            continue

        if (not found) or (m_dir < EA_min):
            found = True
            EA_min = m_dir
            best_idx = i

    if not found:
        # A baseline collision exists, but no sampled direction is feasible
        # within a_max. This is a right-censored EA value.
        return np.inf, -1, True

    return EA_min, best_idx, True


# Public names used by the organized 11-column calculator.  The numerical
# implementation above is unchanged; these aliases avoid exposing private
# helper names to the rest of the package.
build_local_unit_vectors = _build_local_unit_vectors
check_collision_base = _check_collision_base_support_jit
check_current_obb_collision = _check_collision_obb_fast_params
compute_ea_on_directions = _compute_EA_on_directions_interval_jit
get_sector_unit_vectors = _get_sector_unit_vectors
get_time_grids = _get_time_grids
prepare_support_sat_data = _prepare_support_sat_data_jit
solve_direction_min_acceleration = _solve_direction_min_accel_interval_jit
will_collide_with_extra_accel = _will_collide_with_extra_accel_support_jit
