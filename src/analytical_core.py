import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


__all__ = [
    "ANALYTICAL_OUTPUT_COLUMNS",
    "check_collisions",
    "compute_bbox_distance",
    "compute_centroid_distance",
    "compute_real_time_metrics",
    "compute_real_time_metrics_dict",
]


# ============================================================================
# Constants
# ============================================================================
D_SAFE = 0.0
K_2D_TTC = 1.0
GAMMA = 0.01396
EPSILON = 1e-6


# ============================================================================
# Collision detection
# ============================================================================
def get_projection_offsets(
    length_1: float,
    width_1: float,
    heading_1: float,
    length_2: float,
    width_2: float,
    heading_2: float,
) -> Tuple[List[np.ndarray], List[List[float]], List[List[float]]]:
    """
    Precompute SAT projection offsets for two oriented rectangles centered at
    the origin.
    """
    dx_1 = length_1 / 2.0
    dy_1 = width_1 / 2.0
    dx_2 = length_2 / 2.0
    dy_2 = width_2 / 2.0

    vertices_1 = np.array(
        [
            [dx_1, dy_1],
            [-dx_1, dy_1],
            [-dx_1, -dy_1],
            [dx_1, -dy_1],
        ],
        dtype=float,
    )
    vertices_2 = np.array(
        [
            [dx_2, dy_2],
            [-dx_2, dy_2],
            [-dx_2, -dy_2],
            [dx_2, -dy_2],
        ],
        dtype=float,
    )

    c1, s1 = math.cos(heading_1), math.sin(heading_1)
    c2, s2 = math.cos(heading_2), math.sin(heading_2)

    rotation_matrix_1 = np.array([[c1, -s1], [s1, c1]], dtype=float)
    rotation_matrix_2 = np.array([[c2, -s2], [s2, c2]], dtype=float)

    rotated_vertices_1 = vertices_1 @ rotation_matrix_1.T
    rotated_vertices_2 = vertices_2 @ rotation_matrix_2.T

    axes: List[np.ndarray] = []
    for i in range(2):
        edge = rotated_vertices_1[i] - rotated_vertices_1[i - 1]
        axis = np.array([-edge[1], edge[0]], dtype=float)
        norm = np.linalg.norm(axis)
        axes.append(axis / norm)

    for i in range(2):
        edge = rotated_vertices_2[i] - rotated_vertices_2[i - 1]
        axis = np.array([-edge[1], edge[0]], dtype=float)
        norm = np.linalg.norm(axis)
        axes.append(axis / norm)

    projections_1 = [rotated_vertices_1 @ axis for axis in axes]
    max_and_min_projections_1 = [[float(np.min(p)), float(np.max(p))] for p in projections_1]

    projections_2 = [rotated_vertices_2 @ axis for axis in axes]
    max_and_min_projections_2 = [[float(np.min(p)), float(np.max(p))] for p in projections_2]

    return axes, max_and_min_projections_1, max_and_min_projections_2


def check_collisions_between_series(
    A_series: np.ndarray,
    B_series: np.ndarray,
    axes: Sequence[np.ndarray],
    max_and_min_projections_1: Sequence[Sequence[float]],
    max_and_min_projections_2: Sequence[Sequence[float]],
) -> np.ndarray:
    """
    Vectorized SAT collision checking for two position series.
    """
    proj_A_min_max = []
    proj_B_min_max = []

    for i in range(4):
        proj_A = A_series[:, :2] @ axes[i]
        proj_A_min_max.append(
            [
                proj_A + max_and_min_projections_1[i][0],
                proj_A + max_and_min_projections_1[i][1],
            ]
        )

        proj_B = B_series[:, :2] @ axes[i]
        proj_B_min_max.append(
            [
                proj_B + max_and_min_projections_2[i][0],
                proj_B + max_and_min_projections_2[i][1],
            ]
        )

    proj_A_min_max = np.array(proj_A_min_max)
    proj_B_min_max = np.array(proj_B_min_max)

    if_collision = []
    for i in range(4):
        if_collision.append(
            np.logical_not(
                (proj_A_min_max[i][1][:, np.newaxis] < proj_B_min_max[i][0][np.newaxis, :])
                | (proj_B_min_max[i][1][np.newaxis, :] < proj_A_min_max[i][0][:, np.newaxis])
            )
        )

    if_collision = np.array(if_collision)
    if_collision = np.all(if_collision, axis=0)
    return if_collision


def check_collisions(
    x_A: float,
    y_A: float,
    x_B: float,
    y_B: float,
    h_A: float,
    h_B: float,
    l_A: float,
    w_A: float,
    l_B: float,
    w_B: float,
) -> bool:
    """
    Check whether two oriented bounding boxes currently intersect.
    """
    dist = math.hypot(x_B - x_A, y_B - y_A)

    r_min_ego = 0.5 * min(l_A, w_A)
    r_min_veh = 0.5 * min(l_B, w_B)
    if dist < r_min_ego + r_min_veh + 0.1:
        return True

    rA = 0.5 * math.sqrt(l_A * l_A + w_A * w_A)
    rB = 0.5 * math.sqrt(l_B * l_B + w_B * w_B)
    if dist > rA + rB:
        return False

    x_A_array = np.array([x_A], dtype=float)
    y_A_array = np.array([y_A], dtype=float)
    x_B_array = np.array([x_B], dtype=float)
    y_B_array = np.array([y_B], dtype=float)

    axes, max_and_min_projections_1, max_and_min_projections_2 = get_projection_offsets(
        l_A, w_A, h_A, l_B, w_B, h_B
    )

    A_series = np.array([x_A_array, y_A_array]).T
    B_series = np.array([x_B_array, y_B_array]).T

    if_collision = check_collisions_between_series(
        A_series,
        B_series,
        axes,
        max_and_min_projections_1,
        max_and_min_projections_2,
    )
    return bool(if_collision[0, 0])


# ============================================================================
# Bounding-box distance
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


def compute_bbox_distance(
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
) -> float:
    """
    Compute the shortest distance between two oriented bounding boxes.

    Returns 0.0 if the two boxes intersect or touch.
    """
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
            return 0.0

    for px, py in B_pts:
        if _is_point_in_obb(px, py, xA, yA, hA, lA, wA, eps=eps):
            return 0.0

    for i in range(4):
        a1 = A_pts[i]
        a2 = A_pts[(i + 1) & 3]
        for j in range(4):
            b1 = B_pts[j]
            b2 = B_pts[(j + 1) & 3]
            if _segments_intersect(a1, a2, b1, b2, eps=eps):
                return 0.0

    min_d2 = float("inf")

    for px, py in A_pts:
        for i in range(4):
            qx, qy = B_pts[i]
            rx, ry = B_pts[(i + 1) & 3]

            wx, wy = rx - qx, ry - qy
            vx, vy = px - qx, py - qy
            wlen2 = wx * wx + wy * wy

            t = 0.0 if wlen2 == 0.0 else (vx * wx + vy * wy) / wlen2
            t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t

            cx = qx + t * wx
            cy = qy + t * wy

            dx = px - cx
            dy = py - cy
            d2 = dx * dx + dy * dy

            if d2 < min_d2:
                min_d2 = d2

    for px, py in B_pts:
        for i in range(4):
            qx, qy = A_pts[i]
            rx, ry = A_pts[(i + 1) & 3]

            wx, wy = rx - qx, ry - qy
            vx, vy = px - qx, py - qy
            wlen2 = wx * wx + wy * wy

            t = 0.0 if wlen2 == 0.0 else (vx * wx + vy * wy) / wlen2
            t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t

            cx = qx + t * wx
            cy = qy + t * wy

            dx = px - cx
            dy = py - cy
            d2 = dx * dx + dy * dy

            if d2 < min_d2:
                min_d2 = d2

    return math.sqrt(min_d2)


def compute_centroid_distance(x_A: float, y_A: float, x_B: float, y_B: float) -> float:
    """Compute the centroid distance between two agents."""
    return math.hypot(x_A - x_B, y_A - y_B)


# ============================================================================
# TTC / DRAC / TAdv
# ============================================================================
def compute_ttc_lon_1(
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
) -> Tuple[float, float]:
    theta_A = np.array([math.cos(hA), math.sin(hA)], dtype=float)
    theta_A_perp = np.array([-math.sin(hA), math.cos(hA)], dtype=float)

    xB_prime = np.array([xB - xA, yB - yA], dtype=float)
    s0_lon = float(np.dot(xB_prime, theta_A))
    s0_lat = float(np.dot(xB_prime, theta_A_perp))

    vB_vector = np.array([vB * math.cos(hB), vB * math.sin(hB)], dtype=float)
    vB_lon = float(np.dot(vB_vector, theta_A))
    vB_lat = float(np.dot(vB_vector, theta_A_perp))

    if s0_lon * (vA - vB_lon) > 0:
        gap = abs(s0_lon) - (lA + lB) / 2.0
        if gap > 0:
            ttc_lon_1 = gap / abs(vA - vB_lon)
            drac_1 = (vA - vB_lon) ** 2 / (2.0 * gap)
            if abs(s0_lat + vB_lat * ttc_lon_1) <= K_2D_TTC * ((wA + wB) / 2.0) and ttc_lon_1 >= 0:
                return ttc_lon_1, drac_1

    return np.nan, np.nan


def compute_ttc_lon_2(
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
) -> Tuple[float, float]:
    theta_B = np.array([math.cos(hB), math.sin(hB)], dtype=float)
    theta_B_perp = np.array([-math.sin(hB), math.cos(hB)], dtype=float)

    xA_prime = np.array([xA - xB, yA - yB], dtype=float)
    s0_lon = float(np.dot(xA_prime, theta_B))
    s0_lat = float(np.dot(xA_prime, theta_B_perp))

    vA_vector = np.array([vA * math.cos(hA), vA * math.sin(hA)], dtype=float)
    vA_lon = float(np.dot(vA_vector, theta_B))
    vA_lat = float(np.dot(vA_vector, theta_B_perp))

    if s0_lon * (vB - vA_lon) > 0:
        gap = abs(s0_lon) - (lA + lB) / 2.0
        if gap > 0:
            ttc_lon_2 = gap / abs(vB - vA_lon)
            drac_2 = (vB - vA_lon) ** 2 / (2.0 * gap)
            if abs(s0_lat + vA_lat * ttc_lon_2) <= K_2D_TTC * ((wA + wB) / 2.0) and ttc_lon_2 >= 0:
                return ttc_lon_2, drac_2

    return np.nan, np.nan


def compute_tadv(
    xA: float,
    yA: float,
    vA: float,
    hA: float,
    lA: float,
    xB: float,
    yB: float,
    vB: float,
    hB: float,
    lB: float,
) -> float:
    angle_difference = abs(hA - hB)
    if angle_difference > np.pi:
        angle_difference = 2.0 * np.pi - angle_difference

    delta_x = xB - xA
    delta_y = yB - yA
    norm_delta = math.sqrt(delta_x ** 2 + delta_y ** 2)

    if 0 <= angle_difference <= GAMMA:
        if delta_x * math.cos(hA) + delta_y * math.sin(hA) > 0:
            tadv = (norm_delta - lB / 2.0 - lA / 2.0) / vA
        else:
            tadv = (norm_delta - lB / 2.0 - lA / 2.0) / vB
    else:
        denominator_ac = vA * math.sin(hB - hA)
        if abs(denominator_ac) < EPSILON:
            t_ac = np.nan
        else:
            t_ac = (delta_x * math.sin(hB) - delta_y * math.cos(hB)) / denominator_ac

        denominator_bc = vB * math.sin(hA - hB)
        if abs(denominator_bc) < EPSILON:
            t_bc = np.nan
        else:
            t_bc = ((-delta_x) * math.sin(hA) - (-delta_y) * math.cos(hA)) / denominator_bc

        tadv = abs(t_ac - t_bc)

    return tadv if 0 <= tadv else np.nan


# ============================================================================
# ACT / shortest distance / closest-point relative speed
# ============================================================================
def get_rect_corners(x: float, y: float, h: float, l: float, w: float) -> List[List[float]]:
    ch, sh = math.cos(h), math.sin(h)
    corners = [
        [-l / 2.0 * ch - w / 2.0 * sh, -l / 2.0 * sh + w / 2.0 * ch],
        [l / 2.0 * ch - w / 2.0 * sh, l / 2.0 * sh + w / 2.0 * ch],
        [l / 2.0 * ch + w / 2.0 * sh, l / 2.0 * sh - w / 2.0 * ch],
        [-l / 2.0 * ch + w / 2.0 * sh, -l / 2.0 * sh - w / 2.0 * ch],
    ]
    corners = [[x + corner[0], y + corner[1]] for corner in corners]
    return corners


def distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def point_to_segment_distance(
    p: Sequence[float],
    v1: Sequence[float],
    v2: Sequence[float],
) -> Tuple[float, List[float]]:
    line_len = distance(v1, v2)
    if line_len == 0:
        return distance(p, v1), [v1[0], v1[1]]

    t = max(
        0.0,
        min(
            1.0,
            ((p[0] - v1[0]) * (v2[0] - v1[0]) + (p[1] - v1[1]) * (v2[1] - v1[1]))
            / (line_len ** 2),
        ),
    )
    closest = [v1[0] + t * (v2[0] - v1[0]), v1[1] + t * (v2[1] - v1[1])]
    return distance(p, closest), closest


def get_shortest_distance(
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
            dist, closest = point_to_segment_distance(p1, p2, p3)
            if dist < min_distance:
                min_distance = dist
                closest_A = p1
                closest_B = closest

        p1 = corners_B[i]
        for k in range(4):
            p2 = corners_A[k]
            p3 = corners_A[(k + 1) % 4]
            dist, closest = point_to_segment_distance(p1, p2, p3)
            if dist < min_distance:
                min_distance = dist
                closest_A = closest
                closest_B = p1

    return min_distance, closest_A, closest_B


def compute_shortest_distance(
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
    corners_A = get_rect_corners(x_A, y_A, h_A, l_A, w_A)
    corners_B = get_rect_corners(x_B, y_B, h_B, l_B, w_B)

    min_distance, closest_A, closest_B = get_shortest_distance(corners_A, corners_B)

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


# ============================================================================
# EI / MEI support
# ============================================================================
def compute_v_br(
    x_A: float,
    y_A: float,
    v_A: float,
    h_A: float,
    x_B: float,
    y_B: float,
    v_B: float,
    h_B: float,
) -> float:
    delta_x = x_B - x_A
    delta_y = y_B - y_A
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
        v_br = -float(np.dot(unit_vector, velocity_diff))
    else:
        v_br = 0.0

    return v_br


def compute_tdm_indepth(
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

    d_A1 = np.linalg.norm(AA1 - np.dot(AA1, theta_B_prime) * theta_B_prime)
    d_A2 = np.linalg.norm(AA2 - np.dot(AA2, theta_B_prime) * theta_B_prime)
    d_A3 = np.linalg.norm(AA3 - np.dot(AA3, theta_B_prime) * theta_B_prime)
    d_A4 = np.linalg.norm(AA4 - np.dot(AA4, theta_B_prime) * theta_B_prime)
    d_A_max = np.max(np.array([d_A1, d_A2, d_A3, d_A4], dtype=float))

    BB1 = np.array([l_B / 2 * chB - w_B / 2 * -shB, l_B / 2 * shB - w_B / 2 * chB], dtype=float)
    BB2 = np.array([l_B / 2 * chB + w_B / 2 * -shB, l_B / 2 * shB + w_B / 2 * chB], dtype=float)
    BB3 = np.array([-l_B / 2 * chB - w_B / 2 * -shB, -l_B / 2 * shB - w_B / 2 * chB], dtype=float)
    BB4 = np.array([-l_B / 2 * chB + w_B / 2 * -shB, -l_B / 2 * shB + w_B / 2 * chB], dtype=float)

    d_B1 = np.linalg.norm(BB1 - np.dot(BB1, theta_B_prime) * theta_B_prime)
    d_B2 = np.linalg.norm(BB2 - np.dot(BB2, theta_B_prime) * theta_B_prime)
    d_B3 = np.linalg.norm(BB3 - np.dot(BB3, theta_B_prime) * theta_B_prime)
    d_B4 = np.linalg.norm(BB4 - np.dot(BB4, theta_B_prime) * theta_B_prime)
    d_B_max = np.max(np.array([d_B1, d_B2, d_B3, d_B4], dtype=float))

    mfd = d_t1 - (d_A_max + d_B_max)
    d_B_prime = -float(np.dot(delta, theta_B_prime))
    tdm = d_B_prime / v_diff_norm if v_diff_norm != 0 else None
    indepth = D_SAFE - mfd

    return tdm, indepth


# ============================================================================
# TTC2D
# ============================================================================
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


def compute_ttc2d(
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


# ============================================================================
# Core public output contract
# ============================================================================
ANALYTICAL_OUTPUT_COLUMNS = [
    "DRAC",
    "TTC",
    "TAdv",
    "ACT",
    "v_closest",
    "Shortest_D",
    "EI",
    "InDepth",
    "MEI",
    "TTC2D",
    "DTC",
    "v_norm",
    "BBox distance (m)",
    "Centroid distance (m)",
]


def compute_real_time_metrics(
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
) -> Tuple[
    float, float, float, float, float,
    float, float, float, float, float,
    float, float, float, float
]:
    """
    Compute the analytical real-time metrics for one interaction frame.

    Output order is fixed and must match ANALYTICAL_OUTPUT_COLUMNS exactly.

    Current normalization rules are intentionally preserved:
      - DRAC, EI, MEI: NaN -> 0.0
      - TTC, ACT, TTC2D: NaN -> +inf
    """
    collision_result = check_collisions(
        x_A, y_A, x_B, y_B,
        h_A, h_B, l_A, w_A, l_B, w_B,
    )

    centroid_distance = compute_centroid_distance(x_A, y_A, x_B, y_B)

    if not collision_result:
        ttc_lon_1, drac_1 = compute_ttc_lon_1(
            x_A, y_A, v_A, h_A, l_A, w_A,
            x_B, y_B, v_B, h_B, l_B, w_B,
        )
        ttc_lon_2, drac_2 = compute_ttc_lon_2(
            x_A, y_A, v_A, h_A, l_A, w_A,
            x_B, y_B, v_B, h_B, l_B, w_B,
        )

        drac = np.nan if np.isnan(drac_1) and np.isnan(drac_2) else np.nanmin([drac_1, drac_2])
        ttc = np.nan if np.isnan(ttc_lon_1) and np.isnan(ttc_lon_2) else np.nanmin([ttc_lon_1, ttc_lon_2])
        tadv = compute_tadv(
            x_A, y_A, v_A, h_A, l_A,
            x_B, y_B, v_B, h_B, l_B,
        )

        shortest_distance, _closest_points, v_closest = compute_shortest_distance(
            x_A, y_A, v_A, h_A, l_A, w_A,
            x_B, y_B, v_B, h_B, l_B, w_B,
        )

        bbox_distance = shortest_distance
        ttc2d, dtc, v_norm = compute_ttc2d(
            x_A, y_A, v_A, h_A, l_A, w_A,
            x_B, y_B, v_B, h_B, l_B, w_B,
        )

        if not np.isnan(ttc2d) and ttc2d < 0:
            ttc2d = np.inf
            dtc = np.nan
            v_norm = np.nan

        if v_closest > 0:
            tdm, indepth = compute_tdm_indepth(
                x_A, y_A, v_A, h_A, l_A, w_A,
                x_B, y_B, v_B, h_B, l_B, w_B,
            )
            tdm = np.nan if tdm is None or tdm < 0 else tdm
            indepth = np.nan if indepth is None else indepth

            if np.isfinite(indepth) and indepth >= 0:
                ei = indepth / tdm if not np.isnan(tdm) and tdm != 0 else np.nan
                act = shortest_distance / v_closest
                mei = indepth / ttc2d if not np.isnan(ttc2d) and ttc2d != 0 else np.nan
            else:
                ei = np.nan
                mei = np.nan
                act = np.nan
        else:
            ei = np.nan
            indepth = np.nan
            mei = np.nan
            act = np.nan

    else:
        bbox_distance = 0.0

        drac = np.nan
        ttc = np.nan
        tadv = np.nan
        act = np.nan
        v_closest = np.nan
        shortest_distance = np.nan
        ei = np.nan
        indepth = np.nan
        mei = np.nan
        ttc2d = np.nan
        dtc = np.nan
        v_norm = np.nan

    # Preserve the original post-processing contract.
    drac = 0.0 if np.isnan(drac) else drac
    ei = 0.0 if np.isnan(ei) else ei
    mei = 0.0 if np.isnan(mei) else mei

    ttc = np.inf if np.isnan(ttc) else ttc
    act = np.inf if np.isnan(act) else act
    ttc2d = np.inf if np.isnan(ttc2d) else ttc2d

    return (
        drac,
        ttc,
        tadv,
        act,
        v_closest,
        shortest_distance,
        ei,
        indepth,
        mei,
        ttc2d,
        dtc,
        v_norm,
        bbox_distance,
        centroid_distance,
    )


def compute_real_time_metrics_dict(
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
) -> Dict[str, float]:
    """
    Dictionary wrapper for compute_real_time_metrics().
    """
    vals = compute_real_time_metrics(
        x_A, y_A, v_A, h_A, l_A, w_A,
        x_B, y_B, v_B, h_B, l_B, w_B,
    )
    return dict(zip(ANALYTICAL_OUTPUT_COLUMNS, vals))


# ============================================================================
# Minimal single-case runnable demo
# ============================================================================
def main() -> None:
    """
    Minimal single-case demo for analytical metric computation.
    """
    x_A, y_A, v_A, h_A, l_A, w_A = 504.0451, -271.9787, 22.9184, 2.5530, 17.0237, 2.5907
    x_B, y_B, v_B, h_B, l_B, w_B = 501.8724, -278.5692, 24.9702, 2.4877, 16.3289, 2.5973

    result = compute_real_time_metrics_dict(
        x_A, y_A, v_A, h_A, l_A, w_A,
        x_B, y_B, v_B, h_B, l_B, w_B,
    )

    print("analytical_core single-case demo")
    print("--------------------------------")
    for key in ANALYTICAL_OUTPUT_COLUMNS:
        print(f"{key}: {result[key]}")


if __name__ == "__main__":
    main()