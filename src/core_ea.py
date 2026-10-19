"""Backward-compatible entry points for the verified finite-horizon EA solver.

The frozen paper reference and its verified runtime live in
evasive_acceleration._paper. The runtime verifies ray candidates and uses
the fine grid for every retained local-search result.
All four CV/CTRV pairings use the same finite-horizon numerical search. The
legacy signatures and CSV component names are retained; CT means CTRV.
Results retain full precision: paper CSV rounding is not applied here.

Current OBB contact or no feasible direction within the search returns +inf.
Invalid inputs return NaN; no predicted contact on the checked grids gives 0.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    from numba import njit as _numba_njit

    _NUMBA_OK = True
except Exception:
    _NUMBA_OK = False

__all__ = [
    "CORE_ID", "get_core_provenance", "aggregate_ea_modes",
    "compute_single_mode_ea", "compute_single_mode_ea_timed",
    "compute_ea_modes", "compute_ea_modes_timed", "compute_ea", "compute_ea_timed",
]

# Compatibility defaults, checked against the frozen EAConfig.
DEFAULT_T_TOTAL = 10.0
DEFAULT_DT_COARSE = 0.1
DEFAULT_DT_FINE = 0.02
DEFAULT_A_MAX = 100.0
DEFAULT_TOL = 1e-3
DEFAULT_COARSE_SECTOR_NUM = 72
DEFAULT_LOCAL_FINE_HALF_WINDOW_DEG = 5.0
DEFAULT_LOCAL_FINE_DIR_NUM = 101
DEFAULT_FINE_REFINE_RATIO = 1.15
CORE_ID = "paper-aligned-verified-cv-ctrv-20261020"

# The paper aggregates in CV-first arithmetic order, before CSV rounding.
_MODE_KEYS = ("EA_CVCV", "EA_CVCT", "EA_CTCV", "EA_CTCT")


def get_core_provenance() -> dict:
    """Return the frozen-reference identity and current runtime source hashes."""
    manifest = (
        Path(__file__).resolve().parent / "evasive_acceleration"
        / "_paper" / "provenance.json"
    )
    provenance = json.loads(manifest.read_text(encoding="utf-8"))
    provenance["runtime_source_sha256"] = {
        name: hashlib.sha256((manifest.parent / name).read_bytes()).hexdigest()
        for name in ("verified_solver.py", "verified_calculator.py", "four_modes.py")
    }
    return provenance


@lru_cache(maxsize=1)
def _get_bridge():
    # Lazy imports avoid the core_ea -> package -> single_frame import cycle.
    from evasive_acceleration._paper.config import EAConfig, MotionModel
    from evasive_acceleration._paper.four_modes import compute_selected_modes

    return EAConfig, MotionModel, compute_selected_modes


def aggregate_ea_modes(mode_dict: dict[str, float]) -> float:
    """Mean four raw values, propagating missing/NaN components as NaN."""
    try:
        values = [float(mode_dict[key]) for key in _MODE_KEYS]
    except (KeyError, TypeError, ValueError, OverflowError):
        return float("nan")
    if any(math.isnan(value) for value in values):
        return float("nan")
    return float(sum(values) / 4.0)


def _prepare_inputs(
    values, *, coarse_sector_num, local_fine_half_window_deg,
    local_fine_dir_num, fine_refine_ratio, a_max, tol, T_total,
    dt_coarse, dt_fine, size_scale,
):
    """Translate legacy inputs to the canonical state and config, or None."""
    try:
        values = [float(value) for value in values]
        scale = float(size_scale)
        numeric_settings = tuple(
            float(value) for value in (
                local_fine_half_window_deg, fine_refine_ratio, a_max,
                tol, T_total, dt_coarse, dt_fine,
            )
        )
        coarse_count = int(coarse_sector_num)
        local_count = int(local_fine_dir_num)
        if coarse_count != coarse_sector_num or local_count != local_fine_dir_num:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in (*values, scale, *numeric_settings)):
        return None
    half_window, refine_ratio, maximum, tolerance, horizon, coarse_dt, fine_dt = (
        numeric_settings
    )
    if (
        scale <= 0.0 or coarse_count < 4 or local_count < 3
        or half_window < 0.0 or refine_ratio <= 0.0
        or min(maximum, tolerance, horizon, coarse_dt, fine_dt) <= 0.0
    ):
        return None

    x_a, y_a, v_a, h_a, x_b, y_b, v_b, h_b, l_a, w_a, l_b, w_b, yaw_a, yaw_b = values
    if min(l_a, w_a, l_b, w_b) <= 0.0 or min(v_a, v_b) < 0.0:
        return None
    dimensions = tuple(value * scale for value in (l_a, w_a, l_b, w_b))
    if not all(math.isfinite(value) and value > 0.0 for value in dimensions):
        return None
    l_a, w_a, l_b, w_b = dimensions

    config_type, _model, _compute = _get_bridge()
    config = config_type(
        prediction_horizon_s=horizon,
        coarse_step_s=coarse_dt,
        fine_step_s=fine_dt,
        max_extra_acceleration_mps2=maximum,
        solver_tolerance=tolerance,
        coarse_direction_count=coarse_count,
        local_direction_half_window_deg=half_window,
        local_direction_count=local_count,
        fine_refine_ratio=refine_ratio,
    )
    state = np.asarray(
        (x_a, y_a, v_a, h_a, l_a, w_a, yaw_a,
         x_b, y_b, v_b, h_b, l_b, w_b, yaw_b),
        dtype=np.float64,
    )
    return state, config


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
    """Compute one finite-horizon hypothesis; zero yaw specifies CV."""
    prepared = _prepare_inputs(
        (xA, yA, vA, hA, xB, yB, vB, hB, lA, wA, lB, wB, yawA, yawB),
        coarse_sector_num=coarse_sector_num,
        local_fine_half_window_deg=local_fine_half_window_deg,
        local_fine_dir_num=local_fine_dir_num,
        fine_refine_ratio=fine_refine_ratio,
        a_max=a_max,
        tol=tol,
        T_total=T_total,
        dt_coarse=dt_coarse,
        dt_fine=dt_fine,
        size_scale=1.0,
    )
    if prepared is None:
        return float("nan")
    _config_type, model, compute_selected = _get_bridge()
    pair = (model.CTRV, model.CTRV)
    return compute_selected(*prepared, pairs=(pair,))[pair]


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
) -> tuple[float, float]:
    """Return one hypothesis value and elapsed seconds."""
    started = time.perf_counter()
    value = compute_single_mode_ea(
        xA, yA, vA, hA, xB, yB, vB, hB, lA, wA, lB, wB, yawA, yawB,
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
    return value, time.perf_counter() - started


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
) -> dict[str, float]:
    """Return raw EA_CTCT/CTCV/CVCT/CVCV; CT is the legacy name for CTRV.

    size_scale multiplies both users' OBB length and width. Only the four
    original CV/CTRV hypotheses are evaluated, not the CTRA research extension.
    """
    prepared = _prepare_inputs(
        (xA, yA, vA, hA, xB, yB, vB, hB, lA, wA, lB, wB, yawA, yawB),
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
    if prepared is None:
        return dict.fromkeys(("EA_CTCT", "EA_CTCV", "EA_CVCT", "EA_CVCV"), float("nan"))
    _config_type, model, compute_selected = _get_bridge()
    values = compute_selected(*prepared)
    return {
        "EA_CTCT": values[(model.CTRV, model.CTRV)],
        "EA_CTCV": values[(model.CTRV, model.CV)],
        "EA_CVCT": values[(model.CV, model.CTRV)],
        "EA_CVCV": values[(model.CV, model.CV)],
    }


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
) -> tuple[dict[str, float], float]:
    """Return four raw component values and elapsed seconds."""
    started = time.perf_counter()
    modes = compute_ea_modes(
        xA, yA, vA, hA, xB, yB, vB, hB, lA, wA, lB, wB, yawA, yawB,
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
    return modes, time.perf_counter() - started


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
    """Return the mean of the four raw finite-horizon component values."""
    modes = compute_ea_modes(
        xA, yA, vA, hA, xB, yB, vB, hB, lA, wA, lB, wB, yawA, yawB,
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
    return aggregate_ea_modes(modes)


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
) -> tuple[float, dict[str, float], float]:
    """Return aggregate, raw components, and elapsed seconds."""
    started = time.perf_counter()
    modes = compute_ea_modes(
        xA, yA, vA, hA, xB, yB, vB, hB, lA, wA, lB, wB, yawA, yawB,
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
    return aggregate_ea_modes(modes), modes, time.perf_counter() - started


if __name__ == "__main__":
    value, modes, elapsed = compute_ea_timed(
        0.0, 0.0, 10.0, 0.0, 20.0, 0.0, 8.0, math.pi,
        4.5, 1.8, 4.7, 1.9, 0.0, 0.0,
    )
    print("EA =", value, "m/s^2")
    print("Components =", modes)
    print("Runtime =", elapsed, "s")
