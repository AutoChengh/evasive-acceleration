"""Coarse-to-fine EA search with an explicit fine-grid feasibility contract.

The four frozen paper source files remain an archival reference. This module
keeps their motion and SAT geometry but makes every reported finite candidate
pass the full validation grid. The coarse grid is used to locate a promising
direction, never as an unverified final-result fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from . import numerical_solver as geometry
from . import verified_solver as solver
from .config import EAConfig


@dataclass(frozen=True)
class PredictionResult:
    """Private search diagnostics; the public API continues returning scalars."""

    value: float
    direction_x: float
    direction_y: float
    verified: bool
    baseline_collision: bool
    search_grid: str
    evaluated_fine_directions: int


@lru_cache(maxsize=64)
def make_time_grid(horizon: float, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``min(k * step, horizon)`` and explicitly include both endpoints.

    The final interval may be shorter than ``step``. A quotient within a few
    floating-point ulps of an integer is treated as that integer, avoiding a
    duplicate final sample caused solely by division rounding.
    """
    horizon, step = float(horizon), float(step)
    quotient = horizon / step
    nearest = round(quotient)
    close_integer = abs(quotient - nearest) <= 4.0 * math.ulp(quotient)
    intervals = max(1, int(nearest if close_integer else math.ceil(quotient)))
    times = np.minimum(np.arange(intervals + 1, dtype=np.float64) * step, horizon)
    times[-1] = horizon
    squared = 0.5 * times * times
    times.setflags(write=False)
    squared.setflags(write=False)
    return times, squared


def _baseline_collision(data) -> bool:
    collides, _ = geometry.check_collision_base(
        data["relx_arr"], data["rely_arr"], data["axes_x"], data["axes_y"],
        data["sum_radius"], data["t_arr"], float(data["R2"]),
    )
    return bool(collides)


def _candidate_collision(data, value: float, direction_x: float, direction_y: float) -> bool:
    """Evaluate the actual translated OBB candidate, not interval endpoints."""
    return bool(geometry.will_collide_with_extra_accel(
        data["relx_arr"], data["rely_arr"], data["axes_x"], data["axes_y"],
        data["sum_radius"], data["t_arr"], data["t2_arr"],
        float(value * direction_x), float(value * direction_y), float(data["R2"]),
    ))


def _search_directions(data, direction_x, direction_y, config: EAConfig):
    return solver.compute_ea_on_directions(
        data["relx_arr"], data["rely_arr"], data["axes_x"], data["axes_y"],
        data["sum_radius"], data["t_arr"], data["t2_arr"], float(data["R2"]),
        np.asarray(direction_x, dtype=np.float64),
        np.asarray(direction_y, dtype=np.float64),
        config.max_extra_acceleration_mps2, config.solver_tolerance,
    )


def solve_prediction(coarse_data, fine_data, config: EAConfig) -> PredictionResult:
    """Solve one model pairing and retain a fine-grid-verified direction.

    All local directions, plus the original best coarse direction, are solved
    anew on ``fine_data``. ``fine_refine_ratio`` is accepted by the surrounding
    compatibility API but no longer selects which directions receive a fine
    calculation. No collision-at-incumbent pruning is used: feasibility along
    an acceleration ray need not be monotone.

    A fine-only baseline collision is localized with a full-circle directional
    search on the fine grid. If local refinement finds no feasible candidate,
    the same full-circle fine-grid search is a verified fallback; a raw coarse
    value is never returned.
    """
    if not _baseline_collision(fine_data):
        return PredictionResult(0.0, math.nan, math.nan, True, False, "fine-zero", 0)

    direction_x, direction_y = geometry.get_sector_unit_vectors(
        config.coarse_direction_count
    )
    locate_data = coarse_data if _baseline_collision(coarse_data) else fine_data
    location = "coarse-local" if locate_data is coarse_data else "fine-only-local"
    coarse_value, best_index, _ = _search_directions(
        locate_data, direction_x, direction_y, config
    )
    fine_count = config.coarse_direction_count if locate_data is fine_data else 0

    # A coarse failure is not evidence that the fine-grid search must fail.
    if not math.isfinite(float(coarse_value)) or best_index < 0:
        if locate_data is not fine_data:
            coarse_value, best_index, _ = _search_directions(
                fine_data, direction_x, direction_y, config
            )
            fine_count += config.coarse_direction_count
            location = "fine-rescue-local"
        if not math.isfinite(float(coarse_value)) or best_index < 0:
            return PredictionResult(
                math.inf, math.nan, math.nan, False, True, "fine-unresolved", fine_count
            )

    direction_center = math.atan2(direction_y[best_index], direction_x[best_index])
    _, local_x, local_y = geometry.build_local_unit_vectors(
        direction_center, config.local_direction_half_window_deg,
        config.local_direction_count,
    )
    # Always retain the original coarse direction, including even-sized local
    # grids that otherwise do not contain their exact centre.
    fine_x = np.concatenate((np.asarray([direction_x[best_index]]), local_x))
    fine_y = np.concatenate((np.asarray([direction_y[best_index]]), local_y))
    value, fine_index, _ = _search_directions(fine_data, fine_x, fine_y, config)
    fine_count += len(fine_x)

    if not math.isfinite(float(value)) or fine_index < 0:
        # Only a result solved and checked on the fine grid may be a fallback.
        value, fine_index, _ = _search_directions(
            fine_data, direction_x, direction_y, config
        )
        fine_count += len(direction_x)
        fine_x, fine_y = direction_x, direction_y
        location = "fine-full-circle-fallback"

    if not math.isfinite(float(value)) or fine_index < 0:
        return PredictionResult(
            math.inf, math.nan, math.nan, False, True, "fine-unresolved", fine_count
        )

    chosen_x, chosen_y = float(fine_x[fine_index]), float(fine_y[fine_index])
    if _candidate_collision(fine_data, float(value), chosen_x, chosen_y):
        raise RuntimeError("Verified EA solver returned a colliding fine-grid candidate")
    return PredictionResult(
        float(value), chosen_x, chosen_y, True, True, location, fine_count
    )
