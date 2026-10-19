"""Frame-level calculation of nine ordered EA modes and two aggregates."""

from __future__ import annotations

import math

import numpy as np

from . import numerical_solver as solver
from .numerical_solver import (
    CURRENT_COLLISION_EPS,
    solve_direction_min_acceleration,
    will_collide_with_extra_accel,
)
from .config import (
    ALL_PAIR_KEYS,
    DEFAULT_CONFIG,
    EA_COLUMN,
    EA_OUTPUT_COLUMNS,
    EA_WITH_CTRA_COLUMN,
    MODEL_ORDER,
    ORIGINAL_EA_PAIR_KEYS,
    EAConfig,
    MotionModel,
    pair_column,
    pair_key,
)
from .motion import PoseTrajectory, predict_all_models


def _pair_prediction_data(
    road_user_1: PoseTrajectory,
    road_user_2: PoseTrajectory,
    length_1: float,
    width_1: float,
    length_2: float,
    width_2: float,
    time_grid: np.ndarray,
    half_time_squared: np.ndarray,
) -> dict[str, np.ndarray | float]:
    (
        relative_x,
        relative_y,
        axes_x,
        axes_y,
        summed_radii,
        broad_phase_radius_squared,
    ) = solver.prepare_support_sat_data(
        road_user_1.x,
        road_user_1.y,
        road_user_1.heading,
        road_user_2.x,
        road_user_2.y,
        road_user_2.heading,
        float(length_1),
        float(width_1),
        float(length_2),
        float(width_2),
    )
    return {
        "t_arr": time_grid,
        "t2_arr": half_time_squared,
        "relx_arr": relative_x,
        "rely_arr": relative_y,
        "axes_x": axes_x,
        "axes_y": axes_y,
        "sum_radius": summed_radii,
        "R2": float(broad_phase_radius_squared),
    }


def _model_speed_profile(
    model: MotionModel,
    speed: float,
    acceleration: float,
    time_grid: np.ndarray,
) -> np.ndarray:
    """Return the non-reversing scalar-speed profile of one motion model."""

    initial_speed = max(0.0, float(speed))
    if model is not MotionModel.CTRA:
        return np.full_like(time_grid, initial_speed, dtype=np.float64)
    return np.maximum(0.0, initial_speed + float(acceleration) * time_grid)


def _possibly_colliding_coarse_intervals(
    road_user_1: PoseTrajectory,
    road_user_2: PoseTrajectory,
    model_1: MotionModel,
    model_2: MotionModel,
    speed_1: float,
    speed_2: float,
    acceleration_1: float,
    acceleration_2: float,
    length_1: float,
    width_1: float,
    length_2: float,
    width_2: float,
    coarse_time: np.ndarray,
) -> np.ndarray:
    """Flag coarse intervals that could contain an unobserved OBB contact.

    The sum of the two OBB circumradii is a conservative broad-phase radius.
    For an interval of duration ``dt``, a contact can occur only if its initial
    centre separation is no larger than that radius plus the largest possible
    relative travel over the interval.  Intervals failing this test cannot hide
    a collision and do not need fine-grid OBB checks.
    """

    interval_count = max(0, len(coarse_time) - 1)
    if interval_count == 0:
        return np.zeros(0, dtype=bool)
    radius = math.hypot(0.5 * length_1, 0.5 * width_1) + math.hypot(
        0.5 * length_2,
        0.5 * width_2,
    )
    separation = np.hypot(
        road_user_2.x[:-1] - road_user_1.x[:-1],
        road_user_2.y[:-1] - road_user_1.y[:-1],
    )
    speed_profile_1 = _model_speed_profile(
        model_1, speed_1, acceleration_1, coarse_time
    )
    speed_profile_2 = _model_speed_profile(
        model_2, speed_2, acceleration_2, coarse_time
    )
    maximum_relative_speed = np.maximum(
        speed_profile_1[:-1], speed_profile_1[1:]
    ) + np.maximum(speed_profile_2[:-1], speed_profile_2[1:])
    interval_duration = np.diff(coarse_time)
    return separation <= radius + maximum_relative_speed * interval_duration


def _fine_indices_for_coarse_intervals(
    coarse_time: np.ndarray,
    fine_time: np.ndarray,
    interval_mask: np.ndarray,
) -> np.ndarray:
    """Return fine-grid samples contained in flagged coarse intervals."""

    selected = np.zeros(len(fine_time), dtype=bool)
    for index in np.flatnonzero(interval_mask):
        lower = coarse_time[index] - 1e-12
        upper = coarse_time[index + 1] + 1e-12
        selected |= (fine_time >= lower) & (fine_time <= upper)
    return selected


def _slice_pair_prediction_data(
    pair_data: dict[str, np.ndarray | float],
    mask: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Restrict time-indexed SAT arrays to selected fine-grid samples."""

    return {
        key: (value[mask] if isinstance(value, np.ndarray) else value)
        for key, value in pair_data.items()
    }


@solver.njit(cache=True, fastmath=False)
def _local_refine_interval_jit(
    coarse_relx,
    coarse_rely,
    coarse_axes_x,
    coarse_axes_y,
    coarse_sum_radius,
    coarse_t,
    coarse_t2,
    coarse_r2,
    fine_relx,
    fine_rely,
    fine_axes_x,
    fine_axes_y,
    fine_sum_radius,
    fine_t,
    fine_t2,
    fine_r2,
    use_fine,
    local_direction_x,
    local_direction_y,
    coarse_ea,
    fine_refine_ratio,
    max_acceleration,
    tolerance,
):
    best = np.nan
    fine_limit = coarse_ea * fine_refine_ratio + tolerance

    for index in range(local_direction_x.shape[0]):
        direction_x = local_direction_x[index]
        direction_y = local_direction_y[index]

        if not np.isnan(best):
            if will_collide_with_extra_accel(
                coarse_relx,
                coarse_rely,
                coarse_axes_x,
                coarse_axes_y,
                coarse_sum_radius,
                coarse_t,
                coarse_t2,
                best * direction_x,
                best * direction_y,
                coarse_r2,
            ):
                continue

        coarse_value = solve_direction_min_acceleration(
            coarse_relx,
            coarse_rely,
            coarse_axes_x,
            coarse_axes_y,
            coarse_sum_radius,
            coarse_t,
            coarse_t2,
            coarse_r2,
            direction_x,
            direction_y,
            max_acceleration,
            tolerance,
        )
        if np.isnan(coarse_value):
            continue
        value = coarse_value

        if use_fine and coarse_value <= fine_limit:
            if not np.isnan(best):
                if will_collide_with_extra_accel(
                    fine_relx,
                    fine_rely,
                    fine_axes_x,
                    fine_axes_y,
                    fine_sum_radius,
                    fine_t,
                    fine_t2,
                    best * direction_x,
                    best * direction_y,
                    fine_r2,
                ):
                    continue

            fine_value = solve_direction_min_acceleration(
                fine_relx,
                fine_rely,
                fine_axes_x,
                fine_axes_y,
                fine_sum_radius,
                fine_t,
                fine_t2,
                fine_r2,
                direction_x,
                direction_y,
                max_acceleration,
                tolerance,
            )
            if np.isnan(fine_value):
                continue
            value = fine_value

        if np.isnan(best) or value < best:
            best = value

    return best


def _compute_ea_for_prediction(
    coarse_data: dict[str, np.ndarray | float],
    fine_data: dict[str, np.ndarray | float] | None,
    config: EAConfig,
) -> float:
    baseline_collision, _ = solver.check_collision_base(
        coarse_data["relx_arr"],
        coarse_data["rely_arr"],
        coarse_data["axes_x"],
        coarse_data["axes_y"],
        coarse_data["sum_radius"],
        coarse_data["t_arr"],
        float(coarse_data["R2"]),
    )
    if not baseline_collision:
        return 0.0

    coarse_direction_x, coarse_direction_y = solver.get_sector_unit_vectors(
        config.coarse_direction_count
    )
    coarse_ea, best_index, _ = solver.compute_ea_on_directions(
        coarse_data["relx_arr"],
        coarse_data["rely_arr"],
        coarse_data["axes_x"],
        coarse_data["axes_y"],
        coarse_data["sum_radius"],
        coarse_data["t_arr"],
        coarse_data["t2_arr"],
        float(coarse_data["R2"]),
        np.asarray(coarse_direction_x, dtype=np.float64),
        np.asarray(coarse_direction_y, dtype=np.float64),
        config.max_extra_acceleration_mps2,
        config.solver_tolerance,
    )
    coarse_ea = float(coarse_ea)
    if math.isnan(coarse_ea):
        raise RuntimeError("EA solver returned NaN for a valid prediction")
    if math.isinf(coarse_ea) or coarse_ea == 0.0 or best_index < 0:
        return coarse_ea

    direction_center = (best_index + 0.5) * (
        2.0 * math.pi / config.coarse_direction_count
    )
    _, local_direction_x, local_direction_y = solver.build_local_unit_vectors(
        direction_center,
        config.local_direction_half_window_deg,
        config.local_direction_count,
    )

    use_fine = fine_data is not None
    if use_fine:
        fine_collision, _ = solver.check_collision_base(
            fine_data["relx_arr"],
            fine_data["rely_arr"],
            fine_data["axes_x"],
            fine_data["axes_y"],
            fine_data["sum_radius"],
            fine_data["t_arr"],
            float(fine_data["R2"]),
        )
        if not fine_collision:
            return 0.0
        fine_arrays = fine_data
    else:
        fine_arrays = coarse_data

    refined_ea = float(
        _local_refine_interval_jit(
            coarse_data["relx_arr"],
            coarse_data["rely_arr"],
            coarse_data["axes_x"],
            coarse_data["axes_y"],
            coarse_data["sum_radius"],
            coarse_data["t_arr"],
            coarse_data["t2_arr"],
            float(coarse_data["R2"]),
            fine_arrays["relx_arr"],
            fine_arrays["rely_arr"],
            fine_arrays["axes_x"],
            fine_arrays["axes_y"],
            fine_arrays["sum_radius"],
            fine_arrays["t_arr"],
            fine_arrays["t2_arr"],
            float(fine_arrays["R2"]),
            bool(use_fine),
            np.asarray(local_direction_x, dtype=np.float64),
            np.asarray(local_direction_y, dtype=np.float64),
            coarse_ea,
            config.fine_refine_ratio,
            config.max_extra_acceleration_mps2,
            config.solver_tolerance,
        )
    )
    return coarse_ea if math.isnan(refined_ea) else refined_ea


def _mean_ea(
    mode_values: dict[str, float],
    keys: tuple[str, ...],
) -> float:
    values = [float(mode_values[key]) for key in keys]
    if any(math.isnan(value) for value in values):
        raise RuntimeError("Cannot aggregate an undefined motion-model EA")
    if any(math.isinf(value) for value in values):
        return float("inf")
    return float(sum(values) / len(values))


def _round_metric(value: float, decimals: int) -> float:
    value = float(value)
    if math.isnan(value):
        raise RuntimeError("EA output cannot be NaN")
    if math.isinf(value):
        return value
    return round(value, decimals)


def compute_frame_metrics(
    state_values: np.ndarray,
    acceleration_1: float,
    acceleration_2: float,
    config: EAConfig = DEFAULT_CONFIG,
) -> dict[str, float]:
    """Compute the nine ordered modes, ``EA``, and ``EA (with CTRA)``.

    ``state_values`` must follow :data:`src.config.STATE_COLUMNS`.
    The mode in the row name belongs to road user 1; the mode in the column
    name belongs to road user 2. Aggregates are computed from unrounded mode
    values and rounded only once at the end.
    """

    (
        x_1,
        y_1,
        speed_1,
        heading_1,
        length_1,
        width_1,
        yaw_rate_1,
        x_2,
        y_2,
        speed_2,
        heading_2,
        length_2,
        width_2,
        yaw_rate_2,
    ) = [float(value) for value in state_values]

    if solver.check_current_obb_collision(
        x_1,
        y_1,
        heading_1,
        length_1,
        width_1,
        x_2,
        y_2,
        heading_2,
        length_2,
        width_2,
    ):
        return {column: float("inf") for column in EA_OUTPUT_COLUMNS}

    coarse_time, coarse_time_squared, _ = solver.get_time_grids(
        config.prediction_horizon_s,
        config.coarse_step_s,
    )
    coarse_1 = predict_all_models(
        x=x_1,
        y=y_1,
        speed=speed_1,
        heading=heading_1,
        yaw_rate=yaw_rate_1,
        acceleration=acceleration_1,
        time_grid=coarse_time,
    )
    coarse_2 = predict_all_models(
        x=x_2,
        y=y_2,
        speed=speed_2,
        heading=heading_2,
        yaw_rate=yaw_rate_2,
        acceleration=acceleration_2,
        time_grid=coarse_time,
    )

    mode_values: dict[str, float] = {}
    colliding_modes: list[tuple[str, MotionModel, MotionModel, bool]] = []
    coarse_pairs: dict[str, dict[str, np.ndarray | float]] = {}
    possible_fine_intervals: dict[str, np.ndarray] = {}

    for model_1 in MODEL_ORDER:
        for model_2 in MODEL_ORDER:
            key = pair_key(model_1, model_2)
            pair_data = _pair_prediction_data(
                coarse_1[model_1],
                coarse_2[model_2],
                length_1,
                width_1,
                length_2,
                width_2,
                coarse_time,
                coarse_time_squared,
            )
            coarse_pairs[key] = pair_data
            collides, _ = solver.check_collision_base(
                pair_data["relx_arr"],
                pair_data["rely_arr"],
                pair_data["axes_x"],
                pair_data["axes_y"],
                pair_data["sum_radius"],
                pair_data["t_arr"],
                float(pair_data["R2"]),
            )
            if collides:
                colliding_modes.append((key, model_1, model_2, True))
            else:
                mode_values[key] = 0.0
                if config.fine_step_s < config.coarse_step_s:
                    interval_mask = _possibly_colliding_coarse_intervals(
                        coarse_1[model_1],
                        coarse_2[model_2],
                        model_1,
                        model_2,
                        speed_1,
                        speed_2,
                        acceleration_1,
                        acceleration_2,
                        length_1,
                        width_1,
                        length_2,
                        width_2,
                        coarse_time,
                    )
                    if interval_mask.any():
                        possible_fine_intervals[key] = interval_mask

    fine_1 = None
    fine_2 = None
    fine_time = None
    fine_time_squared = None
    if (
        config.fine_step_s < config.coarse_step_s
        and (colliding_modes or possible_fine_intervals)
    ):
        fine_time, fine_time_squared, _ = solver.get_time_grids(
            config.prediction_horizon_s,
            config.fine_step_s,
        )
        fine_1 = predict_all_models(
            x=x_1,
            y=y_1,
            speed=speed_1,
            heading=heading_1,
            yaw_rate=yaw_rate_1,
            acceleration=acceleration_1,
            time_grid=fine_time,
        )
        fine_2 = predict_all_models(
            x=x_2,
            y=y_2,
            speed=speed_2,
            heading=heading_2,
            yaw_rate=yaw_rate_2,
            acceleration=acceleration_2,
            time_grid=fine_time,
        )

    fine_pairs: dict[str, dict[str, np.ndarray | float]] = {}
    if fine_1 is not None and fine_2 is not None:
        # Build full fine predictions only for modes that either collide on the
        # coarse grid or have a physically possible between-sample contact.
        fine_keys = {
            key for key, _model_1, _model_2, _coarse in colliding_modes
        } | set(possible_fine_intervals)
        for model_1 in MODEL_ORDER:
            for model_2 in MODEL_ORDER:
                key = pair_key(model_1, model_2)
                if key not in fine_keys:
                    continue
                fine_pairs[key] = _pair_prediction_data(
                    fine_1[model_1],
                    fine_2[model_2],
                    length_1,
                    width_1,
                    length_2,
                    width_2,
                    fine_time,
                    fine_time_squared,
                )

        # A collision can be missed by 0.1-s samples while still appearing on
        # the configured fine grid.  Promote only such fine-only modes to the
        # EA solver; all other no-collision modes remain exactly zero.
        for model_1 in MODEL_ORDER:
            for model_2 in MODEL_ORDER:
                key = pair_key(model_1, model_2)
                if key not in possible_fine_intervals:
                    continue
                fine_mask = _fine_indices_for_coarse_intervals(
                    coarse_time,
                    fine_time,
                    possible_fine_intervals[key],
                )
                fine_candidate = _slice_pair_prediction_data(
                    fine_pairs[key], fine_mask
                )
                fine_collides, _ = solver.check_collision_base(
                    fine_candidate["relx_arr"],
                    fine_candidate["rely_arr"],
                    fine_candidate["axes_x"],
                    fine_candidate["axes_y"],
                    fine_candidate["sum_radius"],
                    fine_candidate["t_arr"],
                    float(fine_candidate["R2"]),
                )
                if fine_collides:
                    colliding_modes.append((key, model_1, model_2, False))

    for key, _model_1, _model_2, coarse_collision in colliding_modes:
        fine_pair = fine_pairs.get(key)
        if coarse_collision:
            mode_values[key] = _compute_ea_for_prediction(
                coarse_pairs[key],
                fine_pair,
                config,
            )
        else:
            # The baseline collision is visible only on the fine grid, so the
            # EA threshold must be solved on that grid rather than returning a
            # spurious zero from the coarse-grid baseline test.
            assert fine_pair is not None
            mode_values[key] = _compute_ea_for_prediction(
                fine_pair,
                None,
                config,
            )

    raw_output = {
        pair_column(model_1, model_2): mode_values[
            pair_key(model_1, model_2)
        ]
        for model_1 in MODEL_ORDER
        for model_2 in MODEL_ORDER
    }
    raw_output[EA_COLUMN] = _mean_ea(
        mode_values,
        ORIGINAL_EA_PAIR_KEYS,
    )
    raw_output[EA_WITH_CTRA_COLUMN] = _mean_ea(
        mode_values,
        ALL_PAIR_KEYS,
    )

    return {
        column: _round_metric(raw_output[column], config.output_decimals)
        for column in EA_OUTPUT_COLUMNS
    }


@solver.njit(cache=True, fastmath=False)
def _motion_pose_jit(
    x,
    y,
    speed,
    heading,
    yaw_rate,
    acceleration,
    time_value,
):
    speed = max(0.0, speed)
    effective_time = time_value
    if acceleration < 0.0:
        if speed <= 0.0:
            effective_time = 0.0
        else:
            stop_time = speed / (-acceleration)
            if effective_time > stop_time:
                effective_time = stop_time

    if abs(yaw_rate) < 1e-6:
        distance = (
            speed * effective_time
            + 0.5 * acceleration * effective_time * effective_time
        )
        return (
            x + math.cos(heading) * distance,
            y + math.sin(heading) * distance,
            heading,
        )

    theta = heading + yaw_rate * effective_time
    x_new = x + (speed / yaw_rate) * (
        math.sin(theta) - math.sin(heading)
    )
    x_new += acceleration * (
        effective_time * math.sin(theta) / yaw_rate
        + (math.cos(theta) - math.cos(heading)) / (yaw_rate * yaw_rate)
    )
    y_new = y - (speed / yaw_rate) * (
        math.cos(theta) - math.cos(heading)
    )
    y_new += acceleration * (
        -effective_time * math.cos(theta) / yaw_rate
        + (math.sin(theta) - math.sin(heading)) / (yaw_rate * yaw_rate)
    )
    return x_new, y_new, theta


@solver.njit(cache=True, fastmath=False)
def _obb_collision_pose_jit(
    x_1,
    y_1,
    heading_1,
    length_1,
    width_1,
    x_2,
    y_2,
    heading_2,
    length_2,
    width_2,
):
    half_length_1 = 0.5 * length_1
    half_width_1 = 0.5 * width_1
    half_length_2 = 0.5 * length_2
    half_width_2 = 0.5 * width_2

    cos_1 = math.cos(heading_1)
    sin_1 = math.sin(heading_1)
    cos_2 = math.cos(heading_2)
    sin_2 = math.sin(heading_2)

    axis_1_long_x, axis_1_long_y = cos_1, sin_1
    axis_1_lat_x, axis_1_lat_y = -sin_1, cos_1
    axis_2_long_x, axis_2_long_y = cos_2, sin_2
    axis_2_lat_x, axis_2_lat_y = -sin_2, cos_2

    relative_x = x_2 - x_1
    relative_y = y_2 - y_1
    broad_radius = (
        math.hypot(half_length_1, half_width_1)
        + math.hypot(half_length_2, half_width_2)
        + CURRENT_COLLISION_EPS
    )
    if relative_x * relative_x + relative_y * relative_y > broad_radius**2:
        return False

    for axis_index in range(4):
        if axis_index == 0:
            normal_x, normal_y = axis_1_long_x, axis_1_long_y
        elif axis_index == 1:
            normal_x, normal_y = axis_1_lat_x, axis_1_lat_y
        elif axis_index == 2:
            normal_x, normal_y = axis_2_long_x, axis_2_long_y
        else:
            normal_x, normal_y = axis_2_lat_x, axis_2_lat_y

        center_distance = abs(
            relative_x * normal_x + relative_y * normal_y
        )
        projected_radius = (
            half_length_1
            * abs(axis_1_long_x * normal_x + axis_1_long_y * normal_y)
            + half_width_1
            * abs(axis_1_lat_x * normal_x + axis_1_lat_y * normal_y)
            + half_length_2
            * abs(axis_2_long_x * normal_x + axis_2_long_y * normal_y)
            + half_width_2
            * abs(axis_2_lat_x * normal_x + axis_2_lat_y * normal_y)
        )
        if (
            center_distance
            > projected_radius + CURRENT_COLLISION_EPS
        ):
            return False
    return True


@solver.njit(cache=True, fastmath=False)
def _batch_any_model_collision_coarse_jit(
    state_array,
    acceleration_1,
    acceleration_2,
    valid_mask,
    time_grid,
):
    output = np.zeros(state_array.shape[0], dtype=np.bool_)
    for row in range(state_array.shape[0]):
        if not valid_mask[row]:
            continue

        x_1, y_1, speed_1, heading_1, length_1, width_1, yaw_rate_1 = (
            state_array[row, :7]
        )
        x_2, y_2, speed_2, heading_2, length_2, width_2, yaw_rate_2 = (
            state_array[row, 7:]
        )
        if (
            length_1 <= 0.0
            or width_1 <= 0.0
            or length_2 <= 0.0
            or width_2 <= 0.0
        ):
            output[row] = True
            continue

        found_collision = False
        for model_1 in range(3):
            model_yaw_rate_1 = 0.0 if model_1 == 0 else yaw_rate_1
            model_acceleration_1 = (
                acceleration_1[row] if model_1 == 2 else 0.0
            )
            for model_2 in range(3):
                model_yaw_rate_2 = 0.0 if model_2 == 0 else yaw_rate_2
                model_acceleration_2 = (
                    acceleration_2[row] if model_2 == 2 else 0.0
                )
                for time_index in range(time_grid.shape[0]):
                    time_value = time_grid[time_index]
                    pose_1 = _motion_pose_jit(
                        x_1,
                        y_1,
                        speed_1,
                        heading_1,
                        model_yaw_rate_1,
                        model_acceleration_1,
                        time_value,
                    )
                    pose_2 = _motion_pose_jit(
                        x_2,
                        y_2,
                        speed_2,
                        heading_2,
                        model_yaw_rate_2,
                        model_acceleration_2,
                        time_value,
                    )
                    if _obb_collision_pose_jit(
                        pose_1[0],
                        pose_1[1],
                        pose_1[2],
                        length_1,
                        width_1,
                        pose_2[0],
                        pose_2[1],
                        pose_2[2],
                        length_2,
                        width_2,
                    ):
                        found_collision = True
                        break
                if found_collision:
                    break
            if found_collision:
                break
        output[row] = found_collision
    return output


@solver.njit(cache=True, fastmath=False)
def _model_speed_at_time_jit(speed, acceleration, model, time_value):
    """Return non-reversing speed for CV/CTRV/CTRA in the prefilter."""

    initial_speed = max(0.0, speed)
    if model != 2:
        return initial_speed
    return max(0.0, initial_speed + acceleration * time_value)


@solver.njit(cache=True, fastmath=False)
def _add_fine_only_collision_rows_jit(
    coarse_found,
    state_array,
    acceleration_1,
    acceleration_2,
    valid_mask,
    coarse_time,
    fine_time,
):
    """Adaptively find fine-only baseline contacts missed by coarse samples.

    An OBB contact in one coarse interval requires the two centre trajectories
    to enter the sum of their OBB circumradii after at most the largest
    physically possible relative travel in that interval.  Only intervals
    passing this conservative broad-phase test are sampled on the fine grid.
    """

    output = coarse_found.copy()
    if fine_time.shape[0] < 2 or coarse_time.shape[0] < 2:
        return output
    fine_step = fine_time[1] - fine_time[0]
    fine_count = fine_time.shape[0]

    for row in range(state_array.shape[0]):
        if not valid_mask[row] or output[row]:
            continue

        x_1, y_1, speed_1, heading_1, length_1, width_1, yaw_rate_1 = (
            state_array[row, :7]
        )
        x_2, y_2, speed_2, heading_2, length_2, width_2, yaw_rate_2 = (
            state_array[row, 7:]
        )
        radius = (
            math.hypot(0.5 * length_1, 0.5 * width_1)
            + math.hypot(0.5 * length_2, 0.5 * width_2)
        )

        found_collision = False
        for model_1 in range(3):
            model_yaw_rate_1 = 0.0 if model_1 == 0 else yaw_rate_1
            model_acceleration_1 = acceleration_1[row] if model_1 == 2 else 0.0
            for model_2 in range(3):
                model_yaw_rate_2 = 0.0 if model_2 == 0 else yaw_rate_2
                model_acceleration_2 = acceleration_2[row] if model_2 == 2 else 0.0

                for interval in range(coarse_time.shape[0] - 1):
                    start_time = coarse_time[interval]
                    end_time = coarse_time[interval + 1]
                    interval_duration = end_time - start_time
                    pose_1_start = _motion_pose_jit(
                        x_1, y_1, speed_1, heading_1, model_yaw_rate_1,
                        model_acceleration_1, start_time,
                    )
                    pose_2_start = _motion_pose_jit(
                        x_2, y_2, speed_2, heading_2, model_yaw_rate_2,
                        model_acceleration_2, start_time,
                    )
                    separation = math.hypot(
                        pose_2_start[0] - pose_1_start[0],
                        pose_2_start[1] - pose_1_start[1],
                    )
                    max_speed_1 = max(
                        _model_speed_at_time_jit(speed_1, model_acceleration_1, model_1, start_time),
                        _model_speed_at_time_jit(speed_1, model_acceleration_1, model_1, end_time),
                    )
                    max_speed_2 = max(
                        _model_speed_at_time_jit(speed_2, model_acceleration_2, model_2, start_time),
                        _model_speed_at_time_jit(speed_2, model_acceleration_2, model_2, end_time),
                    )
                    if separation > radius + (max_speed_1 + max_speed_2) * interval_duration:
                        continue

                    fine_start = int(math.ceil((start_time - 1e-12) / fine_step))
                    fine_end = int(math.floor((end_time + 1e-12) / fine_step))
                    if fine_start < 0:
                        fine_start = 0
                    if fine_end >= fine_count:
                        fine_end = fine_count - 1
                    for fine_index in range(fine_start, fine_end + 1):
                        time_value = fine_time[fine_index]
                        pose_1 = _motion_pose_jit(
                            x_1, y_1, speed_1, heading_1, model_yaw_rate_1,
                            model_acceleration_1, time_value,
                        )
                        pose_2 = _motion_pose_jit(
                            x_2, y_2, speed_2, heading_2, model_yaw_rate_2,
                            model_acceleration_2, time_value,
                        )
                        if _obb_collision_pose_jit(
                            pose_1[0], pose_1[1], pose_1[2], length_1, width_1,
                            pose_2[0], pose_2[1], pose_2[2], length_2, width_2,
                        ):
                            found_collision = True
                            break
                    if found_collision:
                        break
                if found_collision:
                    break
            if found_collision:
                break
        output[row] = found_collision
    return output


def find_rows_with_any_predicted_collision(
    state_array: np.ndarray,
    acceleration_1: np.ndarray,
    acceleration_2: np.ndarray,
    valid_mask: np.ndarray,
    config: EAConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Fast conservative prefilter used before the full EA direction search."""

    coarse_time, _, _ = solver.get_time_grids(
        config.prediction_horizon_s,
        config.coarse_step_s,
    )
    coarse_found = _batch_any_model_collision_coarse_jit(
        np.asarray(state_array, dtype=np.float64),
        np.asarray(acceleration_1, dtype=np.float64),
        np.asarray(acceleration_2, dtype=np.float64),
        np.asarray(valid_mask, dtype=np.bool_),
        np.asarray(coarse_time, dtype=np.float64),
    )
    if config.fine_step_s >= config.coarse_step_s or not np.any(
        np.asarray(valid_mask, dtype=bool) & (~coarse_found)
    ):
        return coarse_found
    fine_time, _, _ = solver.get_time_grids(
        config.prediction_horizon_s,
        config.fine_step_s,
    )
    return _add_fine_only_collision_rows_jit(
        coarse_found,
        np.asarray(state_array, dtype=np.float64),
        np.asarray(acceleration_1, dtype=np.float64),
        np.asarray(acceleration_2, dtype=np.float64),
        np.asarray(valid_mask, dtype=np.bool_),
        np.asarray(coarse_time, dtype=np.float64),
        np.asarray(fine_time, dtype=np.float64),
    )
