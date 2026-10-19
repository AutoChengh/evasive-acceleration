"""Causal acceleration estimation and CV/CTRV/CTRA prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import EAConfig, MODEL_ORDER, MotionModel


@dataclass(frozen=True)
class PoseTrajectory:
    """Predicted center position and heading over one time grid."""

    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray


def estimate_causal_acceleration(
    times: np.ndarray,
    speeds: np.ndarray,
    config: EAConfig,
) -> np.ndarray:
    """Estimate longitudinal acceleration using only current and past frames.

    The first frame is assigned zero. For frame ``i > 0``,
    ``a[i] = (v[i] - v[i-1]) / (t[i] - t[i-1])`` when both frames are finite
    and time increases. Invalid differences fall back to zero. The result is
    clipped to the configured physical range.
    """

    times = np.asarray(times, dtype=np.float64)
    speeds = np.asarray(speeds, dtype=np.float64)
    acceleration = np.zeros(speeds.shape[0], dtype=np.float64)
    if speeds.shape[0] <= 1:
        return acceleration

    delta_t = times[1:] - times[:-1]
    delta_v = speeds[1:] - speeds[:-1]
    valid = (
        np.isfinite(times[1:])
        & np.isfinite(times[:-1])
        & np.isfinite(speeds[1:])
        & np.isfinite(speeds[:-1])
        & (delta_t > 1e-9)
    )

    acceleration_tail = acceleration[1:]
    acceleration_tail[valid] = delta_v[valid] / delta_t[valid]
    np.clip(
        acceleration,
        config.observed_acceleration_min_mps2,
        config.observed_acceleration_max_mps2,
        out=acceleration,
    )
    return acceleration


def estimate_causal_yaw_rate(
    times: np.ndarray,
    headings: np.ndarray,
) -> np.ndarray:
    """Estimate yaw rate from the current and preceding heading only.

    The heading increment is wrapped to ``[-pi, pi]`` before division, so a
    passage through the angular branch cut does not create a false turn.  The
    first frame and invalid/non-increasing intervals are assigned zero.  This
    is only a missing-data fallback for the stored causal EWMA yaw-rate column;
    it never accesses a later CSV row.
    """

    times = np.asarray(times, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)
    yaw_rate = np.zeros(headings.shape[0], dtype=np.float64)
    if headings.shape[0] <= 1:
        return yaw_rate

    delta_t = times[1:] - times[:-1]
    valid = (
        np.isfinite(times[1:])
        & np.isfinite(times[:-1])
        & np.isfinite(headings[1:])
        & np.isfinite(headings[:-1])
        & (delta_t > 1e-9)
    )
    delta_heading = np.arctan2(
        np.sin(headings[1:] - headings[:-1]),
        np.cos(headings[1:] - headings[:-1]),
    )
    yaw_rate_tail = yaw_rate[1:]
    yaw_rate_tail[valid] = delta_heading[valid] / delta_t[valid]
    return yaw_rate


def _motion_parameters(
    model: MotionModel,
    yaw_rate: float,
    acceleration: float,
) -> tuple[float, float]:
    if model is MotionModel.CV:
        return 0.0, 0.0
    if model is MotionModel.CTRV:
        return float(yaw_rate), 0.0
    if model is MotionModel.CTRA:
        return float(yaw_rate), float(acceleration)
    raise ValueError(f"Unsupported motion model: {model}")


def predict_pose(
    *,
    x: float,
    y: float,
    speed: float,
    heading: float,
    yaw_rate: float,
    acceleration: float,
    time_grid: np.ndarray,
) -> PoseTrajectory:
    """Predict a non-reversing constant-turn-rate-and-acceleration trajectory.

    Setting acceleration to zero gives CTRV; additionally setting yaw rate to
    zero gives CV. Under braking, propagation stops when speed reaches zero so
    a constant negative acceleration never creates artificial reverse motion.
    """

    x = float(x)
    y = float(y)
    speed = max(0.0, float(speed))
    heading = float(heading)
    yaw_rate = float(yaw_rate)
    acceleration = float(acceleration)
    tau = np.asarray(time_grid, dtype=np.float64)

    if acceleration < 0.0:
        if speed <= 0.0:
            effective_tau = np.zeros_like(tau)
        else:
            effective_tau = np.minimum(tau, speed / (-acceleration))
    else:
        effective_tau = tau

    if abs(yaw_rate) < 1e-6:
        distance = (
            speed * effective_tau
            + 0.5 * acceleration * effective_tau * effective_tau
        )
        x_values = x + math.cos(heading) * distance
        y_values = y + math.sin(heading) * distance
        headings = np.full_like(effective_tau, heading, dtype=np.float64)
    else:
        theta = heading + yaw_rate * effective_tau
        x_values = x + (speed / yaw_rate) * (
            np.sin(theta) - math.sin(heading)
        )
        x_values += acceleration * (
            effective_tau * np.sin(theta) / yaw_rate
            + (np.cos(theta) - math.cos(heading)) / (yaw_rate * yaw_rate)
        )
        y_values = y - (speed / yaw_rate) * (
            np.cos(theta) - math.cos(heading)
        )
        y_values += acceleration * (
            -effective_tau * np.cos(theta) / yaw_rate
            + (np.sin(theta) - math.sin(heading)) / (yaw_rate * yaw_rate)
        )
        headings = theta

    return PoseTrajectory(
        x=np.asarray(x_values, dtype=np.float64),
        y=np.asarray(y_values, dtype=np.float64),
        heading=np.asarray(headings, dtype=np.float64),
    )


def predict_all_models(
    *,
    x: float,
    y: float,
    speed: float,
    heading: float,
    yaw_rate: float,
    acceleration: float,
    time_grid: np.ndarray,
) -> dict[MotionModel, PoseTrajectory]:
    """Predict CV, CTRV, and CTRA trajectories for one road user."""

    predictions: dict[MotionModel, PoseTrajectory] = {}
    for model in MODEL_ORDER:
        model_yaw_rate, model_acceleration = _motion_parameters(
            model,
            yaw_rate,
            acceleration,
        )
        predictions[model] = predict_pose(
            x=x,
            y=y,
            speed=speed,
            heading=heading,
            yaw_rate=model_yaw_rate,
            acceleration=model_acceleration,
            time_grid=time_grid,
        )
    return predictions
