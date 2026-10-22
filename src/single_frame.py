"""
single_frame.py

Single-frame entry script for evasive acceleration (EA) computation.

This module is intended for real-time or instant-evaluation use cases, where
the user wants to input the current instantaneous states of two road users and
obtain the EA value for that single frame immediately.

It uses continuous-time EA through ``core_ea.py`` and provides:
    - a clean dataclass-based state interface;
    - basic input validation;
    - a minimal command-line interface;
    - an example runnable entry point.

State definition for each road user (7 values):
    x, y, speed, heading, length, width, yaw_rate

All angles are in radians.
All distances are in meters.
Speeds are in meters/second.
Yaw rates are in radians/second.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

from core_ea import (
    DEFAULT_A_MAX,
    DEFAULT_COARSE_SECTOR_NUM,
    DEFAULT_DT_COARSE,
    DEFAULT_DT_FINE,
    DEFAULT_FINE_REFINE_RATIO,
    DEFAULT_LOCAL_FINE_HALF_WINDOW_DEG,
    DEFAULT_LOCAL_FINE_DIR_NUM,
    DEFAULT_T_TOTAL,
    DEFAULT_TOL,
    compute_ea,
    compute_ea_timed,
)
from yaw_rate import (
    DEFAULT_DEADBAND_RAD_S,
    DEFAULT_DECIMALS,
    DEFAULT_TAU_S,
    CausalYawRateEWMA,
    YawRateEstimate,
    wrap_angle_difference,
)

__all__ = [
    "RoadUserState",
    "RoadUserKinematicState",
    "build_state_with_estimated_yaw_rate",
    "build_compute_kwargs",
    "compute_single_frame_ea",
    "compute_single_frame_ea_from_heading_histories",
    "compute_single_frame_ea_timed",
]


# ============================================================================
# Formatting helpers
# ============================================================================
def _format_float_3(x: float) -> str:
    """Format a float for compact console display."""
    xf = float(x)

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
# Public state container
# ============================================================================
@dataclass(frozen=True)
class RoadUserState:
    """
    Instantaneous state of one road user.

    Parameters
    ----------
    x : float
        Global x position [m].
    y : float
        Global y position [m].
    speed : float
        Speed magnitude [m/s].
    heading : float
        Heading angle [rad].
    length : float
        Length of the oriented bounding box [m].
    width : float
        Width of the oriented bounding box [m].
    yaw_rate : float
        Yaw rate [rad/s].
    """

    x: float
    y: float
    speed: float
    heading: float
    length: float
    width: float
    yaw_rate: float

    def validate(self) -> None:
        """Validate the road-user state."""
        values = (
            self.x,
            self.y,
            self.speed,
            self.heading,
            self.length,
            self.width,
            self.yaw_rate,
        )

        for value in values:
            if not math.isfinite(float(value)):
                raise ValueError("All state values must be finite real numbers.")

        if self.length <= 0.0 or self.width <= 0.0:
            raise ValueError("length and width must be strictly positive.")
        if self.speed < 0.0:
            raise ValueError("speed is a magnitude and must be non-negative.")

    def as_tuple(self) -> Tuple[float, float, float, float, float, float, float]:
        """Return the state as a 7-tuple."""
        return (
            float(self.x),
            float(self.y),
            float(self.speed),
            float(self.heading),
            float(self.length),
            float(self.width),
            float(self.yaw_rate),
        )

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "RoadUserState":
        """
        Build a RoadUserState from a length-7 sequence.

        Expected order:
            [x, y, speed, heading, length, width, yaw_rate]
        """
        if len(values) != 7:
            raise ValueError(
                "A road-user state must contain exactly 7 values: "
                "[x, y, speed, heading, length, width, yaw_rate]."
            )

        state = cls(
            x=float(values[0]),
            y=float(values[1]),
            speed=float(values[2]),
            heading=float(values[3]),
            length=float(values[4]),
            width=float(values[5]),
            yaw_rate=float(values[6]),
        )
        state.validate()
        return state


@dataclass(frozen=True)
class RoadUserKinematicState:
    """Instantaneous road-user state when yaw rate is unavailable."""

    x: float
    y: float
    speed: float
    heading: float
    length: float
    width: float

    def validate(self) -> None:
        values = (
            self.x,
            self.y,
            self.speed,
            self.heading,
            self.length,
            self.width,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("All kinematic state values must be finite real numbers.")
        if self.length <= 0.0 or self.width <= 0.0:
            raise ValueError("length and width must be strictly positive.")
        if self.speed < 0.0:
            raise ValueError("speed is a magnitude and must be non-negative.")


def build_state_with_estimated_yaw_rate(
    state: RoadUserKinematicState,
    times_s: Sequence[float],
    headings_rad: Sequence[float],
    *,
    tau_s: float = DEFAULT_TAU_S,
    deadband_rad_s: float = DEFAULT_DEADBAND_RAD_S,
    decimals: int = DEFAULT_DECIMALS,
    heading_tolerance_rad: float = 1e-6,
) -> Tuple[RoadUserState, YawRateEstimate]:
    """Attach a strictly causal yaw-rate estimate to an instantaneous state.

    ``times_s`` and ``headings_rad`` must end at the state being evaluated; no
    future observation is permitted.  The final history heading is checked
    against ``state.heading`` so an accidentally misaligned history fails
    loudly instead of silently producing the wrong CTRV input.
    """
    state.validate()
    if len(times_s) != len(headings_rad) or len(times_s) == 0:
        raise ValueError("times_s and headings_rad must be equal-length non-empty sequences")
    if heading_tolerance_rad < 0.0 or not math.isfinite(heading_tolerance_rad):
        raise ValueError("heading_tolerance_rad must be finite and non-negative")

    estimator = CausalYawRateEWMA(tau_s, deadband_rad_s, decimals)
    value = 0.0
    for time_s, heading_rad in zip(times_s, headings_rad):
        value = estimator.update(time_s, heading_rad)

    final_heading = float(headings_rad[-1])
    mismatch = abs(wrap_angle_difference(final_heading - float(state.heading)))
    if mismatch > heading_tolerance_rad:
        raise ValueError(
            "the final heading-history observation does not match state.heading "
            f"(wrapped difference {mismatch:.6g} rad)"
        )

    estimate = estimator.last_estimate
    if estimate is None:  # Defensive; a non-empty history always creates one.
        raise RuntimeError("yaw-rate estimator did not produce an estimate")

    return (
        RoadUserState(
            x=float(state.x),
            y=float(state.y),
            speed=float(state.speed),
            heading=float(state.heading),
            length=float(state.length),
            width=float(state.width),
            yaw_rate=float(value),
        ),
        estimate,
    )


# ============================================================================
# Validation helpers
# ============================================================================
def _validate_solver_config(
    *,
    coarse_sector_num: int,
    local_fine_dir_num: int,
    dt_coarse: float,
    dt_fine: float,
    T_total: float,
    a_max: float,
    tol: float,
    size_scale: float,
    local_fine_half_window_deg: float,
    fine_refine_ratio: float,
) -> None:
    """Validate numerical solver settings."""
    settings = {
        "dt_coarse": dt_coarse, "dt_fine": dt_fine, "T_total": T_total,
        "a_max": a_max, "tol": tol, "size_scale": size_scale,
        "local_fine_half_window_deg": local_fine_half_window_deg,
        "fine_refine_ratio": fine_refine_ratio,
    }
    for name, value in settings.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")
    if coarse_sector_num < 4:
        raise ValueError("coarse_sector_num must be at least 4.")
    if local_fine_dir_num < 3:
        raise ValueError("local_fine_dir_num must be at least 3.")
    if dt_coarse <= 0.0 or dt_fine <= 0.0:
        raise ValueError("dt_coarse and dt_fine must be strictly positive.")
    if T_total <= 0.0:
        raise ValueError("T_total must be strictly positive.")
    if a_max <= 0.0:
        raise ValueError("a_max must be strictly positive.")
    if tol <= 0.0:
        raise ValueError("tol must be strictly positive.")
    if size_scale <= 0.0:
        raise ValueError("size_scale must be strictly positive.")
    if not 0.0 <= local_fine_half_window_deg <= 180.0:
        raise ValueError("local_fine_half_window_deg must be between 0 and 180.")
    if fine_refine_ratio <= 0.0:
        raise ValueError("fine_refine_ratio must be strictly positive.")


# ============================================================================
# Public helpers
# ============================================================================
def build_compute_kwargs(
    agent_a: RoadUserState,
    agent_b: RoadUserState,
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
    Convert two road-user states into the keyword arguments expected by core_ea.
    """
    agent_a.validate()
    agent_b.validate()

    _validate_solver_config(
        coarse_sector_num=int(coarse_sector_num),
        local_fine_dir_num=int(local_fine_dir_num),
        dt_coarse=float(dt_coarse),
        dt_fine=float(dt_fine),
        T_total=float(T_total),
        a_max=float(a_max),
        tol=float(tol),
        size_scale=float(size_scale),
        local_fine_half_window_deg=float(local_fine_half_window_deg),
        fine_refine_ratio=float(fine_refine_ratio),
    )

    return {
        "xA": float(agent_a.x),
        "yA": float(agent_a.y),
        "vA": float(agent_a.speed),
        "hA": float(agent_a.heading),
        "xB": float(agent_b.x),
        "yB": float(agent_b.y),
        "vB": float(agent_b.speed),
        "hB": float(agent_b.heading),
        "lA": float(agent_a.length),
        "wA": float(agent_a.width),
        "lB": float(agent_b.length),
        "wB": float(agent_b.width),
        "yawA": float(agent_a.yaw_rate),
        "yawB": float(agent_b.yaw_rate),
        "coarse_sector_num": int(coarse_sector_num),
        "local_fine_half_window_deg": float(local_fine_half_window_deg),
        "local_fine_dir_num": int(local_fine_dir_num),
        "fine_refine_ratio": float(fine_refine_ratio),
        "a_max": float(a_max),
        "tol": float(tol),
        "T_total": float(T_total),
        "dt_coarse": float(dt_coarse),
        "dt_fine": float(dt_fine),
        "size_scale": float(size_scale),
    }


def compute_single_frame_ea(
    agent_a: RoadUserState,
    agent_b: RoadUserState,
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
    Compute the final EA value for one single frame.

    This is the recommended high-level API for instant single-frame evaluation.
    """
    kwargs = build_compute_kwargs(
        agent_a,
        agent_b,
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
    return float(compute_ea(**kwargs))


def compute_single_frame_ea_from_heading_histories(
    agent_a: RoadUserKinematicState,
    times_a_s: Sequence[float],
    headings_a_rad: Sequence[float],
    agent_b: RoadUserKinematicState,
    times_b_s: Sequence[float],
    headings_b_rad: Sequence[float],
    *,
    yaw_rate_tau_s: float = DEFAULT_TAU_S,
    yaw_rate_deadband_rad_s: float = DEFAULT_DEADBAND_RAD_S,
    yaw_rate_decimals: int = DEFAULT_DECIMALS,
    **solver_kwargs,
) -> float:
    """Compute EA when yaw rates must be estimated from causal histories."""
    state_a, _ = build_state_with_estimated_yaw_rate(
        agent_a,
        times_a_s,
        headings_a_rad,
        tau_s=yaw_rate_tau_s,
        deadband_rad_s=yaw_rate_deadband_rad_s,
        decimals=yaw_rate_decimals,
    )
    state_b, _ = build_state_with_estimated_yaw_rate(
        agent_b,
        times_b_s,
        headings_b_rad,
        tau_s=yaw_rate_tau_s,
        deadband_rad_s=yaw_rate_deadband_rad_s,
        decimals=yaw_rate_decimals,
    )
    return compute_single_frame_ea(state_a, state_b, **solver_kwargs)


def compute_single_frame_ea_timed(
    agent_a: RoadUserState,
    agent_b: RoadUserState,
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
) -> Tuple[float, float]:
    """
    Compute the final EA value for one single frame with runtime reporting.

    Returns
    -------
    ea_value : float
        Final EA value.
    elapsed_seconds : float
        Total runtime in seconds.
    """
    kwargs = build_compute_kwargs(
        agent_a,
        agent_b,
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
    ea_value, _, elapsed = compute_ea_timed(**kwargs)
    return float(ea_value), float(elapsed)


# ============================================================================
# CLI helpers
# ============================================================================
def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute the single-frame evasive acceleration (EA) value from the "
            "instantaneous states of two road users."
        )
    )

    parser.add_argument(
        "--agent-a",
        nargs=7,
        type=float,
        metavar=("x", "y", "speed", "heading", "length", "width", "yaw_rate"),
        help="State of agent A: x y speed heading length width yaw_rate",
    )
    parser.add_argument(
        "--agent-b",
        nargs=7,
        type=float,
        metavar=("x", "y", "speed", "heading", "length", "width", "yaw_rate"),
        help="State of agent B: x y speed heading length width yaw_rate",
    )

    # Accepted only for call compatibility; these do not control EA accuracy.
    parser.add_argument("--coarse-sector-num", type=int, default=DEFAULT_COARSE_SECTOR_NUM, help=argparse.SUPPRESS)
    parser.add_argument("--local-fine-half-window-deg", type=float, default=DEFAULT_LOCAL_FINE_HALF_WINDOW_DEG, help=argparse.SUPPRESS)
    parser.add_argument("--local-fine-dir-num", type=int, default=DEFAULT_LOCAL_FINE_DIR_NUM, help=argparse.SUPPRESS)
    parser.add_argument("--fine-refine-ratio", type=float, default=DEFAULT_FINE_REFINE_RATIO, help=argparse.SUPPRESS)
    parser.add_argument("--a-max", type=float, default=DEFAULT_A_MAX)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--T-total", type=float, default=DEFAULT_T_TOTAL)
    parser.add_argument("--dt-coarse", type=float, default=DEFAULT_DT_COARSE, help=argparse.SUPPRESS)
    parser.add_argument("--dt-fine", type=float, default=DEFAULT_DT_FINE, help=argparse.SUPPRESS)
    parser.add_argument("--size-scale", type=float, default=1.0)

    parser.add_argument(
        "--example",
        action="store_true",
        help="Run the built-in example case.",
    )

    return parser


def _get_example_case() -> Tuple[RoadUserState, RoadUserState]:
    """Return a minimal built-in example case."""
    agent_a = RoadUserState(
        x=0.0,
        y=0.0,
        speed=10.0,
        heading=0.0,
        length=4.5,
        width=1.8,
        yaw_rate=0.0,
    )
    agent_b = RoadUserState(
        x=20.0,
        y=0.0,
        speed=8.0,
        heading=math.pi,
        length=4.7,
        width=1.9,
        yaw_rate=0.0,
    )
    return agent_a, agent_b


def _print_result(ea_value: float, elapsed: float) -> None:
    """Print the computation result in a compact human-readable format."""
    print("EA =", _format_float_3(ea_value), "m/s^2")
    print("Runtime after warm-up =", _format_seconds(elapsed))


# ============================================================================
# Main entry
# ============================================================================
def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    try:
        if args.example or (args.agent_a is None and args.agent_b is None):
            agent_a, agent_b = _get_example_case()
        else:
            if args.agent_a is None or args.agent_b is None:
                parser.error("Both --agent-a and --agent-b must be provided together.")
            agent_a = RoadUserState.from_sequence(args.agent_a)
            agent_b = RoadUserState.from_sequence(args.agent_b)

        print("Warming up the solver (the first call may include Numba compilation)...")
        _ = compute_single_frame_ea(
            agent_a,
            agent_b,
            coarse_sector_num=args.coarse_sector_num,
            local_fine_half_window_deg=args.local_fine_half_window_deg,
            local_fine_dir_num=args.local_fine_dir_num,
            fine_refine_ratio=args.fine_refine_ratio,
            a_max=args.a_max,
            tol=args.tol,
            T_total=args.T_total,
            dt_coarse=args.dt_coarse,
            dt_fine=args.dt_fine,
            size_scale=args.size_scale,
        )

        ea_value, elapsed = compute_single_frame_ea_timed(
            agent_a,
            agent_b,
            coarse_sector_num=args.coarse_sector_num,
            local_fine_half_window_deg=args.local_fine_half_window_deg,
            local_fine_dir_num=args.local_fine_dir_num,
            fine_refine_ratio=args.fine_refine_ratio,
            a_max=args.a_max,
            tol=args.tol,
            T_total=args.T_total,
            dt_coarse=args.dt_coarse,
            dt_fine=args.dt_fine,
            size_scale=args.size_scale,
        )

        _print_result(ea_value, elapsed)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
