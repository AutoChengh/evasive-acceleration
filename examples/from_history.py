"""Estimate two yaw rates causally and compute EA for the current frame.

Install the package, then run ``python examples/from_history.py``.
Only headings at or before the current decision time are supplied.
"""

import math

from evasive_acceleration import (
    RoadUserKinematicState,
    build_state_with_estimated_yaw_rate,
    compute_ea,
)


def main() -> None:
    # The current states and final headings refer to t = 0.0 seconds.
    # Available observations are from -0.2, -0.1, and 0.0 seconds.
    times_s = [-0.2, -0.1, 0.0]
    agent_a = RoadUserKinematicState(
        x=0.0,
        y=0.0,
        speed=10.0,
        heading=0.0,
        length=4.5,
        width=1.8,
    )
    agent_b = RoadUserKinematicState(
        x=20.0,
        y=0.0,
        speed=8.0,
        heading=math.pi,
        length=4.7,
        width=1.9,
    )
    state_a, estimate_a = build_state_with_estimated_yaw_rate(
        agent_a,
        times_s,
        [-0.01, -0.005, 0.0],
    )
    state_b, estimate_b = build_state_with_estimated_yaw_rate(
        agent_b,
        times_s,
        [math.pi, math.pi, math.pi],
    )
    for label, estimate in (("A", estimate_a), ("B", estimate_b)):
        print(
            f"{label}: yaw rate = {estimate.value_rad_s:.4f} rad/s, "
            f"initialized = {estimate.initialized}, "
            f"samples = {estimate.sample_count}"
        )
    ea = compute_ea(state_a, state_b)
    print(f"EA = {ea:.3f} m/s^2")


if __name__ == "__main__":
    main()
