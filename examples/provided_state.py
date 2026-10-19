"""Compute one EA value from states that already include yaw rate.

Install the package, then run ``python examples/provided_state.py``.
The expected result is approximately 4.800 m/s² with the paper numerical core.
"""

import math

from evasive_acceleration import RoadUserState, compute_ea


def main() -> None:
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
    ea = compute_ea(agent_a, agent_b)
    print(f"EA = {ea:.3f} m/s^2")


if __name__ == "__main__":
    main()
