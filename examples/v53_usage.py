"""Use v53 with existing state and causal-history helpers.

Install ``.[fast]`` and run ``python -m evasive_acceleration.native_build``
before this example. No compiler is launched by an EA call.
"""

import math

from evasive_acceleration import (
    RoadUserKinematicState,
    RoadUserState,
    build_state_with_estimated_yaw_rate,
    compute_ea,
    get_backend,
    use_backend,
)


def main() -> None:
    agent_a = RoadUserState(
        x=0.0, y=0.0, speed=10.0, heading=0.0,
        length=4.5, width=1.8, yaw_rate=0.0,
    )
    agent_b = RoadUserState(
        x=20.0, y=0.0, speed=8.0, heading=math.pi,
        length=4.7, width=1.9, yaw_rate=0.0,
    )
    previous_backend = get_backend()
    with use_backend("v53"):
        print(f"Backend: {get_backend()}")
        print(f"Provided yaw rates: EA = {compute_ea(agent_a, agent_b):.6f} m/s^2")

        # Only observations at/before the current decision time t = 0 are used.
        current_a = RoadUserKinematicState(
            x=0.0, y=0.0, speed=10.0, heading=0.0,
            length=4.5, width=1.8,
        )
        estimated_a, estimate = build_state_with_estimated_yaw_rate(
            current_a,
            times_s=[-0.2, -0.1, 0.0],
            headings_rad=[-0.01, -0.005, 0.0],
        )
        print(
            f"Causal A yaw rate: {estimate.value_rad_s:.4f} rad/s; "
            f"initialized: {estimate.initialized}"
        )
        print(f"Estimated A yaw rate: EA = {compute_ea(estimated_a, agent_b):.6f} m/s^2")

    assert get_backend() == previous_backend
    print(f"Restored backend: {get_backend()}")


if __name__ == "__main__":
    main()
