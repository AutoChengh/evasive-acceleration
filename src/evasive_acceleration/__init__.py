"""Public EA API with state containers and causal yaw-rate estimation.

``compute_ea(agent_a, agent_b)`` accepts two :class:`RoadUserState` objects.
The compatible keyword-based interface remains available in ``core_ea``.
CSV dependencies are loaded only when the batch command is used.
"""

from single_frame import (
    RoadUserKinematicState,
    RoadUserState,
    build_compute_kwargs,
    build_state_with_estimated_yaw_rate,
    compute_single_frame_ea,
    compute_single_frame_ea_from_heading_histories,
    compute_single_frame_ea_timed,
)
from yaw_rate import (
    DEFAULT_DEADBAND_RAD_S,
    DEFAULT_DECIMALS,
    DEFAULT_TAU_S,
    CausalYawRateEWMA,
    YawRateEstimate,
    causal_yaw_rate,
    causal_yawrate,
    wrap_angle_difference,
)

__version__ = "0.3.0.dev1"

# One high-level entry point; these names refer to the existing implementation.
compute_ea = compute_single_frame_ea
compute_ea_timed = compute_single_frame_ea_timed

__all__ = [
    "__version__",
    "RoadUserState",
    "RoadUserKinematicState",
    "compute_ea",
    "compute_ea_timed",
    "compute_single_frame_ea",
    "compute_single_frame_ea_timed",
    "compute_single_frame_ea_from_heading_histories",
    "build_compute_kwargs",
    "build_state_with_estimated_yaw_rate",
    "CausalYawRateEWMA",
    "YawRateEstimate",
    "causal_yaw_rate",
    "causal_yawrate",
    "wrap_angle_difference",
    "DEFAULT_TAU_S",
    "DEFAULT_DEADBAND_RAD_S",
    "DEFAULT_DECIMALS",
]
