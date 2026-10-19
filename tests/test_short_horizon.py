"""Public APIs support endpoint sampling even when a step exceeds the horizon."""

import math

import pytest

from core_ea import compute_ea as keyword_compute_ea
from evasive_acceleration import RoadUserState, build_compute_kwargs, compute_ea


def _states(*, x_b=100.0, speed_a=10.0):
    return (
        RoadUserState(0.0, 0.0, speed_a, 0.0, 4.0, 2.0, 0.0),
        RoadUserState(x_b, 0.0, 0.0, 0.0, 4.0, 2.0, 0.0),
    )


@pytest.mark.parametrize(
    "options",
    [
        {"T_total": 0.04},
        {"T_total": 0.04, "dt_coarse": 0.1, "dt_fine": 0.2},
        {"T_total": 1.0, "dt_coarse": 3.0, "dt_fine": 0.02},
        {"T_total": 1.0, "dt_coarse": 3.0, "dt_fine": 2.0},
    ],
)
def test_short_horizon_returns_zero_for_separated_trajectories(options):
    agent_a, agent_b = _states()
    assert compute_ea(agent_a, agent_b, **options) == 0.0
    assert keyword_compute_ea(**build_compute_kwargs(agent_a, agent_b, **options)) == 0.0


def test_step_larger_than_horizon_still_checks_the_final_endpoint():
    agent_a, agent_b = _states(x_b=8.0, speed_a=5.0)
    # Only 0 and T are sampled. At T=1 the unmodified centres are 3 m apart,
    # so the 4 m vehicles overlap. A magnitude just above 2 m/s^2 is needed
    # to restore separation; the short grid must not bypass the endpoint.
    options = {"T_total": 1.0, "dt_coarse": 3.0, "dt_fine": 2.0}
    value = compute_ea(agent_a, agent_b, **options)
    assert value > 2.0
    assert value == pytest.approx(2.0, abs=1e-10)
    assert keyword_compute_ea(**build_compute_kwargs(agent_a, agent_b, **options)) == value


@pytest.mark.parametrize("option", ["T_total", "dt_coarse", "dt_fine"])
@pytest.mark.parametrize("value", [0.0, -0.1, math.nan, math.inf, -math.inf])
def test_invalid_time_settings_remain_rejected(option, value):
    agent_a, agent_b = _states()
    options = {option: value}
    with pytest.raises(ValueError):
        compute_ea(agent_a, agent_b, **options)
    kwargs = build_compute_kwargs(agent_a, agent_b)
    kwargs.update(options)
    assert math.isnan(keyword_compute_ea(**kwargs))


def test_short_horizon_preserves_current_contact_infinite_convention():
    agent_a, agent_b = _states(x_b=4.0)
    assert compute_ea(agent_a, agent_b, T_total=0.04) == math.inf
