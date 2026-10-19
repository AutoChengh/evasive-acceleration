"""Regression tests for the fine-grid feasibility contract and search flow."""

from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest

from evasive_acceleration._paper import calculator as frozen
from evasive_acceleration._paper import numerical_solver as geometry
from evasive_acceleration._paper import verified_calculator as verified
from evasive_acceleration._paper import verified_solver
from evasive_acceleration._paper.config import DEFAULT_CONFIG, MotionModel
from evasive_acceleration._paper.four_modes import (
    FOUR_PAIRS, _prediction_pairs, compute_selected_modes_with_diagnostics,
)


def _fixture_state(case_id):
    fixture = json.loads((
        Path(__file__).parent / "fixtures" / "paper_core_reference.json"
    ).read_text(encoding="utf-8"))
    case = next(case for case in fixture["cases"] if case["id"] == case_id)
    return np.asarray([*case["agent_a"], *case["agent_b"]], dtype=np.float64)


def _prune_counterexample():
    # Seed 4391, sample 137 from the independently recorded source audit.
    return np.asarray([
        0.0, 0.0, 30.39263067772737, -1.1308523447998549,
        3.4759324213929874, 2.420372809444441, 0.11359056806944157,
        25.95159218456495, -16.06754496589043, 18.049474687635414,
        -2.7738708745827676, 1.7494284009843821, 1.517793950875798,
        0.6000851650789436,
    ])


def _one_pair(state=None, config=DEFAULT_CONFIG):
    if state is None:
        state = _fixture_state("turning_both")
    pair = (MotionModel.CTRV, MotionModel.CTRV)
    coarse, fine = _prediction_pairs(state, config, (pair,))
    return coarse[pair], fine[pair]


def _ray_value(data, x, y, config=DEFAULT_CONFIG):
    return verified_solver.solve_direction_min_acceleration(
        data["relx_arr"], data["rely_arr"], data["axes_x"], data["axes_y"],
        data["sum_radius"], data["t_arr"], data["t2_arr"], data["R2"],
        float(x), float(y), config.max_extra_acceleration_mps2,
        config.solver_tolerance,
    )


@pytest.mark.parametrize("horizon,step", [(0.25, 0.1), (0.3, 0.1), (0.14, 0.02), (1.0, 2.0)])
def test_time_grid_includes_horizon_without_enlarging_step(horizon, step):
    times, half_squared = verified.make_time_grid(horizon, step)
    assert times[0] == 0.0
    assert times[-1] == horizon
    assert np.all(np.diff(times) > 0)
    assert np.max(np.diff(times)) <= step + 1e-14
    assert np.array_equal(half_squared, 0.5 * times * times)
    assert not times.flags.writeable


def test_time_grid_uses_requested_samples_not_rescaled_linspace():
    times, _ = verified.make_time_grid(0.25, 0.1)
    assert np.array_equal(times, [0.0, 0.1, 0.2, 0.25])


def test_all_local_directions_and_original_coarse_direction_are_fine_solved(monkeypatch):
    coarse, fine = _one_pair()
    calls = []
    original = verified._search_directions

    def record(data, x, y, config):
        calls.append((data is fine, len(x), np.asarray(x).copy(), np.asarray(y).copy()))
        return original(data, x, y, config)

    monkeypatch.setattr(verified, "_search_directions", record)
    result = verified.solve_prediction(coarse, fine, DEFAULT_CONFIG)
    assert calls[0][:2] == (False, 72)
    assert calls[1][:2] == (True, 102)
    assert result.evaluated_fine_directions == 102
    assert result.verified
    assert not verified._candidate_collision(fine, result.value, result.direction_x, result.direction_y)


def test_known_coarse_only_return_is_now_fine_verified():
    coarse, fine = _one_pair(_prune_counterexample())
    assert verified._candidate_collision(
        fine, 4.777277013559797, -0.523985905970079, 0.8517269341430477,
    ), "The recorded frozen-paper counterexample must remain a real collision"
    assert frozen._compute_ea_for_prediction(coarse, fine, DEFAULT_CONFIG) == pytest.approx(
        4.777277013559797, abs=1e-10,
    )
    result = verified.solve_prediction(coarse, fine, DEFAULT_CONFIG)
    assert result.verified
    assert math.isfinite(result.value) and result.value > 0.0
    assert not verified._candidate_collision(fine, result.value, result.direction_x, result.direction_y)


def test_local_answer_equals_unpruned_minimum_of_all_fine_rays():
    coarse, fine = _one_pair(_prune_counterexample())
    x, y = geometry.get_sector_unit_vectors(DEFAULT_CONFIG.coarse_direction_count)
    _, index, _ = verified._search_directions(coarse, x, y, DEFAULT_CONFIG)
    _, local_x, local_y = geometry.build_local_unit_vectors(
        math.atan2(y[index], x[index]), DEFAULT_CONFIG.local_direction_half_window_deg,
        DEFAULT_CONFIG.local_direction_count,
    )
    ray_values = [_ray_value(fine, x[index], y[index])]
    ray_values.extend(_ray_value(fine, xx, yy) for xx, yy in zip(local_x, local_y))
    expected = min(value for value in ray_values if math.isfinite(value))
    assert verified.solve_prediction(coarse, fine, DEFAULT_CONFIG).value == pytest.approx(
        expected, abs=1e-12,
    )


def test_refinement_ratio_no_longer_selects_fine_directions():
    coarse, fine = _one_pair(_prune_counterexample())
    low = verified.solve_prediction(coarse, fine, replace(DEFAULT_CONFIG, fine_refine_ratio=0.01))
    high = verified.solve_prediction(coarse, fine, replace(DEFAULT_CONFIG, fine_refine_ratio=100.0))
    assert low == high


def test_fine_only_contact_uses_full_fine_localization_before_refinement():
    state = _fixture_state("fine_only_contact")
    config = replace(DEFAULT_CONFIG, prediction_horizon_s=0.2)
    coarse, fine = _one_pair(state, config)
    assert not verified._baseline_collision(coarse)
    assert verified._baseline_collision(fine)
    result = verified.solve_prediction(coarse, fine, config)
    assert result.search_grid == "fine-only-local"
    assert result.evaluated_fine_directions == 72 + 102
    assert result.value > 0.0 and result.verified
    assert not verified._candidate_collision(fine, result.value, result.direction_x, result.direction_y)


def test_nominal_fine_step_cannot_make_final_grid_coarser():
    state = _fixture_state("turning_both")
    config = replace(DEFAULT_CONFIG, coarse_step_s=0.02, fine_step_s=0.1)
    coarse_pairs, fine_pairs = _prediction_pairs(state, config)
    for pair in FOUR_PAIRS:
        assert np.array_equal(coarse_pairs[pair]["t_arr"], fine_pairs[pair]["t_arr"])
        assert np.max(np.diff(fine_pairs[pair]["t_arr"])) <= 0.02 + 1e-14


def test_local_failure_uses_a_fine_verified_full_circle_fallback(monkeypatch):
    coarse, fine = _one_pair()
    original = verified._search_directions

    def local_failure(data, x, y, config):
        if data is fine and len(x) == 102:
            return math.inf, -1, True
        return original(data, x, y, config)

    monkeypatch.setattr(verified, "_search_directions", local_failure)
    result = verified.solve_prediction(coarse, fine, DEFAULT_CONFIG)
    assert result.search_grid == "fine-full-circle-fallback"
    assert result.verified
    assert not verified._candidate_collision(fine, result.value, result.direction_x, result.direction_y)


def test_coarse_failure_can_be_rescued_on_fine_grid(monkeypatch):
    coarse, fine = _one_pair()
    original = verified._search_directions

    def coarse_failure(data, x, y, config):
        return (math.inf, -1, True) if data is coarse else original(data, x, y, config)

    monkeypatch.setattr(verified, "_search_directions", coarse_failure)
    result = verified.solve_prediction(coarse, fine, DEFAULT_CONFIG)
    assert result.search_grid == "fine-rescue-local"
    assert result.verified and math.isfinite(result.value)


def test_no_fine_candidate_never_falls_back_to_a_raw_coarse_value(monkeypatch):
    coarse, fine = _one_pair()
    original = verified._search_directions

    def fine_failure(data, x, y, config):
        return (math.inf, -1, True) if data is fine else original(data, x, y, config)

    monkeypatch.setattr(verified, "_search_directions", fine_failure)
    result = verified.solve_prediction(coarse, fine, DEFAULT_CONFIG)
    assert math.isinf(result.value)
    assert result.search_grid == "fine-unresolved"
    assert not result.verified


def test_final_guard_rejects_a_colliding_solver_candidate(monkeypatch):
    coarse, fine = _one_pair()
    original = verified._search_directions

    def corrupt_fine_answer(data, x, y, config):
        return (0.0, 0, True) if data is fine else original(data, x, y, config)

    monkeypatch.setattr(verified, "_search_directions", corrupt_fine_answer)
    with pytest.raises(RuntimeError, match="colliding fine-grid candidate"):
        verified.solve_prediction(coarse, fine, DEFAULT_CONFIG)


def test_current_contact_retains_separate_infinite_output_convention():
    state = np.asarray([0, 0, 0, 0, 4, 2, 0, 0, 2, 0, 0, 4, 2, 0], dtype=float)
    outputs = compute_selected_modes_with_diagnostics(state, DEFAULT_CONFIG)
    assert all(math.isinf(result.value) for result in outputs.values())
    assert all(result.search_grid == "current-contact" for result in outputs.values())


def test_separated_zero_is_checked_on_the_complete_fine_grid():
    state = np.asarray([0, 0, 5, 0, 4, 2, 0, 0, 100, 5, 0, 4, 2, 0], dtype=float)
    outputs = compute_selected_modes_with_diagnostics(state, DEFAULT_CONFIG)
    _, fine = _prediction_pairs(state, DEFAULT_CONFIG)
    assert all(result.value == 0.0 and result.verified for result in outputs.values())
    assert all(not verified._candidate_collision(data, 0.0, 0.0, 0.0) for data in fine.values())
