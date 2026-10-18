import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from batch_compute import process_one_csv
from single_frame import RoadUserKinematicState, build_state_with_estimated_yaw_rate
from yaw_rate import CausalYawRateEWMA, causal_yaw_rate


def test_documented_numerical_example():
    values = causal_yaw_rate(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        [0.00, 0.01, 0.03, 0.06, 0.10, 0.15],
        tau_s=0.6,
    )
    np.testing.assert_array_equal(
        values,
        np.array([0.0, 0.1, 0.1154, 0.1437, 0.1830, 0.2317]),
    )


def test_wraps_across_pi_boundary():
    values = causal_yaw_rate(
        [0.0, 0.1],
        np.deg2rad([179.0, -179.0]),
        deadband_rad_s=0.0,
        decimals=8,
    )
    assert values[1] == pytest.approx(math.radians(20.0), abs=1e-8)


def test_constant_turn_and_irregular_timestamps():
    times = np.array([0.0, 0.07, 0.20, 0.41, 0.68])
    expected_rate = -0.35
    values = causal_yaw_rate(
        times,
        0.4 + expected_rate * times,
        deadband_rad_s=0.0,
        decimals=8,
    )
    np.testing.assert_allclose(values[1:], expected_rate, atol=1e-8)


def test_prefix_consistency_proves_no_future_dependency():
    times = np.array([0.0, 0.1, 0.21, 0.34, 0.49, 0.66])
    headings = np.array([0.0, 0.01, 0.035, 0.07, 0.12, 0.19])
    complete = causal_yaw_rate(times, headings)
    for end in range(1, len(times) + 1):
        prefix = causal_yaw_rate(times[:end], headings[:end])
        np.testing.assert_array_equal(prefix, complete[:end])


def test_invalid_update_does_not_modify_state():
    estimator = CausalYawRateEWMA()
    estimator.update(0.0, 0.0)
    estimator.update(0.1, 0.01)
    snapshot = (
        estimator.previous_time,
        estimator.previous_heading,
        estimator.rate_state,
        estimator.sample_count,
        estimator.last_estimate,
    )
    with pytest.raises(ValueError):
        estimator.update(0.1, 0.02)
    assert snapshot == (
        estimator.previous_time,
        estimator.previous_heading,
        estimator.rate_state,
        estimator.sample_count,
        estimator.last_estimate,
    )


def test_first_value_is_marked_as_initialization_placeholder():
    state = RoadUserKinematicState(0.0, 0.0, 5.0, 0.2, 4.5, 1.8)
    completed, estimate = build_state_with_estimated_yaw_rate(
        state,
        [12.0],
        [0.2],
    )
    assert completed.yaw_rate == 0.0
    assert not estimate.initialized
    assert estimate.sample_count == 1


def test_batch_estimated_mode_accepts_csv_without_yaw_columns(tmp_path):
    source = pd.read_csv(PROJECT_ROOT / "demo_data" / "InD_18_tracks_417_424.csv").head(3)
    source = source.drop(columns=["Yawrate", "2_Yawrate"])
    input_csv = tmp_path / "without_yawrate.csv"
    output_dir = tmp_path / "results"
    source.to_csv(input_csv, index=False)

    stat = process_one_csv(
        str(input_csv),
        str(output_dir),
        yaw_rate_source="estimated",
    )
    result = pd.read_csv(stat["output_file"])

    assert result["EA"].notna().all()
    assert result.loc[0, "EA_Yawrate_A_source"] == "estimated_initialization"
    assert result.loc[1, "EA_Yawrate_A_source"] == "estimated"
    assert not bool(result.loc[0, "EA_Yawrate_A_initialized"])
    assert bool(result.loc[1, "EA_Yawrate_A_initialized"])


def test_batch_auto_prefers_finite_provided_values(tmp_path):
    source = pd.read_csv(PROJECT_ROOT / "demo_data" / "InD_18_tracks_417_424.csv").head(3)
    source.loc[source.index[1], "Yawrate"] = np.nan
    input_csv = tmp_path / "mixed_yawrate.csv"
    output_dir = tmp_path / "results"
    source.to_csv(input_csv, index=False)

    stat = process_one_csv(
        str(input_csv),
        str(output_dir),
        yaw_rate_source="auto",
    )
    result = pd.read_csv(stat["output_file"])

    assert result.loc[0, "EA_Yawrate_A_source"] == "provided"
    assert result.loc[1, "EA_Yawrate_A_source"] == "estimated"
    assert result.loc[2, "EA_Yawrate_A_source"] == "provided"
