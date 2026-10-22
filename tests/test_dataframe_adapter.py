"""Verify explicit units, actor selection, causal alignment and batch reuse."""

import importlib.util
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = PROJECT_ROOT / "examples" / "from_dataframe.py"
spec = importlib.util.spec_from_file_location("ea_dataframe_example", EXAMPLE_PATH)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


@pytest.fixture
def tracks():
    rows = []
    for t in (0.0, 0.1, 0.2):
        rows.extend([
            [t, 101, t * 3, 1.0 + t * 4, 3.0, 4.0, math.atan2(4.0, 3.0), 4.5, 1.8],
            [t, 202, 30.0 - t * 3, 2.0, -3.0, 0.0, math.pi, 4.7, 1.9],
        ])
    return pd.DataFrame(rows, columns=adapter.REQUIRED_FIELDS)


def test_custom_columns_units_reflection_and_body_heading(tracks):
    custom = tracks.rename(columns={field: "my_" + field for field in tracks.columns})
    custom["my_timestamp"] *= 1000.0
    # A small, accepted orientation difference must be preserved, not replaced.
    custom.loc[custom["my_track_id"] == 101, "my_heading"] += 0.01
    custom["my_heading"] = np.rad2deg(custom["my_heading"])
    before = custom.copy(deep=True)
    pair = adapter.to_pair_dataframe(
        custom, actor_a=101, actor_b=202,
        columns={field: "my_" + field for field in tracks.columns},
        time_unit="ms", angle_unit="deg", y_down=True,
    )
    np.testing.assert_allclose(pair["Time (s)"], [0.0, 0.1, 0.2])
    np.testing.assert_allclose(pair["Velocity (m/s)"], 5.0)
    np.testing.assert_allclose(pair["Position Y (m)"], [-1.0, -1.4, -1.8])
    np.testing.assert_allclose(pair["Heading"], -math.atan2(4.0, 3.0) - 0.01)
    # Body yaw is deliberately different from atan2(vy, vx).
    assert pair.loc[0, "Heading"] != pytest.approx(-math.atan2(4.0, 3.0))
    assert "Yawrate" not in pair
    pd.testing.assert_frame_equal(custom, before)


def test_exact_overlap_without_interpolation(tracks):
    tracks.loc[tracks.index[-1], "timestamp"] = 0.3
    pair = adapter.to_pair_dataframe(tracks, actor_a=101, actor_b=202)
    assert pair["Time (s)"].tolist() == [0.0, 0.1]
    assert len(pair) == 2
    prefix = tracks.loc[tracks["timestamp"] <= 0.1]
    pd.testing.assert_frame_equal(
        pair, adapter.to_pair_dataframe(prefix, actor_a=101, actor_b=202)
    )


@pytest.mark.parametrize("timestamp", [0.0, -0.1])
def test_duplicate_or_out_of_order_observations_are_rejected(tracks, timestamp):
    tracks.loc[2, "timestamp"] = timestamp
    with pytest.raises(ValueError, match="strictly increasing.*input order"):
        adapter.to_pair_dataframe(tracks, actor_a=101, actor_b=202)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [("x", np.inf, "finite number"), ("heading", np.nan, "finite number"),
     ("vx", "bad", "finite number"), ("length", 0.0, "strictly positive"),
     ("width", -1.0, "strictly positive")],
)
def test_invalid_selected_state_has_actionable_error(tracks, column, value, message):
    tracks[column] = tracks[column].astype(object)
    tracks.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        adapter.to_pair_dataframe(tracks, actor_a=101, actor_b=202)


def test_missing_actor_and_no_overlap_are_explicit(tracks):
    with pytest.raises(ValueError, match="same ID type"):
        adapter.to_pair_dataframe(tracks, actor_a="101", actor_b="202")
    with pytest.raises(ValueError, match="two different actors"):
        adapter.to_pair_dataframe(tracks, actor_a=101, actor_b=101)
    tracks.loc[tracks["track_id"] == 202, "timestamp"] += 10.0
    with pytest.raises(ValueError, match="no exactly shared timestamps"):
        adapter.to_pair_dataframe(tracks, actor_a=101, actor_b=202)


def test_optional_measured_yaw_rate_converts_units_and_missing_values(tracks):
    tracks["yaw_rate"] = [10.0, 0.0, np.nan, 0.0, 10.0, 0.0]
    tracks["heading"] = np.rad2deg(tracks["heading"])
    pair = adapter.to_pair_dataframe(
        tracks, actor_a=101, actor_b=202, angle_unit="deg", y_down=True
    )
    assert pair.loc[0, "Yawrate"] == pytest.approx(-math.radians(10.0))
    assert math.isnan(pair.loc[1, "Yawrate"])


def test_reverse_motion_cannot_be_enabled_by_relaxing_alignment_tolerance(tracks):
    tracks.loc[tracks["track_id"] == 101, "heading"] += math.pi
    with pytest.raises(ValueError, match="does not support reverse motion"):
        adapter.to_pair_dataframe(
            tracks, actor_a=101, actor_b=202, max_velocity_heading_error_deg=89.0
        )


def test_sideways_motion_is_rejected_but_small_explicit_tolerance_is_supported(tracks):
    tracks.loc[tracks["track_id"] == 101, "heading"] += math.radians(8)
    with pytest.raises(ValueError, match="velocity/body-heading difference.*exceeds"):
        adapter.to_pair_dataframe(tracks, actor_a=101, actor_b=202)
    pair = adapter.to_pair_dataframe(
        tracks, actor_a=101, actor_b=202, max_velocity_heading_error_deg=10.0
    )
    assert pair.loc[0, "Heading"] == pytest.approx(math.atan2(4, 3) + math.radians(8))


@pytest.mark.parametrize("tolerance", [-1.0, 90.0, 180.0, np.inf, np.nan])
def test_alignment_tolerance_cannot_disable_forward_motion_contract(tracks, tolerance):
    with pytest.raises(ValueError, match=r"max_velocity_heading_error_deg.*\[0, 90\)"):
        adapter.to_pair_dataframe(
            tracks, actor_a=101, actor_b=202, max_velocity_heading_error_deg=tolerance
        )


def test_stationary_and_tiny_speed_keep_body_heading_without_course_rejection(tracks):
    tracks.loc[tracks["track_id"] == 101, ["vx", "vy", "heading"]] = [0.0, 0.0, 2.0]
    tracks.loc[tracks["track_id"] == 202, ["vx", "vy", "heading"]] = [1e-7, 0.0, math.pi]
    pair = adapter.to_pair_dataframe(tracks, actor_a=101, actor_b=202)
    np.testing.assert_allclose(pair["Velocity (m/s)"], 0.0)
    np.testing.assert_allclose(pair["Heading"], 2.0)
    np.testing.assert_allclose(pair["2_Heading"], math.pi)


def test_compute_reuses_batch_outputs_and_refuses_overwrite(tracks, tmp_path):
    result = adapter.compute_from_dataframe(
        tracks, actor_a=101, actor_b=202, output_dir=tmp_path, skip_baselines=True
    )
    assert Path(result["input_path"]).is_file()
    assert Path(result["metadata_file"]).is_file()
    output = pd.read_csv(result["output_file"])
    assert result["computed_success_frames"] == 3
    assert result["adapter"]["max_velocity_heading_error_deg"] == 5.0
    assert output["EA_status"].tolist() == ["ok"] * 3
    assert output["EA_Yawrate_A_source"].tolist() == [
        "estimated_initialization", "estimated", "estimated",
    ]
    assert output["EA_Yawrate_A_used_rad_s"].tolist() == [0.0, 0.0, 0.0]
    previous = Path(result["output_file"]).read_bytes()
    with pytest.raises(FileExistsError, match="Already exists"):
        adapter.compute_from_dataframe(
            tracks, actor_a=101, actor_b=202, output_dir=tmp_path, skip_baselines=True
        )
    assert Path(result["output_file"]).read_bytes() == previous


def test_cli_accepts_custom_columns_and_string_actor_ids(tmp_path):
    columns = {
        "timestamp": "time_ms", "track_id": "object_key", "x": "east_m", "y": "north_m",
        "vx": "vx_mps", "vy": "vy_mps", "heading": "yaw_deg",
        "length": "box_length_m", "width": "box_width_m",
    }
    arguments = []
    for field, source in columns.items():
        arguments.extend(["--column", f"{field}={source}"])
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH), "--input",
         str(PROJECT_ROOT / "examples/data/custom_tracks.csv"),
         "--actor-a", "ego", "--actor-b", "other", "--time-unit", "ms",
         "--angle-unit", "deg", "--output-dir", str(tmp_path),
         "--skip-baselines", *arguments],
        text=True, encoding="utf-8", capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    pair = pd.read_csv(tmp_path / "pair_input.csv")
    assert pair["Time (s)"].tolist() == [0.0, 0.1, 0.2, 0.3, 0.4]
    assert pair["Vehicle Number"].tolist() == ["ego"] * 5
