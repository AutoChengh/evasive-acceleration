"""End-to-end checks for the documented batch workflow and failure handling."""

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import batch_compute
from single_frame import RoadUserState, compute_single_frame_ea
from yaw_rate import causal_yaw_rate


def _frames(count=4, *, with_yaw=True):
    """Two non-colliding actors make workflow tests fast and deterministic."""
    data = pd.DataFrame(
        {
            "Time (s)": np.arange(count, dtype=float) * 0.1,
            "Position X (m)": np.arange(count, dtype=float) * 0.5,
            "Position Y (m)": 0.0,
            "Velocity (m/s)": 5.0,
            "Heading": np.arange(count, dtype=float) ** 2 * 0.01,
            "Length (m)": 4.5,
            "Width (m)": 1.8,
            "Vehicle Number": 1,
            "2_Position X (m)": np.arange(count, dtype=float) * 0.5,
            "2_Position Y (m)": 100.0,
            "2_Velocity (m/s)": 5.0,
            "2_Heading": 0.0,
            "2_Length (m)": 4.5,
            "2_Width (m)": 1.8,
            "2_Vehicle Number": 2,
        }
    )
    if with_yaw:
        data["Yawrate"] = 0.123456789
        data["2_Yawrate"] = -0.087654321
    return data


def _run(tmp_path, data, name="case", **kwargs):
    input_file = tmp_path / f"{name}.csv"
    data.to_csv(input_file, index=False)
    stat = batch_compute.process_one_csv(
        str(input_file), str(tmp_path / "results"), **kwargs
    )
    return input_file, stat, pd.read_csv(stat["output_file"])


def test_auto_all_provided_needs_no_time_or_estimator(tmp_path, monkeypatch):
    data = _frames().drop(columns="Time (s)")

    def unexpected_estimator(*args, **kwargs):
        raise AssertionError("No estimation is needed when every yaw rate is finite")

    monkeypatch.setattr(batch_compute, "_estimate_yaw_rate_stream", unexpected_estimator)
    _, stat, result = _run(tmp_path, data)

    np.testing.assert_array_equal(
        result["EA_Yawrate_A_used_rad_s"], data["Yawrate"]
    )
    np.testing.assert_array_equal(
        result["EA_Yawrate_B_used_rad_s"], data["2_Yawrate"]
    )
    assert result["EA_Yawrate_A_source"].eq("provided").all()
    assert result["EA_Yawrate_B_source"].eq("provided").all()
    assert stat["computed_success_frames"] == len(data)
    assert result["EA_error"].fillna("").eq("").all()


def test_auto_estimates_missing_values_using_all_prior_headings(tmp_path):
    data = _frames()
    data.loc[2, "Yawrate"] = np.nan
    expected = causal_yaw_rate(data["Time (s)"], data["Heading"])
    _, _, result = _run(tmp_path, data)

    assert result.loc[2, "EA_Yawrate_A_source"] == "estimated"
    assert result.loc[2, "EA_Yawrate_A_used_rad_s"] == expected[2]
    assert result.loc[2, "EA_Yawrate_A_samples"] == 3
    assert result.loc[0, "EA_Yawrate_A_used_rad_s"] == data.loc[0, "Yawrate"]
    assert result["EA_Yawrate_B_source"].eq("provided").all()
    assert result["EA"].notna().all()


def test_auto_reports_timestamp_column_only_when_estimation_is_needed(tmp_path):
    data = _frames().drop(columns="Time (s)")
    data.loc[1, "Yawrate"] = np.nan
    source = tmp_path / "missing_time.csv"
    data.to_csv(source, index=False)

    with pytest.raises((ValueError, KeyError), match=r"Time \(s\)"):
        batch_compute.process_one_csv(str(source), str(tmp_path / "results"))
    assert not (tmp_path / "results" / "missing_time_EA.csv").exists()


def test_mode_output_paths_are_separate_and_metadata_records_run(tmp_path, monkeypatch):
    output_root = tmp_path / "default_outputs"
    monkeypatch.setattr(batch_compute, "DEFAULT_OUTPUT_DIR", output_root)
    data = _frames(2)
    source = tmp_path / "case.csv"
    data.to_csv(source, index=False)

    for mode in ("provided", "estimated", "auto"):
        stat = batch_compute.process_one_csv(str(source), yaw_rate_source=mode)
        output = Path(stat["output_file"])
        assert output == output_root / mode / "case_EA.csv"
        metadata_file = output.with_suffix(".run.json")
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        assert metadata["requested_yaw_rate_source"] == mode
        assert metadata["solver"]
        assert metadata["yaw_rate"]["tau_s"] == 0.6
        assert metadata["yaw_rate"]["deadband_rad_s"] == 0.002
        assert metadata["statistics"]["total_frames"] == 2
        assert metadata["statistics"]["computed_success_frames"] == 2


@pytest.mark.parametrize("existing_suffix", [".csv", ".run.json"])
def test_existing_result_or_metadata_requires_explicit_overwrite(
    tmp_path, existing_suffix
):
    source = tmp_path / "case.csv"
    _frames(1).to_csv(source, index=False)
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    protected_file = output_dir / f"case_EA{existing_suffix}"
    protected_file.write_text("existing result", encoding="utf-8")

    with pytest.raises(FileExistsError):
        batch_compute.process_one_csv(str(source), str(output_dir))
    assert protected_file.read_text(encoding="utf-8") == "existing result"

    stat = batch_compute.process_one_csv(
        str(source), str(output_dir), overwrite=True
    )
    assert Path(stat["output_file"]).exists()
    assert Path(stat["output_file"]).with_suffix(".run.json").exists()


def test_overwrite_never_allows_output_to_replace_input(tmp_path, monkeypatch):
    source = tmp_path / "case.csv"
    _frames(1).to_csv(source, index=False)
    original_bytes = source.read_bytes()
    monkeypatch.setattr(
        batch_compute, "make_output_csv_path", lambda *args, **kwargs: str(source)
    )

    with pytest.raises((ValueError, FileExistsError)):
        batch_compute.process_one_csv(str(source), str(tmp_path), overwrite=True)
    assert source.read_bytes() == original_bytes


def test_baseline_failure_does_not_discard_successful_ea(tmp_path, monkeypatch):
    def broken_baseline(*args, **kwargs):
        raise RuntimeError("baseline test failure")

    monkeypatch.setattr(batch_compute, "compute_real_time_metrics", broken_baseline)
    _, stat, result = _run(tmp_path, _frames(2))

    assert result["EA"].notna().all()
    assert result["EA_error"].fillna("").eq("").all()
    assert result["EA_baseline_error"].str.contains("baseline test failure").all()
    assert stat["computed_success_frames"] == 2
    assert stat["computation_failed_frames"] == 0


def test_skip_baselines_does_not_invoke_or_publish_comparison_metrics(tmp_path, monkeypatch):
    def unexpected_baseline(*args, **kwargs):
        raise AssertionError("Baseline metrics were explicitly disabled")

    monkeypatch.setattr(batch_compute, "compute_real_time_metrics", unexpected_baseline)
    _, stat, result = _run(tmp_path, _frames(1), skip_baselines=True)
    assert result["EA"].notna().all()
    assert stat["baseline_failed_frames"] == 0
    assert not set(batch_compute.ANALYTICAL_OUTPUT_COLUMNS).intersection(result.columns)
    metadata = json.loads(Path(stat["metadata_file"]).read_text(encoding="utf-8"))
    assert metadata["baselines_enabled"] is False


def test_invalid_timestamp_has_row_reason_and_recovery_uses_actual_dt(tmp_path):
    data = _frames(4, with_yaw=False)
    data["Time (s)"] = [0.0, 0.1, 0.1, 0.4]
    data["Heading"] = [0.0, 0.01, 9.0, 0.10]
    _, _, result = _run(tmp_path, data)

    assert pd.isna(result.loc[2, "EA"])
    assert "strictly increasing" in str(result.loc[2, "EA_Yawrate_A_error"])
    assert "strictly increasing" in str(result.loc[2, "EA_error"])
    assert result.loc[2, "EA_status"] != result.loc[3, "EA_status"]
    expected = causal_yaw_rate([0.0, 0.1, 0.4], [0.0, 0.01, 0.10])
    assert result.loc[3, "EA_Yawrate_A_used_rad_s"] == expected[-1]
    assert result.loc[3, "EA_Yawrate_A_samples"] == 3
    assert math.isfinite(result.loc[3, "EA"])


def test_invalid_heading_is_named_in_output_and_next_frame_recovers(tmp_path):
    data = _frames(3, with_yaw=False)
    data.loc[1, "Heading"] = np.nan
    _, _, result = _run(tmp_path, data)

    assert pd.isna(result.loc[1, "EA"])
    assert "Heading" in str(result.loc[1, "EA_error"])
    assert result.loc[1, "EA_Yawrate_A_error"]
    assert math.isfinite(result.loc[2, "EA"])
    assert result.loc[2, "EA_Yawrate_A_samples"] == 2


def test_track_id_change_resets_only_that_actor(tmp_path):
    data = _frames(4, with_yaw=False)
    data["Vehicle Number"] = [1, 1, 9, 9]
    data["Heading"] = [0.0, 0.01, 2.0, 2.02]
    _, _, result = _run(tmp_path, data)

    assert result.loc[2, "EA_Yawrate_A_source"] == "estimated_initialization"
    assert result.loc[2, "EA_Yawrate_A_samples"] == 1
    assert result.loc[2, "EA_Yawrate_A_used_rad_s"] == 0.0
    assert result.loc[3, "EA_Yawrate_A_used_rad_s"] == pytest.approx(0.2)
    assert result.loc[2, "EA_Yawrate_B_samples"] == 3


def test_appending_future_frames_does_not_change_yawrate_prefix(tmp_path):
    data = _frames(5, with_yaw=False)
    _, _, full = _run(tmp_path, data, name="full")
    _, _, prefix = _run(tmp_path, data.iloc[:3], name="prefix")
    for column in (
        "EA_Yawrate_A_used_rad_s",
        "EA_Yawrate_B_used_rad_s",
        "EA_Yawrate_A_samples",
        "EA_Yawrate_B_samples",
        "EA",
    ):
        np.testing.assert_array_equal(prefix[column], full[column].iloc[:3])


def test_batch_preserves_the_paper_core_single_frame_value(tmp_path):
    agent_a = RoadUserState(0.0, 0.0, 10.0, 0.0, 4.5, 1.8, 0.0)
    agent_b = RoadUserState(20.0, 0.0, 8.0, math.pi, 4.7, 1.9, 0.0)
    original_value = compute_single_frame_ea(agent_a, agent_b)
    assert original_value == pytest.approx(4.800096523170946, abs=1e-10)
    data = _frames(1)
    data.loc[0, [
        "Position X (m)", "Position Y (m)", "Velocity (m/s)", "Heading",
        "Length (m)", "Width (m)", "Yawrate", "2_Position X (m)",
        "2_Position Y (m)", "2_Velocity (m/s)", "2_Heading",
        "2_Length (m)", "2_Width (m)", "2_Yawrate",
    ]] = [0.0, 0.0, 10.0, 0.0, 4.5, 1.8, 0.0, 20.0, 0.0, 8.0,
          math.pi, 4.7, 1.9, 0.0]
    _, _, result = _run(tmp_path, data, yaw_rate_source="provided")

    assert result.loc[0, "EA"] == round(original_value, 4)


def test_cli_single_input_and_failure_statistics(tmp_path, capsys):
    source = tmp_path / "case.csv"
    _frames(1).to_csv(source, index=False)
    assert batch_compute.main([
        "--input", str(source), "--output-dir", str(tmp_path / "single")
    ]) == 0
    assert (tmp_path / "single" / "case_EA.csv").exists()

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _frames(1).to_csv(data_dir / "valid.csv", index=False)
    pd.DataFrame({"wrong_column": [1]}).to_csv(data_dir / "invalid.csv", index=False)
    capsys.readouterr()
    result_code = batch_compute.main([
        "--input-dir", str(data_dir), "--output-dir", str(tmp_path / "batch")
    ])
    output = capsys.readouterr().out.lower()
    assert result_code != 0
    assert re.search(
        r"files discovered:\s*2;\s*successful:\s*1;\s*incomplete:\s*0;\s*failed:\s*1",
        output,
    )


def test_cli_returns_failure_for_row_errors(tmp_path):
    data = _frames(2, with_yaw=False)
    data["Time (s)"] = [0.1, 0.1]
    source = tmp_path / "row_error.csv"
    data.to_csv(source, index=False)
    assert batch_compute.main([
        "--input", str(source), "--output-dir", str(tmp_path / "results")
    ]) != 0


def test_cli_input_and_input_dir_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit) as exc:
        batch_compute.main(["--input", "case.csv", "--input-dir", str(tmp_path)])
    assert exc.value.code == 2


def test_header_only_csv_reports_no_interaction_rows(tmp_path):
    source = tmp_path / "empty.csv"
    _frames(0).to_csv(source, index=False)
    with pytest.raises(ValueError, match="no interaction rows"):
        batch_compute.process_one_csv(str(source), str(tmp_path / "results"))
    assert not (tmp_path / "results").exists()


def test_duplicate_csv_columns_are_reported_before_pandas_renames_them(tmp_path):
    source = tmp_path / "duplicate.csv"
    source.write_text("Heading,Heading\n0,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate column names.*Heading"):
        batch_compute.process_one_csv(str(source), str(tmp_path / "results"))
    assert not (tmp_path / "results").exists()
