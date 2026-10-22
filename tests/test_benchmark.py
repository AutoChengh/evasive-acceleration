"""Portable benchmark checks: correctness and provenance, never speed gates."""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from evasive_acceleration import benchmark


def write_input(path, *, yaw="0", missing=None, duplicate=False):
    fields = list(benchmark.COLUMNS) + ["2_" + name for name in benchmark.COLUMNS]
    row = dict(zip(fields, (0, 0, 10, 0, 4.5, 1.8, yaw, 20, 0, 8, math.pi, 4.7, 1.9, 0)))
    if missing:
        fields.remove(missing)
        row.pop(missing)
    if duplicate:
        fields.append(fields[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    return path


def zero_result(kwargs):
    return dict.fromkeys(benchmark.MODE_KEYS, 0.0), 0.0


def test_csv_loader_keeps_provided_yaw_and_original_row(tmp_path):
    path = write_input(tmp_path / "provided.csv", yaw="0.12")
    cases, files = benchmark.load_cases([path])
    assert cases[0]["kwargs"]["yawA"] == 0.12
    assert cases[0]["csv_row"] == 2
    assert files[0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert cases[0]["kwargs"]["T_total"] == 10


@pytest.mark.parametrize("yaw", ["nan", "inf", "", "bad"])
def test_invalid_rows_are_not_silently_dropped(tmp_path, yaw):
    path = write_input(tmp_path / "invalid.csv", yaw=yaw)
    with pytest.raises(ValueError, match="CSV row 2"):
        benchmark.load_cases([path])


def test_missing_yaw_column_has_actionable_error(tmp_path):
    path = write_input(tmp_path / "missing.csv", missing="Yawrate")
    with pytest.raises(ValueError, match="finite provided yaw rates"):
        benchmark.load_cases([path])


def test_duplicate_headers_are_rejected(tmp_path):
    path = write_input(tmp_path / "duplicate.csv", duplicate=True)
    with pytest.raises(ValueError, match="duplicate CSV"):
        benchmark.load_cases([path])


def test_empty_csv_has_no_frames(tmp_path):
    path = write_input(tmp_path / "empty.csv")
    header = path.read_text(encoding="utf-8-sig").splitlines()[0]
    path.write_text(header + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No input frames"):
        benchmark.load_cases([path])


def test_prefix_selection_is_explicit_and_does_not_select_another_file(tmp_path):
    a = write_input(tmp_path / "a.csv")
    b = write_input(tmp_path / "b.csv")
    cases, files = benchmark.load_cases([a, b], max_frames=1, horizon=3)
    assert len(cases) == len(files) == 1
    assert cases[0]["source_file"] == "a.csv"
    assert cases[0]["kwargs"]["T_total"] == 3


@pytest.mark.parametrize("values", [[], [0], [-1], [math.inf], [math.nan]])
def test_bad_latencies_are_rejected_not_cleaned(values):
    with pytest.raises(ValueError, match="finite, positive"):
        benchmark.latency_summary(values)


def test_strict_threshold_includes_every_slow_observation():
    result = benchmark.latency_summary([1, 15, 20, 25, 600])
    assert result["count"] == 5
    assert result["over_15ms"] == 3
    assert result["over_20ms"] == 2
    assert result["mean_ms"] == np.mean([1, 15, 20, 25, 600])
    assert result["max_ms"] == 600


def test_warmup_is_excluded_and_slow_observation_is_retained(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "_evaluate", zero_result)
    ticks = iter(value * 1_000_000 for value in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 40))
    monkeypatch.setattr(benchmark.time, "perf_counter_ns", lambda: next(ticks))
    output = tmp_path / "result"
    assert benchmark.main(["--output-dir", str(output)]) == 0
    result = json.loads((output / "benchmark_summary.json").read_text())
    assert result["unique_frames"] == result["calls"] == 3
    assert result["warmup_elapsed_ms_excluded"] == [1, 1, 1]
    assert result["latency_all_calls"]["mean_ms"] == pytest.approx(32 / 3)
    assert result["latency_all_calls"]["max_ms"] == 30
    assert result["latency_all_calls"]["over_20ms"] == 1
    with (output / "benchmark_frames.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["elapsed_ms"]) for row in rows] == [1, 1, 30]
    assert all(int(row["elapsed_ns"]) / 1e6 == float(row["elapsed_ms"]) for row in rows)


def test_measured_failures_remain_in_output_and_cause_failure(tmp_path, monkeypatch):
    calls = 0

    def failing(kwargs):
        nonlocal calls
        calls += 1
        if calls == 5:  # Three excluded warm-ups, then measured call two.
            raise RuntimeError("simulated solver failure")
        return zero_result(kwargs)

    monkeypatch.setattr(benchmark, "_evaluate", failing)
    output = tmp_path / "with_error"
    assert benchmark.main(["--output-dir", str(output)]) == 1
    result = json.loads((output / "benchmark_summary.json").read_text())
    assert result["passed"] is False
    assert result["calls"] == result["latency_all_calls"]["count"] == 3
    assert result["status_counts"]["error"] == 1
    assert "simulated solver failure" in (output / "benchmark_frames.csv").read_text()


def test_inf_is_not_an_execution_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "_evaluate", lambda _: (dict.fromkeys(benchmark.MODE_KEYS, math.inf), math.inf))
    output = tmp_path / "inf"
    assert benchmark.main(["--output-dir", str(output)]) == 0
    result = json.loads((output / "benchmark_summary.json").read_text())
    assert result["status_counts"]["inf"] == 3
    assert result["status_counts"]["error"] == 0
    assert "Infinity" not in (output / "benchmark_summary.json").read_text()


def test_existing_output_is_protected_before_computation(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(benchmark, "_evaluate", lambda _: pytest.fail("must not compute"))
    marker = tmp_path / "keep.txt"
    marker.write_text("retain")
    assert benchmark.main(["--output-dir", str(tmp_path)]) == 2
    assert marker.read_text() == "retain"
    assert "fresh path" in capsys.readouterr().err


def test_repeat_count_and_input_hash_are_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "_evaluate", zero_result)
    path = write_input(tmp_path / "provided.csv")
    before = path.read_bytes()
    output = tmp_path / "repeated"
    assert benchmark.main(["--input", str(path), "--repeats", "2", "--output-dir", str(output)]) == 0
    result = json.loads((output / "benchmark_summary.json").read_text())
    assert result["unique_frames"] == 1 and result["calls"] == 2 and result["repeats"] == 2
    assert result["inputs"][0]["sha256"] == hashlib.sha256(before).hexdigest()
    assert "_solver/continuous_adaptive_cells.py" in result["source_sha256"]
    assert "_solver/native_cells.cpp" in result["source_sha256"]
    assert path.read_bytes() == before


def test_changed_source_fails_run_without_discarding_measurements(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "_evaluate", zero_result)
    snapshots = iter([{"core_ea.py": "before"}, {"core_ea.py": "after"}])
    monkeypatch.setattr(benchmark, "_source_hashes", lambda: next(snapshots))
    output = tmp_path / "changed"
    assert benchmark.main(["--output-dir", str(output)]) == 1
    result = json.loads((output / "benchmark_summary.json").read_text())
    assert result["source_and_inputs_unchanged"] is False
    assert result["calls"] == 3
    assert (output / "benchmark_frames.csv").exists()


def test_input_error_never_creates_result_directory(tmp_path):
    source = write_input(tmp_path / "bad.csv", yaw="nan")
    output = tmp_path / "not_created"
    assert benchmark.main(["--input", str(source), "--output-dir", str(output)]) == 2
    assert not output.exists()


def test_directory_discovery_is_sorted_and_skips_generated_EA(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "_evaluate", zero_result)
    source = tmp_path / "inputs"
    source.mkdir()
    write_input(source / "b.csv")
    write_input(source / "a.csv")
    write_input(source / "ignored_EA.csv", yaw="nan")
    output = tmp_path / "selected"
    assert benchmark.main(["--input-dir", str(source), "--max-frames", "1", "--output-dir", str(output)]) == 0
    with (output / "benchmark_frames.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["source_file"] == "a.csv"


@pytest.mark.parametrize("flag,value", [("--repeats", "0"), ("--warmup", "0"), ("--max-frames", "-1"), ("--horizon", "nan")])
def test_invalid_options_exit_before_computation(flag, value):
    with pytest.raises(SystemExit) as exc:
        benchmark.main([flag, value])
    assert exc.value.code == 2


def test_current_core_matches_public_numerical_example(aligned_straight_ea_infimum):
    modes, score = benchmark._evaluate(benchmark._demo_cases(10)[0]["kwargs"])
    assert set(modes) == set(benchmark.MODE_KEYS)
    exact = aligned_straight_ea_infimum(20. - (4.5 + 4.7) / 2., (1.8 + 1.9) / 2., 18.)
    assert -1e-8 <= score - exact <= .001 + 1e-12


def test_installed_cli_forwards_benchmark_help(tmp_path):
    process = subprocess.run([sys.executable, "-I", "-m", "evasive_acceleration", "benchmark", "--help"],
                             cwd=tmp_path, capture_output=True, text=True, encoding="utf-8")
    assert process.returncode == 0, process.stderr
    assert "--input-dir" in process.stdout and "--warmup" in process.stdout


def test_frozen_large_cohort_and_adapter_evidence_are_separately_reported():
    source = Path(__file__).resolve().parents[1] / "docs/data/ea_acceptance_20261022.json"
    result = json.loads(source.read_text())
    frozen = result["frozen_acceptance"]
    integration = result["public_adapter_integration"]
    assert frozen["raw_frames"] == 857820
    assert frozen["warmup_calls_excluded"] == 5993
    assert 0. < frozen["mean_ms"] < frozen["p95_ms"] < frozen["p99_ms"] < frozen["max_ms"]
    assert frozen["max_ms"] < 20. and frozen["over_20ms"] == 0
    assert frozen["errors"] == frozen["nan_frames"] == 0
    assert frozen["global_bracket_tolerance_mps2"] == .001
    assert frozen["max_finite_reference_difference_mps2"] <= .001
    assert integration["mode_comparisons"] == integration["independent_continuous_checks_passed"]
    assert integration["exact_adapter_identity"] is True
    assert integration["max_absolute_adapter_difference_mps2"] == 0.
    assert "Raw inputs and historical observations were not replaced" in result["note"]
