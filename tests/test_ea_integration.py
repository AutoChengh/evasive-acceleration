"""Unique EA integration, independent safety and public provenance checks.

A prebuilt native library is required. Missing dependencies fail the suite;
geometry and precision assertions do not impose machine-specific speed gates.
"""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest

import core_ea
import evasive_acceleration as ea


MODE_KEYS = ("EA_CVCV", "EA_CVCT", "EA_CTCV", "EA_CTCT")
PAIRS = ((0, 0), (0, 1), (1, 0), (1, 1))
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def native_ea():
    import numba
    from evasive_acceleration.native_build import find_native_library
    from evasive_acceleration.runtime import _get_runtime

    assert not numba.config.DISABLE_JIT, "EA requires enabled Numba JIT"
    find_native_library()  # Fail, rather than skip, an incomplete native build.
    return _get_runtime()[0]


def _fictional_state():
    # Published in the independently checked EA example, not private data.
    return np.array([0., 0., 8., 0., 4.8, 1.8, .1,
                     20., 1., 4., math.pi, 4.5, 1.8, 0.])


def _users(state):
    return ea.RoadUserState(*state[:7]), ea.RoadUserState(*state[7:])


def test_missing_numba_fails_explicitly_instead_of_using_an_archived_solver(tmp_path):
    code = """
import builtins
original_import = builtins.__import__
def no_numba(name, *args, **kwargs):
    if name == 'numba' or name.startswith('numba.'):
        raise ImportError('deliberately absent Numba')
    return original_import(name, *args, **kwargs)
builtins.__import__ = no_numba
import evasive_acceleration as ea
a = ea.RoadUserState(0, 0, 1, 0, 4, 2, 0)
b = ea.RoadUserState(0, 30, 1, 0, 4, 2, 0)
try:
    ea.compute_ea(a, b)
except RuntimeError as error:
    assert 'numba' in str(error).lower(), str(error)
else:
    raise AssertionError('EA silently accepted a missing numerical dependency')
print('missing dependency explicitly rejected')
"""
    result = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "missing dependency explicitly rejected" in result.stdout


def test_ea_public_dataclass_legacy_and_four_modes_agree(native_ea):
    state = _fictional_state()
    before = state.copy()
    a, b = _users(state)
    kwargs = ea.build_compute_kwargs(a, b)
    modes = core_ea.compute_ea_modes(**kwargs)
    score = ea.compute_ea(a, b)
    assert set(modes) == set(MODE_KEYS)
    assert score == core_ea.compute_ea(**kwargs)
    assert score == core_ea.aggregate_ea_modes(modes)
    assert score == sum(modes[key] for key in MODE_KEYS) / 4.
    direct = native_ea.compute_selected_modes(state)
    for key, pair in zip(MODE_KEYS, PAIRS):
        assert modes[key] == direct[pair]
    assert score == pytest.approx(1.5366117876437317, abs=.001)
    np.testing.assert_array_equal(state, before)


@pytest.mark.parametrize("yaw_rate", [.1, 6.])
def test_ea_outputs_have_independent_continuous_safety_certificate(native_ea, yaw_rate):
    from evasive_acceleration._experimental import continuous as independent

    state = _fictional_state()
    state[6] = yaw_rate
    if yaw_rate > 5.:
        # A clearly separated spinning body exercises native dispatch cheaply.
        state[2], state[7], state[8], state[9] = 0., 0., 30., 0.
    results = native_ea.compute_selected_modes_with_diagnostics(state)
    for pair, result in results.items():
        assert result.continuous_verified and result.angular_converged
        assert 0. <= result.value - result.lower_bound <= .001 + 1e-12
        ax = result.value * result.direction_x if result.value else 0.
        ay = result.value * result.direction_y if result.value else 0.
        wa = state[6] if pair[0] else 0.
        wb = state[13] if pair[1] else 0.
        status, witness, _ = independent.certify(state, wa, wb, ax, ay,
                                                10., 1e-9, 500000)
        assert status == 1, (pair, result, witness)


def test_ea_causal_history_and_provided_yaw_rate_remain_equivalent(native_ea):
    a = ea.RoadUserKinematicState(0., 0., 8., .02, 4.8, 1.8)
    b = ea.RoadUserKinematicState(20., 1., 4., math.pi, 4.5, 1.8)
    times = [0., .1, .2]
    headings_a, headings_b = [0., .01, .02], [math.pi] * 3
    provided_a, estimate = ea.build_state_with_estimated_yaw_rate(a, times, headings_a)
    provided_b, _ = ea.build_state_with_estimated_yaw_rate(b, times, headings_b)
    assert estimate.initialized and provided_a.yaw_rate == .1
    history = ea.compute_single_frame_ea_from_heading_histories(
        a, times, headings_a, b, times, headings_b,
    )
    assert history == ea.compute_ea(provided_a, provided_b)
    complete = ea.causal_yaw_rate(times + [.3], headings_a + [1.7])
    np.testing.assert_array_equal(ea.causal_yaw_rate(times, headings_a), complete[:3])


def test_ea_catches_contact_between_old_time_samples(native_ea):
    a = ea.RoadUserState(0., 0., 1., 0., .004, .004, 0.)
    b = ea.RoadUserState(.01, -.01, 1., math.pi / 2., .004, .004, 0.)
    coarse = ea.compute_ea(a, b, T_total=.02, dt_coarse=.02, dt_fine=.02)
    fine = ea.compute_ea(a, b, T_total=.02, dt_coarse=.01, dt_fine=.001)
    assert math.isfinite(coarse) and coarse > 0.
    assert abs(coarse - fine) <= .001


def test_ea_rejects_invalid_states_and_options_through_existing_api(native_ea):
    good = ea.RoadUserState(0., 30., 1., 0., 4., 2., 0.)
    for bad in (ea.RoadUserState(0., 0., -1., 0., 4., 2., 0.),
                ea.RoadUserState(0., 0., 1., 0., 0., 2., 0.),
                ea.RoadUserState(math.nan, 0., 1., 0., 4., 2., 0.)):
        with pytest.raises(ValueError):
            ea.compute_ea(bad, good)
    for tolerance in (0., math.nan, math.inf):
        with pytest.raises(ValueError):
            ea.compute_ea(good, good, tol=tolerance)
    kwargs = ea.build_compute_kwargs(good, good)
    kwargs["yawA"] = math.nan
    assert math.isnan(core_ea.compute_ea(**kwargs))


def test_ea_tighter_tolerance_is_respected_for_zero_yaw(native_ea, monkeypatch):
    from evasive_acceleration._experimental import continuous_parallel as preserved

    original = preserved.solve_pair
    observed = []

    def record_budget(state, pair, config, budget=preserved.RetainedBudget()):
        result = original(state, pair, config, budget)
        observed.append((budget.acceleration_tolerance, result))
        return result

    monkeypatch.setattr(preserved, "solve_pair", record_budget)
    a = ea.RoadUserState(0., 0., 8., 0., 4.8, 1.8, 0.)
    b = ea.RoadUserState(20., 1., 4., math.pi, 4.5, 1.8, 0.)
    normal = ea.compute_ea(a, b, tol=.001)
    tighter = ea.compute_ea(a, b, tol=.0001)
    assert normal > 0. and tighter > 0.
    assert abs(normal - tighter) <= .001
    tightened = [result for tolerance, result in observed if tolerance == .0001]
    assert tightened, "custom public tolerance never reached the zero-yaw solver"
    assert all(result.continuous_verified and result.angular_converged
               and 0. <= result.value - result.lower_bound <= .0001 + 1e-12
               for result in tightened)


def test_native_cpp_is_available_as_package_resource():
    from importlib.resources import files

    source = files("evasive_acceleration._experimental").joinpath("native_cells.cpp")
    assert source.is_file(), "source/wheel must include the reproducible native source"
    assert "ea_cells_native_solve_seeded" in source.read_text(encoding="utf-8")


def test_packaged_ea_sources_match_independently_validated_snapshot():
    checker = runpy.run_path(str(ROOT / "tools" / "check_solver_integrity.py"))
    assert checker["check_solver_integrity"]() == []


def _assert_active_ea_provenance(provenance):
    from evasive_acceleration.runtime import CORE_ID

    assert provenance["core_id"] == CORE_ID
    assert provenance["backend"] == "EA"
    hashes = provenance["runtime_source_sha256"]
    assert "_experimental/continuous_adaptive_cells.py" in hashes
    assert "_experimental/native_cells.cpp" in hashes
    package = Path(ea.__file__).resolve().parent
    for relative, expected in hashes.items():
        path = package.parent / relative if relative == "core_ea.py" else package / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
    library = Path(provenance["native_library_path"])
    assert library.is_file()
    assert hashlib.sha256(library.read_bytes()).hexdigest() == provenance["native_library_sha256"]


def _run_cli(*arguments, cwd, env=None):
    return subprocess.run([sys.executable, "-I", "-m", "evasive_acceleration", *arguments],
                          cwd=cwd, env=env, text=True, capture_output=True,
                          encoding="utf-8", timeout=180)


def test_ea_cli_benchmark_records_actual_core_and_source_binary_hashes(native_ea, tmp_path):
    output = tmp_path / "EA benchmark"
    result = _run_cli("benchmark", "--max-frames", "1",
                      "--warmup", "1", "--repeats", "1", "--output-dir", str(output),
                      cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "benchmark_summary.json").read_text(encoding="utf-8"))
    _assert_active_ea_provenance(summary["core_provenance"])
    assert summary["core_id"] == summary["core_provenance"]["core_id"]
    assert summary["passed"] and summary["source_and_inputs_unchanged"]
    assert summary["calls"] == summary["unique_frames"] == 1
    assert summary["status_counts"]["error"] == 0
    assert summary["source_sha256"]["native_library"] == (
        summary["core_provenance"]["native_library_sha256"]
    )
    for key, value in summary["core_provenance"]["runtime_source_sha256"].items():
        assert summary["source_sha256"][key] == value


def test_ea_cli_batch_records_backend_and_causal_yaw_provenance(native_ea, tmp_path):
    source = ROOT / "examples" / "data" / "without_yaw_rate.csv"
    before = source.read_bytes()
    output = tmp_path / "EA batch"
    result = _run_cli("batch", "--input", str(source),
                      "--yaw-rate-source", "estimated", "--skip-baselines",
                      "--output-dir", str(output), cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    metadata_files = list(output.glob("*.run.json"))
    assert len(metadata_files) == 1
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    _assert_active_ea_provenance(metadata["core_provenance"])
    assert metadata["core_id"] == metadata["core_provenance"]["core_id"]
    assert metadata["backend"] == "EA"
    assert metadata["requested_yaw_rate_source"] == "estimated"
    assert metadata["input_sha256"] == hashlib.sha256(before).hexdigest()
    assert metadata["statistics"]["computation_failed_frames"] == 0
    assert metadata["yaw_rate"]["tau_s"] == ea.DEFAULT_TAU_S
    for label in ("A", "B"):
        counts = metadata["yaw_rate_source_counts"][label]
        assert counts["estimated_initialization"] == 1 and counts["estimated"] >= 1
    output_files = list(output.glob("*.csv"))
    assert len(output_files) == 1
    with output_files[0].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["EA_Yawrate_A_source"] == "estimated_initialization"
    assert rows[1]["EA_Yawrate_A_source"] == "estimated"
    assert float(rows[1]["EA_Yawrate_A_used_rad_s"]) == pytest.approx(.05)
    assert source.read_bytes() == before


def test_ea_cli_batch_missing_native_fails_before_creating_outputs(native_ea, tmp_path):
    source = ROOT / "examples" / "data" / "without_yaw_rate.csv"
    before = source.read_bytes()
    output = tmp_path / "must not be created"
    environment = os.environ.copy()
    environment["EA_NATIVE_CELLS_LIBRARY"] = str(tmp_path / "missing library.dll")
    result = _run_cli("batch", "--input", str(source),
                      "--skip-baselines", "--output-dir", str(output),
                      cwd=tmp_path, env=environment)
    assert result.returncode != 0
    assert "EA_NATIVE_CELLS_LIBRARY" in result.stderr
    assert "does not exist" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists(), "dependency errors must not create CSVs full of NaNs"
    assert source.read_bytes() == before


def test_ea_benchmark_warms_every_selected_input_before_any_measurement(tmp_path, monkeypatch):
    from evasive_acceleration import benchmark

    cases = [{"source_file": f"synthetic:{index}", "csv_row": "",
              "kwargs": {"sentinel": index}} for index in range(5)]
    observed = []

    def evaluate(kwargs):
        observed.append(kwargs["sentinel"])
        return dict.fromkeys(MODE_KEYS, 0.), 0.

    # Isolate scheduling from the expensive solver so this regression also
    # exercises scheduling independently of numerical execution.
    monkeypatch.setattr(benchmark, "_demo_cases", lambda horizon: cases)
    monkeypatch.setattr(benchmark, "_evaluate", evaluate)
    monkeypatch.setattr(benchmark, "_environment", lambda: {})
    monkeypatch.setattr(benchmark, "_source_hashes", lambda: {"core_ea.py": "stable"})
    monkeypatch.setattr(core_ea, "get_core_provenance", lambda: {
        "core_id": "test-EA", "backend": "EA",
    })
    output = tmp_path / "fully warmed benchmark"
    assert benchmark.main(["--warmup", "1", "--repeats", "1",
                           "--output-dir", str(output)]) == 0
    assert observed == list(range(5)) + list(range(5))
    summary = json.loads((output / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary["warmup_requested_calls"] == 1
    assert summary["warmup_calls"] == len(summary["warmup_elapsed_ms_excluded"]) == 5
    assert summary["warmup_policy"] == "every selected input at least once"
    assert summary["calls"] == summary["unique_frames"] == 5
    assert summary["passed"] and summary["source_and_inputs_unchanged"]
