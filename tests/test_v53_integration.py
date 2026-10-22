"""Public opt-in v53 integration without replacing the historical default.

Native tests require an explicitly prebuilt library.  Their assertions concern
geometry, precision and interface behavior, not host-specific latency claims.
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
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import core_ea
import evasive_acceleration as ea


MODE_KEYS = ("EA_CVCV", "EA_CVCT", "EA_CTCV", "EA_CTCT")
PAIRS = ((0, 0), (0, 1), (1, 0), (1, 1))
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def native_v53():
    pytest.importorskip("numba", reason="v53 requires the optional fast dependencies")
    from evasive_acceleration.native_build import find_native_library

    try:
        find_native_library()
    except (RuntimeError, FileNotFoundError) as error:
        pytest.skip(f"Build the optional v53 native library first: {error}")
    # Enter through the public loader: it resolves the package's versioned
    # cache and temporarily configures the frozen module's original loader.
    from evasive_acceleration.backends import _get_v53_runtime

    return _get_v53_runtime()[0]


def _fictional_state():
    # Published in the independently checked final-v53 example, not private data.
    return np.array([0., 0., 8., 0., 4.8, 1.8, .1,
                     20., 1., 4., math.pi, 4.5, 1.8, 0.])


def _users(state):
    return ea.RoadUserState(*state[:7]), ea.RoadUserState(*state[7:])


def test_default_backend_and_paper_value_are_preserved():
    assert ea.get_backend() == "paper"
    a = ea.RoadUserState(0., 0., 10., 0., 4.5, 1.8, 0.)
    b = ea.RoadUserState(20., 0., 8., math.pi, 4.7, 1.9, 0.)
    assert ea.compute_ea(a, b) == pytest.approx(4.800096523170946, abs=1e-10)


def test_paper_context_restores_after_nested_context_and_exception():
    assert ea.get_backend() == "paper"
    with pytest.raises(LookupError, match="test sentinel"):
        with ea.use_backend("paper"):
            with ea.use_backend("paper"):
                assert ea.get_backend() == "paper"
                raise LookupError("test sentinel")
    assert ea.get_backend() == "paper"


@pytest.mark.parametrize("invalid", ["v999", "", "V53", None, 53])
def test_unknown_backend_is_rejected_without_changing_default(invalid):
    with pytest.raises((ValueError, TypeError)):
        with ea.use_backend(invalid):
            pytest.fail("invalid backend accepted")
    assert ea.get_backend() == "paper"


def test_import_and_paper_remain_usable_without_numba(tmp_path):
    code = """
import builtins
original_import = builtins.__import__
def no_numba(name, *args, **kwargs):
    if name == 'numba' or name.startswith('numba.'):
        raise ImportError('deliberately absent optional Numba')
    return original_import(name, *args, **kwargs)
builtins.__import__ = no_numba
import evasive_acceleration as ea
assert ea.get_backend() == 'paper'
a = ea.RoadUserState(0, 0, 1, 0, 4, 2, 0)
b = ea.RoadUserState(0, 30, 1, 0, 4, 2, 0)
assert ea.compute_ea(a, b) == 0
try:
    with ea.use_backend('v53'):
        ea.compute_ea(a, b)
except RuntimeError as error:
    assert 'numba' in str(error).lower() or 'fast' in str(error).lower(), str(error)
else:
    raise AssertionError('v53 silently accepted absent required acceleration dependency')
assert ea.get_backend() == 'paper'
print('optional dependency isolation OK')
"""
    result = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "optional dependency isolation OK" in result.stdout


def test_native_backend_context_is_nested_and_thread_local(native_v53):
    with ea.use_backend("v53"):
        assert ea.get_backend() == "v53"
        with ea.use_backend("paper"):
            assert ea.get_backend() == "paper"
        assert ea.get_backend() == "v53"
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(ea.get_backend).result() == "paper"
    assert ea.get_backend() == "paper"


def test_v53_public_dataclass_legacy_and_four_modes_agree(native_v53):
    state = _fictional_state()
    before = state.copy()
    a, b = _users(state)
    with ea.use_backend("v53"):
        kwargs = ea.build_compute_kwargs(a, b)
        modes = core_ea.compute_ea_modes(**kwargs)
        score = ea.compute_ea(a, b)
        assert set(modes) == set(MODE_KEYS)
        assert score == core_ea.compute_ea(**kwargs)
        assert score == core_ea.aggregate_ea_modes(modes)
        assert score == sum(modes[key] for key in MODE_KEYS) / 4.
        direct = native_v53.compute_selected_modes(state)
        for key, pair in zip(MODE_KEYS, PAIRS):
            assert modes[key] == direct[pair]
        assert score == pytest.approx(1.5366117876437317, abs=.001)
    np.testing.assert_array_equal(state, before)


@pytest.mark.parametrize("yaw_rate", [.1, 6.])
def test_v53_outputs_have_independent_continuous_safety_certificate(native_v53, yaw_rate):
    from evasive_acceleration._experimental import continuous as independent

    state = _fictional_state()
    state[6] = yaw_rate
    if yaw_rate > 5.:
        # A clearly separated spinning body exercises native dispatch cheaply.
        state[2], state[7], state[8], state[9] = 0., 0., 30., 0.
    results = native_v53.compute_selected_modes_with_diagnostics(state)
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


def test_v53_causal_history_and_provided_yaw_rate_remain_equivalent(native_v53):
    a = ea.RoadUserKinematicState(0., 0., 8., .02, 4.8, 1.8)
    b = ea.RoadUserKinematicState(20., 1., 4., math.pi, 4.5, 1.8)
    times = [0., .1, .2]
    headings_a, headings_b = [0., .01, .02], [math.pi] * 3
    provided_a, estimate = ea.build_state_with_estimated_yaw_rate(a, times, headings_a)
    provided_b, _ = ea.build_state_with_estimated_yaw_rate(b, times, headings_b)
    assert estimate.initialized and provided_a.yaw_rate == .1
    with ea.use_backend("v53"):
        history = ea.compute_single_frame_ea_from_heading_histories(
            a, times, headings_a, b, times, headings_b,
        )
        assert history == ea.compute_ea(provided_a, provided_b)
    complete = ea.causal_yaw_rate(times + [.3], headings_a + [1.7])
    np.testing.assert_array_equal(ea.causal_yaw_rate(times, headings_a), complete[:3])


def test_v53_catches_contact_between_old_time_samples(native_v53):
    a = ea.RoadUserState(0., 0., 1., 0., .004, .004, 0.)
    b = ea.RoadUserState(.01, -.01, 1., math.pi / 2., .004, .004, 0.)
    with ea.use_backend("v53"):
        coarse = ea.compute_ea(a, b, T_total=.02, dt_coarse=.02, dt_fine=.02)
        fine = ea.compute_ea(a, b, T_total=.02, dt_coarse=.01, dt_fine=.001)
    assert math.isfinite(coarse) and coarse > 0.
    assert abs(coarse - fine) <= .001


def test_v53_rejects_invalid_states_and_options_through_existing_api(native_v53):
    good = ea.RoadUserState(0., 30., 1., 0., 4., 2., 0.)
    with ea.use_backend("v53"):
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


def test_v53_tighter_tolerance_is_respected_for_zero_yaw(native_v53, monkeypatch):
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
    with ea.use_backend("v53"):
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


def test_packaged_v53_sources_match_independently_validated_snapshot():
    checker = runpy.run_path(str(ROOT / "tools" / "check_v53_integrity.py"))
    assert checker["check_v53_integrity"]() == []


def _assert_active_v53_provenance(provenance):
    from evasive_acceleration.backends import V53_CORE_ID

    assert provenance["core_id"] == V53_CORE_ID
    assert provenance["backend"] == "v53"
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


def test_v53_cli_benchmark_records_actual_core_and_source_binary_hashes(native_v53, tmp_path):
    output = tmp_path / "v53 benchmark"
    result = _run_cli("--backend", "v53", "benchmark", "--max-frames", "1",
                      "--warmup", "1", "--repeats", "1", "--output-dir", str(output),
                      cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "benchmark_summary.json").read_text(encoding="utf-8"))
    _assert_active_v53_provenance(summary["core_provenance"])
    assert summary["core_id"] == summary["core_provenance"]["core_id"]
    assert summary["passed"] and summary["source_and_inputs_unchanged"]
    assert summary["calls"] == summary["unique_frames"] == 1
    assert summary["status_counts"]["error"] == 0
    assert summary["source_sha256"]["native_library"] == (
        summary["core_provenance"]["native_library_sha256"]
    )
    for key, value in summary["core_provenance"]["runtime_source_sha256"].items():
        assert summary["source_sha256"][key] == value


def test_v53_cli_batch_records_backend_and_causal_yaw_provenance(native_v53, tmp_path):
    source = ROOT / "examples" / "data" / "without_yaw_rate.csv"
    before = source.read_bytes()
    output = tmp_path / "v53 batch"
    result = _run_cli("batch", "--backend", "v53", "--input", str(source),
                      "--yaw-rate-source", "estimated", "--skip-baselines",
                      "--output-dir", str(output), cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    metadata_files = list(output.glob("*.run.json"))
    assert len(metadata_files) == 1
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    _assert_active_v53_provenance(metadata["core_provenance"])
    assert metadata["core_id"] == metadata["core_provenance"]["core_id"]
    assert metadata["backend"] == "v53"
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


def test_v53_cli_batch_missing_native_fails_before_creating_outputs(native_v53, tmp_path):
    source = ROOT / "examples" / "data" / "without_yaw_rate.csv"
    before = source.read_bytes()
    output = tmp_path / "must not be created"
    environment = os.environ.copy()
    environment["EA_NATIVE_CELLS_LIBRARY"] = str(tmp_path / "missing library.dll")
    result = _run_cli("--backend", "v53", "batch", "--input", str(source),
                      "--skip-baselines", "--output-dir", str(output),
                      cwd=tmp_path, env=environment)
    assert result.returncode != 0
    assert "EA_NATIVE_CELLS_LIBRARY" in result.stderr
    assert "does not exist" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists(), "dependency errors must not create CSVs full of NaNs"
    assert source.read_bytes() == before


def test_v53_benchmark_warms_every_selected_input_before_any_measurement(tmp_path, monkeypatch):
    from evasive_acceleration import benchmark

    cases = [{"source_file": f"synthetic:{index}", "csv_row": "",
              "kwargs": {"sentinel": index}} for index in range(5)]
    observed = []

    def evaluate(kwargs):
        observed.append(kwargs["sentinel"])
        return dict.fromkeys(MODE_KEYS, 0.), 0.

    # Isolate scheduling from the expensive solver so this regression also
    # runs in the deliberately no-Numba CI jobs.
    monkeypatch.setattr(benchmark, "_demo_cases", lambda horizon: cases)
    monkeypatch.setattr(benchmark, "_evaluate", evaluate)
    monkeypatch.setattr(benchmark, "_environment", lambda: {})
    monkeypatch.setattr(benchmark, "_source_hashes", lambda: {"core_ea.py": "stable"})
    monkeypatch.setattr(core_ea, "get_core_provenance", lambda: {
        "core_id": "test-v53", "backend": "v53",
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
