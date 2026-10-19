"""Check paper-core equivalence and time the four-mode API against old v2.

This is an implementation-migration check, not a new paper experiment. The old
core is read from Git history into ignored outputs; it is never shipped in the
package. Both timed functions compute only the same four EA components, with
identical settings and provided yaw rates. Warm-up, reference calculation, CSV
I/O, and report writing are excluded from per-frame timings.

Examples (from the repository root)::

    python tools/validate_paper_migration.py --frames 96 --repeats 3
    python tools/validate_paper_migration.py --frames 403 --repeats 3 \
        --paper-source /path/to/verified/paper/src

Use --skip-legacy if the source checkout has no Git history. Optional
--paper-source must contain the independently frozen calculator.py, config.py,
motion.py, and numerical_solver.py; all four hashes are checked before import.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import core_ea

MODE_KEYS = ("EA_CVCV", "EA_CVCT", "EA_CTCV", "EA_CTCT")
PAPER_KEYS = ("EA_CV_CV", "EA_CV_CTRV", "EA_CTRV_CV", "EA_CTRV_CTRV")
STATE_COLUMNS = (
    "Position X (m)", "Position Y (m)", "Velocity (m/s)", "Heading",
    "Length (m)", "Width (m)", "Yawrate",
    "2_Position X (m)", "2_Position Y (m)", "2_Velocity (m/s)", "2_Heading",
    "2_Length (m)", "2_Width (m)", "2_Yawrate",
)
SETTING_MAP = {
    "T_total": "prediction_horizon_s", "dt_coarse": "coarse_step_s",
    "dt_fine": "fine_step_s", "a_max": "max_extra_acceleration_mps2",
    "tol": "solver_tolerance", "coarse_sector_num": "coarse_direction_count",
    "local_fine_half_window_deg": "local_direction_half_window_deg",
    "local_fine_dir_num": "local_direction_count", "fine_refine_ratio": "fine_refine_ratio",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def state_kwargs(a, b):
    return dict(
        xA=a[0], yA=a[1], vA=a[2], hA=a[3], lA=a[4], wA=a[5], yawA=a[6],
        xB=b[0], yB=b[1], vB=b[2], hB=b[3], lB=b[4], wB=b[5], yawB=b[6],
    )


def paper_state(kwargs, size_scale=1.0):
    return np.array([
        kwargs["xA"], kwargs["yA"], kwargs["vA"], kwargs["hA"],
        kwargs["lA"] * size_scale, kwargs["wA"] * size_scale, kwargs["yawA"],
        kwargs["xB"], kwargs["yB"], kwargs["vB"], kwargs["hB"],
        kwargs["lB"] * size_scale, kwargs["wB"] * size_scale, kwargs["yawB"],
    ], dtype=np.float64)


def same_value(a, b, *, atol=1e-10):
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=atol)


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(item) for item in value]
    return value


def check_fixtures(reference):
    rows = []
    for case in reference["cases"]:
        actual = core_ea.compute_ea_modes(
            **state_kwargs(case["agent_a"], case["agent_b"]), **case["settings"]
        )
        actual["EA"] = core_ea.aggregate_ea_modes(actual)
        expected = {
            key: math.inf if value == "inf" else float(value)
            for key, value in case["expected"].items()
        }
        rows.append({
            "id": case["id"], "passed": all(
                same_value(actual[key], expected[key]) for key in (*MODE_KEYS, "EA")
            ), "expected": expected, "actual": actual,
        })
    return rows


def load_paper_reference(source, hashes, output_dir):
    source = Path(source).resolve()
    actual_hashes = {name: source_hash(source / name) for name in hashes}
    if actual_hashes != hashes:
        raise ValueError("Paper reference hashes do not match the frozen 39103 snapshot")
    # Isolate JIT caches from the external freeze and any older import namespace.
    snapshot = Path(output_dir) / "paper_reference_snapshot"
    snapshot.mkdir()
    for name in hashes:
        (snapshot / name).write_bytes((source / name).read_bytes())
    package_name = "_ea_independent_paper_reference"
    package = types.ModuleType(package_name)
    package.__path__ = [str(snapshot)]
    sys.modules[package_name] = package
    calculator = importlib.import_module(package_name + ".calculator")
    config = importlib.import_module(package_name + ".config")
    return calculator, config, actual_hashes


def read_demo_frames(paths, count):
    frames, provenance = [], []
    for path in paths:
        path = Path(path).resolve()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = set(STATE_COLUMNS) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
            for row_index, row in enumerate(reader, start=2):
                state = [float(row[column]) for column in STATE_COLUMNS]
                if not all(math.isfinite(value) for value in state):
                    raise ValueError(f"Non-finite input in {path.name}:{row_index}")
                frames.append({"file": path.name, "csv_row": row_index,
                               "kwargs": state_kwargs(state[:7], state[7:])})
        provenance.append({"path": str(path), "sha256": sha256(path)})
    if not frames:
        raise ValueError("No input frames")
    count = min(count, len(frames))
    indices = np.linspace(0, len(frames) - 1, count, dtype=int)
    return [frames[int(index)] for index in indices], provenance, len(frames)


def load_legacy(ref, output_dir):
    command = ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT),
               "show", f"{ref}:src/core_ea.py"]
    result = subprocess.run(command, capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(
            "Cannot read legacy core from local Git history; use --skip-legacy "
            "or provide an available --legacy-ref. No network fetch was attempted. "
            + result.stderr.decode("utf-8", errors="replace")
        )
    snapshot = output_dir / "legacy_core_snapshot.py"
    snapshot.write_bytes(result.stdout)  # generated read-only evidence artifact
    spec = importlib.util.spec_from_file_location("_ea_legacy_migration_snapshot", snapshot)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, hashlib.sha256(result.stdout).hexdigest()


def time_summary(samples):
    milliseconds = [sample * 1000.0 for sample in samples]
    return {
        "count": len(samples), "mean_ms": statistics.mean(milliseconds),
        "median_ms": statistics.median(milliseconds),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "total_s": sum(samples),
    }


def versions():
    from evasive_acceleration import __version__

    out = {"python": platform.python_version(), "python_executable": sys.executable,
           "source_package_version": __version__}
    for package in ("numpy", "numba", "evasive-acceleration"):
        key = "installed_distribution_version" if package == "evasive-acceleration" else package
        try:
            out[key] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[key] = None
    return out


def _run(argv=None, *, jit_cache_path=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--input", action="append", type=Path)
    parser.add_argument("--paper-source", type=Path)
    parser.add_argument("--legacy-ref", default="e47598e906453377e1a8f1bd5d35d76ac26375ee")
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.frames < 1 or args.repeats < 1:
        parser.error("--frames and --repeats must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = args.output_dir or ROOT / "outputs" / "validation" / ("paper_migration_" + stamp)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("--output-dir must be absent or empty; existing results are never overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)

    fixture_path = ROOT / "tests" / "fixtures" / "paper_core_reference.json"
    reference = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixture_results = check_fixtures(reference)
    inputs = args.input or sorted((ROOT / "demo_data").glob("*.csv"))
    frames, input_files, total_frames = read_demo_frames(inputs, args.frames)
    settings = reference["reference"]["default_settings"]

    if args.paper_source:
        calculator, paper_config, paper_hashes = load_paper_reference(
            args.paper_source, reference["reference"]["source_sha256_lf_normalized"], output_dir
        )
        reference_origin = str(args.paper_source.resolve())
    else:
        from evasive_acceleration._paper import calculator
        from evasive_acceleration._paper import config as paper_config
        paper_hashes = {
            name: source_hash(ROOT / "src" / "evasive_acceleration" / "_paper" / name)
            for name in reference["reference"]["source_sha256_lf_normalized"]
        }
        if paper_hashes != reference["reference"]["source_sha256_lf_normalized"]:
            raise ValueError("Vendored paper reference no longer matches its frozen hashes")
        reference_origin = "Vendored unchanged canonical calculator; static fixture independently generated"
    cfg = paper_config.EAConfig(**{SETTING_MAP[key]: value for key, value in settings.items()
                                  if key in SETTING_MAP})
    print(f"Checking {len(frames)} frames against canonical four-decimal paper outputs...", flush=True)
    parity, new_values = [], []
    for frame in frames:
        modes = core_ea.compute_ea_modes(**frame["kwargs"], **settings)
        score = core_ea.aggregate_ea_modes(modes)
        canonical = calculator.compute_frame_metrics(paper_state(frame["kwargs"]), 0.0, 0.0, cfg)
        checks = [same_value(round(modes[key], 4), canonical[paper_key])
                  for key, paper_key in zip(MODE_KEYS, PAPER_KEYS)]
        checks.append(same_value(round(score, 4), canonical["EA"]))
        parity.append({"file": frame["file"], "csv_row": frame["csv_row"],
                       "passed": all(checks), "actual_EA": score, "reference_EA": canonical["EA"]})
        new_values.append({**modes, "EA": score})

    legacy, legacy_hash = (None, None) if args.skip_legacy else load_legacy(args.legacy_ref, output_dir)
    engines = {"paper_four_modes": core_ea}
    if legacy is not None:
        engines["legacy_four_modes"] = legacy
    print("Warming both implementations on every selected frame (not timed)...", flush=True)
    warmup = {}
    for name, engine in engines.items():
        start = time.perf_counter()
        for frame in frames:
            engine.compute_ea_modes(**frame["kwargs"], **settings)
        warmup[name] = time.perf_counter() - start
    times = {name: [] for name in engines}
    measured = []
    print(f"Timing {args.repeats} repetitions; engine order alternates...", flush=True)
    for repeat in range(args.repeats):
        ordered = list(engines.items())
        if repeat % 2:
            ordered.reverse()
        for name, engine in ordered:
            for frame in frames:
                start = time.perf_counter()
                engine.compute_ea_modes(**frame["kwargs"], **settings)
                elapsed = time.perf_counter() - start
                times[name].append(elapsed)
                measured.append({"engine": name, "repeat": repeat + 1,
                                 "file": frame["file"], "csv_row": frame["csv_row"],
                                 "elapsed_ms": elapsed * 1000.0})
    changes = []
    if legacy is not None:
        for frame, new in zip(frames, new_values):
            old = legacy.compute_ea_modes(**frame["kwargs"], **settings)
            old["EA"] = legacy.aggregate_ea_modes(old)
            changes.append({"file": frame["file"], "csv_row": frame["csv_row"],
                            "old": old, "new": new,
                            "changed": [key for key in (*MODE_KEYS, "EA")
                                        if not same_value(old[key], new[key], atol=1e-8)]})
    with (output_dir / "timings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(measured[0]))
        writer.writeheader()
        writer.writerows(measured)
    report = {
        "purpose": "Implementation migration only; not a new accuracy or crash-discrimination study",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "processor": platform.processor(),
        "versions": versions(), "numba_import_available": bool(core_ea._NUMBA_OK),
        "NUMBA_DISABLE_JIT": os.environ.get("NUMBA_DISABLE_JIT"),
        "isolated_temporary_numba_cache": jit_cache_path,
        "core_file": str(Path(core_ea.__file__).resolve()), "core_sha256": sha256(core_ea.__file__),
        "core_id": getattr(core_ea, "CORE_ID", None),
        "adapter_source_sha256": {
            relative: sha256(ROOT / relative)
            for relative in ("src/core_ea.py", "src/evasive_acceleration/_paper/four_modes.py",
                             "src/single_frame.py", "src/yaw_rate.py")
        },
        "legacy_ref": None if legacy is None else args.legacy_ref,
        "legacy_source_sha256": legacy_hash,
        "paper_reference_origin": reference_origin, "paper_source_sha256_lf_normalized": paper_hashes,
        "fixture_sha256": sha256(fixture_path), "fixture_cases": fixture_results,
        "input_files": input_files, "available_frames": total_frames,
        "selected_frames": len(frames), "selection": "Evenly spaced across ordered concatenated demo rows",
        "selected_inputs_sha256": hashlib.sha256(json.dumps(frames, sort_keys=True).encode()).hexdigest(),
        "settings": settings, "yaw_rate_source": "Provided CSV yaw rates, identical in both engines",
        "parity_at_paper_output_decimals": parity,
        "warmup_excluded_seconds": warmup, "repeats": args.repeats,
        "timing": {name: time_summary(samples) for name, samples in times.items()},
        "changed_legacy_frames": sum(bool(row["changed"]) for row in changes),
        "legacy_changes": changes,
        "comparison_note": "Legacy output changes are expected from finite-horizon CV-CV and numerical fixes; they are reported, not treated as a requirement to preserve old scores. Timings are machine/input/backend specific.",
    }
    passed = all(row["passed"] for row in fixture_results) and all(row["passed"] for row in parity)
    report["migration_equivalence_passed"] = passed
    (output_dir / "report.json").write_text(
        json.dumps(clean_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Fixture parity: {sum(row['passed'] for row in fixture_results)}/{len(fixture_results)}")
    print(f"Paper frame parity: {sum(row['passed'] for row in parity)}/{len(parity)}")
    for name, samples in times.items():
        summary = time_summary(samples)
        print(f"{name}: mean {summary['mean_ms']:.3f} ms; median {summary['median_ms']:.3f} ms; P95 {summary['p95_ms']:.3f} ms")
    print(f"Report: {output_dir / 'report.json'}")
    return 0 if passed else 1


def main(argv=None):
    # Numba's per-function cache filenames can exceed Windows MAX_PATH in deep
    # research folders. Keep all three implementations' JIT caches together in
    # one short, isolated temporary directory; compilation is warm-up only.
    try:
        from numba import config as numba_config
    except Exception:
        return _run(argv)
    previous = numba_config.CACHE_DIR
    with tempfile.TemporaryDirectory(prefix="ea-migration-") as cache:
        numba_config.CACHE_DIR = cache
        try:
            return _run(argv, jit_cache_path=cache)
        finally:
            numba_config.CACHE_DIR = previous


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
