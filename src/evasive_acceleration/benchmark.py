"""Portable warmed EA timing; no alternative solver or yaw-rate estimator.

Import, CSV parsing/preparation, warm-up, hashing and output writing are
outside each interval. Core input checks and configuration/array construction
remain inside the timed API. All measured observations, including errors, stay.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

import core_ea
from single_frame import RoadUserState, build_compute_kwargs

MODE_KEYS = ("EA_CVCV", "EA_CVCT", "EA_CTCV", "EA_CTCT")
COLUMNS = ("Position X (m)", "Position Y (m)", "Velocity (m/s)",
           "Heading", "Length (m)", "Width (m)", "Yawrate")
STATE_KEYS = {"xA", "yA", "vA", "hA", "xB", "yB", "vB", "hB",
              "lA", "wA", "lB", "wB", "yawA", "yawB"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def _positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def _demo_cases(horizon):
    a = RoadUserState(0, 0, 10, 0, 4.5, 1.8, 0)
    examples = (
        ("head_on", RoadUserState(20, 0, 8, math.pi, 4.7, 1.9, 0)),
        ("parallel_clear", RoadUserState(20, 5, 10, 0, 4.5, 1.8, 0)),
        ("crossing", RoadUserState(15, -15, 10, math.pi / 2, 4.5, 1.8, 0)),
    )
    return [{"source_file": "builtin:" + name, "csv_row": "",
             "kwargs": build_compute_kwargs(a, b, T_total=horizon)}
            for name, b in examples]


def load_cases(paths, *, max_frames=0, horizon=10.0):
    """Parse supplied instantaneous states; never differentiate sparse rows."""
    cases, files = [], []
    required = tuple(COLUMNS) + tuple("2_" + name for name in COLUMNS)
    for path in paths:
        path = Path(path).resolve()
        before = sha256(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            missing = sorted(set(required) - set(fields))
            if missing:
                raise ValueError(f"{path.name}: missing columns: {', '.join(missing)}. "
                                 "Benchmark inputs need finite provided yaw rates.")
            if len(fields) != len(set(fields)):
                raise ValueError(f"{path.name}: duplicate CSV column names")
            count = 0
            for row_id, row in enumerate(reader, start=2):
                try:
                    if None in row:
                        raise ValueError("extra fields without column names")
                    a = RoadUserState.from_sequence([float(row[name]) for name in COLUMNS])
                    b = RoadUserState.from_sequence([float(row["2_" + name]) for name in COLUMNS])
                    kwargs = build_compute_kwargs(a, b, T_total=horizon)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"{path.name}, CSV row {row_id}: {exc}") from exc
                cases.append({"source_file": path.name, "csv_row": row_id, "kwargs": kwargs})
                count += 1
                if max_frames and len(cases) >= max_frames:
                    break
        if sha256(path) != before:
            raise ValueError(f"Input changed while reading: {path}")
        files.append({"path": str(path), "sha256": before, "selected_rows": count})
        if max_frames and len(cases) >= max_frames:
            break
    if not cases:
        raise ValueError("No input frames were found")
    return cases, files


def latency_summary(values):
    times = np.asarray(values, dtype=float)
    if times.ndim != 1 or not times.size or not np.isfinite(times).all() or np.any(times <= 0):
        raise ValueError("Latency observations must be finite, positive and non-empty")
    result = {"count": int(times.size), "mean_ms": float(times.mean()),
              "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
              "p99_ms": float(np.percentile(times, 99)), "max_ms": float(times.max()),
              "min_ms": float(times.min())}
    result.update({f"over_{limit}ms": int(np.count_nonzero(times > limit)) for limit in (15, 20, 50, 100)})
    return result


def _evaluate(kwargs):
    modes = core_ea.compute_ea_modes(**kwargs)
    return modes, core_ea.aggregate_ea_modes(modes)


def _status(modes, score):
    values = [float(modes[key]) for key in MODE_KEYS] + [float(score)]
    if any(math.isnan(value) or value < 0 for value in values):
        return "error"
    if math.isinf(score):
        return "inf"
    return "zero" if score == 0 else "finite_positive"


def _environment():
    versions = {}
    for name in ("evasive-acceleration", "numpy", "numba", "llvmlite"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    processor = platform.processor()
    try:
        if sys.platform.startswith("linux"):
            with Path("/proc/cpuinfo").open() as stream:
                processor = next((line.split(":", 1)[1].strip() for line in stream
                                  if line.startswith("model name")), processor)
        elif sys.platform == "win32":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                processor = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except OSError:
        pass
    return {"python": sys.version, "platform": platform.platform(),
            "processor": processor, "logical_cpus": os.cpu_count(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "packages": versions, "numba_import_available": core_ea._NUMBA_OK,
            "NUMBA_DISABLE_JIT": os.environ.get("NUMBA_DISABLE_JIT"),
            "numba_jit_disabled": bool(sys.modules["numba"].config.DISABLE_JIT) if core_ea._NUMBA_OK else None,
            "thread_environment": {name: os.environ.get(name) for name in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")}}


def _source_hashes():
    adapter = Path(core_ea.__file__).resolve()
    paths = {"core_ea.py": adapter, "benchmark.py": Path(__file__).resolve()}
    paths.update({"_model/" + path.name: path
                  for path in (adapter.parent / "evasive_acceleration/_model").glob("*.py")})
    result = {name: sha256(path) for name, path in sorted(paths.items())}
    provenance = core_ea.get_core_provenance()
    result.update(provenance["runtime_source_sha256"])
    result["native_library"] = provenance["native_library_sha256"]
    return result


def run(args):
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Output directory already exists; use a fresh path: " + str(output))
    paths = []
    if args.input:
        paths = [args.input]
    elif args.input_dir:
        paths = sorted(path for path in args.input_dir.iterdir()
                       if path.is_file() and path.suffix.lower() == ".csv"
                       and not path.name.lower().endswith("_ea.csv"))
        if not paths:
            raise ValueError("No source CSV files in the input directory")
    if paths:
        cases, files = load_cases(paths, max_frames=args.max_frames, horizon=args.horizon)
    else:
        cases, files = _demo_cases(args.horizon), []
        if args.max_frames:
            cases = cases[:args.max_frames]
    environment = _environment()
    provenance = core_ea.get_core_provenance()
    source_hashes = _source_hashes()
    source_before = source_hashes["core_ea.py"]
    warmups = []
    # EA has separate lazy-compiled CV/CV, turning and exact-fallback paths.
    # A prefix of straight-driving inputs cannot warm a later turning path.
    # Exercise every selected input once before any reported measurement.
    warmup_count = max(args.warmup, len(cases))
    for index in range(warmup_count):
        started = time.perf_counter_ns()
        modes, score = _evaluate(cases[index % len(cases)]["kwargs"])
        elapsed = time.perf_counter_ns() - started
        if _status(modes, score) == "error":
            raise ValueError("Warm-up returned an unavailable EA result")
        warmups.append(elapsed / 1e6)
    output.mkdir(parents=True, exist_ok=False)
    times, statuses = [], Counter()
    with (output / "benchmark_frames.csv").open("x", encoding="utf-8", newline="") as stream:
        fields = ["repeat", "selected_index", "source_file", "csv_row", "elapsed_ns", "elapsed_ms",
                  *MODE_KEYS, "EA", "status", "error"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for repeat in range(1, args.repeats + 1):
            for index, case in enumerate(cases):
                error = ""
                started = time.perf_counter_ns()
                try:
                    modes, score = _evaluate(case["kwargs"])
                except Exception as exc:
                    elapsed = time.perf_counter_ns() - started
                    modes, score = dict.fromkeys(MODE_KEYS, math.nan), math.nan
                    error = f"{type(exc).__name__}: {exc}"
                else:
                    elapsed = time.perf_counter_ns() - started
                status = _status(modes, score)
                if status == "error" and not error:
                    error = "EA result is NaN or negative"
                elapsed_ms = elapsed / 1e6
                times.append(elapsed_ms)
                statuses[status] += 1
                writer.writerow({"repeat": repeat, "selected_index": index,
                    "source_file": case["source_file"], "csv_row": case["csv_row"],
                    "elapsed_ns": elapsed, "elapsed_ms": elapsed_ms,
                    **modes, "EA": score, "status": status, "error": error})
    unchanged = (_source_hashes() == source_hashes
                 and core_ea.get_core_provenance() == provenance
                 and all(sha256(item["path"]) == item["sha256"] for item in files))
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not statuses["error"] and unchanged,
        "workload": "provided_csv_states" if files else "three_synthetic_smoke_cases",
        "unique_frames": len(cases), "repeats": args.repeats, "calls": len(times),
        "selection": "all input rows" if not args.max_frames else "first max_frames rows in sorted file order",
        "warmup_calls": warmup_count, "warmup_requested_calls": args.warmup,
        "warmup_policy": "every selected input at least once",
        "warmup_elapsed_ms_excluded": warmups,
        "settings": {key: value for key, value in cases[0]["kwargs"].items() if key not in STATE_KEYS},
        "latency_all_calls": latency_summary(times),
        "status_counts": {name: statuses[name] for name in ("zero", "finite_positive", "inf", "error")},
        "environment": environment, "core_id": provenance["core_id"], "core_provenance": provenance,
        "core_adapter_sha256": source_before, "benchmark_script_sha256": sha256(__file__),
        "source_sha256": source_hashes,
        "inputs": files, "source_and_inputs_unchanged": unchanged,
        "timing_scope": "External perf_counter_ns around four mode evaluations and aggregate mean",
        "excluded": ["imports", "CSV parsing", "CSV-to-keyword state preparation", "provided yaw-rate validation",
                     "yaw-rate estimation (not run)", "warm-up/first compilation", "hashing", "output writes"],
        "all_observations_retained": True,
        "note": "A local benchmark, not the paper's full-data result or a hard real-time guarantee."}
    (output / "benchmark_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    statistics = report["latency_all_calls"]
    print(f"Measured {len(times)} EA calls; {len(cases)} unique frames; {args.repeats} pass(es).")
    print("Mean={mean_ms:.3f} ms; median={median_ms:.3f} ms; P95={p95_ms:.3f} ms; max={max_ms:.3f} ms".format(**statistics))
    print("Saved " + str(output))
    return 0 if report["passed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ea benchmark", description="Measure warmed four-pairing EA on your machine.")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input", type=Path, help="One CSV with finite provided yaw rates")
    inputs.add_argument("--input-dir", type=Path, help="Source CSVs in one directory (not recursive)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs") / ("benchmark_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    parser.add_argument("--repeats", type=_positive_int, default=1)
    parser.add_argument("--warmup", type=_positive_int, default=3)
    parser.add_argument("--max-frames", type=_nonnegative_int, default=0, help="0 selects every input row")
    parser.add_argument("--horizon", type=_positive_float, default=core_ea.DEFAULT_T_TOTAL)
    parser.add_argument("--cpu", type=_nonnegative_int, help="Optional logical CPU affinity (supported platforms only)")
    args = parser.parse_args(argv)
    affinity = None
    try:
        if args.cpu is not None:
            if not hasattr(os, "sched_setaffinity"):
                raise ValueError("--cpu affinity is unsupported on this platform; omit it")
            affinity = os.sched_getaffinity(0)
            os.sched_setaffinity(0, {args.cpu})
        return run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ea benchmark: error: {exc}", file=sys.stderr)
        return 2
    finally:
        if affinity is not None:
            os.sched_setaffinity(0, affinity)


if __name__ == "__main__":
    raise SystemExit(main())
