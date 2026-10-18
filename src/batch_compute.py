"""CSV entry point for EA with explicit yaw-rate provenance and run summaries.

The numerical solver lives in core_ea.py. This module only validates inputs,
selects yaw-rate inputs, calls that solver, and writes auditable results.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from baseline_risk_metrics import ANALYTICAL_OUTPUT_COLUMNS, compute_real_time_metrics
from core_ea import (
    DEFAULT_A_MAX, DEFAULT_COARSE_SECTOR_NUM, DEFAULT_DT_COARSE,
    DEFAULT_DT_FINE, DEFAULT_FINE_REFINE_RATIO, DEFAULT_LOCAL_FINE_DIR_NUM,
    DEFAULT_LOCAL_FINE_HALF_WINDOW_DEG, DEFAULT_TOL, DEFAULT_T_TOTAL,
    aggregate_ea_modes, compute_ea_modes,
)
from yaw_rate import (
    CausalYawRateEWMA, DEFAULT_TAU_S, DEFAULT_DEADBAND_RAD_S, DEFAULT_DECIMALS,
)

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT_DIR / "demo_data"
# Write next to the caller's working directory, including wheel installations.
DEFAULT_OUTPUT_DIR = Path("outputs") / "batch_results"
DEFAULT_TIME_COLUMN = "Time (s)"
DEFAULT_YAW_RATE_TAU_S = DEFAULT_TAU_S
DEFAULT_YAW_RATE_DEADBAND_RAD_S = DEFAULT_DEADBAND_RAD_S
DEFAULT_YAW_RATE_DECIMALS = DEFAULT_DECIMALS
YAW_RATE_SOURCES = ("provided", "estimated", "auto")
YAW_RATE_COLUMNS = ("Yawrate", "2_Yawrate")
OUTPUT_SUFFIX = "EA"
ROUND_DECIMALS = 4
EA_COLUMNS = ("EA_CTCT", "EA_CTCV", "EA_CVCT", "EA_CVCV", "EA")

# Use exactly the same defaults as the legacy core and single-frame API.
SOLVER_DEFAULTS = {
    "coarse_sector_num": DEFAULT_COARSE_SECTOR_NUM,
    "local_fine_half_window_deg": DEFAULT_LOCAL_FINE_HALF_WINDOW_DEG,
    "local_fine_dir_num": DEFAULT_LOCAL_FINE_DIR_NUM,
    "fine_refine_ratio": DEFAULT_FINE_REFINE_RATIO,
    "a_max": DEFAULT_A_MAX,
    "tol": DEFAULT_TOL,
    "T_total": DEFAULT_T_TOTAL,
    "dt_coarse": DEFAULT_DT_COARSE,
    "dt_fine": DEFAULT_DT_FINE,
    "size_scale": 1.0,
}
STATE_COLUMNS = {
    "xA": "Position X (m)", "yA": "Position Y (m)",
    "vA": "Velocity (m/s)", "hA": "Heading",
    "lA": "Length (m)", "wA": "Width (m)",
    "xB": "2_Position X (m)", "yB": "2_Position Y (m)",
    "vB": "2_Velocity (m/s)", "hB": "2_Heading",
    "lB": "2_Length (m)", "wB": "2_Width (m)",
}
BASE_REQUIRED_COLS = list(STATE_COLUMNS.values())
REQUIRED_COLS = BASE_REQUIRED_COLS + list(YAW_RATE_COLUMNS)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute EA from one CSV or a directory. Missing yaw rate is estimated causally.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inputs = parser.add_argument_group("Input")
    location = inputs.add_mutually_exclusive_group()
    location.add_argument("--input", type=Path, help="One input CSV.")
    location.add_argument("--input-dir", type=Path, help="CSV directory; omitted input uses demo_data/.")
    inputs.add_argument("--pattern", default="*.csv", help="Pattern used with --input-dir.")
    yaw = parser.add_argument_group("Yaw rate")
    yaw.add_argument("--yaw-rate-source", choices=YAW_RATE_SOURCES, default="auto",
                     help="Keep finite provided rates, estimate missing values, or force one source.")
    yaw.add_argument("--time-column", default=DEFAULT_TIME_COLUMN,
                     help="Seconds; required only when a yaw rate needs estimation.")
    yaw.add_argument("--yaw-rate-tau-s", type=float, default=DEFAULT_TAU_S,
                     help="Causal EWMA time constant in seconds.")
    yaw.add_argument("--yaw-rate-deadband-rad-s", type=float, default=DEFAULT_DEADBAND_RAD_S,
                     help="Estimated-output deadband in rad/s.")
    yaw.add_argument("--yaw-rate-decimals", type=int, default=DEFAULT_DECIMALS,
                     help="Decimal places for estimated yaw rates.")
    outputs = parser.add_argument_group("Output")
    outputs.add_argument("--output-dir", type=Path,
                         help="Explicit directory; default: outputs/batch_results/<yaw-rate-source>/.")
    outputs.add_argument("--overwrite", action="store_true",
                         help="Allow replacement of existing results; input data are always protected.")
    outputs.add_argument("--skip-baselines", action="store_true",
                         help="Compute EA without the optional comparison metrics.")
    return parser.parse_args(argv)


def to_float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def round_output_value(value) -> float:
    return round(float(value), ROUND_DECIMALS)


def _provided_values(df: pd.DataFrame, column: str) -> List[float]:
    if column not in df.columns:
        return [float("nan")] * len(df)
    return [to_float_or_nan(value) for value in df[column]]


def _needs_estimation(df: pd.DataFrame, source: str, column: str) -> bool:
    return source == "estimated" or (
        source == "auto" and any(not math.isfinite(v) for v in _provided_values(df, column))
    )


def validate_required_columns(
    df: pd.DataFrame, csv_path: str, yaw_rate_source: str = "auto",
    time_column: str = DEFAULT_TIME_COLUMN,
) -> None:
    if yaw_rate_source not in YAW_RATE_SOURCES:
        raise ValueError(f"Unsupported yaw_rate_source: {yaw_rate_source!r}")
    required = list(BASE_REQUIRED_COLS)
    if yaw_rate_source == "provided":
        required.extend(YAW_RATE_COLUMNS)
    elif any(_needs_estimation(df, yaw_rate_source, c) for c in YAW_RATE_COLUMNS):
        required.append(time_column)
    missing = [column for column in required if column not in df.columns]
    if missing:
        hint = (
            f" To estimate missing yaw rate, use --yaw-rate-source auto and supply "
            f"timestamp column {time_column!r} in seconds."
        )
        raise KeyError(f"{Path(csv_path).name}: missing columns {missing}.{hint}")


def make_output_csv_path(csv_path: str, output_dir: str, suffix: str = OUTPUT_SUFFIX) -> str:
    return str(Path(output_dir) / f"{Path(csv_path).stem}_{suffix}.csv")


def _normalise_track_id(value):
    return None if pd.isna(value) else value


def _estimate_yaw_rate_stream(
    df: pd.DataFrame, *, heading_column: str, time_column: str,
    track_id_column: str, tau_s: float, deadband_rad_s: float, decimals: int,
) -> Dict[str, List]:
    """Use original row order, reset on ID changes, and never backfill missing frames."""
    estimator = CausalYawRateEWMA(tau_s, deadband_rad_s, decimals)
    previous_id = None
    have_id = False
    result = {"values": [], "initialized": [], "samples": [], "errors": []}
    for row_number, (_, row) in enumerate(df.iterrows(), start=2):
        if track_id_column in df.columns:
            track_id = _normalise_track_id(row[track_id_column])
            if have_id and track_id != previous_id:
                estimator.reset()
            previous_id, have_id = track_id, True
        try:
            value = estimator.update(
                to_float_or_nan(row[time_column]),
                to_float_or_nan(row[heading_column]),
            )
            estimate = estimator.last_estimate
            initialized = estimate.initialized
            error = ""
        except (TypeError, ValueError, OverflowError) as exc:
            value, initialized = float("nan"), False
            error = (
                f"CSV row {row_number}, {time_column}={row[time_column]!r}, "
                f"{heading_column}={row[heading_column]!r}: {exc}"
            )
        result["values"].append(value)
        result["initialized"].append(initialized)
        result["samples"].append(estimator.sample_count)
        result["errors"].append(error)
    return result


def prepare_yaw_rate_inputs(
    df: pd.DataFrame, *, yaw_rate_source: str, time_column: str,
    tau_s: float, deadband_rad_s: float, decimals: int,
) -> Dict[str, List]:
    """Keep provided values unchanged; estimate a stream only if the policy needs it."""
    # Validate configuration even when every supplied yaw rate is already finite.
    CausalYawRateEWMA(tau_s, deadband_rad_s, decimals)
    if yaw_rate_source not in YAW_RATE_SOURCES:
        raise ValueError(f"Unsupported yaw_rate_source: {yaw_rate_source!r}")
    result = {}
    for label, column, heading, track_id in (
        ("a", "Yawrate", "Heading", "Vehicle Number"),
        ("b", "2_Yawrate", "2_Heading", "2_Vehicle Number"),
    ):
        provided = _provided_values(df, column)
        estimated = None
        if _needs_estimation(df, yaw_rate_source, column):
            estimated = _estimate_yaw_rate_stream(
                df, heading_column=heading, time_column=time_column,
                track_id_column=track_id, tau_s=tau_s,
                deadband_rad_s=deadband_rad_s, decimals=decimals,
            )
        values, sources, ready, samples, errors = [], [], [], [], []
        for i, supplied in enumerate(provided):
            use_supplied = yaw_rate_source == "provided" or (
                yaw_rate_source == "auto" and math.isfinite(supplied)
            )
            if use_supplied:
                valid = math.isfinite(supplied)
                values.append(supplied)
                sources.append("provided" if valid else "missing_provided")
                ready.append(valid)
                samples.append(0)
                errors.append("" if valid else f"Invalid {column}: {df.iloc[i].get(column)!r}")
            else:
                value = estimated["values"][i]
                initialized = estimated["initialized"][i]
                values.append(value)
                sources.append(
                    "estimation_error" if not math.isfinite(value) else
                    "estimated" if initialized else "estimated_initialization"
                )
                ready.append(initialized)
                samples.append(estimated["samples"][i])
                errors.append(estimated["errors"][i])
        for key, items in (
            ("yaw", values), ("source", sources), ("initialized", ready),
            ("samples", samples), ("error", errors),
        ):
            result[f"{key}_{label}"] = items
    return result


def extract_frame_inputs(df: pd.DataFrame, i, yaw_a: float, yaw_b: float) -> Dict[str, float]:
    values = {key: to_float_or_nan(df.at[i, column]) for key, column in STATE_COLUMNS.items()}
    values.update(yawA=float(yaw_a), yawB=float(yaw_b))
    return values


def has_missing_frame_inputs(frame_inputs: Dict[str, float]) -> bool:
    return any(not math.isfinite(value) for value in frame_inputs.values())


def _frame_input_error(frame: Dict[str, float]) -> str:
    errors = []
    for key, value in frame.items():
        if not math.isfinite(value):
            errors.append(f"{STATE_COLUMNS.get(key, key)}={value!r} must be finite")
        elif key in ("lA", "wA", "lB", "wB") and value <= 0:
            errors.append(f"{STATE_COLUMNS[key]}={value!r} must be positive")
        elif key in ("vA", "vB") and value < 0:
            errors.append(f"{STATE_COLUMNS[key]}={value!r} is a speed magnitude and must be non-negative")
    return "; ".join(errors)


def _compute_baselines(frame: Dict[str, float]) -> List[float]:
    order = ("xA", "yA", "vA", "hA", "lA", "wA", "xB", "yB", "vB", "hB", "lB", "wB")
    values = compute_real_time_metrics(*(frame[key] for key in order))
    if len(values) != len(ANALYTICAL_OUTPUT_COLUMNS):
        raise ValueError("baseline output length does not match its column definitions")
    return [round_output_value(value) for value in values]


def _software_version() -> str:
    try:
        return version("evasive-acceleration")
    except PackageNotFoundError:
        return "uninstalled source checkout"


def _read_input_csv(path: Path) -> pd.DataFrame:
    """Validate the original header before pandas can rename duplicate fields."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    columns = [name.strip() for name in header]
    if not columns:
        raise ValueError(f"{path.name}: the CSV is empty")
    duplicates = [name for name, count in Counter(columns).items() if count > 1]
    if duplicates:
        raise ValueError(f"{path.name}: duplicate column names {duplicates}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path.name}: the CSV has column headers but no interaction rows")
    if len(df.columns) != len(columns):
        raise ValueError(f"{path.name}: the header and parsed column counts differ")
    # Normalize harmless surrounding whitespace, without guessing units.
    df.columns = columns
    return df


def process_one_csv(
    csv_path: str, output_dir: Optional[str] = None, *, yaw_rate_source: str = "auto",
    time_column: str = DEFAULT_TIME_COLUMN,
    yaw_rate_tau_s: float = DEFAULT_TAU_S,
    yaw_rate_deadband_rad_s: float = DEFAULT_DEADBAND_RAD_S,
    yaw_rate_decimals: int = DEFAULT_DECIMALS,
    overwrite: bool = False, skip_baselines: bool = False,
) -> dict:
    """Read, validate, prepare yaw rate, compute EA, and save CSV plus run metadata.

    One file represents a continuous pair, in chronological order. Callers that
    need the old input contract should explicitly use yaw_rate_source='provided'.
    """
    started = time.perf_counter()
    input_path = Path(csv_path).resolve()
    output_folder = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR / yaw_rate_source
    output_path = Path(make_output_csv_path(str(input_path), str(output_folder))).resolve()
    metadata_path = output_path.with_suffix(".run.json")
    if input_path in (output_path, metadata_path):
        raise ValueError("The output path would overwrite the input CSV; choose a different output directory.")
    existing = [p for p in (output_path, metadata_path) if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Result already exists: {existing[0]}. Choose --output-dir or explicitly use --overwrite."
        )

    df = _read_input_csv(input_path)
    validate_required_columns(df, str(input_path), yaw_rate_source, time_column)
    yaw = prepare_yaw_rate_inputs(
        df, yaw_rate_source=yaw_rate_source, time_column=time_column,
        tau_s=yaw_rate_tau_s, deadband_rad_s=yaw_rate_deadband_rad_s,
        decimals=yaw_rate_decimals,
    )
    # Keep original input fields, replacing only columns owned by this tool.
    owned = set(EA_COLUMNS) | set(ANALYTICAL_OUTPUT_COLUMNS) | {"EA_CV_Analytical"}
    owned.update(column for column in df.columns if column.startswith("EA_Yawrate_"))
    owned.update(("EA_status", "EA_error", "EA_baseline_error"))
    df = df.drop(columns=[column for column in owned if column in df.columns])

    ea_rows = {column: [] for column in EA_COLUMNS}
    baseline_rows = {column: [] for column in ANALYTICAL_OUTPUT_COLUMNS}
    statuses, errors, baseline_errors = [], [], []
    ea_ms, baseline_ms = 0.0, 0.0
    complete, succeeded, failed, skipped, baseline_failed = 0, 0, 0, 0, 0

    for position, i in enumerate(df.index):
        frame = extract_frame_inputs(df, i, yaw["yaw_a"][position], yaw["yaw_b"][position])
        modes = {column: float("nan") for column in EA_COLUMNS}
        baseline = [float("nan")] * len(ANALYTICAL_OUTPUT_COLUMNS)
        error = "; ".join(filter(None, (
            yaw["error_a"][position], yaw["error_b"][position], _frame_input_error(frame),
        )))
        baseline_error = ""
        if error:
            status = "input_error"
            skipped += 1
        else:
            complete += 1
            tick = time.perf_counter()
            try:
                results = compute_ea_modes(**frame, **SOLVER_DEFAULTS)
                value = aggregate_ea_modes(results)
                if math.isnan(value):
                    raise ValueError("EA solver returned NaN for a complete input frame")
                modes.update(results)
                modes["EA"] = value
                succeeded += 1
                status = "ok"
            except Exception as exc:
                failed += 1
                status, error = "solver_error", str(exc)
            finally:
                ea_ms += (time.perf_counter() - tick) * 1000.0

            if not skip_baselines:
                tick = time.perf_counter()
                try:
                    baseline = _compute_baselines(frame)
                except Exception as exc:
                    baseline_failed += 1
                    baseline_error = str(exc)
                    if status == "ok":
                        status = "ok_baseline_error"
                finally:
                    baseline_ms += (time.perf_counter() - tick) * 1000.0

        for column in EA_COLUMNS:
            ea_rows[column].append(round_output_value(modes[column]))
        for column, value in zip(ANALYTICAL_OUTPUT_COLUMNS, baseline):
            baseline_rows[column].append(value)
        statuses.append(status)
        errors.append(error)
        baseline_errors.append(baseline_error)

    for column, values in ea_rows.items():
        df[column] = values
    for label in ("a", "b"):
        for suffix, key in (
            ("used_rad_s", "yaw"), ("source", "source"), ("initialized", "initialized"),
            ("samples", "samples"), ("error", "error"),
        ):
            df[f"EA_Yawrate_{label.upper()}_{suffix}"] = yaw[f"{key}_{label}"]
    df["EA_status"], df["EA_error"], df["EA_baseline_error"] = statuses, errors, baseline_errors
    if not skip_baselines:
        for column, values in baseline_rows.items():
            df[column] = values

    total_ms = (time.perf_counter() - started) * 1000.0
    stat = {
        "file": str(input_path), "output_file": str(output_path),
        "metadata_file": str(metadata_path), "total_frames": len(df),
        "input_complete_frames": complete, "computed_success_frames": succeeded,
        "computation_failed_frames": failed, "skipped_missing_input_frames": skipped,
        "baseline_failed_frames": baseline_failed,
        "numerical_ea_ms": ea_ms, "analytical_ssm_ms": baseline_ms, "total_ms": total_ms,
    }
    metadata = {
        "schema_version": 1, "software_version": _software_version(),
        "requested_yaw_rate_source": yaw_rate_source,
        "time_column": time_column,
        "yaw_rate": {"tau_s": yaw_rate_tau_s, "deadband_rad_s": yaw_rate_deadband_rad_s,
                     "decimals": yaw_rate_decimals},
        "solver": dict(SOLVER_DEFAULTS), "baselines_enabled": not skip_baselines,
        "yaw_rate_source_counts": {
            label.upper(): dict(Counter(yaw[f"source_{label}"])) for label in ("a", "b")
        },
        "status_counts": dict(Counter(statuses)), "statistics": stat,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a concurrent run from silently replacing results.
    mode = "w" if overwrite else "x"
    with output_path.open(mode, encoding="utf-8-sig", newline="") as stream:
        df.to_csv(stream, index=False)
    with metadata_path.open(mode, encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")

    problem_count = skipped + failed + baseline_failed
    print(f"[{'WARN' if problem_count else 'OK'}] {input_path.name}: "
          f"{succeeded}/{len(df)} EA frames, {skipped} invalid, "
          f"{failed} solver errors, {baseline_failed} baseline errors")
    print(f"Saved: {output_path}")
    print(f"Run settings: {metadata_path}")
    shown_errors = 0
    for row_number, (error, baseline_error) in enumerate(zip(errors, baseline_errors), start=2):
        if error or baseline_error:
            print(f"  CSV row {row_number}: {error or baseline_error}", file=sys.stderr)
            shown_errors += 1
            if shown_errors >= 3:
                break
    return stat


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.input is not None:
        files = [args.input]
    else:
        input_dir = args.input_dir if args.input_dir is not None else DEFAULT_INPUT_DIR
        if not input_dir.is_dir():
            print(f"[ERROR] Input directory not found: {input_dir}. "
                  "Use --input case.csv or --input-dir path/to/csvs.", file=sys.stderr)
            return 2
        files = sorted(p for p in input_dir.glob(args.pattern)
                       if p.is_file() and not p.name.endswith(f"_{OUTPUT_SUFFIX}.csv"))
        if not files:
            print(f"[ERROR] No eligible CSV files in {input_dir}; check --pattern.", file=sys.stderr)
            return 2
    succeeded_files, failed_files, incomplete_files = 0, 0, 0
    for path in files:
        try:
            stat = process_one_csv(
                str(path), str(args.output_dir) if args.output_dir is not None else None,
                yaw_rate_source=args.yaw_rate_source, time_column=args.time_column,
                yaw_rate_tau_s=args.yaw_rate_tau_s,
                yaw_rate_deadband_rad_s=args.yaw_rate_deadband_rad_s,
                yaw_rate_decimals=args.yaw_rate_decimals,
                overwrite=args.overwrite, skip_baselines=args.skip_baselines,
            )
            if (stat["skipped_missing_input_frames"] or stat["computation_failed_frames"]
                    or stat["baseline_failed_frames"]):
                incomplete_files += 1
            else:
                succeeded_files += 1
        except (OSError, ValueError, KeyError) as exc:
            failed_files += 1
            print(f"[ERROR] {path}: {exc}", file=sys.stderr)
    print(f"Files discovered: {len(files)}; successful: {succeeded_files}; "
          f"incomplete: {incomplete_files}; failed: {failed_files}")
    return 1 if failed_files or incomplete_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
