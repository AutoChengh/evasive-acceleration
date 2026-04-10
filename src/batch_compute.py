import argparse
import glob
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from core_ea import aggregate_ea_modes, compute_ea_modes
from analytical_core import ANALYTICAL_OUTPUT_COLUMNS, compute_real_time_metrics


# ============================================================================
# Paths
# ============================================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(SRC_DIR)
DEFAULT_INPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "demo_data")


# ============================================================================
# EA numerical configuration
# ============================================================================
COARSE_SECTOR_NUM = 72
LOCAL_FINE_HALF_WINDOW_DEG = 5.0
LOCAL_FINE_DIR_NUM = 101
FINE_REFINE_RATIO = 1.15

A_MAX = 100.0
TOL = 1e-3
T_TOTAL = 10.0
DT_COARSE = 0.1
DT_FINE = 0.02

ROUND_DECIMALS = 4
OUTPUT_SUFFIX = "EA"


# ============================================================================
# Required input columns
# ============================================================================
REQUIRED_COLS = [
    "Position X (m)",
    "Position Y (m)",
    "Velocity (m/s)",
    "Heading",
    "Length (m)",
    "Width (m)",
    "Yawrate",
    "2_Position X (m)",
    "2_Position Y (m)",
    "2_Velocity (m/s)",
    "2_Heading",
    "2_Length (m)",
    "2_Width (m)",
    "2_Yawrate",
]


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    By default, the script processes all eligible CSV files under demo_data/.
    Users may optionally specify a custom input directory.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Batch-compute EA and baseline analytical metrics for all eligible CSV "
            "files in a target directory."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Directory containing input CSV files. "
            f"Default: {DEFAULT_INPUT_DIR}"
        ),
    )
    return parser.parse_args()


def to_float_or_nan(value) -> float:
    """Convert a value to float; return NaN if conversion fails."""
    try:
        return float(value)
    except Exception:
        return np.nan


def round_output_value(value) -> float:
    """Round a numeric output to the configured number of decimals."""
    if pd.isna(value):
        return np.nan
    return round(float(value), ROUND_DECIMALS)


def make_output_csv_path(csv_path: str, suffix: str = OUTPUT_SUFFIX) -> str:
    """
    Construct the output CSV path by appending '_{suffix}' before the extension.

    Example:
        input.csv -> input_EA.csv
    """
    folder = os.path.dirname(csv_path)
    base_name = os.path.basename(csv_path)
    stem, ext = os.path.splitext(base_name)
    new_name = f"{stem}_{suffix}{ext}"
    return os.path.join(folder, new_name)


def validate_required_columns(df: pd.DataFrame, csv_path: str) -> None:
    """Raise an error if the required input columns are missing."""
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns in '{os.path.basename(csv_path)}': {missing}"
        )


def remove_existing_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove output columns that may already exist in the input file, so that the
    new output remains clean and unambiguous.
    """
    cols_to_drop_if_exist = [
        "EA_CVCV",
        "EA_CVCT",
        "EA_CTCV",
        "EA_CTCT",
        "EA",
        "EA_CV_Analytical",  # Remove any legacy column from older outputs.
    ] + ANALYTICAL_OUTPUT_COLUMNS

    existing = [col for col in cols_to_drop_if_exist if col in df.columns]
    if existing:
        df = df.drop(columns=existing)

    return df


def extract_frame_inputs(df: pd.DataFrame, i) -> Dict[str, float]:
    """Extract and convert the required state variables for one frame."""
    return {
        "xA": to_float_or_nan(df.at[i, "Position X (m)"]),
        "yA": to_float_or_nan(df.at[i, "Position Y (m)"]),
        "vA": to_float_or_nan(df.at[i, "Velocity (m/s)"]),
        "hA": to_float_or_nan(df.at[i, "Heading"]),
        "lA": to_float_or_nan(df.at[i, "Length (m)"]),
        "wA": to_float_or_nan(df.at[i, "Width (m)"]),
        "yawA": to_float_or_nan(df.at[i, "Yawrate"]),
        "xB": to_float_or_nan(df.at[i, "2_Position X (m)"]),
        "yB": to_float_or_nan(df.at[i, "2_Position Y (m)"]),
        "vB": to_float_or_nan(df.at[i, "2_Velocity (m/s)"]),
        "hB": to_float_or_nan(df.at[i, "2_Heading"]),
        "lB": to_float_or_nan(df.at[i, "2_Length (m)"]),
        "wB": to_float_or_nan(df.at[i, "2_Width (m)"]),
        "yawB": to_float_or_nan(df.at[i, "2_Yawrate"]),
    }


def has_missing_frame_inputs(frame_inputs: Dict[str, float]) -> bool:
    """Return True if any required frame input is NaN."""
    return any(pd.isna(v) for v in frame_inputs.values())


def process_one_csv(csv_path: str) -> Dict[str, float]:
    """
    Process one CSV file frame by frame.

    The function:
    1. reads the CSV,
    2. computes the four EA modes and final EA for each valid frame,
    3. computes the analytical metrics from analytical_core,
    4. saves a new CSV without overwriting the original file.
    """
    file_t0 = time.perf_counter()

    df = pd.read_csv(csv_path)
    validate_required_columns(df, csv_path)
    df = remove_existing_output_columns(df)

    ea_cvcv_list: List[float] = []
    ea_cvct_list: List[float] = []
    ea_ctcv_list: List[float] = []
    ea_ctct_list: List[float] = []
    ea_list: List[float] = []

    analytical_result_lists = {col: [] for col in ANALYTICAL_OUTPUT_COLUMNS}

    total_frames = len(df)
    input_complete_frames = 0
    computed_success_frames = 0
    computation_failed_frames = 0
    skipped_missing_input_frames = 0

    numerical_ea_ms_total = 0.0
    analytical_ssm_ms_total = 0.0

    for i in df.index:
        frame_inputs = extract_frame_inputs(df, i)

        if has_missing_frame_inputs(frame_inputs):
            skipped_missing_input_frames += 1

            ea_cvcv_list.append(np.nan)
            ea_cvct_list.append(np.nan)
            ea_ctcv_list.append(np.nan)
            ea_ctct_list.append(np.nan)
            ea_list.append(np.nan)

            for col in ANALYTICAL_OUTPUT_COLUMNS:
                analytical_result_lists[col].append(np.nan)
            continue

        input_complete_frames += 1

        try:
            # -----------------------------------------------------------------
            # EA computation (four modes + final EA)
            # -----------------------------------------------------------------
            t_num0 = time.perf_counter()
            mode_results = compute_ea_modes(
                xA=frame_inputs["xA"],
                yA=frame_inputs["yA"],
                vA=frame_inputs["vA"],
                hA=frame_inputs["hA"],
                xB=frame_inputs["xB"],
                yB=frame_inputs["yB"],
                vB=frame_inputs["vB"],
                hB=frame_inputs["hB"],
                lA=frame_inputs["lA"],
                wA=frame_inputs["wA"],
                lB=frame_inputs["lB"],
                wB=frame_inputs["wB"],
                yawA=frame_inputs["yawA"],
                yawB=frame_inputs["yawB"],
                coarse_sector_num=COARSE_SECTOR_NUM,
                local_fine_half_window_deg=LOCAL_FINE_HALF_WINDOW_DEG,
                local_fine_dir_num=LOCAL_FINE_DIR_NUM,
                fine_refine_ratio=FINE_REFINE_RATIO,
                a_max=A_MAX,
                tol=TOL,
                T_total=T_TOTAL,
                dt_coarse=DT_COARSE,
                dt_fine=DT_FINE,
                size_scale=1.0,
            )
            ea_value = aggregate_ea_modes(mode_results)
            t_num1 = time.perf_counter()
            numerical_ea_ms_total += (t_num1 - t_num0) * 1000.0

            ea_cvcv = mode_results.get("EA_CVCV", np.nan)
            ea_cvct = mode_results.get("EA_CVCT", np.nan)
            ea_ctcv = mode_results.get("EA_CTCV", np.nan)
            ea_ctct = mode_results.get("EA_CTCT", np.nan)

            # -----------------------------------------------------------------
            # Other analytical metrics from analytical_core
            # -----------------------------------------------------------------
            t_ana0 = time.perf_counter()
            analytical_vals = compute_real_time_metrics(
                frame_inputs["xA"],
                frame_inputs["yA"],
                frame_inputs["vA"],
                frame_inputs["hA"],
                frame_inputs["lA"],
                frame_inputs["wA"],
                frame_inputs["xB"],
                frame_inputs["yB"],
                frame_inputs["vB"],
                frame_inputs["hB"],
                frame_inputs["lB"],
                frame_inputs["wB"],
            )
            t_ana1 = time.perf_counter()
            analytical_ssm_ms_total += (t_ana1 - t_ana0) * 1000.0

            if len(analytical_vals) != len(ANALYTICAL_OUTPUT_COLUMNS):
                raise ValueError(
                    "analytical_core.compute_real_time_metrics returned "
                    f"{len(analytical_vals)} values, but "
                    f"{len(ANALYTICAL_OUTPUT_COLUMNS)} were expected."
                )

            computed_success_frames += 1

        except Exception as exc:
            computation_failed_frames += 1
            print(
                f"[WARN] Computation failed for file '{os.path.basename(csv_path)}', "
                f"row index {i}: {exc}"
            )

            ea_cvcv = np.nan
            ea_cvct = np.nan
            ea_ctcv = np.nan
            ea_ctct = np.nan
            ea_value = np.nan

            analytical_vals = [np.nan] * len(ANALYTICAL_OUTPUT_COLUMNS)

        ea_cvcv_list.append(round_output_value(ea_cvcv))
        ea_cvct_list.append(round_output_value(ea_cvct))
        ea_ctcv_list.append(round_output_value(ea_ctcv))
        ea_ctct_list.append(round_output_value(ea_ctct))
        ea_list.append(round_output_value(ea_value))

        for col, val in zip(ANALYTICAL_OUTPUT_COLUMNS, analytical_vals):
            analytical_result_lists[col].append(round_output_value(val))

    # Append outputs to the end of the table.
    df["EA_CVCV"] = ea_cvcv_list
    df["EA_CVCT"] = ea_cvct_list
    df["EA_CTCV"] = ea_ctcv_list
    df["EA_CTCT"] = ea_ctct_list
    df["EA"] = ea_list

    for col in ANALYTICAL_OUTPUT_COLUMNS:
        df[col] = analytical_result_lists[col]

    output_csv_path = make_output_csv_path(csv_path, OUTPUT_SUFFIX)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    file_t1 = time.perf_counter()
    total_ms = (file_t1 - file_t0) * 1000.0

    avg_numerical_ea_ms_per_complete_frame = (
        numerical_ea_ms_total / input_complete_frames
        if input_complete_frames > 0
        else np.nan
    )
    avg_analytical_ssm_ms_per_complete_frame = (
        analytical_ssm_ms_total / input_complete_frames
        if input_complete_frames > 0
        else np.nan
    )
    avg_total_ms_per_frame = total_ms / total_frames if total_frames > 0 else np.nan

    print(f"[OK] Processed: {csv_path}")
    print(f"     Saved: {output_csv_path}")
    print(
        f"     Total frames: {total_frames}, "
        f"input-complete frames: {input_complete_frames}, "
        f"successful frames: {computed_success_frames}, "
        f"failed frames: {computation_failed_frames}, "
        f"skipped (missing input): {skipped_missing_input_frames}"
    )
    print(
        f"     EA total time: {numerical_ea_ms_total:.3f} ms, "
        f"average per input-complete frame: "
        f"{avg_numerical_ea_ms_per_complete_frame:.3f} ms/frame"
    )
    print(
        f"     Analytical metrics total time: {analytical_ssm_ms_total:.3f} ms, "
        f"average per input-complete frame: "
        f"{avg_analytical_ssm_ms_per_complete_frame:.3f} ms/frame"
    )
    print(
        f"     Total file runtime (including I/O): {total_ms:.3f} ms, "
        f"average per frame: {avg_total_ms_per_frame:.3f} ms/frame"
    )

    return {
        "file": csv_path,
        "output_file": output_csv_path,
        "total_frames": total_frames,
        "input_complete_frames": input_complete_frames,
        "computed_success_frames": computed_success_frames,
        "computation_failed_frames": computation_failed_frames,
        "skipped_missing_input_frames": skipped_missing_input_frames,
        "numerical_ea_ms": numerical_ea_ms_total,
        "analytical_ssm_ms": analytical_ssm_ms_total,
        "total_ms": total_ms,
    }


def main() -> None:
    """
    Batch-process all eligible CSV files in the target input directory.

    If no custom directory is provided from the command line, the script uses
    demo_data/ by default.
    """
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)

    batch_t0 = time.perf_counter()

    if not os.path.isdir(input_dir):
        print(f"Input directory was not found: {input_dir}")
        return

    all_csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    csv_files = [
        f for f in all_csv_files
        if not os.path.basename(f).endswith(f"_{OUTPUT_SUFFIX}.csv")
    ]

    if not csv_files:
        print(f"No eligible input CSV files were found in: {input_dir}")
        return

    print(f"Source script directory: {SRC_DIR}")
    print(f"Project root directory: {PROJECT_ROOT_DIR}")
    print(f"Input CSV directory: {input_dir}")
    print("Discovered CSV files:")
    for f in csv_files:
        print(" -", f)

    batch_total_frames = 0
    batch_input_complete_frames = 0
    batch_computed_success_frames = 0
    batch_computation_failed_frames = 0
    batch_skipped_missing_input_frames = 0
    batch_numerical_ea_ms = 0.0
    batch_analytical_ssm_ms = 0.0

    for csv_path in csv_files:
        try:
            stat = process_one_csv(csv_path)
            batch_total_frames += stat["total_frames"]
            batch_input_complete_frames += stat["input_complete_frames"]
            batch_computed_success_frames += stat["computed_success_frames"]
            batch_computation_failed_frames += stat["computation_failed_frames"]
            batch_skipped_missing_input_frames += stat["skipped_missing_input_frames"]
            batch_numerical_ea_ms += stat["numerical_ea_ms"]
            batch_analytical_ssm_ms += stat["analytical_ssm_ms"]
        except Exception as exc:
            print(f"[ERROR] Failed to process '{csv_path}': {exc}")

    batch_t1 = time.perf_counter()
    batch_total_ms = (batch_t1 - batch_t0) * 1000.0

    avg_batch_numerical_ea_ms_per_complete_frame = (
        batch_numerical_ea_ms / batch_input_complete_frames
        if batch_input_complete_frames > 0
        else np.nan
    )
    avg_batch_analytical_ssm_ms_per_complete_frame = (
        batch_analytical_ssm_ms / batch_input_complete_frames
        if batch_input_complete_frames > 0
        else np.nan
    )
    avg_batch_total_ms_per_frame = (
        batch_total_ms / batch_total_frames if batch_total_frames > 0 else np.nan
    )

    print("\n========== Batch Summary ==========")
    print(f"Files processed: {len(csv_files)}")
    print(
        f"Total frames: {batch_total_frames}, "
        f"input-complete frames: {batch_input_complete_frames}, "
        f"successful frames: {batch_computed_success_frames}, "
        f"failed frames: {batch_computation_failed_frames}, "
        f"skipped (missing input): {batch_skipped_missing_input_frames}"
    )
    print(
        f"EA total time: {batch_numerical_ea_ms:.3f} ms, "
        f"average per input-complete frame: "
        f"{avg_batch_numerical_ea_ms_per_complete_frame:.3f} ms/frame"
    )
    print(
        f"Analytical metrics total time: {batch_analytical_ssm_ms:.3f} ms, "
        f"average per input-complete frame: "
        f"{avg_batch_analytical_ssm_ms_per_complete_frame:.3f} ms/frame"
    )
    print(
        f"Total batch runtime (including I/O): {batch_total_ms:.3f} ms, "
        f"average per frame: {avg_batch_total_ms_per_frame:.3f} ms/frame"
    )


if __name__ == "__main__":
    main()
