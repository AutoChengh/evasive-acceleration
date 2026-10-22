"""Adapt a long-form tracking table to the two-actor EA workflow.

Run ``python examples/from_dataframe.py --help`` after installing ``.[batch]``.
The conversion functions can also be imported by a notebook. This module
prepares inputs for EA; it does not define a separate risk algorithm.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


REQUIRED_FIELDS = (
    "timestamp", "track_id", "x", "y", "vx", "vy", "heading", "length", "width",
)
ALLOWED_FIELDS = REQUIRED_FIELDS + ("yaw_rate",)


def _column_mapping(columns: Mapping[str, str] | None) -> dict[str, str]:
    mapping = {field: field for field in ALLOWED_FIELDS}
    if columns is not None:
        unknown = set(columns) - set(ALLOWED_FIELDS)
        if unknown:
            raise ValueError(f"Unknown canonical fields in column mapping: {sorted(unknown)}")
        mapping.update(columns)
    if any(not isinstance(name, str) or not name for name in mapping.values()):
        raise ValueError("Mapped column names must be non-empty strings")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Each canonical field must map to a different source column")
    return mapping


def _actor_rows(
    tracks: pd.DataFrame, actor, mapping: dict[str, str], *,
    time_unit: str, angle_unit: str, y_down: bool,
    max_velocity_heading_error_deg: float,
) -> pd.DataFrame:
    rows = tracks.loc[tracks[mapping["track_id"]].eq(actor)]
    if rows.empty:
        raise ValueError(
            f"Actor {actor!r} ({type(actor).__name__}) was not found. "
            "Use the same ID type as the DataFrame's track_id column."
        )
    fields = [field for field in ALLOWED_FIELDS if mapping[field] in tracks.columns]
    result = rows[[mapping[field] for field in fields]].copy()
    result.columns = fields
    for field in fields:
        if field == "track_id":
            continue
        original = result[field]
        numeric = pd.to_numeric(original, errors="coerce").astype(float)
        # An absent optional yaw rate is supported by batch auto/estimated mode.
        invalid = ~np.isfinite(numeric)
        if field == "yaw_rate":
            invalid &= ~original.isna()
        if invalid.any():
            index = original.index[np.flatnonzero(invalid.to_numpy())[0]]
            raise ValueError(
                f"Actor {actor!r}, column {mapping[field]!r}, row {index!r}: "
                "expected a finite number"
            )
        result[field] = numeric
    if time_unit == "ms":
        result["timestamp"] = result["timestamp"] / 1000.0
    times = result["timestamp"].to_numpy()
    if len(times) > 1 and (np.diff(times) <= 0).any():
        raise ValueError(
            f"Actor {actor!r}: timestamps must be strictly increasing in input order; "
            "duplicate or out-of-order observations are not sorted or merged automatically"
        )
    for field in ("length", "width"):
        if (result[field] <= 0).any():
            raise ValueError(f"Actor {actor!r}: {mapping[field]!r} must be strictly positive")
    if angle_unit == "deg":
        result["heading"] = np.deg2rad(result["heading"])
        if "yaw_rate" in result:
            result["yaw_rate"] = np.deg2rad(result["yaw_rate"])
    if y_down:
        for field in ("y", "vy", "heading", "yaw_rate"):
            if field in result:
                result[field] = -result[field]
    result["speed"] = np.hypot(result["vx"], result["vy"])
    if not np.isfinite(result["speed"]).all():
        raise ValueError(f"Actor {actor!r}: velocity components produce a non-finite speed")
    moving = result["speed"].to_numpy() > 1e-6
    course = np.arctan2(result["vy"], result["vx"]).to_numpy()
    delta = course - result["heading"].to_numpy()
    error_deg = np.rad2deg(np.abs(np.arctan2(np.sin(delta), np.cos(delta))))
    reverse = moving & (error_deg >= 90.0)
    if reverse.any():
        index = result.index[np.flatnonzero(reverse)[0]]
        raise ValueError(
            f"Actor {actor!r}, row {index!r}: velocity points sideways/backward "
            "relative to body heading; this EA adapter does not support reverse motion"
        )
    misaligned = moving & (error_deg > max_velocity_heading_error_deg)
    if misaligned.any():
        position = np.flatnonzero(misaligned)[0]
        raise ValueError(
            f"Actor {actor!r}, row {result.index[position]!r}: velocity/body-heading "
            f"difference {error_deg[position]:.6g} deg exceeds "
            f"max_velocity_heading_error_deg={max_velocity_heading_error_deg:g}; "
            "the EA scalar-speed model assumes forward motion along body heading"
        )
    return result


def to_pair_dataframe(
    tracks: pd.DataFrame, *, actor_a, actor_b,
    columns: Mapping[str, str] | None = None,
    time_unit: str = "s", angle_unit: str = "rad", y_down: bool = False,
    max_velocity_heading_error_deg: float = 5.0,
) -> pd.DataFrame:
    """Return the EA batch wide table at exactly shared timestamps.

    ``columns`` maps canonical names to source names. Positions and dimensions
    must be meters and velocity components must be m/s. Heading is body yaw,
    measured from +x toward +y in the input frame; it is never inferred from
    velocity. ``y_down=True`` reflects that frame into Cartesian y-up.
    The scalar-speed EA model supports forward motion near body heading only.
    Above 1e-6 m/s, course/body-heading disagreement must stay within
    max_velocity_heading_error_deg (default 5 degrees, always below 90).
    Reverse motion and larger sideways motion are rejected.

    Observations must already be chronological within each selected actor.
    Other actors are ignored. No sorting, interpolation or future observations
    are used to complete missing measurements. Optional yaw_rate uses the same
    angular unit as heading, per second; missing values remain missing.
    """
    if not isinstance(tracks, pd.DataFrame):
        raise TypeError("tracks must be a pandas DataFrame")
    if tracks.columns.duplicated().any():
        raise ValueError("The input DataFrame contains duplicate column names")
    if time_unit not in ("s", "ms") or angle_unit not in ("rad", "deg"):
        raise ValueError("time_unit must be 's' or 'ms'; angle_unit must be 'rad' or 'deg'")
    try:
        max_velocity_heading_error_deg = float(max_velocity_heading_error_deg)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_velocity_heading_error_deg must be a finite number in [0, 90)") from exc
    if not np.isfinite(max_velocity_heading_error_deg) or not 0 <= max_velocity_heading_error_deg < 90:
        raise ValueError("max_velocity_heading_error_deg must be a finite number in [0, 90)")
    for actor in (actor_a, actor_b):
        if not pd.api.types.is_scalar(actor) or pd.isna(actor):
            raise ValueError("Actor IDs must be non-missing scalar values")
    if actor_a == actor_b:
        raise ValueError("actor_a and actor_b must identify two different actors")
    mapping = _column_mapping(columns)
    required = [mapping[field] for field in REQUIRED_FIELDS]
    if columns is not None and "yaw_rate" in columns:
        required.append(mapping["yaw_rate"])
    missing = [column for column in required if column not in tracks.columns]
    if missing:
        raise ValueError(f"Missing source columns: {missing}; supply an explicit columns mapping")
    options = {
        "time_unit": time_unit, "angle_unit": angle_unit, "y_down": y_down,
        "max_velocity_heading_error_deg": max_velocity_heading_error_deg,
    }
    a = _actor_rows(tracks, actor_a, mapping, **options)
    b = _actor_rows(tracks, actor_b, mapping, **options)
    # Preserve the original chronological order and use only exact observations.
    a = a.loc[a["timestamp"].isin(b["timestamp"])].reset_index(drop=True)
    if a.empty:
        raise ValueError("The selected actors have no exactly shared timestamps; no interpolation is performed")
    b = b.set_index("timestamp").loc[a["timestamp"]].reset_index()
    pair = pd.DataFrame({"Time (s)": a["timestamp"]})
    for actor_rows, prefix in ((a, ""), (b, "2_")):
        for field, destination in (
            ("track_id", "Vehicle Number"), ("x", "Position X (m)"),
            ("y", "Position Y (m)"), ("speed", "Velocity (m/s)"),
            ("heading", "Heading"), ("length", "Length (m)"), ("width", "Width (m)"),
        ):
            pair[prefix + destination] = actor_rows[field].to_numpy()
        if "yaw_rate" in actor_rows:
            pair[prefix + "Yawrate"] = actor_rows["yaw_rate"].to_numpy()
    return pair


def compute_from_dataframe(
    tracks: pd.DataFrame, *, actor_a, actor_b, output_dir,
    columns: Mapping[str, str] | None = None,
    time_unit: str = "s", angle_unit: str = "rad", y_down: bool = False,
    max_velocity_heading_error_deg: float = 5.0,
    yaw_rate_source: str = "auto", overwrite: bool = False,
    skip_baselines: bool = False,
) -> dict:
    """Persist the converted input and reuse batch EA with causal yaw-rate handling.

    Returns the existing batch statistics plus ``input_path``, the saved wide
    table. ``output_file`` and ``metadata_file`` identify the results and run
    settings. ``adapter`` records the input conversions and alignment tolerance.
    All three files are preserved unless overwrite=True is explicit.
    """
    from batch_compute import process_one_csv, validate_required_columns

    pair = to_pair_dataframe(
        tracks, actor_a=actor_a, actor_b=actor_b, columns=columns,
        time_unit=time_unit, angle_unit=angle_unit, y_down=y_down,
        max_velocity_heading_error_deg=max_velocity_heading_error_deg,
    )
    folder = Path(output_dir).resolve()
    input_path = folder / "pair_input.csv"
    targets = (input_path, folder / "pair_input_EA.csv", folder / "pair_input_EA.run.json")
    validate_required_columns(pair, str(input_path), yaw_rate_source)
    existing = [target for target in targets if target.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Already exists: {existing[0]}; choose another output_dir or set overwrite=True")
    folder.mkdir(parents=True, exist_ok=True)
    pair.to_csv(input_path, index=False, mode="w" if overwrite else "x", encoding="utf-8-sig")
    result = process_one_csv(
        str(input_path), str(folder), yaw_rate_source=yaw_rate_source,
        overwrite=overwrite, skip_baselines=skip_baselines,
    )
    return {
        **result, "input_path": str(input_path),
        "adapter": {
            "time_unit": time_unit, "angle_unit": angle_unit, "y_down": y_down,
            "max_velocity_heading_error_deg": float(max_velocity_heading_error_deg),
            "moving_speed_threshold_m_s": 1e-6,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Your long-form tracking CSV.")
    parser.add_argument("--actor-a", required=True, help="Actor ID, read as a string from the CSV.")
    parser.add_argument("--actor-b", required=True, help="Actor ID, read as a string from the CSV.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--column", action="append", default=[], metavar="FIELD=SOURCE",
                        help="Repeat to map canonical field names to your CSV columns.")
    parser.add_argument("--time-unit", choices=("s", "ms"), default="s")
    parser.add_argument("--angle-unit", choices=("rad", "deg"), default="rad")
    parser.add_argument("--y-down", action="store_true")
    parser.add_argument("--max-velocity-heading-error-deg", type=float, default=5.0,
                        help="Maximum forward course/body-heading mismatch in degrees; must be in [0, 90).")
    parser.add_argument("--yaw-rate-source", choices=("auto", "provided", "estimated"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args(argv)
    try:
        columns = {}
        for item in args.column:
            if "=" not in item:
                raise ValueError("--column must have the form FIELD=SOURCE")
            field, source = item.split("=", 1)
            if field in columns:
                raise ValueError(f"Repeated mapping for {field!r}")
            columns[field] = source
        mapping = _column_mapping(columns)
        with args.input.open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
        if len(header) != len(set(header)):
            raise ValueError("The input CSV contains duplicate column names")
        # Preserve numeric-looking IDs, including leading zeros, for CLI matching.
        tracks = pd.read_csv(args.input, dtype={mapping["track_id"]: "string"})
        generated = [args.output_dir / name for name in (
            "pair_input.csv", "pair_input_EA.csv", "pair_input_EA.run.json",
        )]
        if args.input.resolve() in [path.resolve() for path in generated]:
            raise ValueError("Choose an output directory that does not contain the source CSV under a generated filename")
        result = compute_from_dataframe(
            tracks, actor_a=args.actor_a, actor_b=args.actor_b,
            output_dir=args.output_dir, columns=columns, time_unit=args.time_unit,
            angle_unit=args.angle_unit, y_down=args.y_down,
            max_velocity_heading_error_deg=args.max_velocity_heading_error_deg,
            yaw_rate_source=args.yaw_rate_source, overwrite=args.overwrite,
            skip_baselines=args.skip_baselines,
        )
    except (OSError, TypeError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Converted input: {result['input_path']}")
    problems = sum(result[key] for key in (
        "skipped_missing_input_frames", "computation_failed_frames", "baseline_failed_frames",
    ))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
