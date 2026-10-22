# Use your own tracking data

The example adapter in [`examples/from_dataframe.py`](../examples/from_dataframe.py)
converts one long-form tracking table into the two-actor table accepted by EA.
It calls the `process_one_csv` function, including its causal yaw-rate
estimator and output diagnostics. Only input formatting is adapted; the
EA algorithm, numerical settings and geometry are unchanged.

Install from the repository root:

```bash
python -m pip install -e ".[batch]"
python -m evasive_acceleration.native_build
```

## Input contract

Each row describes one actor at one observation time. Supply these fields, or
map your own names to them:

| Canonical field | Meaning | Input unit |
| --- | --- | --- |
| `timestamp` | Observation timestamp | Seconds, or milliseconds with `time_unit="ms"` |
| `track_id` | Stable actor identifier | String or number |
| `x`, `y` | Bounding-box center in a shared coordinate frame | m |
| `vx`, `vy` | Velocity components in the same frame | m/s |
| `heading` | Body orientation, measured from +x toward +y | Radians, or degrees with `angle_unit="deg"` |
| `length`, `width` | Oriented bounding-box dimensions | m, strictly positive |
| `yaw_rate` | Optional measured yaw rate | rad/s, or deg/s when `angle_unit="deg"` |

The adapter calculates speed as `hypot(vx, vy)` and preserves body heading.
EA then uses that nonnegative speed along the body-heading direction. This
input contract therefore supports **forward motion approximately aligned with
body heading**. It does not support reversing vehicles or substantial sideways
slip. The adapter rejects these cases instead of substituting velocity direction
for body heading or silently passing an incompatible state to the solver.

For speed greater than `1e-6 m/s`, the wrapped angle between `(vx, vy)` and
body heading must be at most `max_velocity_heading_error_deg`, default `5.0`.
The optional tolerance must be finite and in `[0, 90)` degrees; no setting can
enable reverse motion. At speed at or below `1e-6 m/s`, course is numerically
unreliable, so this comparison is skipped and the supplied body heading is
retained. All state fields still require valid finite values.

The default frame has x to the right and y up, with counterclockwise heading
and yaw rate positive. If the source uses y down, use `y_down=True` or
`--y-down`: the adapter negates y, vy, heading, and optional yaw rate together.
This assumes source angles use that same source frame, so positive angles turn
from +x toward its downward +y. Other frame definitions need an explicit
conversion before using this example. Positions must be box centers; a rear
axle or another reference point needs its own conversion first.

Only the two selected actors are validated and used. Within each actor, rows
must already have strictly increasing timestamps. Interleaved actors are fine.
Duplicate timestamps, out-of-order observations, nonnumeric/nonfinite required
values, and nonpositive box dimensions produce a descriptive error. The adapter
does not sort or repair them. Missing optional yaw rates may be empty/NaN for
`auto` processing; other nonfinite or invalid yaw-rate values are rejected.

## Run the included custom-column CSV

[`examples/data/custom_tracks.csv`](../examples/data/custom_tracks.csv) is a
small synthetic approaching pair. It deliberately uses custom column names,
millisecond timestamps, degree headings, string IDs, and no yaw-rate column.
Run this command from the repository root; the single-line form works in
PowerShell and common Unix shells:

```bash
python examples/from_dataframe.py --input examples/data/custom_tracks.csv --actor-a ego --actor-b other --time-unit ms --angle-unit deg --column timestamp=time_ms --column track_id=object_key --column x=east_m --column y=north_m --column vx=vx_mps --column vy=vy_mps --column heading=yaw_deg --column length=box_length_m --column width=box_width_m --output-dir outputs/custom_tracks
```

This writes:

```text
outputs/custom_tracks/
    pair_input.csv              # Converted wide table, seconds and radians
    pair_input_EA.csv           # EA, optional baselines and per-frame diagnostics
    pair_input_EA.run.json      # EA/yaw-rate settings and computation counts
```

There are five shared 10 Hz observations in this example. Both actors travel
straight. With the default `auto` policy and no supplied yaw rate, the first
frame has `estimated_initialization` and a zero placeholder; the remaining
frames have initialized zero yaw-rate estimates. This demonstrates the input
interface and is not a paper benchmark.

The command refuses to replace existing generated files. Select a new output
directory, or pass `--overwrite` to rerun intentionally. The CLI protects the
source CSV even with `--overwrite`. It returns a nonzero exit code for invalid
input or incomplete EA/baseline results. Use `--skip-baselines` if only EA is
needed.

Use `--max-velocity-heading-error-deg 5` to specify the alignment tolerance
explicitly. Raising it admits more forward sideways motion; use a value
justified by your data and the scalar-speed assumption. It cannot be raised to
90 degrees or beyond. Reversing and perpendicular/backward motion remain
unsupported.

CLI IDs are read as strings, including numeric-looking IDs and leading zeros.
For a CSV containing IDs `001` and `002`, use `--actor-a 001 --actor-b 002`.
The Python API instead expects IDs of the same type as the DataFrame column;
use `actor_a=1`, not `"1"`, when the column contains integers.

## Call from a DataFrame or notebook

With the notebook's working directory at the repository root:

```python
import pandas as pd
from examples.from_dataframe import compute_from_dataframe, to_pair_dataframe

tracks = pd.read_csv(
    "examples/data/custom_tracks.csv",
    dtype={"object_key": "string"},
)
columns = {
    "timestamp": "time_ms",
    "track_id": "object_key",
    "x": "east_m",
    "y": "north_m",
    "vx": "vx_mps",
    "vy": "vy_mps",
    "heading": "yaw_deg",
    "length": "box_length_m",
    "width": "box_width_m",
}

# Conversion only: inspect this table before computing EA.
pair = to_pair_dataframe(
    tracks,
    actor_a="ego",
    actor_b="other",
    columns=columns,
    time_unit="ms",
    angle_unit="deg",
)
print(pair.head())

result = compute_from_dataframe(
    tracks,
    actor_a="ego",
    actor_b="other",
    columns=columns,
    time_unit="ms",
    angle_unit="deg",
    output_dir="outputs/custom_tracks_notebook",
)
ea = pd.read_csv(result["output_file"])
print(ea[["Time (s)", "EA", "EA_Yawrate_A_source", "EA_status"]])
```

`columns` always maps **canonical name to your source column name**. Unspecified
fields retain their canonical names. With a table already using the canonical
names, omit `columns`. Add `"yaw_rate": "your_rate_column"` to use an optional
measured rate; the same angular conversion and y-axis reflection apply to it.

`compute_from_dataframe` returns the existing batch statistics, including
`output_file`, `metadata_file`, `total_frames`, and failure counts, plus
`input_path` for the saved converted input and an `adapter` dictionary containing
the input time/angle units, y-axis conversion, alignment tolerance and moving
speed threshold. This adapter provenance is returned to the caller; it is not
added to the batch `.run.json`. Save it with your experiment configuration when
changing the defaults. Both Python functions accept
`max_velocity_heading_error_deg=5.0`. The returned paths can be passed
to the repository's comparison/plotting examples. A zero EA is a valid result;
inspect `EA_status` and yaw-rate source/initialization columns before interpreting
a missing result or an initialization placeholder.

The adapter is an example module distributed with the source checkout, not
part of the installed public package. From an `examples/` working directory,
use `from from_dataframe import ...`. It can also be copied beside your own
notebook once the EA package and batch dependencies are installed.

## Time alignment and causality

Actors may have different sampling rates. Only timestamps present **exactly**
in both actors are retained, in their existing chronological order. There is
no nearest-time match, interpolation, forward fill, backward fill, or use of
a later observation to fill an earlier one. If there is no exact overlap, the
adapter fails with an explicit message. Floating-point timestamp differences
are not silently rounded; synchronize the acquisition clocks upstream.

For example, times A=`[0.0, 0.1, 0.2, 0.3]` and B=`[0.1, 0.2, 0.4]` produce
two output rows at 0.1 and 0.2 seconds. Yaw rate is estimated by the existing
batch estimator **on the retained common observations**. Earlier/unmatched
observations are not carried into its state, so the first retained observation
is its initialization frame and fewer common frames mean less heading history.
If richer per-actor history is important, estimate yaw rate causally on those
full histories first, supply it as `yaw_rate`, and use `provided` or `auto`.

Use `yaw_rate_source="estimated"` to estimate all rates from retained heading
history; `"provided"` to require the measured-rate column; or `"auto"` to retain
finite measured values and estimate missing rates. The exact estimator and
its defaults are documented in [causal_yaw_rate.md](causal_yaw_rate.md).
The conversion never changes the original DataFrame.

This example handles one explicit actor pair. It does not choose dangerous
actors, scan all pairs, associate IDs across logs, or adapt an entire dataset.
