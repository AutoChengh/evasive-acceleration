# EA setup and runtime

EA computes all four CV/CTRV pairings and their arithmetic mean, with
continuous-time collision verification and globally bounded numerical search.
The Python API, command line and notebook all call this algorithm.

## Prepare the native component

Install the package and build once, before computing EA:

```bash
python -m pip install -e .
python -m evasive_acceleration.native_build
ea single
```

Numba is a base dependency. The native build requires a GCC-compatible C++17
compiler (`g++`, or MinGW-w64 `g++` on Windows); it does not install one. Pass
`--compiler` to select an executable. Windows builds statically link the
MinGW runtime, including thread support, so loading does not depend on
compiler-private DLLs.

By default the library is stored in a user cache outside the checkout.
For host-specific optimization, add `--native`. To build into a custom
location, use `--output` with a fresh path and set `EA_NATIVE_CELLS_LIBRARY`
to the resulting library when running EA. See
`python -m evasive_acceleration.native_build --help` for platform options.

An EA call never launches a compiler or substitutes a different algorithm.
Missing dependencies fail explicitly. The first calls can still JIT-compile
Numba functions; benchmark only after warming the needed input paths on the
same CPU.

## Quick benchmark

```bash
ea benchmark
ea benchmark --input examples/data/provided.csv --output-dir outputs/runtime_provided
ea benchmark --input-dir /path/to/csvs --output-dir outputs/runtime_all
```

Without input arguments, the command measures three synthetic states:
head-on, separated parallel motion and crossing. This is a smoke test, not
the paper's dataset benchmark. It needs no Pandas or plotting dependencies.

The directory command reads CSVs in sorted order, non-recursively, skipping
names ending in `_EA.csv`. Each selected row must have finite `Yawrate` and
`2_Yawrate` in the [input schema](../README.md#batch-csv-schema). Invalid rows
are not silently dropped. For missing yaw rates, first use the batch or
history workflow and explicitly supply the resulting used rates under those
two names; preprocessing stays outside the EA timer.

## What is timed

Each `time.perf_counter_ns()` interval includes the complete four-mode solve
and arithmetic mean for one interaction pair. Import, CSV parsing and
CSV-to-keyword preparation, provided yaw-rate validation/estimation,
compilation, warm-up, hashing and output writes are excluded. Core input
checks and configuration/NumPy-array construction remain inside the timed
API call.

Before measurement, the command evaluates every selected input at least once.
The actual excluded warm-up count is `max(--warmup, number of selected inputs)`;
requested and actual counts and warm-up durations are recorded separately.
The default is one measured pass per frame. Longer warm-up is important for
heterogeneous input paths; compilation is not a measured EA operation.

```bash
ea benchmark --input-dir /path/to/csvs --warmup 3 --repeats 1 --max-frames 0 --output-dir outputs/runtime_full
```

`--max-frames 0` includes every row. A positive value selects the first N
rows in sorted file order, not a random or representative sample. `--horizon`
changes the default 10 s horizon and is recorded. Repeated measurements are
retained individually, never reduced to their fastest result.

On systems supporting affinity, optionally use `--cpu 20` to select a logical
CPU. On Windows omit `--cpu`; unsupported affinity requests fail explicitly.
The command records available affinity and numerical-library thread settings.
Set thread environment variables before Python starts rather than changing
them after NumPy has loaded.

## Output

- `benchmark_frames.csv`: every measured duration, source row, repeat, four
  components, mean, numerical status and error.
- `benchmark_summary.json`: counts, mean/median/P95/P99/maximum, counts above
  15/20/50/100 ms, excluded warm-up, settings, environment and source/input hashes.

Existing output directories are protected. `inf` is retained as a numerical
outcome for current contact or no feasible bounded solution, separately from
execution errors and `NaN`. Failed calls remain in both the CSV and all-call
latency statistics and cause a nonzero exit.

## Recorded full-cohort acceptance

A frozen source/build/input set evaluated all 857,820 original interaction
frames, once, serially on one logical CPU of an Intel Xeon Gold 6226R with
Python 3.12.13. One process/thread was used. The measured call included all
four CV/CTRV components and their arithmetic mean.

| Measure | Recorded result |
|---|---:|
| Frames | 857,820 |
| Mean | 0.153683 ms |
| Median | 0.068359 ms |
| P95 | 0.619070 ms |
| P99 | 1.537522 ms |
| Maximum | 14.448126 ms |
| Calls above 20 ms | 0 |
| Execution errors / NaN frames | 0 / 0 |

There were 5,993 excluded same-core warm-up calls, plus a separate untimed
diagnostic pilot, with no JIT signature growth during measurement. The native
build used host-specific optimization, strict float64, no fast-math and
disabled FMA contraction. No slow observation was removed or replaced by a
faster replay. Nine raw-input frames retained `inf` results; corrected inputs
were checked separately and did not replace primary observations.

All finite results were within 0.000949699 m/s² of an independently frozen
continuous-search reference used to check the EA computation. This check is
separate from reproducing historical grid-based research estimates, which
need not equal current outputs frame by frame. See
[historical research records](history.md).

The acceptance suite also included 1,160 expanded cases, 24,792 numerical,
continuous-time safety, global-bound and geometric-invariance checks. The
public adapter matched all 4,640 component outputs of the frozen numerical
entry point; all 4,640 independent continuous-time safety checks passed.
These are acceptance records, not a claim that a new GitHub commit has already
passed hosted tests. The [machine-readable source record](data/ea_acceptance_20261022.json)
preserves the measured results and source/build hashes for traceability.

These times measure one road-user pair, not perception, tracking, trajectory
prediction, yaw-rate estimation or scene-wide evaluation. Rebenchmark another
machine/compiler/workload; an observed maximum is not a universal hard
real-time guarantee.

## Source provenance

EA's numerical operations are validated against the source archive frozen on
2026-10-22. Current file hashes are recorded in the packaged `ea_sources.json`
manifest and run provenance; comments and private package organization need
not be byte-identical to that archive. Raw datasets and compiled libraries are
not bundled. Validation archive SHA256:

```text
37f35ab3b0b62c587cac46b501f398d6283467ccb2220e9715700600bd9c7a15
```

Measured Linux native binary SHA256:

```text
a0037dd2e748bf3af863d2e3b00e07ad1120115bfaa32f4d8f5230ee4ac119c1
```

The portable benchmark measures your installed source and build. It does not
download the full paper cohort or reconstruct the frozen experiment.
