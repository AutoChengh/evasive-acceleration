# Measure EA runtime

Use `ea benchmark` to check warmed EA computation on your machine. It uses
the existing four-pairing core, not a second or simplified solver.

The default remains the paper-compatible solver. For the opt-in v53 solver,
prepare its native component and pass `--backend v53`; see the
[v53 setup and full-cohort acceptance record](runtime_v53.md). The historical
table below measures the earlier paper-compatible implementation, not v53.

## Quick check

After installing the package:

```bash
ea benchmark
```

This measures three explicitly synthetic states: head-on, separated parallel
motion, and crossing. It is a smoke test, not the paper's dataset benchmark.
No Pandas, plotting dependency, or source-checkout data file is required.
The default output is a fresh timestamped directory under `outputs/`.

For a relevant workload, supply finite instantaneous states in the
[provided-yaw-rate CSV schema](../README.md#batch-csv-schema):

```bash
ea benchmark --input examples/data/provided.csv --output-dir outputs/runtime_provided
ea benchmark --input-dir /path/to/csvs --output-dir outputs/runtime_all
```

The directory command reads source CSVs in sorted order, non-recursively,
skipping names ending in `_EA.csv`. An explicit `--input` may select any
valid CSV. Every selected row must have finite `Yawrate` and `2_Yawrate`;
the benchmark does not estimate them or silently discard invalid rows.

For missing yaw rates, use the batch or Python history workflow first and
explicitly supply the resulting used yaw rates under those two input column
names. This keeps preprocessing outside the measured EA interval.

## What is timed

Each interval uses `time.perf_counter_ns()` around the four CV/CTRV mode
evaluations and their aggregate mean. Import, CSV parsing, state conversion,
yaw-rate validation/estimation, warm-up, hashing, and output writes are excluded.
For the default `paper` backend, the defaults are three excluded warm-up calls
and one measured pass per frame. With `--backend v53`, the excluded warm-up
count is `max(--warmup, number of selected inputs)`: every selected input is
evaluated at least once before measurement, including turning and fallback
paths. Both backends default to one measured pass per frame. The JSON records
the requested and actual warm-up counts separately.

```bash
ea benchmark --input-dir /path/to/csvs --warmup 3 --repeats 1 --max-frames 0 --output-dir outputs/runtime_full
```

`--max-frames 0` includes every input row. A positive value selects the first
N rows in sorted file order, not a random or representative sample. `--horizon`
can change the default 10 s prediction horizon; the actual value is recorded.
Use `--repeats` only when you intentionally want repeated measurements. All
repeats are retained, not reduced to their fastest result.

On systems supporting process affinity, optionally select a logical CPU:

```bash
ea benchmark --input-dir /path/to/csvs --cpu 20 --output-dir outputs/runtime_cpu20
```

On Windows, omit `--cpu`; unsupported affinity requests fail explicitly. The
command records affinity where available and existing numerical-library thread
settings. It does not silently change thread settings after NumPy has loaded.
To control threads, set the relevant environment variables before Python starts.

Install the `fast` extra for Numba acceleration. Use separate environments for
accelerated and NumPy-only comparisons. First compilation can be much slower
than warmed calls; excluded warm-up durations are retained in the report.

## Output and interpretation

- `benchmark_frames.csv`: every measured duration in ns and ms, source CSV row,
  repeat number, four EA components, aggregate, numerical status, and error.
- `benchmark_summary.json`: sample count, mean/median/P95/P99/maximum, counts
  above 15/20/50/100 ms, excluded warm-up durations, settings, environment, and
  source/input hashes.

An existing output directory is never overwritten. Use a fresh path for a new
run. `inf` is a valid core outcome for current contact or an unsuccessful bounded
search; it is counted separately from execution errors and `NaN`. Failed calls
remain in both the CSV and all-call latency summary, and cause a nonzero exit.

These are times for one road-user pair per frame. They exclude perception,
tracking, and scene-wide evaluation of many pairs. Do not interpret one run's
maximum as a universal hard real-time bound or expect another machine to
reproduce an identical number of milliseconds.

## Large-cohort implementation check on 2026-10-20

A separate frozen single-pass experiment evaluated 857,820 interaction frames
from 4,418 EA-P90 scenes. Each implementation evaluated every frame once,
serially on logical CPU 20 of an Intel Xeon Gold 6226R (nominal 2.90 GHz),
with balanced order across 200-frame blocks. Numerical-library thread counts
were one. The environment was Linux, Python 3.12.13, NumPy 2.3.5, and
Numba 0.62.1. The core identity was
`paper-aligned-verified-cv-ctrv-20261020` at source commit `7d6535d`.

| Implementation | Mean ms | Median ms | P95 ms | P99 ms | Observed max ms |
|---|---:|---:|---:|---:|---:|
| Frozen reference in the same environment | 4.945 | 2.645 | 16.992 | 19.876 | 28.189 |
| Current verified core | 2.424 | 0.408 | 12.572 | 15.303 | 567.985 |

All original timing observations, including the rare long tail, are retained.
The verified run had seven observations above 20 ms. Later input corrections
and slow-frame replays were analysed separately, not substituted into this table.
The aggregate record and source fingerprints are available in
[the experiment summary](data/runtime_20261020_summary.json); full paper input
trajectories are not bundled in this repository.

A separate one-pass diagnostic replay covered all 10,291 frames previously
observed above 15 ms. Its mean was 15.589 ms, P95 16.716 ms, and maximum
20.488 ms; 8,471 remained above 15 ms and two above 20 ms. All four components
and their aggregate matched the corresponding current-input results exactly.
These selected difficult frames are not a fresh full-cohort benchmark. Their
maximum does not replace the full-pass maximum above.

The portable command measures the current installed implementation only; it
does not reconstruct the reference benchmark or download EA-P90 automatically.
The figures above are an implementation check, not a claim that all paper
statistical analyses have been rerun with this core. Earlier local and
historical-environment timings remain in [validation history](validation_history.md).
