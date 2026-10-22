# v53: globally bounded EA with the existing API

v53 is an explicit, opt-in solver for the same four CV/CTRV motion pairings.
It keeps the existing state containers, arithmetic-mean EA aggregation, and
[causal yaw-rate estimation](causal_yaw_rate.md). Existing calls continue to
use the paper-compatible solver unless you select v53.

## Install and prepare

Install Numba and build the native component once, before computation:

```bash
python -m pip install -e ".[fast]"
python -m evasive_acceleration.native_build
```

The build requires an available GCC-compatible C++17 compiler (`g++`, or
MinGW-w64 `g++` on Windows); it does not install one. Use `--compiler` to select
its executable if needed. Windows builds statically link the MinGW runtime,
including thread support, so loading does not depend on compiler-private DLLs.
The
default build uses a user cache outside the repository. For host-specific
optimization, add `--native`. A custom build location can be selected with
`--output`; use a fresh path, then set `EA_NATIVE_CELLS_LIBRARY` to the resulting
library when computing EA. See `python -m evasive_acceleration.native_build --help`
for the exact options supported on your platform.

Compilation is never hidden inside an EA call. Missing native/Numba dependencies
fail explicitly; requesting v53 does not silently run the paper solver instead.
The first calls can still compile Numba functions. Warm up on the same CPU
before measuring execution time.
The v53 benchmark warms every selected input at least once, rather than only
a short prefix that might miss a lazily compiled turning or fallback path.
Its JSON records the actual warm-up count separately from measured calls.

## Keep your current Python inputs

```python
from evasive_acceleration import compute_ea, get_backend, use_backend

# state_a and state_b are the same RoadUserState objects used by the existing API.
with use_backend("v53"):
    print(get_backend())
    ea = compute_ea(state_a, state_b)

# Leaving the context restores the previous backend.
```

Yaw rate can be supplied directly or obtained with
`build_state_with_estimated_yaw_rate` before entering this context. No future
headings are needed. Run the complete example with:

```bash
python examples/v53_usage.py
```

The CLI accepts the same explicit selection:

```bash
ea single --backend v53
ea batch --backend v53 --input examples/data/without_yaw_rate.csv --output-dir outputs/v53_batch
ea benchmark --backend v53 --input examples/data/provided.csv --output-dir outputs/v53_runtime
```

For batch processing, install the `batch` extra as well. Use fresh output
directories so previous paper-solver results remain available for comparison.
Yaw-rate preprocessing stays outside the numerical solver and benchmark timer.

## What changes numerically

The motion models, body geometry, 10 s prediction horizon, and 100 m/s²
acceleration cap are unchanged. The reported EA remains the arithmetic mean of
four minimum relative-acceleration magnitudes, in m/s².

v53 searches the two-dimensional acceleration space with convex-cell global
branch-and-bound. A collision at an exact time supplies a necessary avoidance
constraint; a cell's minimum norm supplies a global lower bound. A finite upper
bound must pass the original conservative continuous-time body-contact check
over the complete prediction horizon. The target upper/lower difference is
0.001 m/s² per mode, rather than merely the resolution of a local angular scan.

Ordinary turn rates use the Python/Numba path; rapid rotations use the native
C++ frontier. Within one four-mode call, modes may share contact-time and
acceleration-vector proposals. Each mode rebuilds its own constraints and
rechecks safety: neither another mode's scalar answer nor its lower bound is
reused as proof. There is no cross-frame lookup of EA answers.

Source modules under `_experimental` remain internal implementation details;
use the public backend selector rather than importing those modules. The
paper solver remains available for reproducing older outputs. Results are
not required to be bitwise identical across the two numerical methods.

## Recorded full-cohort acceptance

The frozen v53 source was tested on all 857,820 original interaction frames,
including abnormal yaw-rate inputs. One measured pass included all four
CV/CTRV evaluations and the unchanged aggregate mean. No slow observation was
removed or replaced by a faster replay.

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
| Maximum across recorded stress/repeated tests | 15.820041 ms |

The environment was server 39103 (`thicv-PowerEdge-T640`), an Intel Xeon Gold
6226R, logical CPU 20, one process/thread, Python 3.12.13, and g++ 9.4.0.
There were 5,993 excluded same-core warm-up calls and no JIT signature growth
during measurement. Parsing, compilation, warm-up, output writes, and independent
validation were outside the algorithm timer. The native acceptance build used
host-specific optimization, strict float64, no fast-math, and disabled FMA
contraction. A new machine or compiler build must be benchmarked separately;
these observations are not an unconditional hard-real-time guarantee.

All finite full-cohort results differed from the frozen continuous-search
reference (`continuous_parallel`, final v13) by at most
0.000949699 m/s². No original zero became positive, and no finite/Inf
classification changed. Nine raw-input frames retained Inf results; corrected
inputs were tested separately and did not replace the primary records.

That reference is not the retained public `paper` backend. Differences from
the older grid/local-search backend can be larger because v53 checks the
continuous time horizon and searches all acceleration directions globally.

An expanded 1,160-case suite passed 24,792 numerical, continuous-time safety,
global-bound, and geometric-invariance checks. Other acceptance checks included
391 Python regressions, 991 native/geometry checks, 195 context-isolation checks,
20,000 scalar/SIMD comparisons, and an independent 30-gate evidence audit.
These numbers describe the frozen acceptance run, not an assertion that
every hosted CI job for a later review commit has already passed.

The public adapter was additionally checked on those 1,160 expanded cases:
all 4,640 component outputs exactly matched the frozen v53 entry point, and
all 4,640 independent continuous-time safety checks passed. On the same
server/CPU, the warmed public four-mode call plus mean peaked at 15.356282 ms.
An independent rebuild reproduced the frozen native binary hash. Windows
native execution, an isolated installed wheel, and 20,000 scalar/SIMD checks
on each of Linux and Windows also passed. These integration checks are
separate from the earlier full-cohort experiment.
[Machine-readable acceptance summary](data/v53_acceptance_20261022.json).

One rotated expanded test exposed a reference output whose angular search had
not converged. Its larger safe upper bound was not treated as an exact answer;
v53's lower result passed independent continuous safety, global-bound, and
rigid-rotation checks. The original reference record was retained.

## Source provenance

The integrated kernel is the validated dependency closure of the frozen v53
source snapshot dated 2026-10-22. It does not bundle failed experimental
alternatives, raw datasets, or host-specific compiled libraries. The source
snapshot SHA256 is:

```text
37f35ab3b0b62c587cac46b501f398d6283467ccb2220e9715700600bd9c7a15
```

The measured Linux native binary SHA256 was:

```text
a0037dd2e748bf3af863d2e3b00e07ad1120115bfaa32f4d8f5230ee4ac119c1
```

The full-cohort summary and acceptance report are historical evidence tied to
that source/build/input set. For the installed review version, use the
[runtime command](runtime.md) to record fresh observations and environment
information. The separate original EA repository and the v2 main branch are
not replaced by preparing this review branch.
