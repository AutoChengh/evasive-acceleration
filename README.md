# Evasive Acceleration (EA)

Compute two-dimensional interaction risk from the current states of two road
users. EA measures the minimum constant relative acceleration needed to avoid
predicted collision, in **m/s²**. Supply yaw rates directly, or estimate missing
yaw rates from current and historical headings with the same EA solver.

Paper: [Driving risk emerges from the required two-dimensional joint evasive acceleration](https://arxiv.org/abs/2604.17841).

[Install and run](#install-and-run) · [Python usage](#python-usage) · [Input conventions](#input-conventions)

## Visual overview

### Paper Fig. 1: the EA framework

<p align="center">
  <a href="assets/figures/ea_framework_fig1.pdf">
    <img src="assets/figures/ea_framework_fig1.png" alt="Figure 1: motivation for EA, the two-dimensional relative-acceleration formulation, and validation using crashes and naturalistic interactions" width="100%">
  </a>
</p>

Figure 1 from the paper brings together the limitations of time-based risk
measures, the minimum two-dimensional relative acceleration required for
collision avoidance, and the experimental validation.
[Open the vector PDF](assets/figures/ea_framework_fig1.pdf) for a closer look.

## Visual examples

### Naturalistic interaction

<p align="center">
  <img src="assets/gifs/SIND_Tianjin_8_9_1_486_496.gif" alt="SinD Tianjin interaction" width="90%">
</p>

A vehicle–motorcycle intersection interaction from the SinD dataset. The
magnitude of the red arrow represents EA; a longer arrow denotes greater
required evasive acceleration.

### Reconstructed crash

<p align="center">
  <img src="assets/gifs/crash_684_T-bone.gif" alt="CIMSS-TA reconstructed T-bone crash" width="90%">
</p>

A reconstructed T-bone crash from CIMSS-TA, using the same red-arrow encoding.

## Install and run

From a source checkout, with Python 3.10 or newer:

```bash
python -m pip install -e ".[batch,viz,fast]"
ea single
ea batch --input examples/data/without_yaw_rate.csv
```

The single-frame example prints approximately `EA = 4.828 m/s^2`. The batch
command automatically estimates missing yaw rates. Open:

```text
outputs/batch_results/auto/without_yaw_rate_EA.csv
outputs/batch_results/auto/without_yaw_rate_EA.run.json
```

The CSV contains frame-wise EA, yaw-rate provenance, and error diagnostics.
The JSON records the input, settings, and run summary. Existing result files
are protected; rerun with `--overwrite` to replace them.

For computation only, `python -m pip install -e .` installs the minimal package.
The optional `batch`, `viz`, and `fast` extras add CSV processing, plotting, and
Numba acceleration respectively. Numba's first call can include compilation
time. `python -m evasive_acceleration` accepts the same commands as `ea`.

## Three ready-to-run inputs

These small examples describe the same two approaching vehicles at 10 Hz.
They vary only in yaw-rate availability. Run the commands from the checkout
root after installation:

| Input | Command | Yaw-rate handling |
|---|---|---|
| [Provided values](examples/data/provided.csv) | `ea batch --input examples/data/provided.csv --yaw-rate-source provided` | Use both input columns. |
| [No yaw-rate columns](examples/data/without_yaw_rate.csv) | `ea batch --input examples/data/without_yaw_rate.csv` | Estimate both streams causally. |
| [Partially missing values](examples/data/mixed_yaw_rate.csv) | `ea batch --input examples/data/mixed_yaw_rate.csv` | Use finite provided values; estimate individual missing values. |

Default outputs are separated by policy under `outputs/batch_results/`:
`provided/`, `estimated/`, or `auto/`, relative to your current working
directory. To compare policies on the same input in fresh output folders:

```bash
ea batch --input examples/data/provided.csv --yaw-rate-source provided --output-dir outputs/comparison/provided
ea batch --input examples/data/provided.csv --yaw-rate-source estimated --output-dir outputs/comparison/estimated
```

The first row estimated from heading alone uses a zero initialization
placeholder. Its diagnostics distinguish that placeholder from an initialized
zero-turn estimate. Consequently, the two policies need not have identical EA
on that row, even with otherwise identical data.

## Python usage

After installation, imports work from any working directory. The complete
scripts [provided_state.py](examples/provided_state.py) and
[from_history.py](examples/from_history.py) can also be copied into your project.

### Yaw rate is available

```python
import math
from evasive_acceleration import RoadUserState, compute_ea

agent_a = RoadUserState(
    x=0.0, y=0.0, speed=10.0, heading=0.0,
    length=4.5, width=1.8, yaw_rate=0.0,
)
agent_b = RoadUserState(
    x=20.0, y=0.0, speed=8.0, heading=math.pi,
    length=4.7, width=1.9, yaw_rate=0.0,
)
ea = compute_ea(agent_a, agent_b)
print(f"EA = {ea:.3f} m/s^2")  # Approximately 4.828 m/s².
```

### Yaw rate is unavailable

Use `RoadUserKinematicState` with six fields and attach the estimate from each
object's own history:

```python
from evasive_acceleration import (
    RoadUserKinematicState,
    build_state_with_estimated_yaw_rate,
    compute_ea,
)

current = RoadUserKinematicState(
    x=0.0, y=0.0, speed=10.0, heading=0.02,
    length=4.5, width=1.8,
)
state, estimate = build_state_with_estimated_yaw_rate(
    current,
    times_s=[0.0, 0.1, 0.2],
    headings_rad=[0.0, 0.01, 0.02],
)
print(state.yaw_rate, estimate.initialized)  # 0.1, True
```

Repeat this for the other road user, then call `compute_ea(state_a, state_b)`.
Both histories must end at the current decision time. The last history heading
must match the corresponding current state. For continuous online use, keep
one `CausalYawRateEWMA` instance per object; see the
[yaw-rate method and API](docs/causal_yaw_rate.md).

## Input conventions

All inputs for an interaction must use the same Cartesian coordinate frame.
Positions describe the centers of the oriented bounding boxes; heading is
measured counterclockwise from the positive x axis. Positive yaw rate increases
heading. Convert clockwise headings, image coordinates with downward-positive
y, degrees, and rear-axle positions before calling the solver.

| State field | Meaning | Unit |
|---|---|---|
| `x`, `y` | Bounding-box center position | m |
| `speed` | Speed magnitude along heading | m/s |
| `heading` | Body heading | rad |
| `length`, `width` | Positive body dimensions | m |
| `yaw_rate` | Signed heading rotation rate | rad/s |

Use appropriate body dimensions for vehicles, cyclists, and pedestrians.
The heading of a box and the direction of its displacement can differ,
especially at low speed; the estimator uses the headings you supply.

### Batch CSV schema

Each file must describe **one continuous interaction pair in increasing time
order**. Keep observations in their original causal order; do not interleave
different pairs. If `Vehicle Number` or `2_Vehicle Number` changes, the
corresponding estimator resets.

| Agent A column | Agent B column | Unit |
|---|---|---|
| `Position X (m)` | `2_Position X (m)` | m |
| `Position Y (m)` | `2_Position Y (m)` | m |
| `Velocity (m/s)` | `2_Velocity (m/s)` | m/s |
| `Heading` | `2_Heading` | rad |
| `Length (m)` | `2_Length (m)` | m |
| `Width (m)` | `2_Width (m)` | m |

The default yaw-rate policy is `auto`:

| Policy | Extra input | Behavior |
|---|---|---|
| `provided` | `Yawrate`, `2_Yawrate`, both finite | Use the input yaw rates. Timestamps are not needed for EA. |
| `estimated` | `Time (s)` | Estimate both yaw-rate streams; ignore provided values. |
| `auto` | Yaw-rate columns when available; `Time (s)` if any value needs estimation | Prefer each finite supplied value, otherwise estimate it. |

In `auto`, timestamps are only required when estimation is actually needed.
When timestamps are available, estimator histories are maintained on provided
rows too, so a later missing yaw rate can use earlier headings. Additional
input columns are preserved. Baseline metric errors are reported separately
from EA.

The causal estimator wraps heading differences to `[-π, π)`, divides by the
actual elapsed time, and applies a rate EWMA. Defaults are `tau = 0.6 s`,
output deadband `0.002 rad/s`, and four output decimals. The internal state
keeps full precision. These preprocessing settings, initialization behavior,
and limitations are documented in [docs/causal_yaw_rate.md](docs/causal_yaw_rate.md).

## Command-line reference

```bash
ea single --help
ea batch --help
ea batch --input path/to/case.csv
ea batch --input-dir path/to/csvs --yaw-rate-source estimated
ea batch --input path/to/case.csv --output-dir path/to/results
```

`--input` and `--input-dir` are alternatives. A custom `--output-dir` receives
files directly, without a policy subdirectory; use separate directories when
comparing policies. Files ending in `_EA.csv` are skipped during directory
discovery to avoid processing generated results again. The CLI reports file
and row failures and returns a nonzero exit code if a run has errors. An
input file cannot be replaced by an output, including with `--overwrite`.

The original source entry points remain available:

```bash
python src/single_frame.py
python src/batch_compute.py --input examples/data/without_yaw_rate.csv
```

Running batch without an input uses the source checkout's `demo_data/` when
present. `demo_data/`, `examples/`, and `visualization/` are checkout resources;
code installed from a wheel should be given explicit external input paths.

## Reading the results

| Output | Meaning |
|---|---|
| `EA_CVCV`, `EA_CVCT`, `EA_CTCV`, `EA_CTCT` | EA for the four nominal motion combinations, in m/s² |
| `EA` | Arithmetic mean of the four mode-specific values, in m/s² |
| `EA_Yawrate_A_used_rad_s`, `EA_Yawrate_B_used_rad_s` | Actual yaw rates used by the solver |
| `EA_Yawrate_A_source`, `EA_Yawrate_B_source` | Provided, estimated, initialization, or error provenance |
| `EA_Yawrate_A_initialized`, `EA_Yawrate_B_initialized` | Estimated: a backward difference was available; provided: the supplied value was finite |
| `EA_Yawrate_A_samples`, `EA_Yawrate_B_samples` | Valid causal observations in the estimator |
| `EA_Yawrate_A_error`, `EA_Yawrate_B_error` | Specific yaw-rate input or estimation errors |
| `EA_status`, `EA_error` | EA computation status and failure detail |
| `EA_baseline_error` | Separate error detail for optional baseline calculations |

`EA = 0` means no added relative acceleration is required under the evaluated
motion assumptions and finite solver horizon. A yaw-rate value of `0` on the
first estimated observation is an initialization placeholder, not an observed
zero derivative. `NaN` indicates an unavailable result; inspect status and
error columns. `inf` means the search did not find a solution within its
configured limits; it does not establish physically unavoidable collision.

Provided rows report a sample count of zero because no heading-history
estimate was used for that row. Read initialization and sample count together
with the source column.

Batch output also includes baseline quantities such as `DRAC`, `TTC`, `TAdv`,
`ACT`, `EI`, `MEI`, and `TTC2D`. Their definitions and normalization rules are
in `src/baseline_risk_metrics.py`. For example, a non-conflicting TTC can be
`inf`, which has a different meaning from an unbounded EA search result.

## What EA computes

The solver evaluates nominal constant-velocity (CV) and constant-turn-rate
(CT) combinations for the two road users, then averages the four EA values.
CT uses the supplied or estimated yaw rate; CV uses zero yaw rate.

Numerical settings such as `T_total`, `dt_coarse`, `dt_fine`, `a_max`, and
direction-grid sizes control the finite-horizon search and its approximation
accuracy. Report these settings and the yaw-rate preprocessing configuration
when reproducing an experiment. The physical EA definition and the numerical
search remain the same for both yaw-rate input paths.

The paper frames EA in two-dimensional relative motion space, considering the
directions of collision-avoiding acceleration and retaining the least costly
one. It reports evaluation on 44,180 naturalistic interactions from five open
datasets across three countries and 658 reconstructed crashes. Its reported
runtime of approximately 5 ms depends on the implementation and hardware;
first-call JIT compilation should be excluded when measuring steady-state
speed. See the paper for experiment protocols and comparisons.

## Generate GIFs from a source checkout

Install the `batch` and `viz` extras, compute batch results, then run:

```bash
python visualization/visualize_tracks_to_gif.py --input-dir outputs/batch_results/auto --output-dir outputs/gifs
```

Use `python visualization/visualize_tracks_to_gif.py --help` for frame and
time-range options. The rendering script is separate from computation.

## Repository layout

```text
src/evasive_acceleration/        Installable API and command-line entry point
src/core_ea.py                  Original EA solver
src/single_frame.py             State helpers and compatible source entry point
src/batch_compute.py            CSV pipeline and compatible source entry point
src/yaw_rate.py                 Causal yaw-rate estimator
src/baseline_risk_metrics.py    Comparison metrics
examples/                      Small CSV inputs and complete Python examples
demo_data/                     Original interaction demonstrations
docs/                          Method documentation
visualization/                 Source-checkout GIF renderer
tests/                         Numerical, causality, API, and pipeline checks
assets/gifs/                   README animations
assets/figures/                Paper Fig. 1 preview and vector original
pyproject.toml                 Package metadata and optional dependencies
```

## Development checks

```bash
python -m pip install -e ".[batch,viz,fast,dev]"
python -m pytest
```

Tests check the original solver result, yaw-rate numerics and causality,
installed imports, and batch input/output behavior.

## Citation

```bibtex
@article{cheng2026driving,
  title={Driving risk emerges from the required two-dimensional joint evasive acceleration},
  author={Cheng, Hao and Jiang, Yanbo and Yu, Wenhao and Zhou, Rui and Bian, Jiang and Chen, Keyu and Liu, Zhiyuan and Huang, Heye and Zhang, Hailun and Zhang, Fang and others},
  journal={arXiv preprint arXiv:2604.17841},
  year={2026}
}
```

Cheng H, Jiang Y, Yu W, et al. *Driving risk emerges from the required
two-dimensional joint evasive acceleration.* arXiv:2604.17841, 2026.

## License

See [LICENSE](LICENSE) for the project license.
