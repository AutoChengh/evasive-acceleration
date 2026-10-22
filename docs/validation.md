# Validation and reproducibility

All supported entry points use the same EA numerical core. This page describes
current checks; older implementations are available only through the
[historical archives](history.md).

## Run the tests

```bash
python -m pip install -e ".[dev,batch,viz]"
python -m evasive_acceleration.native_build
python tools/check_solver_integrity.py
python -m pytest -q
python -m build --outdir outputs/package_check_dist
python tools/check_wheel.py --wheel-dir outputs/package_check_dist
```

Numba and the explicitly built native library are required. Missing native
dependencies are errors, not a request to use another solver. See
[native setup](runtime.md#prepare-the-native-component).

The suite checks:

- Numerical reference cases, independent continuous-time body-contact
  verification, global objective bounds, boundary cases and invariances.
- The four CV/CTRV components and arithmetic-mean score through the public API.
- Causal yaw-rate estimation, prefix consistency, irregular sampling, track
  resets, initialization and invalid-observation recovery.
- Installed imports, single-frame/batch commands, input preservation, output
  protection, error messages and run metadata.
- Recomputable comparisons, custom-column conversion, notebook execution and
  regular Arial publication typography.
- Benchmark input validation, complete timing retention, excluded warm-up,
  numerical statuses and fresh-output protection.

Source-integrity checks protect the numerical sources, historical baseline
functions and causal yaw-rate estimator. Line endings are normalized for
those checks; other source content is not normalized.

Use a fresh build directory. The package check installs both the wheel and
source distribution in separate clean environments, builds their native
resources, and runs outside the checkout so editable imports cannot hide
missing files.

## Check the examples

```bash
python examples/provided_state.py
python examples/from_history.py
ea batch --input examples/data/without_yaw_rate.csv --skip-baselines --output-dir outputs/check_batch
python examples/comparison_case.py --output-dir outputs/check_comparison
```

The provided-state example prints approximately `4.911 m/s^2`. The two
synthetic equal-TTC cases produce approximately `0.500` and `0.886 m/s²`,
both with TTC `2.00 s`. They are interface illustrations, not paper data.
The Python API retains full precision; batch EA columns are rounded to four
decimals after aggregation from unrounded components.

Publication exports require regular Arial. Point `EA_ARIAL_FONT_PATH` to a
licensed regular Arial font if needed. Explicit font fallback is for labelled
previews/hosted CI only; see [figure style](figure_style.md).

## Runtime and notebook

```bash
ea benchmark
ea benchmark --input examples/data/provided.csv --output-dir outputs/check_runtime
python -m pip install -e ".[notebook]"
python tools/execute_quickstart.py --output outputs/notebooks/quickstart.executed.ipynb
```

The benchmark records every measured call, excluded warm-up, settings,
source/input hashes and environment. Its no-input default is a synthetic
three-state smoke test. The [runtime guide](runtime.md) defines the timer
and reports the frozen 857,820-frame experiment separately.

The output-free notebook calls the same API and example scripts, not an
independent EA implementation. Keep executed notebooks and generated outputs
outside the tracked source.

GitHub Actions checks Linux/Windows, supported Python versions, native execution,
isolated wheel installation, examples and notebook execution. Local success
does not establish that hosted tests passed a new revision; inspect the
[workflow and its runs](../.github/workflows/tests.yml) for that commit.
