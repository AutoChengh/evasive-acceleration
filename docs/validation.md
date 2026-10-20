# Validation and reproducibility

The current EA core evaluates four CV/CTRV pairings using the paper's motion
and collision geometry, finite-horizon search, fine-grid candidate verification,
and arithmetic aggregation. This page identifies the checks and how to run
them. Earlier measurements remain in the [validation history](validation_history.md).

## Run the tests

```bash
python -m pip install -e ".[dev,batch,viz,fast]"
python tools/check_core_integrity.py
python -m pytest -q
python -m build
```

For a NumPy-only check, use a separate environment without the `fast` extra
and without an existing Numba installation. Omitting an extra does not remove
packages already installed in an environment.

The suite checks:

- Independent reference cases, fine-grid contact checks, boundary cases, short
  horizons, and all four components plus their aggregate.
- Causal yaw-rate estimation, prefix consistency, irregular sampling, track
  resets, initialization, and recovery from invalid observations.
- Installed imports, single-frame and batch commands, error messages, input
  preservation, output protection, and run metadata.
- Recomputable comparisons, custom-column conversion, notebook structure, and
  regular Arial publication typography.
- Benchmark input validation, warm-up exclusion, complete timing retention,
  numerical status reporting, and fresh-output protection.

Frozen reference modules, the historical baseline implementation, and the
causal yaw-rate estimator are protected by source-integrity checks. CRLF is
normalized to LF for those checks; other source content is not normalized.
The current runtime is in `verified_calculator.py` and `verified_solver.py`;
the original calculator remains available as a separate reference.

## Check the examples

```bash
python examples/provided_state.py
python examples/from_history.py
ea batch --input examples/data/without_yaw_rate.csv --skip-baselines --output-dir outputs/check_batch
python examples/comparison_case.py --output-dir outputs/check_comparison
```

The provided-state example should print about `4.800 m/s^2`. The two synthetic
equal-TTC cases give about `0.500` and `0.871 m/s²`, both with TTC `2.00 s`.
They illustrate the calculation; they are not data from the paper experiments.

Publication exports require regular Arial. If it is not installed, point
`EA_ARIAL_FONT_PATH` to a legitimately available regular Arial font. Explicit
font fallback is available for labelled previews and hosted CI only; see
[figure style](figure_style.md).

## Measure runtime

```bash
ea benchmark
ea benchmark --input examples/data/provided.csv --output-dir outputs/check_runtime
```

The command uses the existing EA core and requires no CSV or plotting extras.
Its default is a three-state synthetic smoke test. Supply your own CSV or CSV
directory to measure a relevant workload. It saves every measured call and
records warm-up separately, along with parameters, source hashes, input hashes,
and the execution environment. See [runtime measurements](runtime.md) for
the timing boundary and the separate 857,820-frame implementation check.

## Reproduce a source comparison

For a numerical and performance comparison against an independently preserved
earlier implementation:

```bash
python tools/validate_paper_alignment.py --before-source /path/to/before/src --frames 403 --repeats 3 --diagnostic-scenes 64 --output-dir outputs/check_alignment
```

The report checks each component and the aggregate, source identity, inputs,
classification changes, and candidate contact verification. A speed comparison
does not establish numerical equivalence. Input yaw rates must also match:
re-estimating them from headings changes the input experiment, not just the
implementation under comparison.

The Python API retains full precision. Batch exports round EA columns to four
decimal places after computing the aggregate from unrounded components.

## Notebook and installed package

```bash
python -m pip install -e ".[notebook]"
python tools/execute_quickstart.py --output outputs/notebooks/quickstart.executed.ipynb
```

The notebook calls the same API and example scripts. It is not an independent
EA implementation. Keep executed notebooks and generated outputs outside the
tracked, output-free quickstart.

GitHub Actions defines Linux/Windows and Python 3.10/3.12 checks, separate
accelerated execution, wheel isolation, example execution, and a notebook job.
Local test results are not evidence that hosted CI has passed a new revision.
The [workflow](../.github/workflows/tests.yml) and retained
[historical records](validation_history.md) describe those checks in detail.
