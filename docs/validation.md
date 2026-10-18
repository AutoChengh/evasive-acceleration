# Validation and automated checks

Run the commands below from the repository root, preferably in a fresh Python
environment. Python 3.10 or newer is required.

```bash
python -m pip install -e ".[dev,batch,viz]"
python tools/check_core_integrity.py
python -m pytest -q
python -m build
```

For accelerated execution, additionally install the `fast` extra:

```bash
python -m pip install -e ".[dev,batch,viz,fast]"
python -m pytest -q
```

Installing without `fast` does not disable an existing Numba installation. Use
separate environments when comparing the two backends. CI sets
`EA_EXPECT_NUMBA=0` or `1`; the regression suite then fails if the requested
backend was not actually loaded.

## Preserved numerical behavior

The usability release keeps `src/core_ea.py` and
`src/baseline_risk_metrics.py` unchanged. The integrity checker compares SHA-256
hashes after converting CRLF to LF, so Git's Windows line-ending conversion
does not create a false failure. It does not normalize whitespace, comments,
or any other source content. The frozen hashes are stored in
`tools/check_core_integrity.py`.

`tests/test_core_regression.py` compares the original default solver against
fixed numerical values recorded on 2026-10-19:

| Case | CTCT | CTCV | CVCT | CVCV | Final EA (m/s²) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Head-on | 4.8000965232 | 4.8000965232 | 4.8000965232 | 4.9109654797 | 4.8278137623 |
| Turning actors | 4.2627359362 | 3.6453490575 | 4.3855626206 | 4.9109654797 | 4.3011532735 |
| Separated parallel actors | 0 | 0 | 0 | 0 | 0 |

The exact state vectors and unrounded expected values are in the test file.
Tests keep the original 10 s horizon, time grids, direction search, and
tolerances. Expected values are constants, not a comparison between two
wrappers that invoke the same calculation. A small `1e-8` numerical comparison
tolerance permits platform arithmetic differences; it does not change the
solver's own tolerance.

The suite also freezes existing `NaN`/infinite aggregation behavior. For
example, two identical, initially overlapping road-user states currently
produce an undefined final EA (`NaN`). This release preserves that boundary
behavior. A missing or undefined result must not be interpreted as zero risk.
The batch interface records such solver failures explicitly.

The integrity guard is specific to this release. A future deliberate algorithm
change requires its own mathematical review and reviewed numerical baselines;
changing the expected hashes alone does not establish correctness.

## Input and interface checks

The other tests cover causal yaw-rate estimation, angle wrapping, irregular
sampling, initialization, rejected-observation recovery, track-ID resets, and
prefix consistency. They also check provided/estimated/automatic input modes,
output overwrite protection, per-row error reasons, isolated baseline errors,
metadata, installed imports, command-line exits, and documented examples.

These checks establish consistency with the released implementation. They do
not validate yaw-rate accuracy against sensor ground truth or reproduce the
paper's complete data-dependent evaluation.

## Example and notebook smoke checks

```bash
python examples/provided_state.py
python examples/from_history.py
ea batch --input examples/data/without_yaw_rate.csv --skip-baselines --output-dir outputs/validation_batch
python examples/comparison_case.py --output-dir outputs/validation_comparison
python -m pip install -e ".[batch,viz,notebook]"
python tools/execute_quickstart.py --output outputs/notebooks/ea_quickstart.executed.ipynb
```

Use a new output directory for repeated runs, or explicitly select the relevant
overwrite option. Example results and executed notebooks belong under
`outputs/`, not alongside source inputs.

Publication figures require Arial with regular, non-bold text. On hosts that
lack Arial, explicitly setting `EA_ALLOW_FONT_FALLBACK=1` permits the plotting
helper's fallback font for a smoke test. This is what CI does; a successful CI
render is not evidence that a figure uses Arial. `EA_ARIAL_FONT_PATH` can point
to a legitimately available Arial font file. The repository does not download
or distribute proprietary fonts.

## GitHub Actions coverage

`.github/workflows/tests.yml` defines:

- Linux and Windows tests on Python 3.10 and 3.12 without Numba;
- a separate Linux/Python 3.12 run with Numba and the same tests;
- example execution, source/wheel builds, and a wheel smoke test in a fresh
  virtual environment outside the checkout;
- quickstart notebook execution, with the executed notebook uploaded as an
  Actions artifact.

The wheel check uses isolated Python (`-I`) and verifies that imported modules
come from the new environment, preventing an editable checkout from hiding a
broken package. It also checks the fixed head-on EA value through the installed
public API.

Adding the workflow does not execute it remotely. Its actual status is visible
in GitHub Actions after the branch is pushed; local test results and remote CI
results must be reported separately.

## Local verification for this addition (2026-10-19)

Two Windows/Python 3.11.7 environments passed the complete 91-test suite:

| Environment | NumPy | Pandas | Matplotlib | Backend | Result |
|---|---|---|---|---|---|
| Existing validation environment | 1.26.4 | 2.1.4 | 3.8.0 | Numba 0.59.0 | 91 passed |
| Fresh environment, no system site packages | 2.4.6 | 3.0.6 | 3.11.2 | No Numba installed | 91 passed |

The fresh environment uses pytest 9.1.1. Its first run exposed a font-flag API
compatibility issue; the shared style now supports both older integer flags
and newer StyleFlags without deprecated constant access on newer Matplotlib.

The quickstart executed all 8 code cells, produced 4 inline PNGs, and reported
no cell errors. The local figures used actual Arial Regular without font
substitution. Plot tests also check regular weight and the history title/note
remaining inside the canvas. The notebook's printed run directory contains
the exact helper hashes, source commit and dirty-tree status, inputs, settings,
versions, and font provenance; it does not imply an uncommitted tree is a
published release.

Source and wheel builds succeeded. A separate minimal wheel installation,
initially with NumPy only and no Numba/Pandas/Matplotlib, imported from outside
the checkout and reproduced the frozen head-on EA value
`4.827813762295801 m/s²`. Importing the optional plotting module did not load
Matplotlib. Thirty relative documentation links were checked with none broken.

The original raw SHA-256 hashes are still:

```text
core_ea.py:               9225c714840e17ef7de6c512e532c86d2f35d0ed4a3d3a321009f69392413ed8
baseline_risk_metrics.py: b81f1c86dc47a2767f5ff293561bd93f037cf371742bcac78ef56c967d837ee1
```

The build still emits the pre-existing setuptools warning about the TOML
license-table form. It is not a build failure and does not change the license;
modernizing that metadata can be handled in a later packaging release.
Linux and Python 3.10/3.12 checks are configured, not claimed as locally run.
