# Validation and automated checks

Version `0.3.0.dev1` uses a verified finite-horizon runtime with the paper's
motion and collision geometry. Every local direction is solved on the fine
grid and every finite candidate passes a SAT collision check. It follows the
earlier `0.3.0.dev0` paper-core migration but does not preserve that version's
unverified coarse fallback or selective fine verification. Existing Python
states, command-line entry points, batch schemas, and causal yaw-rate estimation
remain available. The four original reference sources remain unchanged.

## Run the checks

Run these commands from a source checkout, preferably in a fresh Python
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

Installing without `fast` does not disable an existing Numba installation.
Use separate environments to compare backends. CI sets `EA_EXPECT_NUMBA=0`
or `1` so the tests can verify which backend was loaded.

## Numerical-core migration

The adapter in `src/core_ea.py` calls the four-mode bridge in
`src/evasive_acceleration/_paper/`. The current runtime is implemented in
`verified_calculator.py` and `verified_solver.py`; it reuses the frozen motion
and OBB geometry. The four original calculator, motion, solver, and configuration
modules are retained separately as a historical numerical reference. The source-integrity
guard checks their normalized SHA-256 hashes, and separately protects the
unchanged historical baseline implementation. CRLF is normalized to LF to
avoid false failures after Git's Windows line-ending conversion; other source
content is not normalized.

The public `EA` result evaluates only these four ordered model pairings:

```text
CV_CV, CV_CTRV, CTRV_CV, CTRV_CTRV
```

All four use finite-horizon numerical search with the paper defaults: a
10 s horizon, 0.1/0.02 s coarse/fine steps, 72 coarse directions, 101 local
directions within ±5°, and a 100 m/s² acceleration bound. The scalar
mode-specific magnitudes are averaged. The separate CTRA nine-pairing
extension is not silently added to the public score.

The migration also adopts the paper core's current-contact and no-feasible-
solution result (`+inf`) and fine-grid checks of potentially missed coarse
contacts. Invalid input or a failed computation remains distinguishable as
`NaN` with diagnostic information in batch output.

The Python API retains unrounded values. Batch CSVs round mode-specific EA
and the aggregate to four decimal places, matching the paper's export style;
the aggregate is calculated before rounding individual modes.

Changing the expected hashes is not itself a correctness test. Numerical
checks should compare each mode and the aggregate against the frozen paper
reference using identical states, yaw rates, geometry, and solver settings.
Compare finite values numerically and compare `0`, `inf`, and `NaN` classifications
explicitly. Recomputing one wrapper through another wrapper is not an independent
reference.

## Inputs and interfaces

The yaw-rate estimator is unchanged by this migration. Tests cover wrapped
angular differences, irregular sampling, first-frame initialization, track-ID
resets, invalid-observation recovery, and prefix consistency. Provided,
estimated, and automatic yaw-rate policies feed the same EA core.

For paper-result consistency, use the supplied yaw-rate observations associated
with those inputs. Applying causal EWMA to headings is an alternative input
workflow, not a promise of identical yaw-rate measurements.

Other checks cover installed imports, command-line exits, output protection,
per-row errors, separate baseline diagnostics, metadata, the custom-data
adapter, recomputable examples, and figure typography.

## Performance measurements

Measure the four-pairing public EA call with the same inputs, settings, machine,
and backend for both versions. Do not compare the old four-pairing result with
the research entry point that also computes nine CTRA combinations and other
metrics. Keep import/JIT compilation time separate from warmed execution.
Report sample count and both median and tail latency; record dependency
versions and backend availability alongside the results.

Faster execution on a measured sample does not establish a universal speedup.
Numerical consistency and performance checks accompany this migration;
full dataset-level and independent solver validation are separate work.

## Review-branch fixes on 2026-10-20

After the four review fixes, the default `python -m pytest -q` command passed
**193 tests in each environment**: the Numba and NumPy-only Windows/Python
3.11.7 environments listed below. The earlier 165-test result is retained as
the pre-review runtime check, not the current suite size.

New regression checks cover explicit, labelled font previews when Arial is
unavailable; valid short horizons with final-endpoint collision checks;
default test discovery excluding retained output snapshots; and migration
reports identifying the current checkout rather than the historical runtime.
Publication rendering still requires regular Arial; non-Arial output requires
an explicit preview opt-in and is not for publication. The frozen paper sources and causal yaw-rate
estimator pass their unchanged integrity checks.

The quickstart executed all 8 code cells with 4 inline PNGs and no cell errors.
Both public state examples and the batch CLI passed. The comparison example
also rendered successfully with simulated missing Arial, with the substitute
font labelled in the figure and recorded in its JSON. Source and wheel builds
passed. Local review artifacts are retained under
`outputs/validation/review_fixes_20261020/` and are not committed.

## Verified runtime checks for 0.3.0.dev1

The full suite passed **165 tests** in each of the two separate environments
listed below, using Windows and Python 3.11.7:

- NumPy 1.26.4, Pandas 2.1.4, Matplotlib 3.8.0, Numba 0.59.0.
- NumPy 2.4.6, Pandas 3.0.6, Matplotlib 3.11.2, without Numba.

The additional checks cover boundary-rounding contacts after `nextafter`,
nonmonotone collision intervals, unsafe incumbent-value pruning, fine-only
baseline collisions, every local direction receiving a fine calculation,
verified fallback searches, non-divisible time steps, and current contact.
All four frozen reference modules, the baseline helper, and the causal yaw-rate
estimator still pass their unchanged source-integrity checks.

A source snapshot was preserved before the repair and loaded with its own
four-mode dependencies, independently of the new runtime. Across all 403 demo
frames, neither the four components nor their aggregate changed by more than
`1e-8 m/s²`; the maximum aggregate difference was `2.676e-14 m/s²`. The nine
frozen reference cases remain reproducible with the original reference
calculator, and the repaired runtime also matched these cases at the stated
tolerance. Old reference values were not replaced with newly generated values.

In 64 seeded synthetic scenes, all 256 finite four-mode outputs passed an
independent NumPy implementation of the fine-grid SAT check: 53 positive
candidates and 203 collision-free zero baselines. Dedicated regression tests
also retain the earlier counterexample whose old finite candidate still
collided, and confirm that the repaired candidate passes its final check.
For that synthetic CTRV pairing, the old value was `4.777277013559797 m/s²`;
the repaired search selects a different verified direction with value
`5.3321215516568055 m/s²`. This is an implementation regression case, not a
measurement of how often paper-data rankings change.

Source and wheel distributions built successfully. A fresh, non-editable
environment with only the wheel and NumPy passed isolated (`-I`) checks from
outside the checkout, including both CLI entry points, public EA and causal
yaw-rate APIs, source provenance, and the repaired counterexample. Numba,
Pandas, and Matplotlib were absent in that environment. The quickstart notebook
also executed all 8 code cells and produced the expected inline figures with
no cell errors; the tracked notebook remains output-free.

The latest paired benchmark used the same i7-12700KF, Python 3.11.7, NumPy
1.26.4, and Numba 0.59.0. Each version evaluated the same 403 frames three
times, with alternating engine order. JIT compilation, warm-up, diagnostics,
and I/O were excluded from the following times:

| Four-mode version | Mean (ms/frame) | Median | P95 |
| --- | ---: | ---: | ---: |
| Preserved `0.3.0.dev0` runtime | 4.087 | 3.031 | 11.405 |
| Verified `0.3.0.dev1` runtime | 5.917 | 6.352 | 16.266 |

This batch measured a 44.8% mean time increase for the added searches and
checks, not a speedup. An earlier run measured a 28.6% increase; both reports
are retained. Absolute timings vary with machine load and clock state. They
are not directly comparable to the separate historical benchmark below.

Reproduce this check with an independently preserved earlier source directory:

```bash
python tools/validate_paper_alignment.py --before-source /path/to/before/src --frames 403 --repeats 3 --diagnostic-scenes 64 --output-dir outputs/validation/paper_alignment
```

The comparison records before/after source hashes, input hashes, settings,
classification changes, dependency isolation, per-frame differences, candidate
checks, and timings. The latest local report is under
`outputs/validation/paper_alignment_20261020/checked/`; source and installed
distribution versions are both `0.3.0.dev1`. Source and input hashes were
unchanged during the run.

No original paper datasets, reported figures, or statistical tables were
recomputed or replaced in this engineering check. Manuscript changes required
to describe the corrected runtime are listed, with original text and English
replacements, in [the paper alignment checklist](paper_alignment_changes_20261020.md).

## Historical paper migration checks for 0.3.0.dev0

The full suite passed in two separate Windows/Python 3.11.7 environments:

- NumPy 1.26.4, Pandas 2.1.4, Matplotlib 3.8.0, Numba 0.59.0: **123 passed**.
- NumPy 2.4.6, Pandas 3.0.6, Matplotlib 3.11.2, no Numba: **123 passed**.

The four paper-source hashes were checked against the canonical paper version.
All **9 independent reference cases** passed. All **403 frames** from the two
bundled demo CSVs matched the paper calculator for each of the four components
and their mean at the paper's four-decimal export precision, using identical
provided yaw rates and settings. The source-integrity checks also passed for
the unchanged causal yaw-rate estimator and historical baseline module.

Both source and wheel distributions built successfully. A new non-editable
environment containing only the wheel and NumPy 2.4.6 passed an isolated
(`-I`) smoke test from outside the checkout, without Numba, Pandas, or
Matplotlib. Installed module origins, the head-on reference, causal yaw rate,
command-line version, packaged provenance, and all four original source hashes
were checked. The source archive includes the reference fixture, visualization
script, and validation tools.

The final warmed benchmark used an Intel Core i7-12700KF on Windows, Python
3.11.7, NumPy 1.26.4, and Numba 0.59.0. Each engine evaluated the same 403 frames
three times (1,209 calls per engine), alternating engine order. Compilation,
warm-up, file I/O, and the independent reference calculation were not timed.

| Four-pairing implementation | Mean (ms/frame) | Median | P95 |
| --- | ---: | ---: | ---: |
| Paper numerical core (`0.3.0.dev0`) | 2.402 | 1.771 | 7.003 |
| Earlier mixed core (`e47598e`) | 5.127 | 5.089 | 10.875 |

The measured mean speedup is about **2.13× on this workload**. The old and new
outputs differ on 133 of 403 frames, chiefly in CV–CV; those differences are
recorded rather than hidden by retaining the old scores. Migration consistency
is assessed against the canonical paper calculation, not against the earlier
mixed implementation.

The table above is an archived `0.3.0.dev0` measurement, not a benchmark of
the current runtime. Running the following command from this checkout compares
the **currently checked-out runtime** with the earlier mixed core (`e47598e`):

```bash
python tools/validate_paper_migration.py --frames 403 --repeats 3 --output-dir outputs/validation/paper_migration
```

The report records source/input hashes, settings, versions, per-frame
comparisons, and timings. `--paper-source /path/to/frozen/paper/src` selects a
separate, hash-verified **reference calculator**, not the runtime being timed.
Reproducing the historical table requires the preserved `0.3.0.dev0` source
snapshot and its recorded environment; this command does not reconstruct that
runtime or guarantee its historical timings. The current runtime identity and
source hashes are recorded in each new report.
`--skip-legacy` runs paper-consistency checks without requiring the historical
Git revision, for example in a source archive or shallow checkout.

These checks verify this implementation migration and its interfaces. They do
not replace the separate dataset-level validation planned for the new release.

## Examples and notebook

```bash
python examples/provided_state.py
python examples/from_history.py
ea batch --input examples/data/without_yaw_rate.csv --skip-baselines --output-dir outputs/validation_batch
python examples/comparison_case.py --output-dir outputs/validation_comparison
python -m pip install -e ".[batch,viz,notebook]"
python tools/execute_quickstart.py --output outputs/notebooks/ea_quickstart.executed.ipynb
```

Use fresh output directories for repeated runs, or explicitly select the
appropriate overwrite option. Executed notebooks and generated validation
artifacts belong under `outputs/`, not alongside source inputs.

Publication figures require regular Arial, with no bold text. Hosts without
Arial can explicitly set `EA_ALLOW_FONT_FALLBACK=1` for a labelled smoke-test
preview. `EA_ARIAL_FONT_PATH` may point to a legitimately available regular
Arial font. The repository does not download or distribute proprietary fonts.

## GitHub Actions coverage

`.github/workflows/tests.yml` defines:

- Linux and Windows tests on Python 3.10 and 3.12 without Numba;
- a separate Linux/Python 3.12 run with Numba;
- example execution, source/wheel builds, and an isolated installed-wheel check;
- headless notebook execution, saving the executed notebook as an artifact.

The wheel check uses isolated Python (`-I`) outside the checkout and verifies
module origins, so an editable install cannot hide a broken wheel. It checks
the migrated head-on reference through the installed public API.

Local test results and hosted CI results are reported separately. A modified
workflow is not evidence of a successful GitHub run until that revision has
actually been pushed and checked.

## Historical migration example checks for 0.3.0.dev0

The migrated public API was exercised locally on Windows/Python 3.11.7 with
Numba 0.59.0. The provided-state example prints `EA = 4.800 m/s^2`; the equal-TTC
comparison produces EA values `0.5000000000` and `0.8706485577 m/s²`, with TTC
`2.00 s` in both cases. The tracked comparison preview was regenerated using
Arial Regular, with no font substitution.

The updated quickstart executed all 8 code cells, rendered 4 inline PNGs, and
reported no cell errors. Its run summary records the new core identity and
source provenance, the vendored modules and four-mode bridge hashes, input
hashes, settings, and font provenance. The separate notebook-structure test
also passed. These are local interface/example checks, not hosted CI or a
reproduction of the full paper experiments.

## Earlier usability-only verification

Before this numerical migration, the `0.2.0.dev0` usability addition passed
91 tests locally on 2026-10-19 in two Windows/Python 3.11.7 environments:
NumPy 1.26.4/Pandas 2.1.4/Matplotlib 3.8.0 with Numba 0.59.0, and a fresh
NumPy 2.4.6/Pandas 3.0.6/Matplotlib 3.11.2 environment without Numba.
That earlier notebook executed 8 code cells and rendered 4 inline PNGs.

Those historical results belong to the old numerical core, whose head-on
aggregate was `4.827813762295801 m/s²`. They are retained for version
traceability, not presented as validation of the migrated core. Results for
the migrated `0.3.0.dev0` core are reported above.
