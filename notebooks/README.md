# Interactive EA quickstart

The notebook reuses the same public API, synthetic comparison, custom-data
adapter, and EA core as the command-line examples. It is not a
reproduction of the paper's statistical experiments.

## Local Jupyter

From a source checkout, install into a virtual environment:

```bash
python -m pip install -e ".[notebook]"
python -m evasive_acceleration.native_build
python -m notebook notebooks/ea_quickstart.ipynb
```

Run all cells in order. Each run creates a fresh directory under
`outputs/notebooks/runs/` containing actual comparison results, converted
input, computed CSV, the two figures, and a provenance summary. No input data
are replaced. The tracked notebook contains no stale outputs or widget state.

The manual controls adjust clearance and closing speed. Press their Run button
to calculate, rather than running an expensive solve on every slider movement.
The geometry arrows are velocities, not an optimal EA control solution.

To test the whole notebook without a browser:

```bash
python tools/execute_quickstart.py --output outputs/notebooks/ea_quickstart.executed.ipynb
```

The executor uses the invoking Python environment, stops on a failed cell,
checks that every code cell executed and at least three PNG figures were
rendered, and saves diagnostics outside the tracked source notebook.

## Colab or another hosted notebook

Open [the public notebook](https://colab.research.google.com/github/AutoChengh/evasive-acceleration/blob/main/notebooks/ea_quickstart.ipynb)
in Colab. Set `ALLOW_PUBLIC_BOOTSTRAP=True` to permit cloning and installing
the public repository in the hosted session. The setup explicitly builds
the native component before any EA call; an available C++17 `g++` is required.
Set `SOURCE_REVISION` to a branch, tag or commit when pinning a run and record
the printed checkout commit. Do not paste access tokens into a notebook.
Local execution never clones or installs automatically; prepare the native
library with the local setup above.

## Match Fig. 1 typography and colors

All new figures share `evasive_acceleration.plotting.figure_style()`:
regular Arial, no bold text, white/warm off-white panels, muted green and gold
vehicle colors, and a coral EA accent. See [the style specification](../docs/figure_style.md).

Arial is not redistributed. On machines without Arial, set EA_ARIAL_FONT_PATH
to a regular Arial .ttf you are licensed to use, before running the notebook.
By default a missing Arial font is an error, not a silent substitution.

For a non-publication preview only, explicitly set
`EA_ALLOW_FONT_FALLBACK=1` before launch, or set ALLOW_FONT_FALLBACK=True in the
notebook configuration. The replacement is printed and recorded. Such an
output is not claimed to meet the Arial publication requirement. This explicit
mode allows hosted Linux CI to test the execution logic without bundling fonts.
The equal-TTC comparison follows the same explicit preview setting, adds a
non-Arial preview label to substituted figures, and records the actual font in
its run manifest. For a strict export despite an enabled preview environment,
pass `allow_font_fallback=False` to `run_comparison` or `plot_comparison`.

## Reading notebook outputs

The notebook compares raw EA and TTC on separate axes; it does not normalize
them into a common score or propose a universal risk threshold. Non-finite
values are omitted from curves. The original baseline's collision/undefined
value conventions are not corrected here; these synthetic examples avoid
already intersecting boxes. Headings and yaw rates define the nominal futures.
Read EA as the required relative-acceleration magnitude in m/s², rather than a
probability or a directly executable ego acceleration command.
