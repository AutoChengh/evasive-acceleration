# Interactive EA quickstart

The notebook reuses the same public API, synthetic comparison, custom-data
adapter, and paper-version finite-horizon EA core as the command-line examples. It is not a
reproduction of the paper's statistical experiments.

## Local Jupyter

From a source checkout, install into a virtual environment:

```bash
python -m pip install -e ".[notebook,fast]"
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

After these files are published to your branch/fork, open
`notebooks/ea_quickstart.ipynb` in Colab through its GitHub notebook picker.
This independent evaluation repository is initially private. Anonymous Colab
cloning is disabled by default; for now, use the local notebook or upload an
authorized source checkout and run from its directory. Do not paste a GitHub
token into a notebook. Once the repository is made public, explicitly enable
ALLOW_PUBLIC_BOOTSTRAP in the setup cell. Set SOURCE_REVISION to the published
branch/tag/commit containing this notebook if it is not yet on main. Record
the printed checkout commit with any reproduced result.
Local notebook execution never clones or installs packages automatically.

This is an optional hosted route, not a claim that an unpublished local
notebook is already available at a public URL.

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

## Reading notebook outputs

The notebook compares raw EA and TTC on separate axes; it does not normalize
them into a common score or propose a universal risk threshold. Non-finite
values are omitted from curves. The original baseline's collision/undefined
value conventions are not corrected here; these synthetic examples avoid
already intersecting boxes. Headings and yaw rates define the nominal futures.
Read EA as the required relative-acceleration magnitude in m/s², rather than a
probability or a directly executable ego acceleration command.
