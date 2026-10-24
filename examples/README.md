# Runnable examples

The examples compute EA through the public interfaces.
Start with the recorded interaction:

```bash
python examples/real_interaction.py --output-dir outputs/real_interaction
```

The source data are included in `demo_data/`. The script recomputes every EA
value and produces a vehicle-motion view and risk history, using Fig. 1's
colors and regular Arial. [View the cases and outputs](../docs/real_interactions.md).

Other entry points:

- [from_dataframe.py](from_dataframe.py): map a custom long-form tracking table
  into the existing pairwise batch pipeline. See
  [the data-adaptation guide](../docs/bring_your_own_data.md).
- [Interactive notebook](../notebooks/ea_quickstart.ipynb): an optional API and
  tracking-table walkthrough with causal yaw-rate diagnostics.
- [comparison_case.py](comparison_case.py): an optional synthetic equal-TTC
  calculation check, separate from the real-data showcase.

New figures require regular Arial; see [font and color setup](../docs/figure_style.md).
For a labelled preview on a host without Arial, explicitly use
`python examples/comparison_case.py --allow-font-fallback` or set
`EA_ALLOW_FONT_FALLBACK=1`. The actual font and substitution are recorded in the
run JSON. Without either opt-in, Arial remains required; publication exports
must not use preview fallback.

Install from the repository root:

```bash
python -m pip install -e ".[batch]"
python -m evasive_acceleration.native_build
python examples/provided_state.py
python examples/from_history.py
ea batch --input examples/data/provided.csv --yaw-rate-source provided
ea batch --input examples/data/without_yaw_rate.csv
ea batch --input examples/data/mixed_yaw_rate.csv
```

The first Python example prints `EA = 4.911 m/s^2`. The history example prints
the estimated yaw rates and initialization diagnostics before EA. Imports use
the installed public package, so both scripts can be copied elsewhere.

The three CSVs describe the same synthetic, physically consistent approaching
pair over eight 10 Hz frames. Agent A travels at 10 m/s with a constant yaw
rate of 0.05 rad/s; Agent B travels straight in the opposite direction at
8 m/s. Their boxes have lengths/widths 4.5/1.8 m and 4.7/1.9 m. Positions are
box centers in Cartesian coordinates; headings are radians. Coordinates are
rounded to six decimal places for readability.

- `provided.csv`: both yaw-rate columns are complete.
- `without_yaw_rate.csv`: both yaw-rate columns are absent.
- `mixed_yaw_rate.csv`: some values are missing independently for each agent.

Estimated-only processing starts with an explicitly marked zero placeholder.
From the second row, the noiseless constant-rate history yields 0.05 rad/s
and 0 rad/s. The first estimated-only row can therefore differ from the
provided input. The mixed input starts with provided values for both agents.

Outputs are under `outputs/batch_results/<yaw-rate-policy>/`; each CSV has a
companion `.run.json`. Read the `EA_*source`, initialization, status, and error
columns when inspecting results. Existing outputs require `--overwrite`.
These small examples demonstrate interfaces; they are not paper experiment data.
