# Runnable examples

The original small examples below still work unchanged. New entry points:

- [comparison_case.py](comparison_case.py): recompute two synthetic equal-TTC
  states with different EA, and export full inputs, results, settings, and a
  Fig. 1-styled PNG. See [the comparison explanation](../docs/comparison_case.md).
- [from_dataframe.py](from_dataframe.py): map a custom long-form tracking table
  into the existing pairwise batch pipeline. See
  [the data-adaptation guide](../docs/bring_your_own_data.md).
- [Interactive notebook](../notebooks/ea_quickstart.ipynb): run both workflows,
  inspect causal yaw-rate provenance, and change an interaction manually.

New figures require regular Arial; see [font and color setup](../docs/figure_style.md).

Install from the repository root:

```bash
python -m pip install -e ".[batch,fast]"
python examples/provided_state.py
python examples/from_history.py
ea batch --input examples/data/provided.csv --yaw-rate-source provided
ea batch --input examples/data/without_yaw_rate.csv
ea batch --input examples/data/mixed_yaw_rate.csv
```

The first Python example prints `EA = 4.828 m/s^2`. The history example prints
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
