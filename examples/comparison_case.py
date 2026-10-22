"""Recompute two synthetic straight-following cases with equal TTC.

Run from a source checkout after installing ``.[viz]`` and building the native component::

    python examples/comparison_case.py --output-dir outputs/comparison_case

EA is computed through the public API; the comparison metrics use the
historical baseline functions. Both use their default numerical settings.
These inputs are illustrations, not paper experiment data.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from baseline_risk_metrics import compute_real_time_metrics_dict
from core_ea import CORE_ID, get_core_provenance
from evasive_acceleration import (
    RoadUserState,
    __version__,
    build_compute_kwargs,
    compute_ea,
)

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure


DEFAULT_CASES_CSV = Path(__file__).resolve().parent / "data" / "comparison_cases.csv"
STATE_FIELDS = (
    "x_m", "y_m", "speed_m_s", "heading_rad", "length_m", "width_m", "yaw_rate_rad_s"
)
SOLVER_FIELDS = ("a_max", "tol", "T_total", "size_scale")


def _state(record, prefix: str) -> RoadUserState:
    values = [float(record[f"{prefix}_{field}"]) for field in STATE_FIELDS]
    state = RoadUserState.from_sequence(values)
    if state.speed < 0:
        raise ValueError("Example speed magnitudes must be non-negative.")
    return state


def _validate_straight_following(a: RoadUserState, b: RoadUserState) -> None:
    """Keep the accompanying gap/closing-speed explanation valid."""
    if not all(math.isclose(value, 0.0, abs_tol=1e-12)
               for value in (a.heading, b.heading, a.yaw_rate, b.yaw_rate, b.y - a.y)):
        raise ValueError("Comparison cases must be aligned, straight, and have zero yaw rate.")
    clearance = b.x - a.x - (a.length + b.length) / 2.0
    if clearance <= 0 or a.speed <= b.speed:
        raise ValueError("Comparison cases must be separated, with A approaching B from behind.")


def _validate_equal_ttc(results: "pd.DataFrame") -> None:
    values = [float(value) for value in results["TTC_s"]]
    if len(values) != 2 or not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("The equal-TTC comparison requires two finite positive TTC values.")
    if not math.isclose(values[0], values[1], rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            "This example requires equal TTC values "
            f"(relative tolerance 1e-9); received {values[0]:.12g} s and {values[1]:.12g} s."
        )


def evaluate_cases(cases: "pd.DataFrame") -> "pd.DataFrame":
    """Return complete input states and computed metrics for independent rows."""
    import pandas as pd

    required = {"case_id", "provenance", "description"}
    required.update(f"{prefix}_{field}" for prefix in ("a", "b") for field in STATE_FIELDS)
    missing = sorted(required - set(cases.columns))
    if missing:
        raise ValueError(f"Comparison CSV is missing columns: {missing}")
    if len(cases) != 2 or cases["case_id"].isna().any() or not cases["case_id"].is_unique:
        raise ValueError("The comparison must contain exactly two uniquely named cases.")
    if not cases["provenance"].eq("synthetic").all():
        raise ValueError("These illustrative inputs must be explicitly labeled synthetic.")

    rows = []
    for record in cases.to_dict(orient="records"):
        a, b = _state(record, "a"), _state(record, "b")
        _validate_straight_following(a, b)
        # The existing baseline expects the first six fields, without yaw rate.
        baseline = compute_real_time_metrics_dict(*a.as_tuple()[:6], *b.as_tuple()[:6])
        row = dict(record)
        row.update(
            clearance_m=b.x - a.x - (a.length + b.length) / 2.0,
            closing_speed_m_s=a.speed - b.speed,
            EA_m_s2=float(compute_ea(a, b)),
            TTC_s=float(baseline["TTC"]),
            DRAC_m_s2=float(baseline["DRAC"]),
        )
        if not all(math.isfinite(row[key]) and row[key] > 0
                   for key in ("EA_m_s2", "TTC_s", "clearance_m", "closing_speed_m_s")):
            raise ValueError(f"Case {record['case_id']!r} did not produce finite positive metrics.")
        rows.append(row)
    results = pd.DataFrame(rows)
    _validate_equal_ttc(results)
    return results


def plot_comparison(
    results: "pd.DataFrame",
    output_path: str | Path | None = None,
    *,
    allow_font_fallback: bool | None = None,
) -> "Figure":
    """Plot initial geometry and separate EA/TTC axes; return an unclosed figure.

    No GUI is opened. A notebook can display the returned figure and close it
    when finished. Plotting dependencies are imported only here. Arial remains
    required unless preview fallback is explicitly enabled through this argument
    or EA_ALLOW_FONT_FALLBACK=1. Pass False to require Arial even in preview mode.
    """
    _validate_equal_ttc(results)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from evasive_acceleration.plotting import FIG1_COLORS, figure_style

    if len(results) != 2:
        raise ValueError("Plotting expects the two computed comparison cases.")
    if output_path is not None and Path(output_path).suffix.lower() != ".png":
        raise ValueError("This example exports PNG only; use a .png output path.")

    colors = FIG1_COLORS
    with figure_style(allow_font_fallback=allow_font_fallback) as font:
        fig = plt.figure(figsize=(11.4, 6.4), facecolor="white")
        grid = fig.add_gridspec(2, 2, width_ratios=[1.65, 1.0])
        geometry_axes = [fig.add_subplot(grid[index, 0]) for index in range(2)]
        ea_axis = fig.add_subplot(grid[0, 1])
        ttc_axis = fig.add_subplot(grid[1, 1])
        fig.subplots_adjust(left=0.065, right=0.98, top=0.84, bottom=0.15,
                            hspace=0.62, wspace=0.32)
        fig.suptitle("Synthetic comparison: equal TTC, different EA", y=0.975,
                     fontsize=16, fontweight="normal", color=colors["ink"])
        fig.text(0.065, 0.91, "Independent instantaneous states; all headings and yaw rates are zero.",
                 fontsize=10, color=colors["muted"], fontweight="normal")
        if font["fallback_used"]:
            fig.text(0.98, 0.015, f"Preview font: {font['actual_family']} (not Arial)",
                     ha="right", fontsize=8, color=colors["muted"], fontweight="normal")

        maximum_x = max(float(row.b_x_m + row.b_length_m / 2.0)
                        for row in results.itertuples()) + 3.0
        labels = ["Case 1", "Case 2"]
        for index, (axis, (_, row)) in enumerate(zip(geometry_axes, results.iterrows())):
            a, b = _state(row, "a"), _state(row, "b")
            axis.set_facecolor(colors["background"])
            axis.axhline(0, color=colors["grid"], linestyle="--", linewidth=1, zorder=0)
            for state, color, name in ((a, colors["green"], "Follower A"),
                                       (b, colors["gold"], "Leader B")):
                axis.add_patch(Rectangle((state.x - state.length / 2.0,
                                          state.y - state.width / 2.0),
                                         state.length, state.width, facecolor=color,
                                         edgecolor=colors["ink"], linewidth=0.8))
                axis.annotate("", xy=(state.x + 0.26 * state.speed, state.y),
                              xytext=(state.x, state.y),
                              arrowprops={"arrowstyle": "->", "color": colors["ink"],
                                          "linewidth": 1.0})
                axis.text(state.x, 1.4, f"{name}\n{state.speed:g} m/s", ha="center",
                          va="bottom", fontsize=9, fontweight="normal", color=colors["ink"])
            first_edge = a.x + a.length / 2.0
            second_edge = b.x - b.length / 2.0
            axis.annotate("", xy=(second_edge, -1.7), xytext=(first_edge, -1.7),
                          arrowprops={"arrowstyle": "<->", "color": colors["muted"],
                                      "linewidth": 0.9})
            axis.text((first_edge + second_edge) / 2.0, -2.25,
                      f"Clearance = {row['clearance_m']:g} m", ha="center", va="top",
                      fontsize=9, color=colors["muted"], fontweight="normal")
            axis.set_title(f"{labels[index]}: closing speed = {row['closing_speed_m_s']:g} m/s",
                           loc="left", fontsize=11, fontweight="normal", color=colors["ink"])
            axis.set_xlim(-4.0, maximum_x)
            axis.set_ylim(-3.5, 3.2)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("Longitudinal position (m)", fontsize=9)
            axis.set_ylabel("Lateral position (m)", fontsize=9)
            axis.tick_params(labelsize=8)

        for axis, column, color, title, unit in (
            (ea_axis, "EA_m_s2", colors["coral"], "Required relative acceleration", "EA (m/s²)"),
            (ttc_axis, "TTC_s", colors["green"], "Time to contact under CV", "TTC (s)"),
        ):
            values = results[column].to_numpy(dtype=float)
            bars = axis.bar(labels, values, width=0.54, color=color,
                            edgecolor=colors["ink"], linewidth=0.7)
            axis.set_ylim(0, float(max(values)) * 1.3)
            axis.set_ylabel(unit, fontsize=10)
            axis.set_title(title, loc="left", fontsize=11, fontweight="normal", color=colors["ink"])
            axis.set_axisbelow(True)
            axis.grid(axis="y", color=colors["grid"], linewidth=0.6)
            axis.tick_params(labelsize=9)
            for bar, value in zip(bars, values):
                axis.text(bar.get_x() + bar.get_width() / 2.0, value + max(values) * 0.045,
                          f"{value:.3f}" if column == "EA_m_s2" else f"{value:.2f}",
                          ha="center", va="bottom", fontsize=10,
                          fontweight="normal", color=colors["ink"])
        fig.text(0.065, 0.055,
                 "Default 10 s EA horizon; unconstrained two-dimensional relative acceleration.\n"
                 "Synthetic illustration only: no empirical ranking, universal threshold, or road-feasibility claim.",
                 fontsize=9, color=colors["muted"], fontweight="normal", va="bottom")
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(destination, dpi=180, facecolor="white")
    return fig


def run_comparison(
    output_dir: str | Path,
    *,
    cases_csv: str | Path | None = None,
    make_plot: bool = True,
    overwrite: bool = False,
    allow_font_fallback: bool | None = None,
) -> "pd.DataFrame":
    """Compute equal-TTC cases and export only after all data/font/plot checks.

    Existing output files require ``overwrite=True``. Inputs are always
    protected, and all figure rendering finishes in memory before any output
    CSV or manifest is changed. Preview fallback requires an explicit True or
    EA_ALLOW_FONT_FALLBACK=1; its actual font is recorded in the run manifest.
    """
    import pandas as pd

    source = Path(cases_csv) if cases_csv is not None else DEFAULT_CASES_CSV
    destination = Path(output_dir)
    filenames = ("comparison_inputs.csv", "comparison_results.csv",
                 "comparison_case.png", "comparison_run.json")
    planned = [destination / name for index, name in enumerate(filenames)
               if index != 2 or make_plot]
    for target in planned:
        if target.resolve() == source.resolve() or (target.is_file() and target.samefile(source)):
            raise ValueError("The output directory must not replace the supplied input file.")
        if target.exists() or target.is_symlink():
            if not target.is_file():
                raise FileExistsError(f"Output target is not a regular file: {target}")
            if not overwrite:
                raise FileExistsError(f"Output already exists: {target}. Use --overwrite to replace it.")

    cases = pd.read_csv(source)
    results = evaluate_cases(cases)
    first = cases.iloc[0]
    compute_kwargs = build_compute_kwargs(_state(first, "a"), _state(first, "b"))
    metadata = {
        "synthetic": True,
        "package_version": __version__,
        "core_id": CORE_ID,
        "core_provenance": get_core_provenance(),
        "input_file": str(source.resolve()),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "solver_settings": {key: compute_kwargs[key] for key in SOLVER_FIELDS},
        "ea_aggregation": "arithmetic mean of CTCT, CTCV, CVCT, CVCV",
        "ttc_source": "baseline_risk_metrics.compute_real_time_metrics_dict",
        "assumptions": [
            "Independent synthetic states; no dataset observations or future truth.",
            "Positions are box centers; heading and yaw rate are zero for both agents.",
            "TTC = positive bumper clearance / closing speed for these aligned CV cases.",
            "EA permits unconstrained two-dimensional relative acceleration; no road boundaries.",
            "No universal risk threshold or empirical superiority is inferred from two examples.",
        ],
        "outputs": {"inputs": filenames[0], "results": filenames[1],
                    "figure": filenames[2] if make_plot else None},
        "font_provenance": None,
    }
    if make_plot:
        import matplotlib.pyplot as plt
        from evasive_acceleration.plotting import resolve_figure_font

        metadata["font_provenance"] = resolve_figure_font(
            allow_font_fallback=allow_font_fallback
        )
        figure = plot_comparison(results, allow_font_fallback=allow_font_fallback)
        try:
            image = io.BytesIO()
            figure.savefig(image, format="png", dpi=180, facecolor="white")
            png_bytes = image.getvalue()
        finally:
            plt.close(figure)
    payloads = {
        filenames[0]: cases.to_csv(index=False).encode("utf-8"),
        filenames[1]: results.to_csv(index=False).encode("utf-8"),
        filenames[3]: (json.dumps(metadata, indent=2) + "\n").encode("utf-8"),
    }
    if make_plot:
        payloads[filenames[2]] = png_bytes
    _write_artifacts(destination, payloads, overwrite=overwrite)
    return results


def _write_artifacts(destination: Path, payloads: dict[str, bytes], *, overwrite: bool) -> None:
    """Reserve new targets exclusively, or replace prepared files individually."""
    destination.mkdir(parents=True, exist_ok=True)
    if overwrite:
        # Prepare all bytes before replacing any old artifact. Each replacement
        # is atomic; this is not a multi-file filesystem transaction.
        with TemporaryDirectory(prefix=".ea-comparison-", dir=destination) as staging:
            staged = Path(staging)
            for name, payload in payloads.items():
                (staged / name).write_bytes(payload)
            for name in payloads:
                os.replace(staged / name, destination / name)
        return

    created = []
    try:
        with ExitStack() as stack:
            streams = {}
            # Reserve every filename first. A concurrent creation cannot be
            # overwritten, and a failed reservation removes only our files.
            for name in payloads:
                target = destination / name
                streams[name] = stack.enter_context(target.open("xb"))
                created.append(target)
            for name, payload in payloads.items():
                streams[name].write(payload)
    except BaseException:
        for target in created:
            target.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison_case"))
    parser.add_argument("--input", type=Path, default=DEFAULT_CASES_CSV,
                        help="Two synthetic straight-following input states.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing comparison outputs; input files remain protected.")
    parser.add_argument("--allow-font-fallback", action="store_true", default=None,
                        help="Explicitly allow a labelled non-Arial preview; never for publication. "
                             "EA_ALLOW_FONT_FALLBACK=1 also enables preview mode.")
    args = parser.parse_args(argv)
    try:
        results = run_comparison(
            args.output_dir, cases_csv=args.input, overwrite=args.overwrite,
            allow_font_fallback=args.allow_font_fallback,
        )
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        parser.exit(1, f"Comparison failed: {error}\n")
    print(results[["case_id", "clearance_m", "closing_speed_m_s", "TTC_s", "EA_m_s2"]]
          .to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Synthetic comparison saved to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
