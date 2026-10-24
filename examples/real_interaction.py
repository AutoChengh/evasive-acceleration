"""Recompute and visualize the real inD interactions bundled with the repository.

python examples/real_interaction.py --output-dir outputs/real_interaction

The recorded paths provide visual context. Each EA value is computed only
from the two recorded states at that timestamp, using their supplied yaw rates
and the unchanged public API with its default settings.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

from core_ea import get_core_provenance
from evasive_acceleration import RoadUserState, build_compute_kwargs, compute_ea

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "crossing": {"input": "InD_05_tracks_266_267.csv",
                 "label": "inD recording 05 · tracks 266 / 267"},
    "low-speed": {"input": "InD_18_tracks_417_424.csv",
                  "label": "inD recording 18 · tracks 417 / 424"},
}
FIELDS = ("Position X (m)", "Position Y (m)", "Velocity (m/s)", "Heading",
          "Length (m)", "Width (m)", "Yawrate")


def state_at(row, prefix=""):
    return RoadUserState.from_sequence([float(row[prefix + name]) for name in FIELDS])


def evaluate_frames(frame):
    """Compute one pair per row, never use later rows to construct a state."""
    import numpy as np

    required = {"Time (s)", *FIELDS, *("2_" + name for name in FIELDS)}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing recorded-state columns: {sorted(missing)}")
    if frame.empty or not np.isfinite(frame[sorted(required)].to_numpy(float)).all():
        raise ValueError("Recorded states and timestamps must be finite and nonempty.")
    if np.any(np.diff(frame["Time (s)"].to_numpy(float)) <= 0):
        raise ValueError("The interaction must be in strictly increasing time order.")
    for column in ("Vehicle Number", "2_Vehicle Number"):
        if column in frame and frame[column].nunique(dropna=False) != 1:
            raise ValueError("One continuous interaction pair is required.")
    results = frame.copy()
    results["EA_m_s2"] = [float(compute_ea(state_at(row), state_at(row, "2_")))
                          for row in frame.to_dict(orient="records")]
    results["Elapsed time (s)"] = frame["Time (s)"].to_numpy(float) - float(frame["Time (s)"].iloc[0])
    return results


def figure_window(results):
    """A fixed +/-2 s display window around the peak; exported data stay full."""
    elapsed=results["Elapsed time (s)"].to_numpy(float)
    peak_time=float(elapsed[int(results["EA_m_s2"].to_numpy(float).argmax())])
    return max(float(elapsed[0]),peak_time-2),min(float(elapsed[-1]),peak_time+2)


def plot_case(results, case="crossing", *, allow_font_fallback=None):
    """Show recorded paths, both current boxes at the peak, and the EA history."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from evasive_acceleration.plotting import FIG1_COLORS, figure_style

    risk = results["EA_m_s2"].to_numpy(float)
    if not np.isfinite(risk).all() or np.any(risk < 0):
        raise ValueError("This finite-history figure requires finite nonnegative EA values.")
    peak = int(np.argmax(risk))
    row = results.iloc[peak]
    a, b = state_at(row), state_at(row, "2_")
    origin = np.array([(a.x + b.x) / 2, (a.y + b.y) / 2])
    colors = FIG1_COLORS
    with figure_style(allow_font_fallback=allow_font_fallback) as font:
        fig, (scene, history) = plt.subplots(1, 2, figsize=(11.6, 3.4),
            gridspec_kw={"width_ratios": [1, 1.35]})
        fig.subplots_adjust(left=.065, right=.975, bottom=.23, top=.91, wspace=.34)
        for prefix, state, color, label in (("", a, colors["green"], "Vehicle A"),
                                          ("2_", b, colors["gold"], "Vehicle B")):
            xy = results[[prefix+"Position X (m)", prefix+"Position Y (m)"]].to_numpy(float) - origin
            scene.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2, label=label)
            local = np.array([state.x, state.y]) - origin
            corners = np.array([[-1,-1],[1,-1],[1,1],[-1,1]], float) * [state.length/2, state.width/2]
            c, s = np.cos(state.heading), np.sin(state.heading)
            corners = corners @ np.array([[c,s],[-s,c]]) + local
            scene.add_patch(Polygon(corners, facecolor=color, edgecolor=colors["ink"], linewidth=.8, zorder=4))
            scene.annotate("", xy=local+np.array([c,s])*3, xytext=local,
                           arrowprops={"arrowstyle":"->", "color":colors["ink"], "linewidth":1}, zorder=5)
        half_width=max(20., abs(a.x-b.x)/2+8., abs(a.y-b.y)/2*1.4+8.)
        scene.set_xlim(-half_width,half_width)
        scene.set_ylim(-half_width/1.4,half_width/1.4)
        scene.set_aspect("equal", adjustable="box")
        scene.set_title("Vehicle positions at peak EA", fontsize=10, loc="left", pad=10)
        scene.set_xlabel("Position X (m)");scene.set_ylabel("Position Y (m)")
        scene.grid(alpha=.45);scene.margins(.12)
        scene.legend(loc="best", frameon=False, fontsize=9)
        elapsed = results["Elapsed time (s)"].to_numpy(float)
        history.plot(elapsed, risk, color=colors["coral"], linewidth=2)
        history.fill_between(elapsed, 0, risk, color=colors["coral"], alpha=.12)
        history.scatter([elapsed[peak]], [risk[peak]], color=colors["coral"], s=32, zorder=5)
        history.axvline(elapsed[peak], color=colors["coral"], linewidth=.8, linestyle="--", alpha=.7)
        history.set_title(f"Peak EA = {risk[peak]:.3f} m/s² at {elapsed[peak]:.2f} s",
                          fontsize=11, loc="left", pad=10)
        history.set_xlabel("Time (s)");history.set_ylabel("EA (m/s²)")
        history.set_xlim(*figure_window(results));history.set_ylim(0, max(.05,float(risk.max())*1.18))
        history.grid(alpha=.45)
        fig.text(.065, .055, CASES[case]["label"],
                 fontsize=9, color=colors["ink"])
        if font["fallback_used"]:
            fig.text(.975,.015,"Preview font: "+font["actual_family"],ha="right",fontsize=8)
    return fig


def run_real_cases(output_dir, *, case="crossing", make_plot=True, allow_font_fallback=None):
    """Create a fresh result directory; inputs and previous runs cannot be replaced."""
    import pandas as pd

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Choose a new output directory; existing inputs and results are protected.")
    selected = list(CASES) if case == "all" else [case]
    if any(name not in CASES for name in selected):
        raise ValueError("Unknown recorded interaction.")
    results, payloads, metadata = {}, {}, {}
    for name in selected:
        source = ROOT / "demo_data" / CASES[name]["input"]
        result = evaluate_frames(pd.read_csv(source))
        results[name] = result
        payloads[name+"_results.csv"] = result.to_csv(index=False).encode("utf-8")
        settings = build_compute_kwargs(state_at(result.iloc[0]), state_at(result.iloc[0], "2_"))
        metadata[name] = {"source": source.relative_to(ROOT).as_posix(),
            "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "synthetic": False,
            "observations": len(result), "peak_ea_m_s2": float(result.EA_m_s2.max()),
            "figure_window_elapsed_s": list(figure_window(result)),
            "solver_settings": {key:settings[key] for key in ("a_max","tol","T_total","size_scale")},
            "yaw_rate_source":"provided", "future_observations_used_for_ea":False}
        if make_plot:
            import matplotlib.pyplot as plt
            from evasive_acceleration.plotting import resolve_figure_font
            metadata[name]["font"] = resolve_figure_font(allow_font_fallback=allow_font_fallback)
            figure = plot_case(result,name,allow_font_fallback=allow_font_fallback)
            try:
                image=io.BytesIO();figure.savefig(image,format="png",dpi=180,facecolor="white")
                payloads[name+"_ea.png"]=image.getvalue()
            finally:
                plt.close(figure)
    payloads["run.json"]=(json.dumps({"cases":metadata,"core_provenance":get_core_provenance()},indent=2)+"\n").encode("utf-8")
    # All computation/font checks/rendering finish before this exclusive mkdir.
    destination.mkdir(parents=True,exist_ok=False)
    for filename,content in payloads.items():
        with (destination/filename).open("xb") as stream:stream.write(content)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",choices=[*CASES,"all"],default="crossing")
    parser.add_argument("--output-dir",type=Path,default=Path("outputs/real_interaction"))
    parser.add_argument("--no-plot",action="store_true")
    args=parser.parse_args()
    try:
        results=run_real_cases(args.output_dir,case=args.case,make_plot=not args.no_plot)
    except (ValueError,OSError,RuntimeError,ImportError) as error:
        parser.exit(1,f"Real-interaction example failed: {error}\n")
    for name,result in results.items():
        print(f"{name}: {len(result)} observations, peak EA = {result.EA_m_s2.max():.3f} m/s^2")
    print(f"Saved to {args.output_dir.resolve()}")


if __name__=="__main__":main()
