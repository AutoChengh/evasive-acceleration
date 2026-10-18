"""Optional, Fig. 1-aligned plotting helpers; the EA solver is not modified.

Matplotlib is imported only when a plotting function is used. Arial is required
by default. An explicitly opted-in fallback is for previews/CI, not publication.
"""

from __future__ import annotations

import math
import os
import warnings
from contextlib import contextmanager
from pathlib import Path


# Sampled from the vector objects of assets/figures/ea_framework_fig1.pdf.
# The stronger coral is its EA accent; the lighter colors are vehicle fills.
FIG1_COLORS = {
    "coral": "#D66A5B",
    "coral_light": "#E89F8C",
    "green": "#81B199",
    "green_light": "#A5C9B5",
    "gold": "#F1CB8D",
    "slate": "#7A8B8C",
    "ink": "#404040",
    "muted": "#A6A6A6",
    "background": "#FCF9F4",
    "grid": "#D9D9D9",
    "white": "#FFFFFF",
}


def _nonregular_face(face, ft2font) -> bool:
    """Support both legacy integer flags and Matplotlib 3.10+ StyleFlags."""
    flag_type = getattr(ft2font, "StyleFlags", None)
    if flag_type is not None:
        flags = flag_type(face.style_flags)
        mask = flag_type.BOLD | flag_type.ITALIC
    else:
        flags = face.style_flags
        mask = ft2font.BOLD | ft2font.ITALIC
    return bool(flags & mask)


def resolve_figure_font(*, allow_font_fallback: bool | None = None) -> dict:
    """Find regular Arial, or explicitly report a preview-only substitution.

    Set EA_ARIAL_FONT_PATH to an existing, legally obtained Arial .ttf file on
    machines without a system installation. No font is downloaded or bundled.
    """
    from matplotlib import font_manager, ft2font

    if allow_font_fallback is None:
        allow_font_fallback = os.environ.get("EA_ALLOW_FONT_FALLBACK", "") == "1"
    custom_path = os.environ.get("EA_ARIAL_FONT_PATH")
    if custom_path:
        path = Path(custom_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"EA_ARIAL_FONT_PATH does not exist: {path}")
        font_manager.fontManager.addfont(str(path))
        family = font_manager.FontProperties(fname=str(path)).get_name()
        if family.casefold() != "arial":
            raise ValueError("EA_ARIAL_FONT_PATH must point to an Arial font, not a substitute")
        if _nonregular_face(font_manager.get_font(str(path)), ft2font):
            raise ValueError("EA_ARIAL_FONT_PATH must be regular Arial, not bold or italic")
    else:
        family = "Arial"

    try:
        path = Path(font_manager.findfont(
            font_manager.FontProperties(family=family, weight="normal", style="normal"),
            fallback_to_default=False,
        )).resolve()
    except ValueError as exc:
        if not allow_font_fallback:
            raise RuntimeError(
                "Arial is required for Fig. 1-matched figures. Install Arial or set "
                "EA_ARIAL_FONT_PATH to your own regular Arial .ttf. For a labelled "
                "preview/CI only, explicitly set EA_ALLOW_FONT_FALLBACK=1."
            ) from exc
        path = None
        for candidate in ("Liberation Sans", "DejaVu Sans"):
            try:
                path = Path(font_manager.findfont(
                    font_manager.FontProperties(family=candidate, weight="normal"),
                    fallback_to_default=False,
                )).resolve()
                family = candidate
                break
            except ValueError:
                continue
        if path is None:
            raise RuntimeError("Neither Arial nor a supported preview font was found") from exc
        warnings.warn(
            f"Preview font substitution: {family}, not Arial. Do not use this output "
            "as a Fig. 1-matched publication figure.", UserWarning, stacklevel=2,
        )

    actual = font_manager.FontProperties(fname=str(path)).get_name()
    face = font_manager.get_font(str(path))
    if _nonregular_face(face, ft2font):
        raise RuntimeError(f"Expected a regular font face, found {face.style_name}: {path}")
    return {
        "requested_family": "Arial", "actual_family": actual,
        "path": str(path), "weight": "normal", "face_style": face.style_name,
        "fallback_used": actual.casefold() != "arial",
    }


@contextmanager
def figure_style(*, allow_font_fallback: bool | None = None):
    """Temporarily apply the shared palette and regular, never-bold type.

    Yields font provenance. It does not change the caller's global rcParams.
    """
    import matplotlib as mpl
    from cycler import cycler

    font = resolve_figure_font(allow_font_fallback=allow_font_fallback)
    family = font["actual_family"]
    settings = {
        "font.family": [family], "font.sans-serif": [family],
        "font.weight": "normal", "font.size": 10,
        "axes.titleweight": "normal", "axes.labelweight": "normal",
        "figure.titleweight": "normal", "axes.titlesize": 11,
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "legend.title_fontsize": 9,
        "text.color": FIG1_COLORS["ink"], "axes.labelcolor": FIG1_COLORS["ink"],
        "xtick.color": FIG1_COLORS["ink"], "ytick.color": FIG1_COLORS["ink"],
        "axes.edgecolor": FIG1_COLORS["muted"], "axes.linewidth": 0.8,
        "figure.facecolor": FIG1_COLORS["white"],
        "axes.facecolor": FIG1_COLORS["background"],
        "savefig.facecolor": FIG1_COLORS["white"],
        "grid.color": FIG1_COLORS["grid"], "grid.alpha": 0.7,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=[FIG1_COLORS[c] for c in (
            "coral", "green", "gold", "slate")]),
        "axes.unicode_minus": False, "figure.dpi": 110, "savefig.dpi": 180,
        "mathtext.fontset": "custom", "mathtext.rm": family,
        "mathtext.it": f"{family}:italic", "mathtext.bf": family,
    }
    with mpl.rc_context(settings):
        yield font


def plot_pair_state(agent_a, agent_b, *, ea_value=None, ttc_value=None):
    """Draw the current boxes and velocity arrows, not predicted/real futures."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import numpy as np

    agent_a.validate()
    agent_b.validate()
    with figure_style():
        figure, axis = plt.subplots(figsize=(8, 3.2), layout="constrained")
        all_corners = []
        for agent, label, color in (
            (agent_a, "A", FIG1_COLORS["green"]),
            (agent_b, "B", FIG1_COLORS["gold"]),
        ):
            local = np.array([
                [-agent.length / 2, -agent.width / 2],
                [agent.length / 2, -agent.width / 2],
                [agent.length / 2, agent.width / 2],
                [-agent.length / 2, agent.width / 2],
            ])
            c, s = math.cos(agent.heading), math.sin(agent.heading)
            corners = local @ np.array([[c, s], [-s, c]]) + [agent.x, agent.y]
            all_corners.extend(corners.tolist())
            axis.add_patch(Polygon(corners, facecolor=color,
                                   edgecolor=FIG1_COLORS["ink"], label=label, linewidth=1))
            axis.annotate("", xy=(agent.x + 0.35 * agent.speed * c,
                                   agent.y + 0.35 * agent.speed * s),
                          xytext=(agent.x, agent.y),
                          arrowprops={"arrowstyle": "->", "color": FIG1_COLORS["ink"]})
        corners = np.asarray(all_corners)
        axis.set_xlim(corners[:, 0].min() - 4, corners[:, 0].max() + 7)
        axis.set_ylim(corners[:, 1].min() - 3, corners[:, 1].max() + 3)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("X position (m)")
        axis.set_ylabel("Y position (m)")
        title = "Current geometry and velocity directions"
        if ea_value is not None:
            title += f" | EA = {ea_value:.3f} m/s\N{SUPERSCRIPT TWO}"
        if ttc_value is not None:
            title += f" | TTC = {ttc_value:.3f} s"
        axis.set_title(title)
        axis.legend(loc="upper right", frameon=False, ncol=2)
        axis.grid(True)
    return figure


def plot_risk_history(frame, *, time_column="Time (s)", output_path=None):
    """Plot computed EA and TTC on separate axes, without recomputing either.

    Missing/nonfinite values are not drawn. This is not a safety classifier.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    required = [time_column, "EA", "TTC"]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"Missing history columns: {missing}")
    times = pd.to_numeric(frame[time_column], errors="raise").to_numpy(dtype=float)
    if len(times) == 0 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("History timestamps must be nonempty, finite and strictly increasing")
    with figure_style():
        figure, axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True,
                                   layout="constrained")
        for axis, metric, color, unit in (
            (axes[0], "EA", "coral", "m/s\N{SUPERSCRIPT TWO}"),
            (axes[1], "TTC", "green", "s"),
        ):
            values = pd.to_numeric(frame[metric], errors="raise").to_numpy(dtype=float)
            values = np.where(np.isfinite(values), values, np.nan)
            axis.plot(times - times[0], values, color=FIG1_COLORS[color], linewidth=1.8,
                      marker="o", markersize=3, label=metric)
            axis.set_ylabel(f"{metric} ({unit})")
            axis.grid(True)
            axis.legend(frameon=False)
        axes[0].set_title("Same interaction, separate metric scales")
        axes[1].set_xlabel("Time since first paired observation (s)")
        # Constrained-layout rect is (left, bottom, width, height), not bounds.
        # Keep the title inside the canvas while reserving room for the note.
        figure.get_layout_engine().set(rect=(0, 0.055, 1, 0.935))
        figure.text(0.5, 0.012, "Non-finite values are omitted; EA is not a collision probability.",
                    ha="center", fontsize=8, color=FIG1_COLORS["slate"])
        if output_path is not None:
            target = Path(output_path)
            if target.suffix.lower() != ".png":
                raise ValueError("Use a .png output path for this example")
            target.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(target, dpi=180)
    return figure
