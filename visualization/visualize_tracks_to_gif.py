#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_tracks_to_gif.py

Generate EA-colored GIF visualizations for interaction CSV files.

This script assumes the following project layout:

project_root/
├─ demo_data/
│  ├─ *_EA.csv
├─ visualization/
│  └─ visualize_tracks_to_gif.py

Input:
- CSV files are read from the sibling folder: ../demo_data/
- Only files whose names end with "_EA.csv" are processed

Output:
- GIF files are saved to: ./gif_visualizations_by_EA/
  i.e., inside the same "visualization" directory as this script

Main features:
1. Read all matching CSV files from the demo_data folder
2. Generate one GIF per CSV, colored by the EA column
3. Show only the current positions of the two agents in each frame
4. Optionally draw historical center trajectories
5. Optionally draw speed arrows and speed text
6. Preserve real time spacing by using frame durations derived from timestamps
7. Use a white, non-transparent background
8. Support slicing the sampled frame sequence by ratio
9. Compute axis limits only from the displayed frame range
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon
from matplotlib.ticker import MultipleLocator

from PIL import Image


# ============================================================================
# Paths
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_DIR = PROJECT_ROOT / "demo_data"
OUTPUT_DIR = SCRIPT_DIR / "gif_visualizations_by_EA"


# ============================================================================
# Visualization settings
# ============================================================================

FRAME_STEP = 1
FIGSIZE = (7.2, 6.0)
FIG_DPI = 220

# Slice the sampled frame sequence by ratio.
# Examples:
# [0.0, 1.0] -> full sequence
# [0.0, 0.6] -> first 60%
# [0.2, 0.8] -> middle 60%
VISUALIZE_TIME_RANGE = [0.0, 1.0]


# Axis margins computed from the displayed frames only.
X_MARGIN_LEFT = 5.0
X_MARGIN_RIGHT = 5.0
Y_MARGIN_BOTTOM = 5.0
Y_MARGIN_TOP = 5.0

# GIF export settings.
GIF_LOOP = 0
GIF_OPTIMIZE = False
GIF_DISPOSAL = 2
MIN_GIF_DURATION_MS = 20
FALLBACK_DURATION_MS = 100

# Historical center trajectory.
DRAW_CENTER_TRAJECTORY = True
HISTORY_TRAJ_COLOR = "#8A8A8A"
HISTORY_TRAJ_LINEWIDTH = 1.0
HISTORY_TRAJ_ALPHA = 0.60

# Vehicle boundary styling.
EDGE_COLOR = "#2B2B2B"
EDGE_WIDTH = 0.70

# Speed arrow settings.
DRAW_SPEED_ARROW = True
SHOW_SPEED_TEXT = False

ARROW_COLOR = "#8C8C8C"
ARROW_ALPHA = 0.78
ARROW_LINEWIDTH = 1.3
ARROW_HEAD_WIDTH = 0.55
ARROW_HEAD_LENGTH = 0.75
ARROW_LENGTH_SCALE = 0.38
SPEED_TEXT_OFFSET = 0.65
SPEED_TEXT_BBOX_ALPHA = 0.60


# ============================================================================
# Matplotlib style
# ============================================================================

rcParams["font.sans-serif"] = ["Arial"]
rcParams["font.family"] = "sans-serif"
rcParams["axes.unicode_minus"] = False

rcParams["axes.linewidth"] = 1.2
rcParams["xtick.major.width"] = 1.2
rcParams["ytick.major.width"] = 1.2
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"

rcParams["axes.labelsize"] = 16
rcParams["xtick.labelsize"] = 13
rcParams["ytick.labelsize"] = 13
rcParams["axes.titlesize"] = 16
rcParams["figure.titlesize"] = 15

GRID_COLOR = "#CFCFCF"
GRID_LINESTYLE = "--"
GRID_LINEWIDTH = 0.8
GRID_ALPHA = 0.62


# ============================================================================
# Colormap
# ============================================================================

RISK_YELLOW = "#FBE38A"
RISK_ORANGE = "#F29E4C"
RISK_RED = "#C73E1D"

RISK_CMAP = LinearSegmentedColormap.from_list(
    "RiskYellowOrangeRed",
    [RISK_YELLOW, RISK_ORANGE, RISK_RED],
    N=256,
)

EA_MIN = 0.0
EA_MAX = 1.5
EA_NORM = Normalize(vmin=EA_MIN, vmax=EA_MAX)


# ============================================================================
# Utility functions
# ============================================================================

def safe_float(value, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def choose_col(df_cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """Return the first matching column name from a list of candidates."""
    for col in candidates:
        if col in df_cols:
            return col
    return None


def clip_val(value: float, xmin: float, xmax: float) -> float:
    """Clip a scalar to the closed interval [xmin, xmax]."""
    return max(xmin, min(xmax, value))


def ea_to_rgba(ea_value, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    """Map EA value to RGBA color."""
    ea = clip_val(safe_float(ea_value, 0.0), EA_MIN, EA_MAX)
    rgba = RISK_CMAP(EA_NORM(ea))
    return (rgba[0], rgba[1], rgba[2], alpha)


# ============================================================================
# Geometry helpers
# ============================================================================

def build_vehicle_polygon(
    x: float,
    y: float,
    heading: float,
    length: float,
    width: float,
) -> np.ndarray:
    """Build a 4-corner polygon for an oriented rectangular vehicle."""
    dx = length / 2.0
    dy = width / 2.0
    c = math.cos(heading)
    s = math.sin(heading)

    corners = np.array([
        [-dx * c - dy * s, -dx * s + dy * c],
        [ dx * c - dy * s,  dx * s + dy * c],
        [ dx * c + dy * s,  dx * s - dy * c],
        [-dx * c + dy * s, -dx * s - dy * c],
    ]) + np.array([x, y])

    return corners


# ============================================================================
# Plot styling
# ============================================================================

def style_axis(ax, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    """Apply consistent axis styling."""
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(rcParams["axes.linewidth"])
    ax.spines["bottom"].set_linewidth(rcParams["axes.linewidth"])

    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.tick_params(axis="both", which="major", pad=4)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(20))

    ax.set_xlabel("Position X (m)", labelpad=8)
    ax.set_ylabel("Position Y (m)", labelpad=8)

    ax.grid(
        True,
        linestyle=GRID_LINESTYLE,
        linewidth=GRID_LINEWIDTH,
        color=GRID_COLOR,
        alpha=GRID_ALPHA,
    )


# ============================================================================
# Drawing helpers
# ============================================================================

def draw_center_trajectory(
    ax,
    frames_history: Sequence[pd.Series],
    posx1_col: str,
    posy1_col: str,
    posx2_col: str,
    posy2_col: str,
    zorder: int = 5,
    alpha: float = HISTORY_TRAJ_ALPHA,
    line_color: str = HISTORY_TRAJ_COLOR,
) -> None:
    """Draw historical center trajectories for both agents."""
    xs1 = np.array([safe_float(r.get(posx1_col, np.nan), np.nan) for r in frames_history], dtype=float)
    ys1 = np.array([safe_float(r.get(posy1_col, np.nan), np.nan) for r in frames_history], dtype=float)
    xs2 = np.array([safe_float(r.get(posx2_col, np.nan), np.nan) for r in frames_history], dtype=float)
    ys2 = np.array([safe_float(r.get(posy2_col, np.nan), np.nan) for r in frames_history], dtype=float)

    valid1 = np.isfinite(xs1) & np.isfinite(ys1)
    valid2 = np.isfinite(xs2) & np.isfinite(ys2)

    if np.sum(valid1) >= 2:
        ax.plot(
            xs1[valid1],
            ys1[valid1],
            color=line_color,
            linewidth=HISTORY_TRAJ_LINEWIDTH,
            alpha=alpha,
            zorder=zorder,
        )

    if np.sum(valid2) >= 2:
        ax.plot(
            xs2[valid2],
            ys2[valid2],
            color=line_color,
            linewidth=HISTORY_TRAJ_LINEWIDTH,
            alpha=alpha,
            zorder=zorder,
        )


def draw_single_frame_on_axis(
    ax,
    row: pd.Series,
    value_to_rgba_func: Callable,
    value_col: str,
    posx1_col: str,
    posy1_col: str,
    posx2_col: str,
    posy2_col: str,
    heading1_col: Optional[str],
    heading2_col: Optional[str],
    len1_col: Optional[str],
    wid1_col: Optional[str],
    len2_col: Optional[str],
    wid2_col: Optional[str],
) -> None:
    """Draw the two agents for the current frame."""
    raw_value = row.get(value_col, np.nan)
    veh_color = value_to_rgba_func(raw_value, alpha=1.0)

    x1 = safe_float(row.get(posx1_col, np.nan), np.nan)
    y1 = safe_float(row.get(posy1_col, np.nan), np.nan)
    a1 = safe_float(row.get(heading1_col, np.nan), np.nan) if heading1_col else np.nan
    l1 = safe_float(row.get(len1_col, 0.0), 0.0) if len1_col else 0.0
    w1 = safe_float(row.get(wid1_col, 0.0), 0.0) if wid1_col else 0.0

    x2 = safe_float(row.get(posx2_col, np.nan), np.nan)
    y2 = safe_float(row.get(posy2_col, np.nan), np.nan)
    a2 = safe_float(row.get(heading2_col, np.nan), np.nan) if heading2_col else np.nan
    l2 = safe_float(row.get(len2_col, 0.0), 0.0) if len2_col else 0.0
    w2 = safe_float(row.get(wid2_col, 0.0), 0.0) if wid2_col else 0.0

    if np.isfinite(x1) and np.isfinite(y1) and np.isfinite(a1) and l1 > 0 and w1 > 0:
        poly1 = build_vehicle_polygon(x1, y1, a1, l1, w1)
        ax.add_patch(Polygon(
            poly1,
            closed=True,
            facecolor=veh_color,
            edgecolor=EDGE_COLOR,
            linewidth=EDGE_WIDTH,
            alpha=veh_color[3],
            zorder=20,
        ))

    if np.isfinite(x2) and np.isfinite(y2) and np.isfinite(a2) and l2 > 0 and w2 > 0:
        poly2 = build_vehicle_polygon(x2, y2, a2, l2, w2)
        ax.add_patch(Polygon(
            poly2,
            closed=True,
            facecolor=veh_color,
            edgecolor=EDGE_COLOR,
            linewidth=EDGE_WIDTH,
            alpha=veh_color[3],
            zorder=20,
        ))


def draw_speed_heading_annotation(
    ax,
    row: pd.Series,
    posx1_col: str,
    posy1_col: str,
    posx2_col: str,
    posy2_col: str,
    heading1_col: Optional[str],
    heading2_col: Optional[str],
    vel1_col: Optional[str],
    vel2_col: Optional[str],
    show_speed_text: bool = True,
) -> None:
    """Draw speed arrows and optional speed labels for both agents."""
    x1 = safe_float(row.get(posx1_col, np.nan), np.nan)
    y1 = safe_float(row.get(posy1_col, np.nan), np.nan)
    a1 = safe_float(row.get(heading1_col, np.nan), np.nan) if heading1_col else np.nan
    v1 = safe_float(row.get(vel1_col, np.nan), np.nan) if vel1_col else np.nan

    x2 = safe_float(row.get(posx2_col, np.nan), np.nan)
    y2 = safe_float(row.get(posy2_col, np.nan), np.nan)
    a2 = safe_float(row.get(heading2_col, np.nan), np.nan) if heading2_col else np.nan
    v2 = safe_float(row.get(vel2_col, np.nan), np.nan) if vel2_col else np.nan

    agent_data = [(x1, y1, a1, v1), (x2, y2, a2, v2)]

    for x, y, heading, speed in agent_data:
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(heading) and np.isfinite(speed)):
            continue

        arrow_len = max(0.8, speed * ARROW_LENGTH_SCALE)
        dx = arrow_len * math.cos(heading)
        dy = arrow_len * math.sin(heading)

        ax.arrow(
            x,
            y,
            dx,
            dy,
            width=0.0,
            head_width=ARROW_HEAD_WIDTH,
            head_length=ARROW_HEAD_LENGTH,
            length_includes_head=True,
            fc=ARROW_COLOR,
            ec=ARROW_COLOR,
            linewidth=ARROW_LINEWIDTH,
            alpha=ARROW_ALPHA,
            zorder=220,
        )

        if not show_speed_text:
            continue

        norm = math.hypot(dx, dy)
        if norm > 1e-8:
            ux = dx / norm
            uy = dy / norm
        else:
            ux, uy = 1.0, 0.0

        tx = x + dx + ux * SPEED_TEXT_OFFSET
        ty = y + dy + uy * SPEED_TEXT_OFFSET

        ha = "left" if ux >= 0 else "right"
        va = "bottom" if uy >= 0 else "top"

        ax.text(
            tx,
            ty,
            f"v={speed:.1f} m/s",
            fontsize=12.0,
            color="#555555",
            ha=ha,
            va=va,
            bbox=dict(
                boxstyle="round,pad=0.22,rounding_size=0.10",
                facecolor="white",
                edgecolor="none",
                alpha=SPEED_TEXT_BBOX_ALPHA,
            ),
            zorder=230,
        )


# ============================================================================
# Frame construction
# ============================================================================

def build_frames_from_df(df: pd.DataFrame, time_col: str) -> List[pd.Series]:
    """
    Build frames directly from the original timestamp column.

    Rows sharing the same timestamp are grouped together, and only the first row
    of each group is kept.
    """
    df2 = df.copy()
    df2["_time_numeric_"] = pd.to_numeric(df2[time_col], errors="coerce")
    df2 = df2[np.isfinite(df2["_time_numeric_"])].copy()
    if df2.empty:
        return []

    df2["_time_key_"] = df2["_time_numeric_"].round(6)
    frames = [group.iloc[0].copy() for _, group in df2.groupby("_time_key_", sort=True)]

    for row in frames:
        row["time_actual"] = safe_float(row.get("_time_key_", np.nan), np.nan)

    return frames


def apply_frame_step(frames: Sequence[pd.Series], frame_step: int) -> Tuple[List[pd.Series], List[int]]:
    """Subsample the frame sequence using the specified step."""
    frame_step = max(1, int(frame_step))
    indices = list(range(0, len(frames), frame_step))
    sampled_frames = [frames[i] for i in indices]
    return sampled_frames, indices


def apply_visualize_time_range(
    frames: Sequence[pd.Series],
    indices: Sequence[int],
    visualize_time_range: Sequence[float],
) -> Tuple[List[pd.Series], List[int], int, int]:
    """
    Slice the sampled frame sequence by ratio.
    """
    if len(frames) == 0:
        return list(frames), list(indices), 0, 0

    if visualize_time_range is None or len(visualize_time_range) != 2:
        raise ValueError("VISUALIZE_TIME_RANGE must be a sequence like [start_ratio, end_ratio].")

    start_ratio = max(0.0, min(1.0, float(visualize_time_range[0])))
    end_ratio = max(0.0, min(1.0, float(visualize_time_range[1])))

    if end_ratio < start_ratio:
        start_ratio, end_ratio = end_ratio, start_ratio

    n = len(frames)

    start_idx = int(math.floor(start_ratio * n))
    end_idx_exclusive = int(math.ceil(end_ratio * n))

    start_idx = max(0, min(start_idx, n - 1))
    end_idx_exclusive = max(start_idx + 1, min(end_idx_exclusive, n))

    sliced_frames = list(frames[start_idx:end_idx_exclusive])
    sliced_indices = list(indices[start_idx:end_idx_exclusive])

    return sliced_frames, sliced_indices, start_idx, end_idx_exclusive


def compute_axis_limits_from_display_frames(
    display_frames: Sequence[pd.Series],
    posx1_col: str,
    posy1_col: str,
    posx2_col: str,
    posy2_col: str,
) -> Tuple[float, float, float, float]:
    """
    Compute axis limits using only the displayed frames.
    """
    xs: List[float] = []
    ys: List[float] = []

    for row in display_frames:
        x1 = safe_float(row.get(posx1_col, np.nan), np.nan)
        y1 = safe_float(row.get(posy1_col, np.nan), np.nan)
        x2 = safe_float(row.get(posx2_col, np.nan), np.nan)
        y2 = safe_float(row.get(posy2_col, np.nan), np.nan)

        if np.isfinite(x1):
            xs.append(x1)
        if np.isfinite(x2):
            xs.append(x2)
        if np.isfinite(y1):
            ys.append(y1)
        if np.isfinite(y2):
            ys.append(y2)

    if len(xs) == 0 or len(ys) == 0:
        return -50.0, 50.0, -50.0, 50.0

    x_min = float(np.min(xs)) - X_MARGIN_LEFT
    x_max = float(np.max(xs)) + X_MARGIN_RIGHT
    y_min = float(np.min(ys)) - Y_MARGIN_BOTTOM
    y_max = float(np.max(ys)) + Y_MARGIN_TOP

    if x_max - x_min < 1e-6:
        x_min -= 10.0
        x_max += 10.0

    if y_max - y_min < 1e-6:
        y_min -= 10.0
        y_max += 10.0

    return x_min, x_max, y_min, y_max


def infer_frame_durations_ms(display_frames: Sequence[pd.Series]) -> List[int]:
    """
    Infer GIF frame durations from actual timestamp gaps.
    """
    if len(display_frames) <= 1:
        return [FALLBACK_DURATION_MS]

    times = [safe_float(row.get("time_actual", np.nan), np.nan) for row in display_frames]
    valid_dts = []

    for i in range(len(times) - 1):
        t0 = times[i]
        t1 = times[i + 1]
        if np.isfinite(t0) and np.isfinite(t1) and (t1 > t0):
            valid_dts.append(t1 - t0)

    if len(valid_dts) == 0:
        return [FALLBACK_DURATION_MS] * len(display_frames)

    median_dt = float(np.median(valid_dts))
    durations: List[int] = []

    for i in range(len(times)):
        if i < len(times) - 1:
            t0 = times[i]
            t1 = times[i + 1]
            if np.isfinite(t0) and np.isfinite(t1) and (t1 > t0):
                dt = t1 - t0
            else:
                dt = median_dt
        else:
            dt = median_dt

        ms = int(round(dt * 1000.0))
        ms = max(MIN_GIF_DURATION_MS, ms)
        durations.append(ms)

    return durations


# ============================================================================
# Rendering
# ============================================================================

def render_one_frame_image(
    stem: str,
    metric_name: str,
    current_row: pd.Series,
    current_frame_idx_in_sequence: int,
    total_frames: int,
    display_frames_history: Sequence[pd.Series],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    value_col: str,
    value_to_rgba_func: Callable,
    colorbar_norm,
    colorbar_cmap,
    colorbar_label: str,
    colorbar_ticks: Sequence[float],
    posx1_col: str,
    posy1_col: str,
    posx2_col: str,
    posy2_col: str,
    heading1_col: Optional[str],
    heading2_col: Optional[str],
    len1_col: Optional[str],
    wid1_col: Optional[str],
    len2_col: Optional[str],
    wid2_col: Optional[str],
    vel1_col: Optional[str],
    vel2_col: Optional[str],
    show_speed_text: bool,
) -> Image.Image:
    """Render a single frame as a PIL image."""
    fig = plt.figure(figsize=FIGSIZE, dpi=FIG_DPI, facecolor="white")

    ax = fig.add_axes([0.10, 0.12, 0.72, 0.78])
    cax = fig.add_axes([0.86, 0.20, 0.025, 0.58])

    sm = ScalarMappable(norm=colorbar_norm, cmap=colorbar_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(colorbar_label, fontsize=15)
    cbar.ax.tick_params(labelsize=12, width=1.0, length=4)
    cbar.outline.set_linewidth(1.0)
    cbar.set_ticks(colorbar_ticks)

    style_axis(ax, x_min, x_max, y_min, y_max)

    if DRAW_CENTER_TRAJECTORY and len(display_frames_history) >= 2:
        draw_center_trajectory(
            ax=ax,
            frames_history=display_frames_history,
            posx1_col=posx1_col,
            posy1_col=posy1_col,
            posx2_col=posx2_col,
            posy2_col=posy2_col,
            zorder=5,
            alpha=HISTORY_TRAJ_ALPHA,
            line_color=HISTORY_TRAJ_COLOR,
        )

    draw_single_frame_on_axis(
        ax=ax,
        row=current_row,
        value_to_rgba_func=value_to_rgba_func,
        value_col=value_col,
        posx1_col=posx1_col,
        posy1_col=posy1_col,
        posx2_col=posx2_col,
        posy2_col=posy2_col,
        heading1_col=heading1_col,
        heading2_col=heading2_col,
        len1_col=len1_col,
        wid1_col=wid1_col,
        len2_col=len2_col,
        wid2_col=wid2_col,
    )

    if DRAW_SPEED_ARROW:
        draw_speed_heading_annotation(
            ax=ax,
            row=current_row,
            posx1_col=posx1_col,
            posy1_col=posy1_col,
            posx2_col=posx2_col,
            posy2_col=posy2_col,
            heading1_col=heading1_col,
            heading2_col=heading2_col,
            vel1_col=vel1_col,
            vel2_col=vel2_col,
            show_speed_text=show_speed_text,
        )

    cur_time = safe_float(current_row.get("time_actual", np.nan), np.nan)
    raw_value = current_row.get(value_col, np.nan)
    cur_value = safe_float(raw_value, np.nan)

    if np.isfinite(cur_value):
        title_line_2 = f"EA = {cur_value:.3f}"
    else:
        title_line_2 = "EA = NaN"

    ax.set_title(f"Colored by {metric_name}\n{title_line_2}", pad=18)

    fig.suptitle(
        f"{stem}\nCurrent time = {cur_time:.3f} s | Frame {current_frame_idx_in_sequence + 1}/{total_frames}",
        y=0.98,
        fontsize=14,
    )

    canvas = FigureCanvas(fig)
    canvas.draw()

    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    image = Image.fromarray(rgba[:, :, :3])

    plt.close(fig)
    return image


def save_gif(
    stem: str,
    metric_name: str,
    display_frames: Sequence[pd.Series],
    durations_ms: Sequence[int],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    value_col: str,
    value_to_rgba_func: Callable,
    colorbar_norm,
    colorbar_cmap,
    colorbar_label: str,
    colorbar_ticks: Sequence[float],
    posx1_col: str,
    posy1_col: str,
    posx2_col: str,
    posy2_col: str,
    heading1_col: Optional[str],
    heading2_col: Optional[str],
    len1_col: Optional[str],
    wid1_col: Optional[str],
    len2_col: Optional[str],
    wid2_col: Optional[str],
    vel1_col: Optional[str],
    vel2_col: Optional[str],
    show_speed_text: bool,
    out_dir: Path,
) -> Optional[Path]:
    """Render and save the GIF for one CSV file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_gif = out_dir / f"{stem}_{metric_name}.gif"

    pil_frames: List[Image.Image] = []
    for i, row in enumerate(display_frames):
        history = display_frames[: i + 1]
        img = render_one_frame_image(
            stem=stem,
            metric_name=metric_name,
            current_row=row,
            current_frame_idx_in_sequence=i,
            total_frames=len(display_frames),
            display_frames_history=history,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            value_col=value_col,
            value_to_rgba_func=value_to_rgba_func,
            colorbar_norm=colorbar_norm,
            colorbar_cmap=colorbar_cmap,
            colorbar_label=colorbar_label,
            colorbar_ticks=colorbar_ticks,
            posx1_col=posx1_col,
            posy1_col=posy1_col,
            posx2_col=posx2_col,
            posy2_col=posy2_col,
            heading1_col=heading1_col,
            heading2_col=heading2_col,
            len1_col=len1_col,
            wid1_col=wid1_col,
            len2_col=len2_col,
            wid2_col=wid2_col,
            vel1_col=vel1_col,
            vel2_col=vel2_col,
            show_speed_text=show_speed_text,
        )
        pil_frames.append(img)

    if len(pil_frames) == 0:
        return None

    first = pil_frames[0]
    rest = pil_frames[1:] if len(pil_frames) > 1 else []

    first.save(
        out_gif,
        save_all=True,
        append_images=rest,
        duration=list(durations_ms),
        loop=GIF_LOOP,
        optimize=GIF_OPTIMIZE,
        disposal=GIF_DISPOSAL,
    )

    return out_gif


# ============================================================================
# CSV processing
# ============================================================================

def process_single_csv(csv_path: Path) -> None:
    """Process one CSV file and export its EA-colored GIF."""
    csv_path = Path(csv_path)
    stem = csv_path.stem
    print(f"\nProcessing file: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"  Failed to read CSV: {exc}")
        return

    if df.empty:
        print("  Empty file. Skipped.")
        return

    cols = df.columns

    time_col = choose_col(cols, ["Time (s)", "Time", "time", "timestamp", "Timestamp"])
    if time_col is None:
        time_col = df.columns[0]

    frames = build_frames_from_df(df, time_col)
    if len(frames) == 0:
        print("  No valid frames. Skipped.")
        return

    sampled_frames, sampled_indices = apply_frame_step(frames, FRAME_STEP)
    if len(sampled_frames) == 0:
        print("  No sampled frames. Skipped.")
        return

    sampled_frames_before_range = len(sampled_frames)

    sampled_frames, sampled_indices, range_start_idx, range_end_idx_exclusive = apply_visualize_time_range(
        sampled_frames,
        sampled_indices,
        VISUALIZE_TIME_RANGE,
    )

    if len(sampled_frames) == 0:
        print("  No frames remain after VISUALIZE_TIME_RANGE slicing. Skipped.")
        return

    durations_ms = infer_frame_durations_ms(sampled_frames)

    ea_col = choose_col(cols, ["EA", "ea", "EA (m/s^2)", "EA_max_1234", "EA_max"])
    if ea_col is None:
        print("  EA column not found. Skipped.")
        return

    posx1_col = choose_col(cols, ["Position X (m)", "pos_x", "x", "X", "position_x", "PositionX"])
    posy1_col = choose_col(cols, ["Position Y (m)", "pos_y", "y", "Y", "position_y", "PositionY"])
    posx2_col = choose_col(cols, ["2_Position X (m)", "Pos2 X", "2_pos_x", "2_x", "2_PositionX", "Position X (m)_2"])
    posy2_col = choose_col(cols, ["2_Position Y (m)", "Pos2 Y", "2_pos_y", "2_y", "2_PositionY", "Position Y (m)_2"])

    if not (posx1_col and posy1_col and posx2_col and posy2_col):
        print("  Position columns missing. Skipped.")
        return

    heading1_col = choose_col(cols, ["Heading", "heading", "yaw", "theta"])
    heading2_col = choose_col(cols, ["2_Heading", "2_heading", "2_yaw", "2_theta"])

    len1_col = choose_col(cols, ["Length (m)", "Length", "length"])
    wid1_col = choose_col(cols, ["Width (m)", "Width", "width"])
    len2_col = choose_col(cols, ["2_Length (m)", "2_Length", "2_length"])
    wid2_col = choose_col(cols, ["2_Width (m)", "2_Width", "2_width"])

    vel1_col = choose_col(cols, ["Velocity (m/s)", "Velocity", "velocity", "Speed", "speed"])
    vel2_col = choose_col(cols, ["2_Velocity (m/s)", "2_Velocity", "2_velocity", "2_Speed", "2_speed"])

    x_min, x_max, y_min, y_max = compute_axis_limits_from_display_frames(
        sampled_frames,
        posx1_col,
        posy1_col,
        posx2_col,
        posy2_col,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_gif_ea = save_gif(
        stem=stem,
        metric_name="EA",
        display_frames=sampled_frames,
        durations_ms=durations_ms,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        value_col=ea_col,
        value_to_rgba_func=ea_to_rgba,
        colorbar_norm=EA_NORM,
        colorbar_cmap=RISK_CMAP,
        colorbar_label="EA",
        colorbar_ticks=[0.0, 0.5, 1.0, 1.5],
        posx1_col=posx1_col,
        posy1_col=posy1_col,
        posx2_col=posx2_col,
        posy2_col=posy2_col,
        heading1_col=heading1_col,
        heading2_col=heading2_col,
        len1_col=len1_col,
        wid1_col=wid1_col,
        len2_col=len2_col,
        wid2_col=wid2_col,
        vel1_col=vel1_col,
        vel2_col=vel2_col,
        show_speed_text=SHOW_SPEED_TEXT,
        out_dir=OUTPUT_DIR,
    )

    seq_times = [safe_float(row.get("time_actual", np.nan), np.nan) for row in sampled_frames]
    seq_times_valid = [t for t in seq_times if np.isfinite(t)]

    if len(seq_times_valid) >= 2:
        dts = np.diff(seq_times_valid)
        median_dt = float(np.median(dts))
    else:
        median_dt = np.nan

    print(f"  Total raw frames: {len(frames)}")
    print(f"  Total sampled frames before VISUALIZE_TIME_RANGE: {sampled_frames_before_range}")
    print(f"  VISUALIZE_TIME_RANGE used: [{VISUALIZE_TIME_RANGE[0]}, {VISUALIZE_TIME_RANGE[1]}]")
    print(f"  Range slice on sampled frames: [{range_start_idx}:{range_end_idx_exclusive}]")
    print(f"  Total displayed frames after VISUALIZE_TIME_RANGE: {len(sampled_frames)}")
    print(f"  Frame step: {FRAME_STEP}")
    print(f"  Axis limits from displayed range only: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}]")

    if np.isfinite(median_dt):
        print(f"  Median displayed dt: {median_dt:.6f} s")
        print(f"  Median displayed duration: {int(round(median_dt * 1000))} ms")
    else:
        print("  Median displayed dt: N/A")

    if len(durations_ms) > 0:
        print(f"  GIF duration range: {min(durations_ms)}–{max(durations_ms)} ms")

    print(f"  EA column used: {ea_col}")
    print(f"  Saved EA GIF: {out_gif_ea}")


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    """Entry point."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.exists():
        print(f"Input directory does not exist: {INPUT_DIR}")
        return

    csv_files = sorted(INPUT_DIR.glob("*_EA.csv"))
    if not csv_files:
        print(f"No CSV files ending with '_EA.csv' found in: {INPUT_DIR}")
        return

    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    print(f"\nDetected {len(csv_files)} CSV file(s):")
    for file_path in csv_files:
        print(f"  - {file_path.name}")

    print(f"VISUALIZE_TIME_RANGE = [{VISUALIZE_TIME_RANGE[0]}, {VISUALIZE_TIME_RANGE[1]}]")

    for csv_file in csv_files:
        try:
            process_single_csv(csv_file)
        except Exception as exc:
            print(f"[ERROR] Failed on file {csv_file.name}: {exc}")


if __name__ == "__main__":
    main()