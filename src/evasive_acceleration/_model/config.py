"""EA parameters, motion-model names, and CSV schema."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .constants import (
    DEFAULT_A_MAX,
    DEFAULT_COARSE_SECTOR_NUM,
    DEFAULT_DT_COARSE,
    DEFAULT_DT_FINE,
    DEFAULT_FINE_REFINE_RATIO,
    DEFAULT_LOCAL_FINE_DIR_NUM,
    DEFAULT_LOCAL_FINE_HALF_WINDOW_DEG,
    DEFAULT_TOL,
)


class MotionModel(IntEnum):
    """Motion-model identifiers recognized in input and result schemas."""

    CV = 0
    CTRV = 1
    CTRA = 2


# Schema cleanup recognizes these labels. The EA computation itself uses
# only the CV/CTRV pairings declared in four_modes.py.
MODEL_ORDER = (
    MotionModel.CV,
    MotionModel.CTRV,
    MotionModel.CTRA,
)

TIME_COLUMN = "Time (s)"

# CSV field names do not prescribe how observations were obtained. EA accepts
# provided yaw rates or estimates missing rates using current/past headings.
# EA uses CV/CTRV only. Acceleration/CTRA and TTR names below are recognized
# for result-schema cleanup and comparison-metric compatibility, not extra EA
# motion hypotheses or alternative definitions of the EA score.
EA_YAW_RATE_COLUMNS = ("Yawrate", "2_Yawrate")
EA_ACCELERATION_COLUMNS = ("A_lon", "2_A_lon")
TTR_YAW_RATE_COLUMNS = (
    "Yawrate_EWMA_tau0p6",
    "2_Yawrate_EWMA_tau0p6",
)
TTR_ACCELERATION_COLUMNS = (
    "A_lon_EWMA_tau0p05",
    "2_A_lon_EWMA_tau0p05",
)

STATE_COLUMNS = (
    "Position X (m)",
    "Position Y (m)",
    "Velocity (m/s)",
    "Heading",
    "Length (m)",
    "Width (m)",
    EA_YAW_RATE_COLUMNS[0],
    "2_Position X (m)",
    "2_Position Y (m)",
    "2_Velocity (m/s)",
    "2_Heading",
    "2_Length (m)",
    "2_Width (m)",
    EA_YAW_RATE_COLUMNS[1],
)

REQUIRED_COLUMNS = (TIME_COLUMN, *STATE_COLUMNS)


def pair_key(model_1: MotionModel, model_2: MotionModel) -> str:
    """Return the internal key for one ordered motion-model pair."""

    return f"{model_1.name}_{model_2.name}"


def pair_column(model_1: MotionModel, model_2: MotionModel) -> str:
    """Return the CSV column name for one ordered motion-model pair."""

    return f"EA_{pair_key(model_1, model_2)}"


PAIR_MODE_COLUMNS = tuple(
    pair_column(model_1, model_2)
    for model_1 in MODEL_ORDER
    for model_2 in MODEL_ORDER
)

EA_COLUMN = "EA"
EA_WITH_CTRA_COLUMN = "EA (with CTRA)"
EA_OUTPUT_COLUMNS = (*PAIR_MODE_COLUMNS, EA_COLUMN, EA_WITH_CTRA_COLUMN)

# Comparison measures used alongside EA in the manuscript experiments. The
# remaining analytical helper quantities stay internal to the baseline module.
COMPARISON_OUTPUT_COLUMNS = (
    "TTC",
    "TTC2D",
    "ACT",
    "DRAC",
    "DRAC2D",
    "MEI",
    "BBox distance (m)",
)

# Each added method exports one pairwise value per frame.  Its two directional
# values are kept internal because this project uses an unordered pair score.
TTR_2018_COLUMN = "TTR_2018"
RISK_FIELD_COLUMN = "Risk field"
ADDITIONAL_RISK_OUTPUT_COLUMNS = (TTR_2018_COLUMN, RISK_FIELD_COLUMN)

OUTPUT_COLUMNS = (
    *EA_OUTPUT_COLUMNS,
    *COMPARISON_OUTPUT_COLUMNS,
    *ADDITIONAL_RISK_OUTPUT_COLUMNS,
)

ORIGINAL_EA_PAIR_KEYS = tuple(
    pair_key(model_1, model_2)
    for model_1 in (MotionModel.CV, MotionModel.CTRV)
    for model_2 in (MotionModel.CV, MotionModel.CTRV)
)

ALL_PAIR_KEYS = tuple(
    pair_key(model_1, model_2)
    for model_1 in MODEL_ORDER
    for model_2 in MODEL_ORDER
)

# Remove stale generated columns before writing EA results, so a processed
# CSV has one unambiguous EA score and its four motion-model components.
LEGACY_GENERATED_COLUMNS = (
    "EA_Original_CTRV_CV_Mean4",
    "EA_CTRA_CTRA",
    "EA_CTRA_CTRV_Mean4",
    "EA_CTRA_CV_Mean4",
    "EA_CTRA_CTRV_CV_Mean9",
    "EA_CVCV",
    "EA_CVCT",
    "EA_CTCV",
    "EA_CTCT",
    "EA_diff",
    "EA_CVCV_diff",
    "EA_CVCT_diff",
    "EA_CTCV_diff",
    "EA_CTCT_diff",
)

GENERATED_COLUMNS_TO_REPLACE = tuple(
    dict.fromkeys((*OUTPUT_COLUMNS, *LEGACY_GENERATED_COLUMNS))
)


@dataclass(frozen=True)
class EAConfig:
    """Numerical settings used by the reproducible EA implementation."""

    prediction_horizon_s: float = 10.0
    coarse_step_s: float = DEFAULT_DT_COARSE
    fine_step_s: float = DEFAULT_DT_FINE
    max_extra_acceleration_mps2: float = DEFAULT_A_MAX
    solver_tolerance: float = DEFAULT_TOL
    coarse_direction_count: int = DEFAULT_COARSE_SECTOR_NUM
    local_direction_half_window_deg: float = DEFAULT_LOCAL_FINE_HALF_WINDOW_DEG
    local_direction_count: int = DEFAULT_LOCAL_FINE_DIR_NUM
    fine_refine_ratio: float = DEFAULT_FINE_REFINE_RATIO
    observed_acceleration_min_mps2: float = -10.0
    observed_acceleration_max_mps2: float = 3.0
    output_decimals: int = 4


DEFAULT_CONFIG = EAConfig()
