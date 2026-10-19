"""Four CV/CTRV pairings using the paper geometry and verified fine search."""

from __future__ import annotations

import math

import numpy as np

from . import calculator as paper
from . import numerical_solver as geometry
from .config import EAConfig, MotionModel
from .motion import predict_pose
from .verified_calculator import PredictionResult, make_time_grid, solve_prediction

FOUR_PAIRS = (
    (MotionModel.CV, MotionModel.CV),
    (MotionModel.CV, MotionModel.CTRV),
    (MotionModel.CTRV, MotionModel.CV),
    (MotionModel.CTRV, MotionModel.CTRV),
)


def _predictions(state, models, time_grid):
    x, y, speed, heading, _length, _width, yaw_rate = state
    return {
        model: predict_pose(
            x=x, y=y, speed=speed, heading=heading,
            yaw_rate=0.0 if model is MotionModel.CV else yaw_rate,
            acceleration=0.0, time_grid=time_grid,
        )
        for model in models
    }


def _prediction_pairs(state_values: np.ndarray, config: EAConfig, pairs=FOUR_PAIRS):
    """Build coarse/fine SAT data for private verification and diagnostics.

    The validation step is ``min(fine_step_s, coarse_step_s)`` so a caller
    cannot accidentally request a final grid less dense than the coarse grid.
    All baseline-zero decisions are checked on the complete final grid.
    """
    state_1 = tuple(float(value) for value in state_values[:7])
    state_2 = tuple(float(value) for value in state_values[7:])
    models_1 = tuple(dict.fromkeys(model_1 for model_1, _ in pairs))
    models_2 = tuple(dict.fromkeys(model_2 for _, model_2 in pairs))
    coarse_time, coarse_squared = make_time_grid(
        config.prediction_horizon_s, config.coarse_step_s,
    )
    fine_time, fine_squared = make_time_grid(
        config.prediction_horizon_s, min(config.fine_step_s, config.coarse_step_s),
    )
    coarse_1 = _predictions(state_1, models_1, coarse_time)
    coarse_2 = _predictions(state_2, models_2, coarse_time)
    if coarse_time is fine_time:
        fine_1, fine_2 = coarse_1, coarse_2
    else:
        fine_1 = _predictions(state_1, models_1, fine_time)
        fine_2 = _predictions(state_2, models_2, fine_time)

    lengths = (state_1[4], state_1[5], state_2[4], state_2[5])
    coarse_pairs, fine_pairs = {}, {}
    for pair in pairs:
        model_1, model_2 = pair
        coarse_pairs[pair] = paper._pair_prediction_data(
            coarse_1[model_1], coarse_2[model_2], *lengths,
            coarse_time, coarse_squared,
        )
        fine_pairs[pair] = coarse_pairs[pair] if coarse_time is fine_time else (
            paper._pair_prediction_data(
                fine_1[model_1], fine_2[model_2], *lengths,
                fine_time, fine_squared,
            )
        )
    return coarse_pairs, fine_pairs


def compute_selected_modes_with_diagnostics(
    state_values: np.ndarray, config: EAConfig, pairs=FOUR_PAIRS,
) -> dict[tuple[MotionModel, MotionModel], PredictionResult]:
    """Private diagnostic entry point; no changes to public result schemas."""
    pairs = tuple(pairs)
    state_1 = tuple(float(value) for value in state_values[:7])
    state_2 = tuple(float(value) for value in state_values[7:])
    if geometry.check_current_obb_collision(
        state_1[0], state_1[1], state_1[3], state_1[4], state_1[5],
        state_2[0], state_2[1], state_2[3], state_2[4], state_2[5],
    ):
        return {
            pair: PredictionResult(
                math.inf, math.nan, math.nan, False, True, "current-contact", 0
            )
            for pair in pairs
        }
    coarse_pairs, fine_pairs = _prediction_pairs(state_values, config, pairs)
    return {
        pair: solve_prediction(coarse_pairs[pair], fine_pairs[pair], config)
        for pair in pairs
    }


def compute_selected_modes(
    state_values: np.ndarray, config: EAConfig, pairs=FOUR_PAIRS,
) -> dict[tuple[MotionModel, MotionModel], float]:
    """Compute unrounded CV/CTRV values using verified fine-grid candidates.

    ``state_values`` retains the canonical 14-value state layout. Validation
    belongs to the public adapter; scalar averaging and CSV rounding remain
    outside this function. Frozen reference sources are not modified.
    """
    diagnostics = compute_selected_modes_with_diagnostics(state_values, config, pairs)
    return {pair: result.value for pair, result in diagnostics.items()}
