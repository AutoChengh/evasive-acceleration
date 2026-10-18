"""Frozen original-solver values, evaluated with its unchanged default settings.

Expected values were recorded from the original core on 2026-10-19. They are
constants, not recomputed through a second wrapper around the same solver.
"""

import math
import os
from pathlib import Path
import runpy
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import core_ea
from evasive_acceleration import RoadUserState, build_compute_kwargs, compute_ea


MODE_KEYS = ("EA_CTCT", "EA_CTCV", "EA_CVCT", "EA_CVCV")


def test_algorithm_source_files_match_frozen_baseline():
    integrity = runpy.run_path(str(PROJECT_ROOT / "tools" / "check_core_integrity.py"))
    assert integrity["check_core_integrity"]() == []


def test_ci_uses_the_requested_acceleration_backend():
    expected = os.environ.get("EA_EXPECT_NUMBA")
    if expected is not None:
        assert expected in ("0", "1")
        assert core_ea._NUMBA_OK is (expected == "1")


@pytest.mark.parametrize(
    ("agent_a", "agent_b", "expected_modes", "expected_final"),
    [
        pytest.param(
            (0.0, 0.0, 10.0, 0.0, 4.5, 1.8, 0.0),
            (20.0, 0.0, 8.0, math.pi, 4.7, 1.9, 0.0),
            (4.800096523170988, 4.800096523170988,
             4.800096523170988, 4.91096547967024),
            4.827813762295801,
            id="head-on",
        ),
        pytest.param(
            (0.0, 0.0, 10.0, 0.0, 4.5, 1.8, 0.08),
            (20.0, 0.0, 8.0, math.pi, 4.7, 1.9, -0.04),
            (4.262735936187927, 3.645349057451058,
             4.385562620602209, 4.91096547967024),
            4.301153273477858,
            id="turning-both-actors",
        ),
        pytest.param(
            (0.0, 0.0, 5.0, 0.0, 4.5, 1.8, 0.0),
            (0.0, 100.0, 5.0, 0.0, 4.5, 1.8, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            0.0,
            id="separated-parallel-actors",
        ),
    ],
)
def test_original_four_modes_and_public_api_keep_frozen_values(
    agent_a, agent_b, expected_modes, expected_final
):
    a, b = RoadUserState(*agent_a), RoadUserState(*agent_b)
    # No shorter horizon, reduced direction grid, or enlarged tolerance here.
    actual_modes = core_ea.compute_ea_modes(**build_compute_kwargs(a, b))
    for key, expected in zip(MODE_KEYS, expected_modes):
        assert actual_modes[key] == pytest.approx(expected, rel=1e-8, abs=1e-8)
    assert compute_ea(a, b) == pytest.approx(expected_final, rel=1e-8, abs=1e-8)


def test_strict_aggregation_preserves_nan_and_infinite_modes():
    assert math.isnan(core_ea.aggregate_ea_modes({"EA_CTCT": 1.0}))
    assert math.isnan(core_ea.aggregate_ea_modes(dict.fromkeys(MODE_KEYS, math.nan)))
    assert math.isinf(core_ea.aggregate_ea_modes(dict.fromkeys(MODE_KEYS, math.inf)))


def test_initial_overlap_is_reported_as_original_undefined_result():
    """Freeze the current boundary behavior rather than assign a new risk meaning."""
    state = RoadUserState(0.0, 0.0, 5.0, 0.0, 4.5, 1.8, 0.0)
    assert math.isnan(compute_ea(state, state))
