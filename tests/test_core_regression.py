"""Frozen-reference preservation and public-interface regression checks.

Expected values in the JSON fixture were generated from the canonical 39103
paper sources, not from the compatibility adapter under test. This migration
suite protects the historical reference separately from the verified runtime.
The repaired runtime is not required to preserve earlier numerical defects.
"""

import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import core_ea
from evasive_acceleration import RoadUserState, build_compute_kwargs, compute_ea
from evasive_acceleration._paper.calculator import compute_frame_metrics
from evasive_acceleration._paper.config import EAConfig


MODE_KEYS = ("EA_CVCV", "EA_CVCT", "EA_CTCV", "EA_CTCT")
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "paper_core_reference.json"
REFERENCE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CASES = REFERENCE["cases"]


def _number(value):
    return math.inf if value == "inf" else float(value)


def test_algorithm_source_files_match_frozen_paper_and_original_helpers():
    integrity = runpy.run_path(str(PROJECT_ROOT / "tools" / "check_core_integrity.py"))
    assert integrity["check_core_integrity"]() == []


def test_ci_uses_the_requested_acceleration_backend():
    expected = os.environ.get("EA_EXPECT_NUMBA")
    if expected is not None:
        assert expected in ("0", "1")
        assert core_ea._NUMBA_OK is (expected == "1")


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_frozen_reference_still_matches_independent_paper_values(case):
    mapping = {
        "T_total": "prediction_horizon_s", "dt_coarse": "coarse_step_s",
        "dt_fine": "fine_step_s", "coarse_sector_num": "coarse_direction_count",
        "local_fine_half_window_deg": "local_direction_half_window_deg",
        "local_fine_dir_num": "local_direction_count",
        "fine_refine_ratio": "fine_refine_ratio", "a_max": "max_extra_acceleration_mps2",
        "tol": "solver_tolerance",
    }
    state = np.array(case["agent_a"] + case["agent_b"], dtype=float)
    scale = case["settings"].get("size_scale", 1.0)
    state[[4, 5, 11, 12]] *= scale
    config = EAConfig(output_decimals=13, **{
        mapping[key]: value for key, value in case["settings"].items()
        if key != "size_scale"
    })
    reference = compute_frame_metrics(state, 0.0, 0.0, config)
    keys = dict(zip(MODE_KEYS, ("EA_CV_CV", "EA_CV_CTRV", "EA_CTRV_CV", "EA_CTRV_CTRV")))
    for key, source in {**keys, "EA": "EA"}.items():
        assert reference[source] == pytest.approx(
            _number(case["expected"][key]), rel=1e-10, abs=1e-10
        )


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_verified_public_four_modes_keep_schema_and_exact_aggregation(case):
    a = RoadUserState(*case["agent_a"])
    b = RoadUserState(*case["agent_b"])
    settings = case["settings"]
    actual = core_ea.compute_ea_modes(**build_compute_kwargs(a, b, **settings))
    assert set(actual) == set(MODE_KEYS), "CTRA must not silently enter main EA"
    for key in MODE_KEYS:
        assert actual[key] >= 0.0
    assert compute_ea(a, b, **settings) == sum(actual[key] for key in MODE_KEYS) / 4.0


def test_aggregation_uses_unrounded_components_before_paper_csv_rounding():
    case = next(case for case in CASES if case["id"] == "turning_both")
    a, b = RoadUserState(*case["agent_a"]), RoadUserState(*case["agent_b"])
    modes = core_ea.compute_ea_modes(**build_compute_kwargs(a, b))
    score = compute_ea(a, b)
    assert score == sum(modes[key] for key in MODE_KEYS) / 4.0
    assert score != sum(round(modes[key], 4) for key in MODE_KEYS) / 4.0


def test_finite_horizon_applies_to_cvcv_as_well_as_turning_modes():
    a = RoadUserState(0.0, 0.0, 10.0, 0.0, 4.5, 1.8, 0.0)
    b = RoadUserState(20.0, 0.0, 8.0, 0.0, 4.5, 1.8, 0.0)
    kwargs = build_compute_kwargs(a, b)
    short = core_ea.compute_ea_modes(**{**kwargs, "T_total": 2.0})
    long = core_ea.compute_ea_modes(**{**kwargs, "T_total": 10.0})
    assert all(value == 0.0 for value in short.values())
    assert all(math.isfinite(value) and value > 0.0 for value in long.values())
    assert len(set(long.values())) == 1


def test_fine_grid_promotes_a_contact_missed_by_every_coarse_sample():
    case = next(case for case in CASES if case["id"] == "fine_only_contact")
    kwargs = build_compute_kwargs(
        RoadUserState(*case["agent_a"]), RoadUserState(*case["agent_b"])
    )
    coarse_only = core_ea.compute_ea_modes(
        **{**kwargs, "T_total": 0.2, "dt_coarse": 0.1, "dt_fine": 0.1}
    )
    with_fine = core_ea.compute_ea_modes(
        **{**kwargs, "T_total": 0.2, "dt_coarse": 0.1, "dt_fine": 0.02}
    )
    assert all(value == 0.0 for value in coarse_only.values())
    assert all(value == pytest.approx(62.5) for value in with_fine.values())


def test_strict_aggregation_preserves_nan_and_infinite_modes():
    assert math.isnan(core_ea.aggregate_ea_modes({"EA_CTCT": 1.0}))
    assert math.isnan(core_ea.aggregate_ea_modes(dict.fromkeys(MODE_KEYS, math.nan)))
    assert math.isinf(core_ea.aggregate_ea_modes(dict.fromkeys(MODE_KEYS, math.inf)))


@pytest.mark.parametrize(
    "first_import", ["core_ea", "single_frame", "evasive_acceleration"]
)
def test_fresh_import_orders_do_not_create_a_circular_import(first_import):
    code = (
        f"import {first_import}; "
        "from evasive_acceleration import RoadUserState, compute_ea; "
        "a=RoadUserState(0,0,5,0,4.5,1.8,0); "
        "b=RoadUserState(0,100,5,0,4.5,1.8,0); "
        "assert compute_ea(a,b)==0.0"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=PROJECT_ROOT, env=env, timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
