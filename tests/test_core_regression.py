"""Frozen-source integrity and unique EA public-interface regression checks.

Analytical cases and independent continuous geometry checks protect the current
solver. Historical grid-estimator results belong to the archived release.
"""

import importlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest

import core_ea
import evasive_acceleration as ea

ROOT = Path(__file__).resolve().parents[1]
MODE_KEYS = ("EA_CVCV", "EA_CVCT", "EA_CTCV", "EA_CTCT")
CASES = json.loads((ROOT / "tests/fixtures/ea_interface_cases.json").read_text(encoding="utf-8"))["cases"]


def test_validated_numerical_sources_and_auxiliary_algorithms_are_unchanged():
    integrity = runpy.run_path(str(ROOT / "tools" / "check_solver_integrity.py"))
    assert integrity["check_solver_integrity"]() == []


def test_new_ea_is_the_only_public_default_and_has_no_backend_selector():
    assert core_ea.CORE_ID == "ea-continuous-certified-20261022"
    assert core_ea.get_core_provenance()["backend"] == "EA"
    for name in ("get_backend", "use_backend", "BACKEND_NAMES"):
        assert not hasattr(ea, name), name
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("evasive_acceleration.backends")


@pytest.mark.parametrize("name", ["calculator", "verified_calculator", "verified_solver", "motion"])
def test_archived_reference_solver_modules_are_not_installable(name):
    assert not (Path(ea.__file__).resolve().parent / "_paper" / (name + ".py")).exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("evasive_acceleration._paper." + name)


def test_compatibility_shims_do_not_expose_the_archived_solver():
    from evasive_acceleration._paper import numerical_solver
    assert not hasattr(numerical_solver, "solve_pair")
    assert not hasattr(numerical_solver, "compute_ea")
    assert not hasattr(numerical_solver, "find_min_acceleration")
    assert set(name for name in vars(numerical_solver) if not name.startswith("_")) == {
        "njit", "DEFAULT_DT_COARSE", "DEFAULT_DT_FINE", "DEFAULT_A_MAX", "DEFAULT_TOL",
        "DEFAULT_COARSE_SECTOR_NUM", "DEFAULT_LOCAL_FINE_HALF_WINDOW_DEG",
        "DEFAULT_LOCAL_FINE_DIR_NUM", "DEFAULT_FINE_REFINE_RATIO",
    }


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_public_four_modes_keep_schema_classification_and_exact_unrounded_aggregation(case):
    a = ea.RoadUserState(*case["agent_a"])
    b = ea.RoadUserState(*case["agent_b"])
    kwargs = ea.build_compute_kwargs(a, b, **case["settings"])
    actual = core_ea.compute_ea_modes(**kwargs)
    assert set(actual) == set(MODE_KEYS), "CTRA must not silently enter main EA"
    assert all(value >= 0. for value in actual.values())
    score = ea.compute_ea(a, b, **case["settings"])
    assert score == sum(actual[key] for key in MODE_KEYS) / 4.
    if case["expected_class"] == "zero":
        assert all(value == 0. for value in actual.values())
    elif case["expected_class"] == "infinite":
        assert all(math.isinf(value) for value in actual.values())
    else:
        assert math.isfinite(score) and score > 0.


def test_finite_horizon_applies_to_cvcv_as_well_as_turning_modes():
    a = ea.RoadUserState(0., 0., 10., 0., 4.5, 1.8, 0.)
    b = ea.RoadUserState(20., 0., 8., 0., 4.5, 1.8, 0.)
    kwargs = ea.build_compute_kwargs(a, b)
    short = core_ea.compute_ea_modes(**{**kwargs, "T_total": 2.})
    long = core_ea.compute_ea_modes(**{**kwargs, "T_total": 10.})
    assert all(value == 0. for value in short.values())
    assert all(math.isfinite(value) and value > 0. for value in long.values())
    assert len(set(long.values())) == 1


def test_continuous_contact_detection_is_not_disabled_by_legacy_grid_settings():
    a = ea.RoadUserState(0., 0., 1., 0., .004, .004, 0.)
    b = ea.RoadUserState(.01, -.01, 1., math.pi / 2., .004, .004, 0.)
    coarse = ea.compute_ea(a, b, T_total=.02, dt_coarse=.02, dt_fine=.02)
    fine = ea.compute_ea(a, b, T_total=.02, dt_coarse=.01, dt_fine=.001)
    assert math.isfinite(coarse) and coarse > 0.
    assert coarse == fine


def test_strict_aggregation_preserves_nan_and_infinite_modes():
    assert math.isnan(core_ea.aggregate_ea_modes({"EA_CTCT": 1.}))
    assert math.isnan(core_ea.aggregate_ea_modes(dict.fromkeys(MODE_KEYS, math.nan)))
    assert math.isinf(core_ea.aggregate_ea_modes(dict.fromkeys(MODE_KEYS, math.inf)))


@pytest.mark.parametrize("first_import", ["core_ea", "single_frame", "evasive_acceleration"])
def test_fresh_import_orders_do_not_create_a_circular_import(first_import):
    code = (
        f"import {first_import}; "
        "from evasive_acceleration import RoadUserState, compute_ea; "
        "a=RoadUserState(0,0,5,0,4.5,1.8,0); "
        "b=RoadUserState(0,100,5,0,4.5,1.8,0); "
        "assert compute_ea(a,b)==0.0"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    process = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=ROOT, env=environment, timeout=90,
    )
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.parametrize("position", ["before", "after"])
@pytest.mark.parametrize("old_name", ["paper", "v53"])
def test_removed_cli_backend_switch_is_rejected_not_silently_ignored(tmp_path, position, old_name):
    args = ["--backend", old_name, "single"] if position == "before" else ["single", "--backend", old_name]
    result = subprocess.run(
        [sys.executable, "-I", "-m", "evasive_acceleration", *args],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=90,
    )
    assert result.returncode == 2
    assert "--backend was removed" in result.stderr
    assert "one implementation" in result.stderr
    assert "Traceback" not in result.stderr
