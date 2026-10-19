"""The alignment audit must not silently reuse the current solver as 'before'."""

import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_ea_alignment_audit_tests", ROOT / "tools" / "validate_paper_alignment.py",
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_old_engine_bridge_is_an_isolated_dependency_graph(tmp_path):
    paper = tmp_path / "evasive_acceleration" / "_paper"
    paper.mkdir(parents=True)
    (paper / "config.py").write_text("EAConfig = object\nMotionModel = object\n", encoding="utf-8")
    (paper / "four_modes.py").write_text(
        "from . import calculator\ndef compute_selected_modes(*args):\n    return calculator.SENTINEL\n",
        encoding="utf-8",
    )
    (paper / "calculator.py").write_text("SENTINEL = 'before-only'\n", encoding="utf-8")
    for name in ("motion.py", "numerical_solver.py"):
        (paper / name).write_text("# complete but unused test dependency\n", encoding="utf-8")
    (tmp_path / "core_ea.py").write_text(
        "def _get_bridge():\n    raise AssertionError('must be relocated')\n"
        "def compute_ea_modes():\n    return _get_bridge()[2]()\n",
        encoding="utf-8",
    )
    namespace = "_ea_alignment_isolation_test"
    try:
        before, bridge = audit.load_before_engine(tmp_path, namespace=namespace)
        assert before.compute_ea_modes() == "before-only"
        assert bridge.__name__ == namespace + ".four_modes"
        assert str(tmp_path) in bridge.__file__
        assert before._get_bridge()[2].__module__ == bridge.__name__
    finally:
        for name in list(sys.modules):
            if name == namespace or name.startswith(namespace + "."):
                del sys.modules[name]


def test_delta_report_distinguishes_numeric_and_semantic_changes():
    old = dict.fromkeys(audit.VALUE_KEYS, 1.)
    new = {**old, "EA_CVCV": 0., "EA_CVCT": math.inf, "EA": math.nan}
    comparison = audit.compare_values(old, new)
    assert set(comparison["changed"]) == {"EA_CVCV", "EA_CVCT", "EA"}
    assert comparison["delta"]["EA_CVCV"] == -1.
    assert comparison["delta"]["EA_CVCT"] is None
    assert comparison["semantic_transition"]["EA_CVCT"] == "finite_positive -> unresolved_or_current_contact_inf"
    # The same helper used to write reports must produce standards-compliant JSON.
    text = json.dumps(audit.clean_json(comparison), allow_nan=False)
    assert '"inf"' in text and '"nan"' in text


def test_independent_sat_recheck_uses_half_time_squared_and_contact():
    pair = {
        "relx_arr": np.array([3.]), "rely_arr": np.array([0.]),
        "axes_x": np.array([[1., 0., 1., 0.]]),
        "axes_y": np.array([[0., 1., 0., 1.]]),
        "sum_radius": np.array([[2., 1., 2., 1.]]),
        "t_arr": np.array([2.]), "t2_arr": np.array([2.]),
    }
    assert audit.independent_sat_collision(pair, 0., 0., 0.) == (False, None)
    # 0.5 m/s² * (0.5 * 2²) reduces the separation by 1 m: exact contact.
    assert audit.independent_sat_collision(pair, .5, 1., 0.) == (True, 2.)
    assert audit.independent_sat_collision(pair, .500001, 1., 0.) == (True, 2.)
    with pytest.raises(ValueError, match="finite"):
        audit.independent_sat_collision(pair, 0., math.nan, math.nan)


def test_synthetic_scenes_are_seeded_and_all_four_categories_present():
    scenes = list(audit.synthetic_scenes(8, 20261020))
    assert scenes == list(audit.synthetic_scenes(8, 20261020))
    assert scenes != list(audit.synthetic_scenes(8, 20261021))
    assert {row["category"] for row in scenes} == {"head_on", "crossing", "turning", "parallel"}
    assert all(all(math.isfinite(v) for v in row["kwargs"].values()) for row in scenes)


def test_change_summary_reports_counts_and_maxima_without_inf_subtraction():
    old = dict.fromkeys(audit.VALUE_KEYS, math.inf)
    new = {**old, "EA_CVCV": 2.}
    row = audit.compare_values(old, new)
    result = audit.summarize_changes([row])
    assert result["changed_frames"] == 1
    assert result["changed_components"]["EA_CVCV"] == 1
    assert result["finite_delta_summary"]["EA_CVCV"]["max_absolute"] is None

