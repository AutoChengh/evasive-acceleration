"""Numerical and artifact checks for the synthetic equal-TTC example."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = str(PROJECT_ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
SPEC = importlib.util.spec_from_file_location(
    "ea_comparison_example", PROJECT_ROOT / "examples" / "comparison_case.py"
)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


@pytest.fixture(scope="module")
def computed_comparison(tmp_path_factory):
    # Two computations with the default finite-horizon paper solver.
    directory = tmp_path_factory.mktemp("synthetic-comparison")
    results = comparison.run_comparison(directory, make_plot=False)
    return results, directory


def test_equal_ttc_has_different_default_ea(computed_comparison):
    results, _ = computed_comparison
    assert list(results["provenance"]) == ["synthetic", "synthetic"]
    assert results["TTC_s"].tolist() == pytest.approx([2.0, 2.0], abs=1e-12)
    assert results["EA_m_s2"].tolist() == pytest.approx(
        [0.5, 0.870648557698041], abs=1e-8
    )
    assert results.iloc[1]["EA_m_s2"] - results.iloc[0]["EA_m_s2"] > 0.3
    assert results["clearance_m"].tolist() == pytest.approx([4.0, 20.0])
    assert results["DRAC_m_s2"].tolist() == pytest.approx([0.5, 2.5])
    assert all(math.isfinite(value) and value > 0 for value in results["EA_m_s2"])


def test_comparison_exports_complete_inputs_and_settings(computed_comparison):
    import pandas as pd

    results, directory = computed_comparison
    saved = pd.read_csv(directory / "comparison_results.csv")
    inputs = pd.read_csv(directory / "comparison_inputs.csv")
    pd.testing.assert_frame_equal(saved, results)
    for prefix in ("a", "b"):
        for field in comparison.STATE_FIELDS:
            column = f"{prefix}_{field}"
            assert saved[column].tolist() == inputs[column].tolist()
    metadata = json.loads((directory / "comparison_run.json").read_text(encoding="utf-8"))
    assert metadata["synthetic"] is True
    assert len(metadata["input_sha256"]) == 64
    assert metadata["solver_settings"]["T_total"] == 10.0
    assert metadata["solver_settings"]["dt_fine"] == 0.02
    assert metadata["solver_settings"]["tol"] == 1e-3
    assert metadata["outputs"]["figure"] is None
    assert metadata["font_provenance"] is None


def test_comparison_plot_has_separate_units_and_regular_text(computed_comparison):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    results, directory = computed_comparison
    destination = directory / "comparison_case.png"
    figure = comparison.plot_comparison(results, destination)
    try:
        assert destination.is_file() and destination.stat().st_size > 1000
        assert "EA (m/s²)" in [axis.get_ylabel() for axis in figure.axes]
        assert "TTC (s)" in [axis.get_ylabel() for axis in figure.axes]
        for text in figure.findobj(match=lambda artist: hasattr(artist, "get_fontweight")):
            assert text.get_fontweight() in ("normal", "regular", 400)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("filename", [
    "comparison_inputs.csv", "comparison_results.csv", "comparison_case.png", "comparison_run.json"
])
def test_existing_artifact_prevents_all_writes(tmp_path, filename):
    existing = tmp_path / filename
    existing.write_bytes(b"existing artifact")
    with pytest.raises(FileExistsError, match="--overwrite"):
        comparison.run_comparison(tmp_path)
    assert existing.read_bytes() == b"existing artifact"
    assert list(tmp_path.iterdir()) == [existing]


def test_explicit_overwrite_replaces_generated_artifacts(tmp_path):
    existing = tmp_path / "comparison_results.csv"
    existing.write_bytes(b"old results")
    results = comparison.run_comparison(tmp_path, make_plot=False, overwrite=True)
    assert existing.read_bytes() != b"old results"
    assert results["TTC_s"].tolist() == pytest.approx([2.0, 2.0])
    assert (tmp_path / "comparison_inputs.csv").is_file()
    assert (tmp_path / "comparison_run.json").is_file()


def test_supplied_input_is_protected_even_with_overwrite(tmp_path):
    supplied = tmp_path / "comparison_inputs.csv"
    original = comparison.DEFAULT_CASES_CSV.read_bytes()
    supplied.write_bytes(original)
    with pytest.raises(ValueError, match="must not replace the supplied input"):
        comparison.run_comparison(tmp_path, cases_csv=supplied, make_plot=False, overwrite=True)
    assert supplied.read_bytes() == original
    assert list(tmp_path.iterdir()) == [supplied]


def test_unequal_ttc_input_is_rejected_before_export(tmp_path):
    import pandas as pd

    cases = pd.read_csv(comparison.DEFAULT_CASES_CSV)
    cases.loc[1, "b_x_m"] = 25.5
    source = tmp_path / "unequal.csv"
    cases.to_csv(source, index=False)
    output = tmp_path / "rejected-output"
    with pytest.raises(ValueError, match="requires equal TTC"):
        comparison.run_comparison(output, cases_csv=source, make_plot=False)
    assert not output.exists()


def test_plot_rejects_unequal_ttc(computed_comparison):
    results, _ = computed_comparison
    unequal = results.copy()
    unequal.loc[1, "TTC_s"] = 2.1
    with pytest.raises(ValueError, match="requires equal TTC"):
        comparison.plot_comparison(unequal)


def test_font_failure_preserves_existing_artifacts(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import evasive_acceleration.plotting as plotting

    filenames = ("comparison_inputs.csv", "comparison_results.csv",
                 "comparison_case.png", "comparison_run.json")
    before = {filename: f"old {filename}".encode() for filename in filenames}
    for filename, content in before.items():
        (tmp_path / filename).write_bytes(content)

    def unavailable_font(**kwargs):
        assert kwargs == {"allow_font_fallback": False}
        raise RuntimeError("Arial unavailable for test")

    monkeypatch.setattr(plotting, "resolve_figure_font", unavailable_font)
    with pytest.raises(RuntimeError, match="Arial unavailable"):
        comparison.run_comparison(tmp_path, overwrite=True)
    assert {filename: (tmp_path / filename).read_bytes() for filename in filenames} == before


def test_plot_failure_does_not_write_csvs_or_manifest(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import evasive_acceleration.plotting as plotting

    monkeypatch.setattr(plotting, "resolve_figure_font", lambda **kwargs: {"actual_family": "Arial"})

    def failed_plot(*args, **kwargs):
        raise RuntimeError("PNG rendering failed")

    monkeypatch.setattr(comparison, "plot_comparison", failed_plot)
    destination = tmp_path / "not-written"
    with pytest.raises(RuntimeError, match="PNG rendering failed"):
        comparison.run_comparison(destination)
    assert not destination.exists()


def test_exclusive_write_failure_removes_only_owned_reservations(tmp_path):
    existing = tmp_path / "second.csv"
    existing.write_bytes(b"created concurrently")
    with pytest.raises(FileExistsError):
        comparison._write_artifacts(
            tmp_path, {"first.csv": b"new", "second.csv": b"replacement"}, overwrite=False
        )
    assert not (tmp_path / "first.csv").exists()
    assert existing.read_bytes() == b"created concurrently"
