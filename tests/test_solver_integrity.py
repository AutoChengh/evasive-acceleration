"""Integrity gates fail on source drift and never rewrite accepted evidence."""

import json
from pathlib import Path
import runpy
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
CHECKER = runpy.run_path(str(ROOT / "tools/check_solver_integrity.py"))


@pytest.mark.parametrize("alteration", ["kernel", "yaw_rate", "baseline", "extra_source", "missing_source"])
def test_integrity_gate_detects_changes_without_refreshing_manifest(tmp_path, alteration):
    snapshot = ROOT / "src/evasive_acceleration/_solver"
    destination = tmp_path / "src/evasive_acceleration/_solver"
    record = json.loads((snapshot / "ea_sources.json").read_text(encoding="utf-8"))
    destination.mkdir(parents=True)
    for name in ["ea_sources.json", *record["sources"]]:
        shutil.copy2(snapshot / name, destination / name)
    for relative in CHECKER["AUXILIARY_HASHES"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, path)
    manifest = destination / "ea_sources.json"
    before = manifest.read_bytes()
    if alteration == "kernel":
        path = destination / "continuous_adaptive_cells.py"
    elif alteration == "yaw_rate":
        path = tmp_path / "src/yaw_rate.py"
    elif alteration == "baseline":
        path = tmp_path / "src/baseline_risk_metrics.py"
    elif alteration == "extra_source":
        path = destination / "unvalidated_kernel.py"
    else:
        (destination / "continuous_adaptive_cells.py").unlink()
        path = None
    if path is not None:
        path.write_bytes((path.read_bytes() if path.exists() else b"") + b"\n# deliberate integrity probe\n")
    failures = CHECKER["check_solver_integrity"](tmp_path)
    assert failures, f"{alteration} escaped the source integrity gate"
    assert manifest.read_bytes() == before, "source drift must not refresh the accepted manifest"
