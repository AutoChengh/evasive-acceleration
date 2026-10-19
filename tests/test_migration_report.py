"""Migration timing reports identify the runtime actually measured."""

import importlib.util
import json
from pathlib import Path

import core_ea


ROOT = Path(__file__).resolve().parents[1]


def test_migration_report_records_current_runtime_not_historical_dev0(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "_ea_migration_report_test", ROOT / "tools" / "validate_paper_migration.py",
    )
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    output = tmp_path / "migration"
    assert audit._run([
        "--frames", "1", "--repeats", "1", "--skip-legacy",
        "--output-dir", str(output),
    ]) == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["timed_runtime_origin"] == "current_checkout"
    assert report["core_id"] == core_ea.CORE_ID
    assert report["runtime_provenance"] == core_ea.get_core_provenance()
    assert report["runtime_provenance"]["runtime_source_sha256"]
    assert Path(report["core_file"]).resolve() == Path(core_ea.__file__).resolve()
    assert report["versions"]["source_package_version"] == audit.versions()["source_package_version"]
    assert report["selected_frames"] == 1
    assert report["migration_equivalence_passed"]
    assert report["timing"]["paper_four_modes"]["count"] == 1
