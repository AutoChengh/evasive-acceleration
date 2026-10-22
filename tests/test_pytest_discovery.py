"""Keep default test discovery independent of retained output snapshots."""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_default_discovery_ignores_tests_inside_output_snapshots(tmp_path):
    # Exercise the real project configuration through pytest itself, without
    # requiring tomllib (Python 3.11+) or a separate TOML dependency.
    (tmp_path / "pyproject.toml").write_text(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"), encoding="utf-8",
    )
    maintained = tmp_path / "tests"
    snapshot = tmp_path / "outputs" / "unpacked-source" / "tests"
    maintained.mkdir()
    snapshot.mkdir(parents=True)
    (maintained / "test_release_probe.py").write_text(
        "def test_expected_probe():\n    pass\n", encoding="utf-8",
    )
    # Matching basenames reproduce the import-mismatch failure caused by
    # source archives retained below outputs/. Never remove real diagnostics.
    (snapshot / "test_release_probe.py").write_text(
        "def test_accidental_probe():\n    pass\n", encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("PYTEST_PLUGINS", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
        encoding="utf-8", timeout=60, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    nodes = [line for line in result.stdout.splitlines() if "::" in line]
    assert nodes == ["tests/test_release_probe.py::test_expected_probe"]
    assert "test_accidental_probe" not in result.stdout
    assert (snapshot / "test_release_probe.py").is_file()
