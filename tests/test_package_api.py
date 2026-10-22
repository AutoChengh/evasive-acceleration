"""Check the installed API, without relying on tests adding src to sys.path.

Install the checkout first with ``python -m pip install -e ".[batch,dev]"``.
"""

import subprocess
import sys
from pathlib import Path

import pytest


LEGACY_GRID_OPTIONS = (
    ("--coarse-sector-num", "coarse_sector_num", "36", 36),
    ("--local-fine-half-window-deg", "local_fine_half_window_deg", "7.5", 7.5),
    ("--local-fine-dir-num", "local_fine_dir_num", "51", 51),
    ("--fine-refine-ratio", "fine_refine_ratio", "1.2", 1.2),
    ("--dt-coarse", "dt_coarse", "0.4", .4),
    ("--dt-fine", "dt_fine", "0.06", .06),
)


def test_editable_path_file_is_safe_for_non_ascii_windows_checkout():
    from importlib.metadata import distribution

    installed = distribution("evasive-acceleration")
    for entry in installed.files or []:
        if str(entry).endswith(".pth"):
            content = installed.locate_file(entry).read_bytes().decode("ascii")
            assert content.startswith("import "), "Use an import hook, not a locale-sensitive source path"


@pytest.mark.parametrize("workflow", ["single", "provided_state.py", "from_history.py"])
def test_console_workflows_also_run_with_windows_gbk_encoding(tmp_path, workflow):
    if workflow == "single":
        command = "from evasive_acceleration.cli import main; main(['single'])"
    else:
        script = Path(__file__).resolve().parents[1] / "examples" / workflow
        command = f"import runpy; runpy.run_path({str(script)!r}, run_name='__main__')"
    result = _run_installed(
        "-c", "import sys; sys.stdout.reconfigure(encoding='gbk'); " + command,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "m/s^2" in result.stdout


def _run_installed(*arguments, cwd):
    return subprocess.run(
        [sys.executable, "-I", *arguments],
        cwd=cwd,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )


def test_installed_package_imports_from_unrelated_directory(tmp_path):
    result = _run_installed(
        "-c",
        "import sys; from importlib.metadata import version; "
        "import evasive_acceleration as ea; "
        "assert ea.__version__ == version('evasive-acceleration'); "
        "assert ea.compute_ea is ea.compute_single_frame_ea; "
        "assert 'batch_compute' not in sys.modules; "
        "assert ea.causal_yaw_rate([0, .1], [0, .01]).tolist() == [0., .1]; "
        "print('installed API OK')",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "installed API OK"


@pytest.mark.parametrize("turn_rate", [0.0, 0.08])
def test_high_level_api_matches_compatible_keyword_solver(turn_rate):
    from core_ea import compute_ea as legacy_compute_ea
    from evasive_acceleration import RoadUserState, build_compute_kwargs, compute_ea

    agent_a = RoadUserState(0.0, 0.0, 10.0, 0.0, 4.5, 1.8, turn_rate)
    agent_b = RoadUserState(20.0, 0.0, 8.0, 3.141592653589793, 4.7, 1.9, 0.0)
    options = {"T_total": 2.0}
    expected = legacy_compute_ea(**build_compute_kwargs(agent_a, agent_b, **options))

    assert compute_ea(agent_a, agent_b, **options) == expected


def test_installed_module_cli_forwards_single_help(tmp_path):
    result = _run_installed(
        "-m", "evasive_acceleration", "single", "--help", cwd=tmp_path
    )
    assert result.returncode == 0, result.stderr
    assert "--agent-a" in result.stdout
    assert "--agent-b" in result.stdout
    for flag, _attribute, _text, _value in LEGACY_GRID_OPTIONS:
        assert flag not in result.stdout, "legacy compatibility options must not be advertised as EA controls"


@pytest.mark.parametrize("flag,attribute,text,value", LEGACY_GRID_OPTIONS)
def test_hidden_grid_options_remain_parseable_without_changing_continuous_ea(flag, attribute, text, value):
    from single_frame import _build_argument_parser
    from evasive_acceleration import RoadUserState, compute_ea

    arguments = _build_argument_parser().parse_args(["--example", flag, text])
    assert getattr(arguments, attribute) == value
    # Nonzero turning rates exercise the actual current solver, not merely
    # argument parsing or the zero-yaw equivalent-pair shortcut.
    a = RoadUserState(0., 0., 8., 0., 4.8, 1.8, .1)
    b = RoadUserState(20., 1., 4., 3.141592653589793, 4.5, 1.8, -.05)
    assert compute_ea(a, b, **{attribute: value}) == compute_ea(a, b)


def test_installed_single_accepts_all_six_hidden_compatibility_flags(tmp_path):
    arguments = [item for flag, _attribute, text, _value in LEGACY_GRID_OPTIONS for item in (flag, text)]
    default = _run_installed("-m", "evasive_acceleration", "single", "--example", cwd=tmp_path)
    compatible = _run_installed("-m", "evasive_acceleration", "single", "--example", *arguments, cwd=tmp_path)
    assert default.returncode == compatible.returncode == 0, default.stderr + compatible.stderr
    default_score = next(line for line in default.stdout.splitlines() if line.startswith("EA ="))
    compatible_score = next(line for line in compatible.stdout.splitlines() if line.startswith("EA ="))
    assert compatible_score == default_score


def test_installed_module_cli_reports_version(tmp_path):
    from evasive_acceleration import __version__

    result = _run_installed(
        "-m", "evasive_acceleration", "--version", cwd=tmp_path
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == __version__


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--agent-a", "0", "0", "10", "0", "-4.5", "1.8", "0",
          "--agent-b", "20", "0", "8", "3.14", "4.7", "1.9", "0"],
         "length and width must be strictly positive"),
        (["--agent-a", "nan", "0", "10", "0", "4.5", "1.8", "0",
          "--agent-b", "20", "0", "8", "3.14", "4.7", "1.9", "0"],
         "All state values must be finite"),
        (["--example", "--dt-fine", "0"],
         "dt_coarse and dt_fine must be strictly positive"),
    ],
)
def test_installed_single_cli_reports_invalid_inputs_without_traceback(
    tmp_path, arguments, message
):
    result = _run_installed(
        "-m", "evasive_acceleration", "single", *arguments, cwd=tmp_path
    )
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert message in result.stderr
    assert "Traceback" not in result.stderr
