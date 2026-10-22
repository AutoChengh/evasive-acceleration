"""Non-destructive optional-native build and discovery regression checks."""

import hashlib
import json
import subprocess

import pytest

from evasive_acceleration import native_build


def test_explicit_missing_native_library_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("EA_NATIVE_CELLS_LIBRARY", str(tmp_path / "absent library.so"))
    with pytest.raises((RuntimeError, FileNotFoundError), match="(?i)(build|native|library)"):
        native_build.find_native_library()


def test_explicit_native_library_path_can_contain_spaces_and_unicode(tmp_path, monkeypatch):
    library = tmp_path / "native 文件夹" / "test library.so"
    library.parent.mkdir()
    library.write_bytes(b"discovery-only placeholder; never loaded")
    monkeypatch.setenv("EA_NATIVE_CELLS_LIBRARY", str(library))
    assert native_build.find_native_library().resolve() == library.resolve()


def test_incomplete_default_cache_is_rejected_without_a_build_manifest(tmp_path, monkeypatch):
    library = tmp_path / "incomplete cache.so"
    library.write_bytes(b"interrupted build output")
    monkeypatch.delenv("EA_NATIVE_CELLS_LIBRARY", raising=False)
    monkeypatch.setattr(native_build, "default_library_path", lambda **kwargs: library)
    with pytest.raises(RuntimeError, match="(?i)(incomplete|changed|not built)"):
        native_build.find_native_library()
    assert library.read_bytes() == b"interrupted build output"


@pytest.mark.parametrize("changed", ["binary", "source_hash"])
def test_default_cache_requires_current_source_and_untampered_binary(
    tmp_path, monkeypatch, changed,
):
    library = tmp_path / "cache.so"
    library.write_bytes(b"original cache artifact")
    record = {
        "source_sha256": hashlib.sha256(native_build.native_source_path().read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }
    manifest = library.with_suffix(library.suffix + ".build.json")
    manifest.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.delenv("EA_NATIVE_CELLS_LIBRARY", raising=False)
    monkeypatch.setattr(native_build, "default_library_path", lambda **kwargs: library)
    assert native_build.find_native_library() == library
    if changed == "binary":
        library.write_bytes(b"changed cache artifact")
    else:
        record["source_sha256"] = "0" * 64
        manifest.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RuntimeError, match="(?i)(incomplete|changed|not built)"):
        native_build.find_native_library()


def test_build_does_not_overwrite_an_existing_output(tmp_path):
    output = tmp_path / "keep.so"
    output.write_bytes(b"preexisting library must remain byte-identical")
    before = output.read_bytes()
    with pytest.raises((FileExistsError, RuntimeError, ValueError)):
        native_build.build_native_library(output=output)
    assert output.read_bytes() == before


def test_missing_compiler_is_reported_without_installing_software(tmp_path, monkeypatch):
    monkeypatch.setattr(native_build.shutil, "which", lambda _: None)
    with pytest.raises((RuntimeError, FileNotFoundError), match="(?i)(compiler|g\\+\\+|C\\+\\+)"):
        native_build.build_native_library(output=tmp_path / "new.so", compiler="missing-cxx")
    assert not (tmp_path / "new.so").exists()


@pytest.mark.parametrize("host_specific", [False, True])
def test_gnu_build_uses_argv_strict_floating_point_and_explicit_cpu_choice(
    tmp_path, monkeypatch, host_specific,
):
    calls = []

    class StopBeforeCompilation(RuntimeError):
        pass

    def intercept(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and any(str(x).endswith(".cpp") for x in command):
            calls.append((command, kwargs))
            raise StopBeforeCompilation("inspect command without compiling")
        return subprocess.CompletedProcess(command, 0, stdout="test compiler", stderr="")

    monkeypatch.setattr(native_build.shutil, "which", lambda _: "C:/compiler path/g++.exe")
    monkeypatch.setattr(native_build.subprocess, "run", intercept)
    with pytest.raises(RuntimeError):
        native_build.build_native_library(output=tmp_path / "new output.so",
                                          compiler="g++", native=host_specific)
    assert len(calls) == 1
    command, options = calls[0]
    assert isinstance(command, (list, tuple)), "never interpolate a shell command"
    assert not options.get("shell", False)
    assert "-fno-fast-math" in command and "-ffp-contract=off" in command
    assert "-ffast-math" not in command and "-Ofast" not in command
    assert ("-march=native" in command) is host_specific
    assert any(str(x).endswith("native_cells.cpp") for x in command)
    assert not (tmp_path / "new output.so").exists()
