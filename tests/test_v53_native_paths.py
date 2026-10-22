"""Native path handling must work on the first v53 load, not just a cached one.

The real-library cases use isolated interpreters so an earlier successful
absolute-path import cannot hide failures in relative, tilde or empty paths.
The guard and failed-import tests also run without optional Numba/native builds.
"""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from evasive_acceleration import backends


ENVIRONMENT_KEY = "EA_NATIVE_CELLS_LIBRARY"
PATH_FORMS = ("unset", "empty", "absolute", "relative", "tilde", "unicode")


def _set_path_form(monkeypatch, tmp_path, form):
    """Create a harmless file and configure each supported path spelling."""
    home = tmp_path / "home"
    library = home / "native folder 库" / "libea_cells.dll"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"path guard fixture; not a loadable native library")
    monkeypatch.chdir(library.parent)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    if form == "unset":
        monkeypatch.delenv(ENVIRONMENT_KEY, raising=False)
    else:
        value = {
            "empty": "",
            "absolute": str(library),
            "relative": library.name,
            "tilde": str(Path("~") / library.relative_to(home)),
            "unicode": str(library),
        }[form]
        monkeypatch.setenv(ENVIRONMENT_KEY, value)
    return library.resolve()


@pytest.mark.parametrize("form", PATH_FORMS)
def test_loaded_library_guard_accepts_equivalent_path_spellings(monkeypatch, tmp_path, form):
    library = _set_path_form(monkeypatch, tmp_path, form)
    previous = (ENVIRONMENT_KEY in os.environ, os.environ.get(ENVIRONMENT_KEY))
    backends._check_library_override(library)
    assert (ENVIRONMENT_KEY in os.environ, os.environ.get(ENVIRONMENT_KEY)) == previous


def test_loaded_library_guard_still_rejects_a_different_explicit_path(monkeypatch, tmp_path):
    library = _set_path_form(monkeypatch, tmp_path, "absolute")
    monkeypatch.setenv(ENVIRONMENT_KEY, str(tmp_path / "another library.dll"))
    with pytest.raises(RuntimeError, match="changed after v53 was loaded"):
        backends._check_library_override(library)


@pytest.mark.parametrize("form", ("unset", "empty", "relative", "tilde"))
def test_failed_frozen_import_restores_exact_native_environment(monkeypatch, tmp_path, form):
    from evasive_acceleration import native_build

    library = _set_path_form(monkeypatch, tmp_path, form)
    previous = (ENVIRONMENT_KEY in os.environ, os.environ.get(ENVIRONMENT_KEY))
    observed = []

    def fail_import(name):
        observed.append(os.environ.get(ENVIRONMENT_KEY))
        raise ImportError("deliberate failed frozen import")

    # This checks loader plumbing without requiring Numba or opening the fake
    # library. The numerical source and native solver are never involved.
    monkeypatch.setitem(sys.modules, "numba", SimpleNamespace(
        config=SimpleNamespace(DISABLE_JIT=False),
    ))
    monkeypatch.setattr(native_build, "find_native_library", lambda: library)
    monkeypatch.setattr(backends.importlib, "import_module", fail_import)
    backends._get_v53_runtime.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="Cannot load the v53 native runtime"):
            backends._get_v53_runtime()
        assert observed == [str(library)]
        assert (ENVIRONMENT_KEY in os.environ, os.environ.get(ENVIRONMENT_KEY)) == previous
    finally:
        backends._get_v53_runtime.cache_clear()


@pytest.fixture
def native_library():
    numba = pytest.importorskip("numba", reason="v53 requires optional fast dependencies")
    if numba.config.DISABLE_JIT:
        pytest.skip("v53 requires enabled Numba JIT")
    from evasive_acceleration.native_build import find_native_library

    try:
        return find_native_library()
    except (RuntimeError, FileNotFoundError) as error:
        pytest.skip(f"Build the optional v53 native library first: {error}")


_FRESH_LOAD_CODE = r'''
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

import evasive_acceleration as ea
from evasive_acceleration import backends, native_build

form, original, sandbox = sys.argv[1:]
original, sandbox = Path(original), Path(sandbox)
key = "EA_NATIVE_CELLS_LIBRARY"
home = sandbox / "home"
home.mkdir(parents=True)
os.environ["HOME"] = str(home)
os.environ["USERPROFILE"] = str(home)
os.environ["LOCALAPPDATA"] = str(sandbox / "cache")
os.environ["XDG_CACHE_HOME"] = str(sandbox / "cache")

if form in ("unset", "empty"):
    # Create a valid portable cache entry using the real discovery rules.
    # This also covers machines where the parent used a custom --output build;
    # it does not replace or mock find_native_library.
    library = native_build.default_library_path(native=False)
    library.parent.mkdir(parents=True)
    shutil.copy2(original, library)
    record = {
        "source_sha256": hashlib.sha256(native_build.native_source_path().read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }
    library.with_suffix(library.suffix + ".build.json").write_text(json.dumps(record), encoding="utf-8")
    os.environ.pop(key, None)
    if form == "empty":
        os.environ[key] = ""
elif form == "unicode":
    library = home / "native folder 库" / original.name
    library.parent.mkdir(parents=True)
    shutil.copy2(original, library)
    os.environ[key] = str(library)
else:
    library = home / "native folder" / original.name
    library.parent.mkdir(parents=True)
    shutil.copy2(original, library)
    os.chdir(library.parent)
    os.environ[key] = {
        "absolute": str(library),
        "relative": library.name,
        "tilde": str(Path("~") / library.relative_to(home)),
    }[form]

before = (key in os.environ, os.environ.get(key))
assert ea.get_backend() == "paper"
assert not any(name.endswith(".continuous_native_cells") for name in sys.modules)
a = ea.RoadUserState(0., 0., 8., 0., 4.8, 1.8, .1)
b = ea.RoadUserState(20., 1., 4., math.pi, 4.5, 1.8, 0.)
with ea.use_backend("v53"):
    adaptive, loaded = backends._get_v53_runtime()
    assert loaded == library.resolve(), (loaded, library)
    assert (key in os.environ, os.environ.get(key)) == before
    first = ea.compute_ea(a, b)
    assert math.isclose(first, 1.5366117876437317, abs_tol=.001), first
    assert (key in os.environ, os.environ.get(key)) == before
    assert ea.compute_ea(a, b) == first
assert ea.get_backend() == "paper"
assert (key in os.environ, os.environ.get(key)) == before

# The normalization fix must not silently permit switching an already loaded
# runtime to a different explicitly named binary, even a byte-identical copy.
different = sandbox / ("different_" + original.name)
shutil.copy2(library, different)
os.environ[key] = str(different)
try:
    with ea.use_backend("v53"):
        ea.compute_ea(a, b)
except RuntimeError as error:
    assert "changed after v53 was loaded" in str(error), str(error)
else:
    raise AssertionError("different native library was accepted after initial load")
assert ea.get_backend() == "paper"
if before[0]:
    os.environ[key] = before[1]
else:
    os.environ.pop(key, None)
with ea.use_backend("v53"):
    assert ea.compute_ea(a, b) == first
assert (key in os.environ, os.environ.get(key)) == before
print("native path first-load and environment restoration OK: " + form)
'''


@pytest.mark.parametrize("form", PATH_FORMS)
def test_native_path_first_load_and_computation_in_fresh_process(native_library, tmp_path, form):
    # -I rules out the repository CWD and PYTHONPATH: use the installed package,
    # as a user following the README would, with no parent module cache.
    result = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", "-c", _FRESH_LOAD_CODE,
         form, str(native_library), str(tmp_path)],
        cwd=tmp_path, text=True, capture_output=True, encoding="utf-8", timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "native path first-load and environment restoration OK: " + form in result.stdout
