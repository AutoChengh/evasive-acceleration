"""Build the EA native library explicitly, never during EA timing.

The C++ source is shipped in wheels and source distributions. A pre-existing
GCC-compatible C++17 compiler is required; this module installs no software.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def native_source_path() -> Path:
    return Path(__file__).resolve().parent / "_solver" / "native_cells.cpp"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_library_path(*, native: bool = False) -> Path:
    """Return a versioned user-cache path, not a checkout-relative binary."""
    source = native_source_path()
    if not source.is_file():
        raise RuntimeError("EA C++ source is missing; reinstall the complete package")
    if sys.platform == "win32":
        cache = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        suffix = ".dll"
    else:
        cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
    flavor = "native" if native else "portable"
    machine = platform.machine().replace("/", "_").replace("\\", "_")
    return (cache / "evasive-acceleration" / "continuous" / _sha256(source)[:20]
            / f"{sys.platform}-{machine}-{flavor}" / ("libea_cells" + suffix))


def find_native_library() -> Path:
    """Resolve an explicitly supplied or previously built library; never build."""
    supplied = os.environ.get("EA_NATIVE_CELLS_LIBRARY")
    if supplied:
        path = Path(supplied).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"EA_NATIVE_CELLS_LIBRARY does not exist: {path}")
        return path
    # Host-optimized output is preferred only when explicitly built beforehand.
    for native in (True, False):
        path = default_library_path(native=native)
        manifest = path.with_suffix(path.suffix + ".build.json")
        if path.is_file() and manifest.is_file():
            try:
                record = json.loads(manifest.read_text(encoding="utf-8"))
                if (record["source_sha256"] == _sha256(native_source_path())
                        and record["output_sha256"] == _sha256(path)):
                    return path
            except (OSError, ValueError, KeyError):
                pass
    raise RuntimeError(
        "EA native library is not built. Run 'python -m "
        "evasive_acceleration.native_build' before computing EA. "
        "Incomplete or changed cache artifacts are not accepted. For a custom "
        "build, use a fresh --output path and set EA_NATIVE_CELLS_LIBRARY to it."
    )


def build_native_library(output=None, compiler: str = "g++", native: bool = False) -> Path:
    """Build a fresh library and SHA-256 manifest, refusing to replace either.

    ``native=True`` adds the host-specific flags used in the validated Linux
    timing experiment. Rebuild and retest when moving to a different CPU.
    """
    source = native_source_path()
    if not source.is_file():
        raise RuntimeError(f"Missing EA native source: {source}")
    target = (Path(output).expanduser().resolve() if output is not None
              else default_library_path(native=native))
    manifest = target.with_suffix(target.suffix + ".build.json")
    if target.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite native library/build manifest: {target}")
    executable = shutil.which(compiler)
    if executable is None:
        raise RuntimeError(
            f"C++17 compiler not found: {compiler}. Install a GCC-compatible "
            "compiler separately, or use --compiler with its executable path."
        )
    flags = ["-std=c++17", "-O3", "-fno-math-errno", "-shared",
             "-fno-fast-math", "-ffp-contract=off"]
    if native:
        flags.extend(["-march=native", "-mtune=native"])
    if sys.platform == "win32":
        # MinGW-w64 exports the unchanged extern-C ABI from the frozen source.
        # Also link MinGW's thread support statically. Otherwise libstdc++
        # can introduce libwinpthread-1.dll, which ctypes' restricted Windows
        # DLL search does not resolve from the compiler's PATH on clean hosts.
        flags.extend(["-static", "-static-libgcc", "-static-libstdc++",
                      "-Wl,--export-all-symbols"])
    else:
        flags.append("-fPIC")
        if sys.platform == "darwin":
            flags[flags.index("-shared")] = "-dynamiclib"
    command = [executable, *flags, str(source), "-o", str(target)]
    source_hash = _sha256(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_suffix(target.suffix + ".building")
    # Exclusivity avoids two explicit builds racing on the same output.
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        if target.exists() or manifest.exists():
            raise FileExistsError(f"Refusing to overwrite native artifacts: {target}")
        subprocess.run(command, check=True)
        if _sha256(source) != source_hash:
            raise RuntimeError("Native source changed during build; output must not be used")
        library = ctypes.CDLL(str(target))
        abi_values = {}
        for name in ("ea_cells_native_abi", "ea_cells_native_seed_abi"):
            function = getattr(library, name)
            function.argtypes = []
            function.restype = ctypes.c_int
            abi_values[name] = function()
            if abi_values[name] != 1:
                raise RuntimeError(f"Unexpected native ABI: {name}={abi_values[name]}")
        record = {
            "version": "EA", "source_sha256": source_hash,
            "output_sha256": _sha256(target), "command": command,
            "compiler": subprocess.check_output([executable, "--version"], text=True),
            "platform": platform.platform(), "machine": platform.machine(),
            "host_optimized": bool(native), "abi": abi_values,
            "compile_inside_measured_call": False,
        }
        with manifest.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
    finally:
        lock.unlink()
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Fresh library output; default: versioned user cache")
    parser.add_argument("--compiler", default="g++", help="GCC-compatible C++17 compiler")
    parser.add_argument("--native", action="store_true", help="Optimize for this CPU; rebuild on a different host")
    parser.add_argument("--show", action="store_true", help="Show a previously built library without compiling")
    args = parser.parse_args(argv)
    try:
        path = find_native_library() if args.show else build_native_library(
            args.output, args.compiler, args.native,
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"ea native build: {exc}\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
