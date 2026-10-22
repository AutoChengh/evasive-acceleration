"""Install source and wheel distributions in fresh environments and test EA.

Imports run outside the checkout, and the native library is built from each
installed C++ resource. Editable installs cannot mask missing package contents.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


CHECK = r'''
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import core_ea
import evasive_acceleration as ea
from evasive_acceleration import native_build

prefix = Path(sys.prefix).resolve()
for module in (ea, core_ea, native_build):
    assert Path(module.__file__).resolve().is_relative_to(prefix), module.__file__
package = Path(ea.__file__).resolve().parent
root = package / '_solver'
assert {p.name for p in root.glob('*.json')} == {'ea_sources.json'}
manifest = json.loads((root / 'ea_sources.json').read_text(encoding='utf-8'))
assert manifest['method'] == 'Evasive Acceleration (EA)'
assert manifest['core_id'] == 'ea-continuous-certified-20261022'
assert {p.name for p in root.glob('*.py')} == {
    name for name in manifest['sources'] if name.endswith('.py')
} | {'__init__.py'}
for name, expected in manifest['sources'].items():
    assert hashlib.sha256((root / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == expected
for name in ('get_backend', 'use_backend', 'BACKEND_NAMES'):
    assert not hasattr(ea, name), name
for name in ('backends', '_experimental', '_paper', '_model.calculator',
             '_model.verified_solver', '_model.verified_calculator', '_model.motion',
             '_model.numerical_solver'):
    assert not (Path(ea.__file__).resolve().parent / (name.replace('.', '/') + '.py')).exists(), name
    try:
        importlib.import_module('evasive_acceleration.' + name)
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError('Unsupported solver module is installed: ' + name)
assert not (package / '_experimental').exists()
assert not (package / '_paper').exists()
assert (package / '_model' / 'constants.py').is_file()
a = ea.RoadUserState(0, 0, 8, 0, 4.8, 1.8, .1)
b = ea.RoadUserState(20, 1, 4, math.pi, 4.5, 1.8, 0)
value = ea.compute_ea(a, b)
assert abs(value - 1.5366117876437317) <= .001, value
provenance = core_ea.get_core_provenance()
assert provenance['core_id'] == 'ea-continuous-certified-20261022'
assert provenance['backend'] == 'EA'
assert provenance['native_library_sha256']
assert ea.causal_yaw_rate([0, .1], [0, .01]).tolist() == [0., .1]
print('Isolated installation: frozen EA resources, native build, public API and causal yaw-rate passed')
'''


def check_distribution(distribution: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="ea-package-check-") as directory:
        sandbox = Path(directory)
        environment = sandbox / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run([str(python), "-m", "pip", "install", str(distribution)],
                       check=True, cwd=sandbox)
        subprocess.run([str(python), "-m", "pip", "check"], check=True, cwd=sandbox)
        library = sandbox / ("libea_cells.dll" if os.name == "nt" else
                             "libea_cells.dylib" if sys.platform == "darwin" else "libea_cells.so")
        subprocess.run([str(python), "-I", "-m", "evasive_acceleration.native_build",
                        "--output", str(library)], check=True, cwd=sandbox)
        environment_variables = os.environ.copy()
        environment_variables["EA_NATIVE_CELLS_LIBRARY"] = str(library)
        subprocess.run([str(python), "-I", "-c", CHECK],
                       check=True, cwd=sandbox, env=environment_variables)
        subprocess.run([str(python), "-I", "-m", "evasive_acceleration", "benchmark",
                        "--max-frames", "1", "--output-dir", str(sandbox / "benchmark")],
                       check=True, cwd=sandbox, env=environment_variables)
        result = subprocess.run(
            [str(python), "-I", "-m", "evasive_acceleration", "--backend", "paper", "single"],
            cwd=sandbox, env=environment_variables, capture_output=True, text=True,
        )
        assert result.returncode == 2, "obsolete selector was accepted"
        assert "--backend was removed" in result.stderr
        assert "one implementation" in result.stderr
        print(f"[OK] Isolated source/wheel verification: {distribution.name}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, required=True,
                        help="Fresh build directory containing exactly one wheel and one sdist")
    args = parser.parse_args(argv)
    directory = args.wheel_dir.resolve()
    wheels = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        parser.error("expected exactly one wheel and one source distribution")
    for distribution in wheels + sources:
        check_distribution(distribution)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
