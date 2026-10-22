"""Install a built wheel in a fresh environment and exercise packaged v53.

Tests import from outside the source checkout and build from the installed
C++ resource, so editable installs cannot mask missing package contents.
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
root = Path(ea.__file__).resolve().parent / '_experimental'
manifest = json.loads((root / 'v53_sources.json').read_text(encoding='utf-8'))
assert {p.name for p in root.glob('*.py')} == {
    name for name in manifest['sources'] if name.endswith('.py')
} | {'__init__.py'}
for name, expected in manifest['sources'].items():
    assert hashlib.sha256((root / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == expected
with ea.use_backend('v53'):
    a = ea.RoadUserState(0, 0, 8, 0, 4.8, 1.8, .1)
    b = ea.RoadUserState(20, 1, 4, math.pi, 4.5, 1.8, 0)
    value = ea.compute_ea(a, b)
    assert abs(value - 1.5366117876437317) <= .001, value
    provenance = core_ea.get_core_provenance()
    assert provenance['core_id'] == 'continuous-certified-v53-20261022'
    assert provenance['native_library_sha256']
    assert ea.causal_yaw_rate([0, .1], [0, .01]).tolist() == [0., .1]
assert ea.get_backend() == 'paper'
print('Isolated wheel: frozen v53 resources, native build, public API and yaw-rate checks passed')
'''


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    wheels = list(args.wheel_dir.resolve().glob("*.whl"))
    if len(wheels) != 1:
        parser.error("expected exactly one wheel in --wheel-dir")
    with tempfile.TemporaryDirectory(prefix="ea-v53-wheel-") as directory:
        sandbox = Path(directory)
        environment = sandbox / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run([str(python), "-m", "pip", "install", str(wheels[0]) + "[fast]"],
                       check=True, cwd=sandbox)
        library = sandbox / ("libea_cells.dll" if os.name == "nt" else
                             "libea_cells.dylib" if sys.platform == "darwin" else "libea_cells.so")
        subprocess.run([str(python), "-I", "-m", "evasive_acceleration.native_build",
                        "--output", str(library)], check=True, cwd=sandbox)
        env = os.environ.copy()
        env["EA_NATIVE_CELLS_LIBRARY"] = str(library)
        subprocess.run([str(python), "-I", "-c", CHECK], check=True, cwd=sandbox, env=env)
        subprocess.run([str(python), "-I", "-m", "evasive_acceleration", "--backend", "v53",
                        "benchmark", "--max-frames", "1", "--output-dir", str(sandbox / "benchmark")],
                       check=True, cwd=sandbox, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
