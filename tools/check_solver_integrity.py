"""Check that EA kernels still match the independently tested source.

Normalize checkout line endings only. Never rewrite a source or refresh the
baseline hashes automatically: a numerical change needs a new validation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUXILIARY_HASHES = {
    "src/baseline_risk_metrics.py": "cba9c7dcf2377391e6c11eb5003e7ed240ed61bc430f8e5a618350f82529f0d9",
    "src/yaw_rate.py": "69f73016813d64ea5a43775678841bea072ee9377dbb5a093724cdc67f332d8e",
}


def check_solver_integrity(project_root: Path = PROJECT_ROOT) -> list[str]:
    root = project_root / "src" / "evasive_acceleration" / "_experimental"
    manifest = root / "v53_sources.json"
    if not manifest.is_file():
        return ["Missing frozen EA source manifest"]
    record = json.loads(manifest.read_text(encoding="utf-8"))
    failures = []
    for name, expected in record["sources"].items():
        path = root / name
        if path.parent != root or not path.is_file():
            failures.append(f"Missing or invalid EA source: {name}")
            continue
        actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            failures.append(f"EA source changed: {name}; expected {expected}, got {actual}")
    actual_sources = {path.name for path in root.iterdir()
                      if path.suffix in (".py", ".cpp") and path.name != "__init__.py"}
    if actual_sources != set(record["sources"]):
        failures.append("Only the frozen EA transitive source closure may be packaged")
    # The public release changes routing, not the causal estimator or baselines.
    for relative, expected in AUXILIARY_HASHES.items():
        path = project_root / relative
        if not path.is_file():
            failures.append(f"Missing auxiliary source: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            failures.append(f"Auxiliary algorithm changed: {relative}; expected {expected}, got {actual}")
    return failures


def main() -> int:
    failures = check_solver_integrity()
    for failure in failures:
        print("[FAIL] " + failure)
    if not failures:
        print("[OK] EA kernels, baseline metrics and causal yaw-rate sources match validated snapshots")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
