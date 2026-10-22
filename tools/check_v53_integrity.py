"""Check that packaged v53 kernels still match the independently tested source.

Normalize checkout line endings only. Never rewrite a source or refresh the
baseline hashes automatically: a numerical change needs a new validation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_v53_integrity(project_root: Path = PROJECT_ROOT) -> list[str]:
    root = project_root / "src" / "evasive_acceleration" / "_experimental"
    manifest = root / "v53_sources.json"
    if not manifest.is_file():
        return ["Missing frozen v53 source manifest"]
    record = json.loads(manifest.read_text(encoding="utf-8"))
    failures = []
    for name, expected in record["sources"].items():
        path = root / name
        if path.parent != root or not path.is_file():
            failures.append(f"Missing or invalid v53 source: {name}")
            continue
        actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            failures.append(f"v53 source changed: {name}; expected {expected}, got {actual}")
    actual_sources = {path.name for path in root.iterdir()
                      if path.suffix in (".py", ".cpp") and path.name != "__init__.py"}
    if actual_sources != set(record["sources"]):
        failures.append("Only the frozen v53 transitive source closure may be packaged")
    return failures


def main() -> int:
    failures = check_v53_integrity()
    for failure in failures:
        print("[FAIL] " + failure)
    if not failures:
        print("[OK] All 15 v53 kernel sources match the validated snapshot")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
