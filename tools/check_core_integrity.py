"""Verify that release usability changes have not altered the EA algorithms.

Only CRLF/LF line endings are normalized, because Git may convert them on
checkout. All other source bytes must match the frozen v2 preparation baseline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASHES = {
    "src/core_ea.py": "af9306d32d3ca6a72f277df6e0838c59f579ed94867944eaf732ccf6f2c85cad",
    "src/baseline_risk_metrics.py": "cba9c7dcf2377391e6c11eb5003e7ed240ed61bc430f8e5a618350f82529f0d9",
}


def check_core_integrity(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Return actionable failures; never rewrite source files or expected hashes."""
    failures = []
    for relative_path, expected in EXPECTED_HASHES.items():
        path = project_root / relative_path
        if not path.is_file():
            failures.append(f"Missing frozen source file: {relative_path}")
            continue
        data = path.read_bytes().replace(b"\r\n", b"\n")
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            failures.append(
                f"{relative_path}: source changed; expected {expected}, got {actual}. "
                "Review the algorithm change explicitly; do not refresh this baseline "
                "as part of a documentation or usability change."
            )
    return failures


def main() -> int:
    failures = check_core_integrity()
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    for relative_path in EXPECTED_HASHES:
        print(f"[OK] Unchanged algorithm: {relative_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
