"""Verify the approved paper EA snapshot and unchanged auxiliary algorithms.

Only CRLF/LF line endings are normalized, because Git may convert them on
checkout. The core transition on 2026-10-20 was explicitly approved: these four
paper files replace the original hybrid analytical/numerical core as the frozen
reference. The baseline metrics and causal yaw estimator remain unchanged.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASHES = {
    "src/evasive_acceleration/_paper/calculator.py": "ba4aad2ed1c5da9841a421af8ddad0193c405a770176ff703427b564757a4693",
    "src/evasive_acceleration/_paper/config.py": "6f51954a8d24590796b747638522f927378de4273aaaad293387e41dc5b4ae33",
    "src/evasive_acceleration/_paper/motion.py": "709e4bf8663bcf580e27ede058d548bf0d40a8da35f199823fb09c5650867625",
    "src/evasive_acceleration/_paper/numerical_solver.py": "dc90a8ce7aa67afda0be250c88a9389cd0168bb36a7aadee4d5e9cb0e52ee7ff",
    "src/baseline_risk_metrics.py": "cba9c7dcf2377391e6c11eb5003e7ed240ed61bc430f8e5a618350f82529f0d9",
    "src/yaw_rate.py": "69f73016813d64ea5a43775678841bea072ee9377dbb5a093724cdc67f332d8e",
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
        print(f"[OK] Frozen source: {relative_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
