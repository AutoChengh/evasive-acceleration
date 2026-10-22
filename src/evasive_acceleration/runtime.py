"""The sole EA runtime, with explicit preparation and numerical provenance.

The validated numerical kernels are unchanged. The native library must be
built before evaluation; no EA call invokes a compiler or another solver.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib
import math
import os
from pathlib import Path
import threading


CORE_ID = "ea-continuous-certified-20261022"
_native_import_lock = threading.Lock()
_loaded_native_sha256: dict[Path, str] = {}
_SOLVER_MODULES = (
    "circle_proposal", "continuous", "continuous_adaptive_cells",
    "continuous_cells", "continuous_compact_kernel", "continuous_native_cells",
    "continuous_parallel", "continuous_penalty_fallback",
    "continuous_resilient_cells", "continuous_tail", "continuous_tail_kernel",
    "normalized_refinement", "penalty_safety", "priority_safety",
)

@lru_cache(maxsize=1)
def _get_runtime():
    """Load the validated kernels lazily, after checking required dependencies."""
    try:
        import numba
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            'EA requires Numba. Install the package dependencies with '
            'python -m pip install -e ., then build the native '
            'library as described in docs/runtime.md.'
        ) from exc
    if numba.config.DISABLE_JIT:
        raise RuntimeError(
            "EA requires enabled Numba JIT; unset NUMBA_DISABLE_JIT."
        )

    from .native_build import find_native_library

    library_path = Path(find_native_library()).resolve()
    # The frozen loader reads this variable at import time. Use a temporary
    # override rather than leaving process-global backend configuration behind.
    with _native_import_lock:
        initial_digest = _check_loaded_native_digest(library_path)
        previous = os.environ.get("EA_NATIVE_CELLS_LIBRARY")
        # Discovery accepts relative/tilde paths and treats an empty value as
        # unset. The frozen loader does not normalize those spellings, so it
        # must always receive the resolved absolute path during first import.
        os.environ["EA_NATIVE_CELLS_LIBRARY"] = str(library_path)
        try:
            adaptive = importlib.import_module(
                "evasive_acceleration._experimental.continuous_adaptive_cells"
            )
            native = importlib.import_module(
                "evasive_acceleration._experimental.continuous_native_cells"
            )
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Cannot load the EA native runtime. Rebuild the native library "
                "for this Python/CPU and start a fresh Python process."
            ) from exc
        finally:
            if previous is None:
                os.environ.pop("EA_NATIVE_CELLS_LIBRARY", None)
            else:
                os.environ["EA_NATIVE_CELLS_LIBRARY"] = previous
        if Path(native._library_path).resolve() != library_path:
            raise RuntimeError(
                "A different EA native library is already loaded. Start a fresh "
                "Python process after selecting EA_NATIVE_CELLS_LIBRARY."
            )
        # Keep the pre-load identity even on a concurrent-replacement error;
        # retrying in the same process must not relabel already mapped code.
        _loaded_native_sha256.setdefault(library_path, initial_digest)
        final_digest = hashlib.sha256(library_path.read_bytes()).hexdigest()
        if final_digest != initial_digest:
            raise RuntimeError(
                "The EA native library changed while loading. Start a fresh "
                "Python process after restoring/rebuilding the library."
            )
    return adaptive, library_path


def _check_loaded_native_digest(library_path: Path) -> str:
    """Check disk-vs-loaded identity outside the timed per-frame path."""
    digest = hashlib.sha256(library_path.read_bytes()).hexdigest()
    original = _loaded_native_sha256.get(library_path)
    if original is not None and digest != original:
        raise RuntimeError(
            "The EA native library was replaced after loading. Start a fresh "
            "Python process before recording results or provenance."
        )
    return digest


def _check_library_override(library_path: Path) -> None:
    """Reject a changed explicit path after the native runtime was loaded."""
    override = os.environ.get("EA_NATIVE_CELLS_LIBRARY")
    if override and override != str(library_path):
        if Path(override).expanduser().resolve() != library_path:
            raise RuntimeError(
            "EA_NATIVE_CELLS_LIBRARY changed after EA was loaded. Start a "
                "fresh Python process to use a different native library."
            )


def _preflight_runtime() -> None:
    """Check required runtime dependencies before a CLI/batch creates outputs."""
    _adaptive, library_path = _get_runtime()
    _check_library_override(library_path)
    _check_loaded_native_digest(library_path)


def _effective_yaw(state_values, pair) -> tuple[float, float]:
    rates = (float(state_values[6]) if pair[0] else 0.0,
             float(state_values[13]) if pair[1] else 0.0)
    return tuple(0.0 if abs(rate) < 1e-6 else rate for rate in rates)


def compute_modes(state_values, config, pairs=None):
    """Bridge the unchanged public state/config layout to EA.

    The default 0.001 m/s² path is the frozen, fully benchmarked entry point.
    The public tolerance also reaches the affine CV/CV primitive when callers
    explicitly request another tolerance; legacy grid/direction options remain
    accepted but do not define EA's continuous-time accuracy.
    """
    from ._paper.four_modes import FOUR_PAIRS

    adaptive, library_path = _get_runtime()
    _check_library_override(library_path)
    selected_pairs = FOUR_PAIRS if pairs is None else tuple(pairs)
    tolerance = float(config.solver_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("solver_tolerance must be finite and positive")
    budget = adaptive.CellBudget(acceleration_tolerance=tolerance)
    if tolerance == 0.001:
        return adaptive.compute_selected_modes(
            state_values, config, pairs=selected_pairs, budget=budget,
        )

    # Frozen v53's zero-yaw branch uses the preserved primitive's default
    # budget. Adapt only this branch so a public custom tolerance is honoured.
    zero_pairs = tuple(pair for pair in selected_pairs
                       if _effective_yaw(state_values, pair) == (0.0, 0.0))
    turning_pairs = tuple(pair for pair in selected_pairs if pair not in zero_pairs)
    values = adaptive.compute_selected_modes(
        state_values, config, pairs=turning_pairs, budget=budget,
    ) if turning_pairs else {}
    if zero_pairs:
        from ._experimental import continuous_parallel as preserved

        result = preserved.solve_pair(
            state_values, zero_pairs[0], config,
            preserved.RetainedBudget(acceleration_tolerance=tolerance),
        )
        if not (result.continuous_verified and result.angular_converged) and (
            result.status not in (
                "current-contact-or-uncertainty", "certified-cap-infeasible",
            )
        ):
            raise RuntimeError(
                "The EA zero-yaw search is unresolved at the requested "
                "tolerance; no inaccurate scalar is returned."
            )
        values.update({pair: result.value for pair in zero_pairs})
    return {pair: values[pair] for pair in selected_pairs}


def get_runtime_provenance() -> dict:
    """Hash the actual frozen runtime and native binary, not the old solver."""
    _adaptive, library_path = _get_runtime()
    _check_library_override(library_path)
    native_digest = _check_loaded_native_digest(library_path)
    package = Path(__file__).resolve().parent
    paths = {"_experimental/" + name + ".py":
             package / "_experimental" / (name + ".py") for name in _SOLVER_MODULES}
    paths.update({
        "_experimental/native_cells.cpp": package / "_experimental/native_cells.cpp",
        "runtime.py": Path(__file__).resolve(),
        "native_build.py": package / "native_build.py",
        "core_ea.py": package.parent / "core_ea.py",
    })
    paths.update({"_paper/" + name: package / "_paper" / name
                  for name in ("__init__.py", "config.py", "four_modes.py", "numerical_solver.py")})
    return {
        "core_id": CORE_ID,
        "backend": "EA",
        "solver": "continuous-time bounded acceleration search",
        "runtime_source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(paths.items())
        },
        "native_library_path": str(library_path),
        "native_library_sha256": native_digest,
        "acceleration_tolerance_default_mps2": 0.001,
        "time_validation": "continuous whole-horizon conservative certificate",
        "unresolved_policy": "raise; never return an uncertified scalar",
    }
