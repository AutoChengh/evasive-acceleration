"""Explicit, context-local selection of the preserved and v53 EA engines.

The existing API uses ``paper`` unless the caller opts in with
``with use_backend("v53")``. Selecting v53 never falls back to the paper grid
search. Its native library must be built before evaluation; no measured call
invokes a compiler. Backend selection does not modify the yaw-rate estimator.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import hashlib
import importlib
import math
import os
from pathlib import Path
import threading


BACKEND_NAMES = ("paper", "v53")
V53_CORE_ID = "continuous-certified-v53-20261022"
_backend: ContextVar[str] = ContextVar("evasive_acceleration_backend", default="paper")
_native_import_lock = threading.Lock()
_loaded_native_sha256: dict[Path, str] = {}
_V53_MODULES = (
    "circle_proposal", "continuous", "continuous_adaptive_cells",
    "continuous_cells", "continuous_compact_kernel", "continuous_native_cells",
    "continuous_parallel", "continuous_penalty_fallback",
    "continuous_resilient_cells", "continuous_tail", "continuous_tail_kernel",
    "normalized_refinement", "penalty_safety", "priority_safety",
)

__all__ = ["BACKEND_NAMES", "get_backend", "use_backend"]


def get_backend() -> str:
    """Return the selected backend in the current thread/async context."""
    return _backend.get()


@contextmanager
def use_backend(name: str):
    """Select an EA backend temporarily, restoring it even after an error.

    Existing state, history, CSV and timed APIs all honour this selection.
    Dependencies are checked when v53 is evaluated, not when importing EA or
    requesting CLI help. A missing v53 dependency is an explicit error.
    """
    if not isinstance(name, str) or name not in BACKEND_NAMES:
        raise ValueError("backend must be 'paper' or 'v53'")
    token = _backend.set(name)
    try:
        yield
    finally:
        _backend.reset(token)


@lru_cache(maxsize=1)
def _get_v53_runtime():
    """Load the frozen v53 modules only after explicit backend selection."""
    try:
        import numba
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            'The v53 backend requires Numba. Install the accelerated extras '
            'with python -m pip install -e ".[fast]", then build the v53 native '
            'library as described in docs/runtime_v53.md.'
        ) from exc
    if numba.config.DISABLE_JIT:
        raise RuntimeError(
            "The v53 backend requires enabled Numba JIT; unset NUMBA_DISABLE_JIT."
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
                "Cannot load the v53 native runtime. Rebuild the native library "
                "for this Python/CPU and start a fresh Python process."
            ) from exc
        finally:
            if previous is None:
                os.environ.pop("EA_NATIVE_CELLS_LIBRARY", None)
            else:
                os.environ["EA_NATIVE_CELLS_LIBRARY"] = previous
        if Path(native._library_path).resolve() != library_path:
            raise RuntimeError(
                "A different v53 native library is already loaded. Start a fresh "
                "Python process after selecting EA_NATIVE_CELLS_LIBRARY."
            )
        # Keep the pre-load identity even on a concurrent-replacement error;
        # retrying in the same process must not relabel already mapped code.
        _loaded_native_sha256.setdefault(library_path, initial_digest)
        final_digest = hashlib.sha256(library_path.read_bytes()).hexdigest()
        if final_digest != initial_digest:
            raise RuntimeError(
                "The v53 native library changed while loading. Start a fresh "
                "Python process after restoring/rebuilding the library."
            )
    return adaptive, library_path


def _check_loaded_native_digest(library_path: Path) -> str:
    """Check disk-vs-loaded identity outside the timed per-frame path."""
    digest = hashlib.sha256(library_path.read_bytes()).hexdigest()
    original = _loaded_native_sha256.get(library_path)
    if original is not None and digest != original:
        raise RuntimeError(
            "The v53 native library was replaced after loading. Start a fresh "
            "Python process before recording results or provenance."
        )
    return digest


def _check_library_override(library_path: Path) -> None:
    """Reject a changed explicit path after the native runtime was loaded."""
    override = os.environ.get("EA_NATIVE_CELLS_LIBRARY")
    if override and override != str(library_path):
        if Path(override).expanduser().resolve() != library_path:
            raise RuntimeError(
                "EA_NATIVE_CELLS_LIBRARY changed after v53 was loaded. Start a "
                "fresh Python process to use a different native library."
            )


def _preflight_backend() -> None:
    """Check optional runtime dependencies before a CLI/batch creates outputs."""
    if get_backend() == "v53":
        _adaptive, library_path = _get_v53_runtime()
        _check_library_override(library_path)
        _check_loaded_native_digest(library_path)


def _effective_yaw(state_values, pair) -> tuple[float, float]:
    rates = (float(state_values[6]) if pair[0] else 0.0,
             float(state_values[13]) if pair[1] else 0.0)
    return tuple(0.0 if abs(rate) < 1e-6 else rate for rate in rates)


def compute_v53_modes(state_values, config, pairs=None):
    """Bridge the unchanged public state/config layout to certified v53.

    The default 0.001 m/s² path is the frozen, fully benchmarked entry point.
    The public tolerance also reaches the affine CV/CV primitive when callers
    explicitly request another tolerance; legacy grid/direction options remain
    accepted but do not define v53's continuous-time accuracy.
    """
    from ._paper.four_modes import FOUR_PAIRS

    adaptive, library_path = _get_v53_runtime()
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
                "The v53 zero-yaw search is unresolved at the requested "
                "tolerance; no inaccurate scalar is returned."
            )
        values.update({pair: result.value for pair in zero_pairs})
    return {pair: values[pair] for pair in selected_pairs}


def _v53_provenance() -> dict:
    """Hash the actual frozen runtime and native binary, not the old solver."""
    _adaptive, library_path = _get_v53_runtime()
    _check_library_override(library_path)
    native_digest = _check_loaded_native_digest(library_path)
    package = Path(__file__).resolve().parent
    paths = {"_experimental/" + name + ".py":
             package / "_experimental" / (name + ".py") for name in _V53_MODULES}
    paths.update({
        "_experimental/native_cells.cpp": package / "_experimental/native_cells.cpp",
        "backends.py": Path(__file__).resolve(),
        "native_build.py": package / "native_build.py",
        "core_ea.py": package.parent / "core_ea.py",
    })
    return {
        "core_id": V53_CORE_ID,
        "backend": "v53",
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
