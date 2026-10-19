"""Audit the verified fine-grid update against a preserved four-mode snapshot.

This is a local engineering comparison, not a dataset-wide accuracy study.
Both timed engines compute only the same four CV/CTRV pairings. Compilation,
warm-up, diagnostics, file I/O, and JSON generation are excluded from timings.
Numerical changes are recorded rather than treated as regressions by default.

Run from the checkout root, preserving the old source tree first::

    python tools/validate_paper_alignment.py --before-source /path/to/before/src \
        --frames 403 --repeats 3 --diagnostic-scenes 64

The old core is loaded with its own frozen dependencies under a private import
namespace. Its lazy dependency loader is relocated to that namespace; no old
solver function or numerical setting is modified. No Git or network access is
required and the preserved source is never changed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "tools"))
import core_ea
from validate_paper_migration import (
    MODE_KEYS, clean_json, read_demo_frames, same_value, sha256,
    source_hash, state_kwargs, time_summary, versions,
)

VALUE_KEYS = (*MODE_KEYS, "EA")
PUBLIC_FUNCTIONS = (
    "aggregate_ea_modes", "compute_single_mode_ea", "compute_single_mode_ea_timed",
    "compute_ea_modes", "compute_ea_modes_timed", "compute_ea", "compute_ea_timed",
)


def load_before_engine(source, namespace="_ea_alignment_before"):
    """Load the complete old four-mode dependency graph without current imports."""
    source = Path(source).resolve()
    paper_path = source / "evasive_acceleration" / "_paper"
    required = (source / "core_ea.py", *(paper_path / name for name in (
        "four_modes.py", "calculator.py", "config.py", "motion.py", "numerical_solver.py",
    )))
    for path in required:
        if not path.is_file():
            raise ValueError(f"Incomplete --before-source snapshot: missing {path}")
    if any(name == namespace or name.startswith(namespace + ".") for name in sys.modules):
        raise ValueError(f"Snapshot namespace is already in use: {namespace}")
    package = types.ModuleType(namespace)
    package.__path__ = [str(paper_path)]
    sys.modules[namespace] = package
    config = importlib.import_module(namespace + ".config")
    bridge = importlib.import_module(namespace + ".four_modes")
    spec = importlib.util.spec_from_file_location(namespace + ".compatibility_core", source / "core_ea.py")
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    engine._get_bridge = lambda: (config.EAConfig, config.MotionModel, bridge.compute_selected_modes)
    # An explicit guard against accidentally timing the current four-mode code.
    if Path(inspect.getfile(bridge)).resolve() != (paper_path / "four_modes.py").resolve():
        raise RuntimeError("Old engine dependency isolation failed")
    return engine, bridge


def classify_value(value):
    value = float(value)
    if math.isnan(value):
        return "invalid_nan"
    if value == math.inf:
        return "unresolved_or_current_contact_inf"
    if value == -math.inf or value < 0:
        return "invalid_negative"
    return "zero" if value == 0 else "finite_positive"


def compare_values(before, after, *, atol=1e-8):
    changes = []
    delta = {}
    transitions = {}
    for key in VALUE_KEYS:
        old, new = float(before[key]), float(after[key])
        if not same_value(old, new, atol=atol):
            changes.append(key)
        delta[key] = new - old if math.isfinite(old) and math.isfinite(new) else None
        transitions[key] = classify_value(old) + " -> " + classify_value(new)
    return {"before": before, "after": after, "delta": delta,
            "changed": changes, "semantic_transition": transitions}


def compute_values(engine, kwargs, settings):
    values = engine.compute_ea_modes(**kwargs, **settings)
    return {**values, "EA": engine.aggregate_ea_modes(values)}


def tree_hashes(source):
    source = Path(source)
    return {
        path.relative_to(source).as_posix(): sha256(path)
        for path in sorted(source.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".json"}
        and "__pycache__" not in path.parts
    }


def assert_before_frozen_hashes(source, reference):
    paper_path = Path(source) / "evasive_acceleration" / "_paper"
    actual = {name: source_hash(paper_path / name)
              for name in reference["reference"]["source_sha256_lf_normalized"]}
    expected = reference["reference"]["source_sha256_lf_normalized"]
    if actual != expected:
        raise ValueError("Before snapshot does not contain the independently frozen paper modules")
    return actual


def independent_sat_collision(pair, magnitude, direction_x, direction_y):
    """Vectorized SAT recheck, independent of the solver's candidate verifier.

    Uses the returned four-axis OBB geometry directly, including contact. No
    broad-phase shortcut and no internal ``verified`` flag are used as proof.
    ``t2_arr`` already contains one half of squared prediction time.
    """
    if not all(math.isfinite(float(value)) for value in (magnitude, direction_x, direction_y)):
        raise ValueError("Candidate magnitude and direction must be finite for SAT verification")
    relx = pair["relx_arr"] - magnitude * direction_x * pair["t2_arr"]
    rely = pair["rely_arr"] - magnitude * direction_y * pair["t2_arr"]
    projections = relx[:, None] * pair["axes_x"] + rely[:, None] * pair["axes_y"]
    contact = np.all(np.abs(projections) <= pair["sum_radius"], axis=1)
    indices = np.flatnonzero(contact)
    return bool(indices.size), None if not indices.size else float(pair["t_arr"][indices[0]])


def synthetic_scenes(count, seed):
    """Deterministic head-on, crossing, turning and parallel synthetic scenes."""
    rng = np.random.default_rng(seed)
    for index in range(count):
        category = index % 4
        a = [0., 0., float(rng.uniform(0.5, 20)), 0.,
             float(rng.uniform(3.5, 5.5)), float(rng.uniform(1.5, 2.4)),
             float(rng.uniform(-0.35, 0.35))]
        if category == 0:
            b = [float(rng.uniform(8, 55)), float(rng.uniform(-3, 3)),
                 float(rng.uniform(0.5, 18)), float(rng.uniform(2.8, 3.5))]
        elif category == 1:
            b = [float(rng.uniform(8, 45)), float(rng.uniform(-25, -6)),
                 float(rng.uniform(2, 18)), float(rng.uniform(1.2, 1.9))]
        elif category == 2:
            b = [float(rng.uniform(7, 35)), float(rng.uniform(-6, 6)),
                 float(rng.uniform(0, 15)), float(rng.uniform(-0.5, 0.5))]
        else:
            b = [float(rng.uniform(-12, 20)), float(rng.uniform(5, 12)),
                 float(rng.uniform(0, 20)), float(rng.uniform(-0.15, 0.15))]
        b.extend([float(rng.uniform(3.5, 5.5)), float(rng.uniform(1.5, 2.4)),
                  float(rng.uniform(-0.35, 0.35))])
        yield {"id": index, "category": ("head_on", "crossing", "turning", "parallel")[category],
               "kwargs": state_kwargs(a, b)}


def diagnostic_check(count, seed, settings):
    from evasive_acceleration._paper import four_modes

    if not hasattr(four_modes, "compute_selected_modes_with_diagnostics"):
        raise RuntimeError("Verified fine-grid diagnostics interface is unavailable")
    rows = []
    counts = Counter()
    for scene in synthetic_scenes(count, seed):
        kwargs = scene["kwargs"]
        ordered = (kwargs["xA"], kwargs["yA"], kwargs["vA"], kwargs["hA"],
                   kwargs["xB"], kwargs["yB"], kwargs["vB"], kwargs["hB"],
                   kwargs["lA"], kwargs["wA"], kwargs["lB"], kwargs["wB"],
                   kwargs["yawA"], kwargs["yawB"])
        state, config = core_ea._prepare_inputs(ordered, **settings)
        results = four_modes.compute_selected_modes_with_diagnostics(state, config)
        _coarse_pairs, fine_pairs = four_modes._prediction_pairs(state, config, four_modes.FOUR_PAIRS)
        api_values = core_ea.compute_ea_modes(**kwargs, **settings)
        scene_rows = []
        for key, pair in zip(MODE_KEYS, four_modes.FOUR_PAIRS):
            result = results[pair]
            value = float(result.value)
            counts[classify_value(value)] += 1
            collides, contact_time = (None, None)
            passed = same_value(api_values[key], value)
            if math.isfinite(value) and value >= 0:
                dx = float(result.direction_x) if value else 0.
                dy = float(result.direction_y) if value else 0.
                collides, contact_time = independent_sat_collision(fine_pairs[pair], value, dx, dy)
                passed = passed and bool(result.verified) and not collides
                if value:
                    passed = passed and math.isclose(math.hypot(dx, dy), 1., abs_tol=1e-10)
                counts["finite_candidates_independently_rechecked"] += 1
                if collides:
                    counts["independent_fine_grid_collisions"] += 1
            else:
                # +inf is an explicitly unresolved/current-contact result, not
                # a feasible candidate and not proof of physical impossibility.
                passed = passed and value == math.inf
            scene_rows.append({"mode": key, "value": value, "passed": passed,
                               "verified_flag": bool(result.verified),
                               "fine_grid_collision": collides, "first_contact_s": contact_time,
                               "direction_x": float(result.direction_x),
                               "direction_y": float(result.direction_y),
                               "search_grid": result.search_grid,
                               "evaluated_fine_directions": int(result.evaluated_fine_directions)})
        rows.append({**scene, "modes": scene_rows, "passed": all(row["passed"] for row in scene_rows)})
    return {"scene_count": count, "seed": seed, "counts": dict(counts), "scenes": rows,
            "passed": all(row["passed"] for row in rows),
            "scope": "Independent sampled-time SAT recheck, not a continuous-time safety proof"}


def api_checks(before):
    signatures = {
        name: {"before": str(inspect.signature(getattr(before, name))),
               "after": str(inspect.signature(getattr(core_ea, name)))}
        for name in PUBLIC_FUNCTIONS
    }
    for value in signatures.values():
        value["passed"] = value["before"] == value["after"]
    valid = state_kwargs([0, 0, 5, 0, 4.5, 1.8, 0], [0, 100, 5, 0, 4.5, 1.8, 0])
    invalid = {**valid, "vA": -1.}
    contact = {**valid, "yB": 1.8}
    semantic = {
        "noncolliding_zero": all(value == 0 for value in core_ea.compute_ea_modes(**valid).values()),
        "invalid_input_nan": all(math.isnan(value) for value in core_ea.compute_ea_modes(**invalid).values()),
        "current_contact_inf": all(value == math.inf for value in core_ea.compute_ea_modes(**contact).values()),
        "component_keys_preserved": set(core_ea.compute_ea_modes(**valid)) == set(MODE_KEYS),
        "aggregate_nan_propagation": math.isnan(core_ea.aggregate_ea_modes(dict.fromkeys(MODE_KEYS, math.nan))),
        "aggregate_inf_propagation": core_ea.aggregate_ea_modes(dict.fromkeys(MODE_KEYS, math.inf)) == math.inf,
    }
    return {"signatures": signatures, "semantic_checks": semantic,
            "passed": all(row["passed"] for row in signatures.values()) and all(semantic.values())}


def summarize_changes(rows):
    finite_deltas = {key: [row["delta"][key] for row in rows if row["delta"][key] is not None]
                     for key in VALUE_KEYS}
    return {
        "frames": len(rows), "changed_frames": sum(bool(row["changed"]) for row in rows),
        "changed_components": {key: sum(key in row["changed"] for row in rows) for key in VALUE_KEYS},
        "semantic_transitions": {
            key: dict(Counter(row["semantic_transition"][key] for row in rows)) for key in VALUE_KEYS
        },
        "finite_delta_summary": {
            key: {"count": len(values), "mean": float(np.mean(values)) if values else None,
                  "max_absolute": max(map(abs, values), default=None)}
            for key, values in finite_deltas.items()
        },
    }


def _run(argv=None, *, jit_cache_path=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-source", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=403)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--diagnostic-scenes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--input", action="append", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if min(args.frames, args.repeats) < 1 or args.diagnostic_scenes < 0:
        parser.error("frames/repeats must be positive; diagnostic-scenes cannot be negative")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = (args.output_dir or ROOT / "outputs" / "validation" / ("paper_alignment_" + stamp)).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("--output-dir must be absent or empty; existing results are never overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = ROOT / "tests" / "fixtures" / "paper_core_reference.json"
    reference = json.loads(fixture_path.read_text(encoding="utf-8"))
    old_hashes = assert_before_frozen_hashes(args.before_source, reference)
    before, old_bridge = load_before_engine(args.before_source)
    preserved_hashes = tree_hashes(args.before_source)
    current_hashes = tree_hashes(ROOT / "src")
    frames, input_files, available = read_demo_frames(
        args.input or sorted((ROOT / "demo_data").glob("*.csv")), args.frames,
    )
    settings = reference["reference"]["default_settings"]
    checks = api_checks(before)
    print(f"Comparing {len(frames)} demo frames and {len(reference['cases'])} frozen fixtures...", flush=True)
    fixture_rows = []
    for case in reference["cases"]:
        kwargs = state_kwargs(case["agent_a"], case["agent_b"])
        case_settings = {**settings, **case["settings"]}
        old = compute_values(before, kwargs, case_settings)
        new = compute_values(core_ea, kwargs, case_settings)
        expected = {key: math.inf if value == "inf" else float(value) for key, value in case["expected"].items()}
        fixture_rows.append({"id": case["id"], **compare_values(old, new),
                             "before_matches_frozen_reference": all(same_value(old[key], expected[key]) for key in VALUE_KEYS)})
    frame_rows = []
    for frame in frames:
        old = compute_values(before, frame["kwargs"], settings)
        new = compute_values(core_ea, frame["kwargs"], settings)
        frame_rows.append({"file": frame["file"], "csv_row": frame["csv_row"], **compare_values(old, new)})

    engines = {"before_four_modes": before, "after_verified_four_modes": core_ea}
    warmup, timings, measured = {}, {name: [] for name in engines}, []
    print("Warming both complete four-mode engines on every selected frame (excluded)...", flush=True)
    for name, engine in engines.items():
        start = time.perf_counter()
        for frame in frames:
            engine.compute_ea_modes(**frame["kwargs"], **settings)
        warmup[name] = time.perf_counter() - start
    print(f"Timing {args.repeats} repetitions with alternating engine order...", flush=True)
    for repeat in range(args.repeats):
        ordered = list(engines.items())
        if repeat % 2:
            ordered.reverse()
        for name, engine in ordered:
            for frame in frames:
                start = time.perf_counter()
                engine.compute_ea_modes(**frame["kwargs"], **settings)
                elapsed = time.perf_counter() - start
                timings[name].append(elapsed)
                measured.append({"engine": name, "repeat": repeat + 1, "file": frame["file"],
                                 "csv_row": frame["csv_row"], "elapsed_ms": elapsed * 1000.})
    print(f"Independently rechecking {args.diagnostic_scenes} synthetic scenes on the fine grid...", flush=True)
    diagnostics = diagnostic_check(args.diagnostic_scenes, args.seed, settings) if args.diagnostic_scenes else None
    snapshot_unchanged = preserved_hashes == tree_hashes(args.before_source)
    current_source_unchanged = current_hashes == tree_hashes(ROOT / "src")
    inputs_unchanged = all(sha256(row["path"]) == row["sha256"] for row in input_files)
    passed = (checks["passed"] and snapshot_unchanged and current_source_unchanged and inputs_unchanged
              and all(row["before_matches_frozen_reference"] for row in fixture_rows)
              and (diagnostics is None or diagnostics["passed"]))
    report = {
        "purpose": "Code-to-paper numerical engineering alignment only; no new empirical method-validity claim",
        "created_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "processor": platform.processor(), "versions": versions(),
        "core_id_before": before.CORE_ID, "core_id_after": core_ea.CORE_ID,
        "before_source": str(args.before_source.resolve()), "before_source_sha256": preserved_hashes,
        "before_frozen_paper_sha256_lf_normalized": old_hashes,
        "before_bridge_module": old_bridge.__name__,
        "dependency_isolation": "Complete frozen private namespace; only lazy import target relocated, no solver changes",
        "after_source_sha256": current_hashes,
        "after_source_unchanged_during_run": current_source_unchanged,
        "validation_tool_sha256": sha256(Path(__file__)),
        "input_files": input_files, "inputs_unchanged": inputs_unchanged,
        "before_source_unchanged": snapshot_unchanged,
        "available_frames": available, "selected_frames": len(frames),
        "selected_inputs_sha256": hashlib.sha256(json.dumps(frames, sort_keys=True).encode()).hexdigest(),
        "fixture_sha256": sha256(fixture_path), "settings": settings,
        "yaw_rate_source": "Provided yaw rates; identical observations in both engines",
        "api": checks, "fixture_comparisons": fixture_rows,
        "demo_summary": summarize_changes(frame_rows), "demo_comparisons": frame_rows,
        "diagnostics": diagnostics, "warmup_excluded_seconds": warmup,
        "numba_import_available": bool(core_ea._NUMBA_OK),
        "NUMBA_DISABLE_JIT": os.environ.get("NUMBA_DISABLE_JIT"),
        "isolated_temporary_numba_cache": jit_cache_path,
        "repeats": args.repeats,
        "timing": {name: time_summary(samples) for name, samples in timings.items()},
        "timing_scope": "Same four pairings, warm compute_ea_modes only; no reference nine-mode CTRA calls",
        "numerical_comparison_note": "Changes are expected from final feasibility checks and uniform fine refinement; reported rather than required to equal old results",
        "passed": passed,
    }
    with (output_dir / "timings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(measured[0]))
        writer.writeheader()
        writer.writerows(measured)
    (output_dir / "report.json").write_text(json.dumps(clean_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Changed demo frames: {report['demo_summary']['changed_frames']}/{len(frames)}")
    print(f"Old snapshot fixture check: {sum(row['before_matches_frozen_reference'] for row in fixture_rows)}/{len(fixture_rows)}")
    for name, samples in timings.items():
        summary = time_summary(samples)
        print(f"{name}: mean {summary['mean_ms']:.3f} ms; median {summary['median_ms']:.3f} ms; P95 {summary['p95_ms']:.3f} ms")
    print(f"Passed: {passed}; report: {output_dir / 'report.json'}")
    return 0 if passed else 1


def main(argv=None):
    try:
        from numba import config as numba_config
    except Exception:
        return _run(argv)
    previous = numba_config.CACHE_DIR
    with tempfile.TemporaryDirectory(prefix="ea-align-") as cache:
        numba_config.CACHE_DIR = cache
        try:
            return _run(argv, jit_cache_path=cache)
        finally:
            numba_config.CACHE_DIR = previous


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
