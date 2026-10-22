"""Validate the public adapter against the frozen v53 four-mode entry point.

Use synthetic inputs by default, or a local NPZ containing an (N,14) ``states``
array. Private datasets are never bundled or modified. Timing observations
cover the public four-mode call plus its mean; independent checks are outside.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import core_ea
import evasive_acceleration as ea
from evasive_acceleration.backends import _get_v53_runtime


MODE_KEYS = ("EA_CVCV", "EA_CVCT", "EA_CTCV", "EA_CTCT")
PAIRS = ((0, 0), (0, 1), (1, 0), (1, 1))


def synthetic_states():
    rng = np.random.default_rng(20261022)
    result = []
    for index in range(64):
        result.append([0., 0., rng.uniform(0., 15.), rng.uniform(-math.pi, math.pi),
                       4.8, 1.8, (0., .1, -.2, 6.)[index % 4],
                       rng.uniform(7., 25.), rng.uniform(-10., 10.),
                       rng.uniform(0., 15.), rng.uniform(-math.pi, math.pi),
                       4.5, 1.8, (0., -.1, .2, -6.)[(index // 4) % 4]])
    return np.asarray(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--independent-check", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("refusing to overwrite validation evidence")
    affinity = None
    if args.cpu is not None:
        if not hasattr(os, "sched_setaffinity"):
            parser.error("--cpu is unsupported on this platform")
        affinity = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {args.cpu})
    try:
        if args.states is None:
            states = synthetic_states()
        else:
            with np.load(args.states, allow_pickle=False) as data:
                states = np.asarray(data["states"], dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != 14 or not len(states):
            parser.error("expected nonempty (N,14) states array")
        runtime, _library = _get_v53_runtime()
        from evasive_acceleration._experimental import continuous as independent
        kwargs = [ea.build_compute_kwargs(ea.RoadUserState(*s[:7]),
                                         ea.RoadUserState(*s[7:])) for s in states]
        failures, times, maximum_delta, certified = [], [], 0., 0
        with ea.use_backend("v53"):
            provenance = core_ea.get_core_provenance()
            # Exercise every branch before measuring. No observed slow frame
            # is removed, capped, substituted or measured again for replacement.
            for values in kwargs:
                core_ea.aggregate_ea_modes(core_ea.compute_ea_modes(**values))
            for index, (state, values) in enumerate(zip(states, kwargs)):
                start = time.perf_counter_ns()
                actual = core_ea.compute_ea_modes(**values)
                mean = core_ea.aggregate_ea_modes(actual)
                times.append((time.perf_counter_ns() - start) / 1e6)
                expected = runtime.compute_selected_modes_with_diagnostics(state)
                for key, pair in zip(MODE_KEYS, PAIRS):
                    reference = expected[pair]
                    if actual[key] != reference.value:
                        failures.append({"index": index, "pair": list(pair), "test": "exact_adapter_identity"})
                    if math.isfinite(actual[key]) and math.isfinite(reference.value):
                        maximum_delta = max(maximum_delta, abs(actual[key] - reference.value))
                    if args.independent_check and math.isfinite(reference.value):
                        wa = state[6] if pair[0] else 0.
                        wb = state[13] if pair[1] else 0.
                        wa = 0. if abs(wa) < 1e-6 else wa
                        wb = 0. if abs(wb) < 1e-6 else wb
                        value = reference.value
                        ax = value * reference.direction_x if value else 0.
                        ay = value * reference.direction_y if value else 0.
                        status, _witness, _nodes = independent.certify(
                            state, wa, wb, ax, ay, 10., 1e-9, 500000,
                        )
                        certified += 1
                        if status != 1 or not (reference.continuous_verified and reference.angular_converged):
                            failures.append({"index": index, "pair": list(pair), "test": "independent_safety"})
                expected_mean = sum(expected[pair].value for pair in PAIRS) / 4.
                if mean != expected_mean:
                    failures.append({"index": index, "test": "unchanged_mean"})
            if core_ea.get_core_provenance() != provenance:
                failures.append({"test": "source_or_binary_changed"})
        report = {
            "passed": not failures, "cases": len(states), "mode_checks": 4 * len(states),
            "max_absolute_adapter_difference_mps2": maximum_delta,
            "independent_continuous_checks": certified, "failures": failures,
            "public_call_latency_ms": {
                "mean": float(np.mean(times)), "p95": float(np.percentile(times, 95)),
                "p99": float(np.percentile(times, 99)), "max": max(times),
                "over_20ms": sum(value > 20. for value in times),
            },
            "all_latency_observations_ms": times, "warmup_calls": len(states),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "provenance": provenance,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({key: report[key] for key in
                          ("passed", "cases", "mode_checks", "max_absolute_adapter_difference_mps2",
                           "independent_continuous_checks", "public_call_latency_ms")}))
        return int(bool(failures))
    finally:
        if affinity is not None:
            os.sched_setaffinity(0, affinity)


if __name__ == "__main__":
    raise SystemExit(main())
