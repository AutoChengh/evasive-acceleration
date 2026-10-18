"""Strictly causal yaw-rate estimation from timestamped heading observations.

The estimator implements the method documented for the EA release:

1. wrapped backward heading difference using the actual elapsed time;
2. a rate EWMA with ``alpha = 1 - exp(-dt / tau)``;
3. an output-only deadband and rounding step.

Only the current and earlier observations are used.  The full-precision EWMA
state is never replaced by the rounded public output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


DEFAULT_TAU_S = 0.6
DEFAULT_DEADBAND_RAD_S = 0.002
DEFAULT_DECIMALS = 4


__all__ = [
    "DEFAULT_TAU_S",
    "DEFAULT_DEADBAND_RAD_S",
    "DEFAULT_DECIMALS",
    "CausalYawRateEWMA",
    "YawRateEstimate",
    "causal_yaw_rate",
    "causal_yawrate",
    "wrap_angle_difference",
]


@dataclass(frozen=True)
class YawRateEstimate:
    """Diagnostics for the most recent causal yaw-rate update."""

    value_rad_s: float
    initialized: bool
    sample_count: int
    raw_rate_rad_s: Optional[float]
    state_rad_s: Optional[float]


def wrap_angle_difference(delta_rad: float) -> float:
    """Map an angular difference to ``[-pi, pi)``."""
    delta_rad = float(delta_rad)
    if not math.isfinite(delta_rad):
        raise ValueError("heading difference must be finite")
    return (delta_rad + math.pi) % (2.0 * math.pi) - math.pi


class CausalYawRateEWMA:
    """Online, strictly causal yaw-rate estimator.

    The first observation returns an initialization placeholder of zero.  The
    first backward difference initializes the internal state on the second
    observation.  Callers should inspect :attr:`last_estimate` when they need
    to distinguish the placeholder from a measured zero yaw rate.
    """

    def __init__(
        self,
        tau_s: float = DEFAULT_TAU_S,
        deadband_rad_s: float = DEFAULT_DEADBAND_RAD_S,
        decimals: int = DEFAULT_DECIMALS,
    ) -> None:
        if not math.isfinite(tau_s) or tau_s <= 0.0:
            raise ValueError("tau_s must be a finite positive value")
        if not math.isfinite(deadband_rad_s) or deadband_rad_s < 0.0:
            raise ValueError("deadband_rad_s must be a finite non-negative value")
        if not isinstance(decimals, int) or isinstance(decimals, bool) or decimals < 0:
            raise ValueError("decimals must be a non-negative integer")

        self.tau_s = float(tau_s)
        self.deadband_rad_s = float(deadband_rad_s)
        self.decimals = decimals
        self.previous_time: Optional[float] = None
        self.previous_heading: Optional[float] = None
        self.rate_state: Optional[float] = None
        self.sample_count = 0
        self.last_estimate: Optional[YawRateEstimate] = None

    def reset(self) -> None:
        """Discard the history, for example after a track-ID change."""
        self.previous_time = None
        self.previous_heading = None
        self.rate_state = None
        self.sample_count = 0
        self.last_estimate = None

    def _format_output(self, state: float) -> float:
        if abs(state) < self.deadband_rad_s:
            return 0.0
        output = float(np.round(state, self.decimals))
        return 0.0 if output == 0.0 else output

    def update(self, time_s: float, heading_rad: float) -> float:
        """Consume one observation and return the current yaw rate in rad/s.

        Invalid observations and non-increasing timestamps raise ``ValueError``
        without modifying an already established estimator state.
        """
        time_s = float(time_s)
        heading_rad = float(heading_rad)
        if not (math.isfinite(time_s) and math.isfinite(heading_rad)):
            raise ValueError("time_s and heading_rad must be finite")

        if self.previous_time is None:
            self.previous_time = time_s
            self.previous_heading = heading_rad
            self.sample_count = 1
            self.last_estimate = YawRateEstimate(
                value_rad_s=0.0,
                initialized=False,
                sample_count=1,
                raw_rate_rad_s=None,
                state_rad_s=None,
            )
            return 0.0

        dt = time_s - self.previous_time
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("timestamps must be finite and strictly increasing")

        delta = wrap_angle_difference(heading_rad - float(self.previous_heading))
        raw_rate = delta / dt
        if not math.isfinite(raw_rate):
            raise ValueError("the backward-difference yaw rate is not finite")

        if self.rate_state is None:
            state = raw_rate
        else:
            alpha = 1.0 - math.exp(-dt / self.tau_s)
            state = alpha * raw_rate + (1.0 - alpha) * self.rate_state

        # Commit only after all validation and arithmetic have succeeded.
        self.previous_time = time_s
        self.previous_heading = heading_rad
        self.rate_state = state
        self.sample_count += 1

        output = self._format_output(state)
        self.last_estimate = YawRateEstimate(
            value_rad_s=output,
            initialized=True,
            sample_count=self.sample_count,
            raw_rate_rad_s=raw_rate,
            state_rad_s=state,
        )
        return output


def causal_yaw_rate(
    times_s: Iterable[float],
    headings_rad: Iterable[float],
    tau_s: float = DEFAULT_TAU_S,
    deadband_rad_s: float = DEFAULT_DEADBAND_RAD_S,
    decimals: int = DEFAULT_DECIMALS,
) -> np.ndarray:
    """Estimate a complete sequence using the same online recursion."""
    times = np.asarray(times_s, dtype=float)
    headings = np.asarray(headings_rad, dtype=float)
    if times.ndim != 1 or headings.ndim != 1 or times.shape != headings.shape:
        raise ValueError("times_s and headings_rad must be equal-length 1-D sequences")

    estimator = CausalYawRateEWMA(tau_s, deadband_rad_s, decimals)
    return np.asarray(
        [estimator.update(t, h) for t, h in zip(times, headings)],
        dtype=float,
    )


# Backward-compatible spelling used in the accompanying method note.
causal_yawrate = causal_yaw_rate
