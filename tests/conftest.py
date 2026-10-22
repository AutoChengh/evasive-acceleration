"""Independent analytical references for synthetic continuous-time EA cases."""

import math

import pytest


@pytest.fixture
def aligned_straight_ea_infimum():
    """Exact infimum for separated, axis-aligned, collinear closing rectangles.

    With initial bumper clearance D, summed half-width W and closing speed V,
    the pure-braking infimum is V²/(2D). A mixed lateral/longitudinal manoeuvre
    first clears laterally at t with ax=2(D-Vt)/t² and ay=2W/t². Stationarity
    of its squared magnitude gives q²-Dq+2W²=0, where q=Vt-D. Enumerating those
    stationary points and pure braking yields the exact continuous-time
    infimum for these long-horizon cases. Contact counts as collision, so a
    certified solver may return an upper bound just above this infimum.
    """
    def reference(clearance, lateral_half_width, closing_speed, horizon=10.):
        d, w, v = clearance, lateral_half_width, closing_speed
        assert d > 0. and w > 0. and v > 0.
        # Restrict this independent derivation to its proven long-horizon case.
        assert horizon >= 2. * d / v
        candidates = [v * v / (2. * d)]
        discriminant = d * d - 8. * w * w
        if discriminant >= 0.:
            for sign in (-1., 1.):
                q = (d + sign * math.sqrt(discriminant)) / 2.
                time = (d + q) / v
                if time <= horizon:
                    candidates.append(2. * math.hypot(q, w) / (time * time))
        return min(candidates)
    return reference
