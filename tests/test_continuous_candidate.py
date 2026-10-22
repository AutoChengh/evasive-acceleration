"""Conservative bounds must not turn sampled contact checks into proofs."""
import math
import numpy as np

from evasive_acceleration._experimental import continuous as c


def test_taylor_certificate_detects_contact_between_grid_points():
    # The 0.04 m wide boxes cross between 0 and .02 s, at .01 s.
    state = np.array([0., 0., 10., 0., .04, .04, 0., .1, -.1, 10., math.pi/2, .04, .04, 0.])
    assert c.separation(state, 0., 0., 0., 0., 0.) > 0
    assert c.separation(state, 0., 0., .02, 0., 0.) > 0
    status, t, _ = c.certify(state, 0., 0., 0., 0., .02)
    assert status == 0 and 0 < t < .02


def test_separated_stationary_bodies_are_certified():
    state = np.array([0., 0., 0., 0., 4., 2., 0., 10., 0., 0., 0., 4., 2., 0.])
    assert c.certify(state, 0., 0., 0., 0., 10.)[0] == 1


def test_sector_bound_never_exceeds_sampled_ray_minima():
    rng = np.random.default_rng(412)
    for _ in range(6):
        state = np.array([0., 0., 8., 0., 4.8, 2., .1, 12., -5., 5., 1.5, 4., 2., -.2])
        state[7:9] += rng.normal(0., 1., 2)
        times, squared = np.empty(31), np.empty(31)
        p, r, nx, ny = [np.empty((31, 4)) for _ in range(4)]
        n = 0
        for t in np.linspace(0, 10, 31):
            n = c._store_time(state, .1, -.2, t, times, squared, p, r, nx, ny, n)
        lower, upper = np.empty(n), np.empty(n)
        for center in np.linspace(-math.pi, math.pi, 13):
            half = .17
            lb = c.sector_lower_bound(p, r, nx, ny, squared, n, center, half, 100., 1e-9, lower, upper)
            for angle in np.linspace(center - half, center + half, 31):
                value = c._ray(p, r, nx, ny, squared, n, math.cos(angle), math.sin(angle), 100., 0., lower, upper)
                assert lb <= value + 1e-8


def test_reported_bound_includes_sectors_pruned_earlier():
    # A global bound cannot be replaced by the next active node after the
    # winning region has already been pruned within the requested tolerance.
    state = np.array([0., 0., 8., 0., 4.8, 2., .1, 12., -5., 5., 1.5, 4., 2., -.2])
    from evasive_acceleration._paper.four_modes import FOUR_PAIRS
    result = c.solve_pair(state, FOUR_PAIRS[3])
    assert result.continuous_verified
    assert result.lower_bound <= result.value
    assert result.value - result.lower_bound <= c.ContinuousBudget().acceleration_tolerance + 1e-12
