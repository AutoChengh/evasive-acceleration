import math
import numpy as np
from evasive_acceleration._experimental import continuous_cells as c


def test_affine_ray_closes_false_gaps_between_sampled_contacts():
    # Continuous crossing of x=-9+9t-.5*a*t^2 through [-1,1]. The discrete
    # times t=1,2 have a false free gap; the continuous ray has no such gap.
    p, v = np.array([-9., 0., -9., 0.]), np.array([9., 0., 9., 0.])
    r, nx, ny = np.ones(4), np.array([1., 0., 1., 0.]), np.array([0., 1., 0., 1.])
    hit, lo, hi = c.affine_interval(p, v, r, nx, ny, 0., 2., 0., 0., 100., 0., False)
    assert hit and lo == 0.
    assert abs(hi - 81. / 16.) < 1e-10


def test_affine_sector_interval_is_a_subset_of_every_ray_interval():
    p, v = np.array([-9., 0., -9., 0.]), np.array([9., 0., 9., 0.])
    r, nx, ny = np.ones(4), np.array([1., 0., 1., 0.]), np.array([0., 1., 0., 1.])
    hit, lo, hi = c.affine_interval(p, v, r, nx, ny, 0., 2., .1, .08, 100., -1e-9, True)
    assert hit
    for angle in np.linspace(.02, .18, 31):
        valid, lower, upper = c.affine_interval(p, v, r, nx, ny, 0., 2., angle, 0., 100., 0., False)
        assert valid and lower <= lo + 1e-8 and upper >= hi - 1e-8


def test_time_zero_positive_displacement_is_not_zero_acceleration():
    p, v = np.array([5., 0., 5., 0.]), np.zeros(4)
    r, nx, ny = np.ones(4), np.array([1., 0., 1., 0.]), np.array([0., 1., 0., 1.])
    hit, lo, hi = c.affine_interval(p, v, r, nx, ny, 0., 10., 0., 0., 100., 0., False)
    assert hit and abs(lo - .08) < 1e-12 and hi == 100.
