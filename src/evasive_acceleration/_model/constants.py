"""Shared numerical defaults and JIT decorator for the EA calculation.

Some scalar defaults also support compatible input signatures. EA's numerical
accuracy is controlled by its acceleration tolerance and safety certificates.
"""
from numba import njit

DEFAULT_DT_COARSE = 0.1
DEFAULT_DT_FINE = 0.02
DEFAULT_A_MAX = 100.0
DEFAULT_TOL = 1e-3
DEFAULT_COARSE_SECTOR_NUM = 72
DEFAULT_LOCAL_FINE_HALF_WINDOW_DEG = 5
DEFAULT_LOCAL_FINE_DIR_NUM = 101
DEFAULT_FINE_REFINE_RATIO = 1.15
