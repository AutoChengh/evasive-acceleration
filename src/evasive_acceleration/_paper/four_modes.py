"""Ordered motion-pair constants required by the unchanged EA kernels.

There is no solver entry point in this compatibility module.
"""
from .config import MotionModel

FOUR_PAIRS = (
    (MotionModel.CV, MotionModel.CV),
    (MotionModel.CV, MotionModel.CTRV),
    (MotionModel.CTRV, MotionModel.CV),
    (MotionModel.CTRV, MotionModel.CTRV),
)
