"""The four ordered motion-model hypotheses aggregated into the EA score."""
from .config import MotionModel

FOUR_PAIRS = (
    (MotionModel.CV, MotionModel.CV),
    (MotionModel.CV, MotionModel.CTRV),
    (MotionModel.CTRV, MotionModel.CV),
    (MotionModel.CTRV, MotionModel.CTRV),
)
