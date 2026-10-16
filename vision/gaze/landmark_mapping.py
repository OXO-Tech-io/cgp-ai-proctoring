"""Per-eye landmark indices used by the gaze pipeline.

Single source of truth is vision/landmark_indices.py (subject's left/right),
so the gaze code can never disagree with the head-pose code about which eye
is which.
"""

from vision.landmark_indices import (
    LEFT_EYE_BOTTOM,
    LEFT_EYE_INNER_CORNER,
    LEFT_EYE_OUTER_CORNER,
    LEFT_EYE_TOP,
    RIGHT_EYE_BOTTOM,
    RIGHT_EYE_INNER_CORNER,
    RIGHT_EYE_OUTER_CORNER,
    RIGHT_EYE_TOP,
)

__all__ = [
    "LEFT_EYE_BOTTOM",
    "LEFT_EYE_INNER_CORNER",
    "LEFT_EYE_OUTER_CORNER",
    "LEFT_EYE_TOP",
    "RIGHT_EYE_BOTTOM",
    "RIGHT_EYE_INNER_CORNER",
    "RIGHT_EYE_OUTER_CORNER",
    "RIGHT_EYE_TOP",
]
