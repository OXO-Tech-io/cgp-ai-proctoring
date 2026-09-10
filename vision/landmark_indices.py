"""
MediaPipe Face Landmarker index reference.\
Based on the 478-point mesh topology (468 base points + 10 iris points)
Verify against official MediaPipe face mesh diagram before relying on these for production accuracy.
"""

NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER_CORNER = 263
RIGHT_EYE_OUTER_CORNER = 33
LEFT_MOUTH_CORNER = 287
RIGHT_MOUTH_CORNER = 57

SOLVEPNP_6POINT = [
    NOSE_TIP,
    CHIN,
    LEFT_EYE_OUTER_CORNER,
    RIGHT_EYE_OUTER_CORNER,
    LEFT_MOUTH_CORNER,
    RIGHT_MOUTH_CORNER
]

LEFT_EYE_INNER_CORNER = 362
RIGHT_EYE_INNER_CORNER = 133

LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473