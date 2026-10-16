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

# Left/right follow the subject's anatomy (same as the eye corners above):
# MediaPipe iris 468-472 is the subject's RIGHT eye (image-left, next to
# corners 33/133); 473-477 is the subject's LEFT eye (next to 263/362).
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_CENTER = 468

LEFT_IRIS_INDICES = [473, 474, 475, 476, 477]
RIGHT_IRIS_INDICES = [468, 469, 470, 471, 472]

# Upper/lower eyelid midpoints, used to normalise iris height per eye.
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145
