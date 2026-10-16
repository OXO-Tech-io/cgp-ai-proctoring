"""Tunable constants for the gaze pipeline."""

# Calibration
CALIB_DELAY_S = 4.0
CALIB_SAMPLES = 50
CALIB_MAX_IRIS_STD = 0.02   # max std of normalised iris position
CALIB_MAX_HEAD_STD_DEG = 2.0  # max std of yaw / pitch during calibration

# Classifier thresholds (placeholders, tune from session B)
TX, TY = 0.06, 0.10
ENABLE_VERTICAL = False  # UP/DOWN unvalidated; ignore dy until session B

# Head-pose gate: beyond this, gaze is not estimated (HEAD_AWAY)
MAX_YAW_DEG, MAX_PITCH_DEG = 20.0, 15.0

# Head-pose compensation (placeholders; fit per direction from sessions C/D)
YAW_GAIN_POS, YAW_GAIN_NEG = 0.0027, 0.0027
PITCH_GAIN_POS, PITCH_GAIN_NEG = -0.0021, -0.0021
COMP_MAX_ANGLE_DEG = 25.0

# Filter
FILTER_WINDOW = 5
FILTER_ALPHA = 0.4
FILTER_MAX_STEP = 0.05
FILTER_MAX_DIVERGENCE = 0.05
