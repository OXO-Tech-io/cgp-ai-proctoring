class HeadPoseCompensator:
    """Compensate raw gaze measurements using head pose.

    Gains can differ for positive and negative turns, and the angle is
    clamped so large turns are never extrapolated.
    """

    def __init__(
        self,
        yaw_gain: float = 0.0027,
        pitch_gain: float = -0.0021,
        yaw_gain_pos: float | None = None,
        yaw_gain_neg: float | None = None,
        pitch_gain_pos: float | None = None,
        pitch_gain_neg: float | None = None,
        max_angle: float = 25.0,
    ) -> None:
        self.yaw_gain_pos = yaw_gain if yaw_gain_pos is None else yaw_gain_pos
        self.yaw_gain_neg = yaw_gain if yaw_gain_neg is None else yaw_gain_neg
        self.pitch_gain_pos = pitch_gain if pitch_gain_pos is None else pitch_gain_pos
        self.pitch_gain_neg = pitch_gain if pitch_gain_neg is None else pitch_gain_neg
        self.max_angle = max_angle

    def compensate(self, gaze_x, gaze_y, yaw, pitch):
        yaw = max(-self.max_angle, min(self.max_angle, yaw))
        pitch = max(-self.max_angle, min(self.max_angle, pitch))
        yaw_gain = self.yaw_gain_pos if yaw >= 0 else self.yaw_gain_neg
        pitch_gain = self.pitch_gain_pos if pitch >= 0 else self.pitch_gain_neg
        return gaze_x - yaw_gain * yaw, gaze_y - pitch_gain * pitch
