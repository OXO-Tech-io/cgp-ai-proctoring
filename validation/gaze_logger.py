import csv
import os
from datetime import datetime

class GazeDataLogger:
    """Log raw gaze estimation data to CSV."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.file = None
        self.writer = None

    def start(self) -> None:
        """Create the CSV file and write the header."""

        directory = os.path.dirname(self.file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        self.file = open(
            self.file_path,
            "w",
            newline="",
        )

        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "timestamp",

            "left_iris_x",
            "left_iris_y",
            "left_normalized_x",
            "left_normalized_y",
            "left_offset_x",
            "left_offset_y",

            "right_iris_x",
            "right_iris_y",
            "right_normalized_x",
            "right_normalized_y",
            "right_offset_x",
            "right_offset_y",

            "left_comp_x",
            "left_comp_y",
            "right_comp_x",
            "right_comp_y",
            "gaze_dx",
            "gaze_dy",

            "pitch",
            "yaw",
            "roll",
            "delta_yaw",
            "delta_pitch",

            "filtered_dx",
            "filtered_dy",
            "reliable",
            "gaze_label",
            "session",
            "target",

            "detection_time_ms",
        ])

    def log(
        self,
        left_iris: tuple[float, float],
        left_normalized: tuple[float, float],
        left_offset: tuple[float, float],
        right_iris: tuple[float, float],
        right_normalized: tuple[float, float],
        right_offset: tuple[float, float],
        left_compensated: tuple[float, float],
        right_compensated: tuple[float, float],
        gaze: tuple[float, float],
        pitch: float,
        yaw: float,
        roll: float,
        delta_yaw: float,
        delta_pitch: float,
        filtered: tuple[float, float],
        reliable: bool,
        gaze_label: str,
        session: str,
        detection_time_ms: float,
        target: str = "",
    ) -> None:
        """Write one gaze sample to the CSV file."""

        if self.writer is None:
            raise RuntimeError("Logger has not been started.")

        self.writer.writerow([
            datetime.now().isoformat(),

            left_iris[0],
            left_iris[1],
            left_normalized[0],
            left_normalized[1],
            left_offset[0],
            left_offset[1],

            right_iris[0],
            right_iris[1],
            right_normalized[0],
            right_normalized[1],
            right_offset[0],
            right_offset[1],

            left_compensated[0],
            left_compensated[1],
            right_compensated[0],
            right_compensated[1],
            gaze[0],
            gaze[1],

            pitch,
            yaw,
            roll,
            delta_yaw,
            delta_pitch,

            filtered[0],
            filtered[1],
            int(reliable),
            gaze_label,
            session,
            target,

            detection_time_ms,
        ])

    def close(self) -> None:
        """Close the CSV file."""

        if self.file:
            self.file.close()
            self.file = None
            self.writer = None