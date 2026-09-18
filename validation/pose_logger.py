from datetime import datetime, timezone
from pathlib import Path
import csv

class PoseDataLogger:
    FILEDNAMES = [
        "timestamp",
        "test",
        "sample",
        "pitch",
        "yaw",
        "roll",
        "relative_pitch",
        "relative_yaw",
        "relative_roll",
        "detection_time_ms"
    ]

    def __init__(self, output_path: str = "data/pose_validation.csv"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self._initialize_file()

    def _initialize_file(self):
        """Create the CSV file with headers if it does not exist."""

        if self.output_path.exists() and self.output_path.stat().st_size > 0:
            return

        with self.output_path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
                writer = csv.DictWriter(
                     file,
                     fieldnames=self.FILEDNAMES,
                )
                writer.writeheader()

    def log_sample(
        self,
        test: str,
        sample: int,
        pitch: float,
        yaw: float,
        roll: float,
        relative_pose=None,
        detection_time_ms: float | None=None,
    ):
        """Save one pose sample to the CSV file."""

        relative_pose = None
        relative_yaw = None
        relative_roll = None

        if relative_pose is not None:
             (
                  relative_pitch,
                  relative_yaw,
                  relative_roll,
             ) = relative_pose

        row ={
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test": test,
            "sample": sample,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "relative_pitch": relative_pitch,
            "relative_yaw": relative_yaw,
            "relative_roll": relative_roll,
            "detection_time_ms": detection_time_ms,
        }

        with self.output_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                 file,
                 fieldnames=self.FILEDNAMES,
            )
            writer.writerow(row)


    def log_baseline(
        self,
        pitch: float,
        yaw: float,
        roll: float,    
    ):
        """Save the neutral calibration baseline as a seperate record."""
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test": "NEUTRAL_BASELINE",
            "sample": "",
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "relative_pitch": 0.0,
            "relative_yaw": 0.0,
            "relative_roll": 0.0,
            "detection_time_ms": "",
        }

        with self.output_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.FIELDNAMES,
            )
            writer.writerow(row)

    def get_output_path(self) -> Path:
        return self.output_path