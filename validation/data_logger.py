import csv
from datetime import datetime
from pathlib import Path


class PoseDataLogger:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.file = None
        self.writer = None

    def start(self):
        self.file = self.output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )

        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "timestamp",
            "test",
            "sample",
            "pitch",
            "yaw",
            "roll",
            "relative_pitch",
            "relative_yaw",
            "relative_roll",
            "detection_time_ms",
        ])

        self.file.flush()

    def log(
        self,
        test: str,
        sample: int,
        pitch: float,
        yaw: float,
        roll: float,
        relative_pitch: float | None,
        relative_yaw: float | None,
        relative_roll: float | None,
        detection_time_ms: float | None,
    ) -> None:
        """Log a single head-pose measurement to the CSV dataset."""
        if self.writer is None:
            return

        self.writer.writerow([ 
            datetime.now().isoformat(timespec="milliseconds"), 
            test, 
            sample, 
            f"{pitch:.4f}", 
            f"{yaw:.4f}", 
            f"{roll:.4f}", 
            f"{relative_pitch:.4f}" if relative_pitch is not None else "", 
            f"{relative_yaw:.4f}" if relative_yaw is not None else "", 
            f"{relative_roll:.4f}" if relative_roll is not None else "", 
            f"{detection_time_ms:.4f}" if detection_time_ms is not None else "", 
        ])

        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()

        self.file = None
        self.writer = None