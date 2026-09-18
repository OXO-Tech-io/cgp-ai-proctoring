from dataclasses import dataclass

import numpy as np

def angle_difference(current: float, reference: float) -> float:
    return (current - reference + 180.0) % 360.0 - 180.0

@dataclass
class NeutralBaseline:
    pitch: float
    yaw: float
    roll: float


class NeutralPoseCalibrator:
    def __init__(self, sample_count: int = 50):
        self.sample_count = sample_count

        self.active = False
        self.samples = []

        self.baseline = None

    def start(self):
        self.active = True
        self.samples = []

        print()
        print("=" * 60)
        print("STARTING NEUTRAL POSE CALIBRATION")
        print(f"Collecting {self.sample_count} samples...")
        print("Keep your head in your normal exam position.")
        print("=" * 60)

    def cancel(self):
        if self.active:
            print()
            print("NEUTRAL CALIBRATION CANCELLED")

        self.active = False
        self.samples = []


    def add_sample(self, pitch: float, yaw: float, roll: float):
        if not self.active:
            return None

        if len(self.samples) >= self.sample_count:
            return None

        self.samples.append(
            (pitch, yaw, roll)
        )

        if len(self.samples) == self.sample_count:
            return self.finish()

        return None

    def finish(self):
        pitches = np.array(
            [sample[0] for sample in self.samples]
        )

        yaws = np.array(
            [sample[1] for sample in self.samples]
        )

        rolls = np.array(
            [sample[2] for sample in self.samples]
        )

        self.baseline = NeutralBaseline(
            pitch=float(np.median(pitches)),
            yaw = float(np.median(yaws)),
            roll=float(np.median(rolls)),
        )

        print()
        print("=" * 60)
        print("NEUTRAL POSE CALIBRATION COMPLETE")
        print("=" * 60)

        print(f"Samples: {len(self.samples)}")

        print()
        print("Neutral Baseline:")
        print(f"  Pitch: {self.baseline.pitch:.2f}°")
        print(f"  Yaw  : {self.baseline.yaw:.2f}°")
        print(f"  Roll : {self.baseline.roll:.2f}°")

        print()
        print("Noise:")
        print(f"  Pitch Std: {np.std(pitches):.2f}°")
        print(f"  Yaw Std  : {np.std(yaws):.2f}°")
        print(f"  Roll Std : {np.std(rolls):.2f}°")

        print("=" * 60)

        self.active = False
        self.samples = []

        return self.baseline

    def get_progress(self):
        return len(self.samples), self.sample_count

    def get_relative_pose(
        self,
        pitch: float,
        yaw: float,
        roll: float
    ):
        if self.baseline is None:
            return None

        delta_pitch = angle_difference(
            pitch,
            self.baseline.pitch,
        )

        delta_yaw = angle_difference(
            yaw, 
            self.baseline.yaw,
        )

        delta_roll = angle_difference(
            roll,
            self.baseline.roll,
        )

        return (
            delta_pitch,
            delta_yaw,
            delta_roll
        )
        