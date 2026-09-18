from dataclasses import dataclass

import numpy as np


@dataclass
class PoseSample:
    pitch: float
    yaw: float
    roll: float


class PoseTestRecorder:
    TESTS = {
        ord("1"): "NEUTRAL",
        ord("2"): "TURN LEFT",
        ord("3"): "TURN RIGHT",
        ord("4"): "LOOK UP",
        ord("5"): "LOOK DOWN",
        ord("6"): "TILT LEFT",
        ord("7"): "TILT RIGHT",
    }

    def __init__(self, sample_count: int = 50):
        self.sample_count = sample_count
        self.active_test = None
        self.samples: list[PoseSample] = []

        # Stores completed test results
        self.results = {}

        # Stores the most recently completed test
        self.last_completed_test = None

    def handle_key(self, key: int):
        if key in self.TESTS:
            self.start_test(self.TESTS[key])
            return

        if key == ord("0"):
            self.stop_test()

    def start_test(self, test_name: str):
        self.active_test = test_name
        self.samples = []

        print()
        print("=" * 60)
        print(f"STARTING TEST: {test_name}")
        print(f"Collecting {self.sample_count} samples...")
        print("=" * 60)

    def stop_test(self):
        if self.active_test is not None:
            print()
            print(f"TEST CANCELLED: {self.active_test}")

        self.active_test = None
        self.samples = []

    def add_sample(self, pitch: float, yaw: float, roll: float):
        if self.active_test is None:
            return None

        if len(self.samples) >= self.sample_count:
            return None

        self.samples.append(
            PoseSample(
                pitch=pitch,
                yaw=yaw,
                roll=roll,
            )
        )

        if len(self.samples) == self.sample_count:
            return self.finish_test()

        return None

    def get_progress(self):
        return len(self.samples), self.sample_count

    def finish_test(self):
        test_name = self.active_test

        pitches = np.array([sample.pitch for sample in self.samples])
        yaws = np.array([sample.yaw for sample in self.samples])
        rolls = np.array([sample.roll for sample in self.samples])

        result = {
            "samples": len(self.samples),

            "pitch_mean": float(np.mean(pitches)),
            "pitch_median": float(np.median(pitches)),
            "pitch_std": float(np.std(pitches)),

            "yaw_mean": float(np.mean(yaws)),
            "yaw_median": float(np.median(yaws)),
            "yaw_std": float(np.std(yaws)),

            "roll_mean": float(np.mean(rolls)),
            "roll_median": float(np.median(rolls)),
            "roll_std": float(np.std(rolls)),
        }

        self.results[test_name] = result
        self.last_completed_test = test_name

        self.print_result(test_name, result)

        # Reset current recording
        self.active_test = None
        self.samples = []

        return test_name, result

    def print_result(self, test_name: str, result: dict):
        print()
        print("=" * 60)
        print(f"TEST COMPLETE: {test_name}")
        print("=" * 60)

        print(f"Samples: {result['samples']}")

        print()
        print("Pitch:")
        print(f"  Mean   : {result['pitch_mean']:.2f}°")
        print(f"  Median : {result['pitch_median']:.2f}°")
        print(f"  Std    : {result['pitch_std']:.2f}°")

        print()
        print("Yaw:")
        print(f"  Mean   : {result['yaw_mean']:.2f}°")
        print(f"  Median : {result['yaw_median']:.2f}°")
        print(f"  Std    : {result['yaw_std']:.2f}°")

        print()
        print("Roll:")
        print(f"  Mean   : {result['roll_mean']:.2f}°")
        print(f"  Median : {result['roll_median']:.2f}°")
        print(f"  Std    : {result['roll_std']:.2f}°")

        print("=" * 60)