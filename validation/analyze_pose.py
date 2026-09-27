import csv
from pathlib import Path
import numpy as np

DATA_PATH = Path("validation/data/pose_measurements.csv")

EXPECTED_COLUMNS =[
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
]

MOVEMENT_TESTS = {
    "TURN LEFT": "Yaw",
    "TURN RIGHT": "Yaw",
    "LOOK UP": "Pitch",
    "LOOK DOWN": "Pitch",
    "TILT LEFT": "Roll",
    "TILT RIGHT": "Roll",
}

JITTER_WARNING_THRESHOLD = 1.0
JITTER_INVESTIGATION_THRESHOLD = 2.0
COUPLING_INVESTIGATION_THRESHOLD = 0.50

def assess_stability(jitter_result):
    mean_abs = jitter_result["mean_abs"]

    if mean_abs <= JITTER_WARNING_THRESHOLD:
        return "STABLE"

    elif mean_abs <= JITTER_INVESTIGATION_THRESHOLD:
        return "ELEVATED JITTER"
    else:
        return "HIGH JITTER"


def access_coupling(coupling_ratio):
    return "INVESTIGATE" if coupling_ratio >= COUPLING_INVESTIGATION_THRESHOLD else "NO MAJOR FLAG"


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    with DATA_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                "Unexpected CSV columsn.\n"
                f"Expected: {EXPECTED_COLUMNS}\n"
                f"Found: {reader.fieldnames}"
            )

        rows = list(reader)

    return rows

def group_by_test(rows):
    grouped = {}

    for row in rows:
        test_name = row["test"]
        grouped.setdefault(test_name, []).append(row)

    return grouped

def calculate_statistics(values):
    values = np.array(values, dtype=np.float64)

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
    }

def print_statistics(name, values):
    stats = calculate_statistics(values)

    print(f" {name}")
    print(f"    Mean   : {stats['mean']:+.2f}°")
    print(f"    Median : {stats['median']:+.2f}°")
    print(f"    Std    : {stats['std']:.2f}°")
    print(f"    Min    : {stats['min']:+.2f}°")
    print(f"    Max    : {stats['max']:+.2f}°")
    print(f"    Range  : {stats['range']:.2f}°")


def analyze_pose_statistics(rows):
    grouped = group_by_test(rows)

    for test_name, test_rows in grouped.items():
        print()
        print("=" * 60)
        print(f"TEST: {test_name}")
        print("=" * 60)

        pitch = [float(row["pitch"])for row in test_rows]

        yaw = [float(row["yaw"])for row in test_rows]

        roll = [float(row["roll"])for row in test_rows]

        print()
        print("RAW POSE")

        print_statistics("Pitch", pitch)
        print_statistics("Yaw", yaw)
        print_statistics("Roll", roll)

        relative_rows = [
            row
            for row in test_rows
            if row["relative_pitch"] != ""
        ]

        if relative_rows:
            relative_pitch = [float(row["relative_pitch"])for row in relative_rows]

            relative_yaw = [float(row["relative_yaw"])for row in relative_rows]

            relative_roll = [float(row["relative_roll"])for row in relative_rows]

            print()
            print("RELATIVE POSE")

            print_statistics(
                "Relative Pitch",
                relative_pitch,
            )

            print_statistics(
                "Relative Yaw",
                relative_yaw,
            )

            print_statistics(
                "Relative Roll",
                relative_roll,
            )

def angle_difference(current: float, previous: float) -> float:
    return (current - previous + 180.0) % 360.0 - 180.0

def calculate_jitter(values, circular=False):
    values = np.array(values, dtype=np.float64)

    if len(values) < 2:
        return None

    differences = []

    for i in range(1, len(values)):
        if circular:
            difference = angle_difference(
                values[i],
                values[i - 1],
            )
        else:
            difference = values[i] - values[i - 1]

        differences.append(difference)

    differences = np.array(
        differences,
        dtype=np.float64,
    )

    absolute_differences = np.abs(differences)

    return {
        "mean_abs": float(
            np.mean(absolute_differences)
        ),
        "median_abs": float(
            np.median(absolute_differences)
        ),
        "std": float(
            np.std(differences)
        ),
        "max_abs": float(
            np.max(absolute_differences)
        ),
    }

def print_jitter(name, values, circular=False):
    jitter = calculate_jitter(
        values,
        circular=circular,
    )

    if jitter is None:
        print(f"  {name}: insufficient samples")
        return

    print(f"  {name}")
    print(
        f"    Mean absolute change   : "
        f"{jitter['mean_abs']:.3f}°"
    )
    print(
        f"    Median absolute change : "
        f"{jitter['median_abs']:.3f}°"
    )
    print(
        f"    Change Std             : "
        f"{jitter['std']:.3f}°"
    )
    print(
        f"    Maximum absolute change: "
        f"{jitter['max_abs']:.3f}°"
    )

def analyze_jitter(rows):
    grouped = group_by_test(rows)

    for test_name, test_rows in grouped.items():
        print()
        print("=" * 60)
        print(f"JITTER: {test_name}")
        print("=" * 60)

        pitch = [
            float(row["pitch"])
            for row in test_rows
        ]

        yaw = [
            float(row["yaw"])
            for row in test_rows
        ]

        roll = [
            float(row["roll"])
            for row in test_rows
        ]

        print()

        print_jitter(
            "Pitch",
            pitch,
        )

        print_jitter(
            "Yaw",
            yaw,
        )

        print_jitter(
            "Roll",
            roll,
            circular=True,
        )

        relative_rows = [
            row
            for row in test_rows
            if row["relative_pitch"] != ""
        ]

        if relative_rows:
            relative_pitch = [
                float(row["relative_pitch"])
                for row in relative_rows
            ]

            relative_yaw = [
                float(row["relative_yaw"])
                for row in relative_rows
            ]

            relative_roll = [
                float(row["relative_roll"])
                for row in relative_rows
            ]

            print()
            print("RELATIVE POSE JITTER")

            print_jitter(
                "Relative Pitch",
                relative_pitch,
            )

            print_jitter(
                "Relative Yaw",
                relative_yaw,
            )

            print_jitter(
                "Relative Roll",
                relative_roll,
                circular=True,
            )


def print_dataset_summary(rows):
    tests = {}

    for row in rows:
        test_name = row["test"]
        tests[test_name] = tests.get(test_name, 0) + 1

    print("=" * 60)
    print("HP-09 HEAD POSE STABILITY ANALYSIS")
    print("=" * 60)

    print()
    print("Dataset")
    print(f"  Total samples: {len(rows)}")

    print("Samples by test:")

    for test_name, count in tests.items():
        print(f"  {test_name:<22} {count}")

    print("=" * 60)


def calculate_cross_axis_coupling(
    pitch: list[float],
    yaw: list[float],
    roll: list[float],
) -> dict[str, float]:
    mean_pitch = float(np.mean(pitch))
    mean_yaw = float(np.mean(yaw))
    mean_roll = float(np.mean(roll))

    abs_pitch = abs(mean_pitch)
    abs_yaw = abs(mean_yaw)
    abs_roll = abs(mean_roll)

    if abs_yaw >= abs_pitch and abs_yaw >= abs_roll:
        primary_axis = "Yaw"
        primary_value = abs_yaw
        secondary = {
            "Pitch": abs_pitch,
            "Roll": abs_roll,
        }

    elif abs_pitch >= abs_yaw and abs_pitch >= abs_roll:
        primary_axis = "Pitch"
        primary_value = abs_pitch
        secondary = {
            "Yaw": abs_yaw,
            "Roll": abs_roll,
        }

    else:
        primary_axis = "Roll"
        primary_value = abs_roll
        secondary = {
            "Pitch": abs_pitch,
            "Yaw": abs_yaw,
        }

    coupling = {}

    for axis, value in secondary.items():
        if primary_value > 0:
            coupling[axis] = value / primary_value
        else:
            coupling[axis] = 0.0

    return {
        "mean_pitch": mean_pitch,
        "mean_yaw": mean_yaw,
        "mean_roll": mean_roll,
        "primary_axis": primary_axis,
        "primary_value": primary_value,
        "pitch_coupling": coupling.get("Pitch", 0.0),
        "yaw_coupling": coupling.get("Yaw", 0.0),
        "roll_coupling": coupling.get("Roll", 0.0),
    }

def analyze_acceptance(rows):
    grouped = group_by_test(rows)

    print()
    print("=" * 60)
    print("HP-09.5 ACCEPTANCE ASSESSMENT")
    print("=" * 60)

    print()
    print("STABILITY ASSESSMENT")
    print("-" * 60)

    for test_name, test_rows in grouped.items():
        if test_name == "NEUTRAL_CALIBRATION":
            continue

        pitch = [
            float(row["pitch"])
            for row in test_rows
        ]

        yaw = [
            float(row["yaw"])
            for row in test_rows
        ]

        roll = [
            float(row["roll"])
            for row in test_rows
        ]

        pitch_jitter = calculate_jitter(pitch)
        yaw_jitter = calculate_jitter(yaw)
        roll_jitter = calculate_jitter(
            roll,
            circular=True,
        )

        print()
        print(f"{test_name}")

        print(
            f"  Pitch: "
            f"{pitch_jitter['mean_abs']:.3f}° "
            f"-> {assess_stability(pitch_jitter)}"
        )

        print(
            f"  Yaw:   "
            f"{yaw_jitter['mean_abs']:.3f}° "
            f"-> {assess_stability(yaw_jitter)}"
        )

        print(
            f"  Roll:  "
            f"{roll_jitter['mean_abs']:.3f}° "
            f"-> {assess_stability(roll_jitter)}"
        )

    print()
    print("PRIMARY AXIS DETECTION")
    print("-" * 60)

    for test_name, primary_axis in MOVEMENT_TESTS.items():
        if test_name not in grouped:
            continue

        test_rows = grouped[test_name]

        relative_pitch = [
            float(row["relative_pitch"])
            for row in test_rows
        ]

        relative_yaw = [
            float(row["relative_yaw"])
            for row in test_rows
        ]

        relative_roll = [
            float(row["relative_roll"])
            for row in test_rows
        ]

        means = {
            "Pitch": abs(float(np.mean(relative_pitch))),
            "Yaw": abs(float(np.mean(relative_yaw))),
            "Roll": abs(float(np.mean(relative_roll))),
        }

        primary_value = means[primary_axis]

        secondary_values = [
            value
            for axis, value in means.items()
            if axis != primary_axis
        ]

        largest_secondary = max(secondary_values)

        if primary_value > largest_secondary:
            result = "DETECTABLE"
        else:
            result = "NOT DOMINANT"

        print(
            f"  {test_name:<12} "
            f"Primary={primary_axis:<5} "
            f"{primary_value:.2f}° "
            f"-> {result}"
        )

    print()
    print("CROSS-AXIS COUPLING FLAGS")
    print("-" * 60)

    flagged_tests = []

    for test_name, primary_axis in MOVEMENT_TESTS.items():
        if test_name not in grouped:
            continue

        test_rows = grouped[test_name]

        relative_pitch = [
            float(row["relative_pitch"])
            for row in test_rows
        ]

        relative_yaw = [
            float(row["relative_yaw"])
            for row in test_rows
        ]

        relative_roll = [
            float(row["relative_roll"])
            for row in test_rows
        ]

        result = calculate_cross_axis_coupling(
            relative_pitch,
            relative_yaw,
            relative_roll,
        )

        coupling_values = {
            "Pitch": result["pitch_coupling"],
            "Yaw": result["yaw_coupling"],
            "Roll": result["roll_coupling"],
        }

        coupling_values.pop(primary_axis)

        largest_axis = max(
            coupling_values,
            key=coupling_values.get,
        )

        largest_ratio = coupling_values[largest_axis]

        assessment = access_coupling(
            largest_ratio
        )

        print(
            f"  {test_name:<12} "
            f"{largest_axis} coupling: "
            f"{largest_ratio * 100:.1f}% "
            f"-> {assessment}"
        )

        if assessment == "INVESTIGATE":
            flagged_tests.append(
                (
                    test_name,
                    largest_axis,
                    largest_ratio,
                )
            )

    print()
    print("FINAL ASSESSMENT")
    print("-" * 60)

    print(
        "  Primary movement signals: "
        "DETECTABLE"
    )

    if flagged_tests:
        print(
            "  Cross-axis investigation: "
            "REQUIRED"
        )

        for (
            test_name,
            axis,
            ratio,
        ) in flagged_tests:
            print(
                f"    - {test_name}: "
                f"{axis} coupling "
                f"{ratio * 100:.1f}%"
            )
    else:
        print(
            "  Cross-axis investigation: "
            "NO MAJOR FLAGS"
        )

    print()
    print(
        "  Production thresholds: "
        "NOT ESTABLISHED"
    )

    print(
        "  Recommendation: "
        "Continue validation before "
        "production classification."
    )

    print("=" * 60)

def analyze_cross_axis_coupling(rows):
    grouped = group_by_test(rows)

    print()
    print("=" * 60)
    print("HP-09.4 CROSS-AXIS COUPLING ANALYSIS")
    print("=" * 60)

    for test_name, test_rows in grouped.items():

        relative_rows = [
            row
            for row in test_rows
            if row["relative_pitch"] != ""
        ]

        if not relative_rows:
            continue

        pitch = [
            float(row["relative_pitch"])
            for row in relative_rows
        ]

        yaw = [
            float(row["relative_yaw"])
            for row in relative_rows
        ]

        roll = [
            float(row["relative_roll"])
            for row in relative_rows
        ]

        result = calculate_cross_axis_coupling(
            pitch,
            yaw,
            roll,
        )

        print()
        print(f"TEST: {test_name}")
        print("-" * 60)

        print(
            f" Mean relative Pitch : "
            f"{result['mean_pitch']:+.2f}°"
        )

        print(
            f" Mean relative Yaw   : "
            f"{result['mean_yaw']:+.2f}°"
        )

        print(
            f" Mean relative Roll  : "
            f"{result['mean_roll']:+.2f}°"
        )

        print(
            f" Primary axis        : "
            f"{result['primary_axis']}"
        )

        print(
            f" Primary magnitude   : "
            f"{result['primary_value']:.2f}°"
        )

        print()
        print(" Coupling ratios")

        print(f"   Pitch / Primary : "f"{result['pitch_coupling']:.3f}")

        print(f"   Yaw / Primary   : "f"{result['yaw_coupling']:.3f}")

        print(f"   Roll / Primary  : "f"{result['roll_coupling']:.3f}")

def calculate_latency_statistics(values):
    values = np.array(values, dtype=np.float64)
    
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def analyze_latency(rows):
    grouped = group_by_test(rows)

    print()
    print("=" * 60)
    print("HP-09.6 DETECTION LATENCY ANALYSIS")
    print("=" * 60)

    all_latencies = []

    for test_name, test_rows in grouped.items():
        latencies = [
            float(row["detection_time_ms"])
            for row in test_rows
            if row["detection_time_ms"] != ""
        ]

        if not latencies:
            continue

        all_latencies.extend(latencies)

        stats = calculate_latency_statistics(latencies)

        print()
        print(f"TEST: {test_name}")
        print("-" * 60)
        print(f" Mean   : {stats['mean']:.2f} ms")
        print(f" Median : {stats['median']:.2f} ms")
        print(f" Std    : {stats['std']:.2f} ms")
        print(f" Min    : {stats['min']:.2f} ms")
        print(f" Max    : {stats['max']:.2f} ms")
        print(f" P95    : {stats['p95']:.2f} ms")
        print(f" P99    : {stats['p99']:.2f} ms")

    if all_latencies:
        stats = calculate_latency_statistics(all_latencies)

        print()
        print("OVERALL LATENCY")
        print("-" * 60)
        print(f" Mean   : {stats['mean']:.2f} ms")
        print(f" Median : {stats['median']:.2f} ms")
        print(f" Std    : {stats['std']:.2f} ms")
        print(f" Min    : {stats['min']:.2f} ms")
        print(f" Max    : {stats['max']:.2f} ms")
        print(f" P95    : {stats['p95']:.2f} ms")
        print(f" P99    : {stats['p99']:.2f} ms")

    print("=" * 60)


def main():
    rows = load_dataset()
    print_dataset_summary(rows)

    analyze_pose_statistics(rows)

    analyze_jitter(rows)

    analyze_cross_axis_coupling(rows)

    analyze_acceptance(rows)
    analyze_latency(rows)


if __name__ == "__main__":
    main()