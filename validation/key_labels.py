from typing import Sequence


def digit_key_labels(labels: Sequence[str]) -> dict[int, str]:
    """Map the digit keys '1', '2', ... to labels, as returned by cv2.waitKey."""
    return {ord(str(i)): label for i, label in enumerate(labels, start=1)}
