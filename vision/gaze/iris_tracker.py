from typing import Sequence


class IrisExtractor:
    """Extract iris landmarks and calculate iris centers."""

    def __init__(
        self,
        left_iris_indices: Sequence[int],
        right_iris_indices: Sequence[int],
    ) -> None:
        self.left_iris_indices = left_iris_indices
        self.right_iris_indices = right_iris_indices

    def extract(self, landmarks):
        """Extract left and right iris landmarks."""

        left_iris = [
            landmarks[index]
            for index in self.left_iris_indices
        ]

        right_iris = [
            landmarks[index]
            for index in self.right_iris_indices
        ]

        return left_iris, right_iris

    @staticmethod
    def calculate_center(landmarks) -> tuple[float, float]:
        """Calculate the center point of an iris."""

        x = sum(point.x for point in landmarks) / len(landmarks)
        y = sum(point.y for point in landmarks) / len(landmarks)

        return x, y