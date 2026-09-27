import math

import cv2
import numpy as np

from vision.landmark_indices import (
    NOSE_TIP,
    CHIN,
    LEFT_EYE_OUTER_CORNER,
    RIGHT_EYE_OUTER_CORNER,
    LEFT_MOUTH_CORNER,
    RIGHT_MOUTH_CORNER,
)


MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),           # Nose tip
        (0.0, 200.0, 150.0),      # Chin: Deep and far down
        (150.0, -120.0, 150.0),   # Left Eye: Far left, deep
        (-150.0, -120.0, 150.0),  # Right Eye: Far right, deep
        (80.0, 120.0, 100.0),     # Left Mouth: Mid left, mid deep
        (-80.0, 120.0, 100.0),    # Right Mouth: Mid right, mid deep
    ],
    dtype=np.float64,
)



class HeadPoseEstimator:
    def __init__(self, width: int, height: int) -> None:
        self.camera_matrix = self._create_camera_matrix(
            width,
            height,
        )

        self.distortion = np.zeros(
            (4, 1),
            dtype=np.float64,
        )

    @staticmethod
    def _create_camera_matrix(
        width: int,
        height: int,
    ) -> np.ndarray:

        focal_length = float(width)

        return np.array(
            [
                [focal_length, 0.0, width / 2.0],
                [0.0, focal_length, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _get_image_points(
        self,
        face_landmarks,
        width: int,
        height: int,
    ) -> np.ndarray:

        indices = [
            NOSE_TIP,
            CHIN,
            LEFT_EYE_OUTER_CORNER,
            RIGHT_EYE_OUTER_CORNER,
            LEFT_MOUTH_CORNER,
            RIGHT_MOUTH_CORNER,
        ]

        points = []

        for index in indices:
            landmark = face_landmarks[index]

            points.append(
                (
                    landmark.x * width,
                    landmark.y * height,
                )
            )

        return np.asarray(
            points,
            dtype=np.float64,
        )

    def estimate(
        self,
        face_landmarks,
        width: int,
        height: int,
    ) -> tuple[float, float, float]:

        image_points = self._get_image_points(
            face_landmarks,
            width,
            height,
        )

        success, rotation_vector, _ = cv2.solvePnP(
            MODEL_POINTS,
            image_points,
            self.camera_matrix,
            self.distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            raise RuntimeError(
                "solvePnP failed to estimate head pose"
            )

        rotation_matrix, _ = cv2.Rodrigues(
            rotation_vector
        )

        sy = math.hypot(
            rotation_matrix[0, 0],
            rotation_matrix[1, 0],
        )

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(
                rotation_matrix[2, 1],
                rotation_matrix[2, 2],
            )

            y = math.atan2(
                -rotation_matrix[2, 0],
                sy,
            )

            z = math.atan2(
                rotation_matrix[1, 0],
                rotation_matrix[0, 0],
            )

        else:
            x = math.atan2(
                -rotation_matrix[1, 2],
                rotation_matrix[1, 1],
            )

            y = math.atan2(
                -rotation_matrix[2, 0],
                sy,
            )

            z = 0.0

        angles = np.degrees(
            [x, y, z]
        )

        return (
            float(angles[0]),  # pitch
            float(angles[1]),  # yaw
            float(angles[2]),  # roll
        )