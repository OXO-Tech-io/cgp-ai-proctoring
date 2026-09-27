import cv2


class FacePositionGuide:
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        target_width_ratio: float = 0.35,
        target_height_ratio: float = 0.55,
        center_tolerance: float = 0.08,
        size_tolerance: float = 0.20,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.target_width = int(frame_width * target_width_ratio)
        self.target_height = int(frame_height * target_height_ratio)

        self.center_tolerance = center_tolerance
        self.size_tolerance = size_tolerance

        self.target_center_x = frame_width // 2
        self.target_center_y = frame_height // 2

    def get_face_bbox(self, face_landmarks):
        x_coordinates = [
            landmark.x for landmark in face_landmarks
        ]

        y_coordinates = [
            landmark.y for landmark in face_landmarks
        ]

        min_x = max(0.0, min(x_coordinates))
        max_x = min(1.0, max(x_coordinates))

        min_y = max(0.0, min(y_coordinates))
        max_y = min(1.0, max(y_coordinates))

        x1 = int(min_x * self.frame_width)
        y1 = int(min_y * self.frame_height)

        x2 = int(max_x * self.frame_width)
        y2 = int(max_y * self.frame_height)

        return x1, y1, x2, y2

    def evaluate(self, face_landmarks):
        x1, y1, x2, y2 = self.get_face_bbox(face_landmarks)

        face_width = x2 - x1
        face_height = y2 - y1

        face_center_x = (x1 + x2) // 2
        face_center_y = (y1 + y2) // 2

        center_error_x = (
            face_center_x - self.target_center_x
        ) / self.frame_width

        center_error_y = (
            face_center_y - self.target_center_y
        ) / self.frame_height

        expected_width = self.target_width
        expected_height = self.target_height

        width_error = (
            face_width - expected_width
        ) / expected_width

        height_error = (
            face_height - expected_height
        ) / expected_height

        messages = []

        if abs(center_error_x) > self.center_tolerance:
            if center_error_x > 0:
                messages.append("MOVE LEFT")
            else:
                messages.append("MOVE RIGHT")

        if abs(center_error_y) > self.center_tolerance:
            if center_error_y > 0:
                messages.append("MOVE UP")
            else:
                messages.append("MOVE DOWN")

        if width_error < -self.size_tolerance:
            messages.append("MOVE CLOSER")

        elif width_error > self.size_tolerance:
            messages.append("MOVE BACK")

        if height_error < -self.size_tolerance:
            if "MOVE CLOSER" not in messages:
                messages.append("MOVE CLOSER")

        elif height_error > self.size_tolerance:
            if "MOVE BACK" not in messages:
                messages.append("MOVE BACK")

        if not messages:
            status = "GOOD POSITION"
        else:
            status = " | ".join(messages)

        return {
            "bbox": (x1, y1, x2, y2),
            "center": (face_center_x, face_center_y),
            "width": face_width,
            "height": face_height,
            "status": status,
            "position_ok": not messages,
        }

    def draw(self, frame, result):
        # Target guide
        target_x1 = (
            self.target_center_x - self.target_width // 2
        )

        target_y1 = (
            self.target_center_y - self.target_height // 2
        )

        target_x2 = (
            self.target_center_x + self.target_width // 2
        )

        target_y2 = (
            self.target_center_y + self.target_height // 2
        )

        cv2.rectangle(
            frame,
            (target_x1, target_y1),
            (target_x2, target_y2),
            (255, 255, 255),
            2,
        )

        # Actual face bounding box
        x1, y1, x2, y2 = result["bbox"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        # Face center
        center_x, center_y = result["center"]

        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (255, 255, 255),
            -1,
        )

        # Target center
        cv2.circle(
            frame,
            (
                self.target_center_x,
                self.target_center_y,
            ),
            5,
            (255, 255, 255),
            -1,
        )

        # Status
        cv2.putText(
            frame,
            result["status"],
            (30, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )