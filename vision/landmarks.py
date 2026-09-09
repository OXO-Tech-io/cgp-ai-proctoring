import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
from vision.landmark_indices import SOLVEPNP_6POINT, LEFT_IRIS_CENTER,RIGHT_IRIS_CENTER

base_options = python.BaseOptions(model_asset_path="/home/aanish/cgp-ai-proctoring/face_landmarker.task")
options = vision.FaceLandmarkerOptions(base_options=base_options)
landmarker = vision.FaceLandmarker.create_from_options(options)
print("Loaded successfully")

class FaceLandmarkerWrapper:
    def __init__(self, model_path: str):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options)
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        print("Loaded successfully")
        self._timings = []

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        start = time.perf_counter()
        detection_result = self.landmarker.detect(mp_image)
        elapsed_ms = (time.perf_counter() - start) * 1000
        # print(f"Detection time: {elapsed_ms:.2f} ms")

        self._timings.append(elapsed_ms)
        if len(self._timings) == 30:
            avg = sum(self._timings) / len(self._timings)
            print(f"Average detection time over last 30 frames: {avg:.2f} ms")
            self._timings.clear()

        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                for landmark in face_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    print(f"Landmark: ({x}, {y})")

                for idx in SOLVEPNP_6POINT:
                    lm = face_landmarks[idx]
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

                for idx in [LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER]:
                    lm = face_landmarks[idx]
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

        return detection_result.face_landmarks
