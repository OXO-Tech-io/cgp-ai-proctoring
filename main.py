import time
import cv2

from vision.landmarks import FaceLandmarkerWrapper
from vision.head_pose import HeadPoseEstimator
from validation.data_logger import PoseDataLogger


MODEL_PATH = "/home/aanish/cgp-ai-proctoring/face_landmarker.task"


landmarker = FaceLandmarkerWrapper(MODEL_PATH)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Failed to open the webcam stream.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# Get actual camera resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize head pose estimator once
head_pose = HeadPoseEstimator(width, height)

pose_logger = PoseDataLogger(
    "validation/data/pose_measurements.csv"
)

pose_logger.start()


print("====================================================")
print("Webcam capture loop active.")
print("-> Press [Q] to quit")
print("====================================================")


prev_time = time.time()
fps = 0


while cap.isOpened():

    success, frame = cap.read()

    if not success:
        continue

   

    # FPS
    curr_time = time.time()
    fps = (
        1 / (curr_time - prev_time)
        if curr_time != prev_time
        else fps
    )
    prev_time = curr_time


    # ----------------------------------------
    # MediaPipe Face Landmarks
    # ----------------------------------------

    faces = landmarker.detect(frame)


    # ----------------------------------------
    # Head Pose Estimation
    # ----------------------------------------

    if faces:

        # Use first detected face
        face_landmarks = faces[0]

        pitch, yaw, roll = head_pose.estimate(
            face_landmarks,
            width,
            height,
        )

        pose_logger.log(
            test="LIVE",
            sample=0,
            yaw=yaw,
            roll=roll,
            pitch=pitch,
            relative_pitch=None,
            relative_yaw=None,
            relative_roll=None,
            detection_time_ms=landmarker.last_detection_time_ms,
        )

        # Display head pose
        cv2.putText(
            frame,
            f"Pitch: {pitch:.1f}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Yaw: {yaw:.1f}",
            (30, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Roll: {roll:.1f}",
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


    # FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


    cv2.imshow("Webcam Capture", frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
pose_logger.close()
cv2.destroyAllWindows()