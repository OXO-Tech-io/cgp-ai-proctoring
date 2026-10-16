import time
import cv2
from uuid import uuid4
from vision.landmarks import FaceLandmarkerWrapper
from vision.head_pose import HeadPoseEstimator
from validation.data_logger import PoseDataLogger
from validation.gaze_logger import GazeDataLogger
from vision.gaze.gaze_pipeline import GazePipeline
from validation.guided_session import GuidedSession

DEBUG = False  # per-frame console prints

MODEL_PATH = "/home/aanish/cgp-ai-proctoring/face_landmarker.task"

SESSION_KEYS = {
    ord("1"): "A_BASELINE",
    ord("2"): "B_EYES_ONLY",
    ord("3"): "C_HEAD_YAW",
    ord("4"): "D_HEAD_PITCH",
}

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)


def put_text(frame, text, y, color=GREEN, scale=0.8):
    cv2.putText(
        frame, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA
    )


def turn_state(yaw, pitch):
    if yaw < -10.0:
        return "Turn Left"
    if yaw > 10.0:
        return "Turn Right"
    if pitch < -10.0:
        return "Turn Up"
    if pitch > 10.0:
        return "Turn Down"
    return "CENTER"


landmarker = FaceLandmarkerWrapper(MODEL_PATH)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Failed to open the webcam stream.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

head_pose = HeadPoseEstimator(width, height)
gaze_pipeline = GazePipeline()

session_label = "NONE"
guided = GuidedSession()
target_label = ""

pose_logger = PoseDataLogger(f"validation/data/live/pose_{uuid4().hex}.csv")
pose_logger.start()

gaze_logger = GazeDataLogger(f"validation/data/live/gaze_{uuid4().hex}.csv")
gaze_logger.start()

print("====================================================")
print("Webcam capture loop active.")
print("-> Press [Q] to quit, [C] to recalibrate gaze")
print("-> Keys 1-4 set the session label (A/B/C/D)")
print("-> Key 5 runs the guided eye-movement session (after calibration)")
print("-> Gaze calibration: look at the center of the screen, keep head still")
print("====================================================")

prev_time = time.time()
fps = 0

try:
    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            continue

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else fps
        prev_time = curr_time

        faces = landmarker.detect(frame)

        if faces:
            face_landmarks = faces[0]

            pitch, yaw, roll = head_pose.estimate(face_landmarks, width, height)

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

            result = gaze_pipeline.update(face_landmarks, yaw, pitch)

            step = guided.update(time.time())
            if step is not None:
                target_label, prompt = step
                put_text(frame, prompt, 330, YELLOW, 1.2)
            elif target_label:
                target_label = ""
                session_label = "NONE"
                print("Guided session finished.")

            for line in result.events:
                print(line)

            if result.message:
                put_text(frame, result.message, 290, ORANGE)

            if result.has_gaze:
                gaze_logger.log(
                    left_iris=result.left_iris,
                    left_normalized=result.left_normalized,
                    left_offset=result.left_offset,
                    right_iris=result.right_iris,
                    right_normalized=result.right_normalized,
                    right_offset=result.right_offset,
                    left_compensated=result.left_compensated,
                    right_compensated=result.right_compensated,
                    gaze=result.gaze,
                    pitch=pitch,
                    yaw=yaw,
                    roll=roll,
                    delta_yaw=result.delta_yaw,
                    delta_pitch=result.delta_pitch,
                    detection_time_ms=landmarker.last_detection_time_ms,
                    filtered=result.filtered,
                    reliable=result.reliable,
                    gaze_label=result.label,
                    session=session_label,
                    target=target_label,
                )

                if DEBUG:
                    print(
                        f"Gaze | {result.status} yaw={yaw:+.2f} pitch={pitch:+.2f} | "
                        f"raw=({result.left_offset[0]:+.4f},{result.left_offset[1]:+.4f}) "
                        f"comp=({result.gaze[0]:+.4f},{result.gaze[1]:+.4f}) "
                        f"filt=({result.filtered[0]:+.4f},{result.filtered[1]:+.4f}) "
                        f"{result.label}"
                    )

            put_text(frame, f"Pitch: {pitch:.1f}", 80)
            put_text(frame, f"Yaw: {yaw:.1f}", 115)
            put_text(frame, f"Roll: {roll:.1f}", 150)
            put_text(frame, turn_state(yaw, pitch), 185)

            if result.label is not None:
                put_text(
                    frame,
                    f"Gaze: {result.label}{'' if result.reliable else ' (?)'}",
                    220,
                )
            elif result.status == "BLINK" and result.calibrated:
                put_text(frame, "Gaze: (eyes closed)", 220, ORANGE)

            put_text(frame, f"Session: {session_label}", 255, YELLOW)

        else:
            gaze_pipeline.on_face_lost()

        put_text(frame, f"FPS: {fps:.1f}", 40, scale=1)

        cv2.imshow("Webcam Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in SESSION_KEYS:
            session_label = SESSION_KEYS[key]
            print(f"Session -> {session_label}")

        if key == ord("5"):
            if gaze_pipeline.calibrated:
                session_label = "B_EYES_ONLY"
                guided.start(time.time())
                print("Guided session started: follow the on-screen prompts, move only your eyes.")
            else:
                print("Calibrate first (wait for 'Gaze calibration complete').")

        if key == ord("c"):
            guided.stop()
            target_label = ""

            gaze_pipeline.start_calibration()

            print("\n========================================")
            print("Gaze calibration started.")
            print("Look naturally at the center of the screen.")
            print("Collecting 50 samples...")
            print("========================================\n")

        elif key == ord("q"):
            break
finally:
    landmarker.landmarker.close()
    cap.release()
    pose_logger.close()
    gaze_logger.close()
    cv2.destroyAllWindows()
