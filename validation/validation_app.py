import cv2

from vision.landmarks import FaceLandmarkerWrapper
from vision.head_pose import HeadPoseEstimator
from validation.face_guide import FacePositionGuide
from validation.pose_test import PoseTestRecorder
from validation.neutral_calibration import NeutralPoseCalibrator
from validation.data_logger import PoseDataLogger

class HeadPoseValidationApp:
    def __init__(self, model_path: str, camera_index: int = 0):
        self.model_path = model_path
        self.camera_index = camera_index

        self.landmarker = FaceLandmarkerWrapper(model_path)

        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open the webcam stream.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.head_pose = HeadPoseEstimator(self.width, self.height)

        self.face_guide = FacePositionGuide(self.width, self.height)

        self.pose_recorder = PoseTestRecorder(sample_count=10)

        self.neutral_calibrator = NeutralPoseCalibrator(
            sample_count=50
        )

        self.pose_logger = PoseDataLogger(
            "validation/data/pose_measurements.csv"
        )

        self.pose_logger.start()


    def run(self):
        print("====================================================")
        print("Webcam capture loop active.")
        print("-> Press [Q] to quit")
        print("====================================================")

        prev_time = 0
        fps = 0

        while self.cap.isOpened():
            success, frame = self.cap.read()

            if not success:
                continue

            frame = cv2.flip(frame, 1)

            faces = self.landmarker.detect(frame)

            pitch = None
            yaw = None
            roll = None
            relative_pose = None
            face_detected = False

            if faces:
                face_landmarks = faces[0]
                face_detected = True

                try:
                    pitch, yaw, roll = self.head_pose.estimate(
                        face_landmarks,
                        self.width,
                        self.height,
                    )

                    test_name = self.pose_recorder.active_test

                    if test_name:
                        sample_number = len(self.pose_recorder.samples) + 1
                    else:
                        sample_number = 0

                    # ----------------------------------------
                    # Face position guide
                    # ----------------------------------------

                    guide_result = self.face_guide.evaluate(
                        face_landmarks
                    )

                    self.face_guide.draw(
                        frame,
                        guide_result,
                    )

                    # ----------------------------------------
                    # Neutral calibration
                    # ----------------------------------------

                    calibration_sample = None

                    if self.neutral_calibrator.active:
                        calibration_sample = (
                            len(self.neutral_calibrator.samples) + 1
                        )

                        self.neutral_calibrator.add_sample(
                            pitch,
                            yaw,
                            roll,
                        )

                    # ----------------------------------------
                    # Relative pose
                    # ----------------------------------------

                    relative_pose = (
                        self.neutral_calibrator.get_relative_pose(
                            pitch,
                            yaw,
                            roll,
                        )
                    )

                    # ----------------------------------------
                    # Log neutral calibration
                    # ----------------------------------------

                    if calibration_sample is not None:
                        self.pose_logger.log(
                            test="NEUTRAL_CALIBRATION",
                            sample=calibration_sample,
                            pitch=pitch,
                            yaw=yaw,
                            roll=roll,
                            relative_pitch=None,
                            relative_yaw=None,
                            relative_roll=None,
                            detection_time_ms=(
                                self.landmarker.last_detection_time_ms
                            ),
                        )

                    # ----------------------------------------
                    # Log controlled pose test
                    # ----------------------------------------

                    if test_name is not None:

                        relative_pitch = None
                        relative_yaw = None
                        relative_roll = None

                        if relative_pose is not None:
                            (
                                relative_pitch,
                                relative_yaw,
                                relative_roll,
                            ) = relative_pose

                        self.pose_logger.log(
                            test=test_name,
                            sample=sample_number,
                            pitch=pitch,
                            yaw=yaw,
                            roll=roll,
                            relative_pitch=relative_pitch,
                            relative_yaw=relative_yaw,
                            relative_roll=relative_roll,
                            detection_time_ms=(
                                self.landmarker.last_detection_time_ms
                            ),
                        )

                    # ----------------------------------------
                    # Add sample to pose recorder
                    # ----------------------------------------

                    self.pose_recorder.add_sample(
                        pitch,
                        yaw,
                        roll,
                    )

                except Exception as e:
                    print(f"Error estimating head pose: {e}")

            self.draw_pose_info(
                frame, 
                pitch, 
                yaw, 
                roll, 
                relative_pose, 
                face_detected,
                
            )


            cv2.imshow("Head Pose Validation", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                self.neutral_calibrator.start()

            elif key == ord("x"):
                self.neutral_calibrator.cancel()

            if key == ord('q'):
                break


            self.pose_recorder.handle_key(key)

        self.release()

    def draw_pose_info(
        self,
        frame,
        pitch,
        yaw,
        roll,
        relative_pose,
        face_detected,
    ):
        # Face detection status
        cv2.putText(
            frame,
            f"Face Detected: {'Yes' if face_detected else 'No'}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if face_detected else (0, 0, 255),
            2,
        )

        # Face status
        status = (
            "Face Detected"
            if face_detected
            else "No Face Detected"
        )

        cv2.putText(
            frame,
            status,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if face_detected else (0, 0, 255),
            2,
        )

        # Raw head-pose values
        if pitch is not None:
            cv2.putText(
                frame,
                f"Pitch: {pitch:.1f}",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Yaw: {yaw:.1f}",
                (30, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Roll: {roll:.1f}",
                (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if relative_pose is not None:
            delta_pitch, delta_yaw, delta_roll = relative_pose

            cv2.putText(
                frame,
                f"dPitch: {delta_pitch:+.1f}",
                (30, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"dYaw:   {delta_yaw:+.1f}",
                (30, 235),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"dRoll:  {delta_roll:+.1f}",
                (30, 265),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # --------------------------------------------------
        # Pose test information
        # --------------------------------------------------

        if self.pose_recorder.active_test:
            cv2.putText(
                frame,
                f"TEST: {self.pose_recorder.active_test}",
                (30, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            current, total = self.pose_recorder.get_progress()

            cv2.putText(
                frame,
                f"Samples: {current}/{total}",
                (30, 255),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if self.neutral_calibrator.active:
            cv2.putText(
                frame,
                "CALIBRATING NEUTRAL POSE",
                (30, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            current, total = self.neutral_calibrator.get_progress()

            cv2.putText(
                frame,
                f"Calibration: {current}/{total}",
                (30, 335),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # --------------------------------------------------
        # Keyboard instructions
        # --------------------------------------------------

        cv2.putText(
            frame,
            "1 Neutral | 2 Left | 3 Right",
            (30, self.height - 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "4 Up | 5 Down | 6 Tilt L | 7 Tilt R | 0 Stop",
            (30, self.height - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (30, self.height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        

    def release(self):
        self.pose_logger.close()
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    model_path = "/home/aanish/cgp-ai-proctoring/face_landmarker.task"

    app = HeadPoseValidationApp(model_path=model_path, camera_index=0,)

    app.run()

if __name__ == "__main__":
    main()