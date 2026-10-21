import time
from typing import Sequence
from uuid import uuid4

import cv2
import numpy as np

from config import gaze_config as cfg
from temporal import FrameSample, SessionRecorder, SlidingWindow
from validation.data_logger import PoseDataLogger
from validation.gaze_logger import GazeDataLogger
from validation.guided_session import GuidedSession
from validation.key_labels import digit_key_labels
from vision.gaze.gaze_pipeline import (
    CalibrationEvent,
    CalibrationStatus,
    GazePipeline,
    GazeResult,
)
from vision.gaze.types import Landmark
from vision.head_pose import HeadPoseEstimator
from vision.landmarks import FaceLandmarkerWrapper
from temporal import DeviationThresholds, TemporalMonitor
from temporal.output import EventLog

DEBUG = False  

MODEL_PATH = "/home/aanish/cgp-ai-proctoring/face_landmarker.task"

SESSION_KEYS = digit_key_labels(
    ["A_BASELINE", "B_EYES_ONLY", "C_HEAD_YAW", "D_HEAD_PITCH"]
)

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)


def put_text(
    frame: np.ndarray,
    text: str,
    y: int,
    color: tuple[int, int, int] = GREEN,
    scale: float = 0.8,
) -> None:
    cv2.putText(
        frame, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA
    )


def turn_state(yaw: float, pitch: float) -> str:
    if yaw < -10.0:
        return "Turn Left"
    elif yaw > 10.0:
        return "Turn Right"
    elif pitch < -10.0:
        return "Turn Up"
    elif pitch > 10.0:
        return "Turn Down"
    else:
        return "CENTER"


def format_calibration_status(status: CalibrationStatus) -> str:
    """On-screen calibration text."""
    if status.phase == "WAITING":
        return f"Calibrating in {status.remaining_s:.0f}s - look at the center"
    return f"Calibrating {status.samples}/{status.required} - hold still"


def format_calibration_event(event: CalibrationEvent) -> str:
    """Console line for a one-off calibration milestone."""
    d = event.data
    if event.kind == "SAMPLE":
        return f"Gaze calibration: {d['samples']:.0f}/{d['required']:.0f}"
    elif event.kind == "SPREAD":
        return (
            f"Calibration spread: lx={d['left_x']:.4f}, ly={d['left_y']:.4f}, "
            f"rx={d['right_x']:.4f}, ry={d['right_y']:.4f} | head std: "
            f"yaw={d['head_yaw']:.2f}, pitch={d['head_pitch']:.2f} deg"
        )
    elif event.kind == "UNSTABLE":
        return (
            f"Calibration unstable (iris std > {d['iris_threshold']} or "
            f"head std > {d['head_threshold']} deg). "
            "Restarting - keep head still, look at center."
        )
    else:
        return (
            "\n========================================\n"
            "Gaze calibration complete.\n"
            f"Left neutral: ({d['left_neutral_x']:.4f}, {d['left_neutral_y']:.4f})\n"
            f"Right neutral: ({d['right_neutral_x']:.4f}, {d['right_neutral_y']:.4f})\n"
            f"Neutral pose: yaw={d['neutral_yaw']:+.2f}, pitch={d['neutral_pitch']:+.2f}\n"
            "========================================\n"
        )


class LiveSession:
    """Webcam loop: head pose + gaze estimation with on-screen overlay,
    CSV logging, and Phase 3 temporal recording."""

    def __init__(self) -> None:
        self.landmarker = FaceLandmarkerWrapper(MODEL_PATH)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open the webcam stream.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.head_pose = HeadPoseEstimator(self.width, self.height)
        self.gaze_pipeline = GazePipeline()

        self.session_label = "NONE"
        self.guided = GuidedSession()
        self.target_label = ""

        self.pose_logger = PoseDataLogger(f"validation/data/live/pose_{uuid4().hex}.csv")
        self.pose_logger.start()

        self.gaze_logger = GazeDataLogger(f"validation/data/live/gaze_{uuid4().hex}.csv")
        self.gaze_logger.start()

        self.prev_time = time.time()
        self.fps = 0.0

        # Phase 3: temporal layer
        self.window = SlidingWindow(duration_s=3.0)
        self.frame_idx = 0
        self.recording_path = f"data/sessions/{int(time.time())}.jsonl"
        self.session_id = uuid4().hex
        meta = {"camera": 0, "width": self.width, "height": self.height}
        self.recorder = SessionRecorder(self.recording_path, session_id=self.session_id, meta=meta)
        self.recorder.__enter__()

        self.thresholds = DeviationThresholds(gaze_yaw_enter=0.055)
        self.event_logs = EventLog(self.recording_path, self.session_id, self.thresholds, meta=meta)
        self.monitor = TemporalMonitor(self.thresholds)

    def print_banner(self) -> None:
        print("====================================================")
        print("Webcam capture loop active.")
        print("-> Press [Q] to quit, [C] to recalibrate gaze")
        print("-> Keys 1-4 set the session label (A/B/C/D)")
        print("-> Key 5 runs the guided eye-movement session (after calibration)")
        print("-> Gaze calibration: look at the center of the screen, keep head still")
        print("====================================================")

    def run(self) -> None:
        self.print_banner()

        while self.cap.isOpened():
            success, frame = self.cap.read()

            if not success:
                continue

            self._update_fps()
            self._process_frame(frame)

            cv2.imshow("Webcam Capture", frame)

            if self._handle_key(cv2.waitKey(1) & 0xFF):
                break

    def close(self) -> None:
        for ev in self.monitor.stop(time.monotonic()):
            self.event_logs.add(ev)
        self.landmarker.landmarker.close()
        self.cap.release()
        self.pose_logger.close()
        self.gaze_logger.close()
        self.recorder.__exit__(None, None, None)
        path = self.event_logs.write()
        print(f"Events Written to {path}")
        cv2.destroyAllWindows()

    def _update_fps(self) -> None:
        curr_time = time.time()
        if curr_time != self.prev_time:
            self.fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time

    def _emit_sample(
        self,
        face_count: int,
        yaw: float | None = None,
        pitch: float | None = None,
        roll: float | None = None,
        gaze_yaw: float | None = None,
        gaze_pitch: float | None = None,
    ) -> None:
        """Build one FrameSample per frame; record it and push it to the window."""
        sample = FrameSample(
            ts=time.monotonic(),
            frame_idx=self.frame_idx,
            face_count=face_count,
            confidence=1.0 if face_count else 0.0,
            head_yaw=yaw,
            head_pitch=pitch,
            head_roll=roll,
            gaze_yaw=gaze_yaw,
            gaze_pitch=gaze_pitch,
        )
        self.frame_idx += 1
        self.recorder.write(sample)
        self.window.push(sample)

        for ev in self.monitor.update(sample):
            self.event_logs.add(ev)

    def _process_frame(self, frame: np.ndarray) -> None:
        faces = self.landmarker.detect(frame)

        if faces:
            self._process_face(frame, faces[0], face_count=len(faces))
        else:
            self.gaze_pipeline.on_face_lost()
            self._emit_sample(face_count=0)

        put_text(frame, f"FPS: {self.fps:.1f}", 40, scale=1)


    def _process_face(
        self,
        frame: np.ndarray,
        face_landmarks: Sequence[Landmark],
        face_count: int = 1,
    ) -> None:
        pitch, yaw, roll = self.head_pose.estimate(
            face_landmarks, self.width, self.height
        )

        self.pose_logger.log(
            test="LIVE",
            sample=0,
            yaw=yaw,
            roll=roll,
            pitch=pitch,
            relative_pitch=None,
            relative_yaw=None,
            relative_roll=None,
            detection_time_ms=self.landmarker.last_detection_time_ms,
        )

        result = self.gaze_pipeline.update(face_landmarks, yaw, pitch)

        gaze_ok = result.has_gaze and result.reliable and result.filtered is not None
        self._emit_sample(
            face_count=face_count,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            gaze_yaw=result.filtered[0] if gaze_ok else None,
            gaze_pitch=result.filtered[1] if gaze_ok else None,
        )

        self._update_guided_session(frame)
        
       

        for event in result.events:
            if event.kind not in ("SAMPLE", "SPREAD", "UNSTABLE"):
                d = event.data
                self.monitor.start(d["neutral_yaw"], d["neutral_pitch"])
                self.event_logs.add_calibration(d["neutral_yaw"], d["neutral_pitch"], time.monotonic())
                print(format_calibration_event(event))

        if result.calibration is not None:
            put_text(frame, format_calibration_status(result.calibration), 290, ORANGE)

        if result.has_gaze:
            self._log_gaze(result, pitch, yaw, roll)

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

        put_text(frame, f"Session: {self.session_label}", 255, YELLOW)

    def _update_guided_session(self, frame: np.ndarray) -> None:
        step = self.guided.update(time.time())
        if step is not None:
            self.target_label, prompt = step
            put_text(frame, prompt, 330, YELLOW, 1.2)
        elif self.target_label:
            self.target_label = ""
            self.session_label = "NONE"
            print("Guided session finished.")

    def _log_gaze(self, result: GazeResult, pitch: float, yaw: float, roll: float) -> None:
        self.gaze_logger.log(
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
            detection_time_ms=self.landmarker.last_detection_time_ms,
            filtered=result.filtered,
            reliable=result.reliable,
            gaze_label=result.label,
            session=self.session_label,
            target=self.target_label,
        )

        if DEBUG:
            print(
                f"Gaze | {result.status} yaw={yaw:+.2f} pitch={pitch:+.2f} | "
                f"raw=({result.left_offset[0]:+.4f},{result.left_offset[1]:+.4f}) "
                f"comp=({result.gaze[0]:+.4f},{result.gaze[1]:+.4f}) "
                f"filt=({result.filtered[0]:+.4f},{result.filtered[1]:+.4f}) "
                f"{result.label}"
            )

    def _handle_key(self, key: int) -> bool:
        """Apply a keypress. Returns True when the user asked to quit."""
        if key in SESSION_KEYS:
            self.session_label = SESSION_KEYS[key]
            print(f"Session -> {self.session_label}")

        if key == ord("5"):
            if self.gaze_pipeline.calibrated:
                self.session_label = "B_EYES_ONLY"
                self.guided.start(time.time())
                print("Guided session started: follow the on-screen prompts, move only your eyes.")
            else:
                print("Calibrate first (wait for 'Gaze calibration complete').")

        if key == ord("c"):
            self.guided.stop()
            self.target_label = ""

            self.gaze_pipeline.start_calibration()
            self.monitor.reset()

            print("\n========================================")
            print("Gaze calibration started.")
            print("Look naturally at the center of the screen.")
            print(f"Collecting {cfg.CALIB_SAMPLES} samples...")
            print("========================================\n")

        return key == ord("q")


def main() -> None:
    session = LiveSession()
    try:
        session.run()
    except KeyboardInterrupt:
        print("\nInterrupted - shutting down.")
    finally:
        session.close()


if __name__ == "__main__":
    main()