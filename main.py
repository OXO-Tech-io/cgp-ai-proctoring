import time
import cv2
from vision.landmarks import FaceLandmarkerWrapper

landmarker = FaceLandmarkerWrapper("/home/aanish/cgp-ai-proctoring/face_landmarker.task")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Failed to open the webcam stream.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

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

    frame = cv2.flip(frame, 1)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else fps
    prev_time = curr_time

    landmarker.detect(frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2 ,cv2.LINE_AA)
    
    cv2.imshow("Webcam Capture", frame)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()