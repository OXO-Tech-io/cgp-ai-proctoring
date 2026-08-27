# CGP AI Proctoring System

AI-based proctoring pipeline using MediaPipe for gaze/head-pose detection, YOLOv8 for object detection, and multi-signal risk scoring — built with a human-in-the-loop review design rather than automatic flagging.

## Project Status

**Current Phase:** Phase 0 — Foundation (in progress)

## Roadmap

- [x] Phase 0 — Foundation
  - [x] Environment setup
  - [x] Webcam capture loop
  - [ ] MediaPipe Face Landmarker integration
  - [ ] Landmark index mapping
  - [ ] Debug overlay scaffold
- [ ] Phase 1 — Head Pose Estimation
- [ ] Phase 2 — Gaze Estimation
- [ ] Phase 3 — Temporal Aggregation Layer
- [ ] Phase 4 — Face Count / Proxy Detection
- [ ] Phase 5 — Object Detection (YOLOv8)
- [ ] Phase 6 — Audio (VAD)
- [ ] Phase 7 — Fusion / Risk Scoring
- [ ] Phase 8 — Human Review Layer

## Progress Log

### Phase 0 — Foundation

#### Webcam Capture Loop ✅
Implemented basic OpenCV `VideoCapture` loop at 1280x720 with a rolling FPS overlay and clean exit handling.

---

## Setup

```bash
uv sync
uv run python main.py
```

## Project Structure

```
cgp-ai-proctoring/
├── main.py
├── vision/
│   ├── landmarks.py
│   ├── head_pose.py
│   └── gaze.py
├── debug/
│   └── overlay.py
└── config/
```
