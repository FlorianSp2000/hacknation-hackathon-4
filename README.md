# World2Data

This repo currently contains two related toolchains:

1. `world2data_cam` CLI (`src/world2data_cam`) for webcam/video capture, frame export, detection/tracking, hand-contact, and door-state signals.
2. A multi-model pipeline app (remote project files under `core/`, `models/`, `pipeline/`, `app.py`) for detection + segmentation + VLM workflows.

## Quick Start (CLI)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m world2data_cam hands_contact --help
```

## Quick Start (Remote App)

```bash
uv sync
streamlit run app.py
```

## Model Notes

- Detection: YOLO (`ultralytics`)
- Segmentation: SAM 3
- VLM: LFM2.5
- Hand pipeline: MediaPipe Hands

## Common CLI flows

Record:

```bash
PYTHONPATH=src python -m world2data_cam record --camera 0 --out recordings/session.mp4 --fps 30 --width 1280 --height 720
```

Hands + contact + optional W2D update:

```bash
PYTHONPATH=src python -m world2data_cam hands_contact \
  --video input/open-door.mp4 \
  --out outputs/open-door_hands \
  --fps 10 \
  --handle-roi "280,330,380,470" \
  --w2d frames/session/world2data_stub.json
```

Door state from ROI:

```bash
PYTHONPATH=src python -m world2data_cam hands_contact \
  --video input/open-door.mp4 \
  --out outputs/open-door_hands \
  --fps 10 \
  --handle-roi "280,330,380,470" \
  --door-roi "430,120,860,560" \
  --door-baseline-seconds 2.0 \
  --door-smooth-alpha 0.2 \
  --door-ajar-on 0.08 --door-ajar-off 0.05 \
  --door-open-on 0.16 --door-open-off 0.10
```
