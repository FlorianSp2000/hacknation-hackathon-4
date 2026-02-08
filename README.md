# world2data-cam

A lightweight Python CLI for macOS (Apple Silicon, including M4) that:
- captures webcam video with a live preview,
- records video with on-frame timestamps,
- runs YOLO object detection + tracking on saved video,
- runs offline MediaPipe hand detection with ROI-based contact scoring,
- extracts frames from video,
- creates and updates a structured World2Data JSON.

## Requirements
- macOS on Apple Silicon
- Python 3.11+
- Webcam permission granted to Terminal/iTerm (System Settings -> Privacy & Security -> Camera)

## Setup (macOS Apple Silicon)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ultralytics Notes (macOS)
- `ultralytics` is installed from `requirements.txt`.
- First run with `--model yolov8n.pt` may download model weights automatically if they are not already cached.
- For offline runs, pass a local model path to `--model`.
- This project runs detection on CPU by default (no CUDA required).

## Quickstart
```bash
PYTHONPATH=src python -m world2data_cam record --camera 0 --out recordings/session.mp4 --fps 30 --width 1280 --height 720
PYTHONPATH=src python -m world2data_cam frames --video recordings/session.mp4 --out frames/session --fps 10
PYTHONPATH=src python -m world2data_cam stub --frames frames/session --out frames/session/world2data_stub.json
PYTHONPATH=src python -m world2data_cam detect --video recordings/session.mp4 --out outputs/session --fps 10 --model yolov8n.pt --w2d frames/session/world2data_stub.json
```

## CLI Usage
### Record from webcam
```bash
PYTHONPATH=src python -m world2data_cam record --camera 0 --out recordings/session.mp4 --fps 30 --width 1280 --height 720
```

Controls:
- `r`: start/stop recording
- `q`: quit preview and exit

Behavior notes:
- Live preview always runs until quit.
- Recording overlays:
  - current timestamp (`YYYY-MM-DD HH:MM:SS`)
  - red `REC` indicator while recording
- Recording writes MP4 output using `mp4v`.

If camera open fails, try camera indices `0`, `1`, `2`.

### Extract frames from a video
```bash
PYTHONPATH=src python -m world2data_cam frames --video recordings/session.mp4 --out frames/session --fps 10
```

### Create World2Data JSON stub
```bash
PYTHONPATH=src python -m world2data_cam stub --frames frames/session --out frames/session/world2data_stub.json
```

The stub now includes a pre-filled door-opening ontology:
- objects: `person`, `door`, `handle`, `doorway`
- states: `door.state` in `closed | ajar | open`
- affordances: `openable`, `graspable`, `traversable`
- actions: `approach`, `reach`, `grasp`, `pull`, `push`, `open`, `pass_through`

### Detect + Track objects and update World2Data JSON
```bash
PYTHONPATH=src python -m world2data_cam detect --video recordings/session.mp4 --out outputs/session --fps 10 --model yolov8n.pt --w2d frames/session/world2data_stub.json
```

Outputs:
- `outputs/session/overlay.mp4`
  - video with bbox, class, track id, confidence
- `outputs/session/summary.json`
  - counts per class and number of tracks
- `frames/session/world2data_stub.json`
  - updated **in place** with real tracks (persons and other detected classes)
  - ontology auto-filled if missing
  - `events` and `state_changes` kept empty

### Offline hand contact vs handle ROI
```bash
PYTHONPATH=src python -m world2data_cam hands_contact \
  --video input/open-door.mp4 \
  --out outputs/open-door_hands \
  --fps 10 \
  --handle-roi "540,280,660,420" \
  --max-hands 2 \
  --min-score 0.5 \
  --contact-thresh 0.15
```

Outputs:
- `outputs/open-door_hands/overlay.mp4`
- `outputs/open-door_hands/frame_stream.jsonl`
- `outputs/open-door_hands/metrics.json`

Door state (closed/ajar/open) from ROI + World2Data updates:
```bash
PYTHONPATH=src python -m world2data_cam hands_contact \
  --video input/open-door.mp4 \
  --out outputs/open-door_hands \
  --fps 10 \
  --handle-roi "540,280,660,420" \
  --door-roi "430,120,860,560" \
  --w2d frames/session/world2data_stub.json \
  --baseline-frames 10 \
  --ema 0.2 \
  --t1 0.18 \
  --t2 0.35 \
  --min-state-len 5
```

This adds per-frame `door` fields in `frame_stream.jsonl`, writes `state_changes`, and updates `door_1` track in the W2D JSON.

Debug ROI overlap on one sampled frame first:
```bash
PYTHONPATH=src python -m world2data_cam hands_contact \
  --video input/open-door.mp4 \
  --out outputs/open-door_hands \
  --fps 10 \
  --handle-roi "540,280,660,420" \
  --frame-idx 80 \
  --debug-png outputs/debug_contact.png
```

Debug door score/state on one sampled frame:
```bash
PYTHONPATH=src python -m world2data_cam hands_contact \
  --video input/open-door.mp4 \
  --out outputs/open-door_hands \
  --fps 10 \
  --handle-roi "540,280,660,420" \
  --door-roi "430,120,860,560" \
  --debug-door-frame 80 \
  --debug-door-png outputs/debug_door.png
```

Then run full metrics:
```bash
PYTHONPATH=src python -m world2data_cam hands_contact \
  --video input/open-door.mp4 \
  --out outputs/open-door_hands \
  --fps 10 \
  --handle-roi "540,280,660,420" \
  --max-hands 2 \
  --min-score 0.5 \
  --contact-thresh 0.15
```

How to pick `--handle-roi`:
- Open a representative frame/screenshot from the video.
- Use any pixel-inspector tool (Preview, screenshot annotator, or editor) to read `(x, y)` values.
- Mark top-left and bottom-right corners around the handle area.
- Pass as `"x1,y1,x2,y2"`.

## Dev helper script
Run:
```bash
bash scripts/dev.sh
```

This script creates/uses `.venv`, installs dependencies, and launches a demo record session.
