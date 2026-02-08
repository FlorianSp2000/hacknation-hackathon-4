# World2Data — Navigation Ground Truth from Video

AI-powered pipeline that converts raw video into structured, navigation-relevant ground truth for humanoid robots. Detects objects, tracks them across frames, classifies navigation states (open/closed/blocked), and exports temporal ground truth with accuracy evaluation.

## Models

| Category | Model | Sizes | Notes |
|---|---|---|---|
| Detection | YOLOv8 | n/s/m/l/x | Ultralytics |
| Detection | YOLO11 | n/s/m/l/x | Ultralytics |
| Detection | RF-DETR | n/s/b/m/l | DINOv2 backbone, real-time transformer |
| Segmentation | SAM 3 | — | Text-prompted, streaming video sessions |
| Segmentation | FastSAM | — | Ultralytics, segment-everything mode |
| Segmentation | RF-DETR Seg | s/m/l | Instance segmentation with COCO classes |
| Tracking | YOLO Tracker | n/s/m/l/x | ByteTrack with persistent IDs |
| VLM | LFM2.5-VL | — | Transformers backend (CUDA/CPU) |
| VLM | LFM2.5-VL ONNX | — | ONNX Runtime (fp16 encoder + q4 decoder) |
| VLM | LFM2.5-VL MLX | — | Apple Silicon (mlx-vlm, 8-bit) |

## Features

- **Object tracking** with persistent IDs across frames
- **Navigation state classification** (doors, drawers, handles → open/closed/ajar/blocked)
- **Temporal diff** — VLM compares consecutive frames to detect state transitions
- **Hand-object interactions** via MediaPipe
- **Navigation timeline** — per-object state timeline with transition events
- **Ground truth export** — structured JSON with per-frame annotations
- **Accuracy evaluation** — compare predictions against manual labels
- **Per-frame timing** — inference latency breakdown per model stage

## Quick Start

```bash
uv sync
streamlit run app.py
```

If `uv` is not installed:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
streamlit run app.py
```

Requires GPU with >=8GB VRAM (sequential offload) or >=16GB (all models loaded).

## Architecture

```
core/           — types, ABCs, registry, video I/O
models/         — one file per model variant, registered via @register
  detection/    — yolov8, yolo11, rf-detr
  segmentation/ — sam3, fastsam, rf-detr-seg
  vlm/          — lfm2.5-vl (transformers, onnx, mlx)
  tracking/     — yolo-tracker
pipeline/       — orchestration, visualization, interactions, nav state, export, evaluation
app.py          — Streamlit UI (upload-based inference)
pages/          — additional Streamlit pages
```

## Adding a Model

1. Create `models/<category>/<name>.py`
2. Implement the ABC with `load()`, `predict()`, `unload()`
3. Decorate with `@register("category", "name")` and import in `models/__init__.py`

```python
from core.base import Detector
from core.registry import register

@register("detector", "my-detector")
class MyDetector(Detector):
    VALID_SIZES = ("s", "l")  # optional, enables size selector in UI
    def __init__(self, model_size="s"): ...
    def load(self): ...
    def predict(self, frame): ...
    def unload(self): ...
```

## Cluster Deployment (Singularity + SLURM)

```bash
# Build image (after dep changes)
singularity build world2data.sif world2data.def

# Deploy
source .env && scp -r . ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_PATH}

# Submit
sbatch start_app.sbatch

# Tunnel
ssh -L 8501:localhost:8501 -J ${CLUSTER_USER}@${CLUSTER_HOST} -p 7665 ${CLUSTER_USER}@<compute-node>
```

See [CLAUDE.md](CLAUDE.md) for full architecture details and compatibility notes.
