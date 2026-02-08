# World2Data — Multi-Model Video Inference Pipeline

AI-powered ground truth generation from raw video. Feeds frames through detection, segmentation, and vision-language models to produce structured, navigation-relevant annotations.

## Models

| Type | Model | Package |
|---|---|---|
| Detection | YOLOv8, YOLO11 | `ultralytics` |
| Segmentation | SAM 3 (text-prompted) | `transformers` |
| Vision-Language | LFM2.5-VL-1.6B | `transformers` |

## Quick Start

```bash
uv sync
streamlit run app.py
```

Requires GPU with >=8GB VRAM (sequential offload) or >=12GB (all models loaded).

## Architecture

```
core/       — types, ABCs, registry, video I/O
models/     — one file per model variant, registered via @register
pipeline/   — orchestration (runner.py) + visualization (visualizer.py)
app.py      — Streamlit UI
```

## Adding a Model

1. Create `models/<category>/<name>.py`
2. Implement the ABC (`Detector`, `Segmenter`, or `VLM`) with `load()`, `predict()`, `unload()`
3. Decorate with `@register("category", "name")` and import in `models/__init__.py`

```python
from core.base import Detector
from core.registry import register

@register("detector", "my-detector")
class MyDetector(Detector):
    def load(self): ...
    def predict(self, frame): ...
    def unload(self): ...
```

## Adding a Modality

1. Add dataclass to `core/types.py`, add field to `FrameResult`
2. Create ABC in `core/base.py`, add to `CATEGORY_BASE` in `core/registry.py`
3. Wire into `pipeline/runner.py` and `pipeline/visualizer.py`
4. Add UI controls in `app.py`

## Cluster Deployment

See [CLAUDE.md](CLAUDE.md) for Singularity container + SLURM setup.
