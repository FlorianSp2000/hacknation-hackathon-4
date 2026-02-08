# World2Data — AI-Powered Ground Truth for Humanoid Navigation

**Track:** VC | **Core idea:** Replace manual data labeling with AI that auto-generates ground truth from raw video of humans in physical spaces.

## Problem

Humanoids fail in real environments because the physical world isn't labeled. Manual labeling is slow, expensive, and optimized for static images — not continuous, interaction-aware, temporal data humanoids need.

## Build

A pipeline that converts **raw video of humans navigating/interacting** into **structured, navigation-relevant ground truth**:

- Objects & affordances (graspable, openable, traversable)
- Human motion & intent
- Interaction events & state changes (open/closed, movable/fixed, blocked/clear)
- Temporal boundaries for actions/transitions

## Evaluation

| Criteria | Focus |
|---|---|
| **GT Accuracy** | Correct objects, states, interactions |
| **Temporal Precision** | Accurate action start/end boundaries |
| **Zero-Shot vs Fine-Tuned** | Measurable improvement from minimal data |
| **Demo Clarity** | World-to-GT transformation is obvious and convincing |
| **Technical Depth** | Complexity, implementation & engineering quality |
| **Communication** | Documentation & presentation (incl. video) |
| **Innovation & Creativity** | Originality and creative approach |

## Development

### Local

```bash
uv sync
streamlit run app.py
```

### Cluster (Singularity + SLURM)

```bash
# Build image (once, or after pyproject.toml changes)
singularity build world2data.sif world2data.def

# Deploy to cluster
source .env && scp -r . ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_PATH}

# Submit job
sbatch start_app.sbatch

# SSH tunnel to access UI
ssh -L 8501:localhost:8501 -J ${CLUSTER_USER}@${CLUSTER_HOST} -p 7665 ${CLUSTER_USER}@<compute-node>
# Then open localhost:8501
```

## Architecture

```
core/       — types, ABCs, registry, video I/O
models/     — one file per model variant, registered via @register
pipeline/   — orchestration (runner.py) + visualization (visualizer.py)
app.py      — Streamlit UI
```

### Adding a model

1. Create `models/<category>/<name>.py`
2. Implement the ABC (`Detector`, `Segmenter`, or `VLM`) with `load()`, `predict()`, `unload()`
3. Decorate with `@register("category", "name")` and import in `models/__init__.py`

### Adding a modality

1. Add dataclass to `core/types.py`, add field to `FrameResult`
2. Create ABC in `core/base.py`, add to `CATEGORY_BASE` in `core/registry.py`
3. Wire into `pipeline/runner.py` and `pipeline/visualizer.py`
4. Add UI controls in `app.py`

### Container structure

- **Image** (`world2data.sif`): CUDA + Python + all pip deps in `/opt/venv/`. No source code.
- **Bind mounts** (at runtime via sbatch):
  - `$(pwd)` → `/app` — source code
  - `$(pwd)/cache` → `/app/cache` — HuggingFace/torch model weight cache
- `PYTHONPATH=/app` set in `%environment` so imports resolve
