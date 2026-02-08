"""SigLIP-based zero-shot navigation state classifier.

Crops tracked bounding boxes from frames and classifies their state
(open/closed/ajar) using SigLIP image-text similarity. ~50ms per crop
vs ~10s for VLM generation — 200x speedup.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from core.base import StateClassifier
from core.registry import register


# Default candidate labels for navigation state classification.
# Each label maps to a normalized state name.
DEFAULT_CANDIDATE_LABELS = [
    "an open door",
    "a closed door",
    "a partially open door",
    "an open drawer",
    "a closed drawer",
    "an open cabinet",
    "a closed cabinet",
    "an open refrigerator",
    "a closed refrigerator",
    "a clear passage",
    "a blocked passage",
    "an obstacle blocking the path",
]

# Map candidate labels to simplified state strings
_LABEL_TO_STATE = {
    "an open door": "open",
    "a closed door": "closed",
    "a partially open door": "ajar",
    "an open drawer": "open",
    "a closed drawer": "closed",
    "an open cabinet": "open",
    "a closed cabinet": "closed",
    "an open refrigerator": "open",
    "a closed refrigerator": "closed",
    "a clear passage": "clear",
    "a blocked passage": "blocked",
    "an obstacle blocking the path": "blocked",
}


def _crop_bbox(frame: np.ndarray, bbox: tuple, padding: int = 10) -> np.ndarray | None:
    """Crop a bounding box region from a BGR frame with optional padding."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    # Validate
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return crop


@register("state_classifier", "siglip")
class SigLIPStateClassifier(StateClassifier):
    """Zero-shot navigation state classifier using SigLIP.

    Classifies cropped bbox regions against navigation state labels
    using image-text similarity. Fast (~50ms/crop) and zero-shot.
    """

    MODEL_ID = "google/siglip-base-patch16-224"

    def __init__(self, candidate_labels: list[str] | None = None):
        self.candidate_labels = candidate_labels or DEFAULT_CANDIDATE_LABELS
        self._pipe = None

    def load(self) -> None:
        from transformers import pipeline as hf_pipeline

        device = 0 if torch.cuda.is_available() else -1
        self._pipe = hf_pipeline(
            "zero-shot-image-classification",
            model=self.MODEL_ID,
            device=device,
        )

    def classify(self, frame: np.ndarray, boxes: list[tuple]) -> list[tuple[str, float]]:
        """Classify state of each cropped bbox region.

        Args:
            frame: BGR numpy (H,W,3)
            boxes: list of (x1, y1, x2, y2, track_id) tuples

        Returns: list of (state, confidence) pairs, one per box.
        """
        assert self._pipe is not None, "Call load() first"

        results = []
        for bbox in boxes:
            crop = _crop_bbox(frame, bbox[:4])
            if crop is None:
                results.append(("unknown", 0.0))
                continue

            # BGR -> RGB -> PIL
            pil_crop = Image.fromarray(crop[:, :, ::-1])

            # Run zero-shot classification
            preds = self._pipe(pil_crop, candidate_labels=self.candidate_labels)

            if preds:
                top = preds[0]  # Highest confidence
                label = top["label"]
                score = float(top["score"])
                state = _LABEL_TO_STATE.get(label, "unknown")
                results.append((state, score))
            else:
                results.append(("unknown", 0.0))

        return results

    def unload(self) -> None:
        del self._pipe
        self._pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
