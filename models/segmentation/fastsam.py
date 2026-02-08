from __future__ import annotations

import numpy as np
import torch

from core.base import Segmenter
from core.registry import register
from core.types import Mask, SegmentationResult


@register("segmenter", "fastsam")
class FastSAMSegmenter(Segmenter):
    """FastSAM segmenter via Ultralytics.

    Uses the CASIA FastSAM checkpoint by default (`FastSAM-s.pt`).
    """

    def __init__(self, model_name: str = "FastSAM-s.pt", conf: float = 0.25, iou: float = 0.9):
        self.model_name = model_name
        self.conf = conf
        self.iou = iou
        self._model = None
        self._device = None

    def load(self) -> None:
        from ultralytics import FastSAM

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = FastSAM(self.model_name)

    def predict(self, frame: np.ndarray, text_prompt: str | None = None) -> SegmentationResult:
        del text_prompt  # FastSAM is used in "segment everything" mode in this pipeline.
        assert self._model is not None, "Call load() first"
        assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected (H,W,3), got {frame.shape}"

        result = self._model(
            frame,
            device=self._device,
            retina_masks=True,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )[0]

        if result.masks is None or result.masks.data is None:
            return SegmentationResult(masks=[], frame_idx=-1)

        mask_data = result.masks.data
        if hasattr(mask_data, "cpu"):
            mask_data = mask_data.cpu().numpy()

        scores = []
        if result.boxes is not None and getattr(result.boxes, "conf", None) is not None:
            conf = result.boxes.conf
            scores = conf.cpu().numpy().tolist() if hasattr(conf, "cpu") else conf.tolist()

        masks = []
        for i in range(mask_data.shape[0]):
            score = float(scores[i]) if i < len(scores) else 1.0
            masks.append(
                Mask(
                    binary_mask=mask_data[i].astype(bool),
                    score=score,
                )
            )

        return SegmentationResult(masks=masks, frame_idx=-1)

    def unload(self) -> None:
        del self._model
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
