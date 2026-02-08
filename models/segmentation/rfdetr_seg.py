from __future__ import annotations

import numpy as np
from PIL import Image

from core.base import Segmenter
from core.registry import register
from core.types import Mask, SegmentationResult
from models._rfdetr_compat import patch_transformers_for_rfdetr


@register("segmenter", "rf-detr-seg")
class RFDETRSegmenter(Segmenter):
    """RF-DETR instance segmentation via DINOv2 backbone."""

    VALID_SIZES = ("s", "m", "l")
    _CLASS_MAP = {
        "s": "RFDETRSegSmall",
        "m": "RFDETRSegMedium",
        "l": "RFDETRSegLarge",
    }

    def __init__(self, model_size: str = "m", threshold: float = 0.5):
        assert model_size in self.VALID_SIZES, \
            f"Invalid size: {model_size}. Must be one of {self.VALID_SIZES}"
        self.model_size = model_size
        self.threshold = threshold
        self._model = None
        self._coco_classes = None

    def load(self) -> None:
        patch_transformers_for_rfdetr()
        import rfdetr
        from rfdetr.util.coco_classes import COCO_CLASSES

        cls_name = self._CLASS_MAP[self.model_size]
        model_cls = getattr(rfdetr, cls_name)
        self._model = model_cls()
        self._coco_classes = COCO_CLASSES

    def predict(self, frame: np.ndarray, text_prompt: str | None = None) -> SegmentationResult:
        assert self._model is not None, "Call load() first"
        assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected (H,W,3), got {frame.shape}"

        # BGR -> RGB -> PIL
        pil_image = Image.fromarray(frame[:, :, ::-1])
        detections = self._model.predict(pil_image, threshold=self.threshold)

        masks = []
        if detections.mask is not None:
            for i in range(len(detections.mask)):
                cid = int(detections.class_id[i])
                masks.append(Mask(
                    binary_mask=detections.mask[i].astype(bool),
                    score=float(detections.confidence[i]),
                    label=self._coco_classes[cid],
                ))

        return SegmentationResult(masks=masks, frame_idx=-1)

    def unload(self) -> None:
        del self._model
        self._model = None
        self._coco_classes = None
