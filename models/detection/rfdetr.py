from __future__ import annotations

import numpy as np
from PIL import Image

from core.base import Detector
from core.registry import register
from core.types import BBox, DetectionResult
from models._rfdetr_compat import patch_transformers_for_rfdetr


@register("detector", "rf-detr")
class RFDETRDetector(Detector):
    VALID_SIZES = ("n", "s", "b", "m", "l")
    _CLASS_MAP = {
        "n": "RFDETRNano",
        "s": "RFDETRSmall",
        "b": "RFDETRBase",
        "m": "RFDETRMedium",
        "l": "RFDETRLarge",
    }

    def __init__(self, model_size: str = "b", threshold: float = 0.5):
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

    def predict(self, frame: np.ndarray) -> DetectionResult:
        assert self._model is not None, "Call load() first"
        assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected (H,W,3), got {frame.shape}"

        # BGR -> RGB -> PIL
        pil_image = Image.fromarray(frame[:, :, ::-1])
        detections = self._model.predict(pil_image, threshold=self.threshold)

        boxes = []
        for i in range(len(detections.xyxy)):
            x1, y1, x2, y2 = detections.xyxy[i].tolist()
            cid = int(detections.class_id[i])
            boxes.append(BBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(detections.confidence[i]),
                class_id=cid,
                class_name=self._coco_classes[cid],
            ))
        return DetectionResult(boxes=boxes, frame_idx=-1)

    def unload(self) -> None:
        del self._model
        self._model = None
        self._coco_classes = None
