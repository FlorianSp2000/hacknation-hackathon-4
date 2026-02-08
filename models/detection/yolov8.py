from __future__ import annotations

import numpy as np

from core.base import Detector
from core.registry import register
from core.types import BBox, DetectionResult


@register("detector", "yolov8")
class YOLOv8Detector(Detector):
    VALID_SIZES = ("n", "s", "m", "l", "x")

    def __init__(self, model_size: str = "n"):
        assert model_size in self.VALID_SIZES, \
            f"Invalid size: {model_size}. Must be one of {self.VALID_SIZES}"
        self.model_size = model_size
        self._model = None

    def load(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(f"yolov8{self.model_size}.pt")

    def predict(self, frame: np.ndarray) -> DetectionResult:
        assert self._model is not None, "Call load() first"
        assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected (H,W,3), got {frame.shape}"

        results = self._model(frame, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append(BBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(box.conf[0]),
                class_id=int(box.cls[0]),
                class_name=results.names[int(box.cls[0])],
            ))
        return DetectionResult(boxes=boxes, frame_idx=-1)

    def unload(self) -> None:
        del self._model
        self._model = None
