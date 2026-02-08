from __future__ import annotations

import numpy as np

from core.base import Tracker
from core.registry import register
from core.types import TrackedBBox, TrackingResult


@register("tracker", "yolo11-botsort")
class YOLO11BoTSORTTracker(Tracker):
    """YOLO11 detector with BoT-SORT multi-object tracking."""

    VALID_SIZES = ("n", "s", "m", "l", "x")

    def __init__(self, model_size: str = "n", tracker_type: str = "botsort.yaml"):
        assert model_size in self.VALID_SIZES, \
            f"Invalid size: {model_size}. Must be one of {self.VALID_SIZES}"
        self.model_size = model_size
        self.tracker_type = tracker_type
        self._model = None

    def load(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(f"yolo11{self.model_size}.pt")

    def update(self, frame: np.ndarray) -> TrackingResult:
        assert self._model is not None, "Call load() first"
        assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected (H,W,3), got {frame.shape}"

        results = self._model.track(
            frame, persist=True, tracker=self.tracker_type, verbose=False
        )[0]

        boxes = []
        for box in results.boxes:
            track_id = int(box.id[0]) if box.id is not None else -1
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append(TrackedBBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(box.conf[0]),
                class_id=int(box.cls[0]),
                class_name=results.names[int(box.cls[0])],
                track_id=track_id,
            ))
        return TrackingResult(boxes=boxes, frame_idx=-1)

    def reset(self) -> None:
        """Reset tracker state for a new video."""
        if self._model is not None:
            self._model.predictor = None

    def unload(self) -> None:
        del self._model
        self._model = None


@register("tracker", "yolo11-bytetrack")
class YOLO11ByteTracker(YOLO11BoTSORTTracker):
    """YOLO11 detector with ByteTrack multi-object tracking."""

    def __init__(self, model_size: str = "n"):
        super().__init__(model_size=model_size, tracker_type="bytetrack.yaml")
