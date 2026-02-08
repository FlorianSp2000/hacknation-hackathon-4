from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


@dataclass
class DetectionResult:
    boxes: list[BBox]
    frame_idx: int


@dataclass
class Mask:
    binary_mask: np.ndarray  # (H, W) bool
    score: float
    label: str | None = None


@dataclass
class SegmentationResult:
    masks: list[Mask]
    frame_idx: int


@dataclass
class VLMResult:
    raw_text: str
    parsed: dict | None  # JSON parse attempt, None if failed
    frame_idx: int


@dataclass
class FrameResult:
    frame_idx: int
    frame: np.ndarray
    detection: DetectionResult | None = None
    segmentation: SegmentationResult | None = None
    vlm: VLMResult | None = None
