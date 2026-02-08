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
class TrackedBBox(BBox):
    track_id: int = -1


@dataclass
class DetectionResult:
    boxes: list[BBox]
    frame_idx: int


@dataclass
class TrackingResult:
    boxes: list[TrackedBBox]
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
class StateChange:
    object_name: str
    before_state: str
    after_state: str
    confidence: str  # "high" / "medium" / "low"


@dataclass
class TemporalChange:
    state_changes: list[StateChange]
    actions_detected: list[str]
    frame_idx_before: int
    frame_idx_after: int
    raw_text: str


@dataclass
class ObjectStateEntry:
    frame_start: int
    frame_end: int
    state: str  # open/closed/ajar/blocked/clear/unknown


@dataclass
class TrackedNavObject:
    track_id: int
    object_type: str  # door/drawer/handle/cabinet/passage/obstacle
    name: str  # human-readable name from VLM
    class_name: str  # from YOLO tracker
    state_timeline: list[ObjectStateEntry] = field(default_factory=list)
    transitions: list[tuple[int, str, str]] = field(default_factory=list)  # (frame, from_state, to_state)


@dataclass
class NavigationTimeline:
    objects: list[TrackedNavObject] = field(default_factory=list)
    total_transitions: int = 0
    video_fps: float = 30.0


@dataclass
class FrameResult:
    frame_idx: int
    frame: np.ndarray
    detection: DetectionResult | None = None
    segmentation: SegmentationResult | None = None
    vlm: VLMResult | None = None
    tracking: TrackingResult | None = None
    temporal_changes: TemporalChange | None = None
