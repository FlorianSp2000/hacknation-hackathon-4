from __future__ import annotations

import cv2
import numpy as np

from core.types import (
    DetectionResult, SegmentationResult, VLMResult, FrameResult,
    TrackingResult, TemporalChange,
)


# Deterministic colors for segmentation masks
_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
]


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic color for a track ID using a hash-based approach."""
    if track_id < 0:
        return (128, 128, 128)
    # Use a simple hash to map track IDs to distinct colors
    h = (track_id * 41 + 7) % 180  # hue in [0, 180)
    hsv = np.array([[[h, 255, 220]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_detections(frame: np.ndarray, det: DetectionResult) -> np.ndarray:
    out = frame.copy()
    for box in det.boxes:
        pt1 = (int(box.x1), int(box.y1))
        pt2 = (int(box.x2), int(box.y2))
        cv2.rectangle(out, pt1, pt2, (0, 255, 0), 2)
        label = f"{box.class_name} {box.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, pt1, (pt1[0] + tw, pt1[1] - th - 4), (0, 255, 0), -1)
        cv2.putText(out, label, (pt1[0], pt1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out


def draw_tracking(frame: np.ndarray, tracking: TrackingResult) -> np.ndarray:
    """Draw tracked bboxes with persistent color-coded IDs."""
    out = frame.copy()
    for box in tracking.boxes:
        color = _track_color(box.track_id)
        pt1 = (int(box.x1), int(box.y1))
        pt2 = (int(box.x2), int(box.y2))
        cv2.rectangle(out, pt1, pt2, color, 2)

        label = f"ID:{box.track_id} {box.class_name} {box.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (pt1[0], pt1[1] - th - 8), (pt1[0] + tw, pt1[1]), color, -1)
        cv2.putText(out, label, (pt1[0], pt1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out


def draw_segmentation(frame: np.ndarray, seg: SegmentationResult) -> np.ndarray:
    out = frame.copy()
    overlay = frame.copy()
    for i, mask in enumerate(seg.masks):
        color = _COLORS[i % len(_COLORS)]
        overlay[mask.binary_mask] = color
    cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)

    # Draw mask scores
    for i, mask in enumerate(seg.masks):
        ys, xs = np.where(mask.binary_mask)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            label = mask.label or f"#{i}"
            cv2.putText(out, f"{label} {mask.score:.2f}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return out


def draw_vlm_text(frame: np.ndarray, vlm: VLMResult) -> np.ndarray:
    out = frame.copy()
    text = vlm.raw_text[:300]  # truncate for display
    y = 30
    for line in text.split("\n"):
        cv2.putText(out, line[:80], (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 18
    return out


def draw_temporal_changes(frame: np.ndarray, tc: TemporalChange) -> np.ndarray:
    """Overlay state change annotations on the frame."""
    out = frame.copy()
    y = 30

    # Header
    header = f"Changes ({tc.frame_idx_before} -> {tc.frame_idx_after})"
    cv2.putText(out, header, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    y += 25

    # State changes
    for sc in tc.state_changes:
        text = f"{sc.object_name}: {sc.before_state} -> {sc.after_state} [{sc.confidence}]"
        cv2.putText(out, text[:80], (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        y += 20

    # Actions
    for action in tc.actions_detected:
        cv2.putText(out, f"Action: {action[:60]}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
        y += 18

    return out


def draw_all(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    out = frame.copy()
    if result.segmentation:
        out = draw_segmentation(out, result.segmentation)
    if result.tracking:
        out = draw_tracking(out, result.tracking)
    elif result.detection:
        out = draw_detections(out, result.detection)
    if result.temporal_changes:
        out = draw_temporal_changes(out, result.temporal_changes)
    if result.vlm:
        out = draw_vlm_text(out, result.vlm)
    return out
