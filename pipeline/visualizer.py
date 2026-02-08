from __future__ import annotations

import cv2
import numpy as np

from core.types import DetectionResult, SegmentationResult, VLMResult, FrameResult


# Deterministic colors for segmentation masks
_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
]


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


def draw_hands_and_interactions(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    out = frame.copy()
    hand_poses = getattr(result, "hand_poses", [])
    interactions = getattr(result, "interactions", [])

    # Hands
    for hand in hand_poses:
        x1, y1, x2, y2 = hand.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(
            out,
            f"{hand.hand_id} {hand.handedness} {hand.score:.2f}",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 180, 0),
            1,
        )
        for x, y in hand.keypoints:
            cv2.circle(out, (x, y), 2, (0, 140, 255), -1)

    # Interactions
    y = 20
    for inter in interactions:
        text = f"{inter.hand_id} -> {inter.target_class}[{inter.target_index}] ({inter.contact_score:.2f})"
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 80), 1)
        y += 18

    return out


def draw_all(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    out = frame.copy()
    hand_poses = getattr(result, "hand_poses", [])
    interactions = getattr(result, "interactions", [])
    if result.segmentation:
        out = draw_segmentation(out, result.segmentation)
    if result.detection:
        out = draw_detections(out, result.detection)
    if hand_poses or interactions:
        out = draw_hands_and_interactions(out, result)
    if result.vlm:
        out = draw_vlm_text(out, result.vlm)
    return out
