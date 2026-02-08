from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from core.types import FrameResult, TrackedBBox, NavigationTimeline


def mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Encode a binary mask as run-length encoding (COCO-compatible)."""
    pixels = binary_mask.flatten(order="F")  # Fortran order (column-major) for COCO
    runs = []
    prev = 0
    count = 0
    for p in pixels:
        if p == prev:
            count += 1
        else:
            runs.append(count)
            count = 1
            prev = p
        # note: 'p' is already bool/int
    runs.append(count)
    # COCO RLE starts with 0-count, so if first pixel is 1, prepend a 0
    if pixels[0]:
        runs = [0] + runs
    return {
        "counts": runs,
        "size": [binary_mask.shape[0], binary_mask.shape[1]],
    }


def _frame_to_dict(r: FrameResult, fps: float) -> dict:
    """Convert a single FrameResult to a JSON-serializable dict."""
    frame_data: dict = {
        "frame_idx": r.frame_idx,
        "timestamp_s": round(r.frame_idx / fps, 4) if fps > 0 else 0.0,
    }

    # Objects from tracking (preferred) or detection
    objects = []
    if r.tracking:
        for box in r.tracking.boxes:
            objects.append({
                "track_id": box.track_id,
                "class": box.class_name,
                "confidence": round(box.confidence, 4),
                "bbox": [round(box.x1, 1), round(box.y1, 1),
                         round(box.x2, 1), round(box.y2, 1)],
            })
    elif r.detection:
        for box in r.detection.boxes:
            obj = {
                "class": box.class_name,
                "confidence": round(box.confidence, 4),
                "bbox": [round(box.x1, 1), round(box.y1, 1),
                         round(box.x2, 1), round(box.y2, 1)],
            }
            if isinstance(box, TrackedBBox):
                obj["track_id"] = box.track_id
            objects.append(obj)
    frame_data["objects"] = objects

    # Segmentation (RLE-encoded masks)
    if r.segmentation:
        frame_data["segmentation"] = [
            {
                "label": m.label,
                "score": round(m.score, 4),
                "rle": mask_to_rle(m.binary_mask),
            }
            for m in r.segmentation.masks
        ]

    # VLM output
    if r.vlm:
        frame_data["vlm"] = r.vlm.parsed if r.vlm.parsed else {"raw_text": r.vlm.raw_text}

    # Temporal changes
    if r.temporal_changes:
        tc = r.temporal_changes
        frame_data["temporal_changes"] = {
            "frame_before": tc.frame_idx_before,
            "frame_after": tc.frame_idx_after,
            "state_changes": [
                {
                    "object": sc.object_name,
                    "before": sc.before_state,
                    "after": sc.after_state,
                    "confidence": sc.confidence,
                }
                for sc in tc.state_changes
            ],
            "actions": tc.actions_detected,
        }

    return frame_data


def _build_tracks_summary(results: list[FrameResult]) -> dict:
    """Aggregate per-track information across all frames."""
    tracks: dict[int, dict] = defaultdict(lambda: {
        "class": None,
        "first_frame": None,
        "last_frame": None,
        "frame_count": 0,
        "avg_confidence": 0.0,
    })

    for r in results:
        if not r.tracking:
            continue
        for box in r.tracking.boxes:
            if box.track_id < 0:
                continue
            t = tracks[box.track_id]
            if t["class"] is None:
                t["class"] = box.class_name
            if t["first_frame"] is None or r.frame_idx < t["first_frame"]:
                t["first_frame"] = r.frame_idx
            if t["last_frame"] is None or r.frame_idx > t["last_frame"]:
                t["last_frame"] = r.frame_idx
            t["frame_count"] += 1
            # Running average
            n = t["frame_count"]
            t["avg_confidence"] = t["avg_confidence"] + (box.confidence - t["avg_confidence"]) / n

    # Round confidence
    for tid in tracks:
        tracks[tid]["avg_confidence"] = round(tracks[tid]["avg_confidence"], 4)

    return dict(tracks)


def _build_nav_gt(nav_timeline: NavigationTimeline) -> dict:
    """Build navigation ground truth section from a NavigationTimeline."""
    fps = nav_timeline.video_fps

    objects = []
    for obj in nav_timeline.objects:
        obj_data = {
            "track_id": obj.track_id,
            "name": obj.name,
            "type": obj.object_type,
            "yolo_class": obj.class_name,
            "state_timeline": [
                {
                    "state": entry.state,
                    "frame_start": entry.frame_start,
                    "frame_end": entry.frame_end,
                    "timestamp_start_s": round(entry.frame_start / fps, 3) if fps > 0 else 0,
                    "timestamp_end_s": round(entry.frame_end / fps, 3) if fps > 0 else 0,
                }
                for entry in obj.state_timeline
            ],
            "transitions": [
                {
                    "frame": frame,
                    "timestamp_s": round(frame / fps, 3) if fps > 0 else 0,
                    "from_state": from_s,
                    "to_state": to_s,
                    "event": f"{obj.name} {to_s}" if to_s != from_s else "no_change",
                }
                for frame, from_s, to_s in obj.transitions
            ],
        }
        objects.append(obj_data)

    return {
        "objects": objects,
        "summary": {
            "total_nav_objects": len(objects),
            "total_transitions": nav_timeline.total_transitions,
            "object_types": list(set(o.object_type for o in nav_timeline.objects)),
        },
    }


def export_ground_truth(
    results: list[FrameResult],
    video_info: dict,
    nav_timeline: NavigationTimeline | None = None,
) -> dict:
    """Convert pipeline results to structured ground truth JSON."""
    fps = video_info.get("fps", 30.0)

    gt = {
        "video": {
            "fps": fps,
            "width": video_info.get("width"),
            "height": video_info.get("height"),
            "frame_count": video_info.get("frame_count"),
        },
        "frames": [_frame_to_dict(r, fps) for r in results],
        "summary": {
            "total_frames_processed": len(results),
            "modalities": [],
        },
    }

    # Detect which modalities were used
    modalities = set()
    for r in results:
        if r.detection:
            modalities.add("detection")
        if r.tracking:
            modalities.add("tracking")
        if r.segmentation:
            modalities.add("segmentation")
        if r.vlm:
            modalities.add("vlm")
        if r.temporal_changes:
            modalities.add("temporal_changes")
    gt["summary"]["modalities"] = sorted(modalities)

    # Add tracks summary if tracking was used
    if "tracking" in modalities:
        gt["tracks_summary"] = _build_tracks_summary(results)

    # Add navigation ground truth if available
    if nav_timeline and nav_timeline.objects:
        gt["navigation_ground_truth"] = _build_nav_gt(nav_timeline)

    return gt


def export_ground_truth_json(
    results: list[FrameResult],
    video_info: dict,
    nav_timeline: NavigationTimeline | None = None,
) -> str:
    """Export ground truth as a formatted JSON string."""
    gt = export_ground_truth(results, video_info, nav_timeline=nav_timeline)
    return json.dumps(gt, indent=2, default=str)
