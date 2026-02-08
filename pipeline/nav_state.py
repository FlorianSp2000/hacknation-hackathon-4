"""Navigation state timeline builder.

Merges VLM navigation-state classifications with tracked object identities
to produce per-object state timelines with transition boundaries.
"""
from __future__ import annotations

from collections import defaultdict

from core.types import (
    FrameResult, TrackedBBox,
    ObjectStateEntry, TrackedNavObject, NavigationTimeline,
)


# Navigation-relevant COCO classes that YOLO can detect
_NAV_RELEVANT_COCO = {
    "refrigerator", "oven", "microwave", "toaster", "sink",
    "toilet", "door",  # COCO doesn't have door but custom models might
    "chair", "couch", "bed", "dining table", "tv",
    "suitcase", "backpack", "handbag",
}

# VLM nav_object types we care about
_NAV_TYPES = {"door", "drawer", "handle", "cabinet", "passage", "obstacle"}


def _bbox_iou(box_a: tuple[float, float, float, float],
              box_b: tuple[float, float, float, float]) -> float:
    """Compute IoU between two (x1, y1, x2, y2) bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_vlm_to_tracks(
    vlm_objects: list[dict],
    tracked_boxes: list[TrackedBBox],
    iou_threshold: float = 0.15,
) -> list[tuple[dict, int]]:
    """Match VLM nav_objects to tracked bboxes by best IoU overlap.

    VLM objects often don't have bbox info, so we also fall back to
    name/type matching against the YOLO class when IoU isn't available.

    Returns list of (vlm_object, track_id) pairs.
    """
    matches = []

    for vlm_obj in vlm_objects:
        vlm_bbox = vlm_obj.get("bbox_approx") or vlm_obj.get("bbox")
        vlm_name = str(vlm_obj.get("name", "")).lower()
        vlm_type = str(vlm_obj.get("type", "")).lower()

        best_track_id = -1
        best_score = -1.0

        for tbox in tracked_boxes:
            score = 0.0

            # IoU matching if VLM provided a bbox
            if vlm_bbox and len(vlm_bbox) == 4:
                try:
                    iou = _bbox_iou(
                        tuple(float(v) for v in vlm_bbox),
                        (tbox.x1, tbox.y1, tbox.x2, tbox.y2),
                    )
                    score = iou
                except (ValueError, TypeError):
                    pass

            # Name-based soft matching as fallback/boost
            yolo_class = tbox.class_name.lower()
            if vlm_name and (vlm_name in yolo_class or yolo_class in vlm_name):
                score += 0.3
            if vlm_type and (vlm_type in yolo_class or yolo_class in vlm_type):
                score += 0.2

            if score > best_score:
                best_score = score
                best_track_id = tbox.track_id

        # Accept match if we got any positive signal
        if best_track_id >= 0 and best_score > 0:
            matches.append((vlm_obj, best_track_id))
        else:
            # No track match — assign a synthetic negative ID based on name hash
            # so we can still build a timeline for unmatched VLM objects
            synthetic_id = -(abs(hash(vlm_name + vlm_type)) % 10000 + 1)
            matches.append((vlm_obj, synthetic_id))

    return matches


def build_nav_timeline(results: list[FrameResult], fps: float = 30.0) -> NavigationTimeline:
    """Build a NavigationTimeline from pipeline results.

    Merges tracked object identities with VLM navigation-state classifications
    to produce per-object state timelines with transition boundaries.
    """
    # Collect raw state observations: track_id -> [(frame_idx, state, name, type)]
    observations: dict[int, list[tuple[int, str, str, str]]] = defaultdict(list)
    # Track ID -> YOLO class name (from tracker)
    track_classes: dict[int, str] = {}

    # Record YOLO classes for all tracks
    for r in results:
        if r.tracking:
            for box in r.tracking.boxes:
                if box.track_id >= 0:
                    track_classes[box.track_id] = box.class_name

    # For each frame with VLM output, extract nav objects and match to tracks
    for r in results:
        if not r.vlm or not r.vlm.parsed:
            continue

        nav_objects = r.vlm.parsed.get("nav_objects", [])
        if not isinstance(nav_objects, list):
            continue

        # Get tracked boxes for this frame (if tracking enabled)
        tracked_boxes = r.tracking.boxes if r.tracking else []

        if tracked_boxes:
            matches = _match_vlm_to_tracks(nav_objects, tracked_boxes)
        else:
            # No tracking — use synthetic IDs
            matches = [
                (obj, -(abs(hash(str(obj.get("name", "")) + str(obj.get("type", "")))) % 10000 + 1))
                for obj in nav_objects
            ]

        for vlm_obj, track_id in matches:
            state = str(vlm_obj.get("state", "unknown")).lower()
            name = str(vlm_obj.get("name", "unknown"))
            obj_type = str(vlm_obj.get("type", "unknown")).lower()
            observations[track_id].append((r.frame_idx, state, name, obj_type))

    # Build per-object timelines
    nav_objects = []
    total_transitions = 0

    # Also include state changes from temporal_changes if available
    temporal_events: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
    for r in results:
        if r.temporal_changes:
            for sc in r.temporal_changes.state_changes:
                # Try to associate with a track by name matching
                sc_name = sc.object_name.lower()
                for tid, obs_list in observations.items():
                    for _, _, name, _ in obs_list:
                        if sc_name in name.lower() or name.lower() in sc_name:
                            temporal_events[tid].append(
                                (r.frame_idx, sc.before_state, sc.after_state)
                            )
                            break
                    else:
                        continue
                    break

    for track_id, obs_list in sorted(observations.items(), key=lambda x: x[0]):
        if not obs_list:
            continue

        # Sort by frame index
        obs_list.sort(key=lambda x: x[0])

        # Determine object identity from most common name/type
        names = [o[2] for o in obs_list]
        types = [o[3] for o in obs_list]
        most_common_name = max(set(names), key=names.count)
        most_common_type = max(set(types), key=types.count)
        yolo_class = track_classes.get(track_id, "unknown")

        # Build state timeline segments
        timeline: list[ObjectStateEntry] = []
        transitions: list[tuple[int, str, str]] = []

        current_state = obs_list[0][1]
        segment_start = obs_list[0][0]

        for i in range(1, len(obs_list)):
            frame_idx, state, _, _ = obs_list[i]
            if state != current_state:
                # Close previous segment
                timeline.append(ObjectStateEntry(
                    frame_start=segment_start,
                    frame_end=frame_idx,
                    state=current_state,
                ))
                transitions.append((frame_idx, current_state, state))
                current_state = state
                segment_start = frame_idx

        # Close final segment — use last frame from results
        last_frame = results[-1].frame_idx if results else segment_start
        timeline.append(ObjectStateEntry(
            frame_start=segment_start,
            frame_end=last_frame,
            state=current_state,
        ))

        total_transitions += len(transitions)

        nav_objects.append(TrackedNavObject(
            track_id=track_id,
            object_type=most_common_type,
            name=most_common_name,
            class_name=yolo_class,
            state_timeline=timeline,
            transitions=transitions,
        ))

    return NavigationTimeline(
        objects=nav_objects,
        total_transitions=total_transitions,
        video_fps=fps,
    )
