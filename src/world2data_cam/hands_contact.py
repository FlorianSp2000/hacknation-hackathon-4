"""Offline hand detection, contact scoring, and door-state inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class HandsContactConfig:
    """Configuration for hands/door extraction."""

    video: Path
    out_dir: Path
    fps: float
    handle_roi: tuple[int, int, int, int]
    max_hands: int
    min_score: float
    contact_thresh: float
    w2d: Path | None = None
    min_event_len: int = 3
    merge_gap: int = 2
    frame_idx: int | None = None
    debug_png: Path | None = None
    door_roi: tuple[int, int, int, int] | None = None
    door_baseline_seconds: float = 2.0
    door_smooth_alpha: float = 0.2
    door_ajar_on: float = 0.18
    door_ajar_off: float = 0.12
    door_open_on: float = 0.35
    door_open_off: float = 0.28
    debug_door_frame: int | None = None
    debug_door_png: Path | None = None


@dataclass(slots=True)
class _TrackState:
    """Internal state for simple hand ID association."""

    track_id: str
    centroid: tuple[float, float]
    last_frame_idx: int


@dataclass(slots=True)
class _ActiveSegment:
    """Internal running contact segment state."""

    start_frame: int
    start_t: float
    last_contact_frame: int
    last_contact_t: float
    gap_count: int
    hand_scores: dict[str, float]
    conf_sum: float
    conf_count: int
    contact_sum: float
    contact_count: int


@dataclass(slots=True)
class _DoorStateTracker:
    """Tracks smoothed door score and hysteresis state transitions."""

    baseline_sum: np.ndarray | None = None
    baseline_count: int = 0
    baseline_mean: np.ndarray | None = None
    baseline_scores: list[float] | None = None
    baseline_closed: float = 0.0
    baseline_ready: bool = False
    open_score: float = 0.0
    delta_smooth: float = 0.0
    state: str = "closed"


@dataclass(slots=True)
class _DoorEventSegment:
    """Internal segment for contiguous door-state interval."""

    state: str
    start_frame: int
    start_t: float
    end_frame: int
    end_t: float
    peak_delta: float
    delta_sum: float
    delta_count: int


def parse_roi(roi_text: str) -> tuple[int, int, int, int]:
    """Parse ROI text in the form 'x1,y1,x2,y2'."""
    try:
        parts = [int(x.strip()) for x in roi_text.split(",")]
    except ValueError as exc:
        raise ValueError("ROI must be integers: x1,y1,x2,y2") from exc

    if len(parts) != 4:
        raise ValueError("ROI must have exactly 4 values: x1,y1,x2,y2")

    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI must satisfy x2>x1 and y2>y1")
    return x1, y1, x2, y2


def _open_video(video_path: Path) -> cv2.VideoCapture:
    """Open input video with validation."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    return cap


def _load_mediapipe_solutions():
    """Load MediaPipe Hands + drawing utils across package variants."""
    try:
        import mediapipe as mp  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is not installed in this environment. "
            "Run: pip install mediapipe"
        ) from exc

    if hasattr(mp, "solutions"):
        return mp.solutions.hands, mp.solutions.drawing_utils

    try:
        from mediapipe.python import solutions as mp_solutions  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Installed mediapipe package does not expose Hands solution. "
            "Reinstall with: pip install --upgrade mediapipe"
        ) from exc

    return mp_solutions.hands, mp_solutions.drawing_utils


def _create_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Create MP4 video writer."""
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create overlay writer: {path}")
    return writer


def _clamp_point(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    """Clamp normalized landmark to image bounds."""
    px = int(round(x * (width - 1)))
    py = int(round(y * (height - 1)))
    return max(0, min(width - 1, px)), max(0, min(height - 1, py))


def _contact_score(points: list[tuple[int, int]], roi: tuple[int, int, int, int]) -> float:
    """Compute fraction of hand keypoints inside ROI."""
    x1, y1, x2, y2 = roi
    inside = sum(1 for x, y in points if x1 <= x <= x2 and y1 <= y <= y2)
    return inside / 21.0


def _associate_tracks(
    detections: list[dict[str, object]],
    tracks: dict[str, _TrackState],
    frame_idx: int,
    next_track_idx: int,
    max_distance_px: float = 120.0,
) -> int:
    """Assign stable IDs to detections using centroid distance."""
    unmatched_dets = set(range(len(detections)))
    unmatched_tracks = set(tracks.keys())

    candidates: list[tuple[float, str, int]] = []
    for track_id, track in tracks.items():
        tx, ty = track.centroid
        for det_idx, det in enumerate(detections):
            dx, dy = det["centroid"]
            dist = float(np.hypot(dx - tx, dy - ty))
            if dist <= max_distance_px:
                candidates.append((dist, track_id, det_idx))

    for _, track_id, det_idx in sorted(candidates, key=lambda item: item[0]):
        if det_idx not in unmatched_dets or track_id not in unmatched_tracks:
            continue
        detections[det_idx]["id"] = track_id
        tracks[track_id].centroid = detections[det_idx]["centroid"]
        tracks[track_id].last_frame_idx = frame_idx
        unmatched_dets.remove(det_idx)
        unmatched_tracks.remove(track_id)

    for det_idx in sorted(unmatched_dets):
        track_id = f"hand_{next_track_idx}"
        next_track_idx += 1
        detections[det_idx]["id"] = track_id
        tracks[track_id] = _TrackState(
            track_id=track_id,
            centroid=detections[det_idx]["centroid"],
            last_frame_idx=frame_idx,
        )

    stale_ids = [
        track_id
        for track_id, track in tracks.items()
        if frame_idx - track.last_frame_idx > 10
    ]
    for track_id in stale_ids:
        del tracks[track_id]

    return next_track_idx


def _finalize_segment(
    segments: list[dict[str, Any]],
    active: _ActiveSegment,
    min_event_len: int,
) -> None:
    """Finalize active segment into event schema if long enough."""
    frames = active.last_contact_frame - active.start_frame + 1
    if frames < min_event_len:
        return

    actor = (
        max(active.hand_scores.items(), key=lambda kv: kv[1])[0]
        if active.hand_scores
        else "hand_unknown"
    )
    avg_contact = (active.contact_sum / active.contact_count) if active.contact_count else 0.0
    avg_conf = (active.conf_sum / active.conf_count) if active.conf_count else 0.0

    segments.append(
        {
            "action": "contact",
            "actor": actor,
            "targets": ["handle_1"],
            "t_start": round(active.start_t, 2),
            "t_end": round(active.last_contact_t, 2),
            "confidence": round(avg_conf, 2),
            "metrics": {
                "avg_contact_score": round(avg_contact, 6),
                "avg_hand_conf": round(avg_conf, 6),
                "frames": frames,
            },
        }
    )


def _apply_door_hysteresis(
    state: str,
    delta_smooth: float,
    ajar_on: float,
    ajar_off: float,
    open_on: float,
    open_off: float,
) -> str:
    """Transition door state with hysteresis thresholds."""
    if state == "closed":
        if delta_smooth >= open_on:
            return "open"
        if delta_smooth >= ajar_on:
            return "ajar"
        return "closed"
    if state == "ajar":
        if delta_smooth >= open_on:
            return "open"
        if delta_smooth <= ajar_off:
            return "closed"
        return "ajar"
    if delta_smooth <= ajar_off:
        return "closed"
    if delta_smooth <= open_off:
        return "ajar"
    return "open"


def _door_state_confidence(state: str, delta_smooth: float, cfg: HandsContactConfig) -> float:
    """Heuristic confidence for door state-change events."""
    if state == "closed":
        margin = cfg.door_ajar_off - delta_smooth
        scale = max(cfg.door_ajar_off, 1e-6)
    elif state == "ajar":
        upper = cfg.door_open_on - delta_smooth
        lower = delta_smooth - cfg.door_ajar_on
        margin = min(upper, lower)
        scale = max(cfg.door_open_on - cfg.door_ajar_on, 1e-6)
    else:
        margin = delta_smooth - cfg.door_open_on
        scale = max(1.0 - cfg.door_open_on, 1e-6)
    raw = margin / scale
    return float(max(0.0, min(1.0, raw)))


def _door_segment_to_event(seg: _DoorEventSegment, idx: int) -> dict[str, Any]:
    """Convert door-state segment to event format."""
    frames = max(1, seg.end_frame - seg.start_frame + 1)
    avg_delta = seg.delta_sum / max(seg.delta_count, 1)
    confidence = max(0.0, min(1.0, seg.peak_delta))
    return {
        "event_id": f"evt_door_{idx:03d}",
        "action": "door_state",
        "actor": "door_1",
        "targets": ["door_1"],
        "t_start": round(seg.start_t, 2),
        "t_end": round(seg.end_t, 2),
        "confidence": round(confidence, 3),
        "metrics": {
            "state": seg.state,
            "peak_delta": round(seg.peak_delta, 6),
            "avg_delta": round(avg_delta, 6),
            "frames": frames,
        },
    }


def _ensure_door_ontology(payload: dict[str, Any]) -> None:
    """Ensure ontology contains hand/handle/door objects and door.state."""
    ontology = payload.get("ontology")
    if not isinstance(ontology, dict):
        ontology = {"objects": [], "states": [], "affordances": [], "actions": []}

    objects = ontology.get("objects")
    if not isinstance(objects, list):
        objects = []
    for obj_name in ("hand", "handle", "door"):
        if obj_name not in objects:
            objects.append(obj_name)
    ontology["objects"] = objects

    states = ontology.get("states")
    if not isinstance(states, list):
        states = []
    has_door_state = any(
        isinstance(item, dict)
        and item.get("object") == "door"
        and item.get("key") == "door.state"
        for item in states
    )
    if not has_door_state:
        states.append(
            {
                "object": "door",
                "key": "door.state",
                "values": ["closed", "ajar", "open"],
            }
        )
    ontology["states"] = states
    payload["ontology"] = ontology


def _update_w2d_payload(
    w2d_path: Path,
    events: list[dict[str, Any]],
    door_track_entries: list[dict[str, Any]],
    state_changes: list[dict[str, Any]],
) -> None:
    """Append events and door state outputs into World2Data JSON in place."""
    if not w2d_path.exists():
        raise FileNotFoundError(f"World2Data JSON not found: {w2d_path}")

    with w2d_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid World2Data format: {w2d_path}")

    _ensure_door_ontology(payload)

    existing_events = payload.get("events")
    if not isinstance(existing_events, list):
        existing_events = []
    existing_count = len(existing_events)
    for idx, event in enumerate(events, start=1):
        event_copy = dict(event)
        event_copy.setdefault("event_id", f"evt_event_{existing_count + idx:03d}")
        existing_events.append(event_copy)
    payload["events"] = existing_events

    tracks = payload.get("tracks")
    if not isinstance(tracks, list):
        tracks = []

    if door_track_entries:
        door_track = {
            "track_id": "door_1",
            "label": "door",
            "category": "object",
            "bbox_format": "xyxy",
            "per_frame": door_track_entries,
        }
        replaced = False
        for idx, track in enumerate(tracks):
            if isinstance(track, dict) and track.get("track_id") == "door_1":
                tracks[idx] = door_track
                replaced = True
                break
        if not replaced:
            tracks.append(door_track)
    payload["tracks"] = tracks

    existing_changes = payload.get("state_changes")
    if not isinstance(existing_changes, list):
        existing_changes = []
    existing_changes.extend(state_changes)
    payload["state_changes"] = existing_changes

    w2d_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_hands_contact(config: HandsContactConfig) -> None:
    """Run MediaPipe Hands and export overlay/jsonl/metrics."""
    if config.fps <= 0:
        raise ValueError("--fps must be > 0")
    if not (0.0 <= config.min_score <= 1.0):
        raise ValueError("--min-score must be between 0 and 1")
    if not (0.0 <= config.contact_thresh <= 1.0):
        raise ValueError("--contact-thresh must be between 0 and 1")
    if config.min_event_len <= 0:
        raise ValueError("--min-event-len must be >= 1")
    if config.merge_gap < 0:
        raise ValueError("--merge-gap must be >= 0")
    if config.door_baseline_seconds <= 0:
        raise ValueError("--door-baseline-seconds must be > 0")
    if not (0.0 <= config.door_smooth_alpha <= 1.0):
        raise ValueError("--door-smooth-alpha must be between 0 and 1")
    if not (
        0.0 <= config.door_ajar_off < config.door_ajar_on < config.door_open_on <= 1.0
    ):
        raise ValueError(
            "Door thresholds must satisfy 0 <= ajar_off < ajar_on < open_on <= 1"
        )
    if not (config.door_ajar_on < config.door_open_off < config.door_open_on):
        raise ValueError(
            "Door thresholds must satisfy ajar_on < open_off < open_on"
        )

    debug_contact_mode = config.frame_idx is not None
    debug_door_mode = config.debug_door_frame is not None
    if debug_contact_mode and debug_door_mode:
        raise ValueError("Use either contact debug flags or door debug flags, not both.")
    if debug_contact_mode and config.debug_png is None:
        raise ValueError("--debug-png is required when using --frame-idx")
    if debug_door_mode and config.debug_door_png is None:
        raise ValueError("--debug-door-png is required when using --debug-door-frame")
    if debug_door_mode and config.door_roi is None:
        raise ValueError("--door-roi is required when using --debug-door-frame")
    if config.frame_idx is not None and config.frame_idx < 0:
        raise ValueError("--frame-idx must be >= 0")
    if config.debug_door_frame is not None and config.debug_door_frame < 0:
        raise ValueError("--debug-door-frame must be >= 0")

    cap = _open_video(config.video)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if source_fps <= 0:
        source_fps = config.fps

    hx1, hy1, hx2, hy2 = config.handle_roi
    if hx1 < 0 or hy1 < 0 or hx2 >= width or hy2 >= height:
        raise ValueError(f"Handle ROI {config.handle_roi} out of bounds for {width}x{height}")

    door_roi = config.door_roi
    if door_roi is not None:
        dx1, dy1, dx2, dy2 = door_roi
        if dx1 < 0 or dy1 < 0 or dx2 >= width or dy2 >= height:
            raise ValueError(f"Door ROI {door_roi} out of bounds for {width}x{height}")

    print(f"Processing frame size: {width}x{height}")
    print(f"Final handle ROI used: {config.handle_roi}")
    if door_roi is not None:
        print(f"Final door ROI used: {door_roi}")

    config.out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = config.out_dir / "overlay.mp4"
    jsonl_path = config.out_dir / "frame_stream.jsonl"
    door_stream_path = config.out_dir / "door_stream.jsonl"
    events_path = config.out_dir / "events.json"
    metrics_path = config.out_dir / "metrics.json"

    debug_mode = debug_contact_mode or debug_door_mode
    debug_target_idx = (
        config.frame_idx if debug_contact_mode else config.debug_door_frame
    )
    writer = None if debug_mode else _create_writer(overlay_path, config.fps, width, height)

    frame_idx = 0
    processed_idx = 0
    next_emit_time = 0.0
    emit_interval = 1.0 / config.fps

    frames_with_hands = 0
    contact_frames = 0
    longest_contact_run = 0
    current_contact_run = 0
    hand_score_sum = 0.0
    hand_score_count = 0
    contact_score_sum = 0.0
    contact_score_count = 0

    tracks: dict[str, _TrackState] = {}
    next_track_idx = 1
    event_segments: list[dict[str, Any]] = []
    active_segment: _ActiveSegment | None = None

    door_tracker = _DoorStateTracker()
    door_tracker.baseline_scores = []
    door_track_entries: list[dict[str, Any]] = []
    state_changes: list[dict[str, Any]] = []
    door_events: list[dict[str, Any]] = []
    door_active_segment: _DoorEventSegment | None = None
    state_frame_counts: dict[str, int] = {"closed": 0, "ajar": 0, "open": 0}
    door_scores: list[float] = []

    mp_hands, mp_draw = _load_mediapipe_solutions()
    stream = None if debug_mode else jsonl_path.open("w", encoding="utf-8")
    door_stream = None if debug_mode else door_stream_path.open("w", encoding="utf-8")

    try:
        with mp_hands.Hands(
            static_image_mode=bool(debug_mode),
            max_num_hands=config.max_hands,
            min_detection_confidence=config.min_score,
            min_tracking_confidence=config.min_score,
        ) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                t_sec = frame_idx / source_fps
                if t_sec + 1e-9 < next_emit_time:
                    frame_idx += 1
                    continue
                next_emit_time += emit_interval

                if debug_mode and processed_idx != debug_target_idx:
                    processed_idx += 1
                    frame_idx += 1
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 165, 255), 2)
                cv2.putText(
                    frame,
                    "HANDLE ROI",
                    (hx1, max(18, hy1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

                door_info: dict[str, Any] | None = None
                if door_roi is not None:
                    dx1, dy1, dx2, dy2 = door_roi
                    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (255, 0, 0), 2)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_roi = gray[dy1 : dy2 + 1, dx1 : dx2 + 1].astype(np.float32)

                    if door_tracker.baseline_count < 10:
                        if door_tracker.baseline_sum is None:
                            door_tracker.baseline_sum = gray_roi.copy()
                        else:
                            door_tracker.baseline_sum += gray_roi
                        door_tracker.baseline_count += 1
                        if door_tracker.baseline_count == 10:
                            door_tracker.baseline_mean = (
                                door_tracker.baseline_sum / 10.0
                            )
                        open_score_raw = 0.0
                    else:
                        assert door_tracker.baseline_mean is not None
                        diff = np.abs(gray_roi - door_tracker.baseline_mean)
                        open_score_raw = float(np.mean(diff) / 255.0)

                    door_tracker.open_score = open_score_raw
                    door_scores.append(open_score_raw)

                    assert door_tracker.baseline_scores is not None
                    if t_sec <= config.door_baseline_seconds and door_tracker.baseline_mean is not None:
                        door_tracker.baseline_scores.append(open_score_raw)
                    if (
                        not door_tracker.baseline_ready
                        and t_sec >= config.door_baseline_seconds
                        and door_tracker.baseline_scores
                    ):
                        door_tracker.baseline_closed = float(
                            np.median(door_tracker.baseline_scores)
                        )
                        door_tracker.baseline_ready = True
                        door_tracker.delta_smooth = 0.0
                        door_tracker.state = "closed"

                    baseline_closed = (
                        door_tracker.baseline_closed if door_tracker.baseline_ready else 0.0
                    )
                    if door_tracker.baseline_ready:
                        delta = door_tracker.open_score - baseline_closed
                        door_tracker.delta_smooth = (
                            config.door_smooth_alpha * delta
                            + (1.0 - config.door_smooth_alpha) * door_tracker.delta_smooth
                        )

                        prev_state = door_tracker.state
                        door_tracker.state = _apply_door_hysteresis(
                            state=door_tracker.state,
                            delta_smooth=door_tracker.delta_smooth,
                            ajar_on=config.door_ajar_on,
                            ajar_off=config.door_ajar_off,
                            open_on=config.door_open_on,
                            open_off=config.door_open_off,
                        )
                        curr_state = door_tracker.state

                        if door_active_segment is None:
                            door_active_segment = _DoorEventSegment(
                                state=curr_state,
                                start_frame=processed_idx,
                                start_t=t_sec,
                                end_frame=processed_idx,
                                end_t=t_sec,
                                peak_delta=door_tracker.delta_smooth,
                                delta_sum=door_tracker.delta_smooth,
                                delta_count=1,
                            )
                        else:
                            door_active_segment.end_frame = processed_idx
                            door_active_segment.end_t = t_sec
                            door_active_segment.peak_delta = max(
                                door_active_segment.peak_delta,
                                door_tracker.delta_smooth,
                            )
                            door_active_segment.delta_sum += door_tracker.delta_smooth
                            door_active_segment.delta_count += 1

                        if curr_state != prev_state:
                            if door_active_segment is not None:
                                door_active_segment.end_frame = max(
                                    door_active_segment.start_frame,
                                    processed_idx - 1,
                                )
                                door_active_segment.end_t = max(
                                    door_active_segment.start_t,
                                    t_sec - emit_interval,
                                )
                                door_events.append(
                                    _door_segment_to_event(
                                        door_active_segment,
                                        idx=len(door_events) + 1,
                                    )
                                )
                            door_active_segment = _DoorEventSegment(
                                state=curr_state,
                                start_frame=processed_idx,
                                start_t=t_sec,
                                end_frame=processed_idx,
                                end_t=t_sec,
                                peak_delta=door_tracker.delta_smooth,
                                delta_sum=door_tracker.delta_smooth,
                                delta_count=1,
                            )
                            state_changes.append(
                                {
                                    "object_id": "door_1",
                                    "state_key": "door.state",
                                    "from": prev_state,
                                    "to": curr_state,
                                    "t": round(t_sec, 2),
                                    "confidence": round(
                                        _door_state_confidence(
                                            state=curr_state,
                                            delta_smooth=door_tracker.delta_smooth,
                                            cfg=config,
                                        ),
                                        3,
                                    ),
                                }
                            )
                    else:
                        delta = 0.0
                        door_tracker.delta_smooth = 0.0
                        curr_state = "closed"

                    state_frame_counts[curr_state] += 1
                    door_track_entries.append(
                        {
                            "t": round(t_sec, 6),
                            "bbox": [dx1, dy1, dx2, dy2],
                            "open_score": round(door_tracker.open_score, 6),
                            "state": curr_state,
                        }
                    )
                    door_info = {
                        "roi": [dx1, dy1, dx2, dy2],
                        "open_score": round(door_tracker.open_score, 6),
                        "state": curr_state,
                        "baseline_closed": round(baseline_closed, 6),
                        "delta": round(delta, 6),
                        "delta_smooth": round(door_tracker.delta_smooth, 6),
                    }
                    cv2.putText(
                        frame,
                        (
                            f"DOOR score={door_tracker.open_score:.3f} "
                            f"delta={door_tracker.delta_smooth:.3f} state={curr_state}"
                        ),
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

                detections: list[dict[str, object]] = []
                if result.multi_hand_landmarks and result.multi_handedness:
                    for hand_landmarks, handedness in zip(
                        result.multi_hand_landmarks,
                        result.multi_handedness,
                    ):
                        classif = handedness.classification[0]
                        conf = float(classif.score)
                        if conf < config.min_score:
                            continue
                        pts = [_clamp_point(lm.x, lm.y, width, height) for lm in hand_landmarks.landmark]
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                        centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
                        c_score = _contact_score(pts, config.handle_roi)
                        detections.append(
                            {
                                "id": "",
                                "handedness": str(classif.label),
                                "conf": conf,
                                "bbox": bbox,
                                "centroid": centroid,
                                "keypoints": [[int(px), int(py)] for px, py in pts],
                                "contact_score": c_score,
                                "contact": c_score >= config.contact_thresh,
                                "landmarks_obj": hand_landmarks,
                            }
                        )

                next_track_idx = _associate_tracks(
                    detections=detections,
                    tracks=tracks,
                    frame_idx=processed_idx,
                    next_track_idx=next_track_idx,
                )

                any_contact = False
                output_hands: list[dict[str, object]] = []
                debug_stats: list[dict[str, object]] = []
                for det in detections:
                    hand_id = str(det["id"])
                    conf = float(det["conf"])
                    bbox = det["bbox"]
                    c_score = float(det["contact_score"])
                    contact = bool(det["contact"])
                    handedness = str(det["handedness"])
                    kpts_in_roi = int(round(c_score * 21.0))

                    if contact:
                        any_contact = True

                    hand_score_sum += conf
                    hand_score_count += 1
                    contact_score_sum += c_score
                    contact_score_count += 1

                    x_min, y_min, x_max, y_max = bbox
                    color = (0, 0, 255) if contact else (0, 255, 0)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(
                        frame,
                        f"{hand_id} {handedness} conf={conf:.2f} contact={c_score:.2f}",
                        (x_min, max(18, y_min - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                    if contact:
                        cv2.putText(
                            frame,
                            "CONTACT",
                            (x_min, min(height - 10, y_max + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    mp_draw.draw_landmarks(frame, det["landmarks_obj"], mp_hands.HAND_CONNECTIONS)

                    output_hands.append(
                        {
                            "id": hand_id,
                            "handedness": handedness,
                            "conf": round(conf, 6),
                            "bbox": bbox,
                            "keypoints": det["keypoints"],
                            "contact_score": round(c_score, 6),
                            "contact": contact,
                        }
                    )
                    debug_stats.append(
                        {
                            "id": hand_id,
                            "bbox": bbox,
                            "conf": round(conf, 6),
                            "contact_score": round(c_score, 6),
                            "kpts_in_roi": kpts_in_roi,
                        }
                    )

                if output_hands:
                    frames_with_hands += 1
                    if any_contact:
                        contact_frames += 1

                contact_hands = [h for h in output_hands if bool(h["contact"])]
                if contact_hands:
                    if active_segment is None:
                        active_segment = _ActiveSegment(
                            start_frame=processed_idx,
                            start_t=t_sec,
                            last_contact_frame=processed_idx,
                            last_contact_t=t_sec,
                            gap_count=0,
                            hand_scores={},
                            conf_sum=0.0,
                            conf_count=0,
                            contact_sum=0.0,
                            contact_count=0,
                        )
                    active_segment.last_contact_frame = processed_idx
                    active_segment.last_contact_t = t_sec
                    active_segment.gap_count = 0
                    for hand in contact_hands:
                        hid = str(hand["id"])
                        cscore = float(hand["contact_score"])
                        conf = float(hand["conf"])
                        active_segment.hand_scores[hid] = active_segment.hand_scores.get(hid, 0.0) + cscore
                        active_segment.conf_sum += conf
                        active_segment.conf_count += 1
                        active_segment.contact_sum += cscore
                        active_segment.contact_count += 1
                elif active_segment is not None:
                    active_segment.gap_count += 1
                    if active_segment.gap_count > config.merge_gap:
                        _finalize_segment(event_segments, active_segment, config.min_event_len)
                        active_segment = None

                if any_contact:
                    current_contact_run += 1
                    longest_contact_run = max(longest_contact_run, current_contact_run)
                else:
                    current_contact_run = 0

                if debug_mode:
                    if debug_contact_mode:
                        cv2.putText(
                            frame,
                            f"frame={processed_idx} size={width}x{height} roi={hx1},{hy1},{hx2},{hy2}",
                            (10, 48 if door_info is not None else 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        if debug_stats:
                            y_txt = 72 if door_info is not None else 48
                            for stat in debug_stats:
                                cv2.putText(
                                    frame,
                                    (
                                        f"{stat['id']} conf={stat['conf']:.2f} "
                                        f"kpts_in_roi={stat['kpts_in_roi']}/21 "
                                        f"contact_score={stat['contact_score']:.2f}"
                                    ),
                                    (10, y_txt),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    2,
                                    cv2.LINE_AA,
                                )
                                y_txt += 20
                        else:
                            cv2.putText(
                                frame,
                                "No hands detected on this sampled frame",
                                (10, 72 if door_info is not None else 48),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )
                        assert config.debug_png is not None
                        config.debug_png.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(config.debug_png), frame)
                        print(
                            f"Debug frame={processed_idx} t={t_sec:.3f}s "
                            f"size={width}x{height} roi={config.handle_roi}"
                        )
                        if debug_stats:
                            for stat in debug_stats:
                                print(
                                    f"{stat['id']} bbox={stat['bbox']} conf={stat['conf']:.3f} "
                                    f"kpts_in_roi={stat['kpts_in_roi']}/21 "
                                    f"contact_score={stat['contact_score']:.3f}"
                                )
                        else:
                            print("No hands detected for selected frame.")
                        print(f"Debug image written: {config.debug_png}")
                    else:
                        assert config.debug_door_png is not None
                        config.debug_door_png.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(config.debug_door_png), frame)
                        if door_info is None:
                            print("Door ROI not configured; no door debug data.")
                        else:
                            print(
                                f"Door debug frame={processed_idx} t={t_sec:.3f}s "
                                f"roi={door_info['roi']} open_score={door_info['open_score']:.3f} "
                                f"state={door_info['state']}"
                            )
                        print(f"Door debug image written: {config.debug_door_png}")
                    break

                assert writer is not None
                assert stream is not None
                if active_segment is not None:
                    cv2.putText(
                        frame,
                        f"CONTACT SEGMENT start={active_segment.start_t:.2f}s",
                        (10, max(32, height - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    seg_frames = processed_idx - active_segment.start_frame + 1
                    bar_w = min(width - 20, seg_frames * 8)
                    cv2.rectangle(frame, (10, 8), (10 + bar_w, 16), (0, 0, 255), -1)
                writer.write(frame)

                line_payload: dict[str, Any] = {
                    "t": round(t_sec, 6),
                    "frame_idx": processed_idx,
                    "hands": output_hands,
                }
                if door_info is not None:
                    line_payload["door"] = door_info
                stream.write(json.dumps(line_payload) + "\n")
                if door_info is not None and door_stream is not None:
                    door_stream.write(
                        json.dumps(
                            {
                                "t": round(t_sec, 6),
                                "frame_idx": processed_idx,
                                "roi": door_info["roi"],
                                "open_score": door_info["open_score"],
                                "baseline_closed": door_info["baseline_closed"],
                                "delta": door_info["delta"],
                                "delta_smooth": door_info["delta_smooth"],
                                "state": door_info["state"],
                            }
                        )
                        + "\n"
                    )

                processed_idx += 1
                frame_idx += 1
    finally:
        if stream is not None:
            stream.close()
        if door_stream is not None:
            door_stream.close()
        cap.release()
        if writer is not None:
            writer.release()

    if active_segment is not None:
        _finalize_segment(event_segments, active_segment, config.min_event_len)
    if door_active_segment is not None:
        door_events.append(_door_segment_to_event(door_active_segment, idx=len(door_events) + 1))

    if debug_mode:
        return

    for idx, event in enumerate(event_segments, start=1):
        event["event_id"] = f"evt_contact_{idx:03d}"
    all_events = event_segments + door_events
    events_path.write_text(json.dumps(all_events, indent=2), encoding="utf-8")
    if config.w2d is not None:
        _update_w2d_payload(
            w2d_path=config.w2d,
            events=all_events,
            door_track_entries=door_track_entries,
            state_changes=state_changes,
        )

    total_contact_seconds = sum(max(0.0, float(e["t_end"]) - float(e["t_start"])) for e in event_segments)
    longest_contact_event_seconds = max(
        (max(0.0, float(e["t_end"]) - float(e["t_start"])) for e in event_segments),
        default=0.0,
    )

    time_in_each_state_seconds = {
        state: round(count / config.fps if config.fps > 0 else 0.0, 6)
        for state, count in state_frame_counts.items()
    }
    open_score_stats = {
        "min": round(float(min(door_scores)), 6) if door_scores else 0.0,
        "mean": round(float(np.mean(door_scores)), 6) if door_scores else 0.0,
        "max": round(float(max(door_scores)), 6) if door_scores else 0.0,
    }

    metrics = {
        "processed_frames": processed_idx,
        "hand_detect_rate": (frames_with_hands / processed_idx) if processed_idx else 0.0,
        "contact_frames_rate": (contact_frames / processed_idx) if processed_idx else 0.0,
        "longest_contact_run_frames": longest_contact_run,
        "longest_contact_run_seconds": (longest_contact_run / config.fps if config.fps > 0 else 0.0),
        "num_contact_events": len(event_segments),
        "total_contact_seconds": round(total_contact_seconds, 6),
        "longest_contact_event_seconds": round(longest_contact_event_seconds, 6),
        "num_state_changes": len(state_changes),
        "time_in_each_state_seconds": time_in_each_state_seconds,
        "open_score_stats": open_score_stats,
        "avg_hand_score": (hand_score_sum / hand_score_count) if hand_score_count else 0.0,
        "avg_contact_score_when_detected": (contact_score_sum / contact_score_count if contact_score_count else 0.0),
        "roi_scaled": False,
        "final_roi_used": list(config.handle_roi),
        "parameters_used": {
            "fps": config.fps,
            "min_score": config.min_score,
            "contact_thresh": config.contact_thresh,
            "roi": list(config.handle_roi),
            "max_hands": config.max_hands,
            "min_event_len": config.min_event_len,
            "merge_gap": config.merge_gap,
            "door_roi": list(config.door_roi) if config.door_roi else None,
            "door_baseline_seconds": config.door_baseline_seconds,
            "door_smooth_alpha": config.door_smooth_alpha,
            "door_ajar_on": config.door_ajar_on,
            "door_ajar_off": config.door_ajar_off,
            "door_open_on": config.door_open_on,
            "door_open_off": config.door_open_off,
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Overlay written: {overlay_path}")
    print(f"Frame stream JSONL written: {jsonl_path}")
    print(f"Door stream JSONL written: {door_stream_path}")
    print(f"Events written: {events_path}")
    if config.w2d is not None:
        print(f"World2Data updated in place: {config.w2d}")
    print(f"Metrics written: {metrics_path}")
