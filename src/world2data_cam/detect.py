"""Video object detection + tracking utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from .export import ensure_ontology


@dataclass(slots=True)
class DetectConfig:
    """Settings for detection/tracking export."""

    video: Path
    out_dir: Path
    model: str
    fps: float
    w2d_path: Path


def _open_video(video_path: Path) -> cv2.VideoCapture:
    """Open input video with basic validation."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    return cap


def _create_writer(out_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Create MP4 writer for overlay output."""
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create overlay writer at: {out_path}")
    return writer


def _xyxy_from_box(values: Any) -> list[float]:
    """Convert tensor-like xyxy values to python floats."""
    return [float(values[0]), float(values[1]), float(values[2]), float(values[3])]


def _load_w2d_payload(w2d_path: Path) -> dict[str, Any]:
    """Load World2Data JSON payload from disk."""
    if not w2d_path.exists():
        raise FileNotFoundError(f"World2Data JSON not found: {w2d_path}")

    with w2d_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid World2Data JSON format: {w2d_path}")
    return payload


def run_detection_and_tracking(config: DetectConfig) -> tuple[Path, Path]:
    """Run YOLO detection+tracking and update World2Data JSON in place."""
    if config.fps <= 0:
        raise ValueError("--fps must be > 0")

    cap = _open_video(config.video)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if source_fps <= 0:
        source_fps = config.fps
        print("Warning: source FPS unavailable; processing all frames as they arrive.")

    config.out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = config.out_dir / "overlay.mp4"
    summary_path = config.out_dir / "summary.json"

    writer = _create_writer(overlay_path, config.fps, width, height)
    model = YOLO(config.model)

    frame_index = 0
    processed_index = 0
    next_emit_time = 0.0
    emit_interval = 1.0 / config.fps

    tracks_by_id: dict[int, dict[str, Any]] = {}
    class_counts: dict[str, int] = {}
    conf_sum = 0.0
    conf_count = 0

    print(
        f"Running detection+tracking on {config.video} "
        f"at target {config.fps:.2f} fps using model {config.model}"
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_sec = frame_index / source_fps
            if timestamp_sec + 1e-9 < next_emit_time:
                frame_index += 1
                continue
            next_emit_time += emit_interval

            results = model.track(frame, persist=True, verbose=False, device="cpu")
            result = results[0]
            boxes = result.boxes
            names = result.names

            if boxes is not None:
                for idx in range(len(boxes)):
                    xyxy = _xyxy_from_box(boxes.xyxy[idx].tolist())
                    conf = float(boxes.conf[idx].item()) if boxes.conf is not None else 0.0
                    cls_id = int(boxes.cls[idx].item()) if boxes.cls is not None else -1
                    cls_name = str(names.get(cls_id, str(cls_id))) if isinstance(names, dict) else str(cls_id)
                    track_id = (
                        int(boxes.id[idx].item())
                        if boxes.id is not None
                        else -(processed_index * 1000 + idx + 1)
                    )

                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    label = f"{cls_name} id={track_id} conf={conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    conf_sum += conf
                    conf_count += 1

                    if track_id not in tracks_by_id:
                        tracks_by_id[track_id] = {
                            "track_id": track_id,
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "detections": [],
                        }

                    tracks_by_id[track_id]["detections"].append(
                        {
                            "frame_index": processed_index,
                            "timestamp_sec": round(timestamp_sec, 6),
                            "bbox_xyxy": [round(v, 3) for v in xyxy],
                            "confidence": round(conf, 6),
                            "class_id": cls_id,
                            "class_name": cls_name,
                        }
                    )

            writer.write(frame)
            processed_index += 1
            frame_index += 1
    finally:
        writer.release()
        cap.release()

    w2d_payload = _load_w2d_payload(config.w2d_path)
    w2d_payload = ensure_ontology(w2d_payload)
    w2d_payload["video_id"] = str(w2d_payload.get("video_id") or config.video.stem)
    w2d_payload["fps"] = config.fps
    w2d_payload["tracks"] = list(tracks_by_id.values())
    w2d_payload["events"] = []
    w2d_payload["state_changes"] = []

    frames_payload = w2d_payload.get("frames")
    if not isinstance(frames_payload, dict):
        w2d_payload["frames"] = {
            "directory": "",
            "metadata": "",
            "files": [],
        }

    config.w2d_path.parent.mkdir(parents=True, exist_ok=True)
    config.w2d_path.write_text(json.dumps(w2d_payload, indent=2), encoding="utf-8")

    summary_payload = {
        "video_id": config.video.stem,
        "model": config.model,
        "source_video": str(config.video),
        "world2data_json": str(config.w2d_path),
        "overlay_video": str(overlay_path),
        "target_fps": config.fps,
        "processed_frames": processed_index,
        "counts_per_class": class_counts,
        "num_tracks": len(tracks_by_id),
        "avg_confidence": round(conf_sum / conf_count, 6) if conf_count else 0.0,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Overlay video written: {overlay_path}")
    print(f"Updated World2Data JSON in place: {config.w2d_path}")
    print(f"Summary JSON written: {summary_path}")
    return overlay_path, summary_path
