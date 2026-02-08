"""Frame extraction and World2Data stub export utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2


DOOR_MVP_ONTOLOGY: dict[str, Any] = {
    "objects": ["person", "door", "handle", "doorway"],
    "states": [
        {
            "object": "door",
            "key": "door.state",
            "values": ["closed", "ajar", "open"],
        }
    ],
    "affordances": ["openable", "graspable", "traversable"],
    "actions": ["approach", "reach", "grasp", "pull", "push", "open", "pass_through"],
}


def default_door_ontology() -> dict[str, Any]:
    """Return a copy-safe default ontology for door-opening MVP."""
    return json.loads(json.dumps(DOOR_MVP_ONTOLOGY))


def ensure_ontology(payload: dict[str, Any]) -> dict[str, Any]:
    """Ensure ontology exists and has the required MVP keys."""
    ontology = payload.get("ontology")
    if not isinstance(ontology, dict):
        payload["ontology"] = default_door_ontology()
        return payload

    defaults = default_door_ontology()
    for key in ("objects", "states", "affordances", "actions"):
        value = ontology.get(key)
        if not isinstance(value, list) or len(value) == 0:
            ontology[key] = defaults[key]
    payload["ontology"] = ontology
    return payload


def extract_frames(video_path: Path, out_dir: Path, target_fps: float) -> Path:
    """Extract JPG frames at target fps and write metadata.json.

    Returns the metadata path.
    """
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if source_fps <= 0:
        print("Warning: source fps unavailable; extracting every frame.")
        source_fps = target_fps

    next_emit_time = 0.0
    emit_interval = 1.0 / target_fps

    frame_idx = 0
    written_count = 0
    frame_files: list[str] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_sec = frame_idx / source_fps
            if timestamp_sec + 1e-9 >= next_emit_time:
                written_count += 1
                filename = f"frame_{written_count:06d}.jpg"
                out_path = out_dir / filename
                cv2.imwrite(str(out_path), frame)
                frame_files.append(filename)
                next_emit_time += emit_interval

            frame_idx += 1
    finally:
        cap.release()

    metadata = {
        "video_path": str(video_path),
        "extracted_fps": target_fps,
        "frame_count": written_count,
        "width": width,
        "height": height,
        "frames": frame_files,
    }

    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        f"Extracted {written_count} frame(s) from {video_path} "
        f"at target {target_fps} fps into {out_dir}"
    )
    print(f"Metadata written: {metadata_path}")

    return metadata_path


def _load_metadata_if_available(frames_dir: Path) -> dict[str, Any]:
    """Load frame metadata when present, otherwise return defaults."""
    metadata_path = frames_dir / "metadata.json"
    if not metadata_path.exists():
        return {"extracted_fps": None, "frames": []}

    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid metadata format in {metadata_path}")

    return data


def create_world2data_stub(frames_dir: Path, out_path: Path) -> Path:
    """Create a World2Data-compatible JSON stub from extracted frames."""
    if not frames_dir.exists() or not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    metadata = _load_metadata_if_available(frames_dir)
    extracted_fps = metadata.get("extracted_fps")

    jpg_files = sorted(path.name for path in frames_dir.glob("*.jpg"))
    if isinstance(metadata.get("frames"), list) and metadata["frames"]:
        frame_files = [str(name) for name in metadata["frames"]]
    else:
        frame_files = jpg_files

    stub = {
        "video_id": frames_dir.name,
        "fps": extracted_fps,
        "ontology": default_door_ontology(),
        "tracks": [],
        "events": [],
        "state_changes": [],
        "frames": {
            "directory": str(frames_dir),
            "metadata": str(frames_dir / "metadata.json"),
            "files": frame_files,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stub, indent=2), encoding="utf-8")
    print(f"World2Data stub written: {out_path}")
    return out_path
