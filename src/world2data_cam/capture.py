"""Webcam capture and recording utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2


@dataclass(slots=True)
class RecordConfig:
    """Settings for webcam preview and recording."""

    camera: int
    out: Path
    fps: int
    width: int
    height: int


def _open_camera(camera_index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    """Open a webcam with best-effort macOS backend selection."""
    backend = getattr(cv2, "CAP_AVFOUNDATION", 0)
    cap = cv2.VideoCapture(camera_index, backend)

    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open camera index "
            f"{camera_index}. Try --camera 0, --camera 1, or --camera 2 "
            "and confirm camera permissions for your terminal app."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))
    return cap


def _create_video_writer(out_path: Path, fps: int, size: Tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
    """Create an MP4 writer using mp4v codec."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mp4_path = out_path if out_path.suffix.lower() == ".mp4" else out_path.with_suffix(".mp4")
    mp4_writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size)
    if mp4_writer.isOpened():
        return mp4_writer, mp4_path

    mp4_writer.release()
    raise RuntimeError(
        "Failed to initialize MP4 writer with mp4v codec. "
        "Try a different output path or camera settings."
    )


def _draw_overlay(frame) -> None:
    """Draw timestamp onto the frame in-place."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        frame,
        timestamp,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_rec_indicator(frame) -> None:
    """Draw REC indicator onto the frame in-place."""
    cv2.circle(frame, (22, 58), 8, (0, 0, 255), -1)
    cv2.putText(
        frame,
        "REC",
        (38, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def record_webcam(config: RecordConfig) -> Optional[Path]:
    """Run live webcam preview and optionally record to disk.

    Controls:
    - r: start/stop recording
    - q: quit

    Returns the final recording path if a recording was created.
    """
    cap = _open_camera(config.camera, config.width, config.height, config.fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or config.width
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.height
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS)) or config.fps
    size = (actual_width, actual_height)

    print(
        f"Preview started: camera={config.camera}, "
        f"resolution={actual_width}x{actual_height}, fps={actual_fps}"
    )
    print("Controls: press 'r' to start/stop recording, 'q' to quit")

    writer: Optional[cv2.VideoWriter] = None
    output_path: Optional[Path] = None
    recording = False

    window_name = "world2data-cam"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read frame from camera.")
                continue

            _draw_overlay(frame)

            if recording and writer is not None:
                _draw_rec_indicator(frame)
                writer.write(frame)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Quitting preview.")
                break

            if key == ord("r"):
                if not recording:
                    try:
                        writer, output_path = _create_video_writer(config.out, config.fps, size)
                    except RuntimeError as exc:
                        print(f"Error: {exc}")
                        continue

                    recording = True
                    print(f"Recording started: {output_path}")
                else:
                    recording = False
                    if writer is not None:
                        writer.release()
                        writer = None
                    print(f"Recording stopped. Saved: {output_path}")
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()

    return output_path
