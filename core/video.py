from __future__ import annotations

from pathlib import Path
from typing import Iterator, Callable

import cv2
import numpy as np

from core.types import FrameResult


def read_frames(video_path: str | Path, max_frames: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yields (frame_idx, bgr_frame). Fails hard on bad file."""
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Cannot open video: {video_path}"
    idx = 0
    try:
        while True:
            if max_frames is not None and idx >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def get_video_info(video_path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Cannot open video: {video_path}"
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def export_annotated_video(
    results: list[FrameResult],
    output_path: str | Path,
    fps: float,
    draw_fn: Callable[[np.ndarray, FrameResult], np.ndarray],
) -> Path:
    """Write annotated video. draw_fn overlays results onto each frame."""
    assert len(results) > 0, "No results to export"
    output_path = Path(output_path)

    h, w = results[0].frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    assert writer.isOpened(), f"Cannot create video writer: {output_path}"

    try:
        for r in results:
            annotated = draw_fn(r.frame, r)
            writer.write(annotated)
    finally:
        writer.release()

    return output_path
