from __future__ import annotations

import shutil
import subprocess
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


def _reencode_h264(input_path: Path, output_path: Path) -> None:
    """Re-encode mp4v file to H.264 via ffmpeg subprocess."""
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None, "ffmpeg not found — install ffmpeg for H.264 export"
    result = subprocess.run(
        [ffmpeg, "-y", "-i", str(input_path),
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(output_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"ffmpeg failed: {result.stderr[:500]}"


def export_annotated_video(
    results: list[FrameResult],
    output_path: str | Path,
    fps: float,
    draw_fn: Callable[[np.ndarray, FrameResult], np.ndarray],
) -> Path:
    """Write annotated video. Outputs H.264 mp4 (ffmpeg re-encode from mp4v)."""
    assert len(results) > 0, "No results to export"
    output_path = Path(output_path)

    h, w = results[0].frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Write raw mp4v to temp file, then re-encode to H.264
    raw_path = output_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (w, h))
    assert writer.isOpened(), f"Cannot create video writer: {raw_path}"

    try:
        for r in results:
            annotated = draw_fn(r.frame, r)
            writer.write(annotated)
    finally:
        writer.release()

    _reencode_h264(raw_path, output_path)
    raw_path.unlink(missing_ok=True)

    return output_path
