"""CLI entrypoint for world2data_cam."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .capture import RecordConfig, record_webcam
from .detect import DetectConfig, run_detection_and_tracking
from .export import create_world2data_stub, extract_frames
from .hands_contact import HandsContactConfig, parse_roi, run_hands_contact


def _build_parser() -> argparse.ArgumentParser:
    """Create CLI parser and subcommands."""
    parser = argparse.ArgumentParser(prog="world2data_cam")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record", help="Preview webcam and record video")
    record_parser.add_argument("--camera", type=int, default=0, help="Camera index (0/1/2)")
    record_parser.add_argument("--out", type=Path, required=True, help="Output video path")
    record_parser.add_argument("--fps", type=int, default=30, help="Target frames per second")
    record_parser.add_argument("--width", type=int, default=1280, help="Capture width")
    record_parser.add_argument("--height", type=int, default=720, help="Capture height")

    frames_parser = subparsers.add_parser("frames", help="Extract JPG frames from a video")
    frames_parser.add_argument("--video", type=Path, required=True, help="Input video path")
    frames_parser.add_argument("--out", type=Path, required=True, help="Output frame directory")
    frames_parser.add_argument("--fps", type=float, default=10.0, help="Extraction fps")

    detect_parser = subparsers.add_parser("detect", help="Run YOLO detection + tracking on a video")
    detect_parser.add_argument("--video", type=Path, required=True, help="Input video path")
    detect_parser.add_argument("--out", type=Path, required=True, help="Output directory")
    detect_parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path/name")
    detect_parser.add_argument("--fps", type=float, default=10.0, help="Processing/output fps")
    detect_parser.add_argument(
        "--w2d",
        type=Path,
        required=True,
        help="Path to existing World2Data JSON (updated in place)",
    )

    stub_parser = subparsers.add_parser("stub", help="Create World2Data JSON stub")
    stub_parser.add_argument("--frames", type=Path, required=True, help="Extracted frames directory")
    stub_parser.add_argument("--out", type=Path, required=True, help="Output stub JSON path")

    hands_parser = subparsers.add_parser(
        "hands_contact",
        help="Run MediaPipe hand detection and compute contact score vs ROI",
    )
    hands_parser.add_argument("--video", type=Path, required=True, help="Input video path")
    hands_parser.add_argument("--out", type=Path, required=True, help="Output directory")
    hands_parser.add_argument("--fps", type=float, default=10.0, help="Processing/output fps")
    hands_parser.add_argument(
        "--handle-roi",
        type=str,
        required=True,
        help="Handle ROI as x1,y1,x2,y2 in pixel coordinates",
    )
    hands_parser.add_argument("--max-hands", type=int, default=2, help="Maximum hands per frame")
    hands_parser.add_argument("--min-score", type=float, default=0.5, help="Min hand confidence")
    hands_parser.add_argument(
        "--contact-thresh",
        type=float,
        default=0.15,
        help="Contact threshold as fraction of keypoints in ROI",
    )
    hands_parser.add_argument(
        "--w2d",
        type=Path,
        default=None,
        help="Optional World2Data JSON path to update in place with contact events",
    )
    hands_parser.add_argument(
        "--min-event-len",
        type=int,
        default=3,
        help="Minimum contact event length in frames",
    )
    hands_parser.add_argument(
        "--merge-gap",
        type=int,
        default=2,
        help="Merge tiny contact gaps up to this many frames",
    )
    hands_parser.add_argument(
        "--frame-idx",
        type=int,
        default=None,
        help="Debug: sampled frame index to inspect (uses fps-downsample indexing)",
    )
    hands_parser.add_argument(
        "--debug-png",
        type=Path,
        default=None,
        help="Debug: output PNG path for the selected --frame-idx",
    )
    hands_parser.add_argument(
        "--door-roi",
        type=str,
        default=None,
        help="Door ROI as x1,y1,x2,y2 in pixel coordinates",
    )
    hands_parser.add_argument(
        "--door-baseline-seconds",
        type=float,
        default=2.0,
        help="Seconds used to estimate closed-door baseline from early open_score values",
    )
    hands_parser.add_argument(
        "--door-smooth-alpha",
        type=float,
        default=0.2,
        help="EMA alpha for smoothing door delta",
    )
    hands_parser.add_argument(
        "--door-ajar-on",
        type=float,
        default=0.18,
        help="Hysteresis ON threshold for closed -> ajar (delta_smooth)",
    )
    hands_parser.add_argument(
        "--door-ajar-off",
        type=float,
        default=0.12,
        help="Hysteresis OFF threshold for ajar -> closed (delta_smooth)",
    )
    hands_parser.add_argument(
        "--door-open-on",
        type=float,
        default=0.35,
        help="Hysteresis ON threshold for ajar -> open (delta_smooth)",
    )
    hands_parser.add_argument(
        "--door-open-off",
        type=float,
        default=0.28,
        help="Hysteresis OFF threshold for open -> ajar (delta_smooth)",
    )
    hands_parser.add_argument(
        "--debug-door-frame",
        type=int,
        default=None,
        help="Debug: sampled frame index to inspect door score/state",
    )
    hands_parser.add_argument(
        "--debug-door-png",
        type=Path,
        default=None,
        help="Debug: output PNG path for --debug-door-frame",
    )

    return parser


def main() -> None:
    """Run the world2data_cam CLI."""
    if sys.version_info < (3, 11):
        raise RuntimeError("Python 3.11+ is required.")

    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "record":
        config = RecordConfig(
            camera=args.camera,
            out=args.out,
            fps=args.fps,
            width=args.width,
            height=args.height,
        )
        result = record_webcam(config)
        if result is None:
            print("No recording file created.")
        return

    if args.command == "frames":
        extract_frames(video_path=args.video, out_dir=args.out, target_fps=args.fps)
        return

    if args.command == "detect":
        config = DetectConfig(
            video=args.video,
            out_dir=args.out,
            model=args.model,
            fps=args.fps,
            w2d_path=args.w2d,
        )
        run_detection_and_tracking(config)
        return

    if args.command == "stub":
        create_world2data_stub(frames_dir=args.frames, out_path=args.out)
        return

    if args.command == "hands_contact":
        config = HandsContactConfig(
            video=args.video,
            out_dir=args.out,
            fps=args.fps,
            handle_roi=parse_roi(args.handle_roi),
            max_hands=args.max_hands,
            min_score=args.min_score,
            contact_thresh=args.contact_thresh,
            w2d=args.w2d,
            min_event_len=args.min_event_len,
            merge_gap=args.merge_gap,
            frame_idx=args.frame_idx,
            debug_png=args.debug_png,
            door_roi=parse_roi(args.door_roi) if args.door_roi else None,
            door_baseline_seconds=args.door_baseline_seconds,
            door_smooth_alpha=args.door_smooth_alpha,
            door_ajar_on=args.door_ajar_on,
            door_ajar_off=args.door_ajar_off,
            door_open_on=args.door_open_on,
            door_open_off=args.door_open_off,
            debug_door_frame=args.debug_door_frame,
            debug_door_png=args.debug_door_png,
        )
        run_hands_contact(config)
        return


if __name__ == "__main__":
    main()
