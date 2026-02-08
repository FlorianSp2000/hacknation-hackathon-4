from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Iterator

import numpy as np

from core.base import Detector, Segmenter, Tracker, VLM
from core.types import BBox, FrameResult, Mask, SegmentationResult, StateChange, TemporalChange
from core.video import read_frames
from pipeline.interactions import InteractionEngine


TEMPORAL_DIFF_PROMPT = """Compare these two frames from a video. Frame A (earlier) and Frame B (later).
Focus ONLY on navigation-relevant objects: doors, drawers, handles, cabinets, passages, and obstacles.
What navigation-relevant state changes occurred between these frames?
Output ONLY valid JSON, no other text:
{"state_changes": [{"object": "str", "type": "door|drawer|handle|cabinet|passage|obstacle", "before": "open|closed|ajar|blocked|clear", "after": "open|closed|ajar|blocked|clear", "confidence": "high|medium|low"}], "actions": ["str"]}"""


@dataclass
class _TrackState:
    track_id: int
    class_name: str
    bbox: tuple[float, float, float, float]
    last_seen_frame: int
    last_refresh_frame: int
    refresh_bbox: tuple[float, float, float, float] | None = None
    last_mask: np.ndarray | None = None


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _pick_best_mask(masks: list[Mask]) -> np.ndarray | None:
    best_mask = None
    best_score = -1.0
    for m in masks:
        area = float(np.count_nonzero(m.binary_mask))
        score = area * float(m.score)
        if score > best_score:
            best_score = score
            best_mask = m.binary_mask
    return best_mask


class Pipeline:
    def __init__(
        self,
        detector: Detector | None = None,
        segmenter: Segmenter | None = None,
        vlm: VLM | None = None,
        tracker: Tracker | None = None,
        vlm_prompt: str = "",
        seg_prompt: str = "",
        frame_skip: int = 1,
        vlm_skip: int = 5,
        sequential_offload: bool = False,
        interaction_engine: InteractionEngine | None = None,
        segment_on_demand: bool = True,
        seg_refresh_interval: int = 6,
        seg_match_iou: float = 0.5,
        seg_resegment_iou: float = 0.7,
        seg_min_box_area: int = 64,
        enable_temporal_diff: bool = False,
    ):
        self.detector = detector
        self.segmenter = segmenter
        self.vlm = vlm
        self.tracker = tracker
        self.vlm_prompt = vlm_prompt
        self.seg_prompt = seg_prompt
        self.frame_skip = frame_skip
        self.vlm_skip = vlm_skip
        self.sequential_offload = sequential_offload
        self.interaction_engine = interaction_engine
        self.segment_on_demand = segment_on_demand
        self.seg_refresh_interval = max(1, seg_refresh_interval)
        self.seg_match_iou = seg_match_iou
        self.seg_resegment_iou = seg_resegment_iou
        self.seg_min_box_area = max(1, seg_min_box_area)
        self.enable_temporal_diff = enable_temporal_diff
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 1

    def _run_detector(self, frame, idx):
        if self.sequential_offload:
            self.detector.load()
        result = self.detector.predict(frame)
        result.frame_idx = idx
        if self.sequential_offload:
            self.detector.unload()
        return result

    def _run_segmenter(self, frame, idx):
        if self.sequential_offload:
            self.segmenter.load()
        result = self.segmenter.predict(frame, text_prompt=self.seg_prompt or None)
        result.frame_idx = idx
        if self.sequential_offload:
            self.segmenter.unload()
        return result

    def _run_vlm(self, frame, idx):
        assert self.vlm_prompt, "VLM requires a prompt"
        if self.sequential_offload:
            self.vlm.load()
        result = self.vlm.predict(frame, self.vlm_prompt)
        result.frame_idx = idx
        if self.sequential_offload:
            self.vlm.unload()
        return result

    def _run_tracker(self, frame, idx):
        result = self.tracker.update(frame)
        result.frame_idx = idx
        return result

    def _run_temporal_diff(
        self, prev_frame: np.ndarray, prev_idx: int, curr_frame: np.ndarray, curr_idx: int
    ) -> TemporalChange:
        h1, w1 = prev_frame.shape[:2]
        h2, w2 = curr_frame.shape[:2]
        max_h = max(h1, h2)
        if h1 < max_h:
            pad = np.zeros((max_h - h1, w1, 3), dtype=prev_frame.dtype)
            prev_padded = np.vstack([prev_frame, pad])
        else:
            prev_padded = prev_frame
        if h2 < max_h:
            pad = np.zeros((max_h - h2, w2, 3), dtype=curr_frame.dtype)
            curr_padded = np.vstack([curr_frame, pad])
        else:
            curr_padded = curr_frame

        combined = np.hstack([prev_padded, curr_padded])
        prompt = (
            f"This image shows two video frames side by side. LEFT = Frame A (frame {prev_idx}, earlier). "
            f"RIGHT = Frame B (frame {curr_idx}, later).\n{TEMPORAL_DIFF_PROMPT}"
        )

        if self.sequential_offload:
            self.vlm.load()
        vlm_result = self.vlm.predict(combined, prompt)
        if self.sequential_offload:
            self.vlm.unload()

        state_changes: list[StateChange] = []
        actions: list[str] = []
        raw_text = vlm_result.raw_text
        parsed = vlm_result.parsed
        if parsed is None:
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError:
                if "```" in raw_text:
                    json_str = raw_text.split("```")[1]
                    if json_str.startswith("json"):
                        json_str = json_str[4:]
                    try:
                        parsed = json.loads(json_str.strip())
                    except json.JSONDecodeError:
                        pass

        if parsed and isinstance(parsed, dict):
            for sc in parsed.get("state_changes", []):
                if isinstance(sc, dict):
                    state_changes.append(
                        StateChange(
                            object_name=str(sc.get("object", "unknown")),
                            before_state=str(sc.get("before", "unknown")),
                            after_state=str(sc.get("after", "unknown")),
                            confidence=str(sc.get("confidence", "medium")),
                        )
                    )
            actions = [str(a) for a in parsed.get("actions", [])]

        return TemporalChange(
            state_changes=state_changes,
            actions_detected=actions,
            frame_idx_before=prev_idx,
            frame_idx_after=curr_idx,
            raw_text=raw_text,
        )

    def _update_tracks(self, boxes: list[BBox], frame_idx: int) -> list[int]:
        det_track_ids = [-1] * len(boxes)
        used_tracks: set[int] = set()
        det_order = sorted(range(len(boxes)), key=lambda i: float(boxes[i].confidence), reverse=True)

        for det_idx in det_order:
            box = boxes[det_idx]
            det_bbox = (float(box.x1), float(box.y1), float(box.x2), float(box.y2))
            best_tid = None
            best_iou = 0.0
            for tid, track in self._tracks.items():
                if tid in used_tracks or track.class_name != box.class_name:
                    continue
                iou = _bbox_iou(det_bbox, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None and best_iou >= self.seg_match_iou:
                track = self._tracks[best_tid]
                track.bbox = det_bbox
                track.last_seen_frame = frame_idx
                det_track_ids[det_idx] = best_tid
                used_tracks.add(best_tid)
            else:
                tid = self._next_track_id
                self._next_track_id += 1
                self._tracks[tid] = _TrackState(
                    track_id=tid,
                    class_name=box.class_name,
                    bbox=det_bbox,
                    last_seen_frame=frame_idx,
                    last_refresh_frame=-10_000,
                    refresh_bbox=None,
                    last_mask=None,
                )
                det_track_ids[det_idx] = tid
                used_tracks.add(tid)

        stale = [tid for tid, tr in self._tracks.items() if frame_idx - tr.last_seen_frame > 60]
        for tid in stale:
            self._tracks.pop(tid, None)

        return det_track_ids

    def _run_segmenter_on_detections(
        self, frame: np.ndarray, idx: int, boxes: list[BBox], det_track_ids: list[int]
    ) -> SegmentationResult:
        if self.sequential_offload:
            self.segmenter.load()
        h, w = frame.shape[:2]
        masks: list[Mask] = []

        try:
            for det_idx, box in enumerate(boxes):
                tid = det_track_ids[det_idx]
                track = self._tracks.get(tid)
                if track is None:
                    continue

                x1 = max(0, int(np.floor(box.x1)))
                y1 = max(0, int(np.floor(box.y1)))
                x2 = min(w, int(np.ceil(box.x2)))
                y2 = min(h, int(np.ceil(box.y2)))
                if x2 <= x1 or y2 <= y1:
                    continue
                area = (x2 - x1) * (y2 - y1)
                if area < self.seg_min_box_area:
                    continue

                current_bbox = (float(box.x1), float(box.y1), float(box.x2), float(box.y2))
                iou_to_refresh = (
                    _bbox_iou(current_bbox, track.refresh_bbox) if track.refresh_bbox is not None else 0.0
                )
                should_refresh = (
                    track.last_mask is None
                    or (idx - track.last_refresh_frame) >= self.seg_refresh_interval
                    or iou_to_refresh < self.seg_resegment_iou
                )

                if should_refresh:
                    crop = frame[y1:y2, x1:x2]
                    seg_crop = self.segmenter.predict(crop, text_prompt=(self.seg_prompt or box.class_name))
                    local_mask = _pick_best_mask(seg_crop.masks)
                    if local_mask is not None:
                        full_mask = np.zeros((h, w), dtype=bool)
                        full_mask[y1:y2, x1:x2] = local_mask
                        track.last_mask = full_mask
                        track.last_refresh_frame = idx
                        track.refresh_bbox = current_bbox

                if track.last_mask is not None:
                    masks.append(
                        Mask(
                            binary_mask=track.last_mask.copy(),
                            score=float(box.confidence),
                            label=f"{box.class_name}#{tid}",
                        )
                    )
        finally:
            if self.sequential_offload:
                self.segmenter.unload()

        return SegmentationResult(masks=masks, frame_idx=idx)

    def run(self, video_path: str, max_frames: int | None = None) -> Iterator[FrameResult]:
        if self.segmenter and hasattr(self.segmenter, "reset_session"):
            self.segmenter.reset_session()
        if self.tracker:
            self.tracker.reset()
            if self.sequential_offload:
                self.tracker.load()

        prev_vlm_frame = None
        prev_vlm_idx = None
        executor = None if self.sequential_offload else ThreadPoolExecutor(max_workers=4)

        try:
            for idx, frame in read_frames(video_path, max_frames):
                if idx % self.frame_skip != 0:
                    continue

                frame_start = perf_counter()
                result = FrameResult(frame_idx=idx, frame=frame)
                guided_seg = bool(
                    self.segment_on_demand and self.segmenter is not None and self.detector is not None
                )

                if self.sequential_offload:
                    if self.detector:
                        t0 = perf_counter()
                        result.detection = self._run_detector(frame, idx)
                        result.timings["detector"] = perf_counter() - t0

                    if self.tracker:
                        t0 = perf_counter()
                        result.tracking = self._run_tracker(frame, idx)
                        result.timings["tracker"] = perf_counter() - t0

                    if self.segmenter:
                        t0 = perf_counter()
                        if guided_seg and result.detection is not None:
                            det_track_ids = self._update_tracks(result.detection.boxes, idx)
                            result.segmentation = self._run_segmenter_on_detections(
                                frame, idx, result.detection.boxes, det_track_ids
                            )
                        else:
                            result.segmentation = self._run_segmenter(frame, idx)
                        result.timings["segmenter"] = perf_counter() - t0

                    if self.vlm and idx % self.vlm_skip == 0:
                        t0 = perf_counter()
                        result.vlm = self._run_vlm(frame, idx)
                        result.timings["vlm"] = perf_counter() - t0
                        if self.enable_temporal_diff and prev_vlm_frame is not None:
                            result.temporal_changes = self._run_temporal_diff(
                                prev_vlm_frame, prev_vlm_idx, frame, idx
                            )
                        prev_vlm_frame = frame.copy()
                        prev_vlm_idx = idx
                else:
                    detection_future = None
                    vlm_future = None
                    seg_future = None

                    if self.detector:
                        t0 = perf_counter()
                        detection_future = (executor.submit(self._run_detector, frame, idx), t0)

                    if self.segmenter and not guided_seg:
                        t0 = perf_counter()
                        seg_future = (executor.submit(self._run_segmenter, frame, idx), t0)

                    if self.vlm and idx % self.vlm_skip == 0:
                        t0 = perf_counter()
                        vlm_future = (executor.submit(self._run_vlm, frame, idx), t0)

                    if detection_future is not None:
                        fut, t0 = detection_future
                        result.detection = fut.result()
                        result.timings["detector"] = perf_counter() - t0

                    if self.tracker:
                        t0 = perf_counter()
                        result.tracking = self._run_tracker(frame, idx)
                        result.timings["tracker"] = perf_counter() - t0

                    if self.segmenter:
                        if guided_seg and result.detection is not None:
                            t0 = perf_counter()
                            det_track_ids = self._update_tracks(result.detection.boxes, idx)
                            result.segmentation = self._run_segmenter_on_detections(
                                frame, idx, result.detection.boxes, det_track_ids
                            )
                            result.timings["segmenter"] = perf_counter() - t0
                        elif seg_future is not None:
                            fut, t0 = seg_future
                            result.segmentation = fut.result()
                            result.timings["segmenter"] = perf_counter() - t0

                    if vlm_future is not None:
                        fut, t0 = vlm_future
                        result.vlm = fut.result()
                        result.timings["vlm"] = perf_counter() - t0
                        if self.enable_temporal_diff and prev_vlm_frame is not None:
                            t1 = perf_counter()
                            result.temporal_changes = self._run_temporal_diff(
                                prev_vlm_frame, prev_vlm_idx, frame, idx
                            )
                            result.timings["temporal_diff"] = perf_counter() - t1
                        prev_vlm_frame = frame.copy()
                        prev_vlm_idx = idx

                if self.interaction_engine:
                    t0 = perf_counter()
                    hands, interactions = self.interaction_engine.process(frame, result.detection)
                    result.hand_poses = hands
                    result.interactions = interactions
                    result.timings["interactions"] = perf_counter() - t0

                result.timings["frame_total"] = perf_counter() - frame_start
                yield result
        finally:
            if executor:
                executor.shutdown(wait=False)
            if self.interaction_engine:
                self.interaction_engine.close()
            if self.tracker and self.sequential_offload:
                self.tracker.unload()
