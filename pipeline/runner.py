from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator
from time import perf_counter

import numpy as np

from core.base import Detector, Segmenter, VLM, Tracker, StateClassifier
from core.types import FrameResult, StateChange, TemporalChange, StateLabel, StateClassificationResult
from core.video import read_frames
from pipeline.interactions import InteractionEngine


TEMPORAL_DIFF_PROMPT = """Compare these two frames from a video. Frame A (earlier) and Frame B (later).
Focus ONLY on navigation-relevant objects: doors, drawers, handles, cabinets, passages, and obstacles.
What navigation-relevant state changes occurred between these frames?
Output ONLY valid JSON, no other text:
{"state_changes": [{"object": "str", "type": "door|drawer|handle|cabinet|passage|obstacle", "before": "open|closed|ajar|blocked|clear", "after": "open|closed|ajar|blocked|clear", "confidence": "high|medium|low"}], "actions": ["str"]}"""


class Pipeline:
    def __init__(
        self,
        detector: Detector | None = None,
        segmenter: Segmenter | None = None,
        vlm: VLM | None = None,
        tracker: Tracker | None = None,
        state_classifier: StateClassifier | None = None,
        vlm_prompt: str = "",
        seg_prompt: str = "",
        frame_skip: int = 1,
        vlm_skip: int = 5,
        sequential_offload: bool = False,
        interaction_engine: InteractionEngine | None = None,
        enable_temporal_diff: bool = False,
    ):
        self.detector = detector
        self.segmenter = segmenter
        self.vlm = vlm
        self.tracker = tracker
        self.state_classifier = state_classifier
        self.vlm_prompt = vlm_prompt
        self.seg_prompt = seg_prompt
        self.frame_skip = frame_skip
        self.vlm_skip = vlm_skip
        self.sequential_offload = sequential_offload
        self.interaction_engine = interaction_engine
        self.enable_temporal_diff = enable_temporal_diff

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
        # Tracker is stateful — no sequential offload (must stay loaded)
        result = self.tracker.update(frame)
        result.frame_idx = idx
        return result

    def _run_state_classifier(self, frame, tracking_result, idx):
        """Run state classifier on tracked bbox crops."""
        if not tracking_result or not tracking_result.boxes:
            return StateClassificationResult(labels=[], frame_idx=idx)

        if self.sequential_offload:
            self.state_classifier.load()

        boxes = [
            (b.x1, b.y1, b.x2, b.y2, b.track_id)
            for b in tracking_result.boxes
        ]
        classifications = self.state_classifier.classify(frame, boxes)

        labels = []
        for i, (state, conf) in enumerate(classifications):
            b = tracking_result.boxes[i]
            labels.append(StateLabel(
                track_id=b.track_id,
                state=state,
                confidence=conf,
                bbox=(b.x1, b.y1, b.x2, b.y2),
            ))

        if self.sequential_offload:
            self.state_classifier.unload()

        return StateClassificationResult(labels=labels, frame_idx=idx)

    def _run_temporal_diff(self, prev_frame: np.ndarray, prev_idx: int,
                           curr_frame: np.ndarray, curr_idx: int) -> TemporalChange:
        """Run VLM on a pair of frames to detect state changes."""

        # Build a side-by-side comparison
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
            f"This image shows two video frames side by side. "
            f"LEFT = Frame A (frame {prev_idx}, earlier). RIGHT = Frame B (frame {curr_idx}, later).\n"
            + TEMPORAL_DIFF_PROMPT
        )

        if self.sequential_offload:
            self.vlm.load()
        vlm_result = self.vlm.predict(combined, prompt)
        if self.sequential_offload:
            self.vlm.unload()

        # Parse temporal changes from VLM output
        state_changes = []
        actions = []
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
                    state_changes.append(StateChange(
                        object_name=str(sc.get("object", "unknown")),
                        before_state=str(sc.get("before", "unknown")),
                        after_state=str(sc.get("after", "unknown")),
                        confidence=str(sc.get("confidence", "medium")),
                    ))
            actions = [str(a) for a in parsed.get("actions", [])]

        return TemporalChange(
            state_changes=state_changes,
            actions_detected=actions,
            frame_idx_before=prev_idx,
            frame_idx_after=curr_idx,
            raw_text=raw_text,
        )

    def run(self, video_path: str, max_frames: int | None = None) -> Iterator[FrameResult]:
        # Reset streaming session for new video (if segmenter supports it)
        if self.segmenter and hasattr(self.segmenter, "reset_session"):
            self.segmenter.reset_session()

        # Reset tracker state for new video
        if self.tracker:
            self.tracker.reset()

        # State for temporal differencing
        prev_vlm_frame = None
        prev_vlm_idx = None

        executor = None if self.sequential_offload else ThreadPoolExecutor(max_workers=4)

        try:
            for idx, frame in read_frames(video_path, max_frames):
                if idx % self.frame_skip != 0:
                    continue

                frame_start = perf_counter()
                result = FrameResult(frame_idx=idx, frame=frame)

                if self.sequential_offload:
                    if self.detector:
                        t0 = perf_counter()
                        result.detection = self._run_detector(frame, idx)
                        result.timings["detector"] = perf_counter() - t0
                    if self.tracker:
                        t0 = perf_counter()
                        result.tracking = self._run_tracker(frame, idx)
                        result.timings["tracker"] = perf_counter() - t0
                    if self.state_classifier and result.tracking:
                        t0 = perf_counter()
                        result.state_classification = self._run_state_classifier(
                            frame, result.tracking, idx)
                        result.timings["state_classifier"] = perf_counter() - t0
                    if self.segmenter:
                        t0 = perf_counter()
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
                    # Parallel: run independent models concurrently
                    futures = {}
                    if self.detector:
                        t0 = perf_counter()
                        futures["detection"] = (
                            executor.submit(self._run_detector, frame, idx),
                            t0,
                        )
                    if self.segmenter:
                        t0 = perf_counter()
                        futures["segmentation"] = (
                            executor.submit(self._run_segmenter, frame, idx),
                            t0,
                        )

                    # Tracker must run sequentially (stateful)
                    if self.tracker:
                        t0 = perf_counter()
                        result.tracking = self._run_tracker(frame, idx)
                        result.timings["tracker"] = perf_counter() - t0

                    # State classifier runs on tracked bboxes (fast, every frame)
                    if self.state_classifier and result.tracking:
                        t0 = perf_counter()
                        result.state_classification = self._run_state_classifier(
                            frame, result.tracking, idx)
                        result.timings["state_classifier"] = perf_counter() - t0

                    # VLM also sequential (GPU-bound + temporal diff needs ordering)
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

                    for key, (future, t0) in futures.items():
                        setattr(result, key, future.result())
                        result.timings[key] = perf_counter() - t0

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
