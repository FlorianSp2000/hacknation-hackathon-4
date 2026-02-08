from __future__ import annotations

from typing import Iterator

from core.base import Detector, Segmenter, VLM
from core.types import FrameResult
from core.video import read_frames


class Pipeline:
    def __init__(
        self,
        detector: Detector | None = None,
        segmenter: Segmenter | None = None,
        vlm: VLM | None = None,
        vlm_prompt: str = "",
        seg_prompt: str = "",
        frame_skip: int = 1,
        vlm_skip: int = 5,
        sequential_offload: bool = False,
    ):
        self.detector = detector
        self.segmenter = segmenter
        self.vlm = vlm
        self.vlm_prompt = vlm_prompt
        self.seg_prompt = seg_prompt
        self.frame_skip = frame_skip
        self.vlm_skip = vlm_skip
        self.sequential_offload = sequential_offload

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

    def run(self, video_path: str, max_frames: int | None = None) -> Iterator[FrameResult]:
        for idx, frame in read_frames(video_path, max_frames):
            if idx % self.frame_skip != 0:
                continue

            result = FrameResult(frame_idx=idx, frame=frame)

            if self.detector:
                result.detection = self._run_detector(frame, idx)

            if self.segmenter:
                result.segmentation = self._run_segmenter(frame, idx)

            if self.vlm and idx % self.vlm_skip == 0:
                result.vlm = self._run_vlm(frame, idx)

            yield result
