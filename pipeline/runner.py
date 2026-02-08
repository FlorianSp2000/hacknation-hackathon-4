from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterator
from time import perf_counter

from core.base import Detector, Segmenter, VLM
from core.types import FrameResult
from core.video import read_frames
from pipeline.interactions import InteractionEngine


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
        interaction_engine: InteractionEngine | None = None,
    ):
        self.detector = detector
        self.segmenter = segmenter
        self.vlm = vlm
        self.vlm_prompt = vlm_prompt
        self.seg_prompt = seg_prompt
        self.frame_skip = frame_skip
        self.vlm_skip = vlm_skip
        self.sequential_offload = sequential_offload
        self.interaction_engine = interaction_engine

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
        executor = None if self.sequential_offload else ThreadPoolExecutor(max_workers=3)

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
                    if self.segmenter:
                        t0 = perf_counter()
                        result.segmentation = self._run_segmenter(frame, idx)
                        result.timings["segmenter"] = perf_counter() - t0
                    if self.vlm and idx % self.vlm_skip == 0:
                        t0 = perf_counter()
                        result.vlm = self._run_vlm(frame, idx)
                        result.timings["vlm"] = perf_counter() - t0
                else:
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
                    if self.vlm and idx % self.vlm_skip == 0:
                        t0 = perf_counter()
                        futures["vlm"] = (
                            executor.submit(self._run_vlm, frame, idx),
                            t0,
                        )

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
