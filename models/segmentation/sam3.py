from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from core.base import Segmenter
from core.registry import register
from core.types import Mask, SegmentationResult


@register("segmenter", "sam3")
class SAM3Segmenter(Segmenter):
    """SAM3 video segmenter with streaming frame-by-frame inference."""

    def __init__(self, threshold: float = 0.5, mask_threshold: float = 0.5):
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self._model = None
        self._processor = None
        self._device = None
        self._dtype = None
        self._session = None
        self._session_prompt = None

    def load(self) -> None:
        from transformers import Sam3VideoProcessor, Sam3VideoModel

        if torch.cuda.is_available():
            self._device = "cuda"
            self._dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            self._device = "mps"
            self._dtype = torch.float32
        else:
            self._device = "cpu"
            self._dtype = torch.float32
        self._model = Sam3VideoModel.from_pretrained(
            "facebook/sam3",
        ).to(self._device, dtype=self._dtype)
        self._processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

    def _init_session(self, text_prompt: str) -> None:
        """Initialize streaming video session with text prompt."""
        self._session = self._processor.init_video_session(
            inference_device=self._device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self._dtype,
        )
        self._session = self._processor.add_text_prompt(
            inference_session=self._session,
            text=text_prompt,
        )
        self._session_prompt = text_prompt

    def predict(self, frame: np.ndarray, text_prompt: str | None = None) -> SegmentationResult:
        assert self._model is not None, "Call load() first"
        assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected (H,W,3), got {frame.shape}"

        pil_image = Image.fromarray(frame[:, :, ::-1])  # BGR → RGB → PIL

        # Initialize or reset session if prompt changed
        if text_prompt and (self._session is None or text_prompt != self._session_prompt):
            self._init_session(text_prompt)

        # Streaming: feed frame into video session
        if self._session is not None:
            inputs = self._processor(images=pil_image, device=self._device, return_tensors="pt")

            with torch.inference_mode():
                outputs = self._model(
                    inference_session=self._session,
                    frame=inputs.pixel_values[0],
                    reverse=False,
                )

            results = self._processor.postprocess_outputs(
                self._session,
                outputs,
                original_sizes=inputs.original_sizes,
            )

            masks = []
            for i in range(len(results["masks"])):
                masks.append(Mask(
                    binary_mask=results["masks"][i].cpu().numpy().astype(bool),
                    score=float(results["scores"][i]) if "scores" in results else 1.0,
                    label=text_prompt,
                ))
            return SegmentationResult(masks=masks, frame_idx=-1)

        # Fallback: no prompt, no session — skip
        return SegmentationResult(masks=[], frame_idx=-1)

    def reset_session(self) -> None:
        """Reset streaming session (e.g. for a new video)."""
        self._session = None
        self._session_prompt = None

    def unload(self) -> None:
        self._session = None
        self._session_prompt = None
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
