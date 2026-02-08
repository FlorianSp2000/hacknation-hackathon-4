from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from core.base import Segmenter
from core.registry import register
from core.types import Mask, SegmentationResult


@register("segmenter", "sam3")
class SAM3Segmenter(Segmenter):
    def __init__(self, threshold: float = 0.5, mask_threshold: float = 0.5):
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self._model = None
        self._processor = None
        self._device = None

    def load(self) -> None:
        from transformers import Sam3Processor, Sam3Model

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = Sam3Model.from_pretrained("facebook/sam3").to(self._device)
        self._processor = Sam3Processor.from_pretrained("facebook/sam3")

    def predict(self, frame: np.ndarray, text_prompt: str | None = None) -> SegmentationResult:
        assert self._model is not None, "Call load() first"
        assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected (H,W,3), got {frame.shape}"

        # BGR → RGB → PIL
        pil_image = Image.fromarray(frame[:, :, ::-1])

        inputs = self._processor(
            images=pil_image,
            text=text_prompt,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = []
        for i in range(len(results["masks"])):
            masks.append(Mask(
                binary_mask=results["masks"][i].cpu().numpy().astype(bool),
                score=float(results["scores"][i]),
            ))

        return SegmentationResult(masks=masks, frame_idx=-1)

    def unload(self) -> None:
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
