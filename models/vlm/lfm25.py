from __future__ import annotations

import json
import numpy as np
import torch
from PIL import Image

from core.base import VLM
from core.registry import register
from core.types import VLMResult


DEFAULT_JSON_SCHEMA_PROMPT = """Analyze this frame. Output ONLY valid JSON with this exact schema, no other text:
{
  "objects": [{"name": "str", "state": "str", "bbox_approx": [x1, y1, x2, y2]}],
  "actions": ["str"],
  "scene_type": "str",
  "navigation_relevant": [{"type": "door|obstacle|passage|surface", "state": "open|closed|blocked|clear"}]
}"""


@register("vlm", "lfm2.5-vl")
class LFM25VL(VLM):
    MODEL_ID = "LiquidAI/LFM2.5-VL-1.6B"

    def __init__(self, max_new_tokens: int = 512):
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def load(self) -> None:
        from transformers import AutoProcessor, AutoModelForImageTextToText

        if torch.cuda.is_available():
            # CUDA: let accelerate handle multi-GPU placement
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.MODEL_ID,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        else:
            # MPS / CPU: device_map="auto" causes disk offload which breaks on MPS.
            # Load to CPU with float32 (MPS has limited bfloat16 support), then move.
            # Use CPU — MPS unified memory is too constrained for this model
            # and accelerate hooks break on MPS with meta-device offloading.
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.float32,
            ).to("cpu")
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)

    def predict(self, frame: np.ndarray, prompt: str) -> VLMResult:
        assert self._model is not None, "Call load() first"
        assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected (H,W,3), got {frame.shape}"

        # BGR → RGB → PIL
        pil_image = Image.fromarray(frame[:, :, ::-1])

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        inputs = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=0.1,
            repetition_penalty=1.05,
            do_sample=True,
        )

        # Decode only the generated tokens (skip input)
        generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
        raw_text = self._processor.decode(generated_ids, skip_special_tokens=True).strip()

        # Attempt JSON parse
        parsed = None
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown code block
            if "```" in raw_text:
                json_str = raw_text.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                try:
                    parsed = json.loads(json_str.strip())
                except json.JSONDecodeError:
                    pass

        return VLMResult(raw_text=raw_text, parsed=parsed, frame_idx=-1)

    def unload(self) -> None:
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
