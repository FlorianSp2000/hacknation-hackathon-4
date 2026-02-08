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
    MODEL_ID = "mlx-community/LFM2.5-VL-1.6B-8bit"
    FALLBACK_MODEL_ID = "LiquidAI/LFM2.5-VL-1.6B"

    def __init__(self, max_new_tokens: int = 512):
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._backend = ""

    def load(self) -> None:
        # Preferred on Apple Silicon: MLX 8-bit model.
        try:
            from mlx_vlm import load as mlx_load
            from mlx_vlm.utils import load_config

            self._model, self._processor = mlx_load(self.MODEL_ID)
            self._config = load_config(self.MODEL_ID)
            self._backend = "mlx"
            return
        except Exception:
            # Fall back to HF Transformers when MLX stack is unavailable.
            pass

        from transformers import AutoProcessor, AutoModelForImageTextToText

        if torch.cuda.is_available():
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.FALLBACK_MODEL_ID,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        else:
            # Avoid accelerate auto device mapping issues on non-CUDA systems.
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.FALLBACK_MODEL_ID,
                torch_dtype=torch.float32,
            ).to("cpu")
        self._processor = AutoProcessor.from_pretrained(self.FALLBACK_MODEL_ID)
        self._backend = "transformers"

    def predict(self, frame: np.ndarray, prompt: str) -> VLMResult:
        assert self._model is not None, "Call load() first"
        assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected (H,W,3), got {frame.shape}"

        # BGR → RGB → PIL
        pil_image = Image.fromarray(frame[:, :, ::-1])

        if self._backend == "mlx":
            from mlx_vlm import generate as mlx_generate
            from mlx_vlm.prompt_utils import apply_chat_template

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            try:
                formatted_prompt = apply_chat_template(self._processor, messages)
            except TypeError:
                formatted_prompt = apply_chat_template(self._processor, self._config, prompt, num_images=1)

            # mlx-vlm has had minor signature changes; support both common forms.
            try:
                raw_text = mlx_generate(
                    self._model,
                    self._processor,
                    formatted_prompt,
                    [pil_image],
                    max_tokens=self.max_new_tokens,
                    verbose=False,
                )
            except TypeError:
                raw_text = mlx_generate(
                    self._model,
                    self._processor,
                    [pil_image],
                    formatted_prompt,
                    max_tokens=self.max_new_tokens,
                    verbose=False,
                )
            raw_text = str(raw_text).strip()
            parsed = None
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
            return VLMResult(raw_text=raw_text, parsed=parsed, frame_idx=-1)

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
        self._backend = ""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
