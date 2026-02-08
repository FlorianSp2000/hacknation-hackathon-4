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


def _parse_json(raw_text: str) -> dict | None:
    """Extract JSON from model output, handling markdown code blocks."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        if "```" in raw_text:
            json_str = raw_text.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            try:
                return json.loads(json_str.strip())
            except json.JSONDecodeError:
                pass
    return None


def _build_conversation(pil_image: Image.Image, prompt: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected (H,W,3), got {frame.shape}"
    return Image.fromarray(frame[:, :, ::-1])


# =============================================================================
# Transformers (original) — CUDA / CPU
# =============================================================================

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
            # Use CPU — MPS unified memory is too constrained for this model
            # and accelerate hooks break on MPS with meta-device offloading.
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.float32,
            ).to("cpu")
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)

    def predict(self, frame: np.ndarray, prompt: str) -> VLMResult:
        assert self._model is not None, "Call load() first"
        pil_image = _frame_to_pil(frame)

        inputs = self._processor.apply_chat_template(
            _build_conversation(pil_image, prompt),
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

        generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
        raw_text = self._processor.decode(generated_ids, skip_special_tokens=True).strip()

        return VLMResult(raw_text=raw_text, parsed=_parse_json(raw_text), frame_idx=-1)

    def unload(self) -> None:
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# ONNX Runtime — cross-platform GPU (CUDA, DirectML, TensorRT)
# =============================================================================

@register("vlm", "lfm2.5-vl-onnx")
class LFM25VLONNX(VLM):
    """LFM2.5-VL via ONNX Runtime with manual KV-cache decoding."""

    MODEL_ID = "LiquidAI/LFM2.5-VL-1.6B-ONNX"
    # Variant selection: fp16 encoder + q4 decoder (best speed/quality tradeoff)
    EMBED_TOKENS_FILE = "onnx/embed_tokens_fp16.onnx"
    EMBED_IMAGES_FILE = "onnx/embed_images_fp16.onnx"
    DECODER_FILE = "onnx/decoder_q4.onnx"

    ONNX_DTYPE_MAP = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int64)": np.int64,
    }

    def __init__(self, max_new_tokens: int = 512):
        self.max_new_tokens = max_new_tokens
        self._embed_tokens = None
        self._embed_images = None
        self._decoder = None
        self._processor = None

    def load(self) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download, list_repo_files
        from transformers import AutoProcessor

        # Download ONNX model files + associated data shards
        embed_tokens_path = hf_hub_download(self.MODEL_ID, self.EMBED_TOKENS_FILE)
        embed_images_path = hf_hub_download(self.MODEL_ID, self.EMBED_IMAGES_FILE)
        decoder_path = hf_hub_download(self.MODEL_ID, self.DECODER_FILE)

        # Download all split data files for each model component
        for f in list_repo_files(self.MODEL_ID):
            for name in [self.EMBED_TOKENS_FILE, self.EMBED_IMAGES_FILE, self.DECODER_FILE]:
                if f.startswith(name + "_data"):
                    hf_hub_download(self.MODEL_ID, f)

        self._embed_tokens = ort.InferenceSession(embed_tokens_path)
        self._embed_images = ort.InferenceSession(embed_images_path)
        self._decoder = ort.InferenceSession(decoder_path)
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID, trust_remote_code=True)

    def _init_kv_cache(self) -> dict[str, np.ndarray]:
        """Initialize empty KV cache from decoder input metadata."""
        cache = {}
        for inp in self._decoder.get_inputs():
            if inp.name in {"inputs_embeds", "attention_mask", "position_ids"}:
                continue
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            for i, d in enumerate(inp.shape):
                if isinstance(d, str) and "sequence" in d.lower():
                    shape[i] = 0
            cache[inp.name] = np.zeros(
                shape, dtype=self.ONNX_DTYPE_MAP.get(inp.type, np.float32)
            )
        return cache

    @staticmethod
    def _update_kv_cache(cache: dict, decoder_outputs: list, decoder_session) -> None:
        """Update cache in-place from decoder present_* outputs."""
        for i, out in enumerate(decoder_session.get_outputs()[1:], 1):
            name = out.name.replace("present_conv", "past_conv").replace("present.", "past_key_values.")
            if name in cache:
                cache[name] = decoder_outputs[i]

    def predict(self, frame: np.ndarray, prompt: str) -> VLMResult:
        assert self._decoder is not None, "Call load() first"
        pil_image = _frame_to_pil(frame)

        # Tokenize
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        prompt_str = self._processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self._processor(images=[pil_image], text=prompt_str, return_tensors="pt")

        pixel_values = inputs["pixel_values"].numpy().astype(np.float32)
        pixel_attention_mask = inputs["pixel_attention_mask"].numpy().astype(np.int64)
        spatial_shapes = inputs["spatial_shapes"].numpy().astype(np.int64)
        input_ids = inputs["input_ids"].numpy().astype(np.int64)

        # Image embeddings
        image_embeds = self._embed_images.run(None, {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
            "spatial_shapes": spatial_shapes,
        })[0]

        # Token embeddings
        token_embeds = self._embed_tokens.run(None, {"input_ids": input_ids})[0]

        # Merge image embeddings at <image> token positions
        image_token_id = self._processor.tokenizer.convert_tokens_to_ids("<image>")
        image_positions = np.where(input_ids[0] == image_token_id)[0]
        for i, pos in enumerate(image_positions):
            if i < len(image_embeds):
                token_embeds[0, pos] = image_embeds[i]

        # Autoregressive decoding with KV cache
        cache = self._init_kv_cache()
        seq_len = token_embeds.shape[1]
        generated_tokens = []
        embeds = token_embeds.astype(np.float32)

        for step in range(self.max_new_tokens):
            if step > 0:
                last_token = np.array([[generated_tokens[-1]]], dtype=np.int64)
                embeds = self._embed_tokens.run(None, {"input_ids": last_token})[0].astype(np.float32)

            attn_mask = np.ones((1, seq_len + len(generated_tokens)), dtype=np.int64)
            outputs = self._decoder.run(None, {
                "inputs_embeds": embeds,
                "attention_mask": attn_mask,
                **cache,
            })

            next_token = int(np.argmax(outputs[0][0, -1]))
            generated_tokens.append(next_token)
            self._update_kv_cache(cache, outputs, self._decoder)

            if next_token == self._processor.tokenizer.eos_token_id:
                break

        raw_text = self._processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return VLMResult(raw_text=raw_text, parsed=_parse_json(raw_text), frame_idx=-1)

    def unload(self) -> None:
        del self._embed_tokens, self._embed_images, self._decoder, self._processor
        self._embed_tokens = None
        self._embed_images = None
        self._decoder = None
        self._processor = None


# =============================================================================
# MLX — Apple Silicon (M1/M2/M3/M4)
# =============================================================================

@register("vlm", "lfm2.5-vl-mlx")
class LFM25VLMLX(VLM):
    """LFM2.5-VL via mlx-vlm. Optimized for Apple Silicon."""

    MODEL_ID = "mlx-community/LFM2.5-VL-1.6B-8bit"

    def __init__(self, max_new_tokens: int = 512):
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def load(self) -> None:
        from mlx_vlm import load as mlx_load

        self._model, self._processor = mlx_load(self.MODEL_ID)

    def predict(self, frame: np.ndarray, prompt: str) -> VLMResult:
        assert self._model is not None, "Call load() first"
        pil_image = _frame_to_pil(frame)

        from mlx_vlm import generate

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        formatted_prompt = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        raw_text = generate(
            self._model,
            self._processor,
            formatted_prompt,
            image=pil_image,
            max_tokens=self.max_new_tokens,
            temperature=0.1,
            repetition_penalty=1.05,
        ).strip()

        return VLMResult(raw_text=raw_text, parsed=_parse_json(raw_text), frame_idx=-1)

    def unload(self) -> None:
        del self._model, self._processor
        self._model = None
        self._processor = None
