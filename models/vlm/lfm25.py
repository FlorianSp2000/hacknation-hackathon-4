from __future__ import annotations

import json
from typing import Any
import numpy as np
import torch
from PIL import Image

from core.base import VLM
from core.registry import register
from core.types import VLMResult


GENERIC_JSON_SCHEMA_PROMPT = """Analyze this frame. Output ONLY valid JSON with this exact schema, no other text:
{
  "objects": [{"name": "str", "state": "str", "bbox_approx": [x1, y1, x2, y2]}],
  "actions": ["str"],
  "scene_type": "str",
  "navigation_relevant": [{"type": "door|obstacle|passage|surface", "state": "open|closed|blocked|clear"}]
}"""

NAV_STATE_PROMPT = """You are a navigation ground truth labeler for humanoid robots.
For each door, drawer, handle, cabinet, passage, or obstacle visible in this frame, classify its state.

Output ONLY valid JSON, no other text:
{"nav_objects": [{"name": "str", "type": "door|drawer|handle|cabinet|passage|obstacle", "state": "open|closed|ajar|blocked|clear|unknown", "interactable": true}]}

Rules:
- Only include navigation-relevant objects (doors, drawers, handles, cabinets, passages, obstacles)
- "state" must be exactly one of: open, closed, ajar, blocked, clear, unknown
- "interactable" means a humanoid could physically interact with it
- If no navigation objects are visible, return {"nav_objects": []}"""

DEFAULT_JSON_SCHEMA_PROMPT = NAV_STATE_PROMPT


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


def _resize_for_mlx_vlm(pil_image: Image.Image, max_side: int = 448) -> Image.Image:
    """Resize image conservatively for MLX VLM to reduce token/feature mismatch risk."""
    w, h = pil_image.size
    longest = max(w, h)
    if longest <= max_side:
        return pil_image
    scale = float(max_side) / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return pil_image.resize((new_w, new_h), Image.Resampling.BICUBIC)


def _resize_square_for_mlx_vlm(pil_image: Image.Image, side: int) -> Image.Image:
    """Force a square RGB input size for stricter MLX image-token alignment."""
    rgb = pil_image.convert("RGB")
    return rgb.resize((side, side), Image.Resampling.BICUBIC)


def _normalize_mlx_prompt(prompt: str) -> str:
    """Ensure exactly one <image> token is present in the prompt."""
    clean = prompt.strip()
    marker = "<image>"
    count = clean.count(marker)
    if count == 0:
        return f"{marker}\n{clean}"
    if count == 1:
        return clean
    # Keep only the first marker occurrence and remove the rest.
    first = clean.find(marker)
    before = clean[: first + len(marker)]
    after = clean[first + len(marker) :].replace(marker, "")
    return (before + after).strip()


def _normalize_generated_text(output: Any) -> str:
    """Convert heterogeneous backend outputs into a plain string."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="ignore").strip()
    if isinstance(output, dict):
        for key in ("text", "output", "response", "generated_text"):
            value = output.get(key)
            if isinstance(value, str):
                return value.strip()
        choices = output.get("choices")
        if isinstance(choices, list):
            parts: list[str] = []
            for choice in choices:
                if isinstance(choice, dict):
                    # OpenAI-style chat shape.
                    msg = choice.get("message")
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        if isinstance(content, str) and content.strip():
                            parts.append(content.strip())
                            continue
                    for key in ("text", "content"):
                        value = choice.get(key)
                        if isinstance(value, str) and value.strip():
                            parts.append(value.strip())
            if parts:
                return "\n".join(parts).strip()
    for attr in ("text", "output", "response", "generated_text"):
        value = getattr(output, attr, None)
        if isinstance(value, str):
            return value.strip()
    if isinstance(output, (list, tuple)):
        parts: list[str] = []
        for item in output:
            text = _normalize_generated_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(output).strip()


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

        # MLX VLM expects image placeholders in text to match provided image count.
        # Keep exactly one placeholder for one image.
        clean_prompt = _normalize_mlx_prompt(prompt)

        def _gen(img: Image.Image, text_prompt: str, *, max_tokens: int | None = None, temperature: float = 0.1):
            return generate(
                self._model,
                self._processor,
                text_prompt,
                image=img,
                max_tokens=max_tokens if max_tokens is not None else self.max_new_tokens,
                temperature=temperature,
                repetition_penalty=1.05,
            )

        try:
            output = _gen(pil_image.convert("RGB"), clean_prompt)
        except ValueError as exc:
            msg = str(exc)
            is_mismatch = "Image features and image tokens do not match" in msg
            if not is_mismatch:
                raise
            # Retry chain for problematic frames:
            # 1) conservative aspect-preserving resize
            # 2) fixed square sizes (often fixes patch-grid/token alignment)
            candidates = [_resize_for_mlx_vlm(pil_image, max_side=448)]
            candidates.extend(_resize_square_for_mlx_vlm(pil_image, side) for side in (448, 384, 336, 320, 256))
            last_exc: Exception | None = None
            output = None
            for candidate in candidates:
                try:
                    output = _gen(candidate, clean_prompt)
                    last_exc = None
                    break
                except ValueError as inner_exc:
                    if "Image features and image tokens do not match" not in str(inner_exc):
                        raise
                    last_exc = inner_exc
            if output is None and last_exc is not None:
                raise last_exc

        raw_text = _normalize_generated_text(output)

        # Some MLX runs can return empty text without raising; retry once with a shorter, deterministic prompt.
        if not raw_text:
            fallback_prompt = _normalize_mlx_prompt(
                "Return ONLY valid JSON with this schema: "
                '{"objects":[{"name":"str","state":"str","robot_actions":["str"]}]}.'
            )
            fallback_img = _resize_square_for_mlx_vlm(pil_image, 336)
            try:
                fallback_output = _gen(fallback_img, fallback_prompt, max_tokens=min(self.max_new_tokens, 192), temperature=0.0)
                raw_text = _normalize_generated_text(fallback_output)
            except Exception:
                # Preserve empty output; caller/UI now surfaces explicit empty-output diagnostics.
                pass

        return VLMResult(raw_text=raw_text, parsed=_parse_json(raw_text), frame_idx=-1)

    def unload(self) -> None:
        del self._model, self._processor
        self._model = None
        self._processor = None
