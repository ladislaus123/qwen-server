"""Janus-Pro Transformers inference engine."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from PIL import Image

from qwen_service.config import Settings

logger = logging.getLogger(__name__)


class TransformersJanusEngine:
    """Runs Janus-Pro image-understanding models through their custom code path."""

    supports_concurrent_generation = False

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_id = settings.model_id
        self._model: Any | None = None
        self._processor: Any | None = None
        self._tokenizer: Any | None = None
        self._device: str | None = None
        self._dtype: Any | None = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def device(self) -> str | None:
        return self._device

    async def load(self) -> None:
        if self._ready:
            return
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        if not self.settings.trust_remote_code:
            raise RuntimeError(
                "Janus-Pro requires LOCAL_VISION_TRUST_REMOTE_CODE=true because "
                "it loads DeepSeek's custom Janus model code."
            )

        import torch
        from transformers import AutoModelForCausalLM

        try:
            from janus.models import VLChatProcessor
        except ImportError as exc:
            raise RuntimeError(
                "LOCAL_VISION_MODEL_FAMILY=janus requires the Janus package. "
                "Install it with `pip install git+https://github.com/deepseek-ai/Janus.git`."
            ) from exc

        device = self._select_device(torch)
        dtype = self._select_dtype(torch, device)
        self._device = device
        self._dtype = dtype

        logger.info(
            "Loading Janus model %s on %s with dtype=%s",
            self.model_id,
            device,
            dtype,
        )
        self._processor = VLChatProcessor.from_pretrained(self.model_id)
        self._tokenizer = self._processor.tokenizer
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=self.settings.trust_remote_code,
        )
        self._model = self._model.to(dtype).to(device).eval()
        self._ready = True
        logger.info("Janus model ready on %s", device)

    def _select_device(self, torch_module: Any) -> str:
        policy = self.settings.device_policy
        cuda_available = torch_module.cuda.is_available()
        mps_available = (
            hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        )

        if policy == "cuda":
            if not cuda_available:
                raise RuntimeError("LOCAL_VISION_DEVICE=cuda was requested, but CUDA is unavailable")
            return "cuda"

        if policy == "mps":
            if not mps_available:
                raise RuntimeError("LOCAL_VISION_DEVICE=mps was requested, but MPS is unavailable")
            return "mps"

        if policy == "cpu":
            logger.warning("Running Janus on CPU will be slow")
            return "cpu"

        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"

        logger.warning("No CUDA or MPS device detected; falling back to CPU")
        return "cpu"

    def _select_dtype(self, torch_module: Any, device: str) -> Any:
        dtype = self.settings.janus_dtype.lower()
        if dtype == "bfloat16":
            return torch_module.bfloat16
        if dtype == "float16":
            return torch_module.float16
        if dtype == "float32":
            return torch_module.float32

        if device == "cuda":
            try:
                major, _minor = torch_module.cuda.get_device_capability()
            except Exception:
                logger.debug("Unable to detect CUDA capability; using float16", exc_info=True)
                return torch_module.float16
            if major >= 8:
                return torch_module.bfloat16
            return torch_module.float16
        if device == "mps":
            return torch_module.float16
        return torch_module.float32

    async def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        if not self._ready:
            raise RuntimeError("model is not loaded")
        return await asyncio.to_thread(self._generate_sync, image, prompt, max_new_tokens)

    def _generate_sync(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        import torch

        if self._model is None or self._processor is None or self._tokenizer is None:
            raise RuntimeError("model is not loaded")

        image_rgb = image.convert("RGB")
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt}",
                "images": [image_rgb],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        logger.info(
            "Janus generate received image mode=%s size=%sx%s prompt_chars=%d",
            image.mode,
            image.width,
            image.height,
            len(prompt),
        )
        prepare_inputs = self._processor(
            conversations=conversation,
            images=[image_rgb],
            force_batchify=True,
        )
        prepare_inputs = _move_janus_inputs(prepare_inputs, self._input_device())

        with torch.inference_mode():
            logger.info(
                "Janus generate preparing embeddings max_new_tokens=%d",
                max_new_tokens,
            )
            inputs_embeds = self._model.prepare_inputs_embeds(**prepare_inputs)
            generated_ids = self._model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=_get_input_value(prepare_inputs, "attention_mask"),
                pad_token_id=self._tokenizer.eos_token_id,
                bos_token_id=self._tokenizer.bos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        result = self._decode_output(generated_ids).strip()
        logger.info(
            "Janus generate decoded response chars=%d result=%r",
            len(result),
            result,
        )
        return result

    def _decode_output(self, generated_ids: Any) -> str:
        output_ids = generated_ids[0]
        to_cpu = getattr(output_ids, "cpu", None)
        if callable(to_cpu):
            output_ids = to_cpu()
        to_list = getattr(output_ids, "tolist", None)
        if callable(to_list):
            output_ids = to_list()
        return self._tokenizer.decode(output_ids, skip_special_tokens=True)

    def _input_device(self) -> str:
        if self._device == "cuda":
            return "cuda"
        if self._device == "mps":
            return "mps"
        return "cpu"

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)

    def _close_sync(self) -> None:
        model = self._model
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._dtype = None
        self._ready = False
        del model

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("Unable to clear CUDA cache", exc_info=True)


def _move_janus_inputs(inputs: Any, device: str) -> Any:
    move = getattr(inputs, "to", None)
    if callable(move):
        return move(device)
    return inputs


def _get_input_value(inputs: Any, key: str) -> Any:
    try:
        return getattr(inputs, key)
    except AttributeError:
        pass
    try:
        return inputs[key]
    except (KeyError, TypeError):
        return None
