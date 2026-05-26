"""Qwen2.5-VL optimized Transformers inference engine."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from PIL import Image

from qwen_service.config import Settings

logger = logging.getLogger(__name__)


class TransformersQwenEngine:
    """Runs Qwen2.5-VL with its model-specific Transformers class."""

    supports_concurrent_generation = False

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_id = settings.model_id
        self._model: Any | None = None
        self._processor: Any | None = None
        self._device: str | None = None
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
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        device = self._select_device(torch)
        self._device = device

        model_kwargs: dict[str, Any] = {}
        if device == "cuda":
            model_kwargs.update({"torch_dtype": "auto", "device_map": "auto"})
        elif device == "mps":
            model_kwargs.update({"torch_dtype": torch.float16})
        else:
            model_kwargs.update({"torch_dtype": "auto"})

        if self.settings.use_flash_attention and device == "cuda":
            model_kwargs["attn_implementation"] = "flash_attention_2"

        processor_kwargs: dict[str, Any] = {}
        if self.settings.processor_min_pixels is not None:
            processor_kwargs["min_pixels"] = self.settings.processor_min_pixels
        if self.settings.processor_max_pixels is not None:
            processor_kwargs["max_pixels"] = self.settings.processor_max_pixels

        logger.info("Loading %s on %s", self.model_id, device)
        self._processor = AutoProcessor.from_pretrained(self.model_id, **processor_kwargs)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs,
        )

        if device in {"mps", "cpu"}:
            self._model.to(device)

        self._model.eval()
        self._ready = True
        logger.info("Qwen model ready on %s", device)

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
            logger.warning("Running Qwen2.5-VL-7B on CPU will be slow")
            return "cpu"

        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"

        logger.warning("No CUDA or MPS device detected; falling back to CPU")
        return "cpu"

    async def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        if not self._ready:
            raise RuntimeError("model is not loaded")
        return await asyncio.to_thread(self._generate_sync, image, prompt, max_new_tokens)

    def _generate_sync(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        if self._model is None or self._processor is None:
            raise RuntimeError("model is not loaded")

        logger.info(
            "Qwen generate received image mode=%s size=%sx%s prompt_chars=%d",
            image.mode,
            image.width,
            image.height,
            len(prompt),
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        logger.debug("Qwen generate applying chat template")
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        logger.info(
            "Qwen generate processed vision inputs image_count=%d video_count=%d",
            len(image_inputs or []),
            len(video_inputs or []),
        )

        logger.debug("Qwen generate preparing processor tensors")
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._input_device())
        logger.info(
            "Qwen generate prepared inputs device=%s tensors=%s",
            self._input_device(),
            self._describe_processor_inputs(inputs),
        )

        with torch.inference_mode():
            logger.info(
                "Qwen generate starting model.generate max_new_tokens=%d",
                max_new_tokens,
            )
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        logger.info("Qwen generate completed model.generate")

        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        logger.info(
            "Qwen generate trimmed output token_counts=%s",
            [len(output_ids) for output_ids in generated_ids_trimmed],
        )
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        result = output_text[0].strip() if output_text else ""
        logger.info(
            "Qwen generate decoded response chars=%d result=%r",
            len(result),
            result,
        )
        return result

    @staticmethod
    def _describe_processor_inputs(inputs: Any) -> dict[str, str]:
        description: dict[str, str] = {}
        try:
            items = inputs.items()
        except AttributeError:
            return {"inputs": type(inputs).__name__}

        for key, value in items:
            shape = getattr(value, "shape", None)
            dtype = getattr(value, "dtype", None)
            device = getattr(value, "device", None)
            details: list[str] = []
            if shape is not None:
                details.append(f"shape={tuple(shape)}")
            if dtype is not None:
                details.append(f"dtype={dtype}")
            if device is not None:
                details.append(f"device={device}")
            description[key] = " ".join(details) or type(value).__name__
        return description

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
        self._ready = False
        del model

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("Unable to clear CUDA cache", exc_info=True)
