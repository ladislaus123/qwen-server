"""Generic Transformers-backed vision-language inference engine."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from PIL import Image

from qwen_service.config import Settings

logger = logging.getLogger(__name__)


class TransformersVisionLanguageEngine:
    """Runs chat-style Hugging Face vision-language models with Auto classes."""

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
        from transformers import AutoProcessor

        device = self._select_device(torch)
        self._device = device

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.settings.trust_remote_code,
        }
        if device == "cuda":
            model_kwargs.update({"torch_dtype": "auto", "device_map": "auto"})
        elif device == "mps":
            model_kwargs.update({"torch_dtype": torch.float16})
        else:
            model_kwargs.update({"torch_dtype": "auto"})

        if self.settings.use_flash_attention and device == "cuda":
            model_kwargs["attn_implementation"] = "flash_attention_2"

        processor_kwargs: dict[str, Any] = {
            "trust_remote_code": self.settings.trust_remote_code,
        }

        model_class = self._resolve_model_class()
        logger.info(
            "Loading %s with %s on %s",
            self.model_id,
            self.settings.auto_model_class,
            device,
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id, **processor_kwargs)
        self._model = model_class.from_pretrained(self.model_id, **model_kwargs)

        if device in {"mps", "cpu"}:
            self._model.to(device)

        self._model.eval()
        self._ready = True
        logger.info("Local vision model ready on %s", device)

    def _resolve_model_class(self) -> Any:
        from transformers import AutoModelForVision2Seq

        if self.settings.auto_model_class == "auto_vision2seq":
            return AutoModelForVision2Seq

        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            logger.warning(
                "AutoModelForImageTextToText is unavailable; falling back to AutoModelForVision2Seq"
            )
            return AutoModelForVision2Seq
        return AutoModelForImageTextToText

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
            logger.warning("Running a local vision-language model on CPU can be slow")
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

        if self._model is None or self._processor is None:
            raise RuntimeError("model is not loaded")

        logger.info(
            "Local vision generate received image mode=%s size=%sx%s prompt_chars=%d",
            image.mode,
            image.width,
            image.height,
            len(prompt),
        )

        text = self._build_prompt_text(image, prompt)
        inputs = self._prepare_inputs(text, image)
        inputs = inputs.to(self._input_device())

        with torch.inference_mode():
            logger.info(
                "Local vision generate starting model.generate max_new_tokens=%d",
                max_new_tokens,
            )
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        logger.info("Local vision generate completed model.generate")

        input_ids = _get_input_ids(inputs)
        if input_ids is not None:
            generated_ids = [
                output_ids[len(prompt_ids) :]
                for prompt_ids, output_ids in zip(input_ids, generated_ids)
            ]

        output_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        result = output_text[0].strip() if output_text else ""
        logger.info(
            "Local vision generate decoded response chars=%d result=%r",
            len(result),
            result,
        )
        return result

    def _build_prompt_text(self, image: Image.Image, prompt: str) -> str:
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
            return self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            logger.debug("Processor does not support chat templates; using raw prompt", exc_info=True)
            return prompt

    def _prepare_inputs(self, text: str, image: Image.Image) -> Any:
        try:
            return self._processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
        except TypeError:
            return self._processor(
                text=text,
                images=image,
                return_tensors="pt",
            )

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


def _get_input_ids(inputs: Any) -> Any | None:
    try:
        return inputs.input_ids
    except AttributeError:
        pass

    try:
        return inputs.get("input_ids")
    except AttributeError:
        return None
