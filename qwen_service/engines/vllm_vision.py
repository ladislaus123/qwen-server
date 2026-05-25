"""vLLM-backed vision-language inference engine."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any
from uuid import uuid4

from PIL import Image

from qwen_service.config import Settings

logger = logging.getLogger(__name__)


class VllmVisionLanguageEngine:
    """Runs chat-style vision-language models through vLLM."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_id = settings.model_id
        self._llm: Any | None = None
        self._processor: Any | None = None
        self._device: str | None = None
        self._ready = False
        self._closed = False
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def device(self) -> str | None:
        return self._device

    async def load(self) -> None:
        if self._closed:
            raise RuntimeError("engine is closed")
        if self._ready:
            return
        await self._load()

    async def _load(self) -> None:
        try:
            from transformers import AutoProcessor
            AsyncEngineArgs, AsyncLLMEngine = _import_vllm_async_engine()
        except ImportError as exc:
            raise RuntimeError(
                "LOCAL_VISION_BACKEND=vllm requires vLLM. Install it with `pip install vllm`."
            ) from exc

        processor_kwargs: dict[str, Any] = {
            "trust_remote_code": self.settings.trust_remote_code,
        }
        engine_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "trust_remote_code": self.settings.trust_remote_code,
            "limit_mm_per_prompt": {"image": 1},
            "max_num_seqs": self.settings.vllm_max_num_seqs,
        }

        if self.settings.vllm_max_model_len is not None:
            engine_kwargs["max_model_len"] = self.settings.vllm_max_model_len
        if self.settings.vllm_tensor_parallel_size > 1:
            engine_kwargs["tensor_parallel_size"] = self.settings.vllm_tensor_parallel_size

        mm_processor_kwargs = self._mm_processor_kwargs()
        if mm_processor_kwargs:
            engine_kwargs["mm_processor_kwargs"] = mm_processor_kwargs

        logger.info(
            "Loading %s with vLLM args=%s",
            self.model_id,
            _safe_log_kwargs(engine_kwargs),
        )
        self._processor = await asyncio.to_thread(
            AutoProcessor.from_pretrained,
            self.model_id,
            **processor_kwargs,
        )
        engine_args = AsyncEngineArgs(**engine_kwargs)
        self._llm = AsyncLLMEngine.from_engine_args(engine_args)
        self._device = "vllm"
        self._ready = True
        logger.info("vLLM model ready")

    def _mm_processor_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.settings.processor_min_pixels is not None:
            kwargs["min_pixels"] = self.settings.processor_min_pixels
        if self.settings.processor_max_pixels is not None:
            kwargs["max_pixels"] = self.settings.processor_max_pixels
        return kwargs

    async def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        if not self._ready:
            raise RuntimeError("model is not loaded")
        async with self._get_semaphore():
            return await self._generate_async(image, prompt, max_new_tokens)

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(
                self.settings.vllm_max_concurrent_requests
            )
        return self._semaphore

    async def _generate_async(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        from vllm import SamplingParams

        if self._llm is None or self._processor is None:
            raise RuntimeError("model is not loaded")

        logger.info(
            "vLLM generate received image mode=%s size=%sx%s prompt_chars=%d",
            image.mode,
            image.width,
            image.height,
            len(prompt),
        )

        prompt_text = self._build_prompt_text(image, prompt)
        request = {
            "prompt": prompt_text,
            "multi_modal_data": {"image": image},
        }
        sampling_params = _create_sampling_params(SamplingParams, max_new_tokens)
        request_id = f"qwen-vllm-{uuid4().hex}"

        logger.info(
            "vLLM generate starting request_id=%s max_tokens=%d",
            request_id,
            max_new_tokens,
        )
        final_output: Any | None = None
        generator = self._llm.generate(request, sampling_params, request_id)
        try:
            async for output in generator:
                final_output = output
        except asyncio.CancelledError:
            await self._abort_request(request_id)
            await _close_async_generator(generator)
            raise

        result = _extract_output_text(final_output)
        logger.info(
            "vLLM generate decoded request_id=%s response chars=%d result=%r",
            request_id,
            len(result),
            result,
        )
        return result

    def _build_prompt_text(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
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
            logger.debug(
                "Processor does not support chat templates; using vLLM fallback prompt",
                exc_info=True,
            )
            return f"USER: <image>\n{prompt}\nASSISTANT:"

    async def close(self) -> None:
        if self._closed:
            return
        try:
            llm = self._llm
            self._llm = None
            self._processor = None
            self._device = None
            self._ready = False

            if llm is not None:
                await _shutdown_llm(llm)
                del llm

            await asyncio.to_thread(_clear_cuda_cache)
        finally:
            self._closed = True

    async def _abort_request(self, request_id: str) -> None:
        if self._llm is None:
            return

        abort = getattr(self._llm, "abort", None)
        if not callable(abort):
            return

        try:
            result = abort(request_id)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.debug("Unable to abort vLLM request %s", request_id, exc_info=True)


def _safe_log_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in kwargs.items()
        if key not in {"hf_token", "token"}
    }


def _import_vllm_async_engine() -> tuple[Any, Any]:
    try:
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        return AsyncEngineArgs, AsyncLLMEngine
    except ImportError:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        return AsyncEngineArgs, AsyncLLMEngine


def _create_sampling_params(sampling_params_class: Any, max_new_tokens: int) -> Any:
    kwargs: dict[str, Any] = {
        "max_tokens": max_new_tokens,
        "temperature": 0,
    }
    try:
        from vllm.sampling_params import RequestOutputKind

        kwargs["output_kind"] = RequestOutputKind.FINAL_ONLY
    except Exception:
        logger.debug(
            "vLLM RequestOutputKind unavailable; using default output mode",
            exc_info=True,
        )

    try:
        return sampling_params_class(**kwargs)
    except TypeError:
        if "output_kind" not in kwargs:
            raise
        logger.debug(
            "vLLM SamplingParams rejected output_kind; retrying without it",
            exc_info=True,
        )
        kwargs.pop("output_kind")
        return sampling_params_class(**kwargs)


def _extract_output_text(output: Any | None) -> str:
    if output is None:
        return ""

    outputs = getattr(output, "outputs", None)
    if not outputs:
        return ""

    text = getattr(outputs[0], "text", "")
    return str(text).strip()


async def _close_async_generator(generator: Any) -> None:
    close = getattr(generator, "aclose", None)
    if not callable(close):
        return

    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.debug("Unable to close vLLM output generator", exc_info=True)


async def _shutdown_llm(llm: Any) -> None:
    for target in (llm, getattr(llm, "llm_engine", None)):
        if target is None:
            continue
        shutdown = getattr(target, "shutdown_background_loop", None)
        if not callable(shutdown):
            shutdown = getattr(target, "shutdown", None)
        if callable(shutdown):
            try:
                result = shutdown()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.debug("Unable to shut down vLLM cleanly", exc_info=True)
            return


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("Unable to clear CUDA cache", exc_info=True)
