"""Application orchestration for image analysis requests."""

from __future__ import annotations

import asyncio
import logging
import time
from uuid import uuid4

from qwen_service.config import Settings
from qwen_service.engines.base import VisionLanguageEngine
from qwen_service.image_io import ImageDecodeError, decode_base64_image
from qwen_service.schemas import AnalyzeRequest, AnalyzeResponse

logger = logging.getLogger(__name__)


class AnalyzeInputError(ValueError):
    """Raised for request data that should become HTTP 400."""


class LocalVisionAnalyzeService:
    """Coordinates decoding, token capping, and model generation."""

    def __init__(self, engine: VisionLanguageEngine, settings: Settings):
        self.engine = engine
        self.settings = settings
        self._inference_lock = asyncio.Lock()

    async def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        request_id = uuid4().hex[:12]
        started_at = time.perf_counter()
        prompt = request.prompt.strip()
        image_text = request.image.strip()

        logger.info(
            (
                "Analyze request %s received "
                "image_present=%s image_chars=%d data_url=%s "
                "prompt_chars=%d requested_max_new_tokens=%s"
            ),
            request_id,
            bool(image_text),
            len(image_text),
            image_text.startswith("data:"),
            len(prompt),
            request.max_new_tokens,
        )

        try:
            image = decode_base64_image(
                request.image,
                max_pixels=self.settings.max_image_pixels,
                max_bytes=self.settings.max_image_bytes,
            )
        except ImageDecodeError as exc:
            logger.warning(
                "Analyze request %s rejected during image decode after %.1fms: %s",
                request_id,
                _elapsed_ms(started_at),
                exc,
            )
            raise AnalyzeInputError(str(exc)) from exc

        logger.info(
            "Analyze request %s decoded image mode=%s size=%sx%s pixels=%d",
            request_id,
            image.mode,
            image.width,
            image.height,
            image.width * image.height,
        )

        max_new_tokens = self.settings.clamp_max_new_tokens(request.max_new_tokens)
        if request.max_new_tokens is not None and max_new_tokens != request.max_new_tokens:
            logger.info(
                "Analyze request %s clamped max_new_tokens from %d to %d",
                request_id,
                request.max_new_tokens,
                max_new_tokens,
            )
        else:
            logger.info(
                "Analyze request %s using max_new_tokens=%d",
                request_id,
                max_new_tokens,
            )

        try:
            if self.settings.backend.lower() == "vllm":
                logger.info(
                    (
                        "Analyze request %s using concurrent vLLM inference "
                        "engine_ready=%s device=%s"
                    ),
                    request_id,
                    self.engine.ready,
                    self.engine.device,
                )
                result = await self._generate(
                    request_id=request_id,
                    image=image,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                )
            else:
                logger.info(
                    (
                        "Analyze request %s waiting for inference lock "
                        "engine_ready=%s device=%s"
                    ),
                    request_id,
                    self.engine.ready,
                    self.engine.device,
                )
                async with self._inference_lock:
                    result = await self._generate(
                        request_id=request_id,
                        image=image,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                    )

            stripped_result = result.strip()
            logger.info(
                "Analyze request %s sending success response result=%r total_elapsed_ms=%.1f",
                request_id,
                stripped_result,
                _elapsed_ms(started_at),
            )
            return AnalyzeResponse(success=True, result=stripped_result)
        except Exception as exc:
            logger.exception(
                "Analyze request %s failed during inference after %.1fms",
                request_id,
                _elapsed_ms(started_at),
            )
            error = str(exc)
            logger.info(
                "Analyze request %s sending error response error=%r total_elapsed_ms=%.1f",
                request_id,
                error,
                _elapsed_ms(started_at),
            )
            return AnalyzeResponse(success=False, error=error)

    async def _generate(
        self,
        *,
        request_id: str,
        image,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        logger.info(
            (
                "Analyze request %s sending image to model "
                "size=%sx%s prompt_chars=%d max_new_tokens=%d"
            ),
            request_id,
            image.width,
            image.height,
            len(prompt),
            max_new_tokens,
        )
        result = await self.engine.generate(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        logger.info(
            "Analyze request %s model processed image result_chars=%d",
            request_id,
            len(result),
        )
        return result


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000


# Backward-compatible name for older tests/imports.
QwenAnalyzeService = LocalVisionAnalyzeService
