"""Application orchestration for image analysis requests."""

from __future__ import annotations

import asyncio
import logging

from qwen_service.config import Settings
from qwen_service.engines.base import VisionLanguageEngine
from qwen_service.image_io import ImageDecodeError, decode_base64_image
from qwen_service.schemas import AnalyzeRequest, AnalyzeResponse

logger = logging.getLogger(__name__)


class AnalyzeInputError(ValueError):
    """Raised for request data that should become HTTP 400."""


class QwenAnalyzeService:
    """Coordinates decoding, token capping, and model generation."""

    def __init__(self, engine: VisionLanguageEngine, settings: Settings):
        self.engine = engine
        self.settings = settings
        self._inference_lock = asyncio.Lock()

    async def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        try:
            image = decode_base64_image(
                request.image,
                max_pixels=self.settings.max_image_pixels,
                max_bytes=self.settings.max_image_bytes,
            )
        except ImageDecodeError as exc:
            raise AnalyzeInputError(str(exc)) from exc

        max_new_tokens = self.settings.clamp_max_new_tokens(request.max_new_tokens)

        try:
            async with self._inference_lock:
                result = await self.engine.generate(
                    image=image,
                    prompt=request.prompt.strip(),
                    max_new_tokens=max_new_tokens,
                )
            return AnalyzeResponse(success=True, result=result.strip())
        except Exception as exc:
            logger.exception("Qwen inference failed")
            return AnalyzeResponse(success=False, error=str(exc))
