import base64
import os
from io import BytesIO

import pytest
from PIL import Image

from qwen_service.config import Settings
from qwen_service.engines.transformers_qwen import TransformersQwenEngine
from qwen_service.schemas import AnalyzeRequest
from qwen_service.service import LocalVisionAnalyzeService


def _png_base64():
    image = Image.new("RGB", (64, 32), color=(255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@pytest.mark.skipif(
    os.getenv("RUN_QWEN_MODEL_SMOKE") != "1",
    reason="set RUN_QWEN_MODEL_SMOKE=1 to download/load the Qwen model",
)
def test_full_model_smoke():
    import asyncio

    async def run():
        settings = Settings(default_max_new_tokens=8, max_new_tokens_limit=8)
        engine = TransformersQwenEngine(settings)
        await engine.load()
        try:
            service = LocalVisionAnalyzeService(engine=engine, settings=settings)
            response = await service.analyze(
                AnalyzeRequest(
                    image=_png_base64(),
                    prompt="Return the word white.",
                    max_new_tokens=8,
                )
            )
            assert response.success is True
            assert isinstance(response.result, str)
        finally:
            await engine.close()

    asyncio.run(run())
