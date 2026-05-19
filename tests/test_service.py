import base64
import asyncio
from io import BytesIO

from PIL import Image
import pytest

from qwen_service.config import Settings
from qwen_service.schemas import AnalyzeRequest
from qwen_service.service import AnalyzeInputError, LocalVisionAnalyzeService


class FakeEngine:
    model_id = "fake-model"
    ready = True
    device = "test"

    def __init__(self):
        self.calls = []

    async def load(self):
        return None

    async def close(self):
        return None

    async def generate(self, image, prompt, max_new_tokens):
        self.calls.append((image.size, prompt, max_new_tokens))
        return "72.5"


def _png_base64():
    image = Image.new("RGB", (4, 3), color=(0, 0, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_service_calls_engine_with_decoded_image_and_capped_tokens():
    engine = FakeEngine()
    settings = Settings(default_max_new_tokens=100, max_new_tokens_limit=128)
    service = LocalVisionAnalyzeService(engine=engine, settings=settings)

    response = asyncio.run(
        service.analyze(
            AnalyzeRequest(
                image=_png_base64(),
                prompt="  read it  ",
                max_new_tokens=999,
            )
        )
    )

    assert response.success is True
    assert response.result == "72.5"
    assert engine.calls == [((4, 3), "read it", 128)]


def test_service_raises_input_error_for_bad_image():
    engine = FakeEngine()
    service = LocalVisionAnalyzeService(engine=engine, settings=Settings())

    with pytest.raises(AnalyzeInputError):
        asyncio.run(service.analyze(AnalyzeRequest(image="bad", prompt="read")))
