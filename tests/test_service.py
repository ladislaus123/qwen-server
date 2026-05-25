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


class BlockingFakeEngine:
    model_id = "fake-model"
    ready = True
    device = "test"

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.first_entered = asyncio.Event()
        self.two_active = asyncio.Event()
        self.release = asyncio.Event()

    async def load(self):
        return None

    async def close(self):
        return None

    async def generate(self, image, prompt, max_new_tokens):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.first_entered.set()
        if self.active >= 2:
            self.two_active.set()

        try:
            await self.release.wait()
            return "72.5"
        finally:
            self.active -= 1


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


def test_vllm_backend_allows_concurrent_analysis_requests():
    async def run():
        engine = BlockingFakeEngine()
        service = LocalVisionAnalyzeService(
            engine=engine,
            settings=Settings(backend="vllm"),
        )
        request = AnalyzeRequest(image=_png_base64(), prompt="read")

        first = asyncio.create_task(service.analyze(request))
        second = asyncio.create_task(service.analyze(request))
        await asyncio.wait_for(engine.two_active.wait(), timeout=1)

        engine.release.set()
        responses = await asyncio.gather(first, second)

        assert engine.max_active == 2
        assert [response.success for response in responses] == [True, True]

    asyncio.run(run())


def test_transformers_backend_serializes_analysis_requests():
    async def run():
        engine = BlockingFakeEngine()
        service = LocalVisionAnalyzeService(
            engine=engine,
            settings=Settings(backend="transformers"),
        )
        request = AnalyzeRequest(image=_png_base64(), prompt="read")

        first = asyncio.create_task(service.analyze(request))
        second = asyncio.create_task(service.analyze(request))
        await asyncio.wait_for(engine.first_entered.wait(), timeout=1)
        await asyncio.sleep(0.05)

        assert engine.max_active == 1

        engine.release.set()
        responses = await asyncio.gather(first, second)

        assert engine.max_active == 1
        assert [response.success for response in responses] == [True, True]

    asyncio.run(run())
