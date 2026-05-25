import asyncio
import os
import sys
import types
from types import SimpleNamespace

import pytest
from PIL import Image

from qwen_service.config import Settings
from qwen_service.engines.vllm_vision import VllmVisionLanguageEngine


@pytest.fixture
def fake_vllm_modules(monkeypatch):
    state = SimpleNamespace(
        engine_args=None,
        llm=None,
        processor_load=None,
        chat_messages=None,
        block_generate=False,
        blocker=None,
    )

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            state.processor_load = (model_id, kwargs)
            return cls()

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            state.chat_messages = messages
            assert tokenize is False
            assert add_generation_prompt is True
            return "templated prompt"

    class FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            state.engine_args = self

    class FakeTextOutput:
        def __init__(self, text):
            self.text = text

    class FakeRequestOutput:
        def __init__(self, text):
            self.outputs = [FakeTextOutput(text)]

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeRequestOutputKind:
        FINAL_ONLY = "final_only"

    class FakeAsyncLLM:
        def __init__(self):
            self.requests = []
            self.abort_calls = []
            self.shutdown_background_loop_called = False
            self.started = asyncio.Event()

        async def generate(self, prompt, sampling_params, request_id):
            self.requests.append(
                {
                    "prompt": prompt,
                    "sampling_params": sampling_params,
                    "request_id": request_id,
                }
            )
            self.started.set()

            if state.block_generate:
                await state.blocker.wait()

            yield FakeRequestOutput(" 72.5 ")

        async def abort(self, request_id):
            self.abort_calls.append(request_id)

        def shutdown_background_loop(self):
            self.shutdown_background_loop_called = True

    class FakeAsyncLLMEngine:
        @classmethod
        def from_engine_args(cls, engine_args):
            state.llm = FakeAsyncLLM()
            return state.llm

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = FakeAutoProcessor

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.AsyncEngineArgs = FakeAsyncEngineArgs
    fake_vllm.AsyncLLMEngine = FakeAsyncLLMEngine
    fake_vllm.SamplingParams = FakeSamplingParams

    fake_sampling_params = types.ModuleType("vllm.sampling_params")
    fake_sampling_params.RequestOutputKind = FakeRequestOutputKind

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", fake_sampling_params)

    return state


def test_vllm_engine_loads_async_engine_args(fake_vllm_modules):
    async def run():
        settings = Settings(
            backend="vllm",
            model_id="test/model",
            trust_remote_code=True,
            processor_min_pixels=11,
            processor_max_pixels=22,
            vllm_max_model_len=1024,
            vllm_max_num_seqs=8,
            vllm_max_concurrent_requests=3,
            vllm_tensor_parallel_size=2,
            vllm_gpu_memory_utilization=0.7,
            vllm_cpu_offload_gb=4,
            vllm_dtype="float16",
            vllm_quantization="bitsandbytes",
        )
        engine = VllmVisionLanguageEngine(settings)

        await engine.load()

        assert fake_vllm_modules.processor_load == (
            "test/model",
            {"trust_remote_code": True},
        )
        assert fake_vllm_modules.engine_args.kwargs == {
            "model": "test/model",
            "trust_remote_code": True,
            "limit_mm_per_prompt": {"image": 1},
            "max_num_seqs": 8,
            "gpu_memory_utilization": 0.7,
            "cpu_offload_gb": 4,
            "max_model_len": 1024,
            "dtype": "float16",
            "quantization": "bitsandbytes",
            "tensor_parallel_size": 2,
            "mm_processor_kwargs": {
                "min_pixels": 11,
                "max_pixels": 22,
            },
        }

        await engine.close()
        assert fake_vllm_modules.llm.shutdown_background_loop_called is True

    asyncio.run(run())


def test_vllm_engine_honors_cuda_device_policy(fake_vllm_modules):
    async def run():
        engine = VllmVisionLanguageEngine(
            Settings(backend="vllm", device_policy="cuda")
        )

        await engine.load()

        assert fake_vllm_modules.engine_args.kwargs["device"] == "cuda"

        await engine.close()

    asyncio.run(run())


def test_vllm_engine_sets_target_device_before_import(monkeypatch, fake_vllm_modules):
    async def run():
        monkeypatch.delenv("VLLM_TARGET_DEVICE", raising=False)
        engine = VllmVisionLanguageEngine(
            Settings(backend="vllm", device_policy="cuda")
        )

        await engine.load()

        assert fake_vllm_modules.engine_args.kwargs["device"] == "cuda"
        assert os.environ["VLLM_TARGET_DEVICE"] == "cuda"

        await engine.close()

    asyncio.run(run())


def test_vllm_engine_generate_uses_multimodal_prompt_and_final_output(
    fake_vllm_modules,
):
    async def run():
        engine = VllmVisionLanguageEngine(Settings(backend="vllm"))
        image = Image.new("RGB", (4, 3), color=(255, 255, 255))

        await engine.load()
        result = await engine.generate(image=image, prompt="read", max_new_tokens=9)

        assert result == "72.5"
        assert fake_vllm_modules.chat_messages[0]["content"][0]["image"] is image
        request = fake_vllm_modules.llm.requests[0]
        assert request["prompt"] == {
            "prompt": "templated prompt",
            "multi_modal_data": {"image": image},
        }
        assert request["sampling_params"].kwargs == {
            "max_tokens": 9,
            "temperature": 0,
            "output_kind": "final_only",
        }

        await engine.close()

    asyncio.run(run())


def test_vllm_engine_cancellation_aborts_request(fake_vllm_modules):
    async def run():
        fake_vllm_modules.block_generate = True
        fake_vllm_modules.blocker = asyncio.Event()
        engine = VllmVisionLanguageEngine(Settings(backend="vllm"))
        image = Image.new("RGB", (4, 3), color=(255, 255, 255))

        await engine.load()
        task = asyncio.create_task(
            engine.generate(image=image, prompt="read", max_new_tokens=9)
        )
        await asyncio.wait_for(fake_vllm_modules.llm.started.wait(), timeout=1)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        request_id = fake_vllm_modules.llm.requests[0]["request_id"]
        assert fake_vllm_modules.llm.abort_calls == [request_id]

        await engine.close()

    asyncio.run(run())
