import asyncio
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

from PIL import Image

from qwen_service.config import Settings
from qwen_service.engines.transformers_janus import TransformersJanusEngine


def test_janus_engine_loads_and_generates_with_custom_code(monkeypatch):
    state = SimpleNamespace(
        processor_load=None,
        model_load=None,
        model_to=[],
        model_eval=False,
        processor_call=None,
        inputs_to=None,
        prepare_kwargs=None,
        generate_kwargs=None,
        decode_call=None,
        cuda_cache_cleared=False,
    )

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_capability():
            return (8, 0)

        @staticmethod
        def empty_cache():
            state.cuda_cache_cleared = True

    class FakeMps:
        @staticmethod
        def is_available():
            return False

    @contextmanager
    def fake_inference_mode():
        yield

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.cuda = FakeCuda
    fake_torch.backends = SimpleNamespace(mps=FakeMps)
    fake_torch.inference_mode = fake_inference_mode

    class FakeTokenizer:
        eos_token_id = 2
        bos_token_id = 1

        def decode(self, output_ids, skip_special_tokens):
            state.decode_call = (output_ids, skip_special_tokens)
            return " 72.5 "

    class FakeJanusInputs(dict):
        @property
        def attention_mask(self):
            return self["attention_mask"]

        def to(self, device):
            state.inputs_to = device
            return self

    class FakeVLChatProcessor:
        tokenizer = FakeTokenizer()

        @classmethod
        def from_pretrained(cls, model_id):
            state.processor_load = model_id
            return cls()

        def __call__(self, *, conversations, images, force_batchify):
            state.processor_call = {
                "conversations": conversations,
                "images": images,
                "force_batchify": force_batchify,
            }
            return FakeJanusInputs(input_ids=[1, 2], attention_mask="mask")

    class FakeLanguageModel:
        def generate(self, **kwargs):
            state.generate_kwargs = kwargs
            return [[7, 2]]

    class FakeModel:
        def __init__(self):
            self.language_model = FakeLanguageModel()

        def to(self, target):
            state.model_to.append(target)
            return self

        def eval(self):
            state.model_eval = True
            return self

        def prepare_inputs_embeds(self, **kwargs):
            state.prepare_kwargs = kwargs
            return "embeds"

    class FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            state.model_load = (model_id, kwargs)
            return FakeModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM

    fake_janus = types.ModuleType("janus")
    fake_janus_models = types.ModuleType("janus.models")
    fake_janus_models.VLChatProcessor = FakeVLChatProcessor

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "janus", fake_janus)
    monkeypatch.setitem(sys.modules, "janus.models", fake_janus_models)

    async def run():
        engine = TransformersJanusEngine(
            Settings(
                backend="transformers",
                model_family="janus",
                model_id="deepseek-ai/Janus-Pro-1B",
                device_policy="cuda",
                trust_remote_code=True,
            )
        )
        image = Image.new("RGB", (4, 3), color=(255, 255, 255))

        await engine.load()
        result = await engine.generate(image=image, prompt="read", max_new_tokens=9)
        await engine.close()

        return result

    assert asyncio.run(run()) == "72.5"
    assert state.processor_load == "deepseek-ai/Janus-Pro-1B"
    assert state.model_load == (
        "deepseek-ai/Janus-Pro-1B",
        {"trust_remote_code": True},
    )
    assert state.model_to == ["bfloat16", "cuda"]
    assert state.model_eval is True
    conversation = state.processor_call["conversations"]
    assert conversation[0]["content"] == "<image_placeholder>\nread"
    assert conversation[0]["images"][0].mode == "RGB"
    assert state.processor_call["images"][0].mode == "RGB"
    assert state.processor_call["force_batchify"] is True
    assert state.inputs_to == "cuda"
    assert state.prepare_kwargs == {"input_ids": [1, 2], "attention_mask": "mask"}
    assert state.generate_kwargs == {
        "inputs_embeds": "embeds",
        "attention_mask": "mask",
        "pad_token_id": 2,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "max_new_tokens": 9,
        "do_sample": False,
        "use_cache": True,
    }
    assert state.decode_call == ([7, 2], True)
    assert state.cuda_cache_cleared is True
