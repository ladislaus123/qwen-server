from qwen_service.config import Settings
from qwen_service.engines.transformers_qwen import TransformersQwenEngine
from qwen_service.engines.transformers_vision import TransformersVisionLanguageEngine
from qwen_service.engines.vllm_vision import VllmVisionLanguageEngine
from qwen_service.main import create_default_engine


def test_create_default_engine_uses_vllm_backend():
    engine = create_default_engine(Settings(backend="vllm"))

    assert isinstance(engine, VllmVisionLanguageEngine)


def test_create_default_engine_uses_qwen_transformers_backend():
    engine = create_default_engine(
        Settings(backend="transformers", model_family="qwen2_5_vl")
    )

    assert isinstance(engine, TransformersQwenEngine)


def test_create_default_engine_uses_auto_transformers_backend():
    engine = create_default_engine(Settings(backend="transformers", model_family="auto"))

    assert isinstance(engine, TransformersVisionLanguageEngine)
