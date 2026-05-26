import pytest

from qwen_service.config import Settings
from qwen_service.engines.transformers_janus import TransformersJanusEngine
from qwen_service.engines.transformers_qwen import TransformersQwenEngine
from qwen_service.engines.transformers_vision import TransformersVisionLanguageEngine
from qwen_service.engines.vllm_vision import VllmVisionLanguageEngine
from qwen_service.main import create_default_engine
from qwen_service.providers import get_provider


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


def test_create_default_engine_uses_janus_transformers_backend():
    engine = create_default_engine(
        Settings(
            backend="transformers",
            model_family="janus",
            model_id="deepseek-ai/Janus-Pro-1B",
            trust_remote_code=True,
        )
    )

    assert isinstance(engine, TransformersJanusEngine)


def test_janus_vllm_combination_fails_clearly():
    with pytest.raises(ValueError, match="Janus-Pro.*transformers"):
        create_default_engine(
            Settings(
                backend="vllm",
                model_family="janus",
                model_id="deepseek-ai/Janus-Pro-1B",
                trust_remote_code=True,
            )
        )


def test_provider_registry_selects_expected_provider_ids():
    assert get_provider(Settings(model_family="qwen2_5_vl")).id == "qwen2_5_vl"
    assert get_provider(Settings(model_family="auto")).id == "auto"
    assert get_provider(Settings(model_family="janus", trust_remote_code=True)).id == "janus"


def test_janus_requires_trust_remote_code():
    with pytest.raises(ValueError, match="LOCAL_VISION_TRUST_REMOTE_CODE=true"):
        create_default_engine(
            Settings(
                backend="transformers",
                model_family="janus",
                model_id="deepseek-ai/Janus-Pro-1B",
            )
        )
