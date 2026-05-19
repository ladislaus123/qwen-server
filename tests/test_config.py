from qwen_service.config import Settings, get_settings


def test_clamp_max_new_tokens_uses_default():
    settings = Settings(default_max_new_tokens=42, max_new_tokens_limit=100)

    assert settings.clamp_max_new_tokens(None) == 42


def test_clamp_max_new_tokens_caps_request():
    settings = Settings(default_max_new_tokens=42, max_new_tokens_limit=100)

    assert settings.clamp_max_new_tokens(500) == 100


def test_clamp_max_new_tokens_raises_low_values_to_one():
    settings = Settings(default_max_new_tokens=42, max_new_tokens_limit=100)

    assert settings.clamp_max_new_tokens(0) == 1


def test_local_vision_env_names_are_preferred(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("QWEN_MODEL_ID", "legacy/model")
    monkeypatch.setenv("LOCAL_VISION_MODEL_ID", "local/model")
    monkeypatch.setenv("LOCAL_VISION_MODEL_FAMILY", "auto")

    settings = get_settings()

    assert settings.model_id == "local/model"
    assert settings.model_family == "auto"
    get_settings.cache_clear()
