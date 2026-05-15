from qwen_service.config import Settings


def test_clamp_max_new_tokens_uses_default():
    settings = Settings(default_max_new_tokens=42, max_new_tokens_limit=100)

    assert settings.clamp_max_new_tokens(None) == 42


def test_clamp_max_new_tokens_caps_request():
    settings = Settings(default_max_new_tokens=42, max_new_tokens_limit=100)

    assert settings.clamp_max_new_tokens(500) == 100


def test_clamp_max_new_tokens_raises_low_values_to_one():
    settings = Settings(default_max_new_tokens=42, max_new_tokens_limit=100)

    assert settings.clamp_max_new_tokens(0) == 1
