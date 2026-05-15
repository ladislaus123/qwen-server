"""Environment-backed settings for the Qwen service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int, minimum: int | None = None) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    service_name: str = "qwen-vl-service"
    host: str = "0.0.0.0"
    port: int = 6000
    reload: bool = False

    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    backend: str = "transformers"
    device_policy: str = "auto"
    load_model_on_startup: bool = True

    default_max_new_tokens: int = 100
    max_new_tokens_limit: int = 256
    max_image_pixels: int = 16_777_216
    max_image_bytes: int = 20 * 1024 * 1024

    qwen_min_pixels: int | None = 256 * 28 * 28
    qwen_max_pixels: int | None = 1280 * 28 * 28
    use_flash_attention: bool = False

    log_level: str = "info"

    def clamp_max_new_tokens(self, requested: int | None) -> int:
        """Return a safe generation token count for the current request."""
        if requested is None:
            requested = self.default_max_new_tokens
        requested = max(1, requested)
        return min(requested, self.max_new_tokens_limit)


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process."""
    load_dotenv()

    device_policy = os.getenv("QWEN_DEVICE", "auto").strip().lower()
    if device_policy not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("QWEN_DEVICE must be one of: auto, cuda, mps, cpu")

    default_max_new_tokens = _get_int("QWEN_DEFAULT_MAX_NEW_TOKENS", 100, minimum=1)
    max_new_tokens_limit = _get_int("QWEN_MAX_NEW_TOKENS_LIMIT", 256, minimum=1)
    if default_max_new_tokens > max_new_tokens_limit:
        default_max_new_tokens = max_new_tokens_limit

    return Settings(
        service_name=os.getenv("QWEN_SERVICE_NAME", "qwen-vl-service"),
        host=os.getenv("QWEN_HOST", "0.0.0.0"),
        port=_get_int("QWEN_PORT", 6000, minimum=1),
        reload=_get_bool("QWEN_RELOAD", False),
        model_id=os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct"),
        backend=os.getenv("QWEN_BACKEND", "transformers"),
        device_policy=device_policy,
        load_model_on_startup=_get_bool("QWEN_LOAD_MODEL_ON_STARTUP", True),
        default_max_new_tokens=default_max_new_tokens,
        max_new_tokens_limit=max_new_tokens_limit,
        max_image_pixels=_get_int("QWEN_MAX_IMAGE_PIXELS", 16_777_216, minimum=1),
        max_image_bytes=_get_int("QWEN_MAX_IMAGE_BYTES", 20 * 1024 * 1024, minimum=1),
        qwen_min_pixels=_get_int("QWEN_MIN_PIXELS", 256 * 28 * 28, minimum=1),
        qwen_max_pixels=_get_int("QWEN_MAX_PIXELS", 1280 * 28 * 28, minimum=1),
        use_flash_attention=_get_bool("QWEN_USE_FLASH_ATTENTION", False),
        log_level=os.getenv("QWEN_LOG_LEVEL", "info"),
    )
