"""Environment-backed settings for the local vision model service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value
    return None


def _get_str(names: tuple[str, ...], default: str) -> str:
    value = _first_env(*names)
    if value is None:
        return default
    return value


def _get_bool(names: tuple[str, ...], default: bool) -> bool:
    value = _first_env(*names)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(names: tuple[str, ...], default: int, minimum: int | None = None) -> int:
    raw_value = _first_env(*names)
    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{names[0]} must be an integer") from exc

    if minimum is not None and value < minimum:
        raise ValueError(f"{names[0]} must be >= {minimum}")
    return value


def _get_optional_int(
    names: tuple[str, ...],
    default: int | None,
    minimum: int | None = None,
) -> int | None:
    raw_value = _first_env(*names)
    if raw_value is None:
        return default

    raw_value = raw_value.strip()
    if raw_value == "":
        return default
    if raw_value.lower() in {"none", "null"}:
        return None

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{names[0]} must be an integer") from exc

    if minimum is not None and value < minimum:
        raise ValueError(f"{names[0]} must be >= {minimum}")
    return value


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    service_name: str = "local-vision-service"
    host: str = "0.0.0.0"
    port: int = 6000
    reload: bool = False

    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_family: str = "qwen2_5_vl"
    auto_model_class: str = "auto_image_text_to_text"
    backend: str = "transformers"
    device_policy: str = "auto"
    load_model_on_startup: bool = True
    trust_remote_code: bool = False

    default_max_new_tokens: int = 100
    max_new_tokens_limit: int = 256
    max_image_pixels: int = 16_777_216
    max_image_bytes: int = 20 * 1024 * 1024

    processor_min_pixels: int | None = 256 * 28 * 28
    processor_max_pixels: int | None = 1280 * 28 * 28
    use_flash_attention: bool = False

    vllm_max_model_len: int | None = 4096
    vllm_max_num_seqs: int = 8
    vllm_max_concurrent_requests: int = 8
    vllm_tensor_parallel_size: int = 1

    log_level: str = "info"

    def clamp_max_new_tokens(self, requested: int | None) -> int:
        """Return a safe generation token count for the current request."""
        if requested is None:
            requested = self.default_max_new_tokens
        requested = max(1, requested)
        return min(requested, self.max_new_tokens_limit)

    @property
    def qwen_min_pixels(self) -> int | None:
        """Legacy attribute name retained for older imports."""
        return self.processor_min_pixels

    @property
    def qwen_max_pixels(self) -> int | None:
        """Legacy attribute name retained for older imports."""
        return self.processor_max_pixels


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process."""
    load_dotenv()

    device_policy = _get_str(("LOCAL_VISION_DEVICE", "QWEN_DEVICE"), "auto").strip().lower()
    if device_policy not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("LOCAL_VISION_DEVICE must be one of: auto, cuda, mps, cpu")

    model_family = _get_str(("LOCAL_VISION_MODEL_FAMILY", "QWEN_MODEL_FAMILY"), "qwen2_5_vl").strip().lower()
    if model_family not in {"qwen2_5_vl", "auto"}:
        raise ValueError("LOCAL_VISION_MODEL_FAMILY must be one of: qwen2_5_vl, auto")

    auto_model_class = _get_str(
        ("LOCAL_VISION_AUTO_MODEL_CLASS", "QWEN_AUTO_MODEL_CLASS"),
        "auto_image_text_to_text",
    ).strip().lower()
    if auto_model_class not in {"auto_image_text_to_text", "auto_vision2seq"}:
        raise ValueError(
            "LOCAL_VISION_AUTO_MODEL_CLASS must be one of: "
            "auto_image_text_to_text, auto_vision2seq"
        )

    backend = _get_str(("LOCAL_VISION_BACKEND", "QWEN_BACKEND"), "transformers").strip().lower()
    if backend not in {"transformers", "vllm"}:
        raise ValueError("LOCAL_VISION_BACKEND must be one of: transformers, vllm")

    default_max_new_tokens = _get_int(("LOCAL_VISION_DEFAULT_MAX_NEW_TOKENS", "QWEN_DEFAULT_MAX_NEW_TOKENS"), 100, minimum=1)
    max_new_tokens_limit = _get_int(("LOCAL_VISION_MAX_NEW_TOKENS_LIMIT", "QWEN_MAX_NEW_TOKENS_LIMIT"), 256, minimum=1)
    if default_max_new_tokens > max_new_tokens_limit:
        default_max_new_tokens = max_new_tokens_limit

    return Settings(
        service_name=_get_str(("LOCAL_VISION_SERVICE_NAME", "QWEN_SERVICE_NAME"), "local-vision-service"),
        host=_get_str(("LOCAL_VISION_HOST", "QWEN_HOST"), "0.0.0.0"),
        port=_get_int(("LOCAL_VISION_PORT", "QWEN_PORT"), 6000, minimum=1),
        reload=_get_bool(("LOCAL_VISION_RELOAD", "QWEN_RELOAD"), False),
        model_id=_get_str(("LOCAL_VISION_MODEL_ID", "QWEN_MODEL_ID"), "Qwen/Qwen2.5-VL-7B-Instruct"),
        model_family=model_family,
        auto_model_class=auto_model_class,
        backend=backend,
        device_policy=device_policy,
        load_model_on_startup=_get_bool(("LOCAL_VISION_LOAD_MODEL_ON_STARTUP", "QWEN_LOAD_MODEL_ON_STARTUP"), True),
        trust_remote_code=_get_bool(("LOCAL_VISION_TRUST_REMOTE_CODE", "QWEN_TRUST_REMOTE_CODE"), False),
        default_max_new_tokens=default_max_new_tokens,
        max_new_tokens_limit=max_new_tokens_limit,
        max_image_pixels=_get_int(("LOCAL_VISION_MAX_IMAGE_PIXELS", "QWEN_MAX_IMAGE_PIXELS"), 16_777_216, minimum=1),
        max_image_bytes=_get_int(("LOCAL_VISION_MAX_IMAGE_BYTES", "QWEN_MAX_IMAGE_BYTES"), 20 * 1024 * 1024, minimum=1),
        processor_min_pixels=_get_int(("LOCAL_VISION_MIN_PIXELS", "QWEN_MIN_PIXELS"), 256 * 28 * 28, minimum=1),
        processor_max_pixels=_get_int(("LOCAL_VISION_MAX_PIXELS", "QWEN_MAX_PIXELS"), 1280 * 28 * 28, minimum=1),
        use_flash_attention=_get_bool(("LOCAL_VISION_USE_FLASH_ATTENTION", "QWEN_USE_FLASH_ATTENTION"), False),
        vllm_max_model_len=_get_optional_int(
            ("LOCAL_VISION_VLLM_MAX_MODEL_LEN", "QWEN_VLLM_MAX_MODEL_LEN"),
            4096,
            minimum=1,
        ),
        vllm_max_num_seqs=_get_int(
            ("LOCAL_VISION_VLLM_MAX_NUM_SEQS", "QWEN_VLLM_MAX_NUM_SEQS"),
            8,
            minimum=1,
        ),
        vllm_max_concurrent_requests=_get_int(
            (
                "LOCAL_VISION_VLLM_MAX_CONCURRENT_REQUESTS",
                "QWEN_VLLM_MAX_CONCURRENT_REQUESTS",
            ),
            8,
            minimum=1,
        ),
        vllm_tensor_parallel_size=_get_int(
            (
                "LOCAL_VISION_VLLM_TENSOR_PARALLEL_SIZE",
                "QWEN_VLLM_TENSOR_PARALLEL_SIZE",
            ),
            1,
            minimum=1,
        ),
        log_level=_get_str(("LOCAL_VISION_LOG_LEVEL", "QWEN_LOG_LEVEL"), "info"),
    )
