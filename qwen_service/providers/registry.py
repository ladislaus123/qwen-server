"""Provider registry for configured vision-language engines."""

from __future__ import annotations

from dataclasses import dataclass

from qwen_service.config import Settings
from qwen_service.engines.base import VisionLanguageEngine
from qwen_service.providers.base import ModelProvider


@dataclass(frozen=True)
class QwenProvider:
    id: str = "qwen2_5_vl"
    supported_model_families: frozenset[str] = frozenset({"qwen2_5_vl"})
    supported_backends: frozenset[str] = frozenset({"transformers", "vllm"})

    def validate(self, settings: Settings) -> None:
        _validate_backend(self, settings)

    def create_engine(self, settings: Settings) -> VisionLanguageEngine:
        self.validate(settings)
        if settings.backend.lower() == "vllm":
            from qwen_service.engines.vllm_vision import VllmVisionLanguageEngine

            return VllmVisionLanguageEngine(settings)

        from qwen_service.engines.transformers_qwen import TransformersQwenEngine

        return TransformersQwenEngine(settings)


@dataclass(frozen=True)
class AutoTransformersProvider:
    id: str = "auto"
    supported_model_families: frozenset[str] = frozenset({"auto"})
    supported_backends: frozenset[str] = frozenset({"transformers"})

    def validate(self, settings: Settings) -> None:
        _validate_backend(self, settings)

    def create_engine(self, settings: Settings) -> VisionLanguageEngine:
        self.validate(settings)
        from qwen_service.engines.transformers_vision import (
            TransformersVisionLanguageEngine,
        )

        return TransformersVisionLanguageEngine(settings)


@dataclass(frozen=True)
class JanusProvider:
    id: str = "janus"
    supported_model_families: frozenset[str] = frozenset({"janus"})
    supported_backends: frozenset[str] = frozenset({"transformers"})

    def validate(self, settings: Settings) -> None:
        _validate_backend(
            self,
            settings,
            extra=(
                "Janus-Pro is loaded through its Transformers/custom code path. "
                "Use LOCAL_VISION_BACKEND=transformers."
            ),
        )
        if not settings.trust_remote_code:
            raise ValueError(
                "Janus-Pro requires LOCAL_VISION_TRUST_REMOTE_CODE=true because "
                "it loads DeepSeek's custom Janus model code."
            )

    def create_engine(self, settings: Settings) -> VisionLanguageEngine:
        self.validate(settings)
        from qwen_service.engines.transformers_janus import TransformersJanusEngine

        return TransformersJanusEngine(settings)


_PROVIDERS: tuple[ModelProvider, ...] = (
    QwenProvider(),
    AutoTransformersProvider(),
    JanusProvider(),
)


def get_provider(settings: Settings) -> ModelProvider:
    """Return the provider for the configured model family."""
    family = settings.model_family.lower()
    for provider in _PROVIDERS:
        if family in provider.supported_model_families:
            provider.validate(settings)
            return provider

    supported = ", ".join(sorted(_supported_families()))
    raise ValueError(
        f"Unsupported model family: {settings.model_family}. "
        f"Supported model families: {supported}."
    )


def create_engine(settings: Settings) -> VisionLanguageEngine:
    """Create an engine from the configured provider."""
    return get_provider(settings).create_engine(settings)


def _validate_backend(
    provider: ModelProvider,
    settings: Settings,
    *,
    extra: str | None = None,
) -> None:
    backend = settings.backend.lower()
    if backend in provider.supported_backends:
        return

    supported = ", ".join(sorted(provider.supported_backends))
    message = (
        f"Unsupported backend {settings.backend!r} for model family "
        f"{settings.model_family!r}. Supported backend(s): {supported}."
    )
    if extra:
        message = f"{message} {extra}"
    raise ValueError(message)


def _supported_families() -> set[str]:
    families: set[str] = set()
    for provider in _PROVIDERS:
        families.update(provider.supported_model_families)
    return families
