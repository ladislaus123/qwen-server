"""Shared provider interface for model-family engine selection."""

from __future__ import annotations

from typing import Protocol

from qwen_service.config import Settings
from qwen_service.engines.base import VisionLanguageEngine


class ModelProvider(Protocol):
    """Factory and validator for one model-family/backend combination set."""

    id: str
    supported_model_families: frozenset[str]
    supported_backends: frozenset[str]

    def validate(self, settings: Settings) -> None:
        ...

    def create_engine(self, settings: Settings) -> VisionLanguageEngine:
        ...
