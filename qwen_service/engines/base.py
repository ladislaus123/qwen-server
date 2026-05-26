"""Shared engine interface."""

from __future__ import annotations

from typing import Protocol

from PIL import Image


class VisionLanguageEngine(Protocol):
    """Minimal contract for swappable vision-language backends."""

    model_id: str

    @property
    def ready(self) -> bool:
        ...

    @property
    def device(self) -> str | None:
        ...

    @property
    def supports_concurrent_generation(self) -> bool:
        ...

    async def load(self) -> None:
        ...

    async def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        ...

    async def close(self) -> None:
        ...
