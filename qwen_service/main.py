"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from qwen_service import __version__
from qwen_service.config import Settings, get_settings
from qwen_service.engines.base import VisionLanguageEngine
from qwen_service.engines.transformers_qwen import TransformersQwenEngine
from qwen_service.routes import router
from qwen_service.service import QwenAnalyzeService


def create_app(
    *,
    settings: Settings | None = None,
    engine: VisionLanguageEngine | None = None,
    load_model_on_startup: bool | None = None,
) -> FastAPI:
    """Create the FastAPI app with injectable settings and engine for tests."""
    resolved_settings = settings or get_settings()
    resolved_engine = engine or TransformersQwenEngine(resolved_settings)
    should_load = (
        resolved_settings.load_model_on_startup
        if load_model_on_startup is None
        else load_model_on_startup
    )

    logging.basicConfig(
        level=resolved_settings.log_level.upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = resolved_settings
        app.state.engine = resolved_engine
        app.state.analyze_service = QwenAnalyzeService(
            engine=resolved_engine,
            settings=resolved_settings,
        )

        if should_load:
            await resolved_engine.load()

        try:
            yield
        finally:
            await resolved_engine.close()

    app = FastAPI(
        title="Qwen2.5-VL Local Service",
        description="Janus-compatible local ROI analysis service backed by Qwen2.5-VL.",
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
