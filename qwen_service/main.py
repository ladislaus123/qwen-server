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

logger = logging.getLogger(__name__)


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
    logger.info(
        (
            "Creating Qwen app service=%s model_id=%s backend=%s "
            "device_policy=%s load_model_on_startup=%s log_level=%s"
        ),
        resolved_settings.service_name,
        resolved_settings.model_id,
        resolved_settings.backend,
        resolved_settings.device_policy,
        should_load,
        resolved_settings.log_level,
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
            logger.info("Loading model during startup")
            try:
                await resolved_engine.load()
            except Exception:
                logger.exception("Model startup load failed")
                raise
            logger.info(
                "Model startup load complete ready=%s device=%s",
                resolved_engine.ready,
                resolved_engine.device,
            )
        else:
            logger.info("Skipping model startup load")

        try:
            yield
        finally:
            logger.info("Closing Qwen engine")
            await resolved_engine.close()
            logger.info("Qwen engine closed")

    app = FastAPI(
        title="Qwen2.5-VL Local Service",
        description="Janus-compatible local ROI analysis service backed by Qwen2.5-VL.",
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
