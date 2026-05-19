"""HTTP routes for the local vision model service."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from qwen_service.config import Settings
from qwen_service.engines.base import VisionLanguageEngine
from qwen_service.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    MetadataResponse,
)
from qwen_service.service import AnalyzeInputError, LocalVisionAnalyzeService

router = APIRouter()
logger = logging.getLogger(__name__)


def get_settings_from_app(request: Request) -> Settings:
    return request.app.state.settings


def get_engine_from_app(request: Request) -> VisionLanguageEngine:
    return request.app.state.engine


def get_service_from_app(request: Request) -> LocalVisionAnalyzeService:
    return request.app.state.analyze_service


@router.get("/", response_model=MetadataResponse)
async def metadata(settings: Settings = Depends(get_settings_from_app)) -> MetadataResponse:
    return MetadataResponse(
        service=settings.service_name,
        model_id=settings.model_id,
        model_family=settings.model_family,
        backend=settings.backend,
        endpoints=["GET /", "GET /health", "POST /analyze"],
    )


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: Settings = Depends(get_settings_from_app),
    engine: VisionLanguageEngine = Depends(get_engine_from_app),
) -> HealthResponse:
    return HealthResponse(
        status="ok" if engine.ready else "starting",
        ready=engine.ready,
        model_id=settings.model_id,
        model_family=settings.model_family,
        backend=settings.backend,
        device=engine.device,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    payload: AnalyzeRequest,
    service: LocalVisionAnalyzeService = Depends(get_service_from_app),
) -> AnalyzeResponse:
    try:
        return await service.analyze(payload)
    except AnalyzeInputError as exc:
        logger.warning("Analyze request rejected with HTTP 400: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
