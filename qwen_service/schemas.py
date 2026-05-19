"""Pydantic models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    image: str = Field(..., min_length=1, description="Raw base64 image or data URL")
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int | None = Field(default=None, ge=1)


class AnalyzeResponse(BaseModel):
    success: bool
    result: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    ready: bool
    model_id: str
    model_family: str | None = None
    backend: str
    device: str | None = None


class MetadataResponse(BaseModel):
    service: str
    model_id: str
    model_family: str | None = None
    backend: str
    analyze_api_compatible: bool = True
    endpoints: list[str]
