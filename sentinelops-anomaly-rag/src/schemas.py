"""
sentinelops/src/schemas.py
Pydantic v2 schemas for SentinelOps FastAPI service.
STEP 7.1
Raw telemetry field names match src/features.py base columns exactly.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class TelemetryRecord(BaseModel):
    """One raw telemetry row - same shape as data/raw/telemetry.parquet."""
    service_name: str
    timestamp: datetime
    cpu_pct: float
    memory_pct: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    error_rate_pct: float
    queue_depth: float
    thread_count: float


class DetectRequest(BaseModel):
    """
    Send >= 13 consecutive rows per service_name (12 = WINDOW_ROWS rolling
    window + 1 current row) for accurate rolling z-score / rate-of-change
    features. Fewer rows still work but rolling stats will be less reliable.
    """
    records: list[TelemetryRecord] = Field(..., min_length=1)


class ModelVote(BaseModel):
    model_name: str
    score: float
    threshold: float
    is_anomaly: bool


class DetectResult(BaseModel):
    service_name: str
    timestamp: datetime
    is_anomaly: bool
    votes: list[ModelVote]


class DetectResponse(BaseModel):
    results: list[DetectResult]
    anomalies_found: int


class RemediateRequest(BaseModel):
    anomaly_description: str = Field(
        ..., description="Natural-language description, e.g. built from a /detect hit"
    )


class RemediateResponse(BaseModel):
    grounding_confidence: str
    top_similarity_score: float
    result: dict
    retrieved_sources: list[dict]
