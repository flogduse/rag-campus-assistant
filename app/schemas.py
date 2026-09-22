"""Pydantic models for API requests/responses and internal chunks."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Source(BaseModel):
    """One retrieved chunk, exposed to the client for citations."""
    index: int = Field(..., description="Citation number, e.g. 1 for [1]")
    source: str = Field(..., description="Document title/filename")
    page: int | None = Field(None, description="Page number when available")
    score: float = Field(..., description="Similarity score (higher = closer)")
    text: str = Field(..., description="The chunk text (truncated in UI)")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str | None
    chunks: int
    docs: int
