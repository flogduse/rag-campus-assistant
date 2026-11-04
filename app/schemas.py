"""Pydantic models for API requests/responses and internal chunks."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Source(BaseModel):
    """One retrieved chunk, exposed to the client for citations."""
    index: int = Field(..., description="Citation number, e.g. 1 for [1]")
    source: str = Field(..., description="Document title/filename")
    page: int | None = Field(None, description="Page number when available")
    score: float = Field(..., description="Similarity score (higher = closer)")
    text: str = Field(..., description="The chunk text (truncated in UI)")


class Turn(BaseModel):
    """One earlier message, sent by the client so follow-ups keep context."""
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    # Optional recent turns; the backend trims to config.HISTORY_TURNS.
    history: list[Turn] = Field(default_factory=list, max_length=20)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str | None
    chunks: int
    docs: int
