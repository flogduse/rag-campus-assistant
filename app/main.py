"""FastAPI application: chat (streaming), health, re-ingest, static UI."""
from __future__ import annotations

import hmac
import json
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config, ingest, llm, store
from .schemas import ChatRequest, HealthResponse

app = FastAPI(title="RAG Campus Assistant", version="1.0.0")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# --- abuse protection -------------------------------------------------------
# A public demo spends someone's Groq quota per question and re-embeds the whole
# corpus on ingest, so both endpoints get a cheap in-memory guard. No Redis, no
# dependency: a dict of timestamps per client IP is enough for a single process.
_rate_lock = threading.Lock()
_rate_buckets: dict[str, list[float]] = {}


def rate_limit(request: Request) -> None:
    """Allow at most RATE_LIMIT chat requests per RATE_WINDOW seconds per IP."""
    if config.RATE_LIMIT <= 0:
        return
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _rate_lock:
        hits = [t for t in _rate_buckets.get(client, []) if now - t < config.RATE_WINDOW]
        if len(hits) >= config.RATE_LIMIT:
            retry_after = max(1, int(config.RATE_WINDOW - (now - hits[0])))
            raise HTTPException(
                status_code=429,
                detail="Too many questions — please wait a moment and try again.",
                headers={"Retry-After": str(retry_after)},
            )
        hits.append(now)
        _rate_buckets[client] = hits
        # Bound memory if many distinct clients appear.
        if len(_rate_buckets) > 10_000:
            for key in [k for k, v in _rate_buckets.items() if not v]:
                _rate_buckets.pop(key, None)


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    """Gate /api/ingest when ADMIN_TOKEN is configured.

    Unset token = local development default (open). Set one before deploying:
    an open ingest endpoint lets anyone wipe the index and burn CPU re-embedding.
    """
    if not config.ADMIN_TOKEN:
        return
    if not hmac.compare_digest(x_admin_token or "", config.ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Token")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    provider, model = config.get_provider()
    return HealthResponse(
        status="ok",
        provider=provider,
        model=model,
        chunks=store.count(),
        docs=store.unique_docs(),
    )


@app.post("/api/chat", dependencies=[Depends(rate_limit)])
def chat(body: ChatRequest) -> StreamingResponse:
    """Server-Sent Events stream: sources first, then answer tokens, then done.

    Event protocol (one JSON object per `data:` line):
      {"type": "sources", "sources": [...]}
      {"type": "token",   "t": "..."}
      {"type": "done"}
    """
    history = [turn.model_dump() for turn in body.history]
    try:
        hits = llm.retrieve(body.question, history)
    except Exception as exc:  # store/model errors surface as 500 with detail
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    # Wire contract (schemas.Source): source/page/score live at the TOP level,
    # not nested under metadata — the UI and tests rely on this.
    sources_out = [
        {
            "index": h["index"],
            "source": h["metadata"].get("source", "unknown"),
            # Stored as "" when unknown; the contract says int | None.
            "page": h["metadata"].get("page") or None,
            "score": h["score"],
            "text": h["text"],
        }
        for h in hits
    ]

    def generate():
        # 1) sources always go first so the UI can render citations instantly
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources_out})}\n\n"

        if not hits:
            yield f"data: {json.dumps({'type': 'token', 't': 'I could not find anything about that in the ingested documents. Try adding more files to data/docs and re-ingesting.'})}\n\n"
        else:
            try:
                for token in llm.stream_answer(body.question, hits, history):
                    yield f"data: {json.dumps({'type': 'token', 't': token})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'token', 't': f'[stream error: {exc}]'})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/ingest", dependencies=[Depends(require_admin)])
def reingest() -> dict:
    """Drop the vector store and re-ingest everything in data/docs."""
    try:
        store.reset()
        added = ingest.ingest_all()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc
    return {"status": "ok", "chunks": added, "total": store.count()}


# Static UI last so /api/* routes take precedence
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")
