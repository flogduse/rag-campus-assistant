"""FastAPI application: chat (streaming), health, re-ingest, static UI."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config, ingest, llm, store
from .schemas import ChatRequest, HealthResponse

app = FastAPI(title="RAG Campus Assistant", version="1.0.0")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


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


@app.post("/api/chat")
def chat(body: ChatRequest) -> StreamingResponse:
    """Server-Sent Events stream: sources first, then answer tokens, then done.

    Event protocol (one JSON object per `data:` line):
      {"type": "sources", "sources": [...]}
      {"type": "token",   "t": "..."}
      {"type": "done"}
    """
    try:
        hits = llm.retrieve(body.question)
    except Exception as exc:  # store/model errors surface as 500 with detail
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    # Wire contract (schemas.Source): source/page/score live at the TOP level,
    # not nested under metadata — the UI and tests rely on this.
    sources_out = [
        {
            "index": h["index"],
            "source": h["metadata"].get("source", "unknown"),
            "page": h["metadata"].get("page"),
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
                for token in llm.stream_answer(body.question, hits):
                    yield f"data: {json.dumps({'type': 'token', 't': token})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'token', 't': f'[stream error: {exc}]'})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/ingest")
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
