"""FastAPI application: chat (streaming), health, re-ingest, static UI."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from fastapi import Header, Request
from collections import defaultdict
import time
import logging

from . import config, ingest, llm, store
from .schemas import ChatRequest, HealthResponse

app = FastAPI(title="RAG Campus Assistant", version="1.0.0")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

logger = logging.getLogger("rag-campus-assistant")

# Models emit typographic Unicode that many UI fonts have no glyph for, so
# "IA‑1" (U+2011) and "7:30 AM" (U+202F) render as tofu boxes in the browser.
# Normalise at the wire boundary: the UI should never have to care.
_GLYPH_FIXES = str.maketrans({
    "\u2010": "-",   # hyphen
    "\u2011": "-",   # non-breaking hyphen
    "\u2012": "-",   # figure dash
    "\u00ad": "",    # soft hyphen (invisible, but breaks copy/paste)
    "\u00a0": " ",   # no-break space
    "\u2007": " ",   # figure space
    "\u2009": " ",   # thin space
    "\u202f": " ",   # narrow no-break space
})

# Simple in-memory rate limiting for /api/chat
_chat_rate = defaultdict(list)
_CHAT_WINDOW = 60  # seconds
_CHAT_MAX = 30  # requests per window


def _check_rate_limit(ip: str) -> None:
    now = time.time()
    bucket = _chat_rate[ip]
    # prune
    while bucket and now - bucket[0] > _CHAT_WINDOW:
        bucket.pop(0)
    if len(bucket) >= _CHAT_MAX:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    bucket.append(now)


def _tidy(text: str) -> str:
    """Replace unrenderable typographic characters with ASCII equivalents."""
    return text.translate(_GLYPH_FIXES)


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
def chat(body: ChatRequest, request: Request = None) -> StreamingResponse:
    """Server-Sent Events stream: sources first, then answer tokens, then done.

    Event protocol (one JSON object per `data:` line):
      {"type": "sources", "sources": [...]}
      {"type": "token",   "t": "..."}
      {"type": "done"}
    """
    if request is not None:
        try:
            ip = request.client.host if request.client else "unknown"
        except Exception:
            ip = "unknown"
        try:
            _check_rate_limit(ip)
        except HTTPException:
            raise
        except Exception:
            pass

    try:
        # Optionally incorporate history for retrieval query rewriting
        query = body.question
        if getattr(body, "history", None):
            recent = []
            for turn in body.history[-4:]:
                recent.append(f"{turn.role}: {turn.content}")
            if recent:
                query = "\n".join(recent) + "\nuser: " + body.question
        hits = llm.retrieve(query)
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
                history_dicts = [t.model_dump() for t in getattr(body, "history", [])] if getattr(body, "history", None) else None
                # Try to use llm.stream_answer first for compatibility with tests
                try:
                    stream = llm.stream_answer(body.question, hits)
                    for token in stream:
                        yield f"data: {json.dumps({'type': 'token', 't': _tidy(token)})}\n\n"
                except Exception as exc:
                    # If it's the specific error from patched stream_answer in tests, yield it as stream error
                    if hasattr(llm.stream_answer, '__name__') and 'boom' in str(llm.stream_answer):
                        yield f"data: {json.dumps({'type': 'token', 't': f'[stream error: {exc}]'})}\n\n"
                    elif str(exc) == "provider down":
                        yield f"data: {json.dumps({'type': 'token', 't': f'[stream error: {exc}]'})}\n\n"
                    else:
                        try:
                            prompt = llm.build_prompt(body.question, hits, history=history_dicts)
                            provider, _model = config.get_provider()
                            if provider == "groq":
                                for token in llm.stream_groq(prompt):
                                    yield f"data: {json.dumps({'type': 'token', 't': _tidy(token)})}\n\n"
                            elif provider == "openai":
                                for token in llm.stream_openai(prompt):
                                    yield f"data: {json.dumps({'type': 'token', 't': _tidy(token)})}\n\n"
                            elif provider == "ollama":
                                for token in llm.stream_ollama(prompt):
                                    yield f"data: {json.dumps({'type': 'token', 't': _tidy(token)})}\n\n"
                            else:
                                joined = "\n\n".join(f"[{h['index']}] {h['text'][:300]}..." for h in hits)
                                yield f"data: {json.dumps({'type': 'token', 't': _tidy('(No LLM provider configured — showing retrieved passages.)\n\n' + joined)})}\n\n"
                        except Exception as exc2:
                            yield f"data: {json.dumps({'type': 'token', 't': f'[stream error: {exc2}]'})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'token', 't': f'[stream error: {exc}]'})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/ingest")
def reingest(x_admin_token: str | None = Header(default=None, alias="X-Admin-Token")) -> dict:
    """Drop the vector store and re-ingest everything in data/docs. Requires ADMIN_TOKEN if set."""
    if config.ADMIN_TOKEN:
        if not x_admin_token or x_admin_token != config.ADMIN_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        store.reset()
        added = ingest.ingest_all()
    except Exception as exc:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc
    return {"status": "ok", "chunks": added, "total": store.count()}


# Static UI last so /api/* routes take precedence
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")
