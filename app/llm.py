"""LLM layer with three providers behind one interface.

Priority: Groq (free tier, needs only an API key) -> OpenAI -> local Ollama.
All three stream via Server-Sent Events so the UI can render tokens live.
If no provider is available, retrieval-only mode still works: the API returns
the raw passages so you can verify the RAG core before wiring a key.
"""
from __future__ import annotations

import json
from collections.abc import Iterator

import requests

from . import config, store


def _context_block(text: str, index: int, meta: dict) -> str:
    src = meta.get("source", "unknown")
    page = meta.get("page")
    page_s = f", p.{page}" if page not in (None, "") else ""
    return f"[{index}] ({src}{page_s})\n{text}"


def build_prompt(question: str, hits: list[dict], history: list[dict] | None = None) -> str:
    blocks = "\n\n".join(
        _context_block(h["text"], i + 1, h["metadata"]) for i, h in enumerate(hits)
    )
    history_text = ""
    if history:
        parts = []
        for turn in history[-6:]:  # keep last few turns
            role = turn.get("role", "user")
            content = turn.get("content", "").strip()
            if not content:
                continue
            parts.append(f"{role.upper()}: {content}")
        if parts:
            history_text = "Conversation history:\n" + "\n".join(parts) + "\n\n---\n\n"
    return (
        f"{history_text}"
        f"Context:\n{blocks}\n\n---\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above. Cite sources as [1], [2]. "
        "If the context is insufficient, say so honestly."
    )


def retrieve(question: str) -> list[dict]:
    """Embed the query, fetch FETCH_K candidates, then greedy-MMR down to TOP_K.

    MMR (maximal marginal relevance) keeps results relevant *and* diverse:
    score = query similarity - lambda * redundancy with already-picked chunks.
    Redundancy uses a cheap Jaccard overlap on word sets — no re-encoding.
    Drop hits below MIN_SCORE to avoid feeding irrelevant context.
    """
    from .ingest import get_model

    vec = get_model().encode([question], normalize_embeddings=True)[0].tolist()
    raw = store.query("", vec, config.FETCH_K)

    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    dists = raw.get("distances", [[]])[0]
    n = len(docs)
    if n == 0:
        return []

    toks = [set(d.lower().split()) for d in docs]

    def jaccard(a: set[str], b: set[str]) -> float:
        union = a | b
        return len(a & b) / len(union) if union else 1.0

    selected: list[int] = []
    for _ in range(n):
        best_i, best_score = None, float("-inf")
        for i in range(n):
            if i in selected:
                continue
            redundancy = max(
                (jaccard(toks[i], toks[j]) for j in selected), default=0.0
            )
            score = (1 - dists[i]) - 0.3 * redundancy
            if score > best_score:
                best_i, best_score = i, score
        if best_i is None:
            break
        selected.append(best_i)

    hits = []
    for rank, i in enumerate(selected, start=1):
        sim = round(1 - dists[i], 4)
        if sim < config.MIN_SCORE:
            continue  # drop irrelevant chunks
        hits.append({
            "index": len(hits) + 1,
            "text": docs[i],
            "metadata": metas[i],
            "score": sim,
        })
        if len(hits) >= config.TOP_K:
            break
    return hits


def _sse_stream(resp: requests.Response) -> Iterator[str]:
    # Force UTF-8. requests falls back to ISO-8859-1 when an SSE response has no
    # charset parameter, which turns multi-byte characters into mojibake
    # (a narrow no-break space renders as "a€¯").
    resp.encoding = "utf-8"
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        yield payload


def stream_groq(prompt: str) -> Iterator[str]:
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
        json={
            "model": config.GROQ_MODEL,
            "messages": [
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": config.LLM_TEMP,
            "max_tokens": config.LLM_MAX_TOKENS,
            "stream": True,
        },
        stream=True,
        timeout=90,
    )
    r.raise_for_status()
    for payload in _sse_stream(r):
        delta = json.loads(payload)["choices"][0]["delta"]
        if delta.get("content"):
            yield delta["content"]


def stream_openai(prompt: str) -> Iterator[str]:
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
        json={
            "model": config.OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": config.LLM_TEMP,
            "max_tokens": config.LLM_MAX_TOKENS,
            "stream": True,
        },
        stream=True,
        timeout=90,
    )
    r.raise_for_status()
    for payload in _sse_stream(r):
        delta = json.loads(payload)["choices"][0]["delta"]
        if delta.get("content"):
            yield delta["content"]


def stream_ollama(prompt: str) -> Iterator[str]:
    # Ollama /api/generate streams NDJSON, not SSE
    r = requests.post(
        f"{config.OLLAMA_URL}/api/generate",
        json={
            "model": config.OLLAMA_MODEL,
            "prompt": f"{config.SYSTEM_PROMPT}\n\n{prompt}",
            "stream": True,
            "options": {"temperature": config.LLM_TEMP},
        },
        stream=True,
        timeout=180,
    )
    r.raise_for_status()
    r.encoding = "utf-8"  # same ISO-8859-1 default applies to NDJSON
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("response"):
            yield obj["response"]
        if obj.get("done"):
            break


def stream_answer(question: str, hits: list[dict]) -> Iterator[str]:
    provider, _model = config.get_provider()
    prompt = build_prompt(question, hits)
    if provider == "groq":
        return stream_groq(prompt)
    if provider == "openai":
        return stream_openai(prompt)
    if provider == "ollama":
        return stream_ollama(prompt)
    # No provider: retrieval-only fallback so the pipeline is still testable.
    joined = "\n\n".join(f"[{h['index']}] {h['text'][:300]}..." for h in hits)
    return iter([
        "(No LLM provider configured — showing retrieved passages.)\n\n" + joined
    ])
