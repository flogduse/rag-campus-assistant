"""LLM layer with three providers behind one interface.

Priority: Groq (free tier, needs only an API key) -> OpenAI -> local Ollama.
All three stream via Server-Sent Events so the UI can render tokens live.
If no provider is available, retrieval-only mode still works: the API returns
the raw passages so you can verify the RAG core before wiring a key.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterator

import requests

from . import config, store


def _context_block(text: str, index: int, meta: dict) -> str:
    src = meta.get("source", "unknown")
    page = meta.get("page")
    page_s = f", p.{page}" if page not in (None, "") else ""
    return f"[{index}] ({src}{page_s})\n{text}"


# Words that usually mean "same as the previous question" — a bare follow-up
# such as "when is it?" retrieves nothing useful on its own.
_FOLLOWUP_RE = re.compile(
    r"\b(it|its|it's|that|this|those|these|they|them|there|same|again)\b", re.I
)


def format_history(history: list[dict] | None, limit: int) -> str:
    """Render the last `limit` turns as a conversation block (may be empty)."""
    if not history:
        return ""
    recent = history[-limit:]
    lines = [f"{t['role'].upper()}: {t['content']}" for t in recent]
    return "Earlier conversation (context only — not a source to cite):\n" + "\n".join(lines) + "\n\n"


def retrieval_query(question: str, history: list[dict] | None) -> str:
    """Expand short/pronoun follow-ups with the previous user question.

    Embedding "when is it?" alone lands nowhere; prefixing the prior question
    keeps the lookup anchored without needing an extra LLM round-trip.
    """
    if not history:
        return question
    last_user = next(
        (t["content"] for t in reversed(history) if t["role"] == "user"), ""
    )
    if not last_user or last_user == question:
        return question
    words = question.split()
    if len(words) <= 6 or _FOLLOWUP_RE.search(question):
        return f"{last_user} {question}"
    return question


def build_prompt(question: str, hits: list[dict], history: list[dict] | None = None) -> str:
    blocks = "\n\n".join(
        _context_block(h["text"], i + 1, h["metadata"]) for i, h in enumerate(hits)
    )
    convo = format_history(history, config.HISTORY_TURNS)
    return (
        f"{convo}Context:\n{blocks}\n\n---\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above. Cite sources as [1], [2]. "
        "If the context is insufficient, say so honestly."
    )


def retrieve(question: str, history: list[dict] | None = None) -> list[dict]:
    """Embed the query, fetch FETCH_K candidates, then greedy-MMR down to TOP_K.

    MMR (maximal marginal relevance) keeps results relevant *and* diverse:
    score = query similarity - lambda * redundancy with already-picked chunks.
    Redundancy uses a cheap Jaccard overlap on word sets — no re-encoding.

    Hits below config.MIN_SCORE are discarded: an off-topic question should
    reach the "I could not find that" path instead of feeding the model four
    loosely related chunks it would then try to answer from.
    """
    from .ingest import get_model

    query = retrieval_query(question, history)
    vec = get_model().encode([query], normalize_embeddings=True)[0].tolist()
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
    for i in selected:
        score = round(1 - dists[i], 4)
        if score < config.MIN_SCORE:
            continue
        if len(hits) >= config.TOP_K:
            break
        # Re-number so citation markers stay contiguous after filtering.
        hits.append({
            "index": len(hits) + 1,
            "text": docs[i],
            "metadata": metas[i],
            "score": score,
        })
    return hits


def _sse_stream(resp: requests.Response) -> Iterator[str]:
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
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("response"):
            yield obj["response"]
        if obj.get("done"):
            break


def stream_answer(
    question: str, hits: list[dict], history: list[dict] | None = None
) -> Iterator[str]:
    provider, _model = config.get_provider()
    prompt = build_prompt(question, hits, history)
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
