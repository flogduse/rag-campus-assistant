"""ChromaDB vector store wrapper.

A key detail: ChromaDB only accepts scalar metadata values. Page numbers are
ints, but a PDF with no extractable page labels would produce None values that
crash ingestion — so pages are stored as strings (or dropped), and restored to
int|None when read back.
"""
from __future__ import annotations

import logging

import chromadb
from chromadb.config import Settings as ChromaSettings

from . import config

log = logging.getLogger(__name__)

_client = None
_collection = None


def get_collection():
    """Lazy singleton so module import stays cheap (and test-friendly)."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name="campus_docs",
            metadata={"hnsw:space": "cosine"},
            # None embedding_function: we pass pre-computed vectors explicitly.
            embedding_function=None,
        )
    return _collection


def _clean_meta(meta: dict) -> dict:
    """Chroma only allows str|int|float|bool metadata — coerce/drop the rest."""
    out = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    for required in ("source", "page", "chunk_index"):
        out.setdefault(required, "" if required != "chunk_index" else 0)
    return out


def add_chunks(chunks: list[dict], vectors: list[list[float]]) -> int:
    col = get_collection()
    if not chunks:
        return 0
    col.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[_clean_meta(c["metadata"]) for c in chunks],
        embeddings=vectors,
    )
    return len(chunks)


def query(text: str, vector: list[float], k: int) -> dict:
    col = get_collection()
    return col.query(
        query_embeddings=[vector],
        n_results=min(k, max(col.count(), 1)),
        include=["documents", "metadatas", "distances"],
    )


def count() -> int:
    try:
        return get_collection().count()
    except Exception as exc:  # health must not 500 because the store is unreadable
        log.warning("store.count() failed: %s", exc)
        return 0


def reset() -> None:
    """Delete every chunk (used on re-ingest).

    Errors propagate on purpose: a partially wiped store followed by an ingest
    would leave duplicate IDs and mixed old/new chunks with no visible signal.
    """
    col = get_collection()
    got = col.get(include=[])
    if got and got.get("ids"):
        col.delete(ids=got["ids"])


def unique_docs() -> int:
    try:
        col = get_collection()
        if col.count() == 0:
            return 0
        got = col.get(include=["metadatas"])
        return len({m.get("source", "?") for m in got["metadatas"]})
    except Exception as exc:
        log.warning("store.unique_docs() failed: %s", exc)
        return 0
