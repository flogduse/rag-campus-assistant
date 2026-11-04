"""Central configuration for the RAG Campus Assistant.

All knobs live here; secrets live in a .env file (never committed).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"          # source PDFs/TXT to ingest
CHROMA_DIR = DATA_DIR / "chroma"      # persistent vector store

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))       # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # characters

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "4"))           # chunks fetched per query
FETCH_K = int(os.getenv("FETCH_K", "12"))      # candidates before MMR
# 0.45 measured on data/eval/golden.json with hybrid scoring: on-topic top hits
# score 0.45-0.95, off-topic ones stay near 0.20-0.35. Re-tune with
# `python scripts/eval_retrieval.py`.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.45"))  # drop chunks below this hybrid score
# Dense embeddings handle paraphrase but under-score short keyword queries. A
# literal term-overlap bonus recovers them without a second index.
LEXICAL_WEIGHT = float(os.getenv("LEXICAL_WEIGHT", "0.35"))
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "6"))  # prior messages used as context

# --- Server protection ---
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")      # when set, /api/ingest requires it
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "20"))  # chat requests per window per IP
RATE_WINDOW = int(os.getenv("RATE_WINDOW", "60"))  # window length in seconds

# --- Embeddings (local by default — no API key, no cost) ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# --- LLM provider (auto-detected) ---
# Priority: Groq (GROQ_API_KEY) -> OpenAI (OPENAI_API_KEY) -> Ollama (local)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

LLM_TEMP = float(os.getenv("LLM_TEMP", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "700"))

SYSTEM_PROMPT = (
    "You are the Campus Assistant for Adithya's college. Answer strictly from "
    "the provided context. If the context does not contain the answer, say "
    "you don't know and suggest which document or office to check. Be concise "
    "(3-8 sentences). When you use facts from a source, cite it inline as [1], "
    "[2] matching the numbered context blocks."
)


PROVIDER_TTL = float(os.getenv("PROVIDER_TTL", "30"))  # seconds to trust a probe
_provider_cache: tuple[float, tuple[str, str | None]] | None = None


def reset_provider_cache() -> None:
    """Forget the cached detection result (e.g. after editing .env in-process)."""
    global _provider_cache
    _provider_cache = None


def get_provider() -> tuple[str, str | None]:
    """Return (provider_name, model) for the best available LLM.

    Order: Groq cloud (free tier) -> OpenAI -> local Ollama.

    The result is cached for PROVIDER_TTL seconds rather than forever: the
    Ollama probe is a network call, but a permanent cache would keep reporting
    "none" after the user starts Ollama. Key-based providers need no probe.
    """
    global _provider_cache
    now = time.monotonic()
    if _provider_cache is not None and now - _provider_cache[0] < PROVIDER_TTL:
        return _provider_cache[1]

    if GROQ_API_KEY:
        result: tuple[str, str | None] = ("groq", GROQ_MODEL)
    elif OPENAI_API_KEY:
        result = ("openai", OPENAI_MODEL)
    else:
        result = ("none", None)
        try:
            import requests
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            if r.status_code == 200:
                result = ("ollama", OLLAMA_MODEL)
        except Exception:
            pass

    _provider_cache = (now, result)
    return result
