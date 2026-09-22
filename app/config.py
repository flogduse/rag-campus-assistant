"""Central configuration for the RAG Campus Assistant.

All knobs live here; secrets live in a .env file (never committed).
"""
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

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


@lru_cache
def get_provider() -> tuple[str, str | None]:
    """Return (provider_name, model) for the best available LLM.

    Order: Groq cloud (free tier) -> OpenAI -> local Ollama.
    """
    if GROQ_API_KEY:
        return "groq", GROQ_MODEL
    if OPENAI_API_KEY:
        return "openai", OPENAI_MODEL
    try:
        import requests
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if r.status_code == 200:
            return "ollama", OLLAMA_MODEL
    except Exception:
        pass
    return "none", None
